"""STL reading for the guard and the mesh fast path: triangle counts without parsing, and triangles as a numpy array."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

BINARY_HEADER_BYTES = 84  # 80 header + uint32 triangle count
BINARY_TRIANGLE_BYTES = 50  # normal, three vertices (12 floats) + uint16 attribute
_VERTEX = re.compile(rb"vertex\s+(\S+)\s+(\S+)\s+(\S+)")


def _is_binary(data: bytes | memoryview) -> bool:
    """A binary STL's size is exactly 84 + 50 * its declared count; an ASCII STL practically never matches that."""
    if len(data) < BINARY_HEADER_BYTES:
        return False
    count = int(np.frombuffer(data[80:84], dtype="<u4")[0])
    return len(data) == BINARY_HEADER_BYTES + BINARY_TRIANGLE_BYTES * count


def stl_triangle_count(path: Path) -> int:
    """Number of triangles in an STL, read from the header (binary) or by counting facets (ASCII), without parsing the vertices."""
    with path.open("rb") as handle:
        head = handle.read(BINARY_HEADER_BYTES)
        size = path.stat().st_size
        if len(head) == BINARY_HEADER_BYTES:
            count = int(np.frombuffer(head[80:84], dtype="<u4")[0])
            if size == BINARY_HEADER_BYTES + BINARY_TRIANGLE_BYTES * count:
                return count
        return path.read_bytes().count(b"endfacet")


def read_stl(path: Path) -> np.ndarray:
    """The triangles of an STL (binary or ASCII) as an (n, 3, 3) float64 array of vertex coordinates."""
    data = path.read_bytes()
    if _is_binary(data):
        count = int(np.frombuffer(data[80:84], dtype="<u4")[0])
        record = np.dtype([("normal", "<f4", 3), ("v", "<f4", (3, 3)), ("attr", "<u2")])
        return np.frombuffer(data, dtype=record, count=count, offset=BINARY_HEADER_BYTES)["v"].astype(np.float64)
    found = _VERTEX.findall(data)
    if not found or len(found) % 3:
        raise ValueError(f"could not read STL file {path}: no triangles found")
    return np.array(found, dtype=np.float64).reshape(-1, 3, 3)


# --- The mesh fast path: feature and silhouette edges from the triangles, visibility by ray tests against a BVH, no OCC. ---

MESH_WELD_TOL = 1e-4  # mm; vertices closer than this are one vertex (STL repeats every vertex per triangle)
MESH_SMOOTH_ANGLE_DEG = 20.0  # creases flatter than this are one smooth surface and are not drawn (OCC path: geometry.SMOOTH_ANGLE_DEG; the same value)
MESH_FRONT_EPS = 1e-9  # a face whose normal has less than this component toward the viewer is edge-on and counts as facing away
MESH_VIS_STEPS = 400  # an edge is tested every (model diagonal / this) along its projection; occlusion changes are then bisected
MESH_BISECT_STEPS = 8  # halvings used to locate where an edge goes behind something: accurate to a 256th of the sample spacing
MESH_DEPTH_EPS = 1e-5  # of the model diagonal: how far toward the viewer a ray starts, so a surface never occludes the edges lying on it
MESH_LEAF = 8  # triangles per BVH leaf
MESH_QUERY_CHUNK = 20_000  # ray queries traversed at once, which bounds the memory of the frontier


@dataclass(frozen=True)
class MeshModel:
    """A welded triangle mesh with its edge topology, ready to be projected by `project_mesh` without OCC."""

    name: str
    vertices: np.ndarray  # (V, 3)
    faces: np.ndarray  # (F, 3) vertex indices
    normals: np.ndarray  # (F, 3) unit, from the winding
    edges: np.ndarray  # (E, 2) vertex indices, one row per distinct edge
    owners: np.ndarray  # (E, 2) the two faces on an edge; the second is -1 on a boundary edge
    nonmanifold: np.ndarray  # (E,) True where more than two faces share the edge
    diagonal: float  # length of the bounding box diagonal


def build_mesh(triangles: np.ndarray, name: str) -> MeshModel:
    """Weld an (n, 3, 3) triangle soup into a MeshModel: shared vertices, unit normals, one record per edge."""
    if len(triangles) == 0:
        raise ValueError(f"{name} contains no geometry")
    quantised = np.round(triangles.reshape(-1, 3) / MESH_WELD_TOL).astype(np.int64)
    _, first, inverse = np.unique(quantised, axis=0, return_index=True, return_inverse=True)
    vertices = triangles.reshape(-1, 3)[first]
    faces = inverse.reshape(-1, 3).astype(np.int64)
    faces = faces[(faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) & (faces[:, 0] != faces[:, 2])]
    cross_n = np.cross(vertices[faces[:, 1]] - vertices[faces[:, 0]], vertices[faces[:, 2]] - vertices[faces[:, 0]])
    area2 = np.linalg.norm(cross_n, axis=1)
    keep = area2 > 0
    faces, cross_n, area2 = faces[keep], cross_n[keep], area2[keep]
    if len(faces) == 0:
        raise ValueError(f"{name} contains no geometry")
    normals = cross_n / area2[:, None]
    half = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    face_of = np.tile(np.arange(len(faces)), 3)
    lo, hi = half.min(axis=1), half.max(axis=1)
    key = lo * len(vertices) + hi
    order = np.argsort(key, kind="stable")
    key, face_of, lo, hi = key[order], face_of[order], lo[order], hi[order]
    starts = np.flatnonzero(np.r_[True, key[1:] != key[:-1]])
    counts = np.diff(np.r_[starts, len(key)])
    owners = np.full((len(starts), 2), -1, dtype=np.int64)
    owners[:, 0] = face_of[starts]
    two = counts >= 2
    owners[two, 1] = face_of[starts[two] + 1]
    box = vertices.max(axis=0) - vertices.min(axis=0)
    return MeshModel(name, vertices, faces, normals, np.column_stack([lo[starts], hi[starts]]), owners, counts > 2, float(np.linalg.norm(box)))


def load_mesh(path: Path) -> MeshModel:
    """Read an STL into a MeshModel."""
    return build_mesh(read_stl(path), path.name)


class _Bvh:
    """A bounding-volume hierarchy over the projected triangles of one view, answering 'is anything nearer the viewer along this ray?' in bulk.

    Triangles are sorted along a Morton curve of their view-plane centroids and grouped MESH_LEAF to a leaf under a complete binary tree. Each node
    keeps the view-plane box of its triangles and the greatest depth (toward the viewer) among them, so a ray prunes whole subtrees that lie behind it.
    Rays are all parallel to the view direction, so each is a point in the view plane and a depth.
    """

    def __init__(self, xy: np.ndarray, depth: np.ndarray, faces: np.ndarray, tol: float) -> None:
        a, b, c = xy[faces[:, 0]], xy[faces[:, 1]], xy[faces[:, 2]]
        v0, v1 = b - a, c - a
        det = v0[:, 0] * v1[:, 1] - v0[:, 1] * v1[:, 0]
        extent = float(np.ptp(xy, axis=0).max()) or 1.0
        valid = np.abs(det) > 1e-12 * extent * extent  # a face seen edge-on has no area in the view and occludes nothing
        lo = np.minimum(np.minimum(a, b), c)
        hi = np.maximum(np.maximum(a, b), c)
        mid = (lo + hi) / 2
        scale = (mid - xy.min(axis=0)) / extent * 65535
        code = _morton(np.clip(scale[:, 0], 0, 65535).astype(np.uint32), np.clip(scale[:, 1], 0, 65535).astype(np.uint32))
        order = np.argsort(code, kind="stable")
        order = order[valid[order]]
        count = len(order)
        leaves = 1
        while leaves * MESH_LEAF < count:
            leaves *= 2
        padded = leaves * MESH_LEAF
        self.leaves = leaves
        self.tol = tol

        def pad(arr: np.ndarray, fill: float) -> np.ndarray:
            out = np.full((padded, *arr.shape[1:]), fill)
            out[:count] = arr[order]
            return out

        self.a, self.v0, self.v1 = pad(a, 0.0), pad(v0, 0.0), pad(v1, 0.0)
        safe = np.where(valid, det, 1.0)
        self.inv_det = pad(1.0 / safe, 0.0)
        da, db, dc = depth[faces[:, 0]], depth[faces[:, 1]], depth[faces[:, 2]]
        self.da, self.d1, self.d2 = pad(da, 0.0), pad(db - da, 0.0), pad(dc - da, 0.0)
        tri_lo, tri_hi, tri_depth = pad(lo, np.inf), pad(hi, -np.inf), pad(np.maximum(np.maximum(da, db), dc), -np.inf)
        node_lo = tri_lo.reshape(leaves, MESH_LEAF, 2).min(axis=1)
        node_hi = tri_hi.reshape(leaves, MESH_LEAF, 2).max(axis=1)
        node_depth = tri_depth.reshape(leaves, MESH_LEAF).max(axis=1)
        self.levels = [(node_lo, node_hi, node_depth)]
        while len(self.levels[0][0]) > 1:
            lo_, hi_, dep_ = self.levels[0]
            self.levels.insert(0, (lo_.reshape(-1, 2, 2).min(axis=1), hi_.reshape(-1, 2, 2).max(axis=1), dep_.reshape(-1, 2).max(axis=1)))

    def occluded(self, xy: np.ndarray, depth: np.ndarray) -> np.ndarray:
        """For each ray (view-plane point, starting depth): True if some triangle crosses it at a greater depth."""
        out = np.zeros(len(xy), dtype=bool)
        for start in range(0, len(xy), MESH_QUERY_CHUNK):
            sl = slice(start, start + MESH_QUERY_CHUNK)
            out[sl] = self._chunk(xy[sl], depth[sl])
        return out

    def _chunk(self, xy: np.ndarray, depth: np.ndarray) -> np.ndarray:
        hit = np.zeros(len(xy), dtype=bool)
        qi = np.arange(len(xy))
        node = np.zeros(len(xy), dtype=np.int64)
        qi, node = self._prune(0, qi, node, xy, depth)
        for level in range(1, len(self.levels)):
            qi = np.repeat(qi, 2)
            node = np.repeat(node, 2) * 2 + np.tile(np.array([0, 1]), len(node))
            qi, node = self._prune(level, qi, node, xy, depth)
        if len(qi):
            qi = np.repeat(qi, MESH_LEAF)
            tri = np.repeat(node, MESH_LEAF) * MESH_LEAF + np.tile(np.arange(MESH_LEAF), len(node))
            rel = xy[qi] - self.a[tri]
            v0, v1, inv = self.v0[tri], self.v1[tri], self.inv_det[tri]
            u = (rel[:, 0] * v1[:, 1] - rel[:, 1] * v1[:, 0]) * inv
            w = (v0[:, 0] * rel[:, 1] - v0[:, 1] * rel[:, 0]) * inv
            eps = 1e-9
            inside = (u >= -eps) & (w >= -eps) & (u + w <= 1 + eps)
            nearer = self.da[tri] + u * self.d1[tri] + w * self.d2[tri] > depth[qi]
            hit[qi[inside & nearer]] = True
        return hit

    def _prune(self, level: int, qi: np.ndarray, node: np.ndarray, xy: np.ndarray, depth: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        lo, hi, top = self.levels[level]
        p = xy[qi]
        lo_n, hi_n = lo[node] - self.tol, hi[node] + self.tol
        ok = np.all((p >= lo_n) & (p <= hi_n), axis=1)
        ok &= top[node] > depth[qi]
        return qi[ok], node[ok]


def _morton(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Interleave the low 16 bits of x and y."""

    def spread(v: np.ndarray) -> np.ndarray:
        v = v.astype(np.uint64)
        v = (v | (v << 8)) & np.uint64(0x00FF00FF)
        v = (v | (v << 4)) & np.uint64(0x0F0F0F0F)
        v = (v | (v << 2)) & np.uint64(0x33333333)
        spread_bits: np.ndarray = (v | (v << 1)) & np.uint64(0x55555555)
        return spread_bits

    interleaved: np.ndarray = spread(x) | (spread(y) << np.uint64(1))
    return interleaved


def _chain(points: np.ndarray) -> list[np.ndarray]:
    """Join segments (n, 2, 2) end to end into polylines wherever exactly two meet at a point; endpoints compare exactly."""
    n = len(points)
    if n == 0:
        return []
    nodes, inverse = np.unique(points.reshape(-1, 2), axis=0, return_inverse=True)
    ends = inverse.reshape(n, 2)
    flat = ends.reshape(-1)
    seg_of = np.repeat(np.arange(n), 2)
    order = np.argsort(flat, kind="stable")
    bounds = np.searchsorted(flat[order], np.arange(len(nodes) + 1))
    incident = seg_of[order].tolist()
    lo_b = bounds.tolist()
    ends_l = ends.tolist()
    used = [False] * n
    chains: list[list[int]] = []

    def walk(node: int, seg: int) -> list[int]:
        path = [node]
        while True:
            used[seg] = True
            a, b = ends_l[seg]
            node = b if a == node else a
            path.append(node)
            if lo_b[node + 1] - lo_b[node] != 2:
                return path
            nxt = [s for s in incident[lo_b[node] : lo_b[node + 1]] if s != seg]
            if not nxt or used[nxt[0]]:
                return path
            seg = nxt[0]

    degree = np.diff(bounds)
    for node in np.flatnonzero(degree != 2).tolist():
        for seg in incident[lo_b[node] : lo_b[node + 1]]:
            if not used[seg]:
                chains.append(walk(node, seg))
    for seg in range(n):  # what is left are closed loops
        if not used[seg]:
            chains.append(walk(ends_l[seg][0], seg))
    return [nodes[c] for c in chains]


def project_mesh(mesh: MeshModel, toward_viewer: tuple[float, float, float], up: tuple[float, float, float]) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Visible and hidden polylines of `mesh` seen from `toward_viewer` with `up` up the page, in the frame geometry.project uses.

    Drawn edges are creases sharper than MESH_SMOOTH_ANGLE_DEG, boundary and non-manifold edges, and silhouettes (one face toward the viewer, its
    neighbour away). Each is tested for visibility by a ray toward the viewer from sample points along it, and split where the answer changes.
    """
    t = np.asarray(toward_viewer, float)
    t = t / np.linalg.norm(t)
    x_dir = np.cross(up, t)
    x_dir /= np.linalg.norm(x_dir)
    axes = np.column_stack([x_dir, np.cross(t, x_dir)])  # x = up × toward, y = toward × x: HLRAlgo_Projector's frame
    verts = mesh.vertices
    front = (mesh.normals @ t) > MESH_FRONT_EPS
    first, second = mesh.owners[:, 0], mesh.owners[:, 1]
    manifold = (second >= 0) & ~mesh.nonmanifold
    sharp = ~manifold
    sec = np.where(manifold, second, 0)
    sharp |= manifold & (np.einsum("ij,ij->i", mesh.normals[first], mesh.normals[sec]) < np.cos(np.radians(MESH_SMOOTH_ANGLE_DEG)))
    silhouette = manifold & (front[first] != front[sec])
    chosen = np.flatnonzero(sharp | silhouette)
    a3, b3 = verts[mesh.edges[chosen, 0]], verts[mesh.edges[chosen, 1]]
    length = np.linalg.norm((b3 - a3) @ axes, axis=1)
    live = length > 1e-9 * mesh.diagonal  # an edge seen end-on has no length in the view
    a3, b3, length = a3[live], b3[live], length[live]
    if len(a3) == 0:
        raise ValueError(f"nothing is visible looking from {toward_viewer}: the part may be empty")

    tris_xy, tris_depth = verts @ axes, verts @ t
    bvh = _Bvh(tris_xy, tris_depth, mesh.faces, 1e-7 * mesh.diagonal)
    lift = MESH_DEPTH_EPS * mesh.diagonal

    def hidden_at(edge: np.ndarray, s: np.ndarray) -> np.ndarray:
        p = a3[edge] * (1 - s)[:, None] + b3[edge] * s[:, None]
        return bvh.occluded(p @ axes, p @ t + lift)

    pieces = np.maximum(1, np.ceil(length / (mesh.diagonal / MESH_VIS_STEPS))).astype(np.int64)
    edge = np.repeat(np.arange(len(a3)), pieces)
    first_of = np.cumsum(pieces) - pieces
    j = np.arange(len(edge)) - first_of[edge]
    n_e = pieces[edge].astype(float)
    t0, t1, tm = j / n_e, (j + 1) / n_e, (j + 0.5) / n_e
    hid = hidden_at(edge, tm)
    # Where neighbouring samples of one edge disagree, bisect between them for the point where it goes behind something.
    flip = np.flatnonzero((edge[1:] == edge[:-1]) & (hid[1:] != hid[:-1]))
    if len(flip):
        lo_s, hi_s, was = tm[flip].copy(), tm[flip + 1].copy(), hid[flip]
        for _ in range(MESH_BISECT_STEPS):
            mid = (lo_s + hi_s) / 2
            same = hidden_at(edge[flip], mid) == was
            lo_s, hi_s = np.where(same, mid, lo_s), np.where(same, hi_s, mid)
        cut = (lo_s + hi_s) / 2
        t1[flip], t0[flip + 1] = cut, cut
    run_start = np.r_[True, (edge[1:] != edge[:-1]) | (hid[1:] != hid[:-1])]
    starts = np.flatnonzero(run_start)
    stops = np.r_[starts[1:], len(edge)] - 1
    r_edge, r_hid = edge[starts], hid[starts]
    s0, s1 = t0[starts], t1[stops]
    p0 = (a3[r_edge] * (1 - s0)[:, None] + b3[r_edge] * s0[:, None]) @ axes
    p1 = (a3[r_edge] * (1 - s1)[:, None] + b3[r_edge] * s1[:, None]) @ axes
    segs = np.stack([p0, p1], axis=1)
    return _chain(segs[~r_hid]), _chain(segs[r_hid])
