"""The numpy mesh fast path: handedness against hard-coded block geometry, agreement with the exact projection, occlusion splits, and the load dispatch.

The block's true geometry comes from conftest's constants. The comparison against the OCC projection of the STEP file is a second, independent opinion: it
does not share the fast path's code (no common projection, edge classification or visibility test), but both read the same block, so neither catches a
conftest block that is itself wrong.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import blueprint.geometry as geometry
from blueprint.geometry import Projection, load_shape, project, section_view
from blueprint.layout import auto_sheet, view_directions
from blueprint.mesh import MeshModel, build_mesh, load_mesh
from blueprint.sheet import Sheet
from conftest import BLOCK, HOLE_CENTRE, HOLE_R, NOTCH

TOL = 0.06  # mm: the block's STL meshes the 3 mm hole to 0.05 mm, so facet silhouettes of the hole sit up to that far from the exact ones
NOTCH_X, NOTCH_Y = BLOCK[0] - NOTCH, BLOCK[1] - NOTCH


def _fast(path: Path, name: str) -> Projection:
    from blueprint.geometry import _mesh_projection

    d, u = view_directions("+Z", "+Y")[name]
    return _mesh_projection(load_mesh(path), d, u)


def _segments(lines: list[np.ndarray]) -> np.ndarray:
    """(n, 2, 2) array of every segment of every polyline."""
    return np.array([(a, b) for line in lines for a, b in zip(line[:-1], line[1:], strict=True)]).reshape(-1, 2, 2)


def _distance(points: np.ndarray, segs: np.ndarray) -> np.ndarray:
    """Distance from each of n points to the nearest of m segments."""
    a, b = segs[:, 0][None], segs[:, 1][None]
    ab = b - a
    ap = points[:, None, :] - a
    t = np.clip((ap * ab).sum(-1) / np.maximum((ab * ab).sum(-1), 1e-18), 0, 1)
    return np.asarray(np.linalg.norm(ap - t[..., None] * ab, axis=-1).min(axis=1))


def _along(lines: list[np.ndarray], step: float = 0.5) -> np.ndarray:
    """Points spaced about `step` along every segment, ends included."""
    out = []
    for a, b in _segments(lines):
        n = max(2, int(np.linalg.norm(b - a) / step) + 1)
        out.append(a + (b - a) * np.linspace(0, 1, n)[:, None])
    return np.vstack(out)


def _off_hole(points: np.ndarray, view: str) -> np.ndarray:
    """Drop points on the hole's rim, where a polygon and a circle legitimately differ by up to the mesh deflection."""
    if view not in ("FRONT", "BACK"):
        return points
    cx = HOLE_CENTRE[0] if view == "FRONT" else -HOLE_CENTRE[0]
    return points[np.abs(np.hypot(points[:, 0] - cx, points[:, 1] - HOLE_CENTRE[1]) - HOLE_R) > 0.2]


@pytest.mark.parametrize("name", ["FRONT", "BACK", "RIGHT", "LEFT", "TOP", "BOTTOM"])
def test_fast_path_agrees_with_the_exact_projection(block_stl: Path, block_step: Path, name: str) -> None:
    """Same bounds, and every straight edge of one is on the other: visible against visible, hidden against hidden (hole facets excepted)."""
    d, u = view_directions("+Z", "+Y")[name]
    exact = project(load_shape(block_step), d, u)
    fast = _fast(block_stl, name)
    assert fast.bounds() == pytest.approx(exact.bounds(), abs=1e-3)
    exact_vis = _segments(exact.visible)
    fast_vis = _segments(fast.visible)
    exact_pts = _off_hole(_along([p for p in exact.visible if len(p) == 2]), name)
    fast_pts = _off_hole(_along(fast.visible), name)
    assert _distance(exact_pts, fast_vis).max() < TOL, "an exact visible edge is missing from the fast path"
    assert _distance(fast_pts, exact_vis).max() < TOL, "the fast path drew a visible edge the exact projection does not have"
    exact_hid = [p for p in exact.hidden if len(p) == 2]
    if exact_hid:
        assert _distance(_off_hole(_along(exact_hid), name), _segments(fast.hidden)).max() < TOL, "a hidden edge is missing from the fast path"
    if fast.hidden:
        stray = _off_hole(_along(fast.hidden), name)
        assert _distance(stray, _segments(exact.hidden + exact.visible)).max() < TOL, "the fast path drew a hidden edge that is not in the exact one"


def _has_segment(lines: list[np.ndarray], a: tuple[float, float], b: tuple[float, float]) -> bool:
    """True if the segment a-b lies along the drawn lines, within TOL everywhere."""
    pts = np.linspace(a, b, 40)
    return bool(_distance(pts, _segments(lines)).max() < TOL)


def test_fast_path_handedness_on_hard_coded_geometry(block_stl: Path) -> None:
    """Views of the notched block, checked against coordinates worked out by hand from the third-angle convention (x = up x toward, y up).

    Front (+Z): x is model X, y is model Y. Right (+X): x is -Z (the viewer's right looking along -X). Top (+Y, up -Z): x is X and y is -Z, so
    the front face is at the bottom.
    """
    w, h, depth = BLOCK
    front = _fast(block_stl, "FRONT")
    assert front.bounds() == pytest.approx((0, 0, w, h), abs=1e-3)
    assert _has_segment(front.visible, (NOTCH_X, NOTCH_Y), (w, NOTCH_Y))  # the notch's floor, top right
    assert _has_segment(front.visible, (NOTCH_X, NOTCH_Y), (NOTCH_X, h))  # its wall
    top_edge_end = _distance(np.array([[w - 1, h]]), _segments(front.visible))[0]
    assert top_edge_end > 0.5, "the notch's corner must be empty, not filled in"
    right = _fast(block_stl, "RIGHT")
    assert right.bounds() == pytest.approx((-depth, 0, 0, h), abs=1e-3)
    assert _has_segment(right.visible, (-depth, NOTCH_Y), (0, NOTCH_Y))  # the notch floor edge-on as a line across the depth
    assert _has_segment(right.hidden, (-depth, HOLE_CENTRE[1] - HOLE_R), (0, HOLE_CENTRE[1] - HOLE_R))  # the hole, seen through the side
    top = _fast(block_stl, "TOP")
    assert top.bounds() == pytest.approx((0, -depth, w, 0), abs=1e-3)
    assert _has_segment(top.visible, (NOTCH_X, -depth), (NOTCH_X, 0))  # the notch wall, from above
    left = _fast(block_stl, "LEFT")
    assert left.bounds() == pytest.approx((0, 0, depth, h), abs=1e-3)
    assert _has_segment(left.visible, (0, 0), (depth, 0))


def _box(lo: tuple[float, float, float], hi: tuple[float, float, float]) -> np.ndarray:
    """12 outward-wound triangles of an axis-aligned box."""
    x0, y0, z0 = lo
    x1, y1, z1 = hi
    c = np.array([[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0], [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]], float)
    quads = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (2, 3, 7, 6), (1, 2, 6, 5), (3, 0, 4, 7)]
    return np.array([c[[q[0], q[1], q[2]]] for q in quads] + [c[[q[0], q[2], q[3]]] for q in quads])


def test_edge_is_split_where_it_goes_behind_something() -> None:
    """A rear box's top edge runs behind a nearer box: it is drawn solid where seen and hidden where covered, and the cut falls at the nearer box's edge."""
    rear = _box((0, 0, -20), (10, 10, -10))
    near = _box((0, -5, 0), (4, 15, 5))
    mesh = build_mesh(np.concatenate([rear, near]), "two boxes")
    view = geometry._mesh_projection(mesh, (0, 0, 1), (0, 1, 0))
    assert _has_segment(view.visible, (4, 10), (10, 10))
    assert _has_segment(view.hidden, (0, 10), (4, 10))
    assert not _has_segment(view.visible, (0, 10), (3.5, 10))
    # The exact cut, to the resolution of the bisection (a 256th of the sample spacing, about 0.0003 of the diagonal here).
    points = _along(view.visible, 0.05)
    on_edge = points[(np.abs(points[:, 1] - 10) < 1e-6) & (points[:, 0] > 0.5)]  # not the vertical edges that end at (0, 10)
    assert on_edge[:, 0].min() == pytest.approx(4.0, abs=0.01)


def test_fast_path_is_deterministic(block_stl: Path) -> None:
    """Two runs give identical polylines, which byte-stable output depends on."""
    a, b = _fast(block_stl, "ISOMETRIC"), _fast(block_stl, "ISOMETRIC")
    assert len(a.visible) == len(b.visible) and all(np.array_equal(p, q) for p, q in zip(a.visible + a.hidden, b.visible + b.hidden, strict=True))


def test_large_stl_loads_as_mesh_and_draws_a_sheet(block_stl: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Above the fast-path threshold an STL loads as a MeshModel, the sheet lays out unchanged, and a section view says it needs STEP."""
    monkeypatch.setattr(geometry, "MESH_FAST_MIN_TRIANGLES", 10)
    model = load_shape(block_stl)
    assert isinstance(model, MeshModel)
    layout = auto_sheet(Sheet("B"), model, views=("FRONT", "TOP", "RIGHT", "ISOMETRIC"))
    assert set(layout.views) == {"FRONT", "TOP", "RIGHT", "ISOMETRIC"}
    with pytest.raises(ValueError, match="STEP"):
        section_view(model, (20, 15, 10), (0, 0, 1), (0, 1, 0))


def test_small_stl_keeps_the_occ_path(block_stl: Path) -> None:
    """At the default threshold a small STL still goes through sewing, so existing sheets render as before."""
    assert not isinstance(load_shape(block_stl), MeshModel)


# --- Fix round: guard bypasses, padding occluders, winding ---


def _scene_boxes(offset: float = 0.0, seed: int = 0) -> np.ndarray:
    """A few overlapping random boxes, shifted by `offset` along z. The triangle count (12 per box) leaves the last BVH leaf partly filled."""
    rng = np.random.default_rng(seed)
    boxes = []
    for _ in range(int(rng.integers(2, 9))):
        lo = rng.uniform(0, 40, 3)
        boxes.append(_box(tuple(lo + [0, 0, offset]), tuple(lo + rng.uniform(3, 15, 3) + [0, 0, offset])))
    return np.concatenate(boxes)


def _length(lines: list[np.ndarray]) -> float:
    return float(sum(np.linalg.norm(np.diff(p, axis=0), axis=1).sum() for p in lines))


@pytest.mark.parametrize("seed", range(40))
def test_projection_does_not_depend_on_where_the_part_sits_along_the_view(seed: int) -> None:
    """Moving a scene toward or away from the viewer changes no drawn length: padding in a partly filled BVH leaf once occluded rays at negative depth."""
    results = []
    for offset in (-200.0, 0.0, 200.0):
        view = geometry._mesh_projection(build_mesh(_scene_boxes(offset, seed), "boxes"), (0.3, 0.5, 1), (0, 1, 0))
        results.append((_length(view.visible), _length(view.hidden)))
    assert results[0] == pytest.approx(results[1], abs=1e-6)
    assert results[2] == pytest.approx(results[1], abs=1e-6)


def test_bvh_matches_brute_force_over_several_levels() -> None:
    """A BVH of 997 triangles (7 levels, last leaf partly filled) answers random rays, at negative depth, as testing every triangle does."""
    from blueprint.mesh import _Bvh

    rng = np.random.default_rng(3)
    verts = rng.uniform(-50, 50, (2991, 3))
    faces = np.arange(2991).reshape(-1, 3)
    verts[:, 2] = rng.uniform(-80, -20, len(verts))  # all behind the origin, so a padding triangle at depth 0 would wrongly look nearer than every ray
    verts[:, :2] *= 0.2  # small triangles in the view plane, so most rays hit few of them
    xy, depth = verts[:, :2], verts[:, 2]
    bvh = _Bvh(xy, depth, faces, 1e-9)
    q = xy[faces[rng.integers(0, len(faces), 400)]].mean(axis=1) + rng.normal(0, 0.05, (400, 2))  # rays that land on triangles, last leaf included
    q[-60:] = xy[faces[-60:]].mean(axis=1)  # the last triangles, which share the partly filled leaf
    d = rng.uniform(-80, -10, 400)
    expected = np.zeros(400, dtype=bool)
    for tri in faces:
        a, b, c = xy[tri]
        for i in range(400):
            p = q[i]

            def side(u: np.ndarray, v: np.ndarray, w: np.ndarray) -> float:
                return float((v[0] - u[0]) * (w[1] - u[1]) - (v[1] - u[1]) * (w[0] - u[0]))

            s = [side(a, b, p), side(b, c, p), side(c, a, p)]
            if not (all(x >= -1e-9 for x in s) or all(x <= 1e-9 for x in s)):
                continue
            area = side(a, b, c)
            if abs(area) < 1e-12:
                continue
            w = np.array([s[1], s[2], s[0]]) / area
            if (w * depth[tri]).sum() > d[i]:
                expected[i] = True
    assert expected.sum() > 20 and (~expected).sum() > 20  # both answers are exercised
    assert np.array_equal(bvh.occluded(q, d), expected)


def _uv_sphere(radius: float = 10.0, rings: int = 24, segments: int = 48) -> np.ndarray:
    """Outward-wound UV sphere triangles, smooth enough (7.5 degrees a step) that no crease is sharper than the 20 degree limit."""
    tris = []
    for i in range(rings):
        t0, t1 = np.pi * i / rings, np.pi * (i + 1) / rings
        for j in range(segments):
            p0, p1 = 2 * np.pi * j / segments, 2 * np.pi * (j + 1) / segments

            def pt(t: float, p: float) -> list[float]:
                return [radius * np.sin(t) * np.cos(p), radius * np.sin(t) * np.sin(p), radius * np.cos(t)]

            a, b, c, d = pt(t0, p0), pt(t0, p1), pt(t1, p1), pt(t1, p0)
            if i > 0:
                tris.append([a, d, b])
            if i < rings - 1:
                tris.append([b, d, c])
    return np.array(tris, float)


@pytest.mark.parametrize("fraction", [0.0, 0.01, 0.2, 1.0])
def test_inconsistent_winding_draws_the_same_sphere(fraction: float) -> None:
    """Faces wound the wrong way (a 1% scatter, a fifth, or the whole sphere inside out) must not add false creases or silhouettes."""
    tris = _uv_sphere()
    good = geometry._mesh_projection(build_mesh(tris, "good"), (0, 0, 1), (0, 1, 0))
    flipped = tris.copy()
    pick = np.random.default_rng(1).random(len(flipped)) < fraction
    if fraction == 1.0:
        pick[:] = True
    flipped[pick] = flipped[pick][:, [0, 2, 1]]
    bad = geometry._mesh_projection(build_mesh(flipped, "bad"), (0, 0, 1), (0, 1, 0))
    assert _length(bad.visible) == pytest.approx(_length(good.visible), rel=1e-6)
    assert _length(bad.hidden) == pytest.approx(_length(good.hidden), rel=1e-6)
    assert len(bad.visible) == len(good.visible)


def _binary_stl(triangles: np.ndarray, count: int | None = None, pad: int = 0, cut: int = 0) -> bytes:
    record = np.zeros(len(triangles), dtype=[("normal", "<f4", 3), ("v", "<f4", (3, 3)), ("attr", "<u2")])
    record["v"] = triangles
    data = b"\0" * 80 + np.uint32(len(triangles) if count is None else count).tobytes() + record.tobytes() + b"\0" * pad
    return data[: len(data) - cut]


@pytest.mark.parametrize(
    "make",
    [
        pytest.param(lambda t: _binary_stl(t), id="exact"),
        pytest.param(lambda t: _binary_stl(t, pad=1), id="one-trailing-byte"),
        pytest.param(lambda t: _binary_stl(t, pad=300), id="padded"),
        pytest.param(lambda t: _binary_stl(t, cut=10), id="truncated"),
        pytest.param(lambda t: _binary_stl(t, count=0), id="header-count-zero"),
        pytest.param(lambda t: _binary_stl(t, count=0xFFFFFFFF), id="header-count-huge"),
    ],
)
def test_size_guard_cannot_be_bypassed_by_binary_quirks(make, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:  # type: ignore[no-untyped-def]
    """Padding, truncation and a lying header count all still trip the guard; none may fall through to OCC sewing, which is the hang."""
    tris = _uv_sphere(rings=8, segments=12)  # 168 triangles
    path = tmp_path / "part.stl"
    path.write_bytes(make(tris))
    monkeypatch.setattr(geometry, "MESH_MAX_TRIANGLES", 100)
    with pytest.raises(ValueError, match=r"part\.stl has [\d,]+ triangles.*limit is 100"):
        load_shape(path)


def test_size_guard_counts_uppercase_ascii(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    lines = ["SOLID part"]
    for tri in _uv_sphere(rings=8, segments=12):
        lines += ["FACET NORMAL 0 0 1", " OUTER LOOP", *[f"  VERTEX {x} {y} {z}" for x, y, z in tri], " ENDLOOP", "ENDFACET"]
    path = tmp_path / "upper.stl"
    path.write_text("\n".join([*lines, "ENDSOLID part"]))
    from blueprint.mesh import read_stl, stl_triangle_count

    assert stl_triangle_count(path) == 168 == len(read_stl(path))
    monkeypatch.setattr(geometry, "MESH_MAX_TRIANGLES", 100)
    with pytest.raises(ValueError, match="limit is 100"):
        load_shape(path)


def test_damaged_stl_is_named_not_silently_empty(tmp_path: Path) -> None:
    from blueprint.mesh import read_stl

    tris = _uv_sphere(rings=8, segments=12)
    cut = tmp_path / "cut.stl"
    cut.write_bytes(_binary_stl(tris, cut=10))
    with pytest.raises(ValueError, match="truncated"):
        read_stl(cut)
    tiny = tmp_path / "tiny.stl"
    tiny.write_bytes(b"solid\n")
    with pytest.raises(ValueError, match="not an STL"):
        read_stl(tiny)


def test_ascii_parse_is_the_same_whatever_the_chunking(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An ASCII STL is parsed in chunks to bound memory; tiny chunks (cutting mid-line and mid-number) must give the same triangles as one chunk."""
    import blueprint.mesh as mesh_module

    tris = _uv_sphere(rings=8, segments=12)
    lines = ["solid s"]
    for tri in tris:
        lines += ["facet normal 0 0 1", " outer loop", *[f"  vertex {float(x)!r} {float(y)!r} {float(z)!r}" for x, y, z in tri], " endloop", "endfacet"]
    path = tmp_path / "chunky.stl"
    path.write_text("\n".join([*lines, "endsolid s"]))
    whole = mesh_module.read_stl(path)
    assert np.allclose(whole, tris)
    monkeypatch.setattr(mesh_module, "ASCII_CHUNK_BYTES", 37)
    assert np.array_equal(mesh_module.read_stl(path), whole)
    assert mesh_module.stl_triangle_count(path) == len(tris)


# --- Second fix round: int32 overflow in the winding flood fill, ASCII detection, huge tokens ---


def _grid(n: int) -> np.ndarray:
    """A flat n x n grid of quads as 2 n^2 triangles, all wound the same way."""
    xs = np.arange(n + 1, dtype=float)
    gx, gy = np.meshgrid(xs, xs)
    p = np.stack([gx, gy, np.zeros_like(gx)], axis=-1)
    a, b, c, d = p[:-1, :-1], p[:-1, 1:], p[1:, 1:], p[1:, :-1]
    return np.concatenate([np.stack([a, b, c], axis=2).reshape(-1, 3, 3), np.stack([a, c, d], axis=2).reshape(-1, 3, 3)])


def test_winding_is_made_consistent_on_a_large_mesh() -> None:
    """Above 46,340 faces the flood fill's edge keys overflowed int32 and left flipped faces flipped: no edge may be walked the same way twice afterwards."""
    tris = _grid(300)  # 180,000 faces: the old int32 keys wrapped from 46,341 faces up, but only a mesh well beyond that showed it
    flip = np.zeros(len(tris), dtype=bool)
    flip[::997] = True
    tris[flip] = tris[flip][:, [0, 2, 1]]
    mesh = build_mesh(tris, "grid")
    assert len(mesh.faces) > 100_000
    directed = np.concatenate([mesh.faces[:, [0, 1]], mesh.faces[:, [1, 2]], mesh.faces[:, [2, 0]]])
    keys = directed[:, 0] * len(mesh.vertices) + directed[:, 1]
    assert len(np.unique(keys)) == len(keys), "some edge is traversed in the same direction by two faces"


def _ascii_stl(triangles: np.ndarray, header: str = "solid part", keyword_case: str = "lower") -> str:
    lines = [header]
    for tri in triangles:
        body = ["facet normal 0 0 1", " outer loop", *[f"  vertex {float(x)!r} {float(y)!r} {float(z)!r}" for x, y, z in tri], " endloop", "endfacet"]
        lines += [k.upper() if keyword_case == "upper" else k for k in body]
    return "\n".join([*lines, "endsolid"]) + "\n"


@pytest.mark.parametrize("variant", ["bom", "long-name", "bom-long-name"])
def test_ascii_stl_with_bom_or_long_solid_name_is_read_as_ascii(variant: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A UTF-8 byte-order mark or a solid name over a kilobyte once made an ASCII file read as binary: a garbage count and a misleading 'truncated'."""
    from blueprint.mesh import read_stl, stl_triangle_count

    tris = _uv_sphere(rings=8, segments=12)
    header = "solid " + ("n" * 3000 if "long" in variant else "part")
    data = _ascii_stl(tris, header).encode()
    path = tmp_path / "ascii.stl"
    path.write_bytes((b"\xef\xbb\xbf" if "bom" in variant else b"") + data)
    assert stl_triangle_count(path) == len(tris) == len(read_stl(path))
    monkeypatch.setattr(geometry, "MESH_MAX_TRIANGLES", 100)
    with pytest.raises(ValueError, match="limit is 100"):
        load_shape(path)


def test_binary_with_a_solid_facet_header_and_padding_still_trips_the_guard(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Many binary files start 'solid' in their free-text header; with trailing padding the size no longer matches, but it is still binary."""
    from blueprint.mesh import read_stl

    tris = _uv_sphere(rings=8, segments=12)
    data = bytearray(_binary_stl(tris, pad=200))
    data[:12] = b"solid facet "
    path = tmp_path / "solid.stl"
    path.write_bytes(bytes(data))
    assert len(read_stl(path)) == len(tris)
    monkeypatch.setattr(geometry, "MESH_MAX_TRIANGLES", 100)
    with pytest.raises(ValueError, match="limit is 100"):
        load_shape(path)


def test_a_huge_token_in_an_ascii_file_does_not_blow_up_memory(tmp_path: Path) -> None:
    """numpy sized every token to the longest in its chunk, so one 1 MB comment word multiplied by the token count; long tokens are now dropped."""
    import tracemalloc

    from blueprint.mesh import read_stl

    tris = _uv_sphere(rings=8, segments=12)
    text = _ascii_stl(tris, "solid " + "x" * 1_000_000)
    path = tmp_path / "huge.stl"
    path.write_text(text)
    tracemalloc.start()
    got = read_stl(path)
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    assert np.allclose(got, tris)
    assert peak < 20_000_000, f"peak {peak:,} bytes"
