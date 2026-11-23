"""Load a part and project it: hidden-line removal on the exact B-rep (STEP) or on a sewn triangle shell (STL fallback).

Only OCP is used, so the package works beside either caller's CAD library (cadquery pins OCP 7.9, build123d uses 8.x).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
from OCP.Bnd import Bnd_Box
from OCP.BndLib import BndLib_Add3dCurve
from OCP.BRep import BRep_Builder
from OCP.BRepAdaptor import BRepAdaptor_Curve, BRepAdaptor_Surface
from OCP.BRepAlgoAPI import BRepAlgoAPI_Common, BRepAlgoAPI_Section
from OCP.BRepBndLib import BRepBndLib
from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace, BRepBuilderAPI_Sewing
from OCP.BRepLib import BRepLib
from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
from OCP.GeomAbs import GeomAbs_CurveType, GeomAbs_Shape, GeomAbs_SurfaceType
from OCP.gp import gp_Ax2, gp_Dir, gp_Pln, gp_Pnt
from OCP.HLRAlgo import HLRAlgo_Projector
from OCP.HLRBRep import HLRBRep_Algo, HLRBRep_HLRToShape
from OCP.IFSelect import IFSelect_ReturnStatus
from OCP.ShapeUpgrade import ShapeUpgrade_UnifySameDomain
from OCP.STEPControl import STEPControl_Reader
from OCP.StlAPI import StlAPI_Reader
from OCP.TopAbs import TopAbs_Orientation, TopAbs_ShapeEnum
from OCP.TopExp import TopExp_Explorer
from OCP.TopoDS import TopoDS, TopoDS_Shape

from .mesh import stl_triangle_count

log = logging.getLogger(__name__)

Vec = tuple[float, float, float]
Polyline = np.ndarray  # N x 2, view-plane millimetres
SECTION_MARGIN = 1.0  # mm the section's keep-box overhangs the part on every side

CURVE_SAMPLES = 28  # points per projected curved edge
SMOOTH_ANGLE_DEG = 20.0  # edges where faces meet flatter than this are facets of one smooth surface and are not drawn
FULL_CIRCLE_RAD = 2 * math.pi - 1e-3
MESH_SEW_TOL = 1e-4  # mm; STL vertices shared by two triangles are written identically, so this only absorbs float noise
MESH_OCC_MAX_TRIANGLES = 150_000  # more triangles than this cannot be sewn and hidden-line-removed in reasonable time (326k never finished sewing)
MESH_COPLANAR_DEG = 0.05  # triangles whose normals differ by less than this are one flat face

# TopoDS's downcast is Edge_s in OCP 7.9 (cadquery) and Edge in OCP 8 (build123d); the package has to run under both.
_as_edge = getattr(TopoDS, "Edge_s", None) or TopoDS.Edge
_as_face = getattr(TopoDS, "Face_s", None) or TopoDS.Face


@dataclass
class Projection:
    """One view of a part: polylines in view-plane millimetres, x to the viewer's right and y up."""

    visible: list[Polyline]
    hidden: list[Polyline]
    straight: list[Polyline] = field(default_factory=list)  # visible straight edges, two points each
    circles: list[tuple[float, float, float]] = field(default_factory=list)  # visible full circles seen face on: (cx, cy, r)
    extent: tuple[float, float, float, float] | None = None  # exact (xmin, ymin, xmax, ymax) of the visible edges, from their curves
    section: list[Polyline] = field(default_factory=list)  # closed outlines of a cut face, hatched when drawn (see section_view)
    clip: tuple[float, float, float] | None = None  # (cx, cy, r): only what lies inside this circle is drawn (see cropped)
    frame: tuple[Vec, Vec] | None = None  # unit (toward_viewer, up) of this view, to map model points and axes into it (View.centre_line)
    unit: float = 1.0  # view-plane mm per model mm: 1 from project(), times k in scaled(), so a model point still lands on the scaled geometry

    def bounds(self) -> tuple[float, float, float, float]:
        """(xmin, ymin, xmax, ymax) of the visible outline: exact from the curves when known, else from the sampled polylines.

        The exact extent matters for curved outlines: 28 samples round a 20 mm circle would measure 19.93.
        """
        if self.extent is not None:
            return self.extent
        if not self.visible:
            raise ValueError("this view has no visible edges: the part may be empty or seen exactly edge-on")
        pts = np.vstack(self.visible)
        return float(pts[:, 0].min()), float(pts[:, 1].min()), float(pts[:, 0].max()), float(pts[:, 1].max())

    def scaled(self, k: float) -> Projection:
        """The same view with every coordinate multiplied by k."""
        extent = None if self.extent is None else (self.extent[0] * k, self.extent[1] * k, self.extent[2] * k, self.extent[3] * k)
        return Projection(
            [p * k for p in self.visible],
            [p * k for p in self.hidden],
            [p * k for p in self.straight],
            [(x * k, y * k, r * k) for x, y, r in self.circles],
            extent,
            [p * k for p in self.section],
            None if self.clip is None else (self.clip[0] * k, self.clip[1] * k, self.clip[2] * k),
            self.frame,
            self.unit * k,
        )

    def cropped(self, centre: tuple[float, float], r: float) -> Projection:
        """The part of this view inside the circle of radius r about centre, for an enlarged detail view.

        The geometry is kept whole and clipped exactly when drawn, so lines, hatching and centre marks all end on the circle. The view's bounds
        become the circle's, so a detail view is placed and sized by the circle it shows.
        """
        if not r > 0:
            raise ValueError(f"a detail needs a positive radius, not {r!r}")
        cx, cy = centre
        return replace(
            self,
            circles=[c for c in self.circles if math.hypot(c[0] - cx, c[1] - cy) < r + c[2]],
            extent=(cx - r, cy - r, cx + r, cy + r),
            clip=(cx, cy, r),
        )


def load_shape(path: Path) -> TopoDS_Shape:
    """Read a STEP file exactly, or an STL as a sewn triangle shell (round features then draw as polygons)."""
    suffix = path.suffix.lower()
    if suffix in (".step", ".stp"):
        reader = STEPControl_Reader()
        if reader.ReadFile(str(path)) != IFSelect_ReturnStatus.IFSelect_RetDone:
            raise ValueError(f"could not read STEP file {path}")
        reader.TransferRoots()
        shape = reader.OneShape()
    elif suffix == ".stl":
        triangles = stl_triangle_count(path)
        if triangles > MESH_OCC_MAX_TRIANGLES:
            raise ValueError(
                f"{path.name} has {triangles:,} triangles; the mesh path is limited to {MESH_OCC_MAX_TRIANGLES:,} "
                "(sewing and hidden-line removal would not finish). Decimate the mesh, or export a STEP file of the part."
            )
        shape = TopoDS_Shape()
        if not StlAPI_Reader().Read(shape, str(path)):
            raise ValueError(f"could not read STL file {path}")
        shape = _clean_mesh(shape)
        log.warning("%s is a mesh: curved features will draw as facets", path.name)
    else:
        raise ValueError(f"unsupported model format {path.suffix!r}: use .step, .stp or .stl")
    if shape.IsNull():
        raise ValueError(f"{path} contains no geometry")
    log.info("loaded %s", path)
    return shape


def _clean_mesh(shape: TopoDS_Shape) -> TopoDS_Shape:
    """Turn an STL's loose triangles into a sewn shell with coplanar triangles merged into single faces.

    StlAPI_Reader makes every triangle its own face with its own edges, so nothing is shared and the projection would draw every triangle edge,
    including the diagonals across flat faces. Sewing shares the edges (so EncodeRegularity can hide the smooth ones); merging same-domain faces
    turns each flat face back into one face. Curved surfaces stay faceted: their facet edges meet at less than SMOOTH_ANGLE_DEG and are hidden.
    """
    sewing = BRepBuilderAPI_Sewing(MESH_SEW_TOL)
    sewing.Add(shape)
    sewing.Perform()
    unify = ShapeUpgrade_UnifySameDomain(sewing.SewedShape(), True, True, False)
    unify.SetLinearTolerance(MESH_SEW_TOL)
    unify.SetAngularTolerance(math.radians(MESH_COPLANAR_DEG))
    unify.Build()
    return unify.Shape()


def bounding_box(shape: TopoDS_Shape) -> tuple[Vec, Vec]:
    """Axis-aligned (min corner, max corner) of a shape."""
    box = Bnd_Box()
    BRepBndLib.Add_s(shape, box)
    lo, hi = box.CornerMin(), box.CornerMax()  # not Get(): OCP 8 returns a struct it cannot convert
    return (lo.X(), lo.Y(), lo.Z()), (hi.X(), hi.Y(), hi.Z())


def _edges(compound: TopoDS_Shape | None) -> list[object]:
    """Every edge of an HLR result compound, with 3D curves rebuilt (OCC can crash sampling HLR edges without it)."""
    found: list[object] = []
    if compound is None or compound.IsNull():
        return found
    explorer = TopExp_Explorer(compound, TopAbs_ShapeEnum.TopAbs_EDGE)
    while explorer.More():
        edge = _as_edge(explorer.Current())
        BRepLib.BuildCurves3d_s(edge, 1e-5)
        found.append(edge)
        explorer.Next()
    return found


def _sample(edge: object) -> tuple[Polyline, GeomAbs_CurveType, BRepAdaptor_Curve]:
    """Points along one projected edge (two for a straight line), its curve type and the adaptor."""
    curve = BRepAdaptor_Curve(edge)
    kind = curve.GetType()
    n = 2 if kind == GeomAbs_CurveType.GeomAbs_Line else CURVE_SAMPLES
    t0, t1 = curve.FirstParameter(), curve.LastParameter()
    pts = [curve.Value(t0 + (t1 - t0) * i / (n - 1)) for i in range(n)]
    return np.array([[p.X(), p.Y()] for p in pts]), kind, curve


def cross(a: Vec, b: Vec) -> Vec:
    """A × b."""
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def project(shape: TopoDS_Shape, toward_viewer: Vec, up: Vec) -> Projection:
    """Hidden-line projection of `shape` seen from `toward_viewer` with `up` pointing up the page.

    The view's x axis is up × toward_viewer, the viewer's right hand, so a right view placed to the right of the front view shows the face that is
    actually on the right (third-angle). Edges between faces meeting within SMOOTH_ANGLE_DEG are marked regular and left out, so faceted or traced
    surfaces draw as one clean outline rather than a comb of facet lines.
    """
    BRepLib.EncodeRegularity_s(shape, math.radians(SMOOTH_ANGLE_DEG))
    _keep_facet_silhouettes(shape, toward_viewer)
    x_dir = cross(up, toward_viewer)
    algo = HLRBRep_Algo()
    algo.Add(shape)
    algo.Projector(HLRAlgo_Projector(gp_Ax2(gp_Pnt(0, 0, 0), gp_Dir(*toward_viewer), gp_Dir(*x_dir))))
    algo.Update()
    algo.Hide()
    out = HLRBRep_HLRToShape(algo)
    unit_t, unit_u = np.asarray(toward_viewer, float), np.asarray(up, float)
    unit_t, unit_u = unit_t / np.linalg.norm(unit_t), unit_u / np.linalg.norm(unit_u)  # callers may pass any length; the stored frame is unit
    view = Projection([], [], frame=((*map(float, unit_t),), (*map(float, unit_u),)))  # type: ignore[arg-type]
    box = Bnd_Box()
    for edge in _edges(out.VCompound()) + _edges(out.OutLineVCompound()):
        pts, kind, curve = _sample(edge)
        BndLib_Add3dCurve.AddOptimal_s(curve, 0.0, box)  # the curve's true extremes, not just the sampled points
        view.visible.append(pts)
        if kind == GeomAbs_CurveType.GeomAbs_Line:
            view.straight.append(pts)
        elif kind == GeomAbs_CurveType.GeomAbs_Circle and curve.LastParameter() - curve.FirstParameter() >= FULL_CIRCLE_RAD:
            circle = curve.Circle()
            centre = circle.Location()
            view.circles.append((centre.X(), centre.Y(), circle.Radius()))
    for edge in _edges(out.HCompound()) + _edges(out.OutLineHCompound()):
        view.hidden.append(_sample(edge)[0])
    if not view.visible:
        raise ValueError(f"nothing is visible looking from {toward_viewer}: the part may be empty")
    lo, hi = box.CornerMin(), box.CornerMax()
    view.extent = (lo.X(), lo.Y(), hi.X(), hi.Y())
    log.debug("projected toward %s: %d visible, %d hidden, %d circles", toward_viewer, len(view.visible), len(view.hidden), len(view.circles))
    return view


def _keep_facet_silhouettes(shape: TopoDS_Shape, toward_viewer: Vec) -> None:
    """Re-mark every edge between two flat faces for this view: sharp if it is a silhouette, smooth if the faces meet flatter than SMOOTH_ANGLE_DEG.

    On a mesh, a curved surface is many flat facets, and its outline in a given view runs along facet edges where one facet faces the viewer
    and its neighbour faces away. EncodeRegularity marks those edges smooth (the facets meet at a shallow angle), so HLR would drop the outline
    along with the facet seams. Re-marked per view, because which edges are silhouettes depends on the direction. Exact curved surfaces are left
    alone: OCC computes their outlines itself.
    """
    view = np.array(toward_viewer, float)
    cos_smooth = math.cos(math.radians(SMOOTH_ANGLE_DEG))
    faces: list[tuple[object, np.ndarray | None]] = []  # (face, outward normal if planar)
    # Edges keyed by their orientation-independent hash, confirmed with IsSame: OCP's == compares Python objects, not shapes, and OCP 8 no
    # longer exposes TopTools' indexed maps. Each entry: [edge, [indices of the faces it bounds]].
    buckets: dict[int, list[list[object]]] = {}
    explorer = TopExp_Explorer(shape, TopAbs_ShapeEnum.TopAbs_FACE)
    while explorer.More():
        face = _as_face(explorer.Current())
        surface = BRepAdaptor_Surface(face)
        normal = None
        if surface.GetType() == GeomAbs_SurfaceType.GeomAbs_Plane:
            d = surface.Plane().Axis().Direction()
            normal = np.array([d.X(), d.Y(), d.Z()]) * (-1 if face.Orientation() == TopAbs_Orientation.TopAbs_REVERSED else 1)
        index = len(faces)
        faces.append((face, normal))
        edges = TopExp_Explorer(face, TopAbs_ShapeEnum.TopAbs_EDGE)
        while edges.More():
            edge = edges.Current()
            bucket = buckets.setdefault(hash(edge), [])
            for entry in bucket:
                if entry[0].IsSame(edge):  # type: ignore[attr-defined]
                    entry[1].append(index)  # type: ignore[attr-defined]
                    break
            else:
                bucket.append([edge, [index]])
            edges.Next()
        explorer.Next()
    builder = BRep_Builder()
    for bucket in buckets.values():
        for edge, owners in bucket:
            if len(owners) != 2:  # type: ignore[arg-type]
                continue
            (f1, n1), (f2, n2) = faces[owners[0]], faces[owners[1]]  # type: ignore[index]
            if n1 is None or n2 is None:
                continue
            silhouette = float(np.dot(n1, view)) * float(np.dot(n2, view)) < 0
            smooth = float(np.dot(n1, n2)) > cos_smooth
            continuity = GeomAbs_Shape.GeomAbs_C1 if smooth and not silhouette else GeomAbs_Shape.GeomAbs_C0
            builder.Continuity(_as_edge(edge), f1, f2, continuity)


def beyond(shape: TopoDS_Shape, origin: Vec, toward_viewer: Vec, depth: float | None = None) -> TopoDS_Shape:
    """The material of `shape` on the far side of the plane through `origin` facing `toward_viewer`.

    That is what a section view looking along -toward_viewer sees once the near part is cut away. With `depth`, only material within that
    distance of the plane is kept.
    """
    t = np.array(toward_viewer, float)  # a copy: never normalise the caller's array in place
    t = t / np.linalg.norm(t)
    lo, hi = (np.asarray(c) for c in bounding_box(shape))
    # A box built in the plane's frame, wider than the part from wherever the plane sits, so it keeps everything beyond the plane.
    span = 2 * (float(np.linalg.norm(hi - lo)) + float(np.linalg.norm((lo + hi) / 2 - np.asarray(origin, float)))) + SECTION_MARGIN
    reach = span if depth is None else depth
    if not reach > 0:
        raise ValueError(f"a section's depth must be positive, not {depth!r}")
    helper = np.array([1.0, 0.0, 0.0]) if abs(t[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    x_dir = np.cross(helper, t)
    x_dir /= np.linalg.norm(x_dir)
    y_dir = np.cross(t, x_dir)
    corner = np.asarray(origin, float) - t * reach - (x_dir + y_dir) * span / 2
    frame = gp_Ax2(gp_Pnt(*corner), gp_Dir(*t), gp_Dir(*x_dir))  # box z along t, from `reach` behind the plane up to it
    box = BRepPrimAPI_MakeBox(frame, span, span, reach).Shape()
    common = BRepAlgoAPI_Common(shape, box)
    common.Build()
    if not common.IsDone():
        raise ValueError("the section cut failed")
    return common.Shape()


def section_view(shape: TopoDS_Shape, origin: Vec, toward_viewer: Vec, up: Vec, depth: float | None = None) -> Projection:
    """A section view: the material beyond the cutting plane through `origin`, seen from `toward_viewer` with `up` up the page.

    The cut face's outlines are attached for hatching. Hidden lines are left out, as is usual in a section view.

    View.draw hatches the cut face. The frame is project()'s, so cut_plane arrows drawn with sight_direction match it.
    """
    t = np.array(toward_viewer, float)
    u = np.array(up, float)
    t, u = t / np.linalg.norm(t), u / np.linalg.norm(u)  # section_loops needs unit vectors; project() and beyond() accept any length
    toward: Vec = (float(t[0]), float(t[1]), float(t[2]))
    up_unit: Vec = (float(u[0]), float(u[1]), float(u[2]))
    loops = section_loops(shape, origin, toward, cross(up_unit, toward))
    if not loops:  # checked first: a plane clear of the part on the near side would leave nothing to project
        raise ValueError(f"the plane through {origin} facing {toward_viewer} does not cut the part")
    projection = project(beyond(shape, origin, toward, depth), toward, up_unit)
    return replace(projection, hidden=[], section=loops)


def sight_direction(toward_viewer: Vec, up: Vec, section_toward_viewer: Vec) -> tuple[float, float]:
    """The line of sight of a section, in the frame of the view its cutting plane is drawn on (for View.cut_plane).

    The parent view looks along -toward_viewer with x = up × toward_viewer; the section view looks along -section_toward_viewer. The result is
    that viewing direction's components on the parent view's x and up axes. A section seen along the parent's own line of sight projects to
    (0, 0) and cannot be drawn as a cutting-plane line.
    """
    x_dir = cross(up, toward_viewer)
    look = -np.asarray(section_toward_viewer, float)
    return float(np.dot(look, x_dir)), float(np.dot(look, up))


def section_loops(shape: TopoDS_Shape, origin: Vec, toward_viewer: Vec, x_dir: Vec) -> list[Polyline]:
    """Closed outlines of the cut through `shape` on the plane through `origin` facing `toward_viewer`, as the viewer sees them: x along `x_dir`, y up.

    Points are in the same frame as project() from the same direction (model origin at view (0, 0), up = toward_viewer × x_dir), wherever `origin`
    sits in the plane, so a section hatched onto a View of that projection lands on it exactly. `x_dir` must be a unit vector in the plane.
    Each edge is sampled and the samples chained head to tail, so a loop comes back in drawing order whatever the stored edge orientations are.
    """
    if abs(float(np.dot(x_dir, toward_viewer))) > 1e-9 or abs(float(np.linalg.norm(x_dir)) - 1) > 1e-9:
        raise ValueError(f"x_dir {x_dir} must be a unit vector perpendicular to toward_viewer {toward_viewer}")
    plane = gp_Pln(gp_Pnt(*origin), gp_Dir(*toward_viewer))
    section = BRepAlgoAPI_Section(shape, BRepBuilderAPI_MakeFace(plane).Face())
    section.Build()
    y_dir = cross(toward_viewer, x_dir)
    segments: list[np.ndarray] = []
    explorer = TopExp_Explorer(section.Shape(), TopAbs_ShapeEnum.TopAbs_EDGE)
    while explorer.More():
        curve = BRepAdaptor_Curve(_as_edge(explorer.Current()))
        n = 2 if curve.GetType() == GeomAbs_CurveType.GeomAbs_Line else CURVE_SAMPLES
        t0, t1 = curve.FirstParameter(), curve.LastParameter()
        pts3 = np.array([[*curve.Value(t0 + (t1 - t0) * i / (n - 1)).Coord()] for i in range(n)])
        segments.append(np.column_stack([pts3 @ np.array(x_dir), pts3 @ np.array(y_dir)]))
        explorer.Next()
    return _chain(segments)


def _chain(segments: list[np.ndarray], tol: float = 1e-4) -> list[Polyline]:
    """Join sampled edge segments into closed loops by matching endpoints."""
    loops: list[Polyline] = []
    remaining = list(segments)
    while remaining:
        loop = remaining.pop(0)
        grown = True
        while grown and np.linalg.norm(loop[0] - loop[-1]) > tol:
            grown = False
            for i, seg in enumerate(remaining):
                if np.linalg.norm(loop[-1] - seg[0]) <= tol:
                    loop = np.vstack([loop, seg[1:]])
                elif np.linalg.norm(loop[-1] - seg[-1]) <= tol:
                    loop = np.vstack([loop, seg[::-1][1:]])
                else:
                    continue
                remaining.pop(i)
                grown = True
                break
        loops.append(loop)
    return loops
