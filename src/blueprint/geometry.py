"""Load a part and project it: hidden-line removal on the exact B-rep (STEP) or on a sewn triangle shell (STL fallback).

Only OCP is used, so the package works beside either caller's CAD library (cadquery pins OCP 7.9, build123d uses 8.x).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from OCP.Bnd import Bnd_Box
from OCP.BRepAdaptor import BRepAdaptor_Curve
from OCP.BRepAlgoAPI import BRepAlgoAPI_Section
from OCP.BRepBndLib import BRepBndLib
from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace
from OCP.BRepLib import BRepLib
from OCP.GeomAbs import GeomAbs_CurveType
from OCP.gp import gp_Ax2, gp_Dir, gp_Pln, gp_Pnt
from OCP.HLRAlgo import HLRAlgo_Projector
from OCP.HLRBRep import HLRBRep_Algo, HLRBRep_HLRToShape
from OCP.IFSelect import IFSelect_ReturnStatus
from OCP.STEPControl import STEPControl_Reader
from OCP.StlAPI import StlAPI_Reader
from OCP.TopAbs import TopAbs_ShapeEnum
from OCP.TopExp import TopExp_Explorer
from OCP.TopoDS import TopoDS, TopoDS_Shape

log = logging.getLogger(__name__)

Vec = tuple[float, float, float]
Polyline = np.ndarray  # N x 2, view-plane millimetres

CURVE_SAMPLES = 28  # points per projected curved edge
SMOOTH_ANGLE_DEG = 20.0  # edges where faces meet flatter than this are facets of one smooth surface and are not drawn
FULL_CIRCLE_RAD = 2 * math.pi - 1e-3

# TopoDS's downcast is Edge_s in OCP 7.9 (cadquery) and Edge in OCP 8 (build123d); the package has to run under both.
_as_edge = getattr(TopoDS, "Edge_s", None) or TopoDS.Edge


@dataclass
class Projection:
    """One view of a part: polylines in view-plane millimetres, x to the viewer's right and y up."""

    visible: list[Polyline]
    hidden: list[Polyline]
    straight: list[Polyline] = field(default_factory=list)  # visible straight edges, two points each
    circles: list[tuple[float, float, float]] = field(default_factory=list)  # visible full circles seen face on: (cx, cy, r)

    def bounds(self) -> tuple[float, float, float, float]:
        """(xmin, ymin, xmax, ymax) over the visible lines."""
        pts = np.vstack(self.visible)
        return float(pts[:, 0].min()), float(pts[:, 1].min()), float(pts[:, 0].max()), float(pts[:, 1].max())

    def scaled(self, k: float) -> Projection:
        """The same view with every coordinate multiplied by k."""
        return Projection(
            [p * k for p in self.visible], [p * k for p in self.hidden], [p * k for p in self.straight], [(x * k, y * k, r * k) for x, y, r in self.circles]
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
        shape = TopoDS_Shape()
        if not StlAPI_Reader().Read(shape, str(path)):
            raise ValueError(f"could not read STL file {path}")
        log.warning("%s is a mesh: curved features will draw as facets", path.name)
    else:
        raise ValueError(f"unsupported model format {path.suffix!r}: use .step, .stp or .stl")
    if shape.IsNull():
        raise ValueError(f"{path} contains no geometry")
    log.info("loaded %s", path)
    return shape


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
    x_dir = cross(up, toward_viewer)
    algo = HLRBRep_Algo()
    algo.Add(shape)
    algo.Projector(HLRAlgo_Projector(gp_Ax2(gp_Pnt(0, 0, 0), gp_Dir(*toward_viewer), gp_Dir(*x_dir))))
    algo.Update()
    algo.Hide()
    out = HLRBRep_HLRToShape(algo)
    view = Projection([], [])
    for edge in _edges(out.VCompound()) + _edges(out.OutLineVCompound()):
        pts, kind, curve = _sample(edge)
        view.visible.append(pts)
        if kind == GeomAbs_CurveType.GeomAbs_Line:
            view.straight.append(pts)
        elif kind == GeomAbs_CurveType.GeomAbs_Circle and curve.LastParameter() - curve.FirstParameter() >= FULL_CIRCLE_RAD:
            circle = curve.Circle()
            centre = circle.Location()
            view.circles.append((centre.X(), centre.Y(), circle.Radius()))
    for edge in _edges(out.HCompound()) + _edges(out.OutLineHCompound()):
        view.hidden.append(_sample(edge)[0])
    log.debug("projected toward %s: %d visible, %d hidden, %d circles", toward_viewer, len(view.visible), len(view.hidden), len(view.circles))
    return view


def section_loops(shape: TopoDS_Shape, origin: Vec, toward_viewer: Vec, x_dir: Vec) -> list[Polyline]:
    """Closed outlines of the cut through `shape` on the plane at `origin` facing `toward_viewer`, as the viewer sees them: x along `x_dir`, y up.

    Up is toward_viewer × x_dir, the same convention as project(), so a section and a projection from the same direction overlay exactly.
    Each edge is sampled and the samples chained head to tail, so a loop comes back in drawing order whatever the stored edge orientations are.
    """
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
        pts3 = np.array([[*curve.Value(t0 + (t1 - t0) * i / (n - 1)).Coord()] for i in range(n)]) - np.array(origin)
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
