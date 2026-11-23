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
