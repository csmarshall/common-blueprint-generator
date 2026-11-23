"""Projection tests against the notched block, whose true geometry is known from conftest's constants, not from the code under test."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.BRepPrimAPI import BRepPrimAPI_MakeSphere
from OCP.StlAPI import StlAPI_Writer

from blueprint.geometry import load_shape, project, section_loops
from blueprint.layout import view_directions
from conftest import BLOCK, HOLE_CENTRE, HOLE_R, NOTCH


def _area(loop: np.ndarray) -> float:
    """Shoelace area of a closed polyline."""
    x, y = loop[:, 0], loop[:, 1]
    return abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))) / 2


def _point_to_segment(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """Distance from point p to the segment a-b."""
    ab = b - a
    t = 0.0 if not ab.any() else float(np.clip(np.dot(p - a, ab) / np.dot(ab, ab), 0, 1))
    return float(np.linalg.norm(p - (a + t * ab)))


def _corner_points(view_lines: list[np.ndarray]) -> np.ndarray:
    return np.vstack(view_lines)


def test_front_view_extents_and_notch_on_the_right(block: object) -> None:
    v = project(block, (0, 0, 1), (0, 1, 0))
    x0, y0, x1, y1 = v.bounds()
    assert (x1 - x0, y1 - y0) == pytest.approx(BLOCK[:2], abs=1e-6)
    # The notch is at +X, +Y in model space; in the front view +X must be to the viewer's right, so no visible point sits in the top-right square.
    pts = _corner_points(v.visible)
    inside_notch = (pts[:, 0] > x1 - NOTCH + 1e-3) & (pts[:, 1] > y1 - NOTCH + 1e-3)
    assert not inside_notch.any()
    assert ((pts[:, 0] < x0 + 1e-6) & (pts[:, 1] > y1 - 1e-6)).any(), "top-left corner should exist"


def test_front_view_finds_the_hole_where_it_is(block: object) -> None:
    v = project(block, (0, 0, 1), (0, 1, 0))
    assert len(v.circles) == 1  # the far rim of the through-hole is hidden behind the near one, so only the near rim is a visible circle
    cx, cy, r = v.circles[0]
    x0, y0, _, _ = v.bounds()
    assert (cx - x0, cy - y0, r) == pytest.approx((*HOLE_CENTRE, HOLE_R), abs=1e-6)


def test_right_view_is_third_angle(block: object) -> None:
    """Looking from +X, the front face (+Z) must be on the viewer's left, next to where the front view is placed."""
    dirs = view_directions("+Z", "+Y")
    toward, up = dirs["RIGHT"]
    assert toward == (1, 0, 0)
    v = project(block, toward, up)
    x0, _, x1, _ = v.bounds()
    assert x1 - x0 == pytest.approx(BLOCK[2], abs=1e-6)
    # x = up × toward = (0,1,0) × (1,0,0) = (0,0,-1): view x runs toward -Z, so the front face at z = 20 is the view's left edge.
    assert x0 == pytest.approx(-BLOCK[2], abs=1e-6)


def test_top_view_puts_front_face_at_the_bottom(block: object) -> None:
    toward, up = view_directions("+Z", "+Y")["TOP"]
    assert (toward, up) == ((0, 1, 0), (0, 0, -1))
    v = project(block, toward, up)
    _, y0, _, y1 = v.bounds()
    # View y runs toward -Z, so the front face (z = 20) is the lowest edge, adjacent to the front view placed below the top view.
    assert (y0, y1) == pytest.approx((-BLOCK[2], 0.0), abs=1e-6)


def test_views_reject_non_perpendicular_axes() -> None:
    with pytest.raises(ValueError, match="perpendicular"):
        view_directions("+Z", "-Z")


def test_section_through_the_hole(block: object) -> None:
    loops = section_loops(block, (0, 0, 10), (0, 0, 1), (1, 0, 0))
    areas = sorted(_area(loop) for loop in loops)
    w, h, _ = BLOCK
    assert len(loops) == 2
    assert areas[-1] == pytest.approx(w * h - NOTCH * NOTCH, rel=1e-3)
    assert areas[0] == pytest.approx(np.pi * HOLE_R**2, rel=2e-2)  # sampled circle, slightly under the true area


def test_step_round_trip(block_step: Path) -> None:
    v = project(load_shape(block_step), (0, 0, 1), (0, 1, 0))
    x0, y0, x1, y1 = v.bounds()
    assert (x1 - x0, y1 - y0) == pytest.approx(BLOCK[:2], abs=1e-6)
    assert len(v.circles) >= 1


def test_stl_fallback_draws_outline_without_facet_comb(block_stl: Path, block_step: Path) -> None:
    """Every visible straight edge of the mesh must lie on a visible edge of the exact solid: a diagonal across a flat face fails."""
    mesh = project(load_shape(block_stl), (0, 0, 1), (0, 1, 0))
    exact = project(load_shape(block_step), (0, 0, 1), (0, 1, 0))
    x0, y0, x1, y1 = mesh.bounds()
    assert (x1 - x0, y1 - y0) == pytest.approx(BLOCK[:2], abs=1e-3)
    assert mesh.circles == []  # a mesh has no true circles
    exact_segments = [(a, b) for line in exact.visible for a, b in zip(line[:-1], line[1:], strict=True)]
    hole = np.array(HOLE_CENTRE)
    stray = []
    for seg in mesh.straight:
        mid = (seg[0] + seg[-1]) / 2
        if abs(np.linalg.norm(mid - hole) - HOLE_R) < 0.1:
            continue  # a facet of the hole's rim
        if min(_point_to_segment(mid, a, b) for a, b in exact_segments) > 1e-3:
            stray.append(seg.tolist())
    assert not stray, f"{len(stray)} mesh edges cross flat faces, e.g. {stray[:2]}"


def test_curved_mesh_keeps_its_silhouette(tmp_path: Path) -> None:
    """A meshed sphere seen from the front must show its whole outline: facet edges on the silhouette stay drawn although the facets meet flat."""
    radius = 10.0
    sphere = BRepPrimAPI_MakeSphere(radius).Shape()
    BRepMesh_IncrementalMesh(sphere, 0.02)
    path = tmp_path / "sphere.stl"
    assert StlAPI_Writer().Write(sphere, str(path))
    view = project(load_shape(path), (0, -1, 0), (0, 0, 1))
    pts = np.vstack(view.visible)
    r = np.hypot(pts[:, 0], pts[:, 1])
    rim = pts[np.abs(r - radius) < 0.1]
    angles = np.degrees(np.arctan2(rim[:, 1], rim[:, 0]))
    covered = len(np.unique(np.floor(angles / 10)))  # 36 ten-degree sectors round the outline
    assert covered == 36, f"outline covers only {covered} of 36 ten-degree sectors"


def test_unsupported_format(tmp_path: Path) -> None:
    bad = tmp_path / "part.obj"
    bad.write_text("")
    with pytest.raises(ValueError, match="unsupported"):
        load_shape(bad)


def test_oversized_mesh_fails_fast_with_a_clear_message(block_stl: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A mesh over the triangle limit is refused up front, naming the file, the count, the limit and what to do, instead of hanging in sewing."""
    import blueprint.geometry as geometry

    monkeypatch.setattr(geometry, "MESH_OCC_MAX_TRIANGLES", 10)
    with pytest.raises(ValueError, match=r"block\.stl has [\d,]+ triangles.*limited to 10.*Decimate.*STEP"):
        load_shape(block_stl)


def test_stl_triangle_count_reads_ascii_and_binary(block_stl: Path, tmp_path: Path) -> None:
    """The count comes from the header for binary files and from the facet records for ASCII, and matches the parsed triangles."""
    from blueprint.mesh import read_stl, stl_triangle_count

    n = stl_triangle_count(block_stl)  # the fixture is ASCII
    assert n == len(read_stl(block_stl)) > 12
    triangles = read_stl(block_stl).astype("<f4")
    binary = tmp_path / "block_bin.stl"
    record = np.zeros(n, dtype=[("normal", "<f4", 3), ("v", "<f4", (3, 3)), ("attr", "<u2")])
    record["v"] = triangles
    binary.write_bytes(b"\0" * 80 + np.uint32(n).tobytes() + record.tobytes())
    assert stl_triangle_count(binary) == n
    assert np.allclose(read_stl(binary), triangles)
