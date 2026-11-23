"""Section views and detail crops (#12), checked against the notched block's known geometry (conftest), not against the code under test."""

from __future__ import annotations

import numpy as np
import pytest
from matplotlib.patches import Circle, PathPatch

from blueprint import Sheet, View, beyond, project, section_view
from blueprint.geometry import bounding_box
from conftest import BLOCK, HOLE_CENTRE, HOLE_R

W, H, D = BLOCK


def _area(loop: np.ndarray) -> float:
    x, y = loop[:, 0], loop[:, 1]
    return abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))) / 2


@pytest.mark.parametrize(
    ("toward", "depth", "x_range"),
    [((1, 0, 0), None, (0, 12)), ((-1, 0, 0), None, (12, W)), ((1, 0, 0), 5.0, (7, 12)), ((-1, 0, 0), 5.0, (12, 17))],
)
def test_beyond_keeps_the_far_side_of_the_plane(block: object, toward: tuple[int, int, int], depth: float | None, x_range: tuple[float, float]) -> None:
    # A viewer on +X looks along -X, so what lies beyond the plane x = 12 is x < 12; from -X it is x > 12.
    (x0, y0, z0), (x1, y1, z1) = bounding_box(beyond(block, (12, 0, 0), toward, depth))
    assert (x0, x1) == pytest.approx(x_range, abs=1e-6)
    assert (y0, y1, z0, z1) == pytest.approx((0, H, 0, D), abs=1e-6)


def test_section_through_the_hole_has_the_hole_as_a_gap(block: object) -> None:
    proj = section_view(block, (HOLE_CENTRE[0], 0, 0), (1, 0, 0), (0, 1, 0))
    assert proj.hidden == []
    # The plane x = 12 crosses the hole (along Z) across the block's full depth, so the cut face is two strips: below and above the hole.
    areas = sorted(_area(loop) for loop in proj.section)
    below, above = (HOLE_CENTRE[1] - HOLE_R) * D, (H - HOLE_CENTRE[1] - HOLE_R) * D
    assert areas == pytest.approx(sorted([below, above]), rel=1e-3)


def test_section_view_is_hatched_when_drawn(block: object) -> None:
    sheet = Sheet("B")
    View(sheet, section_view(block, (12, 0, 0), (1, 0, 0), (0, 1, 0)), (100, 100), 2.0).draw()
    assert any(isinstance(p, PathPatch) and p.get_hatch() for p in sheet.ax.patches)


def test_a_plane_that_misses_the_part_is_refused(block: object) -> None:
    with pytest.raises(ValueError, match="does not cut the part"):
        section_view(block, (100, 0, 0), (1, 0, 0), (0, 1, 0))
    with pytest.raises(ValueError, match="depth must be positive"):
        beyond(block, (12, 0, 0), (1, 0, 0), 0.0)


def test_cropped_view_is_sized_and_clipped_by_its_circle(block: object) -> None:
    front = project(block, (0, 0, 1), (0, 1, 0))
    detail = front.cropped((30, 20), 6)
    assert detail.bounds() == (24, 14, 36, 26)
    assert detail.circles == [], "the hole (centre 12, 12) is outside the crop"
    assert front.cropped(HOLE_CENTRE, 5).circles == [(pytest.approx(HOLE_CENTRE[0]), pytest.approx(HOLE_CENTRE[1]), pytest.approx(HOLE_R))]
    sheet = Sheet("B")
    first = len(sheet.ax.lines)
    view = View.at_corner(sheet, detail, (200, 100), 3.0).draw()
    drawn = sheet.ax.lines[first:]
    assert drawn and all(line.get_clip_path() is not None for line in drawn), "every line ends on the circle"
    frame = [p for p in sheet.ax.patches if isinstance(p, Circle) and p.radius == pytest.approx(18)]
    assert frame and tuple(frame[0].center) == pytest.approx(tuple(view.to_sheet((30, 20))[0]))
    with pytest.raises(ValueError, match="positive radius"):
        front.cropped((30, 20), 0)


def test_detail_circle_marks_the_parent_view(block: object) -> None:
    sheet = Sheet("B")
    view = View(sheet, project(block, (0, 0, 1), (0, 1, 0)), (50, 100), 2.0)
    view.detail_circle((30, 20), 6, "B")
    (circle,) = [p for p in sheet.ax.patches if isinstance(p, Circle)]
    assert tuple(circle.center) == (110.0, 140.0) and circle.radius == 12.0
    (letter,) = [t for t in sheet.ax.texts if t.get_text() == "B"]
    lx, ly = letter.get_position()
    assert lx > 110 + 12 * 0.7 and ly > 140 + 12 * 0.7, "lettered outside the circle's upper right"


def _distance_to_polylines(pt: np.ndarray, lines: list[np.ndarray]) -> float:
    best = np.inf
    for line in lines:
        a, b = line[:-1], line[1:]
        ab = b - a
        k = np.clip(np.einsum("ij,ij->i", pt - a, ab) / np.maximum(np.einsum("ij,ij->i", ab, ab), 1e-12), 0, 1)
        best = min(best, float(np.min(np.linalg.norm(a + ab * k[:, None] - pt, axis=1))))
    return best


def test_section_hatch_lands_on_the_projection_in_known_places(block: object) -> None:
    # Seen from +X with +Y up, the view's x axis is up x toward = Y x X = -Z, so the block's depth (z 0..20) spans view x -20..0. The cut at
    # x = 12 crosses the hole (y 9..15), leaving a strip below (y 0..9) and above (y 15..30). A mirrored or shifted hatch fails here.
    proj = section_view(block, (12, 0, 0), (1, 0, 0), (0, 1, 0))
    boxes = sorted((float(lp[:, 1].min()), float(lp[:, 1].max()), float(lp[:, 0].min()), float(lp[:, 0].max())) for lp in proj.section)
    lo, hi = HOLE_CENTRE[1] - HOLE_R, HOLE_CENTRE[1] + HOLE_R
    assert boxes == [pytest.approx((0, lo, -D, 0), abs=1e-6), pytest.approx((hi, H, -D, 0), abs=1e-6)]
    for loop in proj.section:
        assert max(_distance_to_polylines(pt, proj.visible) for pt in loop) < 1e-3, "every hatch outline point lies on a drawn edge"


def test_beyond_far_origin_and_oblique_direction(block: object) -> None:
    # The plane x = 12 described by an origin far away in y and z still keeps all of x < 12.
    (x0, y0, z0), (x1, y1, z1) = bounding_box(beyond(block, (12, 1e4, -3e4), (1, 0, 0)))
    assert (x0, x1, y0, y1, z0, z1) == pytest.approx((0, 12, 0, H, 0, D), abs=1e-6)
    # Oblique: keep x + y <= 20 (viewer toward (1, 1, 0)); the kept corner nearest the origin is (0, 0) and nothing has x + y > 20.
    kept = beyond(block, (10, 10, 0), (1, 1, 0))
    (x0, y0, _), (x1, y1, _) = bounding_box(kept)
    assert (x0, y0) == pytest.approx((0, 0), abs=1e-6) and x1 == pytest.approx(20, abs=1e-6) and y1 == pytest.approx(20, abs=1e-6)


def test_direction_arrays_are_not_modified_and_need_not_be_unit(block: object) -> None:
    toward = np.array([2.0, 0.0, 0.0])
    beyond(block, (12, 0, 0), toward)
    assert toward.tolist() == [2.0, 0.0, 0.0]
    long_dirs = section_view(block, (12, 0, 0), (3, 0, 0), (0, 5, 0))
    unit_dirs = section_view(block, (12, 0, 0), (1, 0, 0), (0, 1, 0))
    assert long_dirs.bounds() == pytest.approx(unit_dirs.bounds())


def test_plane_clear_of_the_near_side_is_refused_as_missing(block: object) -> None:
    with pytest.raises(ValueError, match="does not cut the part"):
        section_view(block, (-100, 0, 0), (1, 0, 0), (0, 1, 0))


def test_cropped_section_clips_its_hatch(block: object) -> None:
    sheet = Sheet("B")
    first = len(sheet.ax.patches)
    View.at_corner(sheet, section_view(block, (12, 0, 0), (1, 0, 0), (0, 1, 0)).cropped((-10, 5), 4), (100, 100), 3.0).draw()
    hatches = [p for p in sheet.ax.patches[first:] if isinstance(p, PathPatch) and p.get_hatch()]
    assert hatches and all(p.get_clip_path() is not None for p in hatches)
