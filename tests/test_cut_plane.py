"""Cutting-plane lines (#11): arrows point the way the section is seen, both ends lettered, and bad input is refused."""

from __future__ import annotations

import numpy as np
import pytest
from matplotlib.text import Annotation

from blueprint import Sheet, View, project, sight_direction
from blueprint.layout import view_directions


def _arrows(sheet: Sheet) -> list[tuple[np.ndarray, np.ndarray]]:
    """(tail, tip) of every arrow drawn, in sheet mm."""
    return [(np.array(a.xyann, float), np.array(a.xy, float)) for a in sheet.ax.texts if isinstance(a, Annotation)]


def _letters(sheet: Sheet, letter: str) -> list[np.ndarray]:
    return [np.array(t.get_position(), float) for t in sheet.ax.texts if t.get_text() == letter and not isinstance(t, Annotation)]


@pytest.mark.parametrize(("sight", "side"), [((1, 0), (1, 0)), ((-1, 0), (-1, 0)), ((0.6, 0.8), (1, 0)), ((-0.3, -5), (-1, 0))])
def test_arrows_point_to_the_sight_side_of_a_vertical_line(sight: tuple[float, float], side: tuple[float, float]) -> None:
    sheet = Sheet("B")
    sheet.cut_plane((100, 50), (100, 150), "A", sight)
    arrows = _arrows(sheet)
    assert len(arrows) == 2
    tails = sorted(tuple(t) for t, _ in arrows)
    assert tails == [(100.0, 50.0), (100.0, 150.0)], "the legs start at the line's ends"
    for tail, tip in arrows:
        leg = tip - tail
        assert np.allclose(leg / np.linalg.norm(leg), side), "each leg is square to the line, on the sight side"
        assert np.linalg.norm(leg) == pytest.approx(sheet.style.spacing.cut_leg)
    letters = _letters(sheet, "A")
    assert len(letters) == 2
    for (_, tip), at in zip(sorted(arrows, key=lambda a: a[1][1]), sorted(letters, key=lambda p: p[1]), strict=True):
        assert np.dot(at - tip, side) > 0, "the letter sits beyond the arrow tip"


def test_bad_lines_are_refused() -> None:
    sheet = Sheet("B")
    with pytest.raises(ValueError, match="no length"):
        sheet.cut_plane((10, 10), (10, 10), "A", (1, 0))
    with pytest.raises(ValueError, match="runs along the line"):
        sheet.cut_plane((10, 10), (10, 50), "A", (0, 1))


def test_view_cut_plane_uses_model_coordinates(block: object) -> None:
    sheet = Sheet("B")
    view = View(sheet, project(block, (0, 0, 1), (0, 1, 0)), (60, 100), 2.0)
    view.cut_plane((12, -5), (12, 35), "A", (1, 0))
    tails = sorted(tuple(t) for t, _ in _arrows(sheet))
    assert tails == [(84.0, 90.0), (84.0, 170.0)]


def test_sight_direction_from_known_frames() -> None:
    # Front view looks along -Z with x = +X, y = +Y. A section seen from the right (its viewer on +X, looking along -X) is seen leftward.
    assert sight_direction((0, 0, 1), (0, 1, 0), (1, 0, 0)) == (-1.0, 0.0)
    # The same section from the left (viewer on -X, looking along +X) is seen rightward; one seen from above (looking down -Y) is seen downward.
    assert sight_direction((0, 0, 1), (0, 1, 0), (-1, 0, 0)) == (1.0, 0.0)
    assert sight_direction((0, 0, 1), (0, 1, 0), (0, 1, 0)) == (0.0, -1.0)
    # Consistent with the layout's own frames: from the FRONT view, the RIGHT view's direction is seen leftward.
    dirs = view_directions("+Z", "+Y")
    (front_t, front_up), (right_t, _) = dirs["FRONT"], dirs["RIGHT"]
    assert sight_direction(front_t, front_up, right_t) == (-1.0, 0.0)
