"""Centre lines (#2): a named model axis lands where the part's own edges do in every view, and an axis pointing at the viewer is not drawn.

Expected positions are worked out by hand from the notched block's fixed dimensions (tests/conftest.py: 40 x 30 x 20, X right, Y up, Z toward the
front viewer) and from each view's frame, x = up x toward_viewer; they never call the projection code to say where an axis should go.

Frames with front +Z, up +Y, worked by hand:
  FRONT  toward +Z, up +Y:  x_dir = Y x Z = +X      so view (x, y) = ( X,  Y)
  BACK   toward -Z, up +Y:  x_dir = Y x -Z = -X     so view (x, y) = (-X,  Y)
  RIGHT  toward +X, up +Y:  x_dir = Y x X = -Z      so view (x, y) = (-Z,  Y)
  LEFT   toward -X, up +Y:  x_dir = Y x -X = +Z     so view (x, y) = ( Z,  Y)
  TOP    toward +Y, up -Z:  x_dir = -Z x Y = +X     so view (x, y) = ( X, -Z)
  BOTTOM toward -Y, up +Z:  x_dir = Z x -Y = +X     so view (x, y) = ( X,  Z)
"""

from __future__ import annotations

import numpy as np
import pytest
from matplotlib.lines import Line2D

from blueprint import Sheet, View, project
from blueprint.config import load
from blueprint.layout import auto_sheet, view_directions
from blueprint.sheet import CentreLine

ORIGIN = (60.0, 100.0)
SCALE = 2.0
OVERRUN = Sheet("B").style.spacing.centre_overrun


def _chain_lines(sheet: Sheet, before: int) -> list[Line2D]:
    return sheet.ax.lines[before:]


def _draw(block: object, name: str, point: tuple[float, float, float], axis: tuple[float, float, float], label: str = "") -> tuple[bool, list[np.ndarray]]:
    toward, up = view_directions("+Z", "+Y")[name]
    sheet = Sheet("B")
    view = View(sheet, project(block, toward, up), ORIGIN, SCALE)
    before = len(sheet.ax.lines)
    drawn = view.centre_line(point, axis, label)
    return drawn, [np.column_stack([ln.get_xdata(), ln.get_ydata()]) for ln in _chain_lines(sheet, before)]


def _sheet_x(view_x: float) -> float:
    return ORIGIN[0] + SCALE * view_x


def _sheet_y(view_y: float) -> float:
    return ORIGIN[1] + SCALE * view_y


# A Y-direction axis through model (30, 0, 5): inside the block's width (X 0..40) and depth (Z 0..20), off every centre.
@pytest.mark.parametrize(
    ("view", "view_x"),
    [("FRONT", 30.0), ("BACK", -30.0), ("RIGHT", -5.0), ("LEFT", 5.0)],
)
def test_vertical_axis_lands_at_hand_worked_x(block: object, view: str, view_x: float) -> None:
    drawn, lines = _draw(block, view, (30, 0, 5), (0, 1, 0))
    assert drawn and len(lines) == 1
    (xs, ys) = lines[0][:, 0], lines[0][:, 1]
    assert xs == pytest.approx([_sheet_x(view_x)] * 2), f"{view}: a vertical line at view x {view_x}"
    assert min(ys) == pytest.approx(_sheet_y(0) - OVERRUN), "the part is 30 tall: the line spans it plus the overrun each end"
    assert max(ys) == pytest.approx(_sheet_y(30) + OVERRUN)


# A Z-direction axis through model (12, 12, 0): the hole's axis.
@pytest.mark.parametrize(
    ("view", "view_x", "view_y", "vertical"),
    [
        ("TOP", 12.0, None, True),  # x = X = 12; runs along y = -Z, over the 20 mm depth
        ("BOTTOM", 12.0, None, True),  # x = X = 12; y = Z
        ("RIGHT", None, 12.0, False),  # y = Y = 12; runs along x = -Z
        ("LEFT", None, 12.0, False),  # y = Y = 12; runs along x = Z
    ],
)
def test_z_axis_lands_at_hand_worked_position(block: object, view: str, view_x: float | None, view_y: float | None, vertical: bool) -> None:
    drawn, lines = _draw(block, view, (12, 12, 0), (0, 0, 1))
    assert drawn and len(lines) == 1
    xs, ys = lines[0][:, 0], lines[0][:, 1]
    if vertical:
        assert view_x is not None
        assert xs == pytest.approx([_sheet_x(view_x)] * 2)
        assert (max(ys) - min(ys)) == pytest.approx(20 * SCALE + 2 * OVERRUN), "spans the 20 mm depth plus overrun"
    else:
        assert view_y is not None
        assert ys == pytest.approx([_sheet_y(view_y)] * 2)
        assert (max(xs) - min(xs)) == pytest.approx(20 * SCALE + 2 * OVERRUN)


def test_depth_extent_runs_the_right_way_in_top_and_right(block: object) -> None:
    # In TOP, y = -Z: the block's Z 0..20 occupies view y -20..0, so the line spans sheet y from 100 - 40 - overrun to 100 + overrun.
    _, top = _draw(block, "TOP", (12, 12, 0), (0, 0, 1))
    assert (min(top[0][:, 1]), max(top[0][:, 1])) == pytest.approx((_sheet_y(-20) - OVERRUN, _sheet_y(0) + OVERRUN))
    # In RIGHT, x = -Z: the same Z 0..20 occupies view x -20..0.
    _, right = _draw(block, "RIGHT", (12, 12, 0), (0, 0, 1))
    assert (min(right[0][:, 0]), max(right[0][:, 0])) == pytest.approx((_sheet_x(-20) - OVERRUN, _sheet_x(0) + OVERRUN))
    # In LEFT, x = +Z: the opposite side.
    _, left = _draw(block, "LEFT", (12, 12, 0), (0, 0, 1))
    assert (min(left[0][:, 0]), max(left[0][:, 0])) == pytest.approx((_sheet_x(0) - OVERRUN, _sheet_x(20) + OVERRUN))


@pytest.mark.parametrize(
    ("view", "axis"), [("FRONT", (0, 0, 1)), ("BACK", (0, 0, -3)), ("TOP", (0, 1, 0)), ("BOTTOM", (0, -1, 0)), ("RIGHT", (1, 0, 0)), ("LEFT", (2, 0, 0))]
)
def test_axis_pointing_at_the_viewer_is_not_drawn(block: object, view: str, axis: tuple[float, float, float]) -> None:
    drawn, lines = _draw(block, view, (12, 12, 5), axis, "END ON")
    assert not drawn and lines == []


def test_a_line_that_misses_the_outline_is_not_drawn(block: object) -> None:
    drawn, lines = _draw(block, "FRONT", (100, 0, 0), (0, 1, 0))
    assert not drawn and lines == []


def test_oblique_axis_is_clipped_to_the_outline_box(block: object) -> None:
    # In FRONT (view = X, Y) the line through (0, 0) along (1, 1) leaves the 40 x 30 box at (30, 30).
    drawn, lines = _draw(block, "FRONT", (0, 0, 7), (1, 1, 0))
    assert drawn
    (a, b) = sorted(map(tuple, lines[0]))
    k = OVERRUN / np.hypot(1, 1)
    assert a == pytest.approx((_sheet_x(0) - k, _sheet_y(0) - k))
    assert b == pytest.approx((_sheet_x(30) + k, _sheet_y(30) + k))


def test_label_reads_cl_and_sits_past_the_end(block: object) -> None:
    toward, up = view_directions("+Z", "+Y")["FRONT"]
    sheet = Sheet("B")
    View(sheet, project(block, toward, up), ORIGIN, SCALE).centre_line((30, 0, 5), (0, 1, 0), "SOCKET AXIS")
    (t,) = [t for t in sheet.ax.texts if t.get_text() == "CL SOCKET AXIS"]
    x, y = t.get_position()
    assert x == pytest.approx(_sheet_x(30))
    assert y == pytest.approx(_sheet_y(30) + 2 * OVERRUN), "above the top end of a vertical line, clear of its overrun"


def test_bad_input_is_refused(block: object) -> None:
    for point, axis in (((1, 2, float("nan")), (0, 0, 1)), ((1, 2, 3), (0, 0, 0)), ((1, 2, 3), (float("inf"), 0, 0))):
        with pytest.raises(ValueError, match="finite|direction"):
            CentreLine("X", point, axis)
        toward, up = view_directions("+Z", "+Y")["FRONT"]
        with pytest.raises(ValueError, match="finite|direction"):
            View(Sheet("B"), project(block, toward, up), ORIGIN, SCALE).centre_line(point, axis)


def test_auto_sheet_draws_on_orthographic_views_not_the_isometric(block: object) -> None:
    plain = Sheet("B")
    auto_sheet(plain, block)
    with_cl = Sheet("B")
    auto_sheet(with_cl, block, centre_lines=[CentreLine("AXIS", (30, 0, 5), (0, 1, 0))])
    labels = [t.get_text() for t in with_cl.ax.texts if t.get_text().startswith("CL ")]
    # FRONT and RIGHT see a Y axis as a line; TOP looks down it. The isometric gets none.
    assert labels == ["CL AXIS", "CL AXIS"]
    assert len(with_cl.ax.lines) > len(plain.ax.lines)


def test_no_centre_lines_leaves_the_sheet_unchanged(block: object) -> None:
    a, b = Sheet("B"), Sheet("B")
    auto_sheet(a, block)
    auto_sheet(b, block, centre_lines=[])
    assert len(a.ax.lines) == len(b.ax.lines) and len(a.ax.texts) == len(b.ax.texts)


def _cfg(tmp_path: object, extra: str) -> object:
    from pathlib import Path

    path = Path(str(tmp_path)) / "blueprint.yml"
    path.write_text(f"model: m.step\noutput: out/s\ntitle: T\nlicence: MIT\n{extra}")
    return path


def test_config_reads_centre_lines(tmp_path: object) -> None:
    cfg = load(_cfg(tmp_path, "centre_lines:\n  - {label: SOCKET AXIS, point: [12, 12, 0], axis: [0, 0, 1]}\n"))  # type: ignore[arg-type]
    assert cfg.centre_lines == (CentreLine("SOCKET AXIS", (12, 12, 0), (0, 0, 1)),)


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        ("centre_lines: nope\n", "list of"),
        ("centre_lines: [{label: A, point: [1, 2, 3]}]\n", "exactly label, point and axis"),
        ("centre_lines: [{label: A, point: [1, 2], axis: [0, 0, 1]}]\n", "three numbers"),
        ("centre_lines: [{label: A, point: [1, 2, 3], axis: [0, 0, 0]}]\n", "no direction"),
        ("centre_lines: [{label: A, point: [1, 2, .nan], axis: [0, 0, 1]}]\n", "finite"),
        ("centre_lines: [{label: A, point: [1, 2, x], axis: [0, 0, 1]}]\n", "finite"),
        ("centre_lines: [{label: 3, point: [1, 2, 3], axis: [0, 0, 1]}]\n", "label is text"),
    ],
)
def test_config_refuses_bad_centre_lines(tmp_path: object, extra: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        load(_cfg(tmp_path, extra))  # type: ignore[arg-type]
