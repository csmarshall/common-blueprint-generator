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
from matplotlib import colors as mcolors
from matplotlib.lines import Line2D

from blueprint import Sheet, View, project, section_view
from blueprint.config import load
from blueprint.layout import auto_sheet, view_directions
from blueprint.sheet import CentreLine
from blueprint.style import PT_PER_MM

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


def _label(sheet: Sheet, text: str) -> object:
    (t,) = [t for t in sheet.ax.texts if t.get_text() == text]
    return t


def test_label_of_a_vertical_line_sits_just_inside_the_top_edge(block: object) -> None:
    toward, up = view_directions("+Z", "+Y")["FRONT"]
    sheet = Sheet("B")
    View(sheet, project(block, toward, up), ORIGIN, SCALE).centre_line((30, 0, 5), (0, 1, 0), "SOCKET AXIS")
    t = _label(sheet, "CL SOCKET AXIS")
    x, y = t.get_position()  # type: ignore[attr-defined]
    # The line is at view x 30, in the right half of the 0..40 box, so the label hangs to its left; the box top is view y 30.
    assert t.get_ha() == "right" and t.get_va() == "top"  # type: ignore[attr-defined]
    assert (x, y) == pytest.approx((_sheet_x(30) - OVERRUN, _sheet_y(30) - OVERRUN))


def test_label_of_a_line_in_the_left_half_hangs_to_its_right(block: object) -> None:
    toward, up = view_directions("+Z", "+Y")["FRONT"]
    sheet = Sheet("B")
    View(sheet, project(block, toward, up), ORIGIN, SCALE).centre_line((8, 0, 5), (0, 1, 0), "A")
    t = _label(sheet, "CL A")
    assert t.get_ha() == "left"  # type: ignore[attr-defined]
    assert t.get_position() == pytest.approx((_sheet_x(8) + OVERRUN, _sheet_y(30) - OVERRUN))  # type: ignore[attr-defined]


def test_label_of_a_horizontal_line_sits_inside_its_right_end_above_it(block: object) -> None:
    toward, up = view_directions("+Z", "+Y")["RIGHT"]  # view (x, y) = (-Z, Y): the block's box is x -20..0, y 0..30
    sheet = Sheet("B")
    View(sheet, project(block, toward, up), ORIGIN, SCALE).centre_line((12, 12, 0), (0, 0, 1), "HOLE AXIS")
    t = _label(sheet, "CL HOLE AXIS")
    assert t.get_ha() == "right" and t.get_va() == "bottom"  # type: ignore[attr-defined]
    assert t.get_position() == pytest.approx((_sheet_x(0) - OVERRUN, _sheet_y(12) + OVERRUN))  # type: ignore[attr-defined]


def test_every_label_stays_inside_its_own_views_box(block: object) -> None:
    sheet = Sheet("B")
    layout = auto_sheet(
        sheet,
        block,
        views=("FRONT", "TOP", "BOTTOM", "LEFT", "RIGHT", "BACK"),
        centre_lines=[CentreLine("HOLE AXIS", (12, 12, 0), (0, 0, 1)), CentreLine("OFF-CENTRE", (30, 0, 5), (0, 1, 0)), CentreLine("X", (5, 7, 11), (1, 0, 0))],
    )
    boxes = [v.box() for v in layout.views.values()]
    labels = [t for t in sheet.ax.texts if t.get_text().startswith("CL ")]
    assert len(labels) >= 10
    for t in labels:
        x, y = t.get_position()
        assert any(b[0] <= x <= b[2] and b[1] <= y <= b[3] for b in boxes), f"{t.get_text()} anchor {x, y} is outside every view's outline"


def test_non_unit_frame_gives_the_same_line(block: object) -> None:
    sheet, ref = Sheet("B"), Sheet("B")
    View(sheet, project(block, (0, 0, 2), (0, 3, 0)), ORIGIN, SCALE).centre_line((30, 0, 5), (0, 1, 0))
    View(ref, project(block, (0, 0, 1), (0, 1, 0)), ORIGIN, SCALE).centre_line((30, 0, 5), (0, 1, 0))
    assert np.allclose(sheet.ax.lines[-1].get_xydata(), ref.ax.lines[-1].get_xydata())
    assert sheet.ax.lines[-1].get_xdata() == pytest.approx([_sheet_x(30)] * 2)  # type: ignore[arg-type]


def test_scaled_projection_keeps_the_line_on_the_geometry(block: object) -> None:
    sheet = Sheet("B")
    proj = project(block, (0, 0, 1), (0, 1, 0)).scaled(2.0)  # the view is now 80 x 60 view-mm
    View(sheet, proj, ORIGIN, 1.0).centre_line((30, 0, 5), (0, 1, 0))
    line = sheet.ax.lines[-1]
    assert line.get_xdata() == pytest.approx([ORIGIN[0] + 60] * 2)  # type: ignore[arg-type]
    assert (min(line.get_ydata()), max(line.get_ydata())) == pytest.approx((ORIGIN[1] - OVERRUN, ORIGIN[1] + 60 + OVERRUN))  # type: ignore[type-var]


def test_cropped_detail_clips_line_and_label_to_the_circle(block: object) -> None:
    sheet = Sheet("B")
    proj = project(block, (0, 0, 1), (0, 1, 0)).cropped((30, 15), 8.0)
    before = len(sheet.ax.lines)
    assert View(sheet, proj, ORIGIN, SCALE).centre_line((30, 0, 5), (0, 1, 0), "AX")
    (line,) = sheet.ax.lines[before:]
    text = _label(sheet, "CL AX")
    assert line.get_clip_path() is not None
    assert text.get_clip_path() is not None and text.get_clip_on()  # type: ignore[attr-defined]


def test_uncropped_view_sets_no_clip_path(block: object) -> None:
    sheet = Sheet("B")
    View(sheet, project(block, (0, 0, 1), (0, 1, 0)), ORIGIN, SCALE).centre_line((30, 0, 5), (0, 1, 0), "AX")
    assert sheet.ax.lines[-1].get_clip_path() is None


def test_other_front_and_up_pair(block: object) -> None:
    # front +X, up +Z, by hand: r = up x front = Z x X = +Y.
    #   FRONT toward +X, up +Z:  x_dir = Z x X = +Y   so view (x, y) = ( Y,  Z)
    #   RIGHT toward +Y, up +Z:  x_dir = Z x Y = -X   so view (x, y) = (-X,  Z)
    #   TOP   toward +Z, up -X:  x_dir = -X x Z = +Y  so view (x, y) = ( Y, -X)
    # An X-direction axis through (5, 7, 11) points at the front viewer; in RIGHT it is horizontal at y = Z = 11 across x = -X in -40..0;
    # in TOP it is vertical at x = Y = 7 across y = -X in -40..0.
    dirs = view_directions("+X", "+Z")
    assert dirs["RIGHT"][0] == (0.0, 1.0, 0.0) or tuple(dirs["RIGHT"][0]) == (0, 1, 0)
    results = {}
    for name in ("FRONT", "RIGHT", "TOP"):
        sheet = Sheet("B")
        proj = project(block, *dirs[name])
        before = len(sheet.ax.lines)
        drawn = View(sheet, proj, ORIGIN, SCALE).centre_line((5, 7, 11), (1, 0, 0))
        results[name] = (drawn, sheet.ax.lines[before:])
    assert results["FRONT"] == (False, [])
    (right,) = results["RIGHT"][1]
    assert right.get_ydata() == pytest.approx([_sheet_y(11)] * 2)  # type: ignore[arg-type]
    assert (min(right.get_xdata()), max(right.get_xdata())) == pytest.approx((_sheet_x(-40) - OVERRUN, _sheet_x(0) + OVERRUN))  # type: ignore[type-var]
    (top,) = results["TOP"][1]
    assert top.get_xdata() == pytest.approx([_sheet_x(7)] * 2)  # type: ignore[arg-type]
    assert (min(top.get_ydata()), max(top.get_ydata())) == pytest.approx((_sheet_y(-40) - OVERRUN, _sheet_y(0) + OVERRUN))  # type: ignore[type-var]


def test_section_view_places_axes_in_its_own_frame(block: object) -> None:
    # Section at x = 12 seen from +X, up +Y: x_dir = Y x X = -Z, so view (x, y) = (-Z, Y). The material beyond the plane (x < 12) is the full
    # 20 deep and 30 tall. A Y axis through z = 8 stands at view x = -8; a Z axis through y = 12 lies across at view y = 12 over x -20..0.
    sheet = Sheet("B")
    view = View(sheet, section_view(block, (12, 0, 0), (1, 0, 0), (0, 1, 0)), ORIGIN, SCALE)
    assert view.centre_line((5, 0, 8), (0, 1, 0)) and view.centre_line((0, 12, 0), (0, 0, 1))
    vertical, horizontal = sheet.ax.lines[-2], sheet.ax.lines[-1]
    assert vertical.get_xdata() == pytest.approx([_sheet_x(-8)] * 2)  # type: ignore[arg-type]
    assert (min(vertical.get_ydata()), max(vertical.get_ydata())) == pytest.approx((_sheet_y(0) - OVERRUN, _sheet_y(30) + OVERRUN))  # type: ignore[type-var]
    assert horizontal.get_ydata() == pytest.approx([_sheet_y(12)] * 2)  # type: ignore[arg-type]
    assert (min(horizontal.get_xdata()), max(horizontal.get_xdata())) == pytest.approx((_sheet_x(-20) - OVERRUN, _sheet_x(0) + OVERRUN))  # type: ignore[type-var]


def test_auto_sheet_warns_about_a_centre_line_drawn_nowhere(block: object, caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level("WARNING", logger="blueprint.layout"):
        auto_sheet(
            Sheet("B"), block, views=("FRONT",), centre_lines=[CentreLine("LOST AXIS", (12, 12, 0), (0, 0, 1)), CentreLine("MISSED", (100, 0, 0), (0, 1, 0))]
        )
    warned = [r.getMessage() for r in caplog.records]
    assert any("LOST AXIS" in m for m in warned) and any("MISSED" in m for m in warned)


def test_auto_sheet_is_quiet_when_every_centre_line_draws(block: object, caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level("WARNING", logger="blueprint.layout"):
        auto_sheet(Sheet("B"), block, centre_lines=[CentreLine("OK", (30, 0, 5), (0, 1, 0))])
    assert caplog.records == []


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


def test_label_has_a_paper_knockout_above_the_lines_and_clipped_on_a_detail(block: object) -> None:
    sheet = Sheet("B")
    view = View(sheet, project(block, (0, 0, 1), (0, 1, 0)), ORIGIN, SCALE)
    view.draw()
    view.centre_line((30, 0, 5), (0, 1, 0), "AX")
    text = _label(sheet, "CL AX")
    patch = text.get_bbox_patch()  # type: ignore[attr-defined]
    assert patch is not None and patch.get_edgecolor()[3] == 0
    assert patch.get_facecolor() == pytest.approx(mcolors.to_rgba(sheet.style.palette.paper))
    assert text.get_zorder() > max(ln.get_zorder() for ln in sheet.ax.lines), "drawn after every line, with the text on top of its own patch"
    pad_pt = sheet.style.spacing.label_knockout * PT_PER_MM
    assert text.get_bbox_patch().get_boxstyle().pad * sheet.style.type.dim == pytest.approx(pad_pt)  # type: ignore[attr-defined]
    crop = Sheet("B")
    View(crop, project(block, (0, 0, 1), (0, 1, 0)).cropped((30, 15), 8.0), ORIGIN, SCALE).centre_line((30, 0, 5), (0, 1, 0), "AX")
    cropped = _label(crop, "CL AX")
    cropped.draw(crop.fig.canvas.get_renderer())  # type: ignore[attr-defined]  # the patch takes the text's clip when it is laid out
    assert cropped.get_bbox_patch().get_clip_path() is not None  # type: ignore[attr-defined]
