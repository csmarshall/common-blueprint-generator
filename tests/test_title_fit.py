"""Title-block text fits its cell (#13): long values shrink, then wrap, then shrink further, never cut (SVG and PNG); short values are untouched."""

from __future__ import annotations

from typing import cast

import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.font_manager import FontProperties
from matplotlib.text import Text
from matplotlib.textpath import TextPath

from blueprint.licences import lookup
from blueprint.sheet import CELL_PAD, PNG_DPI, TITLE_BLOCK_W, TITLE_ROW_H, Sheet, TitleInfo
from blueprint.style import PT_PER_MM

LONG = {
    "material": "PETG OR ABS, 4 WALLS, 40% GYROID INFILL, 0.2 MM LAYERS; ABS IF IT SITS WITHIN 300 MM OF THE ENCLOSURE HEATER, OTHERWISE PETG",
    "for_": "Filament rack add-on: shared-wall bracket between two adjacent spool rods, staggered layout, both ends, left and right hand",
    "source": "brackets/parts.py · Params(shared_wall=True, socket_pitch=38.5, bore=8.2, wall=2.4, chamfer=0.6, clip=True)",
}


def _info(**kw: str) -> TitleInfo:
    return TitleInfo(title="T", drawing_number="D-1", drawn_by="C. MARSHALL", licence=lookup("MIT"), **kw)  # type: ignore[arg-type]


def _texts_in_title_block(sheet: Sheet) -> list[tuple[Text, tuple[float, float, float, float]]]:
    """Every lettering artist inside the title block, with its box in sheet mm as the PNG renders it."""
    sheet.fig.set_dpi(PNG_DPI)
    renderer = cast(FigureCanvasAgg, sheet.fig.canvas).get_renderer()  # type: ignore[no-untyped-call]
    x0, y0, x1, y1 = sheet.inner[2] - TITLE_BLOCK_W, sheet.inner[1], sheet.inner[2], sheet.title_block_box[3]
    found = []
    for t in sheet.ax.texts:
        bb = t.get_window_extent(renderer).transformed(sheet.ax.transData.inverted())
        if x0 <= bb.x0 < x1 and y0 <= bb.y0 < y1:
            found.append((t, (bb.x0, bb.y0, bb.x1, bb.y1)))
    return found


def test_scale_with_several_detail_scales_reads_as_one_group() -> None:
    sheet = Sheet("B")
    sheet.title_block(_info(), 0.75, detail_scales=(1.5, 3.5))
    assert "1:1.33 · DETAILS 1.5:1, 3.5:1" in [t.get_text() for t in sheet.ax.texts]
    one = Sheet("B")
    one.title_block(_info(), 0.75, detail_scales=(1.5,))
    assert "1:1.33 · DETAIL 1.5:1" in [t.get_text() for t in one.ax.texts]


def test_long_values_stay_inside_the_title_block() -> None:
    sheet = Sheet("B")
    sheet.title_block(_info(**LONG), 0.75, detail_scales=(1.5, 3.5))
    right = sheet.inner[2]
    texts = _texts_in_title_block(sheet)
    assert texts, "the title block's lettering was found"
    for t, (_, _, x1, _) in texts:
        assert x1 <= right, f"{t.get_text()!r} runs past the title block"
    for t, (_, y0, _, y1) in texts:
        assert y1 - y0 <= TITLE_ROW_H, f"{t.get_text()!r} is taller than its row"
    material, (_, _, _, top) = next((t, b) for t, b in texts if t.get_text().startswith("PETG"))
    _, (_, label_bottom, _, _) = next((t, b) for t, b in texts if t.get_text() == "MATERIAL")
    assert top <= label_bottom, "wrapped lines stay below their cell's label"
    assert "\n" in material.get_text(), "wrapped"
    assert material.get_text().split() == LONG["material"].split(), "every word is shown: nothing is cut"


def test_fit_text_keeps_short_text_and_shrinks_before_wrapping() -> None:
    sheet = Sheet("B")
    assert sheet.fit_text("MM", 50, 5, 6.5, 4.6) == (["MM"], 6.5)
    width = sheet.text_width("ABCDEFGHIJ", 6.5)
    lines, size = sheet.fit_text("ABCDEFGHIJ", width * 0.9, 5, 6.5, 4.6)
    assert lines == ["ABCDEFGHIJ"] and 4.6 <= size < 6.5 and sheet.text_width("ABCDEFGHIJ", size) <= width * 0.9
    lines, size = sheet.fit_text("ONE TWO THREE FOUR", sheet.text_width("ONE TWO", 4.6) + 0.1, 50, 6.5, 4.6)
    assert size == 4.6 and lines == ["ONE TWO", "THREE", "FOUR"]


@pytest.mark.parametrize("size", [4.6, 5.16, 5.5, 6.5])
def test_text_width_covers_the_raster_as_well_as_the_outline(size: float) -> None:
    sheet = Sheet("B")
    s = "PETG, 4 WALLS, 40% GYROID; ABS IF NEAR THE HEATER"
    sheet.fig.set_dpi(PNG_DPI)
    probe = sheet.ax.text(0, 0, s, fontsize=size, fontfamily=sheet.style.type.family)
    bb = probe.get_window_extent(cast(FigureCanvasAgg, sheet.fig.canvas).get_renderer()).transformed(sheet.ax.transData.inverted())  # type: ignore[no-untyped-call]
    assert sheet.text_width(s, size) >= bb.x1 - bb.x0 - 1e-9


def test_text_too_long_for_label_size_shrinks_further_and_warns(caplog: pytest.LogCaptureFixture) -> None:
    sheet = Sheet("B")
    text = "WORD " * 40
    lines, size = sheet.fit_text(text, 50, 5, 6.5, 4.6)
    assert size < 4.6 and " ".join(lines).split() == text.split()
    assert len(lines) * sheet.text_height(size) <= 5 and all(sheet.text_width(line, size) <= 50 for line in lines)
    assert "below the 4.6 pt label size" in caplog.text


def test_whitespace_values_render() -> None:
    sheet = Sheet("B")
    sheet.title_block(_info(material=" ", source="  "), 1.0)
    assert sheet.text_width("   ", 6.5) > 0


def test_fit_text_refuses_no_space_and_always_returns() -> None:
    sheet = Sheet("B")
    for width, height in ((0, 5), (-1, 5), (10, 0), (10, -1)):
        with pytest.raises(ValueError, match="positive width and height"):
            sheet.fit_text("ABC", width, height, 6.5, 4.6)
    lines, size = sheet.fit_text("ABC DEF", 1e-3, 5, 6.5, 4.6)  # cannot fit at any readable size: gives up, shows everything
    assert " ".join(lines) == "ABC DEF" and size > 0


def test_fit_text_never_enlarges() -> None:
    sheet = Sheet("B")
    text = "WORD " * 30
    _, size = sheet.fit_text(text, 40, 50, 4.0, 4.6)
    assert size <= 4.0


def test_text_width_restores_dpi_and_covers_the_outline() -> None:
    sheet = Sheet("B")
    dpi = sheet.fig.dpi
    s = "PETG, 4 WALLS, 40% GYROID; ABS IF NEAR THE HEATER"
    outline = TextPath((0, 0), s, size=6.5, prop=FontProperties(family=sheet.style.type.family)).get_extents().x1 / PT_PER_MM
    assert sheet.text_width(s, 6.5) >= outline  # at 6.5 pt the outline (the SVG) is wider than the raster
    assert sheet.fig.dpi == dpi, "measuring leaves the figure's dpi as it was"
    assert not [t for t in sheet.ax.texts if t.get_text() == s], "the probe is removed"


def test_cell_padding_is_kept() -> None:
    # A value that would only fit by touching the cell's edges is shrunk so it clears them by CELL_PAD.
    sheet = Sheet("B")
    third = TITLE_BLOCK_W / 3
    text = "X" * 40
    size = 6.5 * (third - CELL_PAD) / sheet.text_width(text, 6.5)  # fits the cell less one pad, not less two
    lines, fitted = sheet.fit_text(text, third - 2 * CELL_PAD, 5, size, 4.6)
    assert sheet.text_width(lines[0], fitted) <= third - 2 * CELL_PAD < sheet.text_width(text, size)
