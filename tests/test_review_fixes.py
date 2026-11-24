"""Regression tests for the 2026-11-18 review of main (issue #1): exact curve bounds, layouts without FRONT, annotation clearance measured from the
rendered text (not from the boxes the layout itself computes), section frame, config validation, licences, output names with dots."""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pytest
from matplotlib.text import Text
from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeCylinder

from blueprint.cli import main
from blueprint.config import load
from blueprint.geometry import project, section_loops
from blueprint.layout import ALL_VIEWS, auto_sheet
from blueprint.licences import LICENCES, icon, lookup
from blueprint.readme import embed
from blueprint.sheet import Sheet, TitleInfo
from blueprint.style import SHEET_SIZES

Box = tuple[float, float, float, float]


def _overlap(a: Box, b: Box) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def _text_boxes(sheet: Sheet, skip: int = 0) -> list[Box]:
    """Texts drawn after the first `skip`, as measured by the renderer, in sheet millimetres."""
    renderer = sheet.fig.canvas.get_renderer()  # type: ignore[attr-defined]
    to_data = sheet.ax.transData.inverted()
    boxes = []
    for artist in [t for t in sheet.ax.get_children() if isinstance(t, Text)][skip:]:
        if artist.get_text():
            ext = artist.get_window_extent(renderer)
            (x0, y0), (x1, y1) = to_data.transform([(ext.x0, ext.y0), (ext.x1, ext.y1)])
            boxes.append((float(x0), float(y0), float(x1), float(y1)))
    return boxes


def test_curved_outline_bounds_are_exact() -> None:
    cylinder = BRepPrimAPI_MakeCylinder(10, 5).Shape()
    x0, y0, x1, y1 = project(cylinder, (0, 0, 1), (0, 1, 0)).bounds()
    assert (x1 - x0, y1 - y0) == pytest.approx((20.0, 20.0), abs=1e-9)


def test_section_points_share_the_projection_frame(block: object) -> None:
    """Moving the cut origin within its plane must not move the outline (it used to shift by the origin)."""
    at_zero = section_loops(block, (0, 0, 10), (0, 0, 1), (1, 0, 0))
    off_centre = section_loops(block, (20, 15, 10), (0, 0, 1), (1, 0, 0))
    outer = lambda loops: max(loops, key=lambda loop: float(np.ptp(loop[:, 0])))  # noqa: E731  # widest loop: a sampled hole has more points
    assert outer(off_centre)[:, 0].min() == pytest.approx(outer(at_zero)[:, 0].min())
    assert outer(off_centre)[:, 1].min() == pytest.approx(outer(at_zero)[:, 1].min())
    proj = project(block, (0, 0, 1), (0, 1, 0)).bounds()
    assert outer(at_zero)[:, 0].min() == pytest.approx(proj[0]) and outer(at_zero)[:, 1].max() == pytest.approx(proj[3])


def test_section_rejects_a_bad_x_dir(block: object) -> None:
    with pytest.raises(ValueError, match="unit vector perpendicular"):
        section_loops(block, (0, 0, 10), (0, 0, 1), (0, 0, 1))


SHAPES = {
    "block": None,  # filled from the fixture
    "slab": lambda: BRepPrimAPI_MakeBox(200, 200, 5).Shape(),
    "rod": lambda: BRepPrimAPI_MakeBox(5.7, 34.5, 250.1).Shape(),
}
SUBSETS = [
    ("TOP", "RIGHT"),
    ("TOP", "ISOMETRIC"),
    ("TOP", "RIGHT", "ISOMETRIC"),
    ("FRONT", "RIGHT", "ISOMETRIC"),
    ("FRONT", "TOP", "RIGHT"),
    ("LEFT", "BACK"),
    ("BOTTOM", "TOP"),
    tuple(ALL_VIEWS),
    ("FRONT", "TOP", "RIGHT", "ISOMETRIC"),
]


@pytest.mark.parametrize(("shape_name", "views", "size"), list(itertools.product(SHAPES, SUBSETS, sorted(SHEET_SIZES))))
def test_views_and_their_text_stay_clear(block: object, shape_name: str, views: tuple[str, ...], size: str) -> None:
    shape = block if shape_name == "block" else SHAPES[shape_name]()  # type: ignore[misc]
    sheet = Sheet(size)
    furniture = sum(isinstance(t, Text) for t in sheet.ax.get_children())
    layout = auto_sheet(sheet, shape, views=views)
    x0, y0, x1, y1 = sheet.inner
    keep_out = [sheet.title_block_box, sheet.rev_block_box(2)]
    boxes = [v.box() for v in layout.views.values()]
    for a, b in itertools.combinations(boxes, 2):
        assert not _overlap(a, b), f"views overlap: {a} {b}"
    for box in boxes + _text_boxes(sheet, skip=furniture):
        assert x0 <= box[0] and box[2] <= x1 and y0 <= box[1] and box[3] <= y1, f"{box} leaves the border {sheet.inner}"
        for zone in keep_out:
            assert not _overlap(box, zone), f"{box} runs into {zone}"


def test_sheet_a_is_not_offered() -> None:
    assert "A" not in SHEET_SIZES  # the title block and tolerance column are wider than A's inside


def _config(tmp_path: Path, extra: str) -> Path:
    cfg = tmp_path / "blueprint.yml"
    cfg.write_text(f"model: block.step\noutput: out/sheet\ntitle: T\nlicence: MIT\n{extra}")
    return cfg


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        ("formats: []\n", "at least one"),
        ('style: {aged: "false"}\n', "true or false"),
        ("licence: {note: x}\n", "needs an SPDX id"),
    ],
)
def test_config_rejects(tmp_path: Path, extra: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        load(_config(tmp_path, extra))


def test_config_rejects_a_non_mapping(tmp_path: Path) -> None:
    cfg = tmp_path / "blueprint.yml"
    cfg.write_text("- model: x\n")
    with pytest.raises(ValueError, match="expected a mapping"):
        load(cfg)


def test_config_accepts_scalars_dedupes_views_and_reads_licences(tmp_path: Path) -> None:
    cfg = load(_config(tmp_path, "views: top\nformats: svg\nlicence: cc-by-nc-sa-4.0\n"))
    assert cfg.views == ("TOP",) and cfg.formats == ("svg",)
    assert cfg.licence is LICENCES["CC-BY-NC-SA-4.0"] and cfg.licence_note is None
    cfg = load(_config(tmp_path, "views: [front, front, top]\nlicence: {id: MIT, note: SCRIPTS ONLY}\n"))
    assert cfg.views == ("FRONT", "TOP") and cfg.licence is LICENCES["MIT"] and cfg.licence_note == "SCRIPTS ONLY"


def test_unknown_licence_prints_as_given() -> None:
    odd = lookup("WTFPL")
    assert odd.name == "WTFPL" and not odd.icons and not odd.noncommercial and odd.url.endswith("/WTFPL.html")
    assert lookup("CC-BY-NC-3.0").noncommercial  # not in the table, but -NC- still means NOT FOR RESALE


def test_licences_icons_and_notes() -> None:
    nc = lookup("CC-BY-NC-SA-4.0")
    assert nc.noncommercial and nc.icons == ("cc", "by", "nc", "sa") and nc.url == "https://creativecommons.org/licenses/by-nc-sa/4.0/"
    assert not lookup("MIT").icons and not lookup("MIT").noncommercial
    for name in {i for lic in LICENCES.values() for i in lic.icons}:
        mark = icon(name)
        x0, y0, x1, y1 = mark.glyph.get_extents().bounds[0], mark.glyph.get_extents().bounds[1], *mark.glyph.get_extents().max
        assert x0 >= 0 and x1 <= 1 and y0 >= 0 and y1 <= 1, f"{name} glyph outside its unit box"
        assert mark.disc_radius > 0.4


@pytest.mark.parametrize("spdx", ["CC-BY-NC-SA-4.0", "MIT"])
def test_licence_row_text_and_marks(spdx: str) -> None:
    sheet = Sheet("B")
    lic = lookup(spdx)
    sheet.title_block(TitleInfo(title="T", licence=lic), 1.0)
    texts = [t.get_text() for t in sheet.ax.texts]
    assert any(lic.name in t for t in texts)
    assert any(lic.url.removeprefix("https://") == t for t in texts), "the deed or licence URL must be printed"
    assert any(("NOT FOR RESALE" in t) == lic.noncommercial for t in texts if lic.name in t)
    linked = [p for p in sheet.ax.patches if p.get_url() == lic.url]
    assert len(linked) == (2 * len(lic.icons) if lic.icons else 1)  # disc + glyph per CC icon, or the badge


def test_output_name_with_dots_keeps_its_name(tmp_path: Path, block_step: Path) -> None:
    cfg = tmp_path / "blueprint.yml"
    cfg.write_text(f"model: {block_step.name}\noutput: out/part-v1.2\ntitle: T\nlicence: MIT\nformats: [svg]\nviews: [front]\n")
    assert main(["render", str(cfg)]) == 0
    assert (tmp_path / "out" / "part-v1.2.svg").exists()


def test_two_drawings_cannot_share_a_readme_marker(tmp_path: Path, block_step: Path) -> None:
    (tmp_path / "README.md").write_text("<!-- blueprint:sheet -->\n<!-- /blueprint -->\n")
    a, b = tmp_path / "a.yml", tmp_path / "b.yml"
    for cfg, out in ((a, "one/sheet"), (b, "two/sheet")):
        cfg.write_text(f"model: {block_step.name}\noutput: {out}\ntitle: T\nlicence: MIT\nreadme: README.md\nformats: [svg]\nviews: [front]\n")
    assert main(["render", str(a), str(b)]) == 1
    assert not (tmp_path / "one").exists(), "nothing is rendered when two drawings would fight over one marker"


def test_embed_credits_without_a_coffee_link(tmp_path: Path) -> None:
    """#26: the coffee link lives once per repo (README badge, FUNDING.yml), not under every drawing."""
    text = embed(tmp_path / "README.md", tmp_path / "docs" / "part.svg", "PART", "u1/u1_parts.py")
    assert "[![PART](docs/part.svg)](docs/part.svg)" in text and "`u1/u1_parts.py`" in text and "coffee" not in text.lower()
