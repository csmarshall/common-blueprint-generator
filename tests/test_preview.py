"""Preview tests: inputs it accepts, side-by-side layout at one scale, size limits, byte-stable SVG, README embed, config and CLI."""

from __future__ import annotations

from pathlib import Path

import pytest
from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
from OCP.TopoDS import TopoDS_Shape

from blueprint.cli import main
from blueprint.config import load
from blueprint.preview import PREVIEW_HEIGHT, PREVIEW_WIDTH, as_shape, preview
from blueprint.readme import markers
from conftest import BLOCK

LONG_BOX = (80.0, 20.0, 30.0)  # wider than the block, so order and shared scale are visible


def _long_box() -> TopoDS_Shape:
    return BRepPrimAPI_MakeBox(*LONG_BOX).Shape()


class _Build123dLike:
    """Stands in for a build123d or cadquery Shape: the OCP shape is on .wrapped."""

    def __init__(self, shape: TopoDS_Shape) -> None:
        self.wrapped = shape


class _WorkplaneLike:
    """Stands in for a cadquery Workplane: toOCC() returns the OCP shape."""

    def __init__(self, shape: TopoDS_Shape) -> None:
        self._shape = shape

    def toOCC(self) -> TopoDS_Shape:  # noqa: N802 - cadquery's name
        return self._shape


def test_as_shape_accepts_paths_shapes_and_cad_objects(block: TopoDS_Shape, block_step: Path) -> None:
    assert as_shape(block) is block
    assert as_shape(_Build123dLike(block)) is block
    assert as_shape(_WorkplaneLike(block)) is block
    for path in (block_step, str(block_step)):
        assert isinstance(as_shape(path), TopoDS_Shape)
    with pytest.raises(TypeError, match="cannot draw a int"):
        as_shape(42)


def test_parts_sit_left_to_right_at_one_scale(tmp_path: Path, block: TopoDS_Shape) -> None:
    # Front view (+Z toward the viewer), so each part's drawn width is its X extent times the scale: 40 for the block, 80 for the box.
    lay = preview([(block, "BLOCK"), (_long_box(), "BOX")], tmp_path / "p", view="front")
    (a0, ay0, a1, ay1), (b0, by0, b1, by1) = (v.box() for v in lay.views)
    assert a1 < b0, "parts are drawn in the order given, left to right"
    assert (a1 - a0) / BLOCK[0] == pytest.approx((b1 - b0) / LONG_BOX[0]), "one scale for every part"
    assert (ay0 + ay1) / 2 == pytest.approx((by0 + by1) / 2), "parts line up on their middles"
    for x0, y0, x1, y1 in (v.box() for v in lay.views):
        assert 0 < x0 < x1 < lay.width and 0 < y0 < y1 < lay.height, "parts sit inside the canvas"
    assert lay.width == pytest.approx(PREVIEW_WIDTH)


def test_tall_part_is_capped_and_canvas_narrows(tmp_path: Path) -> None:
    tower = BRepPrimAPI_MakeBox(10, 200, 10).Shape()  # front view 10 wide, 200 tall
    lay = preview([tower], tmp_path / "p", view="front")
    _, y0, _, y1 = lay.views[0].box()
    assert y1 - y0 == pytest.approx(PREVIEW_HEIGHT)
    assert lay.width < PREVIEW_WIDTH


def test_svg_is_byte_stable_and_only_svg(tmp_path: Path, block: TopoDS_Shape) -> None:
    a = preview([block], tmp_path / "a").path
    b = preview([block], tmp_path / "b").path
    assert a.suffix == ".svg" and a.read_bytes() == b.read_bytes()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["a.svg", "b.svg"]


def test_bad_inputs_are_refused(tmp_path: Path, block: TopoDS_Shape) -> None:
    with pytest.raises(ValueError, match="at least one part"):
        preview([], tmp_path / "p")
    with pytest.raises(ValueError, match="unknown view"):
        preview([block], tmp_path / "p", view="sideways")
    with pytest.raises(ValueError, match="no room"):
        preview([block] * 3, tmp_path / "p", width=5)


def test_readme_embed_uses_preview_markers_and_leaves_sheet_alone(tmp_path: Path, block: TopoDS_Shape) -> None:
    readme = tmp_path / "README.md"
    sheet_start, sheet_end = markers("hero")
    start, end = markers("hero", "blueprint-preview")
    readme.write_text(f"# x\n{start}\n{end}\n\n{sheet_start}\nkept\n{sheet_end}\n")
    preview([block], tmp_path / "img" / "hero", readme=readme, title="HERO", source="parts.py")
    text = readme.read_text()
    assert "[![HERO](img/hero.svg)](img/hero.svg)" in text
    assert "Preview generated from `parts.py`" in text
    assert f"{sheet_start}\nkept\n{sheet_end}" in text
    with pytest.raises(ValueError, match="blueprint-preview:other"):
        preview([block], tmp_path / "img" / "other", readme=readme)


def _config(tmp_path: Path, model: Path, extra: str) -> Path:
    cfg = tmp_path / "blueprint.yml"
    cfg.write_text(f"model: {model.name}\noutput: out/sheet\ntitle: TEST BLOCK\nlicence: MIT\nformats: [svg]\n{extra}")
    return cfg


def test_config_preview_true_and_map(tmp_path: Path, block_step: Path) -> None:
    assert load(_config(tmp_path, block_step, "")).preview is None
    simple = load(_config(tmp_path, block_step, "preview: true\n")).preview
    assert simple is not None
    assert simple.output == (tmp_path / "out" / "sheet-preview").resolve()
    assert simple.parts == ((block_step.resolve(), ""),) and simple.view == "ISOMETRIC" and simple.width == PREVIEW_WIDTH
    full = load(
        _config(tmp_path, block_step, f"preview:\n  view: top\n  width: 150\n  output: hero\n  parts:\n    - {{model: {block_step.name}, caption: A}}\n")
    ).preview
    assert full is not None
    assert (full.view, full.width, full.output, full.parts) == ("TOP", 150.0, (tmp_path / "hero").resolve(), ((block_step.resolve(), "A"),))


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        ("preview: yes please\n", "takes true"),
        ("preview: {colour: red}\n", "unknown preview keys"),
        ("preview: {view: sideways}\n", "preview view"),
        ("preview: {width: 0}\n", "positive number"),
        ("preview: {parts: [block.step]}\n", "map of model and caption"),
    ],
)
def test_config_preview_errors(tmp_path: Path, block_step: Path, extra: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        load(_config(tmp_path, block_step, extra))


def test_cli_renders_preview_with_the_sheet(tmp_path: Path, block_step: Path) -> None:
    readme = tmp_path / "README.md"
    start, end = markers("sheet-preview", "blueprint-preview")
    sheet_start, sheet_end = markers("sheet")
    readme.write_text(f"{start}\n{end}\n{sheet_start}\n{sheet_end}\n")
    cfg = _config(tmp_path, block_step, "readme: README.md\npreview: true\n")
    assert main(["render", str(cfg)]) == 0
    assert (tmp_path / "out" / "sheet.svg").exists() and (tmp_path / "out" / "sheet-preview.svg").exists()
    assert "(out/sheet-preview.svg)" in readme.read_text() and "(out/sheet.svg)" in readme.read_text()


def test_cli_refuses_two_configs_sharing_a_preview_marker(tmp_path: Path, block_step: Path) -> None:
    (tmp_path / "README.md").write_text("")
    one = _config(tmp_path, block_step, "readme: README.md\npreview: {output: hero}\n")
    two = tmp_path / "two.yml"
    two.write_text(one.read_text().replace("out/sheet", "out/other"))
    assert main(["render", str(one), str(two)]) == 1
