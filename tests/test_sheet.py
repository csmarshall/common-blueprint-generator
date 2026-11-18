"""Layout, config and CLI tests: the sheet renders, chosen views and only those appear, and config errors name the problem."""

from __future__ import annotations

import subprocess
import sys
from datetime import date
from pathlib import Path

import pytest

from blueprint.cli import main
from blueprint.config import load
from blueprint.layout import auto_sheet
from blueprint.revisions import GitDates, git_dates, revisions
from blueprint.sheet import Sheet
from blueprint.style import fmt, scale_label


def _config(tmp_path: Path, model: Path, extra: str = "") -> Path:
    cfg = tmp_path / "blueprint.yml"
    cfg.write_text(f"model: {model.name}\noutput: out/sheet\ntitle: TEST BLOCK\ncompany: Test Works\nlicence: MIT\nformats: [svg]\n{extra}")
    return cfg


def test_fmt_and_scale_label() -> None:
    assert [fmt(18.0), fmt(3.85), fmt(38.71), fmt(305)] == ["18", "3.85", "38.71", "305"]
    assert [scale_label(2), scale_label(1), scale_label(0.75)] == ["2:1", "1:1", "1:1.33"]


@pytest.mark.parametrize(
    "views", [("FRONT", "TOP", "RIGHT", "ISOMETRIC"), ("TOP",), ("ISOMETRIC",), tuple(["FRONT", "TOP", "BOTTOM", "LEFT", "RIGHT", "BACK"])]
)
def test_auto_sheet_places_exactly_the_chosen_views(block: object, views: tuple[str, ...]) -> None:
    sheet = Sheet("B")
    layout = auto_sheet(sheet, block, views=views)
    assert set(layout.views) == set(views)
    x0, y0, x1, y1 = sheet.inner
    for v in layout.views.values():
        bx0, by0, bx1, by1 = v.box()
        assert x0 < bx0 and bx1 < x1 and y0 < by0 and by1 < y1, "view must sit inside the border"
        assert by0 > sheet.title_block_box[3] or bx1 < sheet.title_block_box[0], "view must clear the title block"


def test_single_view_is_drawn_larger(block: object) -> None:
    full = auto_sheet(Sheet("B"), block)
    alone = auto_sheet(Sheet("B"), block, views=("TOP",))
    assert alone.scale > full.scale


def test_isometric_size_multiplier_and_fill(block: object) -> None:
    same = auto_sheet(Sheet("B"), block, iso_size=1.0)
    filled = auto_sheet(Sheet("B"), block, iso_size=None)
    assert same.iso_scale == pytest.approx(same.scale)
    assert filled.iso_scale > same.iso_scale


def test_third_angle_placement(block: object) -> None:
    lay = auto_sheet(Sheet("B"), block, views=("FRONT", "TOP", "RIGHT", "LEFT", "BOTTOM"))
    f = lay.views["FRONT"].box()
    assert lay.views["TOP"].box()[1] > f[3]
    assert lay.views["BOTTOM"].box()[3] < f[1]
    assert lay.views["RIGHT"].box()[0] > f[2]
    assert lay.views["LEFT"].box()[2] < f[0]


def test_revisions_from_git_dates() -> None:
    rows = revisions(GitDates(date(2026, 10, 13), "abc1234", dirty=False), date(2026, 11, 18))
    assert [(r.rev, r.description, r.when) for r in rows] == [
        ("A", "INITIAL RELEASE", date(2026, 10, 13)),
        ("CUR", "RENDERED FROM abc1234", date(2026, 11, 18)),
    ]
    assert revisions(GitDates(None, None, False), date(2026, 11, 18)) == []


def test_git_dates_in_a_fresh_repo(tmp_path: Path) -> None:
    run = lambda *a: subprocess.run(["git", "-C", str(tmp_path), *a], check=True, capture_output=True)  # noqa: E731
    run("init", "-q")
    (tmp_path / "f").write_text("x")
    run("add", "f")
    run("-c", "user.name=t", "-c", "user.email=t@example.invalid", "-c", "commit.gpgsign=false", "commit", "-q", "-m", "first")
    d = git_dates(tmp_path)
    assert d.started == date.today() and d.head and not d.dirty
    (tmp_path / "f").write_text("y")
    assert git_dates(tmp_path).dirty


def test_config_errors_name_the_problem(tmp_path: Path, block_step: Path) -> None:
    with pytest.raises(ValueError, match="unknown keys"):
        load(_config(tmp_path, block_step, "colour: red\n"))
    with pytest.raises(ValueError, match="unknown views"):
        load(_config(tmp_path, block_step, "views: [front, sideways]\n"))
    with pytest.raises(ValueError, match="isometric must be"):
        load(_config(tmp_path, block_step, "isometric: big\n"))
    with pytest.raises(ValueError, match="sheet must be"):
        load(_config(tmp_path, block_step, "sheet: A4\n"))


def test_cli_renders_svg(tmp_path: Path, block_step: Path) -> None:
    cfg = _config(tmp_path, block_step, "views: [top]\nisometric: fill\nstyle: {aged: false}\nnotes: [Print flat.]\n")
    assert load(cfg).views == ("TOP",)
    assert main(["render", str(cfg)]) == 0
    svg = (tmp_path / "out" / "sheet.svg").read_text()
    assert svg.startswith("<?xml") and "<image" not in svg  # aged: false means no embedded paper bitmap


def test_cli_reports_failure(tmp_path: Path) -> None:
    cfg = tmp_path / "blueprint.yml"
    cfg.write_text("model: missing.step\noutput: out/x\ntitle: X\nlicence: MIT\n")
    assert main(["render", str(cfg)]) == 1


def test_readme_embed_is_rewritten_between_markers(tmp_path: Path, block_step: Path) -> None:
    readme = tmp_path / "README.md"
    readme.write_text("# Part\n\nIntro.\n\n<!-- blueprint:sheet -->\nold image\n<!-- /blueprint -->\n\nMore text.\n")
    cfg = _config(tmp_path, block_step, "readme: README.md\nviews: [front]\n")
    assert main(["render", str(cfg)]) == 0
    text = readme.read_text()
    assert "old image" not in text and text.startswith("# Part\n\nIntro.\n") and text.endswith("More text.\n")
    assert "[![TEST BLOCK](out/sheet.svg)](out/sheet.svg)" in text
    assert "common-blueprint-generator](https://github.com/csmarshall/common-blueprint-generator)" in text
    assert "`block.step`" in text
    assert main(["render", str(cfg)]) == 0
    assert readme.read_text() == text  # a re-render with nothing changed leaves the README alone


def test_readme_without_markers_is_an_error_not_an_insertion(tmp_path: Path, block_step: Path) -> None:
    readme = tmp_path / "README.md"
    readme.write_text("# Part\n")
    cfg = _config(tmp_path, block_step, "readme: README.md\nviews: [front]\n")
    assert main(["render", str(cfg)]) == 1
    assert readme.read_text() == "# Part\n"


def test_svg_is_byte_stable(tmp_path: Path, block_step: Path) -> None:
    cfg = _config(tmp_path, block_step, "views: [front]\n")
    main(["render", str(cfg)])
    first = (tmp_path / "out" / "sheet.svg").read_bytes()
    main(["render", str(cfg)])
    assert (tmp_path / "out" / "sheet.svg").read_bytes() == first


def test_generated_docs_are_current() -> None:
    root = Path(__file__).resolve().parent.parent
    done = subprocess.run([sys.executable, str(root / "tools" / "gen_docs.py"), "--check"], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
