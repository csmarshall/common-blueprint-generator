"""The licence on a drawing comes from the repo's REUSE.toml, for the drawing's own output path, with no default (issue #1 change request)."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from blueprint.cli import main
from blueprint.config import load
from blueprint.reuse_lookup import LicenceLookupError, licence_for

SPLIT = """version = 1

[[annotations]]
path = "**"
SPDX-FileCopyrightText = "2026 Charles Marshall"
SPDX-License-Identifier = "MIT"

[[annotations]]
path = ["**/*.svg", "**/*.step", "models/**"]
precedence = "override"
SPDX-FileCopyrightText = "2024-2026 Charles Marshall"
SPDX-License-Identifier = "CC-BY-NC-SA-4.0"
"""

SINGLE = """version = 1

[[annotations]]
path = "**"
SPDX-FileCopyrightText = "2025 Someone Else"
SPDX-License-Identifier = "CERN-OHL-S-2.0"
"""

ONLY_SCRIPTS = """version = 1

[[annotations]]
path = "**/*.py"
SPDX-FileCopyrightText = "2026 Charles Marshall"
SPDX-License-Identifier = "MIT"
"""


def _repo(tmp_path: Path, reuse_toml: str | None, block_step: Path) -> Path:
    repo = tmp_path / "repo"
    (repo / "part").mkdir(parents=True)
    subprocess.run(["git", "-C", str(repo), "init", "-q"], check=True)
    if reuse_toml is not None:
        (repo / "REUSE.toml").write_text(reuse_toml)
    shutil.copy(block_step, repo / "part" / "block.step")
    (repo / "part" / "make.py").write_text("print('model')\n")
    return repo


def _cfg(repo: Path, extra: str = "") -> Path:
    cfg = repo / "part" / "blueprint.yml"
    cfg.write_text(f"model: block.step\noutput: drawing\ntitle: T\nformats: [svg]\nviews: [front]\n{extra}")
    return cfg


def test_split_licence_resolves_the_drawing_not_the_script(tmp_path: Path, block_step: Path) -> None:
    repo = _repo(tmp_path, SPLIT, block_step)
    drawing = licence_for(repo / "part" / "drawing.svg")  # not rendered yet: the lookup must not need the file
    script = licence_for(repo / "part" / "make.py")
    assert drawing.licence.spdx == "CC-BY-NC-SA-4.0" and drawing.licence.noncommercial
    assert drawing.copyright == "© 2024-2026 Charles Marshall"
    assert script.licence.spdx == "MIT" and script.copyright == "© 2026 Charles Marshall"
    assert not (repo / "part" / "drawing.svg").exists(), "the lookup leaves no placeholder behind"


def test_single_licence_repo_and_unknown_ids_print_as_given(tmp_path: Path, block_step: Path) -> None:
    repo = _repo(tmp_path, SINGLE, block_step)
    found = licence_for(repo / "part" / "drawing.svg")
    assert found.licence.name == "CERN OHL-S 2.0" and found.copyright == "© 2025 Someone Else"
    (repo / "REUSE.toml").write_text(SINGLE.replace("CERN-OHL-S-2.0", "LicenseRef-Shermer-NC-1.0"))
    odd = licence_for(repo / "part" / "drawing.svg")
    assert odd.licence.name == "LicenseRef-Shermer-NC-1.0" and odd.licence.noncommercial and not odd.licence.icons


def test_render_prints_the_reuse_licence(tmp_path: Path, block_step: Path) -> None:
    repo = _repo(tmp_path, SPLIT, block_step)
    assert main(["render", str(_cfg(repo))]) == 0
    assert (repo / "part" / "drawing.svg").exists()


def test_config_override_skips_the_lookup(tmp_path: Path, block_step: Path) -> None:
    repo = _repo(tmp_path, None, block_step)  # no REUSE.toml at all
    cfg = _cfg(repo, "licence: {id: MIT, note: SCRIPTS ONLY}\ncopyright: © 2026 Charles Marshall\n")
    assert load(cfg).licence is not None
    assert main(["render", str(cfg)]) == 0


def test_missing_reuse_toml_fails_naming_the_path(tmp_path: Path, block_step: Path) -> None:
    repo = _repo(tmp_path, None, block_step)
    with pytest.raises(LicenceLookupError, match=r"part/drawing\.svg.*no REUSE\.toml"):
        licence_for(repo / "part" / "drawing.svg")
    assert main(["render", str(_cfg(repo))]) == 1
    assert not (repo / "part" / "drawing.svg").exists(), "nothing is written without a licence"


def test_unmatched_path_fails_naming_the_path(tmp_path: Path, block_step: Path) -> None:
    repo = _repo(tmp_path, ONLY_SCRIPTS, block_step)
    with pytest.raises(LicenceLookupError, match=r"part/drawing\.svg: no REUSE\.toml annotation"):
        licence_for(repo / "part" / "drawing.svg")
    assert main(["render", str(_cfg(repo))]) == 1


def test_preview_path_must_be_licensed_too(tmp_path: Path, block_step: Path) -> None:
    only_sheet = SPLIT.replace('path = ["**/*.svg", "**/*.step", "models/**"]', 'path = ["part/drawing.svg", "**/*.step"]').replace(
        'path = "**"\nSPDX-FileCopyrightText = "2026 Charles Marshall"\nSPDX-License-Identifier = "MIT"',
        'path = "**/*.py"\nSPDX-FileCopyrightText = "2026 Charles Marshall"\nSPDX-License-Identifier = "MIT"',
    )
    repo = _repo(tmp_path, only_sheet, block_step)
    assert main(["render", str(_cfg(repo))]) == 0, "the sheet alone is covered"
    (repo / "part" / "drawing.svg").unlink()
    assert main(["render", str(_cfg(repo, "preview: true\n"))]) == 1
    assert not (repo / "part" / "drawing.svg").exists(), "an unlicensed preview fails before anything is written"
