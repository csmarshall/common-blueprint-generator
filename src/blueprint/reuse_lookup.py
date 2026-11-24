"""Read a drawing's licence and copyright from the repo it is rendered into, using REUSE (https://reuse.software/spec-3.3/).

The licence printed on a sheet must be the licence of the sheet itself, which in a split-licence repo (drawings CC BY-NC-SA, scripts MIT) differs
from the model's and the script's. So the lookup is for the drawing's own output path, and the `reuse` package decides which REUSE.toml annotation
wins by its own precedence rules: this module writes no glob matching of its own.

There is no default. No REUSE.toml, or no annotation covering the drawing, is an error naming the path: a wrong licence on a drawing is worse than
none, and none is not allowed either. `licence:` in blueprint.yml is the explicit override that skips this lookup.

`reuse` is GPL-3.0-or-later, so it is an optional extra (`common-blueprint-generator[reuse]`) rather than a hard dependency of this MIT package; the
reusable workflow always installs it.
"""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .licences import Licence, lookup

log = logging.getLogger(__name__)


class LicenceLookupError(ValueError):
    """The drawing's licence could not be read from the repo's REUSE.toml."""


@dataclass(frozen=True)
class FileLicence:
    """What REUSE says about one path."""

    licence: Licence
    copyright: str  # e.g. "© 2026 Charles Marshall"; several holders joined with "; "
    source: str  # where it came from, for the log: e.g. REUSE.toml


def repo_root(start: Path) -> Path:
    """The git work tree containing `start`, else the topmost ancestor holding a REUSE.toml, else LicenceLookupError."""
    probe = start if start.is_dir() else start.parent
    while not probe.exists():
        probe = probe.parent
    try:
        top = subprocess.run(["git", "-C", str(probe), "rev-parse", "--show-toplevel"], capture_output=True, text=True, check=True, timeout=30)
        return Path(top.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        holders = [d for d in (probe, *probe.parents) if (d / "REUSE.toml").is_file()]
        if holders:
            return holders[-1]
    raise LicenceLookupError(f"{start}: not inside a git repository or a directory with a REUSE.toml, so its licence cannot be looked up")


def licence_for(drawing: Path) -> FileLicence:
    """The licence and copyright REUSE assigns to `drawing` (the sheet's own output file, which need not exist yet)."""
    try:
        from reuse.project import Project
    except ImportError as exc:
        raise LicenceLookupError(
            "reading the licence from REUSE.toml needs the reuse extra: install common-blueprint-generator[reuse], or set licence: in blueprint.yml"
        ) from exc
    drawing = drawing.resolve()
    root = repo_root(drawing)
    rel = drawing.relative_to(root)
    if not (root / "REUSE.toml").is_file():
        raise LicenceLookupError(f"{rel}: {root} has no REUSE.toml, so the drawing's licence is unknown. Add one, or set licence: in blueprint.yml")
    project = Project.from_directory(root)
    # reuse reads the file itself as well as REUSE.toml, so the path must exist, and it must be empty: a previous render printed its © line into
    # the sheet, and reading that back would feed it into this render, growing (and re-escaping) on every pass (#24). So the lookup always sees an
    # empty file at the drawing's path, which leaves REUSE.toml as the only source whatever precedence its annotation uses. An existing sheet's
    # bytes are held aside and put back.
    existing = drawing.read_bytes() if drawing.exists() else None
    created_dirs = [d for d in reversed(drawing.parents) if not d.exists()]
    try:
        drawing.parent.mkdir(parents=True, exist_ok=True)
        drawing.write_bytes(b"")
        infos = project.reuse_info_of(drawing)
    finally:
        if existing is None:
            drawing.unlink(missing_ok=True)
            for d in reversed(created_dirs):
                d.rmdir()
        else:
            drawing.write_bytes(existing)
    expressions = sorted({str(e) for info in infos for e in info.spdx_expressions})
    if not expressions:
        raise LicenceLookupError(f"{rel}: no REUSE.toml annotation in {root} covers this drawing. Add its path to an [[annotations]] entry")
    spdx = expressions[0] if len(expressions) == 1 else " AND ".join(f"({e})" if " " in e else e for e in expressions)
    notices = sorted({_notice(n) for info in infos for n in info.copyright_notices})
    sources = sorted({str(info.source_path or info.path) for info in infos})
    log.info("%s: %s, %s (from %s)", rel, spdx, "; ".join(notices) or "no copyright notice", ", ".join(sources))
    return FileLicence(lookup(spdx), "; ".join(notices), ", ".join(sources))


def _notice(notice: object) -> str:
    """A reuse CopyrightNotice as the title block prints it: © 2026 Charles Marshall (or 2024-2026)."""
    years = getattr(notice, "years", ()) or ()
    spans = [f"{y.start}{y.separator or ''}{y.end or ''}" if getattr(y, "end", None) else str(y.start) for y in years]
    name = str(getattr(notice, "name", "") or "")
    return "© " + " ".join(s for s in (", ".join(spans), name) if s)
