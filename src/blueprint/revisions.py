"""Dates and revision rows for the title block, derived from git so nobody types them.

STARTED is the date of the repository's first commit. UPDATED is the render date. The revision block carries an initial-release row on the first
commit's date and a current row naming the commit the sheet was rendered from, so a sheet always says which source it shows.

A shallow clone's first commit is the clone boundary, not the project's start: the reusable workflow checks out with full history for that reason.
"""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from .sheet import Revision

log = logging.getLogger(__name__)


@dataclass
class GitDates:
    """What git says about the repository the model lives in."""

    started: date | None
    head: str | None  # short sha of HEAD
    dirty: bool  # uncommitted changes, so the sheet does not match HEAD exactly


def _git(repo: Path, *args: str) -> str | None:
    """Run git in `repo` and return stripped stdout, or None if git fails or is missing."""
    try:
        done = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=True, timeout=30)
    except (OSError, subprocess.SubprocessError) as exc:
        log.debug("git %s failed in %s: %s", " ".join(args), repo, exc)
        return None
    return done.stdout.strip()


def git_dates(repo: Path) -> GitDates:
    """First-commit date, HEAD and dirtiness of the repository containing `repo`; fields are None outside a git checkout."""
    first = _git(repo, "log", "--reverse", "--format=%cs", "--max-parents=0", "HEAD")
    if _git(repo, "rev-parse", "--is-shallow-repository") == "true":
        log.warning("%s is a shallow clone: STARTED shows the clone boundary, not the first commit (check out with fetch-depth: 0)", repo)
    started = date.fromisoformat(first.splitlines()[0]) if first else None
    head = _git(repo, "rev-parse", "--short=7", "HEAD")
    dirty = bool(_git(repo, "status", "--porcelain", "--untracked-files=no"))
    return GitDates(started, head, dirty)


def revisions(dates: GitDates, rendered: date) -> list[Revision]:
    """Initial release on the first commit's date, then the commit this sheet was rendered from."""
    rows = []
    if dates.started:
        rows.append(Revision("A", "INITIAL RELEASE", dates.started))
    if dates.head:
        rows.append(Revision("CUR", f"RENDERED FROM {dates.head}" + (" + LOCAL EDITS" if dates.dirty else ""), rendered))
    return rows
