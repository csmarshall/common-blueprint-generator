"""Keep a README's blueprint embed current: the image, linked to its scalable SVG, and a one-line credit linking to this project.

The README marks the spot once with a pair of comments; every render rewrites what is between them, so the embed and credit cannot go stale or be
lost in an edit:

    <!-- blueprint:dock -->
    (generated: do not edit between the markers)
    <!-- /blueprint -->

The marker name is the drawing's output file name (`output: docs/images/dock` → `blueprint:dock`). A preview (blueprint.preview) has its own
kind of marker, `<!-- blueprint-preview:NAME -->` … `<!-- /blueprint-preview -->`, so a README can carry a header preview and the sheet of the same
part. A missing marker is an error rather than an insertion, so the generator never decides where in someone's README a drawing goes.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path

PROJECT_NAME = "common-blueprint-generator"
PROJECT_URL = "https://github.com/csmarshall/common-blueprint-generator"
SUPPORT_URL = "https://buymeacoffee.com/cs_marshall"
# A `uses:` line calling this project's reusable render workflow: from a caller repo by URL, or from this repo itself (publish.yml dogfoods it).
# Matches any trigger (push, dispatch, schedule): the claim is about who renders, and every caller so far renders on push.
RENDER_WORKFLOW = re.compile(
    r"^\s*(?:-\s*)?uses:\s*['\"]?(?:csmarshall/common-blueprint-generator/|\./)\.github/workflows/render\.yml(?=[@'\"\s]|$)", re.MULTILINE | re.IGNORECASE
)
SHEET = "blueprint"  # marker kind for a drawing sheet
PREVIEW = "blueprint-preview"  # marker kind for a preview

log = logging.getLogger(__name__)


def markers(name: str, kind: str = SHEET) -> tuple[str, str]:
    """The start and end comments for the drawing (or, with kind=PREVIEW, the preview) called `name`."""
    return f"<!-- {kind}:{name} -->", f"<!-- /{kind} -->"


def renders_in_ci(readme: Path) -> bool:
    """True if the repo holding `readme` re-renders its drawings in CI: one of its workflows calls this project's reusable render workflow.

    Read from the repo rather than the environment (GITHUB_ACTIONS), so a local render of a CI repo writes the same credit line as CI does and the
    README does not flip between two wordings. The repo root is the nearest directory upward holding `.git` (a directory, or a file in a worktree).
    """
    for root in [readme.resolve().parent, *readme.resolve().parent.parents]:
        if (root / ".git").exists():
            workflows = root / ".github" / "workflows"
            files = sorted(f for f in [*workflows.glob("*.yml"), *workflows.glob("*.yaml")] if f.is_file()) if workflows.is_dir() else []
            for f in files:
                if RENDER_WORKFLOW.search(f.read_text(encoding="utf-8", errors="replace")):
                    log.debug("%s calls the render workflow: the credit claims a re-render on every change", f)
                    return True
            log.debug("no workflow under %s calls the render workflow: the credit makes no re-render claim", workflows)
            return False
    log.debug("%s is not inside a git repo: the credit makes no re-render claim", readme)
    return False


def embed(readme: Path, image: Path, title: str, source: str | Path, what: str = "Drawing", ci: bool | None = None) -> str:
    """The Markdown that goes between the markers: the image linked to itself at full size, then the credit line.

    `source` names what the drawing is generated from: a path (shown relative to the README) or text such as "u1/u1_parts.py". `what` opens the
    credit line ("Drawing", or "Preview" for a preview). The line claims a re-render "on every change to the model" only when `ci` is true, which
    by default is read from the repo (renders_in_ci); a repo that renders locally (make blueprint) gets the line without the claim.
    """
    if ci is None:
        ci = renders_in_ci(readme)
    rel = Path(os.path.relpath(image, readme.parent)).as_posix()
    source = Path(os.path.relpath(source, readme.parent)).as_posix() if isinstance(source, Path) else source
    claim = " on every change to the model" if ci else ""
    credit = f"{what} generated from `{source}` by [{PROJECT_NAME}]({PROJECT_URL}){claim} · [Buy me a coffee]({SUPPORT_URL})"
    return f"[![{title}]({rel})]({rel})\n\n<sub>{credit}</sub>"


def _pattern(readme: Path, name: str, kind: str) -> tuple[re.Pattern[str], str, str]:
    """The regex spanning this drawing's markers and the README text; ValueError naming the lines to add if the markers are missing."""
    start, end = markers(name, kind)
    text = readme.read_text()
    pattern = re.compile(re.escape(start) + r".*?" + re.escape(end), re.DOTALL)
    if not pattern.search(text):
        raise ValueError(f"{readme}: add the line '{start}' and, below it, '{end}' where the drawing should appear")
    return pattern, start, end


def require(readme: Path, name: str, kind: str = SHEET) -> None:
    """Raise ValueError unless `readme` has this drawing's markers: checked before drawing, so a missing marker fails before anything is written."""
    _pattern(readme, name, kind)


def update(readme: Path, name: str, content: str, kind: str = SHEET) -> bool:
    """Replace the text between this drawing's markers in `readme`; True if the file changed. Raises ValueError if the markers are missing."""
    pattern, start, end = _pattern(readme, name, kind)
    text = readme.read_text()
    new = pattern.sub(lambda _: f"{start}\n{content}\n{end}", text, count=1)
    if new == text:
        return False
    readme.write_text(new)
    log.info("updated the %s:%s embed in %s", kind, name, readme)
    return True
