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
SHEET = "blueprint"  # marker kind for a drawing sheet
PREVIEW = "blueprint-preview"  # marker kind for a preview

log = logging.getLogger(__name__)


def markers(name: str, kind: str = SHEET) -> tuple[str, str]:
    """The start and end comments for the drawing (or, with kind=PREVIEW, the preview) called `name`."""
    return f"<!-- {kind}:{name} -->", f"<!-- /{kind} -->"


def embed(readme: Path, image: Path, title: str, source: str | Path, what: str = "Drawing") -> str:
    """The Markdown that goes between the markers: the image linked to itself at full size, then the credit line.

    `source` names what the drawing is generated from: a path (shown relative to the README) or text such as "u1/u1_parts.py". `what` opens the
    credit line ("Drawing", or "Preview" for a preview).
    """
    rel = Path(os.path.relpath(image, readme.parent)).as_posix()
    source = Path(os.path.relpath(source, readme.parent)).as_posix() if isinstance(source, Path) else source
    credit = f"{what} generated from `{source}` by [{PROJECT_NAME}]({PROJECT_URL}) on every change to the model · [Buy me a coffee]({SUPPORT_URL})"
    return f"[![{title}]({rel})]({rel})\n\n<sub>{credit}</sub>"


def update(readme: Path, name: str, content: str, kind: str = SHEET) -> bool:
    """Replace the text between this drawing's markers in `readme`; True if the file changed. Raises ValueError if the markers are missing."""
    start, end = markers(name, kind)
    text = readme.read_text()
    pattern = re.compile(re.escape(start) + r".*?" + re.escape(end), re.DOTALL)
    if not pattern.search(text):
        raise ValueError(f"{readme}: add the line '{start}' and, below it, '{end}' where the drawing should appear")
    new = pattern.sub(lambda _: f"{start}\n{content}\n{end}", text, count=1)
    if new == text:
        return False
    readme.write_text(new)
    log.info("updated the %s:%s embed in %s", kind, name, readme)
    return True
