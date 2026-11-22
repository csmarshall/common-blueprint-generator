"""Sheet-free previews for README headers.

One or more parts are drawn side by side at one scale in the house style and captioned, with no border, title block, dimensions or hidden lines.

A preview is rendered by the same run that draws the sheets (the `preview:` key in blueprint.yml, or `preview()` from a Python render command),
so it cannot show a part that no longer exists. Parts can be files (STEP, or STL as a fallback), OCP shapes, or build123d / cadquery objects built
in memory, so a repo that builds its parts in Python does not have to write a file just to read it back.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from OCP.TopoDS import TopoDS_Shape

from .geometry import load_shape, project
from .layout import ALL_VIEWS, view_directions, view_size
from .readme import PREVIEW, embed, update
from .sheet import TEXT_HEIGHT_EM, Canvas, View
from .style import PT_PER_MM, Style

log = logging.getLogger(__name__)

PREVIEW_WIDTH = 200.0  # mm across the canvas at most. The SVG scales to the README's column, so this sets how heavy the lines look against the parts.
PREVIEW_HEIGHT = 90.0  # mm the parts may stand at most, so a tall part makes a header strip rather than a poster; the canvas then narrows to fit
PREVIEW_VIEW = "ISOMETRIC"


@dataclass
class PreviewLayout:
    """What preview() drew: the shared scale, each part's placed view (left to right) and the SVG written."""

    scale: float
    width: float
    height: float
    views: list[View]
    path: Path


def as_shape(model: Any) -> TopoDS_Shape:
    """An OCP shape from a file path, an OCP shape, a build123d or cadquery Shape (`.wrapped`), or a cadquery Workplane (`.toOCC()`)."""
    if isinstance(model, TopoDS_Shape):
        return model
    if isinstance(model, str | Path):
        return load_shape(Path(model))
    to_occ = getattr(model, "toOCC", None)
    if callable(to_occ):
        return as_shape(to_occ())
    wrapped = getattr(model, "wrapped", None)
    if isinstance(wrapped, TopoDS_Shape):
        return wrapped
    raise TypeError(f"cannot draw a {type(model).__name__}: pass a STEP/STL path, an OCP TopoDS_Shape, or a build123d or cadquery object")


def preview(
    parts: Sequence[Any],
    out: Path,
    front: str = "+Z",
    up: str = "+Y",
    view: str = PREVIEW_VIEW,
    width: float = PREVIEW_WIDTH,
    max_height: float = PREVIEW_HEIGHT,
    style: Style | None = None,
    readme: Path | None = None,
    title: str = "",
    source: str | Path = "",
) -> PreviewLayout:
    """Draw `parts` side by side in `view` at one scale and write out.svg; with `readme`, keep it embedded there.

    Each entry of `parts` is a model (see as_shape) or a (model, caption) pair. `front` and `up` are the model axes the sheets use, so the preview
    shows the part in the same pose. The embed goes between `<!-- blueprint-preview:NAME -->` and `<!-- /blueprint-preview -->`, NAME being the
    output file name, with the same credit line as a sheet; `title` is its alt text and `source` what the credit says it was generated from.

    The parts fill `width` unless that would make them taller than `max_height`; then they are drawn `max_height` tall and the canvas is only as
    wide as they need.
    """
    if not parts:
        raise ValueError("a preview needs at least one part")
    name = view.upper()
    if name not in ALL_VIEWS:
        raise ValueError(f"unknown view {view!r}; choose from {[v.lower() for v in ALL_VIEWS]}")
    st = style or Style()
    sp, ty = st.spacing, st.type
    entries = [p if isinstance(p, tuple) else (p, "") for p in parts]
    toward, up_dir = view_directions(front, up)[name]
    projections = [project(as_shape(model), toward, up_dir) for model, _ in entries]
    sizes = [view_size(p) for p in projections]
    has_captions = any(caption for _, caption in entries)
    caption_h = sp.caption_drop + ty.caption / PT_PER_MM * TEXT_HEIGHT_EM if has_captions else 0.0
    room = width - 2 * sp.margin - sp.view_gap * (len(entries) - 1)
    if room <= 0:
        raise ValueError(f"a {width:g} mm preview has no room for {len(entries)} parts: make it wider")
    scale = min(room / sum(w for w, _ in sizes), max_height / max(h for _, h in sizes))
    row_h = scale * max(h for _, h in sizes)
    width = 2 * sp.margin + scale * sum(w for w, _ in sizes) + sp.view_gap * (len(entries) - 1)
    height = 2 * sp.margin + row_h + caption_h
    canvas = Canvas(width, height, st)
    row_bottom = sp.margin + caption_h
    placed = []
    x = sp.margin
    for (_, caption), projection, (w, h) in zip(entries, projections, sizes, strict=True):
        corner = (x, row_bottom + (row_h - scale * h) / 2)  # each part centred in the row, so differing heights line up on their middles
        v = View.at_corner(canvas, projection, corner, scale).draw(hidden=False, centre_marks=False)
        if caption:
            canvas.caption(x + scale * w / 2, row_bottom - sp.caption_drop, caption)
        placed.append(v)
        x += scale * w + sp.view_gap
    (svg,) = canvas.save(out, ("svg",))
    log.info("preview %s: %d parts at scale %.2f, %.0f x %.0f mm", out.name, len(entries), scale, width, height)
    if readme:
        update(readme, out.name, embed(readme, svg, title or out.name, source, what="Preview"), PREVIEW)
    return PreviewLayout(scale, width, height, placed, svg)
