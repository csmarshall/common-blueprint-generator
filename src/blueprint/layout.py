"""The automatic sheet: six orthographic views in the third-angle cross plus an isometric, at the largest standard scale that fits.

Front sits in the middle, top above it, bottom below, left and right beside it and back at the far right. Each orthographic view gets its overall
width and height, measured from its own projection; the isometric goes in the space right of the cross and is not to scale.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from OCP.TopoDS import TopoDS_Shape

from .geometry import Projection, Vec, cross, project
from .sheet import TEXT_HEIGHT_EM, CentreLine, Sheet, View
from .style import PT_PER_MM, STANDARD_SCALES

log = logging.getLogger(__name__)

ORTHO = ("FRONT", "TOP", "BOTTOM", "LEFT", "RIGHT", "BACK")
ALL_VIEWS = (*ORTHO, "ISOMETRIC")
DEFAULT_VIEWS = ("FRONT", "TOP", "RIGHT", "ISOMETRIC")  # the usual minimum; the other three are opt-in through `views`
DEFAULT_ISO_SIZE = 1.0  # isometric scale as a multiple of the main scale; None fills the free space
ISO_MIN_FRACTION = 0.45  # the isometric beside other views is drawn at least this fraction of the main scale, or the main scale steps down
ISO_FILL = 0.92  # an isometric drawn alone fills this fraction of the free space
CLEARANCE = 2.0  # sheet mm between any annotation and the border, title block or revision block
MIN_EXTENT = 1e-6  # mm; a view seen exactly edge-on has zero width or height
REV_ROWS_RESERVED = 2  # the revision block's rows (initial release + current), as Sheet draws it from git


AXES: dict[str, Vec] = {"+X": (1, 0, 0), "-X": (-1, 0, 0), "+Y": (0, 1, 0), "-Y": (0, -1, 0), "+Z": (0, 0, 1), "-Z": (0, 0, -1)}


def view_directions(front: str, up: str) -> dict[str, tuple[Vec, Vec]]:
    """(toward viewer, up) for every view, from the model axes that face the viewer and point up in the front view."""
    f, u = AXES[front], AXES[up]
    if abs(float(np.dot(f, u))) > 1e-9:
        raise ValueError(f"front {front} and up {up} must be perpendicular")
    r = cross(u, f)  # the viewer's right hand in the front view
    k = 1 / np.sqrt(3)
    iso: Vec = ((f[0] + r[0] + u[0]) * k, (f[1] + r[1] + u[1]) * k, (f[2] + r[2] + u[2]) * k)
    return {
        "FRONT": (f, u),
        "BACK": (_neg(f), u),
        "RIGHT": (r, u),
        "LEFT": (_neg(r), u),
        "TOP": (u, _neg(f)),
        "BOTTOM": (_neg(u), f),
        "ISOMETRIC": (iso, u),
    }


def _neg(v: Vec) -> Vec:
    """-v."""
    return (-v[0], -v[1], -v[2])


@dataclass
class AutoLayout:
    """Where each view went and at what scale."""

    scale: float
    iso_scale: float
    views: dict[str, View]


def view_size(p: Projection) -> tuple[float, float]:
    """Width and height of a view's visible outline, never zero (an edge-on face would otherwise divide by zero when scaling)."""
    x0, y0, x1, y1 = p.bounds()
    return max(x1 - x0, MIN_EXTENT), max(y1 - y0, MIN_EXTENT)


def _pads(sheet: Sheet) -> tuple[float, float]:
    """Sheet mm needed below and to the left of a view for its overall dimension and caption, derived from the style, not tuned."""
    sp, ty = sheet.style.spacing, sheet.style.type
    caption_mm = ty.caption / PT_PER_MM * TEXT_HEIGHT_EM
    dim_mm = ty.dim / PT_PER_MM * TEXT_HEIGHT_EM
    return sp.dim_offset + sp.caption_drop + caption_mm + CLEARANCE, sp.dim_offset + dim_mm + CLEARANCE


def auto_sheet(
    sheet: Sheet,
    shape: TopoDS_Shape,
    front: str = "+Z",
    up: str = "+Y",
    captions: dict[str, str] | None = None,
    views: tuple[str, ...] = DEFAULT_VIEWS,
    iso_size: float | None = DEFAULT_ISO_SIZE,
    centre_lines: Sequence[CentreLine] = (),
) -> AutoLayout:
    """Project `shape`, choose the scale, place and draw the views with overall dimensions and captions.

    `views` picks which views appear, in any combination. Orthographic views keep their third-angle places on a grid (top above, bottom below,
    left and right beside the centre column, back at the far right) whether or not the front view is among them; one view on its own is centred and
    drawn as large as a standard scale allows (the isometric alone is scaled to fill the space). `iso_size` sets the isometric's scale as a multiple
    of the main scale when it sits beside other views; None makes it fill the space to their right. `centre_lines` are drawn on every orthographic
    view where their axis projects to a line (the isometric is not to scale and carries no centre marks, so none there).
    """
    captions = {k.upper(): v for k, v in (captions or {}).items()}
    views = tuple(dict.fromkeys(v.upper() for v in views))  # drop repeats, keep order
    if unknown := [v for v in views if v not in ALL_VIEWS]:
        raise ValueError(f"unknown views {unknown}; choose from {list(ALL_VIEWS)}")
    if not views:
        raise ValueError("at least one view is needed")
    dirs = view_directions(front, up)
    proj = {name: project(shape, d, u) for name, (d, u) in dirs.items() if name in views}
    below, left = _pads(sheet)
    x0, _, x1, y1 = sheet.inner
    band_top = sheet.title_block_box[3]
    # Views and their annotations stay inside the border and above the title-block band; the revision block is handled per layout.
    area = (x0 + left, band_top + below, x1 - CLEARANCE, y1 - CLEARANCE)
    rev = sheet.rev_block_box(REV_ROWS_RESERVED)
    if len(views) == 1:
        return _single(sheet, views[0], proj[views[0]], (area[0], area[1], area[2], rev[1] - CLEARANCE), captions, centre_lines)
    return _grid(sheet, proj, area, rev, captions, iso_size, centre_lines)


def _single(
    sheet: Sheet, name: str, projection: Projection, area: tuple[float, float, float, float], captions: dict[str, str], centre_lines: Sequence[CentreLine] = ()
) -> AutoLayout:
    """One view, centred in `area` (already clear of every annotation and the revision block)."""
    pw, ph = view_size(projection)
    aw, ah = area[2] - area[0], area[3] - area[1]
    if name == "ISOMETRIC":
        s = min(aw / pw, ah / ph) * ISO_FILL
    else:
        fitting = [k for k in STANDARD_SCALES if k * pw <= aw and k * ph <= ah]
        if not fitting:
            raise ValueError(f"{name} view does not fit the sheet at any standard scale: use a larger sheet size")
        s = fitting[0]
    corner = (area[0] + (aw - s * pw) / 2, area[1] + (ah - s * ph) / 2)
    v = _draw(sheet, name, projection, corner, s, captions, centre_lines)
    log.info("single %s view at scale %.2f", name, s)
    return AutoLayout(s, s, {name: v})


def _draw(
    sheet: Sheet,
    name: str,
    projection: Projection,
    corner: tuple[float, float],
    scale: float,
    captions: dict[str, str],
    centre_lines: Sequence[CentreLine] = (),
) -> View:
    """Place, draw and annotate one view."""
    is_iso = name == "ISOMETRIC"
    v = View.at_corner(sheet, projection, corner, scale).draw(hidden=not is_iso, centre_marks=not is_iso)
    if is_iso:
        v.caption("ISOMETRIC", "not to scale", below=sheet.style.spacing.caption_drop)
    else:
        for c in centre_lines:
            v.centre_line(c.point, c.axis, c.label)
        v.overall_dims()
        v.caption(name, captions.get(name, ""))
    return v


# Grid cells of the third-angle layout: (column, row). Column 1 and row 1 are the centre, where the front view sits when it is drawn.
CELLS = {"LEFT": (0, 1), "FRONT": (1, 1), "TOP": (1, 0), "BOTTOM": (1, 2), "RIGHT": (2, 1), "BACK": (3, 1)}


def _grid(
    sheet: Sheet,
    proj: dict[str, Projection],
    area: tuple[float, float, float, float],
    rev: tuple[float, float, float, float],
    captions: dict[str, str],
    iso_size: float | None,
    centre_lines: Sequence[CentreLine] = (),
) -> AutoLayout:
    """Orthographic views on the third-angle grid, and the isometric (if chosen) in the space to their right."""
    gap = sheet.style.spacing.view_gap
    size = {name: view_size(p) for name, p in proj.items()}
    ortho = [n for n in proj if n != "ISOMETRIC"]
    has_iso = "ISOMETRIC" in proj
    cols = sorted({CELLS[n][0] for n in ortho})
    rows = sorted({CELLS[n][1] for n in ortho})
    # Each column is as wide as its widest view and each row as tall as its tallest, so the centre column and row hold whether or not FRONT is drawn.
    col_w = {c: max(size[n][0] for n in ortho if CELLS[n][0] == c) for c in cols}
    row_h = {r: max(size[n][1] for n in ortho if CELLS[n][1] == r) for r in rows}
    area_w, area_h = area[2] - area[0], area[3] - area[1]
    below_rev = rev[1] - CLEARANCE - area[1]  # height available under the revision block

    chosen = None
    for s in STANDARD_SCALES:
        grid_w = s * sum(col_w.values()) + gap * (len(cols) - 1)
        grid_h = s * sum(row_h.values()) + gap * (len(rows) - 1)
        if grid_w > area_w:
            continue
        if not has_iso:
            left_edge = area[0] + (area_w - grid_w) / 2
            # Centred grid: it must stay under the revision block if it reaches across under it.
            room_h = below_rev if left_edge + grid_w > rev[0] - CLEARANCE else area_h
            if grid_h <= room_h:
                chosen = (s, 0.0, grid_w, grid_h, left_edge, room_h)
                break
            continue
        iso_w_room = area_w - grid_w - gap
        if iso_w_room <= 0 or grid_h > (below_rev if area[0] + grid_w > rev[0] - CLEARANCE else area_h):
            continue
        iw, ih = size["ISOMETRIC"]
        room_s = min(iso_w_room / iw, (below_rev - sheet.style.spacing.caption_drop) / ih)
        iso_s = room_s if iso_size is None else min(s * iso_size, room_s)
        if iso_s >= s * ISO_MIN_FRACTION:
            chosen = (s, iso_s, grid_w, grid_h, area[0], below_rev if area[0] + grid_w > rev[0] - CLEARANCE else area_h)
            break
    if chosen is None:
        raise ValueError("part does not fit the sheet at any standard scale: use a larger sheet size")
    s, iso_s, grid_w, grid_h, left_edge, room_h = chosen
    log.info("scale %.2f, isometric %.2f", s, iso_s)

    col_x: dict[int, float] = {}
    x = left_edge
    for c in cols:
        col_x[c] = x
        x += s * col_w[c] + gap
    row_y: dict[int, float] = {}
    y = area[1] + (room_h - grid_h) / 2 + grid_h  # top of the grid, centred vertically in the room it has
    for r in rows:
        y -= s * row_h[r]
        row_y[r] = y
        y -= gap
    placed: dict[str, View] = {}
    for name in ortho:
        c, r = CELLS[name]
        w, h = size[name]
        corner = (col_x[c] + (s * col_w[c] - s * w) / 2, row_y[r] + (s * row_h[r] - s * h) / 2)
        placed[name] = _draw(sheet, name, proj[name], corner, s, captions, centre_lines)
    if has_iso:
        iw, ih = size["ISOMETRIC"]
        iso_left = left_edge + grid_w + gap
        corner = (iso_left + (area[2] - iso_left - iso_s * iw) / 2, rev[1] - CLEARANCE - iso_s * ih)
        placed["ISOMETRIC"] = _draw(sheet, "ISOMETRIC", proj["ISOMETRIC"], corner, iso_s, captions)
    return AutoLayout(s, iso_s, placed)
