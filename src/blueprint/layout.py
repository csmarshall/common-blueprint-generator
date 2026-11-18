"""The automatic sheet: six orthographic views in the third-angle cross plus an isometric, at the largest standard scale that fits.

Front sits in the middle, top above it, bottom below, left and right beside it and back at the far right. Each orthographic view gets its overall
width and height, measured from its own projection; the isometric goes in the space right of the cross and is not to scale.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from OCP.TopoDS import TopoDS_Shape

from .geometry import Projection, Vec, cross, project
from .sheet import Sheet, View
from .style import STANDARD_SCALES

log = logging.getLogger(__name__)

ORTHO = ("FRONT", "TOP", "BOTTOM", "LEFT", "RIGHT", "BACK")
ALL_VIEWS = (*ORTHO, "ISOMETRIC")
DEFAULT_VIEWS = ("FRONT", "TOP", "RIGHT", "ISOMETRIC")  # the usual minimum; the other three are opt-in through `views`
ISO_MAX_ENLARGE = 1.6  # the isometric is drawn at most this much larger than the main scale
ISO_MIN_FRACTION = 0.45  # and at least this fraction of it, or the scale steps down
EDGE_PAD = 16.0  # sheet mm kept clear inside the border for the outermost dimensions and captions


AXES: dict[str, Vec] = {"+X": (1, 0, 0), "-X": (-1, 0, 0), "+Y": (0, 1, 0), "-Y": (0, -1, 0), "+Z": (0, 0, 1), "-Z": (0, 0, -1)}


def view_directions(front: str, up: str) -> dict[str, tuple[Vec, Vec]]:
    """(toward viewer, up) for every view, from the model axes that face the viewer and point up in the front view."""
    f, u = AXES[front], AXES[up]
    if abs(float(np.dot(f, u))) > 1e-9:
        raise ValueError(f"front {front} and up {up} must be perpendicular")
    r = cross(u, f)  # the viewer's right hand in the front view
    neg = lambda v: (-v[0], -v[1], -v[2])  # noqa: E731
    return {
        "FRONT": (f, u), "BACK": (neg(f), u),
        "RIGHT": (r, u), "LEFT": (neg(r), u),
        "TOP": (u, neg(f)), "BOTTOM": (neg(u), f),
        "ISOMETRIC": (tuple(np.add(np.add(f, r), u) / np.sqrt(3)), u),  # type: ignore[dict-item]
    }


@dataclass
class AutoLayout:
    """Where each view went and at what scale."""

    scale: float
    iso_scale: float
    views: dict[str, View]


def _size(p: Projection) -> tuple[float, float]:
    x0, y0, x1, y1 = p.bounds()
    return x1 - x0, y1 - y0


def auto_sheet(sheet: Sheet, shape: TopoDS_Shape, front: str = "+Z", up: str = "+Y", captions: dict[str, str] | None = None,
               views: tuple[str, ...] = DEFAULT_VIEWS) -> AutoLayout:
    """Project `shape`, choose the scale, place and draw the views with overall dimensions and captions.

    `views` picks which views appear. Several orthographic views keep their third-angle positions around the front; one view on its own is centred
    and drawn as large as a standard scale allows (the isometric alone is scaled to fill the space).
    """
    captions = {k.upper(): v for k, v in (captions or {}).items()}
    views = tuple(v.upper() for v in views)
    if unknown := [v for v in views if v not in ALL_VIEWS]:
        raise ValueError(f"unknown views {unknown}; choose from {list(ALL_VIEWS)}")
    if not views:
        raise ValueError("at least one view is needed")
    dirs = view_directions(front, up)
    proj = {name: project(shape, d, u) for name, (d, u) in dirs.items() if name in views}
    gap = sheet.style.spacing.view_gap
    x0, _, x1, y1 = sheet.inner
    band_top = sheet.title_block_box[3]
    area = (x0 + EDGE_PAD, band_top + EDGE_PAD, x1 - EDGE_PAD, y1 - EDGE_PAD)
    rev_h = sheet.rev_block_box(2)[3] - sheet.rev_block_box(2)[1]
    if len(views) == 1:
        return _single(sheet, views[0], proj[views[0]], (area[0], area[1], area[2], area[3] - rev_h), captions)
    return _cross(sheet, proj, area, rev_h, gap, captions)


def _single(sheet: Sheet, name: str, projection: Projection, area: tuple[float, float, float, float], captions: dict[str, str]) -> AutoLayout:
    """One view, centred in `area`: an orthographic view at the largest standard scale that leaves room for its dimensions, the isometric to fill."""
    sp = sheet.style.spacing
    pw, ph = _size(projection)
    room = sp.dim_offset + 2 * sp.caption_drop  # dimension on the left and below, caption under that
    aw, ah = area[2] - area[0] - room, area[3] - area[1] - room
    if name == "ISOMETRIC":
        s = min(aw / pw, ah / ph) * 0.92
    else:
        fitting = [k for k in STANDARD_SCALES if k * pw <= aw and k * ph <= ah]
        if not fitting:
            raise ValueError(f"{name} view does not fit the sheet at any standard scale: use a larger sheet size")
        s = fitting[0]
    corner = (area[0] + room + (aw - s * pw) / 2, area[1] + room + (ah - s * ph) / 2)
    is_iso = name == "ISOMETRIC"
    v = View.at_corner(sheet, projection, corner, s).draw(hidden=not is_iso, centre_marks=not is_iso)
    if is_iso:
        v.caption("ISOMETRIC", "not to scale", below=sp.caption_drop)
    else:
        v.overall_dims()
        v.caption(name, captions.get(name, ""))
    log.info("single %s view at scale %.2f", name, s)
    return AutoLayout(s, s, {name: v})


def _cross(sheet: Sheet, proj: dict[str, Projection], area: tuple[float, float, float, float], rev_h: float, gap: float,
           captions: dict[str, str]) -> AutoLayout:
    """Orthographic views in their third-angle positions around the front, and the isometric (if chosen) in the space to their right."""
    size = {name: _size(p) for name, p in proj.items()}
    has_iso = "ISOMETRIC" in proj

    def w(n: str) -> float:
        return size[n][0] if n in proj and n != "ISOMETRIC" else 0.0

    def h(n: str) -> float:
        return size[n][1] if n in proj and n != "ISOMETRIC" else 0.0

    def gaps(*names: str) -> float:
        return gap * sum(1 for n in names if n in proj)

    chosen = None
    for s in STANDARD_SCALES:
        cross_w = s * (w("LEFT") + w("FRONT") + w("RIGHT") + w("BACK")) + gaps("LEFT", "RIGHT", "BACK")
        cross_h = s * (h("TOP") + h("FRONT") + h("BOTTOM")) + gaps("TOP", "BOTTOM")
        if cross_h > area[3] - area[1] - (0 if has_iso else rev_h) or cross_w > area[2] - area[0]:
            continue
        if not has_iso:
            chosen = (s, 0.0, cross_w, cross_h)
            break
        iso_w_room = area[2] - area[0] - cross_w - gap
        iso_h_room = area[3] - area[1] - rev_h
        if iso_w_room <= 0:
            continue
        iw, ih = size["ISOMETRIC"]
        iso_s = min(s * ISO_MAX_ENLARGE, iso_w_room / iw, iso_h_room / ih)
        if iso_s >= s * ISO_MIN_FRACTION:
            chosen = (s, iso_s, cross_w, cross_h)
            break
    if chosen is None:
        raise ValueError("part does not fit the sheet at any standard scale: use a larger sheet size")
    s, iso_s, cross_w, cross_h = chosen
    log.info("scale %.2f, isometric %.2f", s, iso_s)

    fw, fh = s * w("FRONT"), s * h("FRONT")
    left_w = s * w("LEFT") + gaps("LEFT")
    # Without an isometric the cross is centred across the sheet and below the revision block; with one it hugs the left edge.
    fx = area[0] + left_w + (0 if has_iso else (area[2] - area[0] - cross_w) / 2)
    fy = area[1] + (area[3] - area[1] - (0 if has_iso else rev_h) - cross_h) / 2 + s * h("BOTTOM") + gaps("BOTTOM")
    corners: dict[str, tuple[float, float]] = {"FRONT": (fx, fy)} if "FRONT" in proj else {}
    if "TOP" in proj:
        corners["TOP"] = (fx + (fw - s * w("TOP")) / 2, fy + fh + gap)
    if "BOTTOM" in proj:
        corners["BOTTOM"] = (fx + (fw - s * w("BOTTOM")) / 2, fy - gap - s * h("BOTTOM"))
    if "LEFT" in proj:
        corners["LEFT"] = (fx - gap - s * w("LEFT"), fy + (fh - s * h("LEFT")) / 2)
    right_edge = fx + fw
    if "RIGHT" in proj:
        corners["RIGHT"] = (right_edge + gap, fy + (fh - s * h("RIGHT")) / 2)
        right_edge = corners["RIGHT"][0] + s * w("RIGHT")
    if "BACK" in proj:
        corners["BACK"] = (right_edge + gap, fy + (fh - s * h("BACK")) / 2)
    if has_iso:
        iw, ih = size["ISOMETRIC"]
        iso_left = area[0] + cross_w + gap
        iso_top = area[3] - rev_h
        corners["ISOMETRIC"] = (iso_left + (area[2] - iso_left - iso_s * iw) / 2, iso_top - iso_s * ih)

    placed: dict[str, View] = {}
    for name, corner in corners.items():
        is_iso = name == "ISOMETRIC"
        v = View.at_corner(sheet, proj[name], corner, iso_s if is_iso else s).draw(hidden=not is_iso, centre_marks=not is_iso)
        if is_iso:
            v.caption("ISOMETRIC", "not to scale", below=sheet.style.spacing.caption_drop)
        else:
            v.overall_dims()
            v.caption(name, captions.get(name, ""))
        placed[name] = v
    return AutoLayout(s, iso_s, placed)
