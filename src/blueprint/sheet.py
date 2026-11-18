"""The drawing sheet: paper, border, title block, revision and tolerance blocks, notes, stamp, and View for placing projected geometry with dimensions.

Coordinates are sheet millimetres from the lower-left paper corner. A repo with a hand-laid-out drawing (section views, chain dimensions, callouts)
builds a Sheet, places Views on it and annotates them; layout.auto_sheet does the same for the standard six-view drawing.
"""

from __future__ import annotations

import logging
import math
import textwrap
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import colors as mcolors  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.patches import Circle, PathPatch, Polygon, Rectangle  # noqa: E402
from matplotlib.path import Path as MplPath  # noqa: E402
from scipy.ndimage import gaussian_filter  # noqa: E402

from .geometry import Polyline, Projection  # noqa: E402
from .readme import PROJECT_NAME, PROJECT_URL  # noqa: E402
from .style import PT_PER_MM, SHEET_SIZES, SheetSize, Style, fmt, scale_label  # noqa: E402

log = logging.getLogger(__name__)

PAPER_PX_PER_MM = 4  # resolution of the aged paper texture
TITLE_BLOCK_W = 170.0
TITLE_ROW_H = 9.0
TITLE_HEAD_H = 14.0
SIDE_COLUMN_W = 92.0  # tolerance block and projection symbol, left of the title block
REV_COL_W = (12.0, 76.0, 26.0)  # REV, DESCRIPTION, DATE
REV_ROW_H = 6.0
STAMP_R = 19.0
PNG_DPI = 150  # 2550 px across an ANSI B sheet; the SVG is the scalable copy, so the PNG only needs to be a sharp preview


@dataclass
class Revision:
    """One row of the revision block."""

    rev: str
    description: str
    when: date


@dataclass
class TitleInfo:
    """Everything the title block, revision block and stamp say. Empty fields are left blank on the sheet."""

    title: str
    drawing_number: str = ""
    for_: str = ""
    material: str = ""
    source: str = ""
    drawn_by: str = ""
    company: str = ""  # appears on the stamp
    started: date | None = None
    updated: date | None = None  # the render date when None
    revisions: list[Revision] = field(default_factory=list)
    units: str = "MM"
    tolerances: Sequence[str] = ("X ±0.5", "X.X ±0.2", "X.XX ±0.1", "ANGLES ±1°")


def _date(d: date | None) -> str:
    """A title-block date: 18 NOV 2026."""
    return d.strftime("%d %b %Y").upper() if d else ""


class Sheet:
    """One drawing sheet with its border and furniture, ready for views."""

    def __init__(self, size: str = "B", style: Style | None = None) -> None:
        """Create the figure at true size and draw the paper and border."""
        self.size: SheetSize = SHEET_SIZES[size]
        self.style = style or Style()
        self.w, self.h = self.size.width, self.size.height
        self.fig: Figure = plt.figure(figsize=(self.w / 25.4, self.h / 25.4))
        self.ax: Axes = self.fig.add_axes((0, 0, 1, 1))
        self.ax.set_xlim(0, self.w)
        self.ax.set_ylim(0, self.h)
        self.ax.set_aspect("equal")
        self.ax.axis("off")
        sp = self.style.spacing
        self.inner = (sp.margin + sp.border_gap, sp.margin + sp.border_gap, self.w - sp.margin - sp.border_gap, self.h - sp.margin - sp.border_gap)
        self._paper()
        self._border()

    # ------------------------------------------------------------------ regions the furniture occupies, for layouts to avoid
    @property
    def title_block_box(self) -> tuple[float, float, float, float]:
        """(x0, y0, x1, y1) of the title block plus the tolerance and projection column beside it."""
        x1, y0 = self.inner[2], self.inner[1]
        return x1 - TITLE_BLOCK_W - SIDE_COLUMN_W, y0, x1, y0 + TITLE_HEAD_H + 5 * TITLE_ROW_H

    def rev_block_box(self, rows: int) -> tuple[float, float, float, float]:
        """(x0, y0, x1, y1) of a revision block with `rows` rows, in the top-right corner."""
        x1, y1 = self.inner[2], self.inner[3]
        return x1 - sum(REV_COL_W), y1 - REV_ROW_H * (rows + 2), x1, y1

    @property
    def stamp_centre(self) -> tuple[float, float]:
        """Where the stamp sits: in the open band left of the title block, its ring just clear of the border."""
        x0, y0, _, _ = self.title_block_box
        return x0 - STAMP_R - 8, y0 + STAMP_R + 6

    # ------------------------------------------------------------------ primitives
    def text(self, x: float, y: float, s: str, size: float, color: str | None = None, **kw: object) -> None:
        """Lettering in the house face."""
        self.ax.text(x, y, s, fontsize=size, fontfamily=self.style.type.family, color=color or self.style.palette.ink, **kw)  # type: ignore[arg-type]

    def line(self, xs: Sequence[float], ys: Sequence[float], width: float, color: str | None = None, **kw: object) -> None:
        """A plain line."""
        self.ax.plot(xs, ys, color=color or self.style.palette.ink, lw=width, **kw)  # type: ignore[arg-type]

    def centre_line(self, a: tuple[float, float], b: tuple[float, float]) -> None:
        """A long-dash-dot chain line from a to b."""
        ln = self.style.lines
        self.ax.plot([a[0], b[0]], [a[1], b[1]], color=self.style.palette.ink, lw=ln.centre, dashes=ln.centre_dash, alpha=0.85)

    def dimension(self, a: tuple[float, float], b: tuple[float, float], text: str, vertical: bool = False) -> None:
        """A dimension line with arrows at both ends and its value on a paper patch at the middle."""
        pal, ln = self.style.palette, self.style.lines
        self.ax.annotate(
            "", xy=a, xytext=b, arrowprops={"arrowstyle": "<|-|>", "color": pal.dim, "lw": ln.dim, "shrinkA": 0, "shrinkB": 0, "mutation_scale": 6}
        )
        self.text(
            (a[0] + b[0]) / 2,
            (a[1] + b[1]) / 2,
            text,
            self.style.type.dim,
            pal.dim,
            ha="center",
            va="center",
            rotation=90 if vertical else 0,
            bbox={"boxstyle": "square,pad=0.15", "fc": pal.paper, "ec": "none"},
        )

    def leader(self, at: tuple[float, float], text_at: tuple[float, float], text: str, ha: str = "left") -> None:
        """A callout: text with a thin arrow to the feature."""
        pal = self.style.palette
        self.ax.annotate(
            text,
            xy=at,
            xytext=text_at,
            color=pal.dim,
            fontsize=self.style.type.dim,
            fontfamily=self.style.type.family,
            ha=ha,
            va="center",
            arrowprops={"arrowstyle": "-|>", "color": pal.dim, "lw": self.style.lines.dim, "shrinkA": 1.5, "shrinkB": 0, "mutation_scale": 6},
        )

    def caption(self, x: float, y: float, name: str, detail: str = "") -> None:
        """A view caption: bold name, with an optional explanation in brackets."""
        self.text(x, y, f"{name} ({detail})" if detail else name, self.style.type.caption, ha="center", va="top", weight="bold")

    # ------------------------------------------------------------------ paper and border
    def _paper(self) -> None:
        """Flat blue paper, aged with grain, blotches, two fold creases and a vignette if the toggle is on, and the grid if that one is."""
        st = self.style
        rgb = np.array(mcolors.to_rgb(st.palette.paper))
        if st.toggles.aged:
            wpx, hpx = int(self.w * PAPER_PX_PER_MM), int(self.h * PAPER_PX_PER_MM)
            rng = np.random.default_rng(st.seed)
            grain = rng.normal(0, 0.03, (hpx, wpx))
            blotches = gaussian_filter(rng.normal(0, 1, (hpx, wpx)), sigma=min(wpx, hpx) / 12)
            blotches *= 0.09 / max(1e-9, float(np.abs(blotches).max()))
            yy, xx = np.mgrid[0:hpx, 0:wpx]
            nx, ny = xx / wpx - 0.5, yy / hpx - 0.5
            vignette = -0.5 * (nx**2 + ny**2) ** 1.4
            creases = 0.05 * np.exp(-(((xx - wpx / 2) / 3.0) ** 2)) + 0.04 * np.exp(-(((yy - hpx / 2) / 3.0) ** 2))
            shade = 1.0 + grain + blotches + vignette + creases
            self.ax.imshow(
                np.clip(rgb[None, None, :] * shade[:, :, None], 0, 1), extent=(0, self.w, 0, self.h), origin="lower", interpolation="bilinear", zorder=-10
            )
        else:
            self.ax.add_patch(Rectangle((0, 0), self.w, self.h, fc=st.palette.paper, ec="none", zorder=-10))
        if st.toggles.grid:
            x0, y0, x1, y1 = self.inner
            for x in np.arange(math.ceil(x0 / 10) * 10, x1, 10):
                self.line([x, x], [y0, y1], st.lines.grid, st.palette.grid, zorder=-9)
            for y in np.arange(math.ceil(y0 / 10) * 10, y1, 10):
                self.line([x0, x1], [y, y], st.lines.grid, st.palette.grid, zorder=-9)

    def _border(self) -> None:
        """Double border, with centring marks at the middle of each edge (the top one says which way is up when winks are on)."""
        st, sp = self.style, self.style.spacing
        m = sp.margin
        self.ax.add_patch(Rectangle((m, m), self.w - 2 * m, self.h - 2 * m, fc="none", ec=st.palette.ink, lw=st.lines.border_outer))
        x0, y0, x1, y1 = self.inner
        self.ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fc="none", ec=st.palette.ink, lw=st.lines.border_inner))
        mark = m * 0.6
        cx, cy = self.w / 2, self.h / 2
        for xs, ys in (
            ([cx, cx], [m - mark, m + sp.border_gap]),
            ([cx, cx], [self.h - m - sp.border_gap, self.h - m + mark]),
            ([m - mark, m + sp.border_gap], [cy, cy]),
            ([self.w - m - sp.border_gap, self.w - m + mark], [cy, cy]),
        ):
            self.line(xs, ys, st.lines.border_outer)
        if st.toggles.winks:
            self.text(cx + 2.5, self.h - m + mark / 2, "↑ THIS WAY UP", st.type.label, st.palette.dim, ha="left", va="center")

    # ------------------------------------------------------------------ furniture
    def title_block(self, info: TitleInfo, scale: float, detail_scales: Sequence[float] = ()) -> None:
        """Title block, tolerance block, projection symbol, revision block, standards footer and stamp."""
        st, pal, ty, ln = self.style, self.style.palette, self.style.type, self.style.lines
        x1, y0 = self.inner[2], self.inner[1]
        x0 = x1 - TITLE_BLOCK_W
        top = y0 + TITLE_HEAD_H + 5 * TITLE_ROW_H
        self.ax.add_patch(Rectangle((x0, y0), TITLE_BLOCK_W, top - y0, fc="none", ec=pal.ink, lw=1.0))

        def cell(x: float, y: float, w: float, head: str, body: str, size: float = ty.body) -> None:
            self.ax.add_patch(Rectangle((x, y), w, TITLE_ROW_H, fc="none", ec=pal.ink, lw=ln.table))
            self.text(x + 1.5, y + TITLE_ROW_H - 1.2, head, ty.label, pal.dim, va="top")
            self.text(x + 1.5, y + 1.3, body, size, va="bottom")

        third = TITLE_BLOCK_W / 3
        scale_text = scale_label(scale) + "".join(f" (DETAIL {scale_label(s)})" for s in detail_scales)
        rows = [
            [("STARTED", _date(info.started)), ("UPDATED", _date(info.updated or date.today())), ("SHEET", "1 OF 1")],
            [("DRAWN BY", info.drawn_by), ("DWG NO.", info.drawing_number), ("SIZE", self.size.name + (f" ({self.size.wink})" if st.toggles.winks else ""))],
            [("SCALE", scale_text), ("UNITS", info.units), ("MATERIAL", info.material)],
        ]
        for r, row in enumerate(rows):
            for c, (head, body) in enumerate(row):
                size = ty.label + 0.6 if head == "SIZE" and st.toggles.winks else ty.body
                cell(x0 + c * third, y0 + r * TITLE_ROW_H, third, head, body, size)
        cell(x0, y0 + 3 * TITLE_ROW_H, TITLE_BLOCK_W, "SOURCE", info.source)
        cell(x0, y0 + 4 * TITLE_ROW_H, TITLE_BLOCK_W, "FOR", info.for_)
        self.text(x0 + 3, top - TITLE_HEAD_H / 2, info.title, ty.title, va="center", weight="bold")
        if st.toggles.winks:
            self.text(x1, y0 - st.spacing.border_gap - 1.2, "DRAWN PER ASME Y14.5 (MOSTLY)", ty.label, pal.dim, ha="right", va="top")
        if st.toggles.credit:
            url = PROJECT_URL.removeprefix("https://")
            self.text(
                self.inner[0],
                y0 - st.spacing.border_gap - 1.2,
                f"DRAWN WITH {PROJECT_NAME.upper()} · {url}",
                ty.label,
                pal.dim,
                ha="left",
                va="top",
                url=PROJECT_URL,
            )
        self._side_column(info)
        self._revision_block(info)
        if st.toggles.stamp and info.company:
            self.stamp(*self.stamp_centre, info.company)

    def _side_column(self, info: TitleInfo) -> None:
        """Tolerance block above the third-angle projection symbol, to the left of the title block."""
        st, pal, ty, ln = self.style, self.style.palette, self.style.type, self.style.lines
        x1, y0 = self.inner[2] - TITLE_BLOCK_W, self.inner[1]
        x0 = x1 - SIDE_COLUMN_W
        h_total = TITLE_HEAD_H + 5 * TITLE_ROW_H
        h_sym = 26.0
        self.ax.add_patch(Rectangle((x0, y0), SIDE_COLUMN_W, h_sym, fc="none", ec=pal.ink, lw=ln.table))
        self.ax.add_patch(Rectangle((x0, y0 + h_sym), SIDE_COLUMN_W, h_total - h_sym, fc="none", ec=pal.ink, lw=ln.table))
        lines = ["UNLESS OTHERWISE SPECIFIED:", f"DIMENSIONS IN {info.units}", "   ".join(info.tolerances[:3]), *info.tolerances[3:]]
        if st.toggles.winks:
            lines.append("OR WHATEVER THE PRINTER GIVES")
        for i, s in enumerate(lines):
            self.text(x0 + 2, y0 + h_total - 3 - i * 4.4, s, ty.label + 0.8, pal.dim if i == len(lines) - 1 and st.toggles.winks else pal.ink, va="top")
        self._projection_symbol(x0 + 8, y0 + 10, 9.0)
        self.text(x0 + 2, y0 + h_sym - 2.2, "THIRD ANGLE PROJECTION", ty.label, pal.dim, va="top")
        if st.toggles.winks:
            self.text(x0 + 2, y0 + 2.2, "(THE AMERICAN ONE)", ty.label, pal.dim, va="bottom")

    def _projection_symbol(self, x: float, y: float, d: float) -> None:
        """The third-angle symbol: a truncated cone's end view (two circles) to the right of its side view (a trapezoid), with centre lines."""
        pal, lw = self.style.palette, self.style.lines.visible * 0.7
        h_small = d * 0.55
        side_w = d * 1.15
        self.ax.add_patch(
            Polygon([(x, y + (d - h_small) / 2), (x + side_w, y), (x + side_w, y + d), (x, y + (d + h_small) / 2)], closed=True, fc="none", ec=pal.ink, lw=lw)
        )
        cx, cy = x + side_w + d * 0.6 + d / 2, y + d / 2
        self.ax.add_patch(Circle((cx, cy), d / 2, fc="none", ec=pal.ink, lw=lw))
        self.ax.add_patch(Circle((cx, cy), h_small / 2, fc="none", ec=pal.ink, lw=lw))
        self.centre_line((x - 1.5, cy), (cx + d / 2 + 1.5, cy))
        self.centre_line((cx, y - 1.5), (cx, y + d + 1.5))

    def _revision_block(self, info: TitleInfo) -> None:
        """REV / DESCRIPTION / DATE table in the top-right corner, one row per revision, or one initial-release row when there are none."""
        pal, ty, ln = self.style.palette, self.style.type, self.style.lines
        revisions = info.revisions or [Revision("A", "INITIAL RELEASE", info.started or info.updated or date.today())]
        x0, y0, x1, y1 = self.rev_block_box(len(revisions))
        self.ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fc=pal.paper, ec=pal.ink, lw=1.0, zorder=2))
        self.text((x0 + x1) / 2, y1 - REV_ROW_H / 2, "REVISIONS", ty.body, ha="center", va="center", weight="bold", zorder=3)
        heads = ("REV", "DESCRIPTION", "DATE")
        rows = [heads, *[(r.rev, r.description, _date(r.when)) for r in revisions]]
        for i, row in enumerate(rows):
            ytop = y1 - REV_ROW_H * (i + 1)
            self.line([x0, x1], [ytop, ytop], ln.table, zorder=3)
            x = x0
            for w, s in zip(REV_COL_W, row, strict=True):
                self.text(x + 1.5, ytop - REV_ROW_H / 2, s, ty.label if i == 0 else ty.label + 1.2, pal.dim if i == 0 else pal.ink, va="center", zorder=3)
                x += w
        x = x0
        for w in REV_COL_W[:-1]:
            x += w
            self.line([x, x], [y0, y1 - REV_ROW_H], ln.table, zorder=3)

    def notes(self, lines: Sequence[str], x: float | None = None, y: float | None = None, right: float | None = None) -> None:
        """Numbered general notes, wrapped to the space between `x` and `right` (default: left border to the stamp)."""
        ty = self.style.type
        x = self.inner[0] + 3 if x is None else x
        top = self.title_block_box[3] if y is None else y
        right = (self.stamp_centre[0] - STAMP_R - 6) if right is None else right
        char_w = ty.note / PT_PER_MM * 0.6  # monospace advance is 0.6 em
        width = max(20, int((right - x) / char_w))
        out = ["NOTES"]
        for i, note in enumerate(lines, 1):
            wrapped = textwrap.wrap(note, width - 4) or [""]
            out.append(f"{i}. {wrapped[0]}")
            out += [f"   {w}" for w in wrapped[1:]]
        self.text(x, top, "\n".join(out), ty.note, va="top", linespacing=1.45)
        log.debug("notes: %d lines at %d chars", len(out), width)

    def stamp(self, cx: float, cy: float, company: str) -> None:
        """The approved stamp, in the house's clean lettering: a tilted double ring with the company name over APPROVED FOR PRINTING."""
        pal, ty = self.style.palette, self.style.type
        tilt = 12.0
        for r, w in ((STAMP_R, 1.3), (STAMP_R - 2.4, 0.5)):
            self.ax.add_patch(Circle((cx, cy), r, fc="none", ec=pal.dim, lw=w, alpha=0.85))
        rad = math.radians(tilt)

        def at(dy: float) -> tuple[float, float]:
            return cx - dy * math.sin(rad), cy + dy * math.cos(rad)

        for dy, s, size, weight in (
            (7.0, company.upper(), ty.label + 1.2, "normal"),
            (0.0, "APPROVED", ty.caption + 2.5, "bold"),
            (-7.0, "FOR PRINTING", ty.label + 1.2, "normal"),
        ):
            self.text(*at(dy), s, size, pal.dim, ha="center", va="center", rotation=tilt, weight=weight, alpha=0.9)

    # ------------------------------------------------------------------ output
    def save(self, stem: Path, formats: Sequence[str] = ("png", "svg"), dpi: int = PNG_DPI) -> list[Path]:
        """Write the sheet as stem.<format> for each format and close the figure.

        Output is byte-stable for the same drawing on the same toolchain (fixed SVG id salt, no creation date in the metadata), so a re-render that
        changes nothing commits nothing.
        """
        stem.parent.mkdir(parents=True, exist_ok=True)
        written = []
        no_dates: dict[str, dict[str, str | None]] = {"svg": {"Date": None}, "pdf": {"CreationDate": None, "ModDate": None}, "png": {}}
        for ext in formats:
            path = stem.with_suffix(f".{ext}")
            with plt.rc_context({"svg.hashsalt": "common-blueprint-generator"}):
                self.fig.savefig(path, dpi=dpi if ext == "png" else None, facecolor=self.style.palette.paper, metadata=no_dates.get(ext, {}))
            log.info("wrote %s (%d KiB)", path, path.stat().st_size // 1024)
            written.append(path)
        plt.close(self.fig)
        return written


class View:
    """Projected geometry placed on a sheet: model millimetres times `scale`, model origin at `origin` in sheet millimetres."""

    def __init__(self, sheet: Sheet, projection: Projection, origin: tuple[float, float], scale: float) -> None:
        """Place `projection` with model (0, 0) at sheet `origin`."""
        self.sheet, self.proj, self.origin, self.scale = sheet, projection, np.array(origin, float), scale

    @classmethod
    def at_corner(cls, sheet: Sheet, projection: Projection, corner: tuple[float, float], scale: float) -> View:
        """Place `projection` with the lower-left corner of its visible outline at sheet `corner`."""
        x0, y0, _, _ = projection.bounds()
        return cls(sheet, projection, (corner[0] - x0 * scale, corner[1] - y0 * scale), scale)

    def to_sheet(self, pts: Sequence[float] | np.ndarray) -> np.ndarray:
        """Model view-plane points (N x 2, or one point) to sheet millimetres."""
        return self.origin + np.atleast_2d(np.asarray(pts, float)) * self.scale

    def box(self) -> tuple[float, float, float, float]:
        """(x0, y0, x1, y1) of the visible outline on the sheet."""
        x0, y0, x1, y1 = self.proj.bounds()
        a, b = self.to_sheet((x0, y0))[0], self.to_sheet((x1, y1))[0]
        return float(a[0]), float(a[1]), float(b[0]), float(b[1])

    def draw(self, hidden: bool = True, centre_marks: bool = True) -> View:
        """Draw the hidden lines, then the visible lines, then a centre mark through every circle seen face on."""
        st = self.sheet.style
        ln, pal = st.lines, st.palette
        if hidden:
            for line in self.proj.hidden:
                s = self.to_sheet(line)
                self.sheet.ax.plot(s[:, 0], s[:, 1], color=pal.ink, lw=ln.hidden, alpha=ln.hidden_alpha, dashes=ln.hidden_dash, solid_capstyle="butt")
        for line in self.proj.visible:
            s = self.to_sheet(line)
            self.sheet.ax.plot(s[:, 0], s[:, 1], color=pal.ink, lw=ln.visible, solid_capstyle="round")
        if centre_marks:
            for cx, cy, r in self.proj.circles:
                c = self.to_sheet((cx, cy))[0]
                reach = r * self.scale + st.spacing.centre_overrun
                self.sheet.centre_line((c[0] - reach, c[1]), (c[0] + reach, c[1]))
                self.sheet.centre_line((c[0], c[1] - reach), (c[0], c[1] + reach))
        return self

    def dim_h(self, x0: float, x1: float, y_ref: float, offset: float, text: str | None = None, ext: bool = True) -> None:
        """Horizontal dimension between model x0 and x1, drawn `offset` sheet mm above (positive) or below model y_ref."""
        sp = self.sheet.style.spacing
        a, b = self.to_sheet((x0, y_ref))[0], self.to_sheet((x1, y_ref))[0]
        y = a[1] + offset
        self.sheet.dimension((a[0], y), (b[0], y), text or fmt(abs(x1 - x0)))
        if ext:
            sign = math.copysign(1, offset)
            for p in (a, b):
                self.sheet.line(
                    [p[0], p[0]],
                    [p[1] + sign * sp.extension_gap, y + sign * sp.extension_overrun],
                    self.sheet.style.lines.extension,
                    self.sheet.style.palette.dim,
                )

    def dim_v(self, y0: float, y1: float, x_ref: float, offset: float, text: str | None = None, ext: bool = True) -> None:
        """Vertical dimension between model y0 and y1, drawn `offset` sheet mm right (positive) or left of model x_ref."""
        sp = self.sheet.style.spacing
        a, b = self.to_sheet((x_ref, y0))[0], self.to_sheet((x_ref, y1))[0]
        x = a[0] + offset
        self.sheet.dimension((x, a[1]), (x, b[1]), text or fmt(abs(y1 - y0)), vertical=True)
        if ext:
            sign = math.copysign(1, offset)
            for p in (a, b):
                self.sheet.line(
                    [p[0] + sign * sp.extension_gap, x + sign * sp.extension_overrun],
                    [p[1], p[1]],
                    self.sheet.style.lines.extension,
                    self.sheet.style.palette.dim,
                )

    def overall_dims(self) -> None:
        """Overall width below the view and overall height to its left, measured from the projection."""
        x0, y0, x1, y1 = self.proj.bounds()
        off = self.sheet.style.spacing.dim_offset
        self.dim_h(x0, x1, y0, -off)
        self.dim_v(y0, y1, x0, -off)

    def leader(self, model_xy: tuple[float, float], text_at: tuple[float, float], text: str, ha: str = "left") -> None:
        """A callout from a model point to text at a sheet position."""
        at = self.to_sheet(model_xy)[0]
        self.sheet.leader((float(at[0]), float(at[1])), text_at, text, ha)

    def caption(self, name: str, detail: str = "", below: float | None = None) -> None:
        """Caption centred under the view, `below` sheet mm under its outline (default: clear of an overall dimension)."""
        sp = self.sheet.style.spacing
        x0, y0, x1, _ = self.box()
        drop = sp.dim_offset + sp.caption_drop if below is None else below
        self.sheet.caption((x0 + x1) / 2, y0 - drop, name, detail)

    def hatch(self, loops: Sequence[Polyline]) -> None:
        """Section hatching over closed loops given in this view's model coordinates."""
        verts: list[list[float]] = []
        codes: list[np.uint8] = []
        for loop in loops:
            s = self.to_sheet(loop)
            verts.extend(s.tolist())
            codes.extend([MplPath.MOVETO] + [MplPath.LINETO] * (len(s) - 2) + [MplPath.CLOSEPOLY])
        if verts:
            self.sheet.ax.add_patch(PathPatch(MplPath(verts, codes), fc="none", ec=self.sheet.style.palette.dim, lw=0, hatch="////", alpha=0.9))
