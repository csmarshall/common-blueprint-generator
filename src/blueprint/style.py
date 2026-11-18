"""The house style: one look shared by every repo, decided element by element in the style review (REV-001).

Base drafting is the clean bracket sheet's (flat blue, crisp lines, monospace lettering, light-blue dimension ink). The aged tag-clip sheet contributes the
paper ageing, the approved stamp and the deadpan winks, each of which a repo can switch off in its config.

Text sizes and line widths are in points on a figure drawn at true sheet size, so 1 sheet mm is 1 mm on the printed page.
"""

from __future__ import annotations

from dataclasses import dataclass

PT_PER_MM = 72.0 / 25.4


@dataclass(frozen=True)
class SheetSize:
    """A named drawing sheet, landscape, in millimetres."""

    name: str
    width: float
    height: float
    wink: str  # what the SIZE field says after the name


# US drawing sizes (ASME Y14.1). B is the default: the closest to the A3 the bracket sheet used. A (8.5 x 11) is not offered: the title block and
# tolerance column alone are 262 mm wide, wider than A's 255 mm inside the border.
SHEET_SIZES = {
    "B": SheetSize("B", 431.8, 279.4, "11 × 17, THE TABLOID ONE"),
    "C": SheetSize("C", 558.8, 431.8, "17 × 22, THE PLOTTER ONE"),
}

# Scales a drawing may be drawn at, largest first. Sheet mm per model mm.
STANDARD_SCALES = (5.0, 4.0, 3.0, 2.0, 1.5, 1.0, 0.75, 0.5, 0.4, 0.25, 0.2, 0.1)


@dataclass(frozen=True)
class Toggles:
    """Per-repo switches for the parts of the house style that are personality rather than drafting."""

    aged: bool = True  # paper grain, blotches, fold creases and vignette
    grid: bool = False  # faint 10 mm graph grid
    stamp: bool = True  # the approved rubber stamp
    winks: bool = True  # deadpan captions on the projection symbol, size field, centring marks, standards footer and tolerance block
    credit: bool = True  # a small line in the bottom margin naming and linking the generator


@dataclass(frozen=True)
class Palette:
    """Every colour on the sheet."""

    paper: str = "#0b3a82"
    grid: str = "#1d4f9c"
    ink: str = "#f4f8ff"
    dim: str = "#9fd0ff"  # dimensions, callouts and leaders: a second ink so they read apart from the part


@dataclass(frozen=True)
class Lines:
    """Line widths in points and dash patterns, from the clean sheet."""

    visible: float = 1.15
    hidden: float = 0.45
    hidden_alpha: float = 0.75
    hidden_dash: tuple[float, float] = (4.0, 2.5)
    dim: float = 0.5
    extension: float = 0.4
    centre: float = 0.5
    centre_dash: tuple[float, float, float, float] = (9.0, 2.0, 2.0, 2.0)  # long dash, gap, short dash, gap
    border_outer: float = 1.4
    border_inner: float = 0.5
    table: float = 0.5
    grid: float = 0.25


@dataclass(frozen=True)
class Type:
    """Lettering: one monospace family, sizes in points."""

    family: str = "DejaVu Sans Mono"  # ships with matplotlib, so every runner has it
    title: float = 13.0
    caption: float = 7.5
    body: float = 6.5
    note: float = 5.8
    dim: float = 6.2
    label: float = 4.6


@dataclass(frozen=True)
class Spacing:
    """Sheet-millimetre distances used by the layout and the annotation helpers."""

    margin: float = 10.0  # paper edge to the outer border line
    border_gap: float = 2.0  # outer to inner border line
    view_gap: float = 26.0  # between neighbouring views, room for a dimension and a caption
    dim_offset: float = 8.0  # dimension line distance from its view
    extension_gap: float = 1.2  # gap between the part and an extension line
    extension_overrun: float = 1.5  # extension line run past its dimension line
    caption_drop: float = 7.0  # caption distance below a view's lowest annotation
    centre_overrun: float = 2.5  # centre mark run past a circle


@dataclass(frozen=True)
class Style:
    """The whole house style."""

    palette: Palette = Palette()
    lines: Lines = Lines()
    type: Type = Type()
    spacing: Spacing = Spacing()
    toggles: Toggles = Toggles()
    seed: int = 1932  # fixes the paper ageing, so a rebuild draws the same sheet


def fmt(value: float) -> str:
    """A dimension value: two decimals with trailing zeros dropped (18, 3.85, 305)."""
    return f"{value:.2f}".rstrip("0").rstrip(".")


def scale_label(scale: float) -> str:
    """A scale as drawings write it: 2:1 for enlargements, 1:1.33 for reductions."""
    if scale >= 1:
        return f"{fmt(scale)}:1"
    return f"1:{fmt(1 / scale)}"
