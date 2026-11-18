"""blueprint.yml: one file per drawing, describing the part and what the sheet should say. Paths in it are relative to the file.

KEYS is the schema. Loading, validation and the generated configuration reference (docs/config-reference.md) all read it, so the documentation
cannot describe a key the loader does not accept or miss one it does.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

from .layout import ALL_VIEWS, DEFAULT_ISO_SIZE, DEFAULT_VIEWS
from .style import SHEET_SIZES, Toggles

FORMATS = ("png", "svg", "pdf")
AXIS_NAMES = ("+X", "-X", "+Y", "-Y", "+Z", "-Z")


@dataclass(frozen=True)
class Key:
    """One blueprint.yml key, as the loader checks it and the reference documents it."""

    name: str
    kind: str  # shown in the reference
    default: str  # shown in the reference; "required" for required keys
    doc: str
    example: str = ""


KEYS: tuple[Key, ...] = (
    Key(
        "model", "path", "required", "The part to draw: a STEP file (`.step`, `.stp`), or an STL as a fallback (curves then draw as facets).", "build/dock.step"
    ),
    Key("output", "path", "required", "Where to write the sheet, without an extension; one file per entry in `formats`.", "docs/images/blueprint_dock"),
    Key("title", "text", "required", "The drawing title, in the title block's top row.", "U1 TAG CLIP · FINGER DOCK"),
    Key("drawing_number", "text", '""', "DWG NO. field.", "STC-001"),
    Key("for", "text", '""', "FOR field: what the part fits or belongs to.", "Snapmaker U1 toolhead, filament slot 1"),
    Key("material", "text", '""', "MATERIAL field.", "ABS or PETG. Not PLA."),
    Key("source", "text", '""', "SOURCE field: the file the model comes from.", "u1/u1_parts.py"),
    Key("drawn_by", "text", '""', "DRAWN BY field.", "C. MARSHALL"),
    Key("company", "text", '""', "Name on the approved stamp. No stamp is drawn without one.", "Shermer Works"),
    Key("front", "axis", '"+Z"', f"Model axis pointing at the viewer in the front view: one of {', '.join(AXIS_NAMES)}.", '"+Z"'),
    Key("up", "axis", '"+Y"', "Model axis pointing up the page in the front view; must be perpendicular to `front`.", '"+Y"'),
    Key("sheet", "size", '"B"', f"ANSI sheet size: one of {', '.join(sorted(SHEET_SIZES))}.", '"B"'),
    Key("formats", "list", "[png, svg]", f"Output formats, any of {', '.join(FORMATS)}. SVG is the scalable copy; PNG is a preview.", "[svg]"),
    Key(
        "views",
        "list",
        "[" + ", ".join(v.lower() for v in DEFAULT_VIEWS) + "]",
        f"Which views to draw, any of {', '.join(v.lower() for v in ALL_VIEWS)}. Several keep their third-angle places around the front; "
        "one on its own is centred and drawn as large as a standard scale allows.",
        "[top]",
    ),
    Key(
        "isometric",
        "number or fill",
        f"{DEFAULT_ISO_SIZE:g}",
        "Isometric scale as a multiple of the main scale when it sits beside other views, or `fill` to fill the space to their right.",
        "1.6",
    ),
    Key("captions", "map", "{}", "Extra caption text per view, shown in brackets after the view name.", "{front: socket face, top: snap fingers up}"),
    Key("notes", "list", "[]", "General notes, numbered and wrapped in the lower-left corner.", "[Print flat edge down, no supports.]"),
    Key(
        "readme",
        "path",
        '""',
        "Markdown file to keep the drawing embedded in, between `<!-- blueprint:NAME -->` and `<!-- /blueprint -->` "
        "(NAME is the output file name). Each render rewrites the embed and a credit line linking this project.",
        "README.md",
    ),
    Key(
        "style",
        "map",
        "{}",
        "House-style switches: " + ", ".join(f"`{f.name}` (default {str(f.default).lower()})" for f in fields(Toggles)) + ".",
        "{grid: true, winks: false}",
    ),
)
KNOWN = {k.name for k in KEYS}
REQUIRED = tuple(k.name for k in KEYS if k.default == "required")


@dataclass
class DrawingConfig:
    """One drawing, as read from blueprint.yml. Field meanings are in KEYS."""

    path: Path  # the config file itself
    model: Path
    output: Path  # without extension
    title: str
    drawing_number: str = ""
    for_: str = ""
    material: str = ""
    source: str = ""
    drawn_by: str = ""
    company: str = ""
    front: str = "+Z"
    up: str = "+Y"
    sheet: str = "B"
    formats: tuple[str, ...] = ("png", "svg")
    views: tuple[str, ...] = DEFAULT_VIEWS
    isometric: float | None = DEFAULT_ISO_SIZE  # None fills the free space
    readme: Path | None = None
    captions: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    toggles: Toggles = Toggles()


def load(path: Path) -> DrawingConfig:
    """Read and check a blueprint.yml, raising ValueError naming the first problem."""
    raw: dict[str, Any] = yaml.safe_load(path.read_text()) or {}
    if unknown := set(raw) - KNOWN:
        raise ValueError(f"{path}: unknown keys {sorted(unknown)}; expected some of {sorted(KNOWN)}")
    for key in REQUIRED:
        if not raw.get(key):
            raise ValueError(f"{path}: '{key}' is required")
    base = path.parent
    sheet = str(raw.get("sheet", "B")).upper()
    if sheet not in SHEET_SIZES:
        raise ValueError(f"{path}: sheet must be one of {sorted(SHEET_SIZES)}, not {sheet!r}")
    front, up = str(raw.get("front", "+Z")).upper(), str(raw.get("up", "+Y")).upper()
    for name, axis in (("front", front), ("up", up)):
        if axis not in AXIS_NAMES:
            raise ValueError(f"{path}: {name} must be one of {list(AXIS_NAMES)}, not {axis!r}")
    formats = tuple(raw.get("formats", ("png", "svg")))
    if bad := [f for f in formats if f not in FORMATS]:
        raise ValueError(f"{path}: unsupported formats {bad}; use {list(FORMATS)}")
    views = tuple(str(v).upper() for v in raw.get("views") or DEFAULT_VIEWS)
    if bad_views := [v for v in views if v not in ALL_VIEWS]:
        raise ValueError(f"{path}: unknown views {bad_views}; choose from {[v.lower() for v in ALL_VIEWS]}")
    iso_raw = raw.get("isometric", DEFAULT_ISO_SIZE)
    isometric: float | None
    if iso_raw == "fill":
        isometric = None
    elif isinstance(iso_raw, int | float) and not isinstance(iso_raw, bool) and iso_raw > 0:
        isometric = float(iso_raw)
    else:
        raise ValueError(f"{path}: isometric must be a positive multiple of the main scale (e.g. 1.0, 1.6) or 'fill', not {iso_raw!r}")
    style = raw.get("style") or {}
    toggle_names = {f.name for f in fields(Toggles)}
    if bad_toggles := set(style) - toggle_names:
        raise ValueError(f"{path}: unknown style switches {sorted(bad_toggles)}; expected some of {sorted(toggle_names)}")
    return DrawingConfig(
        path=path,
        model=(base / raw["model"]).resolve(),
        output=(base / raw["output"]).resolve(),
        title=str(raw["title"]),
        drawing_number=str(raw.get("drawing_number", "")),
        for_=str(raw.get("for", "")),
        material=str(raw.get("material", "")),
        source=str(raw.get("source", "")),
        drawn_by=str(raw.get("drawn_by", "")),
        company=str(raw.get("company", "")),
        front=front,
        up=up,
        sheet=sheet,
        formats=formats,
        views=views,
        isometric=isometric,
        readme=(base / raw["readme"]).resolve() if raw.get("readme") else None,
        captions={str(k).upper(): str(v) for k, v in (raw.get("captions") or {}).items()},
        notes=[str(n) for n in raw.get("notes") or []],
        toggles=Toggles(**{k: bool(v) for k, v in style.items()}),
    )
