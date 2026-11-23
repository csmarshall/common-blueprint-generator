"""blueprint.yml: one file per drawing, describing the part and what the sheet should say. Paths in it are relative to the file.

KEYS is the schema. Loading, validation and the generated configuration reference (docs/config-reference.md) all read it, so the documentation
cannot describe a key the loader does not accept or miss one it does.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

from .layout import ALL_VIEWS, DEFAULT_ISO_SIZE, DEFAULT_VIEWS
from .licences import LICENCES, NONCOMMERCIAL_NOTE, Licence, lookup
from .preview import PREVIEW_HEIGHT, PREVIEW_VIEW, PREVIEW_WIDTH
from .sheet import CentreLine
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
        "model",
        "path",
        "required",
        "The part to draw: a STEP file (`.step`, `.stp`), or an STL as a fallback (curves then draw as facets; an STL over 100,000 triangles"
        " takes a fast numpy path and cannot be sectioned).",
        "build/dock.step",
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
        "centre_lines",
        "list",
        "[]",
        "Named axes to draw as chain lines, each `{label, point, axis}` in model coordinates (mm): `point` is any point on the axis, `axis` its "
        "direction (any length, not zero). The line runs across every orthographic view where the axis projects to a line, labelled `CL` plus the "
        "label. A view looking straight down the axis shows a point, and nothing is drawn there (circles seen face on already get their own centre "
        "mark). The isometric carries none.",
        "[{label: SOCKET AXIS, point: [12, 12, 0], axis: [0, 0, 1]}]",
    ),
    Key(
        "licence",
        "SPDX id or map",
        "none",
        "Override for the LICENCE row. Normally omitted: the licence and copyright are read from the repo's REUSE.toml for the drawing's own output "
        "file, and a render with no REUSE.toml or no matching annotation fails. Set this (an SPDX id) to skip that lookup. "
        f"Known ids ({', '.join(LICENCES)}) get their label, CC icons or a badge, and deed link; others print as given. Any id containing -NC- adds "
        f"{NONCOMMERCIAL_NOTE} unless given `{{id: ..., note: ...}}`.",
        "CC-BY-NC-SA-4.0",
    ),
    Key("copyright", "text", "from REUSE.toml", "Copyright line for the LICENCE row when `licence:` overrides the REUSE lookup.", "© 2026 Charles Marshall"),
    Key(
        "readme",
        "path",
        '""',
        "Markdown file to keep the drawing embedded in, between `<!-- blueprint:NAME -->` and `<!-- /blueprint -->` "
        "(NAME is the output file name). Each render rewrites the embed and a credit line linking this project.",
        "README.md",
    ),
    Key(
        "preview",
        "true or map",
        "none",
        "Also draw a sheet-free preview for a README header: the part with no border, title block or dimensions, as SVG. `true` draws this model; "
        f"a map sets `view` (default {PREVIEW_VIEW.lower()}), `width` (most mm across, default {PREVIEW_WIDTH:g}; parts stand at most "
        f"{PREVIEW_HEIGHT:g} mm tall), `output` (default OUTPUT-preview) and "
        "`parts`, a list of `{model, caption}` drawn side by side at one scale (default: this model, uncaptioned). With `readme`, it is kept "
        "embedded between `<!-- blueprint-preview:NAME -->` and `<!-- /blueprint-preview -->` (NAME is the preview's output file name).",
        "{parts: [{model: build/inline.step, caption: IN-LINE}, {model: build/end.step, caption: END}]}",
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


PREVIEW_KEYS = ("view", "width", "output", "parts")


@dataclass(frozen=True)
class PreviewConfig:
    """The `preview:` block: what to draw and where. Field meanings are in KEYS."""

    output: Path  # without extension
    view: str = PREVIEW_VIEW
    width: float = PREVIEW_WIDTH
    parts: tuple[tuple[Path, str], ...] = ()  # (model, caption), left to right


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
    licence: Licence | None = None
    licence_note: str | None = None  # None: the licence's default note
    copyright: str = ""
    front: str = "+Z"
    up: str = "+Y"
    sheet: str = "B"
    formats: tuple[str, ...] = ("png", "svg")
    views: tuple[str, ...] = DEFAULT_VIEWS
    isometric: float | None = DEFAULT_ISO_SIZE  # None fills the free space
    readme: Path | None = None
    captions: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    centre_lines: tuple[CentreLine, ...] = ()
    toggles: Toggles = Toggles()
    preview: PreviewConfig | None = None


def load(path: Path) -> DrawingConfig:
    """Read and check a blueprint.yml, raising ValueError naming the first problem."""
    loaded = yaml.safe_load(path.read_text()) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"{path}: expected a mapping of keys (model:, output:, title:, ...), not {type(loaded).__name__}")
    raw: dict[str, Any] = loaded
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
    formats = tuple(str(f) for f in _as_list(raw.get("formats", ("png", "svg"))))
    if not formats:
        raise ValueError(f"{path}: formats must name at least one of {list(FORMATS)}")
    if bad := [f for f in formats if f not in FORMATS]:
        raise ValueError(f"{path}: unsupported formats {bad}; use {list(FORMATS)}")
    views = tuple(dict.fromkeys(str(v).upper() for v in _as_list(raw.get("views") or DEFAULT_VIEWS)))
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
    if not_bool := [k for k, v in style.items() if not isinstance(v, bool)]:
        raise ValueError(f"{path}: style switches take true or false, unquoted; check {not_bool}")
    licence, licence_note = _licence(path, raw.get("licence"))
    model, output = (base / raw["model"]).resolve(), (base / raw["output"]).resolve()
    return DrawingConfig(
        path=path,
        model=model,
        output=output,
        title=str(raw["title"]),
        drawing_number=str(raw.get("drawing_number", "")),
        for_=str(raw.get("for", "")),
        material=str(raw.get("material", "")),
        source=str(raw.get("source", "")),
        drawn_by=str(raw.get("drawn_by", "")),
        company=str(raw.get("company", "")),
        licence=licence,
        licence_note=licence_note,
        copyright=str(raw.get("copyright", "")),
        front=front,
        up=up,
        sheet=sheet,
        formats=formats,
        views=views,
        isometric=isometric,
        readme=(base / raw["readme"]).resolve() if raw.get("readme") else None,
        captions={str(k).upper(): str(v) for k, v in (raw.get("captions") or {}).items()},
        notes=[str(n) for n in raw.get("notes") or []],
        centre_lines=_centre_lines(path, raw.get("centre_lines")),
        toggles=Toggles(**style),
        preview=_preview(path, raw.get("preview"), model, output),
    )


def _as_list(value: Any) -> list[Any]:
    """A YAML list as given, or a lone scalar (views: top) as a one-item list."""
    return list(value) if isinstance(value, list | tuple) else [value]


def _centre_lines(path: Path, value: Any) -> tuple[CentreLine, ...]:
    """`centre_lines: [{label, point: [x, y, z], axis: [x, y, z]}]` as CentreLines; every entry needs all three keys."""
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ValueError(f"{path}: centre_lines is a list of {{label, point, axis}} maps, not {value!r}")
    out: list[CentreLine] = []
    for entry in value:
        if not isinstance(entry, dict) or set(entry) != {"label", "point", "axis"}:
            raise ValueError(f"{path}: each centre line is a map of exactly label, point and axis, not {entry!r}")
        if not isinstance(entry["label"], str):
            raise ValueError(f"{path}: a centre line's label is text, not {entry['label']!r}")
        triples = []
        for name in ("point", "axis"):
            v = entry[name]
            if not isinstance(v, list) or len(v) != 3:
                raise ValueError(f"{path}: centre line {entry['label']!r}: {name} is a list of three numbers [x, y, z], not {v!r}")
            triples.append(v)
        try:
            out.append(CentreLine(entry["label"], tuple(triples[0]), tuple(triples[1])))
        except ValueError as e:
            raise ValueError(f"{path}: {e}") from None
    return tuple(out)


def _licence(path: Path, value: Any) -> tuple[Licence | None, str | None]:
    """`licence: CC-BY-NC-SA-4.0` or `licence: {id: CC-BY-NC-SA-4.0, note: ...}` as (licence, note override)."""
    if value is None or value == "":
        return None, None
    spdx, note = (value.get("id"), value.get("note")) if isinstance(value, dict) else (value, None)
    if not spdx:
        raise ValueError(f"{path}: licence: needs an SPDX id, e.g. licence: CC-BY-NC-SA-4.0 or licence: {{id: MIT, note: ...}}")
    return lookup(str(spdx)), None if note is None else str(note)


def _preview(path: Path, value: Any, model: Path, output: Path) -> PreviewConfig | None:
    """`preview: true` or `preview: {view, width, output, parts}` as a PreviewConfig; None when absent or false."""
    if value is None or value is False:
        return None
    default_output = output.with_name(f"{output.name}-preview")
    if value is True:
        return PreviewConfig(default_output, parts=((model, ""),))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: preview: takes true, or a map of {list(PREVIEW_KEYS)}, not {value!r}")
    if unknown := set(value) - set(PREVIEW_KEYS):
        raise ValueError(f"{path}: unknown preview keys {sorted(unknown)}; expected some of {list(PREVIEW_KEYS)}")
    view = str(value.get("view", PREVIEW_VIEW)).upper()
    if view not in ALL_VIEWS:
        raise ValueError(f"{path}: preview view must be one of {[v.lower() for v in ALL_VIEWS]}, not {view.lower()!r}")
    width = value.get("width", PREVIEW_WIDTH)
    if not isinstance(width, int | float) or isinstance(width, bool) or not math.isfinite(width) or width <= 0:
        raise ValueError(f"{path}: preview width is a positive number of mm, not {width!r}")
    base = path.parent
    parts: list[tuple[Path, str]] = []
    for entry in _as_list(value.get("parts") or [{"model": None}]):
        if not isinstance(entry, dict) or set(entry) - {"model", "caption"}:
            raise ValueError(f"{path}: each preview part is a map of model and caption, not {entry!r}")
        part_model = entry.get("model")
        if part_model is not None and not isinstance(part_model, str):
            raise ValueError(f"{path}: a preview part's model is a path, not {part_model!r}")
        parts.append(((base / part_model).resolve() if part_model else model, str(entry.get("caption") or "")))
    out_raw = value.get("output")
    if out_raw is not None and not isinstance(out_raw, str):
        raise ValueError(f"{path}: preview output is a path, not {out_raw!r}")
    out = (base / out_raw).resolve() if out_raw else default_output
    if out == output:
        raise ValueError(f"{path}: preview output {out_raw!r} is the sheet's own output: the preview would overwrite the sheet")
    return PreviewConfig(out, view, float(width), tuple(parts))
