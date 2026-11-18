"""blueprint.yml: one file per drawing, describing the part and what the sheet should say. Paths in it are relative to the file."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

from .layout import ALL_VIEWS, DEFAULT_VIEWS
from .style import SHEET_SIZES, Toggles

FORMATS = ("png", "svg", "pdf")


@dataclass
class DrawingConfig:
    """One drawing, as read from blueprint.yml."""

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
    views: tuple[str, ...] = DEFAULT_VIEWS  # any subset; one name alone draws that view by itself, as large as fits
    captions: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    toggles: Toggles = Toggles()


KNOWN = {"model", "output", "title", "drawing_number", "for", "material", "source", "drawn_by", "company", "front", "up", "sheet", "formats",
         "views", "captions", "notes", "style"}


def load(path: Path) -> DrawingConfig:
    """Read and check a blueprint.yml, raising ValueError naming the first problem."""
    raw: dict[str, Any] = yaml.safe_load(path.read_text()) or {}
    unknown = set(raw) - KNOWN
    if unknown:
        raise ValueError(f"{path}: unknown keys {sorted(unknown)}; expected some of {sorted(KNOWN)}")
    for key in ("model", "output", "title"):
        if not raw.get(key):
            raise ValueError(f"{path}: '{key}' is required")
    base = path.parent
    sheet = str(raw.get("sheet", "B")).upper()
    if sheet not in SHEET_SIZES:
        raise ValueError(f"{path}: sheet must be one of {sorted(SHEET_SIZES)}, not {sheet!r}")
    formats = tuple(raw.get("formats", ("png", "svg")))
    if bad := [f for f in formats if f not in FORMATS]:
        raise ValueError(f"{path}: unsupported formats {bad}; use {list(FORMATS)}")
    views = tuple(str(v).upper() for v in raw.get("views") or DEFAULT_VIEWS)
    if bad_views := [v for v in views if v not in ALL_VIEWS]:
        raise ValueError(f"{path}: unknown views {bad_views}; choose from {[v.lower() for v in ALL_VIEWS]}")
    style = raw.get("style") or {}
    toggle_names = {f.name for f in fields(Toggles)}
    if bad_toggles := set(style) - toggle_names:
        raise ValueError(f"{path}: unknown style switches {sorted(bad_toggles)}; expected some of {sorted(toggle_names)}")
    return DrawingConfig(
        path=path, model=(base / raw["model"]).resolve(), output=(base / raw["output"]).resolve(), title=str(raw["title"]),
        drawing_number=str(raw.get("drawing_number", "")), for_=str(raw.get("for", "")), material=str(raw.get("material", "")),
        source=str(raw.get("source", "")), drawn_by=str(raw.get("drawn_by", "")), company=str(raw.get("company", "")),
        front=str(raw.get("front", "+Z")).upper(), up=str(raw.get("up", "+Y")).upper(), sheet=sheet, formats=formats, views=views,
        captions={str(k).upper(): str(v) for k, v in (raw.get("captions") or {}).items()}, notes=[str(n) for n in raw.get("notes") or []],
        toggles=Toggles(**{k: bool(v) for k, v in style.items()}),
    )
