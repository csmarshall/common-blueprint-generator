"""Licences a drawing can be marked with: an SPDX id in, the title block's text, deed link and icons out.

Creative Commons licences get the official CC icon marks (vendored unmodified in licence_icons/, see NOTICE.md there), drawn in their original
colours with the deed URL beside them, as the CC trademark policy requires. Other licences have no official marks the policy lets us use, so they get
a plain drawn badge with the SPDX id. Non-commercial licences add NOT FOR RESALE unless the config gives its own note.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from functools import cache
from pathlib import Path

from matplotlib.path import Path as MplPath
from matplotlib.transforms import Affine2D
from svgpath2mpl import parse_path

ICON_DIR = Path(__file__).with_name("licence_icons")
SVG_NS = "{http://www.w3.org/2000/svg}"


@dataclass(frozen=True)
class Licence:
    """What the title block says and shows for one licence."""

    spdx: str
    name: str  # as printed, e.g. CC BY-NC-SA 4.0
    url: str  # the deed or licence text
    icons: tuple[str, ...] = ()  # CC icon marks, left to right; empty means a drawn SPDX badge
    noncommercial: bool = False  # true for any id containing -NC-; adds NOT FOR RESALE


def _cc(spdx: str, elements: tuple[str, ...]) -> Licence:
    """A Creative Commons 4.0 licence from its elements, e.g. ("by", "nc", "sa")."""
    code = "-".join(elements)
    return Licence(spdx, f"CC {code.upper()} 4.0", f"https://creativecommons.org/licenses/{code}/4.0/", ("cc", *elements), "nc" in elements)


LICENCES: dict[str, Licence] = {
    lic.spdx: lic
    for lic in (
        _cc("CC-BY-4.0", ("by",)),
        _cc("CC-BY-SA-4.0", ("by", "sa")),
        _cc("CC-BY-ND-4.0", ("by", "nd")),
        _cc("CC-BY-NC-4.0", ("by", "nc")),
        _cc("CC-BY-NC-SA-4.0", ("by", "nc", "sa")),
        _cc("CC-BY-NC-ND-4.0", ("by", "nc", "nd")),
        Licence("CC0-1.0", "CC0 1.0", "https://creativecommons.org/publicdomain/zero/1.0/", ("cc", "zero")),
        Licence("MIT", "MIT", "https://opensource.org/license/mit"),
        Licence("Apache-2.0", "Apache 2.0", "https://www.apache.org/licenses/LICENSE-2.0"),
        Licence("BSD-3-Clause", "BSD 3-Clause", "https://opensource.org/license/bsd-3-clause"),
        Licence("GPL-3.0-or-later", "GPL 3.0 or later", "https://www.gnu.org/licenses/gpl-3.0.html"),
        Licence("CERN-OHL-S-2.0", "CERN OHL-S 2.0", "https://ohwr.org/cern_ohl_s_v2.txt"),
    )
}
NONCOMMERCIAL_NOTE = "NOT FOR RESALE"


def lookup(spdx: str) -> Licence:
    """The licence for an SPDX id or expression (case-insensitive).

    Ids in LICENCES get their label, link and icons. Anything else (another SPDX id, or an expression such as "MIT OR Apache-2.0") prints as given,
    with a badge and a link to the SPDX list, rather than failing. Any id containing -NC- is non-commercial.
    """
    spdx = spdx.strip()
    for key, lic in LICENCES.items():
        if key.lower() == spdx.lower():
            return lic
    single = " " not in spdx
    url = f"https://spdx.org/licenses/{spdx}.html" if single else "https://spdx.org/licenses/"
    return Licence(spdx, spdx, url, (), "-NC-" in spdx.upper())


@dataclass(frozen=True)
class Icon:
    """One CC icon as matplotlib geometry in a unit box (0..1 both ways, y up): its white disc and its black glyph."""

    disc_centre: tuple[float, float]
    disc_radius: float
    glyph: MplPath


@cache
def icon(name: str) -> Icon:
    """Load licence_icons/<name>.svg into unit-box geometry, flipping SVG's y-down to y-up."""
    root = ET.parse(ICON_DIR / f"{name}.svg").getroot()  # vendored package files only, never user input (expat fetches no external entities)
    vx, vy, vw, vh = (float(v) for v in root.get("viewBox", "").split())
    to_unit = Affine2D().translate(-vx, -vy).scale(1 / vw, -1 / vh).translate(0, 1)
    circle = next(root.iter(f"{SVG_NS}circle"))
    cx, cy = to_unit.transform((float(circle.get("cx", 0)), float(circle.get("cy", 0))))
    r = float(circle.get("r", 0)) / vw
    paths = [parse_path(p.get("d", "")) for p in root.iter(f"{SVG_NS}path")]
    glyph = MplPath.make_compound_path(*paths).transformed(to_unit)
    return Icon((float(cx), float(cy)), r, glyph)
