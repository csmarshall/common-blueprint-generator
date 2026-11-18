# common-blueprint-generator

<p>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
  <a href="https://buymeacoffee.com/cs_marshall"><img src="https://img.shields.io/badge/Buy%20Me%20a%20Coffee-ffdd00?logo=buy-me-a-coffee&logoColor=black" alt="Buy Me a Coffee"></a>
</p>

Old-school blueprint sheets for 3D-printed parts, drawn straight from the CAD model and kept current by CI.

Point it at a STEP file and it draws a third-angle drawing on an ANSI B sheet: hidden-line projections of the real B-rep, overall dimensions measured from the geometry, a full title block, a revision block that names the commit it was rendered from, and a stamp. One house style is shared by every repo that uses it, and a reusable GitHub workflow re-renders the sheet whenever the model changes, so the drawing in a README never goes stale.

<!-- blueprint:block -->
[![TEST BLOCK](examples/block.svg)](examples/block.svg)

<sub>Drawing generated from `tests/conftest.py (notched_block)` by [common-blueprint-generator](https://github.com/csmarshall/common-blueprint-generator) on every change to the model · [Buy me a coffee](https://buymeacoffee.com/cs_marshall)</sub>
<!-- /blueprint -->

**Docs:** [csmarshall.github.io/common-blueprint-generator](https://csmarshall.github.io/common-blueprint-generator/): gallery, [blueprint.yml reference](docs/config-reference.md), [house style reference](docs/style-reference.md) and the Python API. All of it is generated from the code, and CI fails if a committed reference is out of date.

## Use it in a repo

1. **Export a STEP file** of the part you want drawn. In build123d that's `export_step(part, "build/part.step")`; in cadquery it's `cq.exporters.export(part, "build/part.step")`. An STL also works, but curves then draw as facets.
2. **Add `blueprint.yml`** next to it:

   ```yaml
   model: build/part.step
   output: docs/images/part        # writes part.svg and part.png
   title: MY PART
   drawing_number: MP-001
   material: PETG
   drawn_by: C. MARSHALL
   company: Shermer Works          # goes on the approved stamp
   readme: README.md               # keep the drawing embedded here (optional)
   ```

3. **Mark where it goes in the README**, once:

   ```markdown
   <!-- blueprint:part -->
   <!-- /blueprint -->
   ```

   Each render rewrites what's between the markers: the scalable SVG linked to itself at full size, and a one-line credit linking back here.
4. **Call the reusable workflow** from `.github/workflows/blueprint.yml`:

   ```yaml
   on:
     push:
       branches: [main]
       paths: ["src/**", "blueprint.yml"]   # whatever the model is built from
     workflow_dispatch:
   permissions:
     contents: write
   jobs:
     blueprint:
       uses: csmarshall/common-blueprint-generator/.github/workflows/render.yml@main
       with:
         export-command: uv run --locked python tools/export_step.py
   ```

   It checks out full history (STARTED is the first commit's date), runs your export, renders, and commits the sheet and README back. Commits made with the default token don't trigger workflows, so it can't loop.

Run it locally with `uvx --from "common-blueprint-generator[ocp] @ git+https://github.com/csmarshall/common-blueprint-generator" blueprint render blueprint.yml`.

### Choosing views

The default is front, top and right in their third-angle positions plus an isometric, at the largest standard scale that fits. `views:` takes any subset of `front, top, bottom, left, right, back, isometric`. A single view, e.g. `views: [top]`, is centred and drawn as large as a standard scale allows. `isometric: 1.6` draws the isometric at 1.6× the main scale; `isometric: fill` fills the space beside the views. `front:` and `up:` name the model axes facing the viewer and pointing up (default `+Z` and `+Y`).

### The licence on the drawing

Every sheet states its licence in the title block, and the generator never guesses it. The licence and copyright are read from the repo's [REUSE.toml](https://reuse.software/spec-3.3/) for the drawing's own output file. That matters in a split-licence repo: the drawing gets the drawings' licence (say CC BY-NC-SA 4.0), not the MIT that covers the script that drew it. The [`reuse`](https://reuse.software/) package does the lookup, so REUSE's own precedence rules decide which annotation wins. A repo with no `REUSE.toml`, or with no annotation covering the drawing, fails the render with an error naming the path. `licence:` in `blueprint.yml` is the explicit override.

Creative Commons licences get the official CC icons (unmodified, per the [CC trademark policy](https://creativecommons.org/policies/), with the deed URL beside them and linked in the SVG). Other licences get a drawn SPDX badge, and an id the generator doesn't know prints exactly as given. Any id containing `-NC-` adds NOT FOR RESALE.

A part repo's `REUSE.toml` for the usual split looks like this (the second annotation overrides the first for drawings, models and print files):

```toml
version = 1

[[annotations]]
path = "**"
SPDX-FileCopyrightText = "2026 Charles Marshall"
SPDX-License-Identifier = "MIT"

[[annotations]]
path = ["**/*.svg", "**/*.step", "**/*.stl", "**/*.3mf", "prints/**", "docs/images/**"]
precedence = "override"
SPDX-FileCopyrightText = "2026 Charles Marshall"
SPDX-License-Identifier = "CC-BY-NC-SA-4.0"
```

The workflow installs `reuse` (the `[reuse]` extra; it's GPL-3.0-or-later, so it is optional for this MIT package). Locally: `pip install "common-blueprint-generator[ocp,reuse] @ git+..."`.

### Releases

`release.yml` is a second reusable workflow, run after the render: every merge to main is a patch release (v1.4.2 to v1.4.3), and a minor or major bump is a deliberate manual run with `bump: minor` or `bump: major`. Each release carries `<repo>-source-vX.Y.Z.zip` and, with `prints-paths`, `<repo>-prints-vX.Y.Z.zip`: the print files plus `LICENSES/`, `REUSE.toml` and the README. This repo releases itself the same way and moves the `v1` tag, so callers can pin `@v1`.

### Dates and revisions

Nothing is typed by hand. STARTED is the date of the repo's first commit and UPDATED is the render date. The revision block has an INITIAL RELEASE row on the first-commit date and a CUR row reading `RENDERED FROM <sha>`, marked `+ LOCAL EDITS` if the checkout was dirty.

### Drawings that need more than the automatic layout

Section views, chain dimensions and callouts can't be chosen automatically, so lay those out in Python on the same `Sheet` and `View` API the automatic layout uses. The [filament-rack-addon-brackets](https://github.com/csmarshall/filament-rack-addon-brackets) bracket is the worked example:

```python
from blueprint import Sheet, TitleInfo, View, fmt, git_dates, project, revisions, section_loops

sheet = Sheet("B")
front = View(sheet, project(shape, (1, 0, 0), (0, 0, 1)), (34, 150), 0.75).draw()
for a, b in zip(stations, stations[1:]):
    front.dim_h(a, b, 0, -9)                      # chain dimensions along the sockets
front.leader((x, z), (104, 186), f"8x {fmt(p.bore)} dia")
...
sheet.title_block(TitleInfo(title="SHARED-WALL BRACKET", ...), 0.75, detail_scales=(1.5,))
sheet.save(Path("blueprint"))
```

In the workflow, pass `render-command` instead of `configs`. `$BLUEPRINT_PACKAGE` holds the pip requirement for the generator at the chosen ref.

### Dependencies

Only OCP, matplotlib, numpy, scipy and PyYAML. The package depends on `cadquery-ocp-proxy`, so it uses whichever OCP the caller already has: cadquery 2.8 pins OCP 7.9, build123d 0.13 uses OCP 8, and CI tests both. Install the `ocp` extra when nothing else provides it.

## The house style

The style was settled by reviewing two earlier hand-built sheets element by element: one aged and hand-drafted, one clean. The result is clean drafting on blue paper (crisp lines, monospace lettering, a second ink for dimensions, descriptive captions) on aged paper, with a few straight-faced details: an APPROVED FOR PRINTING stamp, "THIRD ANGLE PROJECTION (THE AMERICAN ONE)", "OR WHATEVER THE PRINTER GIVES" under the tolerances, "↑ THIS WAY UP" on the centring mark, and "DRAWN PER ASME Y14.5 (MOSTLY)". `style:` switches the paper ageing, grid, stamp, winks and credit line on or off per drawing. Every value is in the [style reference](docs/style-reference.md).

## Repos with 3D models

An inventory of [csmarshall](https://github.com/csmarshall)'s repositories with model files (2026-11-18), as candidate users:

| Repo | STEP | STL | 3MF | CAD source | Status |
|---|---:|---:|---:|---|---|
| snapmaker-u1-tag-clip | 2 | 2 | 23 | build123d | migrating: config only |
| filament-rack-addon-brackets | 4 | 4 | 2 | cadquery | migrating: Python layout (section A-A) |
| LD2410C-usb-adapter | 23 | 41 | 0 | build123d, OpenSCAD | candidate: has its own `tools/case_drawing.py` |
| gen-o-lantern | 0 | 15 | 0 | none | candidate (STL fallback) |
| u1_cowl_eyes | 0 | 2 | 2 | none | candidate (STL fallback) |
| big-beads | 0 | 0 | 264 | none | 3MF only, not supported |
| SnapmakerOrca-AI-FullColor, blink1, gods-eye-view, SnapCon | | | | | forks: test models, not candidates |

## Development

```sh
uv sync --extra ocp --group docs
uv run pytest                       # projection, layout, config, git dates, README embed, byte-stable output
uv run ruff check src tests tools && uv run mypy src
uv run python tools/gen_docs.py     # regenerate the references after changing config keys or style values
uv run python tools/gen_docs.py --gallery && uv run zensical serve   # docs site, built with Zensical
```

The tests check handedness against an asymmetric block whose geometry is defined in the test fixtures, not taken from the code under test. A mirrored projection or a view placed on the wrong side fails them. They can't catch a model that was already mirrored before export; that has to be checked against the physical part.

Human-directed and AI-assisted: the design decisions and the house style are the author's, made by reviewing every element. The code was written with Claude and reviewed before merge.

## License

MIT. See [LICENSE](LICENSE).
