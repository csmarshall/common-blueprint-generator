# common-blueprint-generator

Shared blueprint generator for Charles's 3D-printed-part repos: STEP (STL fallback) in, third-angle ANSI sheet out, one house style, re-rendered by a reusable workflow. Callers: snapmaker-u1-tag-clip (config only), filament-rack-addon-brackets (Python layout on the Sheet/View API).

## Golden rules

- **The house style is decided, not tuned.** It came from an element-by-element review (REV-001, 2026-11-18). Change a value in `src/blueprint/style.py` only when Charles asks, and render a before/after for him rather than describing it.
- **Run under both OCP lines.** cadquery 2.8 pins OCP 7.9 and build123d uses OCP 8. Their APIs differ (`TopoDS.Edge_s` vs `TopoDS.Edge`, `Bnd_Box.Get()` unusable in 8). The package depends on `cadquery-ocp-proxy`, never on a concrete OCP wheel, and CI runs a matrix over both.
- **View convention:** x = up × toward_viewer (the viewer's right hand). `section_loops` uses the same convention. The handedness tests use an asymmetric block defined in `tests/conftest.py`, independent of the code under test. Keep it that way.
- **Nothing typed that can be derived.** Dimensions are measured from projections. Dates come from git (first commit and render date). Config keys live in `config.KEYS`, which drives both validation and `docs/config-reference.md`. Style values live in `style.py`, whose trailing comments become `docs/style-reference.md`.
- **Every drawing states its licence; never default it.** The licence comes from the repo's REUSE.toml for the drawing's own output path (via the `reuse` package, never a home-made glob matcher), or from an explicit `licence:` override. No REUSE.toml or no match is an error naming the path.
- **Generated docs are committed and checked.** After changing config keys or style values, run `uv run python tools/gen_docs.py`. CI fails if they are stale. Gallery images are not committed: the docs site build renders them.
- **Output is byte-stable** (fixed SVG hash salt, no metadata dates), so a re-render that changes nothing commits nothing. Don't add timestamps or random ids.
- **Callers pin `@v1`.** Every merge to main is a patch release that moves `v1`; minor and major are manual (`Publish` workflow dispatch). Breaking changes to `blueprint.yml` keys, the `Sheet`/`View` API or `render.yml` inputs need a migration PR in each caller repo.

## Flow

Issue first. Branch in a worktree at `~/work/claude/common-blueprint-generator-worktrees/<n>-slug`. Open a PR with `Closes #n`; CI (`ci.yml`) is the gate. Get a rev-s review, squash-merge, then the docs site deploys from main.

## Commands

```sh
uv sync --extra ocp --group docs
uv run pytest && uv run ruff check src tests tools && uv run mypy src && uv run python tools/gen_docs.py --check
uv run blueprint render examples/blueprint.yml          # re-renders the README example
uv run python tools/gen_docs.py --gallery && uv run zensical serve
```
