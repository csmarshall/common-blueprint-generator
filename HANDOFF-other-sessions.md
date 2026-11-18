Heads-up from another Claude session: we're replacing this repo's blueprint script with a shared generator, `common-blueprint-generator` (a separate repo; it isn't public yet). It draws third-angle ANSI B sheets from a STEP export in one approved house style: clean blue drafting with aged paper, the Shermer Works stamp and the winks. Dates come from git: STARTED is the first commit, UPDATED is the render date, plus a "RENDERED FROM <sha>" revision row. A reusable GitHub workflow will re-render the sheet whenever the model changes and commit the PNG and SVG back.

That session will open a PR here (you don't need to do the migration). The PR will:
- add a STEP export of the drawn part
- add the blueprint config (snapmaker-u1-tag-clip: `blueprint.yml`) or a thin script on the shared Python API (filament-rack-addon-brackets: `shared-wall-bracket/make_blueprint.py` keeps its section A-A, chain dims and callouts)
- add a workflow that calls the shared one
- delete the old drawing code (tag-clip: `tools/render_blueprint.py` and `tools/fonts/`)

Until it lands, please:
- don't edit the blueprint scripts or the committed blueprint images (tag-clip: `tools/render_blueprint.py`, `docs/images/blueprint_dock.png`; brackets: `shared-wall-bracket/make_blueprint.py`, `blueprint.png`, `blueprint.svg`)
- keep the model entry points stable, because the export calls them: tag-clip `u1_parts.dock_fingers_parts(None)["dock"]`; brackets `shared_wall_bracket.Params` and `build_bracket(p)`, plus the Params fields the drawing reads (length, height, thickness, rail_height, leg_length, bar_y, bar_z, bore, bar_depth, chamfer, corner_radius, plan_round, upright_from_end, upright_depth_leg, upright_depth_flat, web, divider, upright_vent_d, tube_od)
- if you need to change any of these, note it in your session-state so the PR can be rebased onto it.
