"""`blueprint render blueprint.yml [...]`: draw the automatic sheet for each config.

Log level comes from --log-level or the BLUEPRINT_LOG_LEVEL environment variable.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import replace
from datetime import date
from pathlib import Path

from .config import DrawingConfig, load
from .geometry import load_shape
from .layout import auto_sheet
from .revisions import GitDates, git_dates, revisions
from .sheet import Sheet, TitleInfo
from .style import Style

LOG_FORMAT = "%(asctime)s %(levelname)-5s %(name)s: %(message)s"
log = logging.getLogger("blueprint")


def title_info(cfg: DrawingConfig, rendered: date, dates: GitDates) -> TitleInfo:
    """Title-block content: the config's words plus git-derived dates and revisions."""
    return TitleInfo(
        title=cfg.title,
        drawing_number=cfg.drawing_number,
        for_=cfg.for_,
        material=cfg.material,
        source=cfg.source,
        drawn_by=cfg.drawn_by,
        company=cfg.company,
        started=dates.started,
        updated=rendered,
        revisions=revisions(dates, rendered),
    )


def render(cfg: DrawingConfig, dates: GitDates, rendered: date | None = None) -> list[Path]:
    """Draw one automatic sheet and write it in every configured format. `dates` is read before any sheet is written (see main)."""
    rendered = rendered or date.today()
    sheet = Sheet(cfg.sheet, replace(Style(), toggles=cfg.toggles))
    layout = auto_sheet(sheet, load_shape(cfg.model), cfg.front, cfg.up, cfg.captions, cfg.views, cfg.isometric)
    sheet.title_block(title_info(cfg, rendered, dates), layout.scale)
    sheet.notes(cfg.notes)
    return sheet.save(cfg.output, cfg.formats)


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    ap = argparse.ArgumentParser(prog="blueprint", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    r = sub.add_parser("render", help="draw the sheet for each blueprint.yml")
    r.add_argument("configs", nargs="+", type=Path)
    ap.add_argument("--log-level", default=os.environ.get("BLUEPRINT_LOG_LEVEL", "INFO"), choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = ap.parse_args(argv)
    logging.basicConfig(level=args.log_level, format=LOG_FORMAT)
    failed = 0
    # Read git for every config first: writing one sheet dirties the checkout, which would mark every later sheet "+ LOCAL EDITS".
    dates = {path: git_dates(path.resolve().parent) for path in args.configs}
    for path in args.configs:
        try:
            for out in render(load(path), dates[path]):
                print(out)
        except Exception:
            log.exception("failed to render %s", path)
            failed += 1
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
