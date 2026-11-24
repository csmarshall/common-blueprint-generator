"""Third-angle blueprint sheets of 3D-printed parts, drawn by hidden-line projection of a STEP (or STL) file.

Config-only repos use the `blueprint render` CLI; repos that need hand-placed views, sections or callouts build a Sheet and Views directly, and
preview() draws sheet-free README header images.
"""

from .geometry import Projection, beyond, load_shape, project, section_loops, section_view, sight_direction
from .layout import auto_sheet, view_directions
from .preview import PreviewLayout, as_shape, preview
from .revisions import git_dates, revisions
from .sheet import Canvas, CentreLine, Revision, Sheet, TitleInfo, View
from .style import Style, Toggles, fmt, scale_label

__all__ = [
    "Canvas",
    "CentreLine",
    "PreviewLayout",
    "Projection",
    "Revision",
    "Sheet",
    "Style",
    "TitleInfo",
    "Toggles",
    "View",
    "as_shape",
    "auto_sheet",
    "beyond",
    "fmt",
    "git_dates",
    "load_shape",
    "preview",
    "project",
    "revisions",
    "scale_label",
    "section_loops",
    "section_view",
    "sight_direction",
    "view_directions",
]
