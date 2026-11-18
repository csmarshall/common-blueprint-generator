"""Test fixtures: small solids built directly in OCP, so the tests need no CAD library and know the true geometry independently of the code under test."""

from __future__ import annotations

from pathlib import Path

import pytest
from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeCylinder
from OCP.gp import gp_Ax2, gp_Dir, gp_Pnt
from OCP.IFSelect import IFSelect_ReturnStatus
from OCP.STEPControl import STEPControl_AsIs, STEPControl_Writer
from OCP.StlAPI import StlAPI_Writer
from OCP.TopoDS import TopoDS_Shape

# The notched block: 40 wide (X), 30 tall (Y), 20 deep (Z), with a 10 x 10 notch cut out of the top-right corner as seen from the front (+Z),
# running the full depth, and a 6 mm through-hole along Z centred at (12, 12). Nothing about it is symmetric, so a mirrored or rotated view is visible.
BLOCK = (40.0, 30.0, 20.0)
NOTCH = 10.0
HOLE_CENTRE = (12.0, 12.0)
HOLE_R = 3.0


def notched_block() -> TopoDS_Shape:
    """The asymmetric test solid described above."""
    w, h, d = BLOCK
    block = BRepPrimAPI_MakeBox(w, h, d).Shape()
    notch = BRepPrimAPI_MakeBox(gp_Pnt(w - NOTCH, h - NOTCH, -1), NOTCH + 1, NOTCH + 1, d + 2).Shape()
    hole = BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(*HOLE_CENTRE, -1), gp_Dir(0, 0, 1)), HOLE_R, d + 2).Shape()
    return BRepAlgoAPI_Cut(BRepAlgoAPI_Cut(block, notch).Shape(), hole).Shape()


@pytest.fixture
def block() -> TopoDS_Shape:
    return notched_block()


@pytest.fixture
def block_step(tmp_path: Path) -> Path:
    path = tmp_path / "block.step"
    writer = STEPControl_Writer()
    writer.Transfer(notched_block(), STEPControl_AsIs)
    assert writer.Write(str(path)) == IFSelect_ReturnStatus.IFSelect_RetDone
    return path


@pytest.fixture
def block_stl(tmp_path: Path) -> Path:
    path = tmp_path / "block.stl"
    shape = notched_block()
    BRepMesh_IncrementalMesh(shape, 0.05)
    assert StlAPI_Writer().Write(shape, str(path))
    return path
