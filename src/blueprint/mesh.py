"""STL reading for the guard and the mesh fast path: triangle counts without parsing, and triangles as a numpy array."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

BINARY_HEADER_BYTES = 84  # 80 header + uint32 triangle count
BINARY_TRIANGLE_BYTES = 50  # normal, three vertices (12 floats) + uint16 attribute
_VERTEX = re.compile(rb"vertex\s+(\S+)\s+(\S+)\s+(\S+)")


def _is_binary(data: bytes | memoryview) -> bool:
    """A binary STL's size is exactly 84 + 50 * its declared count; an ASCII STL practically never matches that."""
    if len(data) < BINARY_HEADER_BYTES:
        return False
    count = int(np.frombuffer(data[80:84], dtype="<u4")[0])
    return len(data) == BINARY_HEADER_BYTES + BINARY_TRIANGLE_BYTES * count


def stl_triangle_count(path: Path) -> int:
    """Number of triangles in an STL, read from the header (binary) or by counting facets (ASCII), without parsing the vertices."""
    with path.open("rb") as handle:
        head = handle.read(BINARY_HEADER_BYTES)
        size = path.stat().st_size
        if len(head) == BINARY_HEADER_BYTES:
            count = int(np.frombuffer(head[80:84], dtype="<u4")[0])
            if size == BINARY_HEADER_BYTES + BINARY_TRIANGLE_BYTES * count:
                return count
        return path.read_bytes().count(b"endfacet")


def read_stl(path: Path) -> np.ndarray:
    """The triangles of an STL (binary or ASCII) as an (n, 3, 3) float64 array of vertex coordinates."""
    data = path.read_bytes()
    if _is_binary(data):
        count = int(np.frombuffer(data[80:84], dtype="<u4")[0])
        record = np.dtype([("normal", "<f4", 3), ("v", "<f4", (3, 3)), ("attr", "<u2")])
        return np.frombuffer(data, dtype=record, count=count, offset=BINARY_HEADER_BYTES)["v"].astype(np.float64)
    found = _VERTEX.findall(data)
    if not found or len(found) % 3:
        raise ValueError(f"could not read STL file {path}: no triangles found")
    return np.array(found, dtype=np.float64).reshape(-1, 3, 3)
