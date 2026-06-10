"""Decode Oblivion LAND record terrain data (the VHGT heightmap).

Reference: UESP "Oblivion Mod:Mod File Format/LAND".
"""
from __future__ import annotations

import struct

GRID = 33                 # vertices per cell edge (33 x 33 = 1089 vertices)
INTERVALS = GRID - 1      # 32 spacings between vertices
VERTEX_SPACING = 128      # game units between adjacent vertices (4096 / 32)
CELL_SIZE = 4096          # game units per exterior cell edge
HEIGHT_SCALE = 8.0        # each accumulated gradient unit = 8 game units

_VHGT_GRAD = "<%db" % (GRID * GRID)   # 1089 signed int8 gradient bytes


def decode_vhgt(data: bytes) -> list:
    """Decode a VHGT subrecord into a 33x33 grid of absolute heights (game units).

    Binary layout: float32 ``offset`` + 33*33 signed int8 gradient bytes + 3 pad.

    Reconstruction: the first byte of each row is the delta from the previous
    row's start; each subsequent byte in a row is the delta from the previous
    vertex. The running sum is seeded with ``offset`` and the final value is
    multiplied by 8 (HEIGHT_SCALE).

    Returned ``grid[y][x]``: ``y`` = row index, increasing NORTH (+Y); ``x`` =
    column index, increasing EAST (+X). ``grid[0][0]`` is the cell's SW corner.
    """
    need = 4 + GRID * GRID
    if len(data) < need:
        raise ValueError("VHGT too short: %d bytes (need >= %d)" % (len(data), need))
    offset = struct.unpack_from("<f", data, 0)[0]
    grad = struct.unpack_from(_VHGT_GRAD, data, 4)

    grid = [[0.0] * GRID for _ in range(GRID)]
    row_start = offset
    k = 0
    for y in range(GRID):
        row_start += grad[k]            # grad[y][0]: row-to-row delta
        k += 1
        cur = row_start
        grid[y][0] = cur * HEIGHT_SCALE
        for x in range(1, GRID):
            cur += grad[k]              # grad[y][x]: across-row delta
            k += 1
            grid[y][x] = cur * HEIGHT_SCALE
    return grid


def encode_vhgt(grid: list) -> bytes:
    """Inverse of :func:`decode_vhgt`. Used by tests to round-trip synthetic data.

    Assumes every height is a multiple of HEIGHT_SCALE and that the resulting
    per-vertex gradients fit in a signed byte (-128..127).
    """
    h = [[int(round(grid[y][x] / HEIGHT_SCALE)) for x in range(GRID)] for y in range(GRID)]
    grad = [[0] * GRID for _ in range(GRID)]
    # offset is fixed at 0, so grad[0][0] carries the SW corner directly.
    grad[0][0] = h[0][0]
    for y in range(1, GRID):
        grad[y][0] = h[y][0] - h[y - 1][0]
    for y in range(GRID):
        for x in range(1, GRID):
            grad[y][x] = h[y][x] - h[y][x - 1]

    flat = [grad[y][x] for y in range(GRID) for x in range(GRID)]
    for v in flat:
        if not -128 <= v <= 127:
            raise ValueError("gradient %d out of int8 range; heights too steep" % v)
    return struct.pack("<f", 0.0) + struct.pack(_VHGT_GRAD, *flat) + b"\x00\x00\x00"
