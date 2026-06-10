import struct

from oblivion2vmf.land import GRID, HEIGHT_SCALE, decode_vhgt, encode_vhgt


def _flat_grid(value):
    return [[float(value)] * GRID for _ in range(GRID)]


def test_decode_flat():
    # offset 0, all gradients 0 -> all heights 0.
    data = struct.pack("<f", 0.0) + bytes(GRID * GRID) + b"\x00\x00\x00"
    grid = decode_vhgt(data)
    assert len(grid) == GRID and len(grid[0]) == GRID
    assert all(h == 0.0 for row in grid for h in row)


def test_decode_corner_and_slope():
    # offset 0; grad[0][0] = 5 -> SW corner = 5 * 8 = 40.
    grad = [0] * (GRID * GRID)
    grad[0] = 5            # corner
    grad[1] = 1            # +1 step east from corner -> 6*8 = 48
    grad[GRID] = 2         # +2 step north on row 1 start -> 7*8 = 56
    data = struct.pack("<f", 0.0) + struct.pack("<%db" % (GRID * GRID), *grad) + b"\x00\x00\x00"
    grid = decode_vhgt(data)
    assert grid[0][0] == 5 * HEIGHT_SCALE
    assert grid[0][1] == 6 * HEIGHT_SCALE
    assert grid[1][0] == 7 * HEIGHT_SCALE


def test_roundtrip_ramp():
    # A north-east ramp; every height a multiple of 8 and within int8 deltas.
    grid = [[float((x + y) * HEIGHT_SCALE) for x in range(GRID)] for y in range(GRID)]
    data = encode_vhgt(grid)
    assert decode_vhgt(data) == grid


def test_roundtrip_flat_offsetish():
    grid = _flat_grid(96)  # 12 * 8
    assert decode_vhgt(encode_vhgt(grid)) == grid
