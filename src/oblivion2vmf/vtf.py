"""Minimal VTF -> RGB decoder for the 3D viewport.

VTK can't read .vtf, so to show real model textures in the hull editor we decode
the largest mip of a VTF back to an (H, W, 3) uint8 array. Handles the formats
this tool itself emits: BGR888 (enum 3, from textures.write_vtf*) and DXT1/3/5
(enums 13/14/15, from dds.dds_to_vtf — block data copied straight from the DDS).
Alpha is ignored (collision authoring only needs the colour). Returns None for
anything unsupported so the caller falls back to a flat colour.
"""
from __future__ import annotations

import struct

import numpy as np


def _rgb565(c):
    r = (c >> 11) & 0x1F
    g = (c >> 5) & 0x3F
    b = c & 0x1F
    return ((r << 3) | (r >> 2), (g << 2) | (g >> 4), (b << 3) | (b >> 2))


def _decode_dxt(raw, w, h, color_off, block_bytes):
    """Decode the DXT *colour* sub-block (DXT1 = whole 8-byte block; DXT3/5 =
    bytes 8..16 after the alpha block). Returns (h, w, 3) uint8."""
    out = np.zeros((h, w, 3), np.uint8)
    bw = max(1, (w + 3) // 4)
    bh = max(1, (h + 3) // 4)
    pos = 0
    for by in range(bh):
        for bx in range(bw):
            blk = raw[pos:pos + block_bytes]
            pos += block_bytes
            if len(blk) < color_off + 8:
                continue
            o = color_off
            c0, c1 = struct.unpack_from("<HH", blk, o)
            bits = struct.unpack_from("<I", blk, o + 4)[0]
            e0, e1 = _rgb565(c0), _rgb565(c1)
            if c0 > c1:
                c2 = tuple((2 * e0[i] + e1[i]) // 3 for i in range(3))
                c3 = tuple((e0[i] + 2 * e1[i]) // 3 for i in range(3))
            else:
                c2 = tuple((e0[i] + e1[i]) // 2 for i in range(3))
                c3 = (0, 0, 0)
            pal = (e0, e1, c2, c3)
            for py in range(4):
                yy = by * 4 + py
                if yy >= h:
                    break
                for px in range(4):
                    xx = bx * 4 + px
                    if xx >= w:
                        continue
                    idx = (bits >> (2 * (py * 4 + px))) & 0x3
                    out[yy, xx] = pal[idx]
    return out


def vtf_to_rgb(path):
    """Decode a VTF file's largest mip to an (H, W, 3) uint8 RGB array, or None."""
    try:
        with open(path, "rb") as f:
            data = f.read()
    except OSError:
        return None
    if len(data) < 80 or data[:4] != b"VTF\x00":
        return None
    header_size = struct.unpack_from("<I", data, 12)[0]
    width, height = struct.unpack_from("<HH", data, 16)
    fmt = struct.unpack_from("<I", data, 52)[0]
    if width <= 0 or height <= 0:
        return None

    # The largest mip is stored LAST in the file (VTF is smallest-mip-first, one
    # frame/face here), so slice it off the tail by its computed size.
    if fmt == 3:                                       # BGR888
        size = width * height * 3
        if len(data) < header_size + size:
            return None
        raw = data[len(data) - size:]
        arr = np.frombuffer(raw, np.uint8).reshape(height, width, 3)
        return np.ascontiguousarray(arr[:, :, ::-1])   # BGR -> RGB
    if fmt in (13, 14, 15):                             # DXT1 / DXT3 / DXT5
        block = 8 if fmt == 13 else 16
        color_off = 0 if fmt == 13 else 8              # DXT3/5: skip 8 alpha bytes
        size = max(1, (width + 3) // 4) * max(1, (height + 3) // 4) * block
        if len(data) < header_size + size:
            return None
        raw = data[len(data) - size:]
        return _decode_dxt(raw, width, height, color_off, block)
    return None
