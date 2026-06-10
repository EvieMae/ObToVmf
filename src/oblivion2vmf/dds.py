"""Transcode Bethesda .dds textures into Source .vtf.

DDS and VTF both store the same raw DXT (S3TC/BCn) blocks, so we copy the
compressed block data with no re-encode -- only the container header differs.
We emit a VTF 7.2 copying the DDS's full mip chain (largest->smallest reordered
to VTF's smallest-first layout); if the DDS has only one level we keep NOMIP.

Only DXT1/DXT3/DXT5 are handled (the vast majority of Oblivion textures);
anything else returns None so the caller can fall back to a placeholder.
"""
from __future__ import annotations

import struct

# fourCC -> (VTF image format enum, bytes per 4x4 block)
_DXT = {b"DXT1": (13, 8), b"DXT3": (14, 16), b"DXT5": (15, 16)}

_NOMIP_NOLOD = 0x0300
_EIGHTBITALPHA = 0x2000


def dds_to_vtf(data):
    """Return VTF bytes for a DXT1/3/5 .dds, or None if unsupported."""
    if len(data) < 128 or data[:4] != b"DDS ":
        return None
    height = struct.unpack_from("<I", data, 12)[0]
    width = struct.unpack_from("<I", data, 16)[0]
    fourcc = data[84:88]
    if fourcc not in _DXT:
        return None
    fmt, block = _DXT[fourcc]
    if width <= 0 or height <= 0:
        return None

    # The DDS may carry a full mip chain (largest->smallest). Copy every level
    # that's actually present so the VTF minifies cleanly at distance.
    dds_mipcount = struct.unpack_from("<I", data, 28)[0] or 1
    levels = []                              # (w, h, dxt_bytes), largest first
    off = 128
    w, hh = width, height
    for _ in range(max(1, dds_mipcount)):
        sz = max(1, (w + 3) // 4) * max(1, (hh + 3) // 4) * block
        if off + sz > len(data):
            break
        levels.append((w, hh, data[off:off + sz]))
        off += sz
        if w == 1 and hh == 1:
            break
        w = max(1, w // 2); hh = max(1, hh // 2)
    if not levels:
        return None
    mipcount = len(levels)

    nomip = _NOMIP_NOLOD if mipcount == 1 else 0
    flags = nomip | (_EIGHTBITALPHA if fmt != 13 else 0)
    h = bytearray(b"VTF\x00")
    h += struct.pack("<II", 7, 2)            # version 7.2
    h += struct.pack("<I", 80)               # headerSize
    h += struct.pack("<HH", width, height)
    h += struct.pack("<I", flags)
    h += struct.pack("<HH", 1, 0)            # frames, firstFrame
    h += b"\x00" * 4                          # padding0
    h += struct.pack("<fff", 0.5, 0.5, 0.5)  # reflectivity
    h += b"\x00" * 4                          # padding1
    h += struct.pack("<f", 1.0)              # bumpmapScale
    h += struct.pack("<I", fmt)              # highResImageFormat (DXT*)
    h += struct.pack("<B", mipcount)         # mipmapCount
    h += struct.pack("<I", 0xFFFFFFFF)       # lowResImageFormat = NONE
    h += struct.pack("<BB", 0, 0)            # lowResWidth/Height
    h += struct.pack("<H", 1)                # depth
    h += b"\x00" * (80 - len(h))
    body = b"".join(blk for _, _, blk in reversed(levels))  # smallest mip first
    return bytes(h) + body
