import struct

from oblivion2vmf.dds import dds_to_vtf
from oblivion2vmf.model import tex_slug


def build_dds(w, h, fourcc, block_data):
    hdr = bytearray(128)
    hdr[0:4] = b"DDS "
    struct.pack_into("<I", hdr, 4, 124)     # dwSize
    struct.pack_into("<I", hdr, 12, h)
    struct.pack_into("<I", hdr, 16, w)
    hdr[84:88] = fourcc
    return bytes(hdr) + block_data


def test_dxt1_transcode():
    block = bytes(range(8))                  # one 4x4 DXT1 block
    vtf = dds_to_vtf(build_dds(4, 4, b"DXT1", block))
    assert vtf is not None
    assert vtf[:4] == b"VTF\x00"
    assert struct.unpack_from("<II", vtf, 4) == (7, 2)
    assert struct.unpack_from("<HH", vtf, 16) == (4, 4)         # width, height
    assert struct.unpack_from("<I", vtf, 52)[0] == 13           # DXT1
    assert vtf[80:] == block                                    # DXT blocks copied verbatim
    assert len(vtf) == 80 + 8


def test_dxt5_transcode_size():
    # 8x8 DXT5 = (8/4)*(8/4)=4 blocks * 16 bytes = 64 bytes
    blocks = bytes(64)
    vtf = dds_to_vtf(build_dds(8, 8, b"DXT5", blocks))
    assert struct.unpack_from("<I", vtf, 52)[0] == 15           # DXT5
    assert len(vtf) == 80 + 64
    flags = struct.unpack_from("<I", vtf, 20)[0]
    assert flags & 0x2000                                       # EIGHTBITALPHA for DXT5


def test_unsupported_format_returns_none():
    assert dds_to_vtf(b"not a dds") is None
    assert dds_to_vtf(build_dds(4, 4, b"RGBA", bytes(64))) is None


def test_tex_slug():
    assert tex_slug("textures\\clutter\\lowerclass\\Barrel01.dds") == "t_clutter_lowerclass_barrel01"
    assert tex_slug("Architecture\\Castle\\Wall.DDS") == "t_architecture_castle_wall"
    assert tex_slug("") == "t_default"
