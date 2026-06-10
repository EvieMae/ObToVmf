import struct

from oblivion2vmf.textures import (
    COLORS,
    MaterialLibrary,
    Texturer,
    classify,
    write_vtf,
)


def test_classify_keywords():
    assert classify("Landscape\\Grass01.dds") == "grass"
    assert classify("Landscape\\TerrainHDDirt.dds") == "dirt"
    assert classify("Rocks01") == "rock"
    assert classify("BeachSand") == "sand"
    assert classify("MountainSnow") == "snow"
    assert classify("Landscape\\DirtPathTrail.dds") == "road"   # 'path'/'trail' wins over 'dirt'
    assert classify("") == "default"
    assert classify("SomethingWeird") == "default"


def test_write_vtf_header(tmp_path):
    p = tmp_path / "t.vtf"
    write_vtf(str(p), (10, 20, 30), size=16)
    data = p.read_bytes()
    assert data[:4] == b"VTF\x00"
    assert struct.unpack_from("<II", data, 4) == (7, 2)          # version 7.2
    assert struct.unpack_from("<I", data, 12)[0] == 80           # headerSize
    assert struct.unpack_from("<HH", data, 16) == (16, 16)       # width, height
    assert struct.unpack_from("<I", data, 20)[0] & 0x0100 == 0   # NOMIP cleared (we have mips)
    # highResImageFormat is at offset 52 = BGR888 (3)
    assert struct.unpack_from("<I", data, 52)[0] == 3
    assert struct.unpack_from("<B", data, 56)[0] == 5            # mips 16,8,4,2,1
    # 80-byte header + full BGR888 mip chain (16..1)
    chain = sum(s * s * 3 for s in (16, 8, 4, 2, 1))
    assert len(data) == 80 + chain


def test_write_vtf_nomip_when_disabled(tmp_path):
    p = tmp_path / "t.vtf"
    write_vtf(str(p), (10, 20, 30), size=16, mips=False)
    data = p.read_bytes()
    assert struct.unpack_from("<I", data, 20)[0] == 0x0300       # NOMIP|NOLOD
    assert struct.unpack_from("<B", data, 56)[0] == 1
    assert len(data) == 80 + 16 * 16 * 3


def test_dds_to_vtf_mip_chain():
    import struct as _s
    from oblivion2vmf.dds import dds_to_vtf
    # 8x8 DXT1 with 2 mips: 8x8 (4 blocks*8=32B) + 4x4 (1 block*8=8B)
    hdr = bytearray(128)
    hdr[0:4] = b"DDS "
    _s.pack_into("<I", hdr, 12, 8)            # height
    _s.pack_into("<I", hdr, 16, 8)            # width
    _s.pack_into("<I", hdr, 28, 2)            # dwMipMapCount
    hdr[84:88] = b"DXT1"
    body = bytes(range(32)) + bytes(range(8))
    vtf = dds_to_vtf(bytes(hdr) + body)
    assert _s.unpack_from("<B", vtf, 56)[0] == 2                 # 2 mips copied
    assert _s.unpack_from("<I", vtf, 20)[0] & 0x0100 == 0        # NOMIP cleared
    # smallest mip (4x4=8B) first, then 8x8 (32B)
    assert vtf[80:88] == bytes(range(8))
    assert vtf[88:120] == bytes(range(32))


def test_material_library_paths_and_write(tmp_path):
    lib = MaterialLibrary(prefix="oblivion2vmf")
    assert lib.material_for("grass", None) == "oblivion2vmf/grass"
    assert lib.material_for("grass", "grass") == "oblivion2vmf/grass"   # same -> single
    assert lib.material_for("dirt", "grass") == "oblivion2vmf/blend_dirt_grass"
    stats = lib.write(str(tmp_path / "materials"))
    base = tmp_path / "materials" / "oblivion2vmf"
    assert (base / "tex_grass.vtf").exists()
    assert (base / "tex_dirt.vtf").exists()
    assert (base / "grass.vmt").exists()
    assert (base / "blend_dirt_grass.vmt").exists()
    blend = (base / "blend_dirt_grass.vmt").read_text()
    assert "WorldVertexTransition" in blend
    assert '"$basetexture" "oblivion2vmf/tex_dirt"' in blend     # base = alpha 0
    assert '"$basetexture2" "oblivion2vmf/tex_grass"' in blend   # overlay = alpha 255
    assert stats["textures"] == 2 and stats["blend_materials"] == 1


def test_texturer_tile_single_and_blend():
    ltex = {
        0x10: {"icon": "Landscape\\Grass01.dds", "edid": "Grass"},
        0x20: {"icon": "Landscape\\Dirt01.dds", "edid": "Dirt"},
    }
    # quadrant 0 (SW): base grass, overlay dirt painted full in the NE-most point
    grid = [[0] * 17 for _ in range(17)]
    grid[16][16] = 255
    cell_textures = {(0, 0): {0: {"base": 0x10, "overs": [(0x20, grid)]}}}
    tx = Texturer(ltex, cell_textures)

    # power-4 tile (n=16) at SW (r0=0,c0=0) -> quadrant 0. base=grass, overlay=dirt;
    # canonicalised to blend_dirt_grass (dirt<grass) with alpha INVERTED.
    mat, blend = tx.tile(0, 0, 0, 0, 16)
    assert mat == "oblivion2vmf/blend_dirt_grass"
    assert blend["mode"] == "alpha"
    a = blend["alphas"]
    assert len(a) == 17 and len(a[0]) == 17
    assert a[16][16] == 0 and a[0][0] == 255          # inverted (was 255 / 0)

    # a cell with no texture data -> default single material, no blend
    mat2, blend2 = tx.tile(5, 5, 0, 0, 16)
    assert mat2 == "oblivion2vmf/default"
    assert blend2 is None


def test_texturer_4way_weights():
    ltex = {0x10: {"icon": "grass", "edid": ""}, 0x20: {"icon": "dirt", "edid": ""},
            0x30: {"icon": "rock", "edid": ""}}
    g_dirt = [[0] * 17 for _ in range(17)]
    g_dirt[16][16] = 255
    g_rock = [[0] * 17 for _ in range(17)]
    g_rock[0][0] = 255
    cell_textures = {(0, 0): {0: {"base": 0x10, "overs": [(0x20, g_dirt), (0x30, g_rock)]}}}
    tx = Texturer(ltex, cell_textures, fourway=True)

    mat, blend = tx.tile(0, 0, 0, 0, 16)
    # canonical (sorted) tokens dirt<grass<rock, padded to 4 channels
    assert mat == "oblivion2vmf/blend4_dirt_grass_rock_rock"
    assert blend["mode"] == "multiblend"
    w = blend["weights"]
    assert abs(sum(w[16][16]) - 1.0) < 1e-6                     # weights normalised
    assert w[16][16][0] > 0.99                                  # dirt channel (idx 0)
    assert w[0][0][2] > 0.99                                    # rock channel (idx 2)
    assert w[8][8] == (0.0, 1.0, 0.0, 0.0)                      # unpainted -> base grass (idx 1)


def test_material_library_4way(tmp_path):
    lib = MaterialLibrary()
    assert lib.material_for_4way(("grass", "dirt", "rock", "sand")) == \
        "oblivion2vmf/blend4_grass_dirt_rock_sand"
    stats = lib.write(str(tmp_path / "materials"))
    base = tmp_path / "materials" / "oblivion2vmf"
    vmt = (base / "blend4_grass_dirt_rock_sand.vmt").read_text()
    assert "Lightmapped_4WayBlend" in vmt
    assert '"$basetexture" "oblivion2vmf/tex_grass"' in vmt
    assert '"$basetexture4" "oblivion2vmf/tex_sand"' in vmt
    assert (base / "tex_sand.vtf").exists()
    assert stats["blend4_materials"] == 1
