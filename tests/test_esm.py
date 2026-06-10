"""Parse a hand-built synthetic TES4 plugin (no real game data needed)."""
import struct
import zlib

from oblivion2vmf.esm import COMPRESSED, TerrainExtractor
from oblivion2vmf.land import GRID, HEIGHT_SCALE, encode_vhgt


def rec(sig, data, flags=0, formid=0):
    if flags & COMPRESSED:
        body = struct.pack("<I", len(data)) + zlib.compress(data)
    else:
        body = data
    return sig + struct.pack("<IIII", len(body), flags, formid, 0) + body


def grup(label, gtype, body):
    assert len(label) == 4
    size = 20 + len(body)
    return b"GRUP" + struct.pack("<I", size) + label + struct.pack("<i", gtype) + struct.pack("<I", 0) + body


def sub(sig, data):
    return sig + struct.pack("<H", len(data)) + data


def fid_bytes(formid):
    return struct.pack("<I", formid)


def btxt(formid, quad):
    return sub(b"BTXT", struct.pack("<I", formid) + bytes([quad]) + b"\x00\xff\xff")


def atxt(formid, quad, layer=0):
    return sub(b"ATXT", struct.pack("<I", formid) + bytes([quad, 0]) + struct.pack("<H", layer))


def vtxt(points):
    """points: list of (position, opacity-float)."""
    body = b"".join(struct.pack("<HBBf", pos, 0, 0, op) for pos, op in points)
    return sub(b"VTXT", body)


def make_cell(cx, cy, formid, grid, compress_land=False, land_extra=b"", refrs=b""):
    cell = rec(b"CELL", sub(b"XCLC", struct.pack("<ii", cx, cy)), formid=formid)
    land_flags = COMPRESSED if compress_land else 0
    land = rec(b"LAND", sub(b"VHGT", encode_vhgt(grid)) + land_extra,
               flags=land_flags, formid=formid + 1)
    temp = grup(fid_bytes(formid), 9, land + refrs)    # Cell Temporary Children
    children = grup(fid_bytes(formid), 6, temp)        # Cell Children
    return cell + children


def refr(base_formid, pos, rot=(0.0, 0.0, 0.0), scale=None, formid=0):
    data = sub(b"NAME", struct.pack("<I", base_formid))
    data += sub(b"DATA", struct.pack("<6f", *pos, *rot))
    if scale is not None:
        data += sub(b"XSCL", struct.pack("<f", scale))
    return rec(b"REFR", data, formid=formid)


def stat_group(entries):
    """entries: list of (formid, modl_path)."""
    body = b""
    for formid, modl in entries:
        body += rec(b"STAT",
                    sub(b"EDID", b"S\x00") + sub(b"MODL", modl.encode() + b"\x00"),
                    formid=formid)
    return grup(b"STAT", 0, body)


def ltex_group(entries):
    """entries: list of (formid, edid, icon)."""
    body = b""
    for formid, edid, icon in entries:
        body += rec(b"LTEX",
                    sub(b"EDID", edid.encode() + b"\x00") + sub(b"ICON", icon.encode() + b"\x00"),
                    formid=formid)
    return grup(b"LTEX", 0, body)


def make_plugin(cells, ws_formid=0x3C, ltex_entries=None, stat_entries=None):
    body = b""
    for (cx, cy), val in cells.items():
        formid, grid, comp = val[0], val[1], val[2]
        land_extra = val[3] if len(val) > 3 else b""
        refrs = val[4] if len(val) > 4 else b""
        body += make_cell(cx, cy, formid, grid, comp, land_extra=land_extra, refrs=refrs)
    subblock = grup(b"\x00\x00\x00\x00", 5, body)      # Exterior Cell Sub-Block
    block = grup(b"\x00\x00\x00\x00", 4, subblock)     # Exterior Cell Block
    wrld = rec(b"WRLD", b"", formid=ws_formid)
    world_children = grup(fid_bytes(ws_formid), 1, block)
    wrld_top = grup(b"WRLD", 0, wrld + world_children)
    # A decoy non-worldspace top group that must be ignored.
    decoy = grup(b"DIAL", 0, rec(b"DIAL", b"junkjunk"))
    tes4 = rec(b"TES4", b"")
    out = tes4 + decoy + wrld_top
    if ltex_entries:
        out += ltex_group(ltex_entries)
    if stat_entries:
        out += stat_group(stat_entries)
    return out


def ramp(seed=0):
    return [[float((x + y + seed) * HEIGHT_SCALE) for x in range(GRID)] for y in range(GRID)]


def test_load_order_override_and_delete():
    g = ramp()
    ex = TerrainExtractor(target_ws=0x3C, models=True)
    # base plugin: one REFR (FormID 0x401) of base 0x300
    base = make_plugin({(0, 0): (0x100, g, False, b"", refr(0x300, (1., 2., 3.), formid=0x401))},
                       stat_entries=[(0x300, "a.nif")])
    ex.parse_bytes(base)
    assert len(ex.placements[(0, 0)]) == 1
    # override plugin: same FormID at a new position -> replaces, not duplicates
    over = make_plugin({(0, 0): (0x100, g, False, b"", refr(0x300, (9., 9., 9.), formid=0x401))})
    ex.parse_bytes(over)
    assert len(ex.placements[(0, 0)]) == 1
    assert ex.placements[(0, 0)][0]["pos"] == (9., 9., 9.)
    # delete plugin: REFR 0x401 with the Deleted flag (0x20) -> removed
    deleted = rec(b"REFR", b"", flags=0x20, formid=0x401)
    ex.parse_bytes(make_plugin({(0, 0): (0x100, g, False, b"", deleted)}))
    assert len(ex.placements.get((0, 0), [])) == 0


def test_read_masters(tmp_path):
    from oblivion2vmf.esm import read_masters
    p = tmp_path / "m.esp"
    hdr = (sub(b"HEDR", b"\x00" * 12) + sub(b"MAST", b"Oblivion.esm\x00") + sub(b"DATA", b"\x00" * 8)
           + sub(b"MAST", b"Res.esm\x00") + sub(b"DATA", b"\x00" * 8))
    p.write_bytes(rec(b"TES4", hdr))
    assert read_masters(str(p)) == ["Oblivion.esm", "Res.esm"]


def test_extract_single_cell():
    grid = ramp()
    plugin = make_plugin({(-19, -1): (0x100, grid, False)})
    ex = TerrainExtractor(target_ws=0x3C).parse_bytes(plugin)
    assert set(ex.cells) == {(-19, -1)}
    assert ex.cells[(-19, -1)] == grid


def test_compressed_land():
    grid = ramp(3)
    plugin = make_plugin({(5, 7): (0x200, grid, True)})
    ex = TerrainExtractor(target_ws=0x3C).parse_bytes(plugin)
    assert ex.cells[(5, 7)] == grid


def test_bounds_filter_and_list():
    cells = {
        (0, 0): (0x100, ramp(0), False),
        (10, 10): (0x110, ramp(1), False),
        (-5, -5): (0x120, ramp(2), False),
    }
    plugin = make_plugin(cells)

    full = TerrainExtractor(target_ws=0x3C, list_only=True).parse_bytes(plugin)
    assert full.cell_coords == {(0, 0), (10, 10), (-5, -5)}
    assert full.bbox() == (-5, -5, 10, 10)

    box = TerrainExtractor(target_ws=0x3C, bounds=(-1, -1, 1, 1)).parse_bytes(plugin)
    assert set(box.cells) == {(0, 0)}


def test_wrong_worldspace_excluded():
    plugin = make_plugin({(0, 0): (0x100, ramp(), False)}, ws_formid=0x3C)
    ex = TerrainExtractor(target_ws=0x99).parse_bytes(plugin)
    assert ex.cells == {}
    assert ex.cell_coords == set()


def test_land_textures_and_ltex():
    # Cell with a base texture in quadrant 0 and a dirt overlay painted at one point.
    land_extra = (
        btxt(0x10, 0)                                   # quad 0 base = grass LTEX
        + atxt(0x20, 0, layer=0)                        # quad 0 overlay = dirt LTEX
        + vtxt([(0, 1.0), (18, 0.5)])                   # pos 0 -> [0][0]=255, pos 18 -> [1][1]=128
    )
    plugin = make_plugin(
        {(0, 0): (0x100, ramp(), False, land_extra)},
        ltex_entries=[(0x10, "Grass", "Landscape\\Grass01.dds"),
                      (0x20, "Dirt", "Landscape\\Dirt01.dds")],
    )
    ex = TerrainExtractor(target_ws=0x3C).parse_bytes(plugin)

    assert ex.ltex[0x10]["icon"] == "Landscape\\Grass01.dds"
    assert ex.ltex[0x20]["edid"] == "Dirt"

    quads = ex.cell_textures[(0, 0)]
    assert quads[0]["base"] == 0x10
    fid, alpha = quads[0]["overs"][0]
    assert fid == 0x20
    assert alpha[0][0] == 255
    assert alpha[1][1] == 128          # round(0.5*255)


def test_multiple_overlays_kept_and_sorted():
    # two overlays with different coverage -> both kept, most-covering first
    land_extra = (
        btxt(0x10, 0)
        + atxt(0x20, 0, layer=0) + vtxt([(0, 0.2)])                       # light coverage
        + atxt(0x30, 0, layer=1) + vtxt([(p, 1.0) for p in range(50)])    # heavy coverage
    )
    plugin = make_plugin({(0, 0): (0x100, ramp(), False, land_extra)})
    ex = TerrainExtractor(target_ws=0x3C).parse_bytes(plugin)
    overs = ex.cell_textures[(0, 0)][0]["overs"]
    assert [fid for fid, _ in overs] == [0x30, 0x20]   # heavier first


def test_textures_disabled():
    land_extra = btxt(0x10, 0)
    plugin = make_plugin({(0, 0): (0x100, ramp(), False, land_extra)})
    ex = TerrainExtractor(target_ws=0x3C, textures=False).parse_bytes(plugin)
    assert ex.cells != {}
    assert ex.cell_textures == {}


def test_refr_placements_and_base_models():
    refrs = (refr(0x300, (4096.0, 8192.0, 100.0), rot=(0.0, 0.0, 1.5), formid=0x401)
             + refr(0x301, (1.0, 2.0, 3.0), scale=2.5, formid=0x402))
    plugin = make_plugin(
        {(0, 0): (0x100, ramp(), False, b"", refrs)},
        stat_entries=[(0x300, "Clutter\\Rock01.nif")],
    )
    ex = TerrainExtractor(target_ws=0x3C, models=True).parse_bytes(plugin)

    assert ex.base_models[0x300] == "Clutter\\Rock01.nif"
    plist = ex.placements[(0, 0)]
    assert len(plist) == 2
    p0 = plist[0]
    assert p0["base"] == 0x300
    assert p0["pos"] == (4096.0, 8192.0, 100.0)
    assert abs(p0["rot"][2] - 1.5) < 1e-6
    assert p0["scale"] == 1.0          # no XSCL -> default
    assert plist[1]["scale"] == 2.5    # XSCL present


def test_models_disabled_skips_placements():
    refrs = refr(0x300, (0.0, 0.0, 0.0), formid=0x401)
    plugin = make_plugin({(0, 0): (0x100, ramp(), False, b"", refrs)},
                         stat_entries=[(0x300, "Clutter\\Rock01.nif")])
    ex = TerrainExtractor(target_ws=0x3C, models=False).parse_bytes(plugin)
    assert ex.placements == {}
    assert ex.base_models == {}
