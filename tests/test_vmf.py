import os

from oblivion2vmf.land import GRID, HEIGHT_SCALE
from oblivion2vmf.vmf import (
    NODRAW_MATERIAL,
    WATER_MATERIAL,
    _Ids,
    _fade_for,
    _greedy_rects,
    DEFAULT_FADE_TABLE,
    build_lighting_entities,
    build_prop_entities,
    build_skybox_enclosure,
    build_solids,
    build_water_solids,
    write_vmf,
)


def flat(value=0.0):
    return [[float(value)] * GRID for _ in range(GRID)]


def ramp():
    return [[float((x + y) * HEIGHT_SCALE) for x in range(GRID)] for y in range(GRID)]


def _block_rows(solid_text, block_name):
    """Return the list of `rowN` value-strings inside the named dispinfo subblock."""
    lines = solid_text.splitlines()
    header = [i for i, l in enumerate(lines) if l.strip() == block_name][0]
    rows = []
    for l in lines[header + 2:]:          # skip the header line and its `{`
        if l.strip() == "}":
            break
        rows.append(l.split('"')[3])      # value is the 4th quoted field
    return rows


def test_water_solid_one_per_watered_cell():
    cells = {(0, 0): flat(100.0), (1, 0): flat(100.0)}
    cell_water = {(0, 0): 200.0}            # only one cell has water
    solids, n = build_water_solids(cells, cell_water, scale=1.0, recenter=False)
    assert n == 1 and len(solids) == 1
    s = solids[0]
    # top face = water material, exactly five nodraw faces
    assert s.count('"material" "%s"' % WATER_MATERIAL) == 1
    assert s.count('"material" "%s"' % NODRAW_MATERIAL) == 5


def test_water_surface_above_terrain_and_volume():
    # water at height 500 over terrain at 100 -> surface above, bottom below terrain
    cells = {(0, 0): flat(100.0)}
    solids, n = build_water_solids(cells, {(0, 0): 500.0}, scale=1.0, recenter=False,
                                   min_depth=64.0, margin=8.0)
    assert n == 1
    zs = []
    for line in solids[0].splitlines():
        if '"plane"' in line:
            # parse z of each plane point
            for grp in line.split("(")[1:]:
                parts = grp.split(")")[0].split()
                zs.append(float(parts[2]))
    assert max(zs) == 500.0                  # surface at water height
    assert min(zs) <= 100.0 - 8.0            # bottom reaches below terrain min


def test_write_vmf_includes_water_and_lod():
    out = os.path.join(os.path.dirname(__file__), "_wtmp.vmf")
    stats = write_vmf({(0, 0): flat(0.0)}, out, scale=1.0, recenter=False,
                      water=True, cell_water={(0, 0): 64.0})
    try:
        txt = open(out).read()
        assert stats["water"] == 1
        assert "water_lod_control" in txt
        assert WATER_MATERIAL in txt
    finally:
        os.remove(out)


def test_no_water_when_disabled():
    out = os.path.join(os.path.dirname(__file__), "_wtmp2.vmf")
    stats = write_vmf({(0, 0): flat(0.0)}, out, scale=1.0, recenter=False,
                      water=False, cell_water={(0, 0): 64.0})
    try:
        assert stats["water"] == 0
        assert "water_lod_control" not in open(out).read()
    finally:
        os.remove(out)


def test_disp_top_textured_sides_nodraw():
    solids, _ = build_solids({(0, 0): flat()}, power=4, material="mymat",
                             lightmapscale=48)
    s = solids[0]
    assert s.count('"material" "mymat"') == 1            # only the top face
    assert s.count('"material" "%s"' % NODRAW_MATERIAL) == 5
    assert '"lightmapscale" "48"' in s


def test_outer_power_lowers_edge_cells():
    # 3x3 cells; inner cell (1,1) at power 4, the 8 edge cells at power 2
    cells = {(cx, cy): flat() for cx in range(3) for cy in range(3)}
    solids, stats = build_solids(cells, power=4, outer_power=2, outer_margin=1)
    # power4 -> 4 disps/cell (1 inner), power2 -> 64 disps/cell (8 edge)
    assert stats["outer_power"] == 2
    assert len(solids) == 1 * 4 + 8 * 64


def test_fade_categories():
    t = DEFAULT_FADE_TABLE
    assert _fade_for("models/oblivion2vmf/bush_shrubazalea.mdl", t) == t["bush"]
    assert _fade_for("models/oblivion2vmf/clutter_barrel01.mdl", t) == t["clutter"]
    assert _fade_for("models/props_foliage/tree_deciduous_01a.mdl", t) == t["tree"]
    assert _fade_for("models/oblivion2vmf/rocks_westweald_01.mdl", t) == t["rock"]
    # buildings fade far (at the fog wall), not never
    assert _fade_for("models/oblivion2vmf/architecture_farmhouse.mdl", t) == t["building"]
    # fade_scale multiplies both distances
    s, e = _fade_for("models/oblivion2vmf/bush_x.mdl", t, scale=0.5)
    assert (s, e) == (t["bush"][0] * 0.5, t["bush"][1] * 0.5)


def test_prop_fade_distances_and_scale():
    placements = {(0, 0): [
        {"base": 1, "pos": (0, 0, 0), "rot": (0, 0, 0), "scale": 1.0},
        {"base": 2, "pos": (0, 0, 0), "rot": (0, 0, 0), "scale": 1.0},
    ]}
    model_map = {1: "models/oblivion2vmf/bush_x.mdl",
                 2: "models/oblivion2vmf/architecture_house.mdl"}
    ents, placed, _ = build_prop_entities(placements, model_map, _Ids(), 1.0,
                                          (0, 0, 0), prop_fade=True)
    assert placed == 2
    bush = next(e for e in ents if "bush_x" in e)
    house = next(e for e in ents if "architecture_house" in e)
    assert '"fademaxdist"' in bush and '"fademaxdist"' in house   # both fade now
    assert '"fademaxdist" "3800"' in bush                         # bush fades sooner
    assert '"fademaxdist" "14000"' in house                       # building at fog wall
    # fade_scale 0.5 halves the distances
    ents2, _, _ = build_prop_entities(placements, model_map, _Ids(), 1.0, (0, 0, 0),
                                      prop_fade=True, fade_scale=0.5)
    bush2 = next(e for e in ents2 if "bush_x" in e)
    assert '"fademaxdist" "1900"' in bush2


def test_greedy_rects_merges_block():
    cells = {(x, y) for x in range(3) for y in range(2)}     # 3x2 block
    rects = _greedy_rects(cells)
    assert rects == [(0, 0, 2, 1)]                            # one rectangle


def test_greedy_rects_L_shape():
    cells = {(0, 0), (1, 0), (0, 1)}                          # L
    rects = _greedy_rects(cells)
    assert len(rects) == 2                                    # can't be one rect
    covered = set()
    for x0, y0, x1, y1 in rects:
        for x in range(x0, x1 + 1):
            for y in range(y0, y1 + 1):
                covered.add((x, y))
    assert covered == cells


def test_water_merge_same_height_one_brush():
    cells = {(0, 0): flat(50.0), (1, 0): flat(50.0), (2, 0): flat(50.0)}
    cw = {(0, 0): 100.0, (1, 0): 100.0, (2, 0): 100.0}        # same height
    merged, nm = build_water_solids(cells, cw, scale=1.0, recenter=False, merge=True)
    unmerged, nu = build_water_solids(cells, cw, scale=1.0, recenter=False, merge=False)
    assert nm == 1 and nu == 3                                # merged into one


def test_water_different_heights_not_merged():
    cells = {(0, 0): flat(50.0), (1, 0): flat(50.0)}
    cw = {(0, 0): 100.0, (1, 0): 200.0}                       # different heights
    _, n = build_water_solids(cells, cw, scale=1.0, recenter=False, merge=True)
    assert n == 2


def test_lighting_entities():
    ents = build_lighting_entities(_Ids(), sun_yaw=200, sun_pitch=-45, fog=True)
    blob = "\n".join(ents)
    assert "light_environment" in blob
    assert "shadow_control" in blob
    assert "env_fog_controller" in blob
    assert '"pitch" "-45"' in blob


def test_write_vmf_lighting_default_on():
    out = os.path.join(os.path.dirname(__file__), "_ltmp.vmf")
    stats = write_vmf({(0, 0): flat(0.0)}, out, scale=1.0, recenter=False)
    try:
        txt = open(out).read()
        assert stats["lighting"] is True
        assert "light_environment" in txt
    finally:
        os.remove(out)


def test_vis_floor_adds_separate_brushes_to_floor():
    cells = {(0, 0): flat(1000.0)}                 # terrain at z=1000 (no recenter)
    thin, ts = build_solids(cells, power=4, recenter=False, thickness=16.0)
    deep, ds = build_solids(cells, power=4, recenter=False, vis_floor=True,
                            vis_floor_depth=512.0)
    # vis-floor adds a separate solid brush per displacement tile (power4 -> 4)
    assert ts.get("vis_brushes", 0) == 0
    assert ds["vis_brushes"] == 4
    assert ds["displacements"] == 4                # disp count not inflated
    # the displacement itself stays thin; only the separate brushes reach the floor
    # min z across all plane points: thin bottom ~984, vis-floor bottom ~488
    def min_z(solid_list):
        zs = []
        for s in solid_list:
            for line in s.splitlines():
                if '"plane"' in line:
                    for grp in line.split("(")[1:]:
                        zs.append(float(grp.split(")")[0].split()[2]))
        return min(zs)
    assert min_z(thin) > 980
    assert abs(min_z(deep) - (1000.0 - 512.0)) < 1.0   # floor at terrain_min - depth


def test_skybox_enclosure_six_brushes():
    cells = {(0, 0): flat(0.0), (1, 0): flat(0.0)}
    solids, n = build_skybox_enclosure(cells, scale=1.0, recenter=False)
    assert n == 6
    assert all("tools/toolsskybox" in s for s in solids)


def test_write_vmf_seal_sky():
    out = os.path.join(os.path.dirname(__file__), "_stmp.vmf")
    stats = write_vmf({(0, 0): flat(0.0)}, out, scale=1.0, recenter=False,
                      seal_sky=True, vis_floor=True)
    try:
        assert stats["skybox"] == 6
        assert "tools/toolsskybox" in open(out).read()
    finally:
        os.remove(out)


def test_disp_counts_power4():
    solids, stats = build_solids({(0, 0): flat()}, power=4)
    assert stats["displacements"] == 4    # 2x2 tiles per cell
    assert len(solids) == 4
    joined = "\n".join(solids)
    assert joined.count("dispinfo") == 4
    assert joined.count('"power" "4"') == 4


def test_distances_dimensions():
    solids, _ = build_solids({(0, 0): flat()}, power=4)
    rows = _block_rows(solids[0], "distances")
    assert len(rows) == 17
    for r in rows:
        assert len(r.split()) == 17


def test_triangle_tags_dimensions():
    solids, _ = build_solids({(0, 0): flat()}, power=4)
    rows = _block_rows(solids[0], "triangle_tags")
    assert len(rows) == 16
    for r in rows:
        assert len(r.split()) == 32


def test_flat_terrain_zero_distances():
    solids, _ = build_solids({(0, 0): flat(100.0)}, power=4)
    for s in solids:
        for r in _block_rows(s, "distances"):
            assert all(float(v) == 0.0 for v in r.split())


def test_ramp_max_distance_matches_relief():
    solids, _ = build_solids({(0, 0): ramp()}, power=4, scale=1.0)
    # within one 16-interval tile, relief = (16 east + 16 north) steps * 8 = 256
    max_d = max(
        float(v)
        for s in solids
        for r in _block_rows(s, "distances")
        for v in r.split()
    )
    assert abs(max_d - 256.0) < 1e-6


def test_write_vmf_balanced_braces(tmp_path):
    out = tmp_path / "t.vmf"
    stats = write_vmf({(0, 0): ramp(), (1, 0): flat()}, str(out), power=4)
    text = out.read_text()
    assert text.count("{") == text.count("}")
    assert text.count("dispinfo") == stats["displacements"] == 8
    assert 'classname" "worldspawn"' in text
    assert "startposition" in text
    assert os.path.getsize(str(out)) > 0


def test_scale_applies(tmp_path):
    out = tmp_path / "s.vmf"
    stats = write_vmf({(0, 0): flat()}, str(out), power=4, scale=0.5, recenter=False)
    # one cell spans 4096 units -> 2048 HU at scale 0.5 (no recenter)
    assert abs(stats["maxs"][0] - 2048.0) < 1e-6


def test_recenter_centers_on_origin():
    # A far-off cell (like West Weald near -19) must still land near the origin.
    solids, stats = build_solids({(-19, -1): flat()}, power=4, scale=1.0, recenter=True)
    mn, mx = stats["mins"], stats["maxs"]
    # bbox spans one cell (4096 units) centered at 0 -> +/-2048 in x and y
    assert abs(mn[0] + 2048.0) < 1e-6 and abs(mx[0] - 2048.0) < 1e-6
    assert abs(mn[1] + 2048.0) < 1e-6 and abs(mx[1] - 2048.0) < 1e-6


def test_recenter_invariant_distances():
    # Recentering is a uniform shift, so displacement distances are unchanged.
    a, _ = build_solids({(-19, -1): ramp()}, power=4, recenter=True)
    b, _ = build_solids({(-19, -1): ramp()}, power=4, recenter=False)
    da = [r for s in a for r in _block_rows(s, "distances")]
    db = [r for s in b for r in _block_rows(s, "distances")]
    assert da == db


def test_prop_static_emission(tmp_path):
    placements = {(0, 0): [
        {"base": 0x300, "pos": (0.0, 0.0, 0.0), "rot": (0.0, 0.0, 0.0), "scale": 1.0},
        {"base": 0x999, "pos": (10.0, 0.0, 0.0), "rot": (0.0, 0.0, 0.0), "scale": 1.0},
    ]}
    model_map = {0x300: "models/oblivion2vmf/rock.mdl"}   # 0x999 not converted -> skipped
    out = tmp_path / "p.vmf"
    stats = write_vmf({(0, 0): flat()}, str(out), power=4,
                      placements=placements, model_map=model_map)
    text = out.read_text()
    assert text.count("{") == text.count("}")
    assert text.count('"classname" "prop_static"') == 1
    assert 'models/oblivion2vmf/rock.mdl' in text
    assert stats["props"] == 1
    assert stats["props_skipped"] == 1


def test_no_props_without_model_map(tmp_path):
    out = tmp_path / "n.vmf"
    stats = write_vmf({(0, 0): flat()}, str(out), power=4)
    assert stats["props"] == 0
    assert "prop_static" not in out.read_text()


def test_power3_tiling():
    solids, stats = build_solids({(0, 0): flat()}, power=3)
    assert stats["displacements"] == 16   # 4x4 tiles per cell
    rows = _block_rows(solids[0], "distances")
    assert len(rows) == 9 and all(len(r.split()) == 9 for r in rows)


def test_build_solids_with_texturer():
    from oblivion2vmf.textures import Texturer
    ltex = {0x10: {"icon": "grass", "edid": ""}, 0x20: {"icon": "dirt", "edid": ""}}
    grid = [[0] * 17 for _ in range(17)]
    grid[16][16] = 255
    cell_textures = {(0, 0): {
        0: {"base": 0x10, "overs": [(0x20, grid)]},  # SW: grass base, dirt overlay
        1: {"base": 0x20}, 2: {"base": 0x10}, 3: {"base": 0x10},
    }}
    tx = Texturer(ltex, cell_textures)
    solids, _ = build_solids({(0, 0): flat()}, power=4, texturer=tx)
    joined = "\n".join(solids)
    assert "oblivion2vmf/blend_dirt_grass" in joined   # SW quadrant (canonicalised)
    assert "oblivion2vmf/dirt" in joined               # SE quadrant single
    # the SW tile (first solid) carries blend alpha (inverted: mostly 255)
    assert any("255" in r for r in _block_rows(solids[0], "alphas"))


def test_build_solids_4way_multiblock(tmp_path):
    from oblivion2vmf.textures import Texturer
    ltex = {0x10: {"icon": "grass", "edid": ""}, 0x20: {"icon": "dirt", "edid": ""}}
    grid = [[0] * 17 for _ in range(17)]
    grid[16][16] = 255
    cell_textures = {(0, 0): {
        0: {"base": 0x10, "overs": [(0x20, grid)]},
        1: {"base": 0x20}, 2: {"base": 0x10}, 3: {"base": 0x10},
    }}
    tx = Texturer(ltex, cell_textures, fourway=True)
    out = tmp_path / "f.vmf"
    write_vmf({(0, 0): flat()}, str(out), power=4, texturer=tx)
    text = out.read_text()
    assert text.count("{") == text.count("}")            # multiblend blocks balanced
    assert "blend4_dirt_grass_grass_grass" in text       # SW quad 4-way material (canonical)
    # SW solid: multiblend = 17 rows of 17*4 floats; color blocks present
    sw = text.split("\tsolid")[1]
    mb = _block_rows(sw, "multiblend")
    assert len(mb) == 17 and all(len(r.split()) == 17 * 4 for r in mb)
    assert _block_rows(sw, "multiblend_color_0")
    assert _block_rows(sw, "alphablend")
