import os

from oblivion2vmf.land import GRID, HEIGHT_SCALE
from oblivion2vmf.skybox import generate_skybox_terrain, _cell_color
from oblivion2vmf.textures import COLORS
from oblivion2vmf.vmf import _Ids, build_skybox_model


def ramp():
    return [[float((x + y) * HEIGHT_SCALE) for x in range(GRID)] for y in range(GRID)]


def test_cell_color_falls_back_to_default():
    assert _cell_color((9, 9), {}, {}) == COLORS["default"]


def test_cell_color_classifies_base():
    ltex = {0x10: {"icon": "Landscape\\Grass01.dds", "edid": "Grass"}}
    cell_textures = {(0, 0): {0: {"base": 0x10}, 1: {"base": 0x10}}}
    assert _cell_color((0, 0), cell_textures, ltex) == COLORS["grass"]


def test_generate_skybox_terrain_writes_mesh_and_texture(tmp_path):
    cells = {(0, 0): ramp(), (1, 0): ramp(), (0, 1): ramp(), (1, 1): ramp()}
    work = tmp_path / "work"; work.mkdir()
    mats = tmp_path / "mats"
    path, bounds = generate_skybox_terrain(
        cells, {}, {}, 0.5625, (0.0, 0.0, 0.0), str(work), str(mats),
        studiomdl=None, gamedir=None, compile_models=False, step=8, tex_per_cell=4)
    assert path == "models/oblivion2vmf/skybox_terrain.mdl"
    assert len(bounds) == 6 and bounds[1] > bounds[0]        # nonzero extent
    assert (work / "skybox_terrain.smd").exists()
    assert (work / "skybox_terrain.qc").exists()
    vtf = mats / "models" / "oblivion2vmf" / "skybox_terrain.vtf"
    assert vtf.exists() and len(vtf.read_bytes()) == 80 + (2 * 4) * (2 * 4) * 3  # 8x8 px


def test_build_skybox_model_room_and_entities():
    bounds = (-16000.0, 16000.0, -12000.0, 12000.0, -300.0, 3000.0)
    solids, ents = build_skybox_model(_Ids(), "models/oblivion2vmf/skybox_terrain.mdl",
                                      bounds, sky_scale=16)
    assert len(solids) == 6                                   # sealed room
    assert all("tools/toolsskybox" in s for s in solids)
    blob = "\n".join(ents)
    assert '"classname" "sky_camera"' in blob and '"scale" "16"' in blob
    assert "skybox_terrain.mdl" in blob
    # prescaled model (compiled at 1/sky_scale) -> prop placed at scale 1.0
    assert '"uniformscale" "1"' in blob


def test_build_skybox_model_uniformscale_when_not_prescaled():
    bounds = (-16000.0, 16000.0, -12000.0, 12000.0, -300.0, 3000.0)
    _, ents = build_skybox_model(_Ids(), "models/oblivion2vmf/ext.mdl",
                                 bounds, sky_scale=16, prescaled=False)
    assert '"uniformscale" "0.0625"' in "\n".join(ents)        # external model: 1/16
