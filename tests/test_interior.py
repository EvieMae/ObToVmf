"""Interior (room) VMF writer + extractor smoke tests."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from oblivion2vmf.vmf import write_interior_vmf, add_instance_to_vmf
from oblivion2vmf.esm import InteriorExtractor


def _room(tmp_path, name="room.vmf", **kw):
    placements = {0x10: [
        {"base": 0xA1, "pos": (100.0, 0.0, 0.0), "rot": (0, 0, 0), "scale": 1.0},
        {"base": 0xA2, "pos": (-100.0, 50.0, 20.0), "rot": (0, 0, 1.5), "scale": 1.0},
    ]}
    model_map = {0xA1: "models/oblivion2vmf/a1.mdl", 0xA2: "models/oblivion2vmf/a2.mdl"}
    out = tmp_path / name
    write_interior_vmf(str(out), placements, model_map, scale=0.5, **kw)
    return out


def test_write_interior_vmf(tmp_path):
    # two props + one light reference in one synthetic cell
    placements = {0x10: [
        {"base": 0xA1, "pos": (100.0, 0.0, 0.0), "rot": (0, 0, 0), "scale": 1.0},
        {"base": 0xA2, "pos": (-100.0, 50.0, 20.0), "rot": (0, 0, 1.5), "scale": 1.0},
        {"base": 0xB1, "pos": (0.0, 0.0, 80.0), "rot": (0, 0, 0), "scale": 1.0},   # the light
    ]}
    model_map = {0xA1: "models/oblivion2vmf/a1.mdl", 0xA2: "models/oblivion2vmf/a2.mdl"}
    light_bases = {0xB1: {"radius": 400, "color": (255, 200, 150)}}

    out = tmp_path / "room.vmf"
    stats = write_interior_vmf(str(out), placements, model_map, scale=0.5,
                               light_bases=light_bases, ambient=(40, 45, 60))
    txt = out.read_text()
    assert stats["props"] == 2                       # 2 models; the LIGH ref isn't a prop
    assert stats["lights"] == 1
    assert txt.count('"prop_static"') == 2
    assert '"classname" "light"' in txt
    assert "tools/toolsblack" in txt                 # sealed black room
    assert "info_player_start" in txt                # spawn point exists
    assert '"_light" "255 200 150' in txt            # light colour from the LIGH base


def test_skybox_room_wraps_in_toolsskybox(tmp_path):
    sealed = _room(tmp_path, "sealed.vmf").read_text()
    sky = _room(tmp_path, "sky.vmf", skybox_room=True).read_text()
    assert "tools/toolsblack" in sealed and "tools/toolsskybox" not in sealed
    assert "tools/toolsskybox" in sky and "tools/toolsblack" not in sky


def test_add_instance_creates_host_and_places(tmp_path):
    room = _room(tmp_path, "MyRoom.vmf")
    host = tmp_path / "main.vmf"          # does not exist yet -> created
    origin, rel = add_instance_to_vmf(str(host), str(room))
    txt = host.read_text()
    assert '"classname" "func_instance"' in txt
    assert rel == "MyRoom.vmf"                       # relative, forward slashes
    assert '"file" "MyRoom.vmf"' in txt
    assert "func_instance" in txt and "cameras" in txt   # inserted before footer


def test_add_instance_avoids_overlap(tmp_path):
    a = _room(tmp_path, "A.vmf")
    b = _room(tmp_path, "B.vmf")
    host = tmp_path / "main.vmf"
    o1, _ = add_instance_to_vmf(str(host), str(a))
    o2, _ = add_instance_to_vmf(str(host), str(b))
    assert o2[0] > o1[0]                              # second laid further along +X
    assert host.read_text().count('"classname" "func_instance"') == 2


def test_add_instance_explicit_origin(tmp_path):
    room = _room(tmp_path, "R.vmf")
    host = tmp_path / "main.vmf"
    o, _ = add_instance_to_vmf(str(host), str(room), origin=(512.0, -256.0, 0.0))
    assert o == (512.0, -256.0, 0.0)
    assert '"origin" "512 -256 0"' in host.read_text()


def test_write_interior_vmf_empty_raises(tmp_path):
    try:
        write_interior_vmf(str(tmp_path / "x.vmf"), {}, {})
    except ValueError:
        return
    raise AssertionError("empty interior should raise ValueError")


def test_interior_extractor_find():
    ex = InteriorExtractor()
    ex.interiors = {
        0x1: {"edid": "SkingradHouseForSale", "full": "Rosethorn Hall", "ambient": None},
        0x2: {"edid": "ChorrolFightersGuild", "full": "Fighters Guild", "ambient": None},
    }
    assert ex.find("SkingradHouseForSale")[0] == 0x1          # exact EDID
    assert ex.find("rosethorn")[0] == 0x1                     # unique substring (name)
    assert ex.find("guild")[0] == 0x2                         # unique substring
    assert ex.find("nope")[0] is None                         # no match


def test_box_and_ramp_hulls():
    from oblivion2vmf.model import box_hull, ramp_hull, _BOX_FACES, _WEDGE_FACES
    bb = (0.0, 0.0, 0.0, 10.0, 4.0, 6.0)
    bverts, bfaces = box_hull(bb)
    assert len(bverts) == 8 and bfaces == _BOX_FACES
    assert (10.0, 4.0, 6.0) in bverts and (0.0, 0.0, 0.0) in bverts

    rverts, rfaces = ramp_hull(bb, "+x")
    assert len(rverts) == 6 and rfaces == _WEDGE_FACES
    # +x ramp: the two raised (z1) verts sit at the high-x edge
    top = [v for v in rverts if v[2] == 6.0]
    assert all(v[0] == 10.0 for v in top) and len(top) == 2
    # every vertex referenced by a face (so studiomdl hulls all of them)
    assert set(i for f in rfaces for i in f) == set(range(6))

    # -y ramp raises the low-y edge
    yv, _ = ramp_hull(bb, "-y")
    assert all(v[1] == 0.0 for v in yv if v[2] == 6.0)


def test_load_model_overrides_lowercases_and_tolerates_missing(tmp_path):
    import json as _j
    from oblivion2vmf.cli import _load_model_overrides
    assert _load_model_overrides(None) is None
    assert _load_model_overrides(str(tmp_path / "nope.json")) is None   # missing -> None
    p = tmp_path / "ov.json"
    p.write_text(_j.dumps({"Architecture/Anvil/House01.NIF": {"collision": "ramp"}}))
    ov = _load_model_overrides(str(p))
    assert "architecture/anvil/house01.nif" in ov                      # key lowercased
    assert ov["architecture/anvil/house01.nif"]["collision"] == "ramp"


def test_hullview_smd_parse_and_bounds(tmp_path):
    from oblivion2vmf.hullview import read_smd_mesh, mesh_bounds
    smd = tmp_path / "m.smd"
    smd.write_text("version 1\nnodes\n0 \"root\" -1\nend\nskeleton\ntime 0\n"
                   "0 0 0 0 0 0 0\nend\ntriangles\n"
                   "phys\n0 -5 0 0 0 0 1 0 0\n0 5 0 0 0 0 1 0 0\n0 5 8 2 0 0 1 0 0\nend\n")
    verts, tris = read_smd_mesh(str(smd))
    assert len(verts) == 3 and tris == [(0, 1, 2)]
    assert mesh_bounds(verts) == (-5, 0, 0, 5, 8, 2)


def test_collision_smd_multi_hull_writes_each_box(tmp_path):
    # the 'hulls' build path writes one convex element per authored box
    from oblivion2vmf.model import box_hull, write_collision_smd
    boxes = [(0, 0, 0, 10, 10, 10), (20, 0, 0, 30, 10, 40)]
    out = tmp_path / "phys.smd"
    write_collision_smd([box_hull(b) for b in boxes], str(out), scale=1.0)
    txt = out.read_text()
    # 2 boxes * 12 tris each = 24 "phys" material lines
    assert txt.count("\nphys\n") + (1 if txt.startswith("phys\n") else 0) == 24
    assert "30.000000 10.000000 40.000000" in txt        # second box max corner present
