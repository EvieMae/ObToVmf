"""VTF decode + grouped-SMD parse for the 3D viewport."""
import os
import struct
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from oblivion2vmf import dds, textures
from oblivion2vmf.vtf import vtf_to_rgb
from oblivion2vmf.hullview import read_smd_grouped


def test_vtf_bgr888_roundtrip(tmp_path):
    p = tmp_path / "a.vtf"
    textures.write_vtf_rgb(str(p), [(255, 0, 0), (0, 255, 0), (0, 0, 255), (200, 200, 200)], 2, 2)
    img = vtf_to_rgb(str(p))
    assert img.shape == (2, 2, 3)
    assert img[0, 0].tolist() == [255, 0, 0]          # RGB, not BGR
    assert img[0, 1].tolist() == [0, 255, 0]


def test_vtf_dxt1_roundtrip(tmp_path):
    # one 4x4 DXT1 block, c0 = red (0xF800), all indices 0 -> solid red
    dh = bytearray(128)
    dh[0:4] = b"DDS "
    struct.pack_into("<I", dh, 12, 4)                  # height
    struct.pack_into("<I", dh, 16, 4)                  # width
    struct.pack_into("<I", dh, 28, 1)                  # mipcount
    dh[84:88] = b"DXT1"
    block = struct.pack("<HHI", 0xF800, 0x001F, 0)
    vtf_bytes = dds.dds_to_vtf(bytes(dh) + block)
    p = tmp_path / "b.vtf"
    p.write_bytes(vtf_bytes)
    img = vtf_to_rgb(str(p))
    assert img.shape == (4, 4, 3)
    assert img[0, 0, 0] > 240 and img[0, 0, 1] < 16 and img[0, 0, 2] < 16


def test_vtf_unsupported_returns_none(tmp_path):
    p = tmp_path / "bad.vtf"
    p.write_bytes(b"NOPE" + b"\x00" * 100)
    assert vtf_to_rgb(str(p)) is None


def test_read_smd_grouped_splits_by_material(tmp_path):
    smd = ("version 1\nnodes\n0 \"root\" -1\nend\nskeleton\ntime 0\n"
           "0 0 0 0 0 0 0\nend\ntriangles\n"
           "matA\n0 0 0 0 0 0 1 0 0\n0 1 0 0 0 0 1 1 0\n0 1 1 0 0 0 1 1 1\n"
           "matB\n0 5 0 0 0 0 1 0 0\n0 6 0 0 0 0 1 1 0\n0 6 1 0 0 0 1 1 1\nend\n")
    p = tmp_path / "g.smd"
    p.write_text(smd)
    g = read_smd_grouped(str(p))
    assert sorted(g) == ["matA", "matB"]
    assert len(g["matA"]["tris"]) == 1 and len(g["matA"]["points"]) == 3
    assert len(g["matA"]["uvs"]) == 3                  # uv per unique corner


def test_acd_parts_write_arbitrary_convex(tmp_path):
    # the build's acd_parts path writes arbitrary (verts, faces) convex pieces
    from oblivion2vmf.model import write_collision_smd
    tetra_v = [(0, 0, 0), (10, 0, 0), (0, 10, 0), (0, 0, 10)]
    tetra_f = [(0, 1, 2), (0, 1, 3), (1, 2, 3), (0, 2, 3)]
    parts = [(tetra_v, tetra_f), (tetra_v, tetra_f)]
    out = tmp_path / "phys.smd"
    write_collision_smd(parts, str(out), scale=1.0)
    txt = out.read_text()
    assert txt.count("\nphys\n") == 8                 # 2 parts * 4 tris
    assert "10.000000 0.000000 0.000000" in txt


def test_trap_hull_and_spec_normalizer():
    from oblivion2vmf.model import trap_hull, hull_from_spec, _BOX_FACES, _WEDGE_FACES
    bb = (0.0, 0.0, 0.0, 10.0, 4.0, 6.0)
    verts, faces = trap_hull(bb, top_scale=0.5)
    assert len(verts) == 8 and faces == _BOX_FACES
    top = [v for v in verts if v[2] == 6.0]
    assert len(top) == 4
    # top scaled 0.5 about centre (5, 2): x in [2.5, 7.5], y in [1, 3]
    assert min(v[0] for v in top) == 2.5 and max(v[0] for v in top) == 7.5
    assert min(v[1] for v in top) == 1.0 and max(v[1] for v in top) == 3.0

    # spec forms: legacy list = box; dicts pick the right shape; junk -> None
    assert hull_from_spec(list(bb))[1] == _BOX_FACES
    assert hull_from_spec({"type": "wedge", "bounds": list(bb), "axis": "-y"})[1] == _WEDGE_FACES
    assert hull_from_spec({"type": "trapezium", "bounds": list(bb)})[1] == _BOX_FACES
    assert hull_from_spec({"type": "box", "bounds": [1, 2]}) is None
    assert hull_from_spec([1, 2, 3]) is None


def test_mixed_shape_hulls_write(tmp_path):
    # the 'hulls' build path accepts mixed boxes/wedges/trapeziums in one model
    from oblivion2vmf.model import hull_from_spec, write_collision_smd
    specs = [[0, 0, 0, 10, 10, 10],
             {"type": "wedge", "bounds": [20, 0, 0, 30, 10, 8], "axis": "+x"},
             {"type": "trap", "bounds": [40, 0, 0, 50, 10, 12], "top_scale": 0.5}]
    parts = [p for p in (hull_from_spec(s) for s in specs) if p is not None]
    assert len(parts) == 3
    out = tmp_path / "phys.smd"
    write_collision_smd(parts, str(out), scale=1.0)
    txt = out.read_text()
    # box 12 + wedge 8 + trap 12 = 32 triangles
    assert txt.count("\nphys\n") == 32
