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


def test_cylinder_hull_rings_on_ellipse():
    from oblivion2vmf.model import cylinder_hull
    bb = (0.0, 0.0, 1.0, 20.0, 10.0, 9.0)              # ellipse radii 10, 5
    for sides in (8, 12, 16):
        verts, faces = cylinder_hull(bb, sides=sides)
        assert len(verts) == 2 * sides
        assert {v[2] for v in verts} == {1.0, 9.0}     # only z0 / z1
        for x, y, _ in verts:                          # on the inscribed ellipse
            r = ((x - 10.0) / 10.0) ** 2 + ((y - 5.0) / 5.0) ** 2
            assert abs(r - 1.0) < 1e-9
        assert {i for f in faces for i in f} == set(range(2 * sides))


def test_plane_hull_is_small_box():
    # a plane is a small box: real extents kept, any too-thin axis padded so the
    # convex piece always has 3D volume (no degenerate VPhysics conflicts)
    from oblivion2vmf.model import plane_hull, _BOX_FACES
    verts, faces = plane_hull((0, 0, 5, 10, 4, 100), thickness=2.0)
    assert faces == _BOX_FACES and len(verts) == 8
    assert {v[2] for v in verts} == {5.0, 100.0}        # thick Z kept (not flattened)
    # flat in X (x0==x1, a wall) -> X padded to >= thickness about the plane
    fv, _ = plane_hull((50, 0, 0, 50, 100, 80), thickness=2.0)
    xs = {v[0] for v in fv}
    assert xs == {49.0, 51.0}                            # padded +/-1 about x=50
    assert max(v[1] for v in fv) - min(v[1] for v in fv) == 100   # Y kept


def test_hull_from_spec_cylinder_plane_and_rot():
    from oblivion2vmf.model import hull_from_spec
    cyl = hull_from_spec({"type": "cylinder", "bounds": [0, 0, 0, 10, 10, 5], "sides": 8})
    assert cyl is not None and len(cyl[0]) == 16
    pl = hull_from_spec({"type": "plane", "bounds": [0, 0, 0, 10, 10, 50]})
    assert pl is not None and {v[2] for v in pl[0]} == {0.0, 50.0}   # real box, not slab
    # rot [0,0,90] about the bounds centre (5,5,5): corner (0,0,0) -> (10,0,0)
    box = hull_from_spec({"type": "box", "bounds": [0, 0, 0, 10, 10, 10],
                          "rot": [0, 0, 90]})
    x, y, z = box[0][0]
    assert abs(x - 10.0) < 1e-6 and abs(y) < 1e-6 and abs(z) < 1e-6
    # no rot / zero rot = identical verts (backward compatible)
    plain = hull_from_spec({"type": "box", "bounds": [0, 0, 0, 10, 10, 10]})
    zero = hull_from_spec({"type": "box", "bounds": [0, 0, 0, 10, 10, 10],
                           "rot": [0, 0, 0]})
    assert plain[0] == zero[0]
    # junk specs still -> None
    assert hull_from_spec({"type": "cylinder", "bounds": [1, 2]}) is None
    assert hull_from_spec([1, 2, 3]) is None


def test_mesh_hull_spec_roundtrip(tmp_path):
    # the face-edit mode stores explicit geometry as {"type":"mesh"} — the build
    # writes it verbatim as one convex piece
    from oblivion2vmf.model import hull_from_spec, box_hull, write_collision_smd
    verts, faces = box_hull((0.0, 0.0, 0.0, 10.0, 10.0, 10.0))
    spec = {"type": "mesh", "verts": [list(v) for v in verts],
            "faces": [list(f) for f in faces]}
    part = hull_from_spec(spec)
    assert part is not None and len(part[0]) == 8 and len(part[1]) == 12
    # rot rotates about the verts' own bbox centre
    spun = hull_from_spec({**spec, "rot": [0, 0, 90]})
    assert abs(spun[0][0][0] - 10.0) < 1e-6      # (0,0,0) -> (10,0,0) about (5,5,5)
    # malformed specs refused
    assert hull_from_spec({"type": "mesh", "verts": [[0, 0, 0]], "faces": []}) is None
    out = tmp_path / "m.smd"
    write_collision_smd([part], str(out), scale=1.0)
    assert out.read_text().count("\nphys\n") == 12


def test_coplanar_convex_pieces_cube_is_six(tmp_path):
    # a cube's Havok-style trimesh (12 tris) -> 6 exact convex wall pieces
    from oblivion2vmf.model import box_hull, coplanar_convex_pieces, write_collision_smd
    verts, faces = box_hull((0.0, 0.0, 0.0, 100.0, 100.0, 100.0))
    subs = [{"verts": [list(v) for v in verts], "tris": [list(f) for f in faces]}]
    parts = coplanar_convex_pieces(subs, thickness=4.0)
    assert parts is not None
    assert len(parts) == 6                          # one prism per cube face
    for pv_, pf in parts:
        assert len(pv_) == 8 and len(pf) >= 4        # quad face -> 8-vert prism
    out = tmp_path / "phys.smd"
    write_collision_smd(parts, str(out), scale=1.0)
    assert out.read_text().count("\nphys\n") >= 6 * 4


def test_coplanar_pieces_concave_patch_falls_back(tmp_path):
    # an L-shaped coplanar patch (non-convex outline) -> per-triangle prisms, not
    # one over-filling hull
    from oblivion2vmf.model import coplanar_convex_pieces
    # L in the z=0 plane from 3 squares' worth of triangles (6 tris)
    v = [[0, 0, 0], [2, 0, 0], [2, 1, 0], [0, 1, 0],   # bottom bar
         [0, 1, 0], [1, 1, 0], [1, 2, 0], [0, 2, 0]]   # upright bar
    tris = [[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7]]
    parts = coplanar_convex_pieces([{"verts": v, "tris": tris}], thickness=2.0)
    # the L outline over-fills its convex hull, so it must NOT collapse to 1 piece
    assert len(parts) >= 2


def test_simplify_collision_welds_soup():
    # triangle-soup: same 2 triangles repeated with DISTINCT duplicate verts ->
    # welding collapses them to the shared corners
    from oblivion2vmf.model import simplify_collision, collision_vert_count
    quad = [[0, 0, 0], [10, 0, 0], [10, 10, 0], [0, 0, 0], [10, 10, 0], [0, 10, 0]]
    tris = [[0, 1, 2], [3, 4, 5]]
    subs = [{"verts": quad, "tris": tris}]
    assert collision_vert_count(subs) == 6
    out = simplify_collision(subs, target_tris=100000)   # no decimate, just weld
    assert collision_vert_count(out) == 4                 # 6 soup verts -> 4 corners
    assert len(out[0]["tris"]) == 2


def test_simplify_collision_empty_safe():
    from oblivion2vmf.model import simplify_collision
    assert simplify_collision([]) == []
    assert simplify_collision(None) is None
