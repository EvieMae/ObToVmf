from oblivion2vmf import model


def _tri_submesh():
    return [{
        "verts": [(0.0, 0.0, 0.0), (10.0, 0.0, 0.0), (0.0, 10.0, 0.0)],
        "tris": [(0, 1, 2)],
        "normals": [(0.0, 0.0, 1.0)] * 3,
        "uvs": [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)],
        "texture": None,
    }]


def test_slugify():
    assert model.slugify("Architecture\\Anvil\\House01.nif") == "architecture_anvil_house01"
    assert model.slugify("rock.NIF") == "rock"
    assert model.slugify("a b/c.nif") == "a_b_c"


def test_write_smd_structure(tmp_path):
    p = tmp_path / "m.smd"
    tris = model.write_smd(_tri_submesh(), str(p), scale=2.0)
    assert tris == 1
    text = p.read_text()
    assert text.startswith("version 1")
    assert '0 "root" -1' in text
    assert text.rstrip().endswith("end")
    assert model.PLACEHOLDER_MATERIAL in text
    # 3 vertex lines under the triangles block
    vlines = [l for l in text.splitlines() if l.startswith("0 ") and l.count(" ") >= 8]
    assert len(vlines) == 3
    # scale applied: second vert x = 10 * 2 = 20
    assert any(l.split()[1] == "20.000000" for l in vlines)


def test_write_smd_winding_and_vflip(tmp_path):
    p = tmp_path / "m.smd"
    model.write_smd(_tri_submesh(), str(p), scale=1.0,
                    flip_winding=True, flip_v=True)
    lines = p.read_text().splitlines()
    vlines = [l for l in lines if l.startswith("0 ") and len(l.split()) == 9]
    # winding flip: emitted order is verts 0,2,1 -> 2nd line is original vert 2 (x=0,y=10)
    assert vlines[1].split()[1:4] == ["0.000000", "10.000000", "0.000000"]
    # v-flip: original v=0 -> 1.0
    assert vlines[0].split()[-1] == "1.000000"


def test_write_qc_with_collision(tmp_path):
    p = tmp_path / "m.qc"
    model.write_qc(str(p), "oblivion2vmf/rock.mdl", "rock.smd", surfaceprop="rock",
                   scale=1.0, collision_smd="rock_phys.smd")
    text = p.read_text()
    assert '$modelname "oblivion2vmf/rock.mdl"' in text
    assert "$staticprop" in text
    assert '$cdmaterials "models/oblivion2vmf"' in text
    assert '$collisionmodel "rock_phys.smd"' in text
    assert "$concave" in text


def test_write_qc_no_collision(tmp_path):
    p = tmp_path / "n.qc"
    model.write_qc(str(p), "oblivion2vmf/x.mdl", "x.smd")   # collision_smd default None
    assert "$collisionmodel" not in p.read_text()


def test_write_qc_lods(tmp_path):
    p = tmp_path / "b.qc"
    model.write_qc(str(p), "oblivion2vmf/bush.mdl", "bush.smd",
                   lods=[(12, "bush_lod1.smd"), (30, "bush_lod2.smd")])
    text = p.read_text()
    assert "$lod 12" in text and "$lod 30" in text
    assert 'replacemodel "bush.smd" "bush_lod1.smd"' in text
    assert 'replacemodel "bush.smd" "bush_lod2.smd"' in text


def test_decimate_welds_and_reduces():
    # pairs of near-coincident vertices weld at cell=0.5; widely-spaced ones stay
    verts = [(0, 0, 0), (0.1, 0, 0), (2, 0, 0), (2.1, 0, 0), (4, 0, 0), (0, 4, 0)]
    tris = [(0, 2, 5), (1, 3, 4)]
    subs = [{"verts": verts, "tris": tris, "normals": [], "uvs": [], "material": "m"}]
    dec = model._decimate(subs, cell=0.5)
    assert len(dec[0]["verts"]) < len(verts)     # the 0/0.1 and 2/2.1 pairs welded
    for a, b, c in dec[0]["tris"]:               # no degenerate triangles
        assert a != b and b != c and a != c
    assert dec[0]["material"] == "m"


def test_qem_preserves_topology_and_uvs():
    import pytest
    if not model._HAVE_QEM:
        pytest.skip("no QEM decimation library installed")
    # a subdivided plane: many connected tris with per-vertex UVs
    N = 12
    verts = [(x, y, 0.0) for y in range(N) for x in range(N)]
    uvs = [(x / (N - 1), y / (N - 1)) for y in range(N) for x in range(N)]
    tris = []
    for y in range(N - 1):
        for x in range(N - 1):
            a = y * N + x
            tris += [(a, a + 1, a + N), (a + 1, a + N + 1, a + N)]
    sub = {"verts": verts, "tris": tris, "uvs": uvs, "normals": [], "material": "m"}
    out = model._qem_submesh(sub, 0.5)
    assert out is not None
    assert len(out["tris"]) < len(tris)                  # reduced
    assert len(out["uvs"]) == len(out["verts"])          # UV per surviving vertex
    # silhouette preserved: x/y extent unchanged (clustering would shrink/distort)
    xs = [v[0] for v in out["verts"]]
    assert min(xs) == 0 and max(xs) == N - 1


def test_model_lods_skips_small_props():
    small = [{"verts": [(0, 0, 0), (1, 0, 0), (0, 1, 0)], "tris": [(0, 1, 2)],
              "normals": [], "uvs": [], "material": "m"}]
    assert model._model_lods(small, "x", ".", 1.0, False, True) == []   # < min_tris


def test_bush_lod_reduces_geometry():
    full = model._bush_submeshes("leaf", lod=0)
    far = model._bush_submeshes("leaf", lod=2)
    assert len(full) == 4 and len(far) == 1     # fewer cones at far LOD
    nfull = sum(len(s["tris"]) for s in full)
    nfar = sum(len(s["tris"]) for s in far)
    assert nfar < nfull


def test_write_collision_smd(tmp_path):
    parts = [
        ([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)], [(0, 1, 2)]),
        ([(5.0, 5.0, 0.0), (6.0, 5.0, 0.0), (5.0, 6.0, 0.0)], [(0, 1, 2)]),
    ]
    p = tmp_path / "phys.smd"
    model.write_collision_smd(parts, str(p), scale=2.0)
    text = p.read_text()
    assert text.startswith("version 1")
    vlines = [l for l in text.splitlines() if l.startswith("0 ") and len(l.split()) == 9]
    assert len(vlines) == 6                       # 2 tris x 3 verts
    assert any(l.split()[1] == "12.000000" for l in vlines)   # 6 * scale 2 = 12


def test_acd_module_importable():
    from oblivion2vmf import acd
    assert isinstance(acd.available(), bool)


def test_generate_tree_model(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    mats = tmp_path / "materials"
    path = model.generate_tree_model(str(work), str(mats), "oblivion2vmf",
                                     0.5625, None, None, compile_models=False)
    assert path == "models/oblivion2vmf/tree01.mdl"
    assert (work / "tree01.smd").exists()
    assert (work / "tree01_phys.smd").exists()       # trunk-only collision
    assert (work / "tree01.qc").exists()
    base = mats / "models" / "oblivion2vmf"
    assert (base / "tree_leaf.vtf").exists() and (base / "tree_bark.vtf").exists()
    smd = (work / "tree01.smd").read_text()
    assert "tree_bark" in smd and "tree_leaf" in smd


def test_tree_submeshes_geometry():
    subs = model._tree_submeshes()
    assert subs[0]["material"] == "tree_bark"          # trunk
    assert all(s["material"] == "tree_leaf" for s in subs[1:])   # canopy cones
    assert all(s["verts"] and s["tris"] for s in subs)


def test_placeholder_material(tmp_path):
    d = model.write_placeholder_material(str(tmp_path / "materials"))
    import os
    assert os.path.isfile(os.path.join(d, model.PLACEHOLDER_MATERIAL + ".vmt"))
    assert os.path.isfile(os.path.join(d, model.PLACEHOLDER_MATERIAL + ".vtf"))
    vmt = open(os.path.join(d, model.PLACEHOLDER_MATERIAL + ".vmt")).read()
    assert "VertexLitGeneric" in vmt


def test_load_json_tolerant_salvages_truncated_cache(tmp_path):
    from oblivion2vmf.model import load_json_tolerant, atomic_write_json
    full = {"a/b.nif": {"sig": "1", "model_path": "x"},
            "c/d.nif": {"sig": "2", "model_path": "y"},
            "e/f.nif": {"sig": "3", "model_path": "z"}}
    p = tmp_path / "cache.json"
    atomic_write_json(str(p), full)
    assert load_json_tolerant(str(p)) == full          # intact round-trips

    # simulate a kill mid-write: truncate inside the last entry
    text = p.read_text()
    cut = text.rfind("z")                              # mid last value
    p.write_text(text[:cut])
    salvaged = load_json_tolerant(str(p))
    assert "a/b.nif" in salvaged and "c/d.nif" in salvaged   # complete entries kept
    assert "e/f.nif" not in salvaged                   # partial tail dropped
    assert load_json_tolerant(str(tmp_path / "missing.json")) == {}
    (tmp_path / "junk.json").write_text("not json at all {{{")
    assert load_json_tolerant(str(tmp_path / "junk.json")) == {}


def test_split_bodies_under_vertex_limit():
    from oblivion2vmf.model import split_bodies, _split_submesh_by_verts
    # one submesh with 90 verts (30 tris), cap 40 -> must split into chunks < 40 verts
    verts = [[float(i), 0.0, 0.0] for i in range(90)]
    tris = [[i, i + 1, i + 2] for i in range(0, 90, 3)]
    sub = {"verts": verts, "tris": tris, "uvs": [], "normals": [], "material": "m"}
    bodies = split_bodies([sub], max_verts=40)
    assert len(bodies) >= 3
    for body in bodies:
        for s in body:
            assert len(s["verts"]) <= 40
    # geometry conserved: total triangles unchanged across the split
    assert sum(len(s["tris"]) for b in bodies for s in b) == 30
    # a small mesh stays a single body
    small = {"verts": verts[:9], "tris": tris[:3], "uvs": [], "normals": [], "material": "m"}
    assert len(split_bodies([small], max_verts=40)) == 1


def test_write_qc_multi_body(tmp_path):
    from oblivion2vmf.model import write_qc
    qc = tmp_path / "m.qc"
    bodies = [str(tmp_path / "m_b0.smd"), str(tmp_path / "m_b1.smd"),
              str(tmp_path / "m_b2.smd")]
    write_qc(str(qc), "oblivion2vmf/m.mdl", str(tmp_path / "m.smd"), bodies=bodies)
    txt = qc.read_text()
    assert txt.count("$body") == 3
    assert '$body "body0" "m_b0.smd"' in txt and '$body "body2" "m_b2.smd"' in txt
