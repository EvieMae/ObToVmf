"""Export the scene (terrain + placed props) to a Wavefront OBJ + MTL.

Geometry is written in the SAME recentered Hammer-unit frame the converter uses
for the real map, so a model built from this OBJ (e.g. decimated/baked in Blender)
drops straight back in as the 3D-skybox backdrop via ``--skybox-model-file``.

Materials are flat colours (terrain by land type, props grey) — enough to navigate
in Blender; re-texture or bake there.
"""
from __future__ import annotations

import os

from .land import GRID, CELL_SIZE, VERTEX_SPACING
from .textures import COLORS, classify


def _grid_indices(step):
    idx = list(range(0, GRID, step))
    if idx[-1] != GRID - 1:
        idx.append(GRID - 1)
    return idx


_MAX_COORD = 16384.0


def read_obj(path):
    """Parse a Wavefront OBJ (as written by export_scene_obj). Returns
    (verts, uvs, faces_by_material, mtl_colors, bounds). Faces are triangulated
    lists of (vert_idx, uv_idx|None) triples (0-based)."""
    verts, uvs = [], []
    faces = {}
    cur = "default"
    with open(path, "r", encoding="ascii", errors="replace") as f:
        for line in f:
            if line.startswith("v "):
                _, x, y, z = line.split()[:4]
                verts.append((float(x), float(y), float(z)))
            elif line.startswith("vt "):
                p = line.split()
                uvs.append((float(p[1]), float(p[2])))
            elif line.startswith("usemtl "):
                cur = line.split(None, 1)[1].strip()
            elif line.startswith("f "):
                idx = []
                for tok in line.split()[1:]:
                    a = tok.split("/")
                    vi = int(a[0]) - 1
                    ti = int(a[1]) - 1 if len(a) > 1 and a[1] else None
                    idx.append((vi, ti))
                fl = faces.setdefault(cur, [])
                for i in range(1, len(idx) - 1):          # fan-triangulate
                    fl.append((idx[0], idx[i], idx[i + 1]))
    colors = {}
    mtl = os.path.splitext(path)[0] + ".mtl"
    if os.path.isfile(mtl):
        name = None
        for line in open(mtl, "r", encoding="ascii", errors="replace"):
            if line.startswith("newmtl "):
                name = line.split(None, 1)[1].strip()
            elif line.startswith("Kd ") and name:
                p = line.split()
                colors[name] = tuple(int(float(c) * 255) for c in p[1:4])
    xs = [v[0] for v in verts]
    ys = [v[1] for v in verts]
    zs = [v[2] for v in verts]
    bounds = (min(xs), max(xs), min(ys), max(ys), min(zs), max(zs)) if verts else (0,) * 6
    return verts, uvs, faces, colors, bounds


def export_scene_obj(cells, placements, base_models, ltex, cell_textures, source,
                     scale, offsets, out_path, angle_sign=-1.0, yaw_offset=-90.0,
                     terrain_step=4, min_prop_size=0.0, max_prop_size=0.0, log=print):
    """Write ``out_path`` (.obj) + matching .mtl. Returns (n_verts, n_tris)."""
    from .nif import extract_meshes, available
    from .props import refr_matrix, placement_origin
    from .model import tex_slug

    x_off, y_off, z_off = offsets
    mtl_path = (out_path[:-4] if out_path.lower().endswith(".obj") else out_path) + ".mtl"
    materials = {}                       # name -> (r,g,b) 0..1
    nv = nt = 0

    with open(out_path, "w", encoding="ascii") as obj:
        obj.write("# oblivion2vmf scene export\nmtllib %s\n" % os.path.basename(mtl_path))

        def write_mesh(name, verts, uvs, tris, color):
            nonlocal nv, nt
            base = nv + 1                # OBJ is 1-indexed
            uvs = uvs or []
            for v in verts:
                obj.write("v %.3f %.3f %.3f\n" % (v[0], v[1], v[2]))
            for i in range(len(verts)):  # exactly one vt per vertex (keeps f indices aligned)
                u = uvs[i] if i < len(uvs) else (0.0, 0.0)
                obj.write("vt %.5f %.5f\n" % (u[0], u[1]))
            obj.write("usemtl %s\n" % name)
            for a, b, c in tris:
                obj.write("f %d/%d %d/%d %d/%d\n"
                          % (base + a, base + a, base + b, base + b, base + c, base + c))
            materials.setdefault(name, color)
            nv += len(verts)
            nt += len(tris)

        # --- terrain (downsampled heightmesh per cell) ---
        idx = _grid_indices(terrain_step)
        n = len(idx)
        for (cx, cy), grid in sorted(cells.items()):
            ox, oy = cx * CELL_SIZE, cy * CELL_SIZE
            verts, uvs, tris = [], [], []
            for r in idx:
                for c in idx:
                    wx, wy, h = ox + c * VERTEX_SPACING, oy + r * VERTEX_SPACING, grid[r][c]
                    verts.append(((wx - x_off) * scale, (wy - y_off) * scale, (h - z_off) * scale))
                    uvs.append((c / (GRID - 1.0), r / (GRID - 1.0)))
            for ri in range(n - 1):
                for ci in range(n - 1):
                    a = ri * n + ci
                    tris.append((a, a + 1, a + n + 1))
                    tris.append((a, a + n + 1, a + n))
            # colour by the cell's dominant land type
            col = COLORS["default"]
            info = cell_textures.get((cx, cy))
            if info:
                for q in info.values():
                    b = q.get("base")
                    if b:
                        col = COLORS.get(classify((ltex.get(b) or {}).get("icon", "")),
                                         COLORS["default"])
                        break
            write_mesh("terrain", verts, uvs, tris, tuple(c / 255.0 for c in col))

        # --- props (each placement baked to world space) ---
        prop_color = (0.55, 0.53, 0.5)
        mesh_cache = {}                  # modl -> (submeshes|None, max_extent_oblivion)
        n_props = n_skipped = 0
        if placements and base_models and source and available():
            for cell in sorted(placements):
                for p in placements[cell]:
                    modl = base_models.get(p["base"])
                    if not modl or modl.lower().endswith(".spt"):
                        continue
                    cached = mesh_cache.get(modl)
                    if cached is None:
                        try:
                            data = source.get_mesh(modl)
                            subs = extract_meshes(data) if data else None
                        except Exception:
                            subs = None
                        ext = 0.0
                        if subs:
                            xs = [v[0] for s in subs for v in s["verts"]]
                            ys = [v[1] for s in subs for v in s["verts"]]
                            zs = [v[2] for s in subs for v in s["verts"]]
                            ext = max(max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs))
                        cached = mesh_cache[modl] = (subs, ext)
                    subs, ext = cached
                    if not subs:
                        continue
                    size_hu = ext * scale * p["scale"]
                    # drop small props (rocks/bushes/clutter) and oversized distant-LOD
                    # landmark meshes (XXXL flags etc.)
                    if min_prop_size and size_hu < min_prop_size:
                        n_skipped += 1
                        continue
                    if max_prop_size and size_hu > max_prop_size:
                        n_skipped += 1
                        continue
                    origin = placement_origin(p["pos"], offsets, scale)
                    # drop props parked outside the map cube (distant-LOD parked at z=-30000)
                    if any(abs(c) > _MAX_COORD for c in origin):
                        n_skipped += 1
                        continue
                    R = refr_matrix(*p["rot"], sign=angle_sign, yaw_offset=yaw_offset)
                    s = scale * p["scale"]
                    for sub in subs:
                        wverts = []
                        for vx, vy, vz in sub["verts"]:
                            lx, ly, lz = vx * s, vy * s, vz * s
                            wverts.append((
                                origin[0] + R[0][0] * lx + R[0][1] * ly + R[0][2] * lz,
                                origin[1] + R[1][0] * lx + R[1][1] * ly + R[1][2] * lz,
                                origin[2] + R[2][0] * lx + R[2][1] * ly + R[2][2] * lz))
                        tex = sub.get("texture")
                        name = tex_slug(tex) if tex else "prop"
                        write_mesh(name, wverts, sub.get("uvs"), sub["tris"], prop_color)
                    n_props += 1

    with open(mtl_path, "w", encoding="ascii") as f:
        for name, (r, g, b) in sorted(materials.items()):
            f.write("newmtl %s\nKd %.3f %.3f %.3f\nKa 0 0 0\n\n" % (name, r, g, b))
    log("    OBJ: %d verts, %d tris, %d props (%d small props skipped), %d materials -> %s"
        % (nv, nt, n_props, n_skipped, len(materials), out_path))
    return nv, nt
