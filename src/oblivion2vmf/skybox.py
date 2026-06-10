"""Bake the whole region into one low-poly terrain model + a baked top-down
texture, for use as a 3D-skybox backdrop.

A model sidesteps the two blockers a brush/displacement skybox hits: it doesn't
count against the 2048 displacement cap, and a point entity can't leak. Placed in
a small skybox room at uniformscale 1/sky_scale under a sky_camera, it renders 1:1
over the real terrain so the surrounding cells show as the distant horizon.
"""
from __future__ import annotations

import os
from collections import Counter

from .land import GRID, CELL_SIZE, VERTEX_SPACING
from .textures import COLORS, classify, write_vtf_rgb
from . import model as _model


def _cell_color(cell, cell_textures, ltex):
    """Average-ish terrain colour for a cell: the most common base land texture
    across its quadrants, mapped through classify()/COLORS."""
    info = cell_textures.get(cell)
    if not info:
        return COLORS["default"]
    cnt = Counter()
    for quad in info.values():
        b = quad.get("base")
        if b:
            cnt[b] += 1
    if not cnt:
        return COLORS["default"]
    fid = cnt.most_common(1)[0][0]
    icon = (ltex.get(fid) or {}).get("icon", "")
    return COLORS.get(classify(icon), COLORS["default"])


def _safe_mat(name):
    """studiomdl truncates material names at the first '.', and chokes on other
    punctuation — Blender adds '.003'-style suffixes. Map to [A-Za-z0-9_] only."""
    return "sky_" + "".join(c if (c.isalnum() or c == "_") else "_" for c in name)


def _write_obj_smd(verts, uvs, faces, path, mat_prefix="sky_"):
    """Write an SMD directly from parsed-OBJ data (global verts/uvs, faces grouped
    by material). Per-face flat normals; materials are prefixed so they don't clash
    with the real prop materials."""
    L = ["version 1", "nodes", '0 "root" -1', "end",
         "skeleton", "time 0", "0 0 0 0 0 0 0", "end", "triangles"]
    for mat, fl in faces.items():
        smat = _safe_mat(mat)
        for (a, ta), (b, tb), (c, tc) in fl:
            pa, pb, pc = verts[a], verts[b], verts[c]
            ux, uy, uz = pb[0] - pa[0], pb[1] - pa[1], pb[2] - pa[2]
            vx, vy, vz = pc[0] - pa[0], pc[1] - pa[1], pc[2] - pa[2]
            nx, ny, nz = uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx
            ln = (nx * nx + ny * ny + nz * nz) ** 0.5 or 1.0
            nx, ny, nz = nx / ln, ny / ln, nz / ln
            L.append(smat)
            for vi, ti in ((a, ta), (b, tb), (c, tc)):
                px, py, pz = verts[vi]
                u, v = uvs[ti] if (ti is not None and ti < len(uvs)) else (0.0, 0.0)
                L.append("0 %.3f %.3f %.3f %.4f %.4f %.4f %.5f %.5f"
                         % (px, py, pz, nx, ny, nz, u, v))
    L.append("end")
    with open(path, "w", encoding="ascii") as f:
        f.write("\n".join(L) + "\n")


def generate_skybox_from_obj(obj_path, work_dir, materials_root, studiomdl, gamedir,
                             compile_models=True, prefix=_model.MODEL_PREFIX,
                             sky_scale=16, log=print):
    """Compile an (already-edited) OBJ to models/<prefix>/skybox_imported.mdl and
    return (model_path, HU bounds) for placement as the 3D-skybox backdrop. Flat
    UnlitGeneric materials are generated from the .mtl colours (prefixed 'sky_' so
    they don't overwrite the real prop materials)."""
    from .obj import read_obj
    from .textures import write_vtf
    verts, uvs, faces, colors, bounds = read_obj(obj_path)
    if not verts or not faces:
        log("[warn] OBJ %s has no geometry." % obj_path)
        return None, None
    slug = "skybox_imported"
    if materials_root:
        d = os.path.join(materials_root, "models", prefix)
        os.makedirs(d, exist_ok=True)
        for mat in faces:
            col = colors.get(mat, (130, 130, 130))
            safe = _safe_mat(mat)
            write_vtf(os.path.join(d, safe + ".vtf"), col, size=8, noise=0, mips=False)
            with open(os.path.join(d, safe + ".vmt"), "w", encoding="ascii") as f:
                f.write('"UnlitGeneric"\n{\n\t"$basetexture" "models/%s/%s"\n'
                        '\t"$nocull" "1"\n}\n' % (prefix, safe))
    ref = os.path.join(work_dir, slug + ".smd")
    _write_obj_smd(verts, uvs, faces, ref)
    qc = os.path.join(work_dir, slug + ".qc")
    # compile at 1/sky_scale so the prop can be placed at scale 1.0 (GMod's 3D
    # skybox pass ignores prop_static uniformscale -> backdrop would float).
    _model.write_qc(qc, "%s/%s.mdl" % (prefix, slug), ref, scale=1.0 / sky_scale)
    n_tris = sum(len(fl) for fl in faces.values())
    if compile_models and studiomdl and gamedir:
        ok, clog = _model.run_studiomdl(studiomdl, gamedir, qc, timeout=600)
        if not ok:
            log("[warn] skybox OBJ compile failed:\n%s" % clog[-600:])
    log("    skybox from OBJ: %d verts, %d tris, %d materials" % (len(verts), n_tris, len(faces)))
    return "models/%s/%s.mdl" % (prefix, slug), bounds


def generate_skybox_terrain(cells, ltex, cell_textures, scale, offsets, work_dir,
                            materials_root, studiomdl, gamedir, compile_models=True,
                            prefix=_model.MODEL_PREFIX, step=8, tex_per_cell=4,
                            sky_scale=16, log=print):
    """Build/compile models/<prefix>/skybox_terrain.mdl from ``cells`` (the region
    + margin), using the REAL terrain ``offsets`` (x_off,y_off,z_off) so it aligns.
    Returns the model path, or None."""
    if not cells:
        return None
    x_off, y_off, z_off = offsets
    cxs = [c[0] for c in cells]
    cys = [c[1] for c in cells]
    bx0, bx1, by0, by1 = min(cxs), max(cxs), min(cys), max(cys)
    WX0, WX1 = bx0 * CELL_SIZE, (bx1 + 1) * CELL_SIZE
    WY0, WY1 = by0 * CELL_SIZE, (by1 + 1) * CELL_SIZE
    spanX, spanY = float(WX1 - WX0), float(WY1 - WY0)

    idx = list(range(0, GRID, step))
    if idx[-1] != GRID - 1:
        idx.append(GRID - 1)
    n = len(idx)

    # --- low-poly heightmesh (per-cell sample grid; border verts duplicated, which
    # is fine for a distant backdrop) in real recentered Hammer units ---------
    verts, uvs, tris = [], [], []
    for (cx, cy), grid in sorted(cells.items()):
        ox, oy = cx * CELL_SIZE, cy * CELL_SIZE
        base = len(verts)
        for r in idx:
            for c in idx:
                wx, wy, hgt = ox + c * VERTEX_SPACING, oy + r * VERTEX_SPACING, grid[r][c]
                verts.append(((wx - x_off) * scale, (wy - y_off) * scale, (hgt - z_off) * scale))
                uvs.append(((wx - WX0) / spanX, (wy - WY0) / spanY))
        for ri in range(n - 1):
            for ci in range(n - 1):
                a = base + ri * n + ci
                b, c2, d = a + 1, a + n, a + n + 1
                tris.append((a, b, d))
                tris.append((a, d, c2))
    sub = {"verts": verts, "tris": tris, "uvs": uvs,
           "normals": [(0.0, 0.0, 1.0)] * len(verts), "material": "skybox_terrain"}

    # --- baked top-down colour texture (classify colour per cell) -------------
    K = max(1, tex_per_cell)
    TW, TH = (bx1 - bx0 + 1) * K, (by1 - by0 + 1) * K
    pixels = []
    for ty in range(TH):
        wy = WY0 + (ty + 0.5) / TH * spanY
        ccy = int(wy // CELL_SIZE)
        for tx in range(TW):
            wx = WX0 + (tx + 0.5) / TW * spanX
            ccx = int(wx // CELL_SIZE)
            pixels.append(_cell_color((ccx, ccy), cell_textures, ltex))

    slug = "skybox_terrain"
    if materials_root:
        d = os.path.join(materials_root, "models", prefix)
        os.makedirs(d, exist_ok=True)
        write_vtf_rgb(os.path.join(d, slug + ".vtf"), pixels, TW, TH)
        with open(os.path.join(d, slug + ".vmt"), "w", encoding="ascii") as f:
            f.write('"UnlitGeneric"\n{\n\t"$basetexture" "models/%s/%s"\n'
                    '\t"$nocull" "1"\n}\n' % (prefix, slug))

    ref = os.path.join(work_dir, slug + ".smd")
    _model.write_smd([sub], ref, scale=1.0, flip_v=False)   # verts already in HU
    qc = os.path.join(work_dir, slug + ".qc")
    # compile at 1/sky_scale (prop placed at scale 1.0; uniformscale is ignored in
    # the 3D-skybox pass and would leave the backdrop floating).
    _model.write_qc(qc, "%s/%s.mdl" % (prefix, slug), ref, scale=1.0 / sky_scale)
    if compile_models and studiomdl and gamedir:
        ok, clog = _model.run_studiomdl(studiomdl, gamedir, qc)
        if not ok:
            log("[warn] skybox terrain compile failed:\n%s" % clog[-400:])
    log("    skybox terrain: %d tris, %dx%d baked texture (%d cells)"
        % (len(tris), TW, TH, len(cells)))
    xs = [v[0] for v in verts]
    ys = [v[1] for v in verts]
    zs = [v[2] for v in verts]
    bounds = (min(xs), max(xs), min(ys), max(ys), min(zs), max(zs))
    return "models/%s/%s.mdl" % (prefix, slug), bounds
