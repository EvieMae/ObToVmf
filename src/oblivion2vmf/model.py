"""NIF geometry -> Source static-prop (.mdl) pipeline.

Per unique Oblivion mesh: extract geometry (nif.py) -> write reference SMD + QC
-> compile with studiomdl into models/<prefix>/<slug>.mdl. A single flat
placeholder material is generated and shared by every model (geometry-first;
real .dds->.vtf texturing is a later pass).

Coordinate notes for the SMD: Source is Z-up, 1 unit = 1 inch, FRONT faces wind
CLOCKWISE, and UV origin is bottom-left. NIF is also Z-up but triangles wind
counter-clockwise (OpenGL) and UVs are top-left, so we reverse winding and flip V
by default. Both are "verify in-game" flips (FLIP_WINDING / FLIP_V).
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor

try:                                    # numpy underpins the QEM LOD path
    import numpy as _np
except Exception:                       # pragma: no cover
    _np = None
try:                                    # preferred: pyfqmr (has preserve_border)
    import pyfqmr as _fqmr
    _HAVE_FQMR = _np is not None
except Exception:                       # pragma: no cover
    _fqmr = None
    _HAVE_FQMR = False
try:                                    # fallback QEM: fast-simplification
    import fast_simplification as _fs
    _HAVE_FS = _np is not None
except Exception:                       # pragma: no cover
    _fs = None
    _HAVE_FS = False
_HAVE_QEM = _HAVE_FQMR or _HAVE_FS

from . import acd
from . import nif
from .dds import dds_to_vtf
from .textures import write_vtf

# Gamebryo/NIF is DirectX-based (CW front faces), same as Source -> do NOT reverse
# winding by default. NIF UV origin is top-left vs Source bottom-left -> flip V.
FLIP_WINDING = False
FLIP_V = True
PLACEHOLDER_MATERIAL = "oblivion2vmf_model"   # SMD material name -> .vmt of same name
MODEL_PREFIX = "oblivion2vmf"
PLACEHOLDER_COLOR = (150, 140, 120)
# Max concurrent CoACD decompositions regardless of --jobs (in-process C++; too
# many at once exhausts memory and hard-crashes the process).
ACD_MAX_CONCURRENCY = 6
# Bump when the NIF->SMD geometry output changes (so the build cache invalidates
# even though the NIF bytes are unchanged). 2 = transpose NIF node rotations;
# 3 = decimated prop LODs; 4 = QEM prop LODs; 5 = 4 QEM LOD stages (50/25/12.5/5%);
# 6 = LOD switch thresholds 3/10/15/20; 7 = distance-based thresholds, gentle near;
# 8 = weld + pyfqmr preserve_border; 9 = sooner switch distances;
# 10 = LOD distances scale with model size; 11 = reproject LOD normals;
# 12 = cap LOD count for big/multi-material models (GMod studiomdl LOD0 corruption).
GEOM_REV = 12
ACD_TIMEOUT = 50            # seconds before a CoACD attempt is abandoned
ACD_COARSE_THRESHOLD = 0.4  # fast retry threshold for meshes too slow at the fine one
# Convert a desired LOD switch distance (Hammer units) to studiomdl's $lod screen
# metric (= 100 / pixels-per-unit ~= distance/5.4 at 1080p/90deg FOV; higher-res
# monitors switch farther, i.e. keep more detail).
_LOD_METRIC_PER_HU = 1.0 / 5.4
_LOD_MIN_BASE_HU = 400.0     # floor so tiny props don't LOD right in your face

_STEAM_HINTS = [
    r"C:\Program Files (x86)\Steam\steamapps\common",
    r"D:\Steam\steamapps\common",
    r"D:\SteamLibrary\steamapps\common",
    r"E:\SteamLibrary\steamapps\common",
]


def slugify(modl_path):
    """Turn a MODL path like 'Architecture\\Anvil\\AnvilHouse01.nif' into a safe,
    unique-ish model filename stem."""
    s = modl_path.replace("/", "\\").lower()
    if s.endswith(".nif"):
        s = s[:-4]
    s = s.replace("\\", "_")
    s = re.sub(r"[^a-z0-9_]+", "_", s).strip("_")
    return s or "model"


def tex_slug(tex_path):
    """Material/texture stem from a NIF texture path, e.g.
    'textures\\clutter\\lowerclass\\Barrel.dds' -> 't_clutter_lowerclass_barrel'."""
    s = tex_path.replace("/", "\\").lower()
    if s.startswith("textures\\"):
        s = s[len("textures\\"):]
    if s.endswith(".dds"):
        s = s[:-4]
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    return ("t_" + s) if s else "t_default"


# --- SMD / QC ----------------------------------------------------------------

def write_smd(submeshes, path, scale=1.0, default_material=PLACEHOLDER_MATERIAL,
              flip_winding=FLIP_WINDING, flip_v=FLIP_V):
    out = ["version 1", "nodes", '0 "root" -1', "end",
           "skeleton", "time 0", "0 0 0 0 0 0 0", "end", "triangles"]
    tri_count = 0
    for sub in submeshes:
        verts, norms, uvs, tris = sub["verts"], sub["normals"], sub["uvs"], sub["tris"]
        material = sub.get("material") or default_material
        nv = len(verts)
        for tri in tris:
            a, b, c = (tri[0], tri[2], tri[1]) if flip_winding else (tri[0], tri[1], tri[2])
            if max(a, b, c) >= nv:
                continue
            out.append(material)
            for vi in (a, b, c):
                px, py, pz = verts[vi]
                nx, ny, nz = norms[vi] if norms else (0.0, 0.0, 1.0)
                u, v = uvs[vi] if uvs else (0.0, 0.0)
                if flip_v:
                    v = 1.0 - v
                out.append("0 %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f"
                           % (px * scale, py * scale, pz * scale, nx, ny, nz, u, v))
            tri_count += 1
    out.append("end")
    with open(path, "w", encoding="ascii") as f:
        f.write("\n".join(out) + "\n")
    return tri_count


def write_collision_smd(parts, path, scale=1.0, material="phys"):
    """Write convex parts [(verts, faces)] as a collision SMD (one element per
    part). studiomdl turns each separate element into one convex hull."""
    out = ["version 1", "nodes", '0 "root" -1', "end",
           "skeleton", "time 0", "0 0 0 0 0 0 0", "end", "triangles"]
    for verts, faces in parts:
        for a, b, c in faces:
            out.append(material)
            for vi in (a, b, c):
                x, y, z = verts[vi]
                out.append("0 %.6f %.6f %.6f 0 0 1 0 0" % (x * scale, y * scale, z * scale))
    out.append("end")
    with open(path, "w", encoding="ascii") as f:
        f.write("\n".join(out) + "\n")


def _aabb(subs):
    """Axis-aligned bounds of all submesh verts as (x0,y0,z0,x1,y1,z1) in MODEL
    units (pre-scale), or None when empty."""
    vs = [v for s in subs for v in s["verts"]]
    if not vs:
        return None
    xs = [v[0] for v in vs]; ys = [v[1] for v in vs]; zs = [v[2] for v in vs]
    return (min(xs), min(ys), min(zs), max(xs), max(ys), max(zs))


# the 12 triangles (as vertex-index triples) of a box given 8 corners ordered
# bottom 0-3 (z0) then top 4-7 (z1), each ring CCW in XY
_BOX_FACES = [(0, 1, 2), (0, 2, 3), (4, 6, 5), (4, 7, 6),
              (0, 4, 5), (0, 5, 1), (1, 5, 6), (1, 6, 2),
              (2, 6, 7), (2, 7, 3), (3, 7, 4), (3, 4, 0)]


def box_hull(bounds):
    """A single convex box (AABB) collision part [(verts, faces)] from bounds."""
    x0, y0, z0, x1, y1, z1 = bounds
    verts = [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
             (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]
    return (verts, list(_BOX_FACES))


# triangular-prism (wedge) faces for 6 verts: bottom rect 0-3, sloped top edge 4-5
_WEDGE_FACES = [(0, 1, 2), (0, 2, 3),          # bottom
                (0, 4, 5), (0, 5, 1),          # the two sloped quads -> ramp surface
                (3, 2, 5), (3, 5, 4),
                (0, 3, 4), (1, 5, 2)]          # the two triangular ends


def ramp_hull(bounds, axis="+x"):
    """A single convex wedge: a ramp that rises from z0 to z1 along ``axis`` (one
    of +x,-x,+y,-y). The player walks UP the slope instead of hitting a wall.
    Returns one collision part [(verts, faces)]."""
    x0, y0, z0, x1, y1, z1 = bounds
    # bottom rectangle (full footprint at z0), then the top edge is a single raised
    # line on the 'high' side -> a triangular prism. Pick which side is high.
    if axis in ("+x", "-x"):
        hi_x = x1 if axis == "+x" else x0
        verts = [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
                 (hi_x, y0, z1), (hi_x, y1, z1)]
    else:
        hi_y = y1 if axis == "+y" else y0
        verts = [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
                 (x0, hi_y, z1), (x1, hi_y, z1)]
    return (verts, list(_WEDGE_FACES))


def trap_hull(bounds, top_scale=0.5):
    """A trapezoidal prism (frustum): full-footprint bottom rectangle at z0, top
    rectangle at z1 scaled by ``top_scale`` about the footprint centre. One convex
    hull — same vertex topology as a box, so it reuses ``_BOX_FACES``."""
    x0, y0, z0, x1, y1, z1 = bounds
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    s = max(0.05, min(1.0, float(top_scale)))
    hx, hy = (x1 - x0) / 2.0 * s, (y1 - y0) / 2.0 * s
    verts = [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
             (cx - hx, cy - hy, z1), (cx + hx, cy - hy, z1),
             (cx + hx, cy + hy, z1), (cx - hx, cy + hy, z1)]
    return (verts, list(_BOX_FACES))


def cylinder_hull(bounds, sides=12):
    """An n-sided prism approximating the cylinder inscribed in ``bounds``: the
    XY-footprint ellipse extruded from z0 to z1. One convex hull; sides clamps to
    >= 3 so the part always has volume."""
    x0, y0, z0, x1, y1, z1 = bounds
    n = max(3, int(sides))
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    rx, ry = (x1 - x0) / 2.0, (y1 - y0) / 2.0
    ring = [(cx + rx * math.cos(2 * math.pi * i / n),
             cy + ry * math.sin(2 * math.pi * i / n)) for i in range(n)]
    verts = [(x, y, z0) for x, y in ring] + [(x, y, z1) for x, y in ring]
    faces = []
    for i in range(n):                      # side quads (two tris each)
        j = (i + 1) % n
        faces.append((i, j, n + j))
        faces.append((i, n + j, n + i))
    for i in range(1, n - 1):               # caps as fans (reference every vert)
        faces.append((0, i + 1, i))                          # bottom
        faces.append((n, n + i, n + i + 1))                  # top
    return (verts, faces)


def plane_hull(bounds, thickness=2.0):
    """A thin slab over the bounds XY footprint, from z0 to z0+thickness. Convex
    physics pieces need volume — a zero-thickness quad is degenerate for
    studiomdl — so 'plane' really means 'very flat box'."""
    x0, y0, z0, x1, y1, _ = bounds
    t = max(0.1, float(thickness))
    return box_hull((x0, y0, z0, x1, y1, z0 + t))


def _rotate_verts(verts, rot, centre):
    """Rotate verts about ``centre`` by Euler angles rot=[rx,ry,rz] in degrees,
    intrinsic XYZ order, right-handed (== world-axis Z then Y then X). Pure math
    so hull specs don't drag in numpy."""
    rx, ry, rz = (math.radians(float(a)) for a in rot)
    cx_, sx = math.cos(rx), math.sin(rx)
    cy_, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    ox, oy, oz = centre
    out = []
    for x, y, z in verts:
        x, y, z = x - ox, y - oy, z - oz
        x, y = x * cz - y * sz, x * sz + y * cz          # about world Z
        x, z = x * cy_ + z * sy, -x * sy + z * cy_       # about world Y
        y, z = y * cx_ - z * sx, y * sx + z * cx_        # about world X
        out.append((x + ox, y + oy, z + oz))
    return out


def hull_from_spec(spec):
    """One convex collision part from a hull spec. Accepts the legacy 6-float box
    list, or a dict {"type": "box"|"wedge"|"trap"|"cylinder"|"plane",
    "bounds": [x0,y0,z0,x1,y1,z1], "axis": "+x" (wedge), "top_scale": 0.5 (trap),
    "sides": 12 (cylinder), "thickness": 2.0 (plane), "rot": [rx,ry,rz] degrees
    (any dict shape, about the bounds centre)}. Returns (verts, faces) or None
    for a malformed spec."""
    if isinstance(spec, dict):
        b = spec.get("bounds") or []
        if len(b) != 6:
            return None
        b = tuple(float(c) for c in b)
        t = (spec.get("type") or "box").lower()
        if t == "wedge":
            part = ramp_hull(b, spec.get("axis", "+x"))
        elif t in ("trap", "trapezoid", "trapezium"):
            part = trap_hull(b, spec.get("top_scale", 0.5))
        elif t == "cylinder":
            part = cylinder_hull(b, spec.get("sides", 12))
        elif t == "plane":
            part = plane_hull(b, spec.get("thickness", 2.0))
        else:
            part = box_hull(b)
        rot = spec.get("rot")
        if rot:
            try:
                rot = [float(a) for a in rot]
                if len(rot) != 3:
                    return None
            except (TypeError, ValueError):
                return None                  # malformed rot = malformed spec
            if any(rot):
                centre = ((b[0] + b[3]) / 2.0, (b[1] + b[4]) / 2.0,
                          (b[2] + b[5]) / 2.0)
                part = (_rotate_verts(part[0], rot, centre), part[1])
        return part
    if len(spec) == 6:
        return box_hull(tuple(float(c) for c in spec))
    return None


def write_qc(path, modelname, ref_smd, surfaceprop="default", scale=1.0,
             cdmaterials="models/" + MODEL_PREFIX, collision_smd=None, max_convex=64,
             lods=None):
    ref = os.path.basename(ref_smd)
    lines = [
        '$modelname "%s"' % modelname,
        "$staticprop",
        "$scale %s" % _num(scale),       # MUST precede $body -- studiomdl applies
        '$body "body" "%s"' % ref,        # $scale only to bodies declared after it
        '$cdmaterials "%s"' % cdmaterials,
        '$surfaceprop "%s"' % surfaceprop,
        '$sequence "idle" "%s" fps 1' % ref,
    ]
    # LOD stages: (switch_threshold, lod_smd_path). studiomdl swaps the body for
    # the lower-poly mesh past the threshold -> far foliage costs far fewer polys.
    for thresh, lod_smd in (lods or []):
        lines += [
            "$lod %d" % thresh,
            "{",
            '\treplacemodel "%s" "%s"' % (ref, os.path.basename(lod_smd)),
            "}",
        ]
    if collision_smd:
        # Concave collision from a (possibly pre-decomposed) SMD. High piece cap
        # + studiomdl -fullcollide keeps open structures from sealing shut.
        lines += [
            '$collisionmodel "%s"' % os.path.basename(collision_smd),
            "{",
            "\t$concave",
            "\t$maxconvexpieces %d" % max_convex,
            "\t$automass",
            "}",
        ]
    with open(path, "w", encoding="ascii") as f:
        f.write("\n".join(lines) + "\n")


def _num(v):
    f = float(v)
    return str(int(f)) if f == int(f) else ("%.4f" % f).rstrip("0").rstrip(".")


def write_placeholder_material(materials_root, prefix=MODEL_PREFIX,
                               name=PLACEHOLDER_MATERIAL):
    """Write materials/models/<prefix>/<name>.vmt + .vtf (VertexLitGeneric)."""
    d = os.path.join(materials_root, "models", prefix)
    os.makedirs(d, exist_ok=True)
    write_vtf(os.path.join(d, name + ".vtf"), PLACEHOLDER_COLOR, size=64)
    with open(os.path.join(d, name + ".vmt"), "w", encoding="ascii") as f:
        f.write(_model_vmt(prefix, name, name))
    return d


def _model_vmt(prefix, name, basetexture):
    return ('"VertexLitGeneric"\n{\n'
            '\t"$basetexture" "models/%s/%s"\n'
            '\t"$surfaceprop" "default"\n}\n' % (prefix, basetexture))


def write_textures(materials_root, needed, source, prefix=MODEL_PREFIX):
    """Convert each needed Oblivion texture to .vtf + write its .vmt. ``needed``
    maps material slug -> NIF texture path. Returns (converted, fallback)."""
    d = os.path.join(materials_root, "models", prefix)
    os.makedirs(d, exist_ok=True)
    ok = fail = 0
    for slug, tex in sorted(needed.items()):
        dds = source.get_texture(tex)
        vtf = dds_to_vtf(dds) if dds else None
        if vtf:
            with open(os.path.join(d, slug + ".vtf"), "wb") as f:
                f.write(vtf)
            base = slug
            ok += 1
        else:
            base = PLACEHOLDER_MATERIAL          # resolves to the grey placeholder .vtf
            fail += 1
        with open(os.path.join(d, slug + ".vmt"), "w", encoding="ascii") as f:
            f.write(_model_vmt(prefix, slug, base))
    return ok, fail


# --- mesh decimation for prop LODs -------------------------------------------

def _decimate(subs, cell):
    """Vertex-clustering decimation: snap each vertex to a ``cell``-sized grid,
    weld coincident vertices, and drop triangles that collapse. Cheap and robust
    (no quadric metric) — good enough for distant LODs. Preserves per-submesh
    material; UV/normal taken from each cluster's first vertex."""
    out = []
    for s in subs:
        verts, tris = s["verts"], s["tris"]
        uvs = s.get("uvs") or []
        norms = s.get("normals") or []
        cluster = {}
        nv, nuv, nn, remap = [], [], [], []
        for i, v in enumerate(verts):
            key = (round(v[0] / cell), round(v[1] / cell), round(v[2] / cell))
            idx = cluster.get(key)
            if idx is None:
                idx = cluster[key] = len(nv)
                nv.append(v)
                nuv.append(uvs[i] if i < len(uvs) else (0.0, 0.0))
                nn.append(norms[i] if i < len(norms) else (0.0, 0.0, 1.0))
            remap.append(idx)
        nt = []
        for a, b, c in tris:
            ra, rb, rc = remap[a], remap[b], remap[c]
            if ra != rb and rb != rc and ra != rc:
                nt.append((ra, rb, rc))
        if nt:
            out.append({"verts": nv, "tris": nt, "normals": nn, "uvs": nuv,
                        "material": s.get("material")})
    return out


WELD_EPS = 0.25        # model-space units: merge duplicate verts to connect soup
PYFQMR_AGG = 7         # pyfqmr aggressiveness (lower = better quality, slower)


def _weld(verts, tris, eps=WELD_EPS):
    """Merge coincident vertices (snap to an ``eps`` grid) so Oblivion's
    triangle-soup meshes (each panel a separate piece with duplicate verts) become
    a connected surface. This is the key prerequisite for clean decimation: it
    turns thousands of fake 'boundary' edges into real shared edges. Returns
    (welded_verts, welded_tris)."""
    keys = {}
    remap = [0] * len(verts)
    nv = []
    inv = eps if eps > 1e-9 else 1.0
    for i, v in enumerate(verts):
        k = (round(v[0] / inv), round(v[1] / inv), round(v[2] / inv))
        j = keys.get(k)
        if j is None:
            j = keys[k] = len(nv)
            nv.append(v)
        remap[i] = j
    nt = []
    for a, b, c in tris:
        ra, rb, rc = remap[a], remap[b], remap[c]
        if ra != rb and rb != rc and ra != rc:
            nt.append((ra, rb, rc))
    return nv, nt


def _nearest_attrs(out_pts, orig_pts, uvs, normals):
    """Reproject UVs and normals: for each output vertex, copy the attributes of
    the nearest original vertex (pyfqmr drops both, and its returned normals don't
    line up). One chunked numpy nearest-neighbour pass for both. Returns
    (uvs_out, normals_out)."""
    P = _np.asarray(orig_pts, dtype=_np.float32)
    O = _np.asarray(out_pts, dtype=_np.float32)
    has_uv, has_n = bool(uvs), bool(normals)
    U = _np.asarray(uvs, dtype=_np.float32) if has_uv else None
    N = _np.asarray(normals, dtype=_np.float32) if has_n else None
    out_uv, out_n = [], []
    for i in range(0, len(O), 512):
        chunk = O[i:i + 512]
        d = ((chunk[:, None, :] - P[None, :, :]) ** 2).sum(2)
        idx = d.argmin(1)
        if has_uv:
            out_uv.extend(tuple(uv) for uv in U[idx].tolist())
        if has_n:
            out_n.extend(tuple(n) for n in N[idx].tolist())
    if not has_uv:
        out_uv = [(0.0, 0.0)] * len(O)
    return out_uv, out_n


def _qem_submesh(sub, reduction):
    """Decimate one (material-merged) submesh, preserving silhouette. ``reduction``
    is the fraction of triangles to REMOVE. Pipeline: weld duplicate verts ->
    pyfqmr quadric collapse with preserve_border (keeps open-shell edges so it
    doesn't explode) -> reproject UVs from nearest original vertex. Falls back to
    fast-simplification if pyfqmr is absent. Returns a submesh dict or None."""
    tris = sub["tris"]
    if not _HAVE_QEM or len(tris) < 24:
        return None
    if _HAVE_FQMR:
        wv, wf = _weld(sub["verts"], tris)
        if len(wf) < 8:
            return None
        V = _np.asarray(wv, dtype=_np.float64)
        F = _np.asarray(wf, dtype=_np.int32)
        target = max(4, int(len(wf) * (1.0 - reduction)))
        try:
            sim = _fqmr.Simplify()
            sim.setMesh(V, F)
            sim.simplify_mesh(target_count=target, aggressiveness=PYFQMR_AGG,
                              preserve_border=True, verbose=0)
            po, fo, no = sim.getMesh()
        except Exception:
            return None
        if len(fo) == 0 or len(fo) >= len(tris):
            return None
        # pyfqmr drops UVs and its normals don't line up -> reproject both from the
        # nearest original vertex (missing normals would flat-shade the whole LOD).
        out_uv, out_n = _nearest_attrs(po, sub["verts"], sub.get("uvs") or [],
                                       sub.get("normals") or [])
        return {"verts": [tuple(map(float, p)) for p in po.tolist()],
                "tris": [tuple(map(int, f)) for f in fo.tolist()],
                "uvs": out_uv, "normals": out_n, "material": sub.get("material")}
    # fallback: fast-simplification (no border preservation)
    pts = _np.asarray(sub["verts"], dtype=_np.float32)
    faces = _np.asarray(tris, dtype=_np.int32)
    try:
        _, _, collapses = _fs.simplify(pts, faces, reduction, return_collapses=True)
        po, fo, imap = _fs.replay_simplification(pts, faces, collapses)
    except Exception:
        return None
    if len(fo) == 0 or len(fo) >= len(tris):
        return None
    nv = len(po)
    uvs, norms = sub.get("uvs") or [], sub.get("normals") or []
    out_uv = [(0.0, 0.0)] * nv
    out_n = [(0.0, 0.0, 1.0)] * nv
    for i in range(len(pts)):
        d = int(imap[i])
        if 0 <= d < nv:
            if i < len(uvs):
                out_uv[d] = uvs[i]
            if i < len(norms):
                out_n[d] = norms[i]
    return {"verts": [tuple(map(float, p)) for p in po.tolist()],
            "tris": [tuple(map(int, f)) for f in fo.tolist()],
            "uvs": out_uv, "normals": out_n, "material": sub.get("material")}


def _merge_by_material(subs):
    """Concatenate submeshes that share a material into one mesh each. QEM only
    collapses connected edges, so merging disconnected same-material parts is safe
    and lets the decimator reduce much deeper (small per-strip submeshes alone hit
    a floor) while preserving material boundaries."""
    groups = {}
    order = []
    for s in subs:
        mat = s.get("material")
        g = groups.get(mat)
        if g is None:
            g = groups[mat] = {"verts": [], "tris": [], "uvs": [], "normals": []}
            order.append(mat)
        base = len(g["verts"])
        nv = len(s["verts"])
        uvs = s.get("uvs") or []
        norms = s.get("normals") or []
        g["verts"].extend(s["verts"])
        g["uvs"].extend(uvs[i] if i < len(uvs) else (0.0, 0.0) for i in range(nv))
        g["normals"].extend(norms[i] if i < len(norms) else (0.0, 0.0, 1.0) for i in range(nv))
        g["tris"].extend((a + base, b + base, c + base) for a, b, c in s["tris"])
    return [{"verts": groups[m]["verts"], "tris": groups[m]["tris"],
             "uvs": groups[m]["uvs"], "normals": groups[m]["normals"], "material": m}
            for m in order]


def _decimate_qem(subs, reduction):
    """Merge by material, then QEM-decimate each group; keep a group unchanged if
    it can't reduce."""
    return [(_qem_submesh(g, reduction) or g) for g in _merge_by_material(subs)]


def _model_lods(subs, slug, work_dir, scale, flip_winding, flip_v, min_tris=300):
    """Generate LOD SMDs for a prop. Uses quadric edge-collapse (fast-simplification)
    when available — topology-preserving, so buildings don't collapse — else falls
    back to gentle vertex clustering. Returns [(threshold, smd_path)]."""
    total = sum(len(s["tris"]) for s in subs)
    if total < min_tris:
        return []
    # LOD switch distances SCALE WITH MODEL SIZE so a switch happens at a consistent
    # on-screen size: a small crate LODs up close, a huge castle keeps full detail
    # until it's genuinely far (otherwise a building bigger than a flat switch
    # distance shows its LOD while you stand next to it). distance = base * mult,
    # base = max(model size HU, floor). The $lod value is a screen metric
    # (~ distance/5.4 at 1080p/90deg).
    xs = [v[0] for s in subs for v in s["verts"]]
    ys = [v[1] for s in subs for v in s["verts"]]
    zs = [v[2] for s in subs for v in s["verts"]]
    size_hu = max(max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs)) * scale
    base = max(size_hu, _LOD_MIN_BASE_HU)
    # GMod's studiomdl corrupts LOD0 when a model has multiple meshes (materials)
    # AND multiple LODs (worse with more geometry) -- garrysmod-issues #4832. So
    # big/complex models get a SINGLE LOD (keep ~25%); simple ones get two (50/25%).
    nmat = len(set(s.get("material") for s in subs))
    if total > 3000 or nmat >= 4:
        stages = [(0.75, 2.0)]                          # one LOD, keep ~25%
    else:
        stages = [(0.5, 2.0), (0.75, 4.0)]              # two LODs (50% then 25%)
    levels = [(red, max(1, round(base * mult * _LOD_METRIC_PER_HU))) for red, mult in stages]
    if not _HAVE_QEM:
        # crude fallback: vertex clustering by grid (gentler than before; 1 stage)
        xs = [v[0] for s in subs for v in s["verts"]]
        ys = [v[1] for s in subs for v in s["verts"]]
        zs = [v[2] for s in subs for v in s["verts"]]
        ext = max(max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs)) or 1.0
        dec = _decimate(subs, ext / 28.0)
        dt = sum(len(s["tris"]) for s in dec)
        if dt < 8 or dt >= total * 0.9:
            return []
        path = os.path.join(work_dir, "%s_lod1.smd" % slug)
        write_smd(dec, path, scale=scale, flip_winding=flip_winding, flip_v=flip_v)
        return [(20, path)]
    lods = []
    prev = total
    for lvl, (reduction, thresh) in enumerate(levels, start=1):
        dec = _decimate_qem(subs, reduction)
        dt = sum(len(s["tris"]) for s in dec)
        # skip a stage that doesn't reduce meaningfully past the previous one (QEM
        # floors out — a building can't always shrink to 5% without falling apart)
        if dt < 8 or dt >= prev * 0.9:
            continue
        path = os.path.join(work_dir, "%s_lod%d.smd" % (slug, lvl))
        write_smd(dec, path, scale=scale, flip_winding=flip_winding, flip_v=flip_v)
        lods.append((thresh, path))
        prev = dt
    return lods


# --- generated geometric tree (placeholder for SpeedTree .spt) ---------------

def _box_mesh(z0, z1, half):
    v = [(-half, -half, z0), (half, -half, z0), (half, half, z0), (-half, half, z0),
         (-half, -half, z1), (half, -half, z1), (half, half, z1), (-half, half, z1)]
    t = [(0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7),
         (0, 1, 5), (0, 5, 4), (1, 2, 6), (1, 6, 5),
         (2, 3, 7), (2, 7, 6), (3, 0, 4), (3, 4, 7)]
    return v, t


def _cylinder_mesh(z0, z1, r0, r1, sides=8, vtile=4.0):
    """Tapered cylinder (r0 at bottom -> r1 at top) with bark UVs wrapping around.
    Returns (verts, tris, uvs). Duplicate seam ring vertex for a clean UV wrap."""
    verts, uvs = [], []
    for i in range(sides + 1):
        a = 2 * math.pi * i / sides
        c, s = math.cos(a), math.sin(a)
        u = i / sides
        verts.append((r0 * c, r0 * s, z0)); uvs.append((u, 0.0))
        verts.append((r1 * c, r1 * s, z1)); uvs.append((u, vtile))
    tris = []
    for i in range(sides):
        b0, t0, b1, t1 = i * 2, i * 2 + 1, (i + 1) * 2, (i + 1) * 2 + 1
        tris.append((b0, t0, t1)); tris.append((b0, t1, b1))
    return verts, tris, uvs


def _cone_mesh(z0, z1, r, sides=8):
    v = [(r * math.cos(2 * math.pi * i / sides), r * math.sin(2 * math.pi * i / sides), z0)
         for i in range(sides)]
    apex = len(v)
    v.append((0.0, 0.0, z1))
    t = [(i, (i + 1) % sides, apex) for i in range(sides)]
    return v, t


def _cone_uv_mesh(z0, z1, r, sides=10, tile=1.0):
    """Cone with UVs + a closed base, so a leaf texture maps cleanly. Returns
    (verts, tris, uvs)."""
    verts, uvs = [], []
    for i in range(sides):
        a = 2 * math.pi * i / sides
        verts.append((r * math.cos(a), r * math.sin(a), z0))
        uvs.append((tile * i / sides, 0.0))
    apex = len(verts)
    verts.append((0.0, 0.0, z1)); uvs.append((0.5 * tile, tile))
    base = len(verts)
    verts.append((0.0, 0.0, z0)); uvs.append((0.5 * tile, 0.0))   # base center
    tris = []
    for i in range(sides):
        j = (i + 1) % sides
        tris.append((i, j, apex))        # side
        tris.append((j, i, base))        # underside (so it's not hollow from below)
    return verts, tris, uvs


def _bush_submeshes(leaf_mat, r=150.0, h=300.0, lod=0):
    """A solid 3D bush: overlapping leaf-textured cones forming a rounded blob.
    ``lod`` 0=4 cones (full), 1=2 cones, 2=1 cone (far)."""
    cones = [
        (0.0,  0.0,  0.0, h,        r),        # central, tallest
        (r * 0.45, 0.0,        -10.0, h * 0.62, r * 0.7),
        (-r * 0.4, r * 0.35,   -10.0, h * 0.55, r * 0.66),
        (-r * 0.35, -r * 0.4,  -10.0, h * 0.6,  r * 0.7),
    ]
    keep = {0: 4, 1: 2, 2: 1}.get(lod, 4)
    subs = []
    for ox, oy, z0, z1, rr in cones[:keep]:
        cv, ct, cuv = _cone_uv_mesh(z0, z1, rr, sides=(10 if lod == 0 else 6))
        cv = [(x + ox, y + oy, z) for (x, y, z) in cv]
        subs.append({"verts": cv, "tris": ct, "normals": [], "uvs": cuv,
                     "material": leaf_mat})
    return subs


def generate_bush_model(slug, leaf_rel, work_dir, materials_root, source,
                        scale, studiomdl, gamedir, compile_models, log=print):
    """Build/compile a self-contained 3D bush (leaf-textured cones) for a shrub
    species, using the real Oblivion leaf texture (alpha-tested). No billboards,
    no external content. Two LOD stages keep distant bushes cheap. Returns the
    model path."""
    leaf_slug = slug + "_leaf"
    if materials_root:
        d = os.path.join(materials_root, "models", MODEL_PREFIX)
        os.makedirs(d, exist_ok=True)
        dds = source.get_texture(leaf_rel) if (source and leaf_rel) else None
        vtf = dds_to_vtf(dds) if dds else None
        if vtf:
            with open(os.path.join(d, leaf_slug + ".vtf"), "wb") as f:
                f.write(vtf)
        else:
            write_vtf(os.path.join(d, leaf_slug + ".vtf"), (58, 92, 44))
        with open(os.path.join(d, leaf_slug + ".vmt"), "w", encoding="ascii") as f:
            f.write(_leaf_vmt(MODEL_PREFIX, leaf_slug))
    ref = os.path.join(work_dir, slug + ".smd")
    write_smd(_bush_submeshes(leaf_slug, lod=0), ref, scale=scale, flip_v=False)
    lods = []
    for lvl, thresh in ((1, 12), (2, 30)):
        lp = os.path.join(work_dir, "%s_lod%d.smd" % (slug, lvl))
        write_smd(_bush_submeshes(leaf_slug, lod=lvl), lp, scale=scale, flip_v=False)
        lods.append((thresh, lp))
    qc = os.path.join(work_dir, slug + ".qc")
    write_qc(qc, "%s/%s.mdl" % (MODEL_PREFIX, slug), ref, scale=1.0, lods=lods)  # no collision
    if compile_models and studiomdl and gamedir:
        ok, clog = run_studiomdl(studiomdl, gamedir, qc)
        if not ok:
            log("[warn] bush %s compile failed:\n%s" % (slug, clog[-300:]))
    return "models/%s/%s.mdl" % (MODEL_PREFIX, slug)


def _tree_submeshes(trunk_h=400.0, top_h=1600.0, trunk_r=45.0, canopy_r=380.0):
    """A simple tree: a trunk box + three stacked canopy cones."""
    subs = []
    tv, tt, tuv = _cylinder_mesh(0.0, trunk_h, trunk_r, trunk_r * 0.6, sides=8)
    subs.append({"verts": tv, "tris": tt, "normals": [], "uvs": tuv, "material": "tree_bark"})
    span = top_h - trunk_h
    for f0, f1, fr in ((0.0, 0.60, 1.0), (0.30, 0.85, 0.72), (0.60, 1.0, 0.42)):
        cv, ct = _cone_mesh(trunk_h + span * f0, trunk_h + span * f1, canopy_r * fr)
        subs.append({"verts": cv, "tris": ct, "normals": [], "uvs": [], "material": "tree_leaf"})
    return subs


def generate_tree_model(work_dir, materials_root, prefix, scale,
                        studiomdl, gamedir, compile_models, log=print):
    """Build/compile one generic tree .mdl (reused for all SpeedTree placements).
    Returns its model path."""
    subs = _tree_submeshes()
    slug = "tree01"
    ref = os.path.join(work_dir, slug + ".smd")
    write_smd(subs, ref, scale=scale)
    phys = os.path.join(work_dir, slug + "_phys.smd")
    write_smd([subs[0]], phys, scale=scale)        # trunk-only collision (walk under canopy)
    qc = os.path.join(work_dir, slug + ".qc")
    write_qc(qc, "%s/%s.mdl" % (prefix, slug), ref, scale=1.0, collision_smd=phys)
    if materials_root:
        d = os.path.join(materials_root, "models", prefix)
        os.makedirs(d, exist_ok=True)
        write_vtf(os.path.join(d, "tree_bark.vtf"), (96, 64, 40))
        write_vtf(os.path.join(d, "tree_leaf.vtf"), (58, 92, 44))
        for nm, sp in (("tree_bark", "wood"), ("tree_leaf", "grass")):
            with open(os.path.join(d, nm + ".vmt"), "w", encoding="ascii") as f:
                f.write(_tree_vmt(prefix, nm, sp))
    if compile_models and studiomdl and gamedir:
        ok, clog = run_studiomdl(studiomdl, gamedir, qc)
        if not ok:
            log("[warn] tree model compile failed:\n%s" % clog[-400:])
    return "models/%s/%s.mdl" % (prefix, slug)


def _tree_vmt(prefix, name, surfaceprop="wood"):
    # two-sided: the generated trunk/canopy geometry is simple and not guaranteed
    # to wind the way Source expects, so $nocull keeps it from being culled away.
    return ('"VertexLitGeneric"\n{\n'
            '\t"$basetexture" "models/%s/%s"\n'
            '\t"$nocull" "1"\n\t"$surfaceprop" "%s"\n}\n' % (prefix, name, surfaceprop))


def _leaf_vmt(prefix, name):
    return ('"VertexLitGeneric"\n{\n'
            '\t"$basetexture" "models/%s/%s"\n'
            '\t"$alphatest" "1"\n\t"$alphatestreference" "0.5"\n'
            '\t"$nocull" "1"\n\t"$surfaceprop" "grass"\n}\n' % (prefix, name))


def _billboard_tree_submeshes(bark_mat, leaf_mat, trunk_h=350.0, top=1500.0,
                              trunk_r=42.0, canopy_w=950.0, canopy_zb=220.0, tile=3.0):
    """Trunk box (bark) + 3 crossed vertical quads (alpha-tested leaf cluster)."""
    subs = []
    tv, tt, tuv = _cylinder_mesh(0.0, trunk_h, trunk_r, trunk_r * 0.6, sides=8)
    subs.append({"verts": tv, "tris": tt, "normals": [], "uvs": tuv, "material": bark_mat})
    hw = canopy_w / 2.0
    for k in range(3):
        a = math.pi * k / 3.0
        cx, cy = math.cos(a), math.sin(a)
        v = [(-hw * cx, -hw * cy, canopy_zb), (hw * cx, hw * cy, canopy_zb),
             (hw * cx, hw * cy, top), (-hw * cx, -hw * cy, top)]
        uv = [(0.0, 0.0), (tile, 0.0), (tile, tile), (0.0, tile)]
        subs.append({"verts": v, "tris": [(0, 1, 2), (0, 2, 3)],
                     "normals": [], "uvs": uv, "material": leaf_mat})
    return subs


def generate_billboard_tree(slug, leaf_rel, bark_rel, work_dir, materials_root, source,
                            scale, studiomdl, gamedir, compile_models, log=print):
    """Build/compile a crossed-billboard tree using the real Oblivion leaf (+bark)
    textures. Returns its model path."""
    leaf_slug, bark_slug = slug + "_leaf", slug + "_bark"
    bark_mat = "tree_bark"
    if materials_root:
        d = os.path.join(materials_root, "models", MODEL_PREFIX)
        os.makedirs(d, exist_ok=True)
        # leaf (alpha-tested)
        dds = source.get_texture(leaf_rel) if source else None
        vtf = dds_to_vtf(dds) if dds else None
        if vtf:
            with open(os.path.join(d, leaf_slug + ".vtf"), "wb") as f:
                f.write(vtf)
        else:
            write_vtf(os.path.join(d, leaf_slug + ".vtf"), (58, 92, 44))
        with open(os.path.join(d, leaf_slug + ".vmt"), "w", encoding="ascii") as f:
            f.write(_leaf_vmt(MODEL_PREFIX, leaf_slug))
        # bark (opaque), else fall back to a flat brown
        bdds = source.get_texture(bark_rel) if (source and bark_rel) else None
        bvtf = dds_to_vtf(bdds) if bdds else None
        if bvtf:
            with open(os.path.join(d, bark_slug + ".vtf"), "wb") as f:
                f.write(bvtf)
            with open(os.path.join(d, bark_slug + ".vmt"), "w", encoding="ascii") as f:
                f.write(_tree_vmt(MODEL_PREFIX, bark_slug, "wood"))
            bark_mat = bark_slug
        else:
            write_vtf(os.path.join(d, "tree_bark.vtf"), (96, 64, 40))
            with open(os.path.join(d, "tree_bark.vmt"), "w", encoding="ascii") as f:
                f.write(_tree_vmt(MODEL_PREFIX, "tree_bark", "wood"))

    subs = _billboard_tree_submeshes(bark_mat, leaf_slug)
    ref = os.path.join(work_dir, slug + ".smd")
    write_smd(subs, ref, scale=scale, flip_v=False)
    phys = os.path.join(work_dir, slug + "_phys.smd")
    write_smd([subs[0]], phys, scale=scale)        # trunk-only collision
    qc = os.path.join(work_dir, slug + ".qc")
    write_qc(qc, "%s/%s.mdl" % (MODEL_PREFIX, slug), ref, scale=1.0, collision_smd=phys)
    if compile_models and studiomdl and gamedir:
        ok, clog = run_studiomdl(studiomdl, gamedir, qc)
        if not ok:
            log("[warn] tree %s compile failed:\n%s" % (slug, clog[-300:]))
    return "models/%s/%s.mdl" % (MODEL_PREFIX, slug)


# --- studiomdl ----------------------------------------------------------------

def find_studiomdl(explicit=None):
    if explicit:
        return explicit if os.path.isfile(explicit) else None
    rels = [
        os.path.join("Source SDK Base 2013 Multiplayer", "bin", "studiomdl.exe"),
        os.path.join("GarrysMod", "bin", "studiomdl.exe"),   # GMod's own (targets garrysmod)
    ]
    for base in _STEAM_HINTS:
        for rel in rels:
            cand = os.path.join(base, rel)
            if os.path.isfile(cand):
                return cand
    return None


def find_gamedir(explicit=None):
    if explicit:
        return explicit if os.path.isdir(explicit) else None
    for base in _STEAM_HINTS:
        cand = os.path.join(base, "GarrysMod", "garrysmod")
        if os.path.isfile(os.path.join(cand, "gameinfo.txt")):
            return cand
    return None


def run_studiomdl(studiomdl, gamedir, qc_path, timeout=120):
    """Compile a QC. Returns (ok, log)."""
    cmd = [studiomdl, "-game", gamedir, "-nop4", "-fullcollide", "-verbose", qc_path]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except Exception as exc:                      # pragma: no cover
        return False, "studiomdl invocation failed: %r" % (exc,)
    log = (p.stdout or "") + (p.stderr or "")
    ok = p.returncode == 0 and ".mdl" in log.lower()
    return ok, log


# --- batch orchestration ------------------------------------------------------

def _horizontal_extent(subs, scale):
    xs = [v[0] for s in subs for v in s["verts"]]
    ys = [v[1] for s in subs for v in s["verts"]]
    if not xs:
        return 0.0
    return max(max(xs) - min(xs), max(ys) - min(ys)) * scale


def build_models(base_models, placements, source, work_dir, scale=1.0,
                 prefix=MODEL_PREFIX, studiomdl=None, gamedir=None,
                 compile_models=True, materials_root=None,
                 flip_winding=FLIP_WINDING, flip_v=FLIP_V,
                 collision="auto", collision_size=400.0, acd_threshold=0.08,
                 acd_max_hulls=-1, acd_jobs=None, ramp_axis="+x", model_lods=True,
                 trees=True, tree_model=None, tree_scale=1.0, tree_map=None,
                 jobs=None, use_cache=True, cache_rebuild=False, model_overrides=None,
                 log=print):
    """Convert every unique mesh referenced by ``placements`` to a .mdl.

    base_models : {base_formid: modl_path}
    placements  : {(cx,cy): [{"base", ...}]}
    source      : bsa.DataSource (resolves modl_path -> .nif bytes)
    work_dir    : scratch dir for .smd/.qc
    Returns (model_map, stats) where model_map = {base_formid: "models/<prefix>/<slug>.mdl"}.
    """
    os.makedirs(work_dir, exist_ok=True)
    # unique modl paths actually placed in this selection
    used = {}
    for plist in placements.values():
        for p in plist:
            modl = base_models.get(p["base"])
            if modl:
                used.setdefault(modl, []).append(p["base"])

    if not nif.available() and compile_models:
        log("[warn] PyFFI not available — cannot parse NIFs. "
            "Install with: pip install pyffi  (the time.clock shim is built in).")
    if collision == "acd" and not acd.available():
        log("[warn] --collision acd needs CoACD (pip install coacd); large props "
            "will be non-solid (walk-through) this run.")

    model_map = {}
    converted = {}        # modl -> model path
    needed_textures = {}  # material slug -> NIF texture path
    failures = []
    tree_modls = [m for m in used if m.lower().endswith(".spt")]
    mesh_modls = sorted(m for m in used if not m.lower().endswith(".spt"))

    # --- build cache: skip meshes whose NIF bytes + build params are unchanged
    # and whose compiled .mdl still exists in the gamedir. Keyed by modl path;
    # the signature pins everything that affects the compiled output.
    cache_path = os.path.join(work_dir, ".build_cache.json")
    cache = {}
    if use_cache and not cache_rebuild:
        try:
            with open(cache_path, encoding="utf-8") as f:
                cache = json.load(f)
        except Exception:
            cache = {}
    cache_lock = threading.Lock()
    # Limit concurrent CoACD calls. CoACD now runs in isolated subprocesses, so the
    # cap is just a memory/CPU budget (no crash risk). Default min(n_jobs,
    # ACD_MAX_CONCURRENCY); override with acd_jobs.
    n_jobs = max(1, jobs or min(8, (os.cpu_count() or 4)))
    acd_limit = max(1, acd_jobs or min(n_jobs, ACD_MAX_CONCURRENCY))
    acd_sem = threading.BoundedSemaphore(acd_limit)

    def _commit(modl, entry):
        """Add one entry and flush the cache to disk under a lock, so a mid-build
        kill still leaves a valid, up-to-date cache."""
        if not use_cache:
            return
        with cache_lock:
            cache[modl] = entry
            try:
                with open(cache_path, "w", encoding="utf-8") as f:
                    json.dump(cache, f)
            except Exception:
                pass

    overrides = model_overrides or {}

    def _override(modl):
        """Per-model settings dict (collision/ramp_axis/scale/surfaceprop/skip),
        looked up case-insensitively by .nif path. Empty when none set."""
        return overrides.get(modl.lower(), overrides.get(modl, {})) or {}

    def _sig(data, ov):
        key = "%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s" % (
            hashlib.sha1(data).hexdigest(), GEOM_REV, prefix, scale, flip_winding,
            flip_v, collision, collision_size, acd_threshold, acd_max_hulls,
            ramp_axis if collision == "ramp" else "-",
            "lod" if model_lods else "nolod",
            json.dumps(ov, sort_keys=True) if ov else "-")
        return hashlib.sha1(key.encode("utf-8")).hexdigest()

    def _build_one(modl):
        """Convert one NIF mesh -> .mdl. Pure per-model work (thread-safe: fresh
        BSA handle per read, unique slug filenames). Returns a result dict that the
        caller merges into shared state serially."""
        slug = slugify(modl)
        model_path = "models/%s/%s.mdl" % (prefix, slug)
        res = {"modl": modl, "model_path": None, "textures": {},
               "failure": None, "logs": [], "cached": False, "cache_entry": None}
        ov = _override(modl)
        if ov.get("skip"):
            res["skipped"] = True             # excluded by the model editor
            return res
        m_collision = ov.get("collision") or collision
        if m_collision == "custom":
            # "(custom)" GUI option: use hand-authored hulls when present (the
            # acd_parts branch below fires first regardless of mode), else fall
            # back to the global collision mode.
            m_collision = "hulls" if ov.get("hulls") else collision
        m_ramp = ov.get("ramp_axis") or ramp_axis
        m_scale = scale * float(ov.get("scale", 1.0) or 1.0)
        m_surf = ov.get("surfaceprop") or "default"
        try:
            data = source.get_mesh(modl)
            if data is None:
                res["failure"] = (modl, "not found in data dir / BSAs")
                return res
            # cache hit: signature matches and the compiled .mdl is still on disk
            sig = _sig(data, ov)
            if use_cache and compile_models and gamedir:
                ent = cache.get(modl)
                if (ent and ent.get("sig") == sig
                        and os.path.exists(os.path.join(gamedir, model_path))):
                    res["model_path"] = model_path
                    res["textures"] = ent.get("textures", {})
                    res["cached"] = True
                    return res
            subs = nif.extract_meshes(data)
            if not subs:
                res["failure"] = (modl, "no renderable geometry")
                return res
            for sub in subs:                       # assign a material per submesh
                tex = sub.get("texture")
                if tex:
                    ts = tex_slug(tex)
                    sub["material"] = ts
                    res["textures"][ts] = tex
                else:
                    sub["material"] = PLACEHOLDER_MATERIAL
            # cache-rebuild mode: don't recompile; just re-derive the cache entry
            # from the existing compiled .mdl (sig + textures), skipping CoACD/studiomdl.
            if cache_rebuild:
                if gamedir and os.path.exists(os.path.join(gamedir, model_path)):
                    res["model_path"] = model_path
                    res["cache_entry"] = {"sig": sig, "model_path": model_path,
                                          "textures": res["textures"], "collision": m_collision}
                    _commit(modl, res["cache_entry"])
                else:
                    res["failure"] = (modl, "no compiled .mdl in gamedir to cache")
                return res
            ref_smd = os.path.join(work_dir, slug + ".smd")
            qc = os.path.join(work_dir, slug + ".qc")
            write_smd(subs, ref_smd, scale=m_scale,
                      flip_winding=flip_winding, flip_v=flip_v)
            lods = (_model_lods(subs, slug, work_dir, m_scale, flip_winding, flip_v)
                    if model_lods else [])

            # Collision source: the NIF's own Havok shell (clean structural mesh)
            # if present, else the render mesh.
            try:
                coll_subs = nif.extract_collision(data)
            except Exception:
                coll_subs = None
            coll_subs = coll_subs or subs

            # Modes: none=off; full=solid all; auto=small solid / big non-solid
            # (walk through buildings); acd=small solid / big convex-decomposed
            # (walk INTO buildings accurately, needs coacd).
            big = _horizontal_extent(subs, m_scale) > collision_size
            phys = os.path.join(work_dir, slug + "_phys.smd")
            coll_smd, maxc = None, 64
            if ov.get("acd_parts"):
                # Convex parts authored/previewed in the GUI 3D editor — bake them
                # verbatim (final SMD units, scale 1.0) instead of recomputing CoACD.
                parts = [(pv_, [tuple(f) for f in pf]) for pv_, pf in ov["acd_parts"]]
                if parts:
                    write_collision_smd(parts, phys, scale=1.0)
                    coll_smd, maxc = phys, max(64, len(parts) + 8)
            elif m_collision == "none":
                pass
            elif m_collision == "hulls":
                # Multiple hand-authored hulls (GUI 3D editor): boxes, wedges, or
                # trapezoidal prisms. Coords are in FINAL (SMD/Hammer) units ->
                # written as-is (scale 1.0).
                parts = [p for p in (hull_from_spec(s) for s in (ov.get("hulls") or []))
                         if p is not None]
                if parts:
                    write_collision_smd(parts, phys, scale=1.0)
                    coll_smd, maxc = phys, max(64, len(parts) + 8)
            elif m_collision in ("bbox", "ramp"):
                # Single convex primitive sized to the mesh bounds: a box (cheap
                # blocking volume) or a wedge ramp (walk UP the slope). Both are
                # one convex hull -> trivial physics.
                bb = _aabb(coll_subs)
                if bb:
                    part = (box_hull(bb) if m_collision == "bbox"
                            else ramp_hull(bb, m_ramp))
                    write_collision_smd([part], phys, scale=m_scale)
                    coll_smd, maxc = phys, 1
            elif big and m_collision in ("auto", "acd"):
                if m_collision == "acd" and acd.available():
                    # isolated subprocess so a crash/timeout on a pathological mesh
                    # can't kill the build. On failure, retry once at a coarser
                    # threshold (fast; keeps walk-in collision), else fall back to a
                    # SOLID hull so the prop is at least not walk-through.
                    def _acd(thr, to):
                        try:
                            with acd_sem:
                                return acd.decompose_isolated(
                                    coll_subs, threshold=thr, max_hulls=acd_max_hulls,
                                    timeout=to)
                        except Exception:
                            return None
                    parts = _acd(acd_threshold, ACD_TIMEOUT)
                    coarse = False
                    if not parts and acd_threshold < ACD_COARSE_THRESHOLD:
                        parts = _acd(ACD_COARSE_THRESHOLD, ACD_TIMEOUT)
                        coarse = bool(parts)
                    if parts:
                        write_collision_smd(parts, phys, scale=m_scale)
                        coll_smd, maxc = phys, max(64, len(parts) + 8)
                        res["logs"].append("    acd %s -> %d hulls%s%s" % (
                            slug, len(parts), " (havok)" if coll_subs is not subs else "",
                            " (coarse)" if coarse else ""))
                    else:
                        # last resort: solid concave-by-component (not walk-through)
                        write_smd(coll_subs, phys, scale=m_scale)
                        coll_smd = phys
                        res["logs"].append("[warn] ACD failed for %s -> solid "
                                           "(no walk-in, but not ghost)" % slug)
                # big auto -> non-solid (None)
            else:
                # small props (any mode) or 'full' (any size): concave-by-component
                write_smd(coll_subs, phys, scale=m_scale)
                coll_smd = phys
            write_qc(qc, "%s/%s.mdl" % (prefix, slug), ref_smd, scale=1.0,
                     surfaceprop=m_surf, collision_smd=coll_smd, max_convex=maxc, lods=lods)
            if compile_models:
                ok, clog = run_studiomdl(studiomdl, gamedir, qc)
                if not ok:
                    res["failure"] = (modl, "studiomdl failed (see log)")
                    res["logs"].append("[warn] studiomdl failed for %s:\n%s"
                                       % (slug, clog[-600:]))
                    # still map it — user can recompile the .qc
            res["model_path"] = model_path
            if not res["failure"] and compile_models:
                res["cache_entry"] = {"sig": sig, "model_path": model_path,
                                      "textures": res["textures"], "collision": m_collision}
                _commit(modl, res["cache_entry"])   # flush now -> kill-safe
        except Exception as exc:
            res["failure"] = (modl, "exception: %r" % (exc,))
        return res

    # Per-model work is independent; studiomdl (subprocess) and CoACD (C++) both
    # release the GIL, so threads give a near-linear speedup. Merge results
    # serially afterwards in a deterministic order.
    if mesh_modls:
        log("    compiling %d meshes with %d worker%s (CoACD<=%d)%s..."
            % (len(mesh_modls), n_jobs, "" if n_jobs == 1 else "s", acd_limit,
               " (cache on)" if use_cache else ""))
    if n_jobs == 1 or len(mesh_modls) <= 1:
        results = [_build_one(m) for m in mesh_modls]
    else:
        with ThreadPoolExecutor(max_workers=n_jobs) as ex:
            results = list(ex.map(_build_one, mesh_modls))
    n_cached = n_rebuilt = n_skipped = 0
    for res in results:                            # deterministic merge (cache already flushed per-model)
        for line in res["logs"]:
            log(line)
        needed_textures.update(res["textures"])
        if res.get("skipped"):
            n_skipped += 1
            continue
        if res["failure"]:
            failures.append(res["failure"])
        if res["model_path"]:
            converted[res["modl"]] = res["model_path"]
        if res["cached"]:
            n_cached += 1
        if cache_rebuild and res["cache_entry"]:
            n_rebuilt += 1
    if cache_rebuild:
        if mesh_modls:
            log("    cache rebuilt: %d entries from existing .mdl, %d missing"
                % (n_rebuilt, len(mesh_modls) - n_rebuilt))
    elif mesh_modls:
        log("    models: %d built, %d reused from cache"
            % (len(mesh_modls) - n_cached, n_cached))

    # Trees for SpeedTree (.spt) placements. Precedence per species:
    #   per-species tree_map  >  tree_map["_default"] / tree_model  >  generate.
    n_trees = 0
    model_scale = {}
    if tree_modls and trees:
        tmap = {k.lower(): v for k, v in (tree_map or {}).items()}
        default_model = tmap.get("_default") or tree_model
        cat = None
        tree_cache = {}            # (leaf, bark) -> generated model path
        geom_path = None
        modeled = generated = 0
        for m in tree_modls:
            name = m.lstrip("\\").rsplit(".", 1)[0]
            is_shrub = "shrub" in name.lower()
            # Explicit per-species mapping always wins. Trees fall back to the
            # default model; shrubs skip the default and generate a 3D bush so they
            # don't become full-size trees.
            chosen = tmap.get(name.lower()) or (None if is_shrub else default_model)
            if chosen:
                converted[m] = chosen
                for base in used[m]:
                    model_scale[base] = tree_scale
                modeled += 1
                continue
            # no model assigned -> generate self-contained geometry
            if cat is None and source:
                from .trees import TreeTextures
                cat = TreeTextures(source)
            leaf, bark = cat.match(name) if cat else (None, None)
            if is_shrub:
                key = ("bush", leaf)
                if key not in tree_cache:
                    lb = (leaf or name).split("\\")[-1].rsplit(".", 1)[0]
                    sslug = "bush_" + re.sub(r"[^a-z0-9]+", "", lb.lower())[:46]
                    tree_cache[key] = generate_bush_model(
                        sslug, leaf, work_dir, materials_root, source,
                        scale, studiomdl, gamedir, compile_models, log=log)
                converted[m] = tree_cache[key]
            elif leaf and source:
                key = (leaf, bark)
                if key not in tree_cache:
                    lb = leaf.split("\\")[-1].rsplit(".", 1)[0]
                    sslug = "tree_" + re.sub(r"[^a-z0-9]+", "", lb.lower())[:48]
                    tree_cache[key] = generate_billboard_tree(
                        sslug, leaf, bark, work_dir, materials_root, source,
                        scale, studiomdl, gamedir, compile_models, log=log)
                converted[m] = tree_cache[key]
            else:
                if geom_path is None:
                    geom_path = generate_tree_model(work_dir, materials_root, prefix, scale,
                                                    studiomdl, gamedir, compile_models, log=log)
                converted[m] = geom_path
            generated += 1
        n_trees = len(tree_modls)
        log("    trees: %d species (%d mapped to .mdl, %d generated)"
            % (n_trees, modeled, generated))

    for base, modl in base_models.items():
        if modl in converted:
            model_map[base] = converted[modl]

    tex_ok = tex_fail = 0
    if materials_root is not None:
        write_placeholder_material(materials_root, prefix)
        tex_ok, tex_fail = write_textures(materials_root, needed_textures, source, prefix)

    stats = {
        "unique_meshes": len(used),
        "converted": len(converted),
        "failed": len(failures),
        "failures": failures,
        "textures": tex_ok,
        "textures_failed": tex_fail,
        "trees": n_trees,
        "cached": n_cached,
        "skipped": n_skipped,
        "model_scale": model_scale,
    }
    return model_map, stats
