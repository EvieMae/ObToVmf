"""Extract static-mesh geometry from Oblivion .nif files via PyFFI.

Oblivion NIF is version 20.0.0.5, which has NO per-block sizes — a hand-rolled
parser desyncs on any unimplemented block type. PyFFI is the nif.xml made
executable and understands every block, so we lean on it. PyFFI 2.2.3 calls the
removed time.clock(); we shim it before import.

extract_meshes() returns a list of submeshes, each a dict with model-space
geometry (the .nif's internal node transforms baked in; placement is applied
later via prop_static):
    {"verts": [(x,y,z),...], "tris": [(a,b,c),...],
     "normals": [(x,y,z),...], "uvs": [(u,v),...], "texture": str|None}
"""
from __future__ import annotations

import time as _time

if not hasattr(_time, "clock"):          # removed in Python 3.8; PyFFI 2.2.3 needs it
    _time.clock = _time.perf_counter

try:
    from pyffi.formats.nif import NifFormat
    HAVE_PYFFI = True
    _IMPORT_ERROR = None
except Exception as exc:                  # pragma: no cover
    NifFormat = None
    HAVE_PYFFI = False
    _IMPORT_ERROR = exc


def available():
    return HAVE_PYFFI


def _identity():
    return ([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]], [0.0, 0.0, 0.0], 1.0)


def _mat3_mul(a, b):
    return [[sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)] for i in range(3)]


def _mat3_vec(m, v):
    return [m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
            m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
            m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2]]


def _compose(parent, local):
    rp, tp, sp = parent
    rc, tc, sc = local
    r = _mat3_mul(rp, rc)
    tcr = _mat3_vec(rp, tc)
    t = [tcr[i] * sp + tp[i] for i in range(3)]
    return (r, t, sp * sc)


def _apply(tf, v):
    r, t, s = tf
    rv = _mat3_vec(r, [v[0] * s, v[1] * s, v[2] * s])
    return (rv[0] + t[0], rv[1] + t[1], rv[2] + t[2])


def _apply_dir(tf, v):
    return tuple(_mat3_vec(tf[0], list(v)))


def _block_transform(block):
    m = block.rotation
    # NIF/Gamebryo stores the rotation for row-vector math (v' = v * M); our
    # _mat3_vec does column-vector math (M * v), so we use the TRANSPOSE. Identity
    # and symmetric (e.g. 180-degree) rotations are transpose-invariant, so most
    # models are unaffected; a node with a real rotation (e.g. SkCastleGatehouse's
    # 90-degree-about-X) would otherwise come out flipped/upside-down.
    r = [[m.m_11, m.m_21, m.m_31],
         [m.m_12, m.m_22, m.m_32],
         [m.m_13, m.m_23, m.m_33]]
    t = [block.translation.x, block.translation.y, block.translation.z]
    return (r, t, float(block.scale))


def _name(block):
    n = getattr(block, "name", None)
    if n is None:
        return ""
    if isinstance(n, bytes):
        return n.decode("latin-1", "replace")
    return str(n)


def _find_texture(shape):
    for prop in shape.properties:
        if isinstance(prop, NifFormat.NiTexturingProperty) and getattr(prop, "has_base_texture", False):
            src = prop.base_texture.source
            if src is not None and getattr(src, "use_external", 1):
                fn = src.file_name
                if isinstance(fn, bytes):
                    fn = fn.decode("latin-1", "replace")
                return str(fn) if fn else None
    return None


def _extract_geom(shape, world):
    geom = shape.data
    if geom is None:
        return None
    verts = [_apply(world, (v.x, v.y, v.z)) for v in geom.vertices]
    tris = [t for t in geom.get_triangles()]
    if not verts or not tris:
        return None
    norms = ([_apply_dir(world, (n.x, n.y, n.z)) for n in geom.normals]
             if geom.has_normals else [])
    uvs = [(t.u, t.v) for t in geom.uv_sets[0]] if geom.uv_sets else []
    return {"verts": verts, "tris": tris, "normals": norms, "uvs": uvs,
            "texture": _find_texture(shape)}


def _walk(block, world, out):
    if isinstance(block, NifFormat.NiAVObject):
        # skip collision/editor-only subtrees
        nm = _name(block).lower()
        if "rootcollision" in nm or "editormarker" in nm:
            return
        world = _compose(world, _block_transform(block))
    if isinstance(block, (NifFormat.NiTriShape, NifFormat.NiTriStrips)):
        sub = _extract_geom(block, world)
        if sub:
            out.append(sub)
    if isinstance(block, NifFormat.NiNode):
        for child in block.children:
            if child is not None:
                _walk(child, world, out)


def _read(path_or_bytes):
    import io
    data = NifFormat.Data()
    if isinstance(path_or_bytes, (bytes, bytearray)):
        data.read(io.BytesIO(path_or_bytes))
    else:
        with open(path_or_bytes, "rb") as f:
            data.read(f)
    return data


def extract_meshes(path_or_bytes):
    """Parse a .nif (path str or raw bytes) and return its renderable submeshes."""
    if not HAVE_PYFFI:
        raise RuntimeError("PyFFI not available: %r" % (_IMPORT_ERROR,))
    data = _read(path_or_bytes)
    out = []
    for root in data.roots:
        _walk(root, _identity(), out)
    return out


# 1 Havok unit = 7 NIF/game units in Oblivion (rigid-body translations, transform
# shapes, packed/convex/box shape coordinates are stored in Havok units; the verts
# referenced by bhkNiTriStripsShape's NiTriStripsData are plain NIF units).
HAVOK_SCALE = 7.0


def _quat_to_mat3(x, y, z, w):
    return [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]


def _hk_compose(parent, r, t):
    rp, tp = parent
    return (_mat3_mul(rp, r), [_mat3_vec(rp, t)[i] + tp[i] for i in range(3)])


def _hk_apply(tf, v):
    r, t = tf
    rv = _mat3_vec(r, list(v))
    return (rv[0] + t[0], rv[1] + t[1], rv[2] + t[2])


def _hk_sub(tf, verts, tris):
    return {"verts": [_hk_apply(tf, v) for v in verts],
            "tris": list(tris), "normals": [], "uvs": [], "material": "phys"}


def _hk_box(tf, hx, hy, hz, out):
    corners = [(sx * hx, sy * hy, sz * hz)
               for sz in (-1, 1) for sy in (-1, 1) for sx in (-1, 1)]
    # any topology referencing all verts works — studiomdl convex-hulls the set
    tris = [(0, 1, 3), (0, 3, 2), (4, 7, 5), (4, 6, 7),
            (0, 5, 1), (0, 4, 5), (2, 3, 7), (2, 7, 6),
            (0, 2, 6), (0, 6, 4), (1, 5, 7), (1, 7, 3)]
    out.append(_hk_sub(tf, corners, tris))


def _hk_walk(sh, tf, out, depth=0):
    """Walk a Havok shape tree, composing transforms, appending one submesh per
    concrete shape (so convex pieces stay separate)."""
    if sh is None or depth > 16:
        return
    if isinstance(sh, NifFormat.bhkMoppBvTreeShape):
        _hk_walk(sh.shape, tf, out, depth + 1)
        return
    if isinstance(sh, NifFormat.bhkListShape):
        for s2 in sh.sub_shapes:
            _hk_walk(s2, tf, out, depth + 1)
        return
    if isinstance(sh, (NifFormat.bhkTransformShape, NifFormat.bhkConvexTransformShape)):
        m = sh.transform
        # row-vector matrix -> transpose for our column convention; translation is
        # the 4th ROW, in Havok units
        r = [[m.m_11, m.m_21, m.m_31],
             [m.m_12, m.m_22, m.m_32],
             [m.m_13, m.m_23, m.m_33]]
        t = [m.m_41 * HAVOK_SCALE, m.m_42 * HAVOK_SCALE, m.m_43 * HAVOK_SCALE]
        _hk_walk(sh.shape, _hk_compose(tf, r, t), out, depth + 1)
        return
    if isinstance(sh, NifFormat.bhkNiTriStripsShape):
        for sd in sh.strips_data:
            if sd is None:
                continue
            verts = [(v.x, v.y, v.z) for v in sd.vertices]          # NIF units
            tris = [(a, c, d) for a, c, d in sd.get_triangles()]
            if verts and tris:
                out.append(_hk_sub(tf, verts, tris))
        return
    if isinstance(sh, NifFormat.bhkPackedNiTriStripsShape):
        d = sh.data
        if d is not None:
            verts = [(v.x * HAVOK_SCALE, v.y * HAVOK_SCALE, v.z * HAVOK_SCALE)
                     for v in d.vertices]
            tris = [(t.triangle.v_1, t.triangle.v_2, t.triangle.v_3)
                    for t in d.triangles]
            if verts and tris:
                out.append(_hk_sub(tf, verts, tris))
        return
    if isinstance(sh, NifFormat.bhkConvexVerticesShape):
        verts = [(v.x * HAVOK_SCALE, v.y * HAVOK_SCALE, v.z * HAVOK_SCALE)
                 for v in sh.vertices]
        if len(verts) >= 4:
            tris = [(0, i, i + 1) for i in range(1, len(verts) - 1)]  # fan; re-hulled
            out.append(_hk_sub(tf, verts, tris))
        return
    if isinstance(sh, NifFormat.bhkBoxShape):
        d = sh.dimensions                                            # half-extents
        _hk_box(tf, d.x * HAVOK_SCALE, d.y * HAVOK_SCALE, d.z * HAVOK_SCALE, out)
        return
    if isinstance(sh, NifFormat.bhkSphereShape):
        r = float(sh.radius) * HAVOK_SCALE
        _hk_box(tf, r, r, r, out)                                    # box approx
        return
    # unknown shape type: ignore (better partial collision than none)


def _collect_collision(block, world, out):
    """Walk the scene graph accumulating node world transforms; for every node
    carrying a bhkCollisionObject, extract its rigid body's shapes with the BODY
    transform (bhkRigidBodyT) composed under the NODE's world transform — both are
    needed: posts rotate via the body quaternion, paintings via the host node."""
    if isinstance(block, NifFormat.NiAVObject):
        world = _compose(world, _block_transform(block))
        co = getattr(block, "collision_object", None)
        body = getattr(co, "body", None) if co is not None else None
        if body is not None and isinstance(body, NifFormat.bhkRigidBody):
            tf = ([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]], [0.0, 0.0, 0.0])
            if isinstance(body, NifFormat.bhkRigidBodyT):
                q = body.rotation
                t = [body.translation.x * HAVOK_SCALE,
                     body.translation.y * HAVOK_SCALE,
                     body.translation.z * HAVOK_SCALE]
                tf = (_quat_to_mat3(q.x, q.y, q.z, q.w), t)
            pieces = []
            _hk_walk(body.shape, tf, pieces)
            for s in pieces:
                s["verts"] = [_apply(world, v) for v in s["verts"]]
                out.append(s)
    if isinstance(block, NifFormat.NiNode):
        for child in block.children:
            if child is not None:
                _collect_collision(child, world, out)


def extract_collision(path_or_bytes):
    """Return the NIF's Havok collision as submeshes [{"verts","tris"}] in model
    space (same coords as the render mesh), or None if it has no usable Havok
    geometry. This is the clean structural shell Bethesda authored for physics --
    a better, lighter collision source than the detailed render mesh.

    Composes (node world transform) @ (bhkRigidBodyT quaternion+translation, Havok
    units x7) @ (transform-shape wrappers) — without these, models like basement
    posts and paintings come back rotated 90 degrees. Handles MOPP/list wrappers,
    (packed) tri-strip meshes, convex-vertex shapes, and box/sphere primitives;
    each concrete shape becomes its own submesh. Animated models (anim NIFs) may
    still mismatch — their collision rides animated nodes."""
    if not HAVE_PYFFI:
        return None
    data = _read(path_or_bytes)
    out = []
    for root in data.roots:
        _collect_collision(root, _identity(), out)
    return out or None
