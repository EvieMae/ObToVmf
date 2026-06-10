"""Write a Valve Map Format (.vmf) file with displacement terrain.

Each exterior cell's 33x33 heightmap is emitted as a grid of displacement
brushes. A power-N displacement has (2^N + 1) vertices per edge, so a 32-interval
cell splits into (32 / 2^N)^2 displacements:

    power 4 -> 17x17 verts -> 2x2 = 4 displacements / cell   (recommended)
    power 3 ->  9x9  verts -> 4x4 = 16 displacements / cell
    power 2 ->  5x5  verts -> 8x8 = 64 displacements / cell

Each displacement is a thin axis-aligned box brush whose +Z (top) face carries a
``dispinfo``. The top face is a flat reference plane at ``ztop`` (the minimum
height in that tile); per-vertex ``distances`` along the +Z normal push each
vertex up to its true terrain height, so distances stay >= 0.

The box-face winding and the ``dispinfo`` block layout were verified against a
real Hammer-exported VMF (ata4/bspsrc test_displacements.vmf).

ORIENTATION NOTE: the displacement grid is written with row index = +Y (north)
and within-row index = +X (east), and ``startposition`` at the SW corner. If
terrain loads transposed/mirrored in Hammer, this is the single knob to flip
(swap the r/c sampling in :func:`_tile_solid`); the geometry is still valid.
"""
from __future__ import annotations

from .land import GRID, INTERVALS, VERTEX_SPACING, CELL_SIZE

DEFAULT_MATERIAL = "dev/dev_measuregeneric01b"
DEFAULT_SKY = "sky_day01_01"
WATER_MATERIAL = "oblivion2vmf/water"      # self-contained %compilewater material
NODRAW_MATERIAL = "tools/toolsnodraw"
MAX_MAP_DISPINFO = 2048            # Source SDK 2013 hard cap (bspfile.h)
MAX_COORD = 16384                  # +/- world boundary per axis (coordsize.h)
ALLOWED_POWERS = (2, 3, 4)

# Player-accurate Oblivion-unit -> Hammer-unit factor: a 1.0-scale Oblivion
# character is 128 units tall; a Source player is 72 units (= 6 ft). 72/128 =
# 0.5625, which also gives 1 unit = 0.5625 in = 1.4288 cm (~70 units/metre).
REAL_WORLD_SCALE = 0.5625


class _Ids:
    def __init__(self):
        self.n = 0

    def next(self):
        self.n += 1
        return self.n


def _num(v):
    """Compact numeric formatting: integers stay integers, floats are trimmed."""
    f = float(v)
    if f == int(f):
        return str(int(f))
    return ("%.6f" % f).rstrip("0").rstrip(".")


def _vec(p):
    return " ".join(_num(c) for c in p)


def _clamp_alpha(a):
    a = int(a)
    return 0 if a < 0 else 255 if a > 255 else a


def _plane(p0, p1, p2):
    return "(%s) (%s) (%s)" % (_vec(p0), _vec(p1), _vec(p2))


def _box_faces(x0, y0, z0, x1, y1, z1, tscale=0.25):
    """Six faces of an axis-aligned box; winding matches Hammer's box export.

    ``tscale`` is the texture UV scale (world units per texel) for every face.
    Returns a list of (name, plane_points, uaxis, vaxis). The ``top`` face
    (+Z, at z1) is where a dispinfo is attached.
    """
    s = _num(tscale)
    ux = "[1 0 0 0] %s" % s
    uy = "[0 1 0 0] %s" % s
    vy = "[0 -1 0 0] %s" % s
    vz = "[0 0 -1 0] %s" % s
    return [
        ("bottom", ((x0, y1, z0), (x0, y0, z0), (x1, y0, z0)), ux, vy),
        ("west",   ((x0, y0, z1), (x0, y0, z0), (x0, y1, z0)), uy, vz),
        ("east",   ((x1, y1, z1), (x1, y1, z0), (x1, y0, z0)), uy, vz),
        ("north",  ((x0, y1, z1), (x0, y1, z0), (x1, y1, z0)), ux, vz),
        ("south",  ((x1, y0, z1), (x1, y0, z0), (x0, y0, z0)), ux, vz),
        ("top",    ((x0, y0, z1), (x0, y1, z1), (x1, y1, z1)), ux, vy),
    ]


def _disp_block(power, startpos, samples, ztop, blend=None, indent="\t\t\t"):
    """Build the ``dispinfo`` block. ``samples`` is a (verts x verts) grid of
    absolute (already-scaled) heights; ``samples[r][c]`` r=+Y, c=+X. ``blend`` is
    None, {"mode":"alpha","alphas": grid 0-255}, or
    {"mode":"multiblend","weights": grid of (w1,w2,w3,w4)} for 4-way."""
    n = 1 << power
    verts = n + 1
    i = indent
    i1, i2 = i + "\t", i + "\t\t"
    out = []
    out.append('%sdispinfo' % i)
    out.append('%s{' % i)
    out.append('%s"power" "%d"' % (i1, power))
    out.append('%s"startposition" "[%s]"' % (i1, _vec(startpos)))
    out.append('%s"flags" "0"' % i1)
    out.append('%s"elevation" "0"' % i1)
    out.append('%s"subdiv" "0"' % i1)

    def grid_block(name, row_builder, nrows):
        out.append('%s%s' % (i1, name))
        out.append('%s{' % i1)
        for r in range(nrows):
            out.append('%s"row%d" "%s"' % (i2, r, row_builder(r)))
        out.append('%s}' % i1)

    up = " ".join(["0 0 1"] * verts)
    zero3 = " ".join(["0 0 0"] * verts)
    zero1 = " ".join(["0"] * verts)
    grid_block("normals", lambda r: up, verts)
    grid_block(
        "distances",
        lambda r: " ".join(_num(samples[r][c] - ztop) for c in range(verts)),
        verts,
    )
    grid_block("offsets", lambda r: zero3, verts)
    grid_block("offset_normals", lambda r: up, verts)
    if blend and blend.get("mode") == "alpha":
        a = blend["alphas"]
        grid_block("alphas",
                   lambda r: " ".join(str(_clamp_alpha(a[r][c])) for c in range(verts)),
                   verts)
    else:
        grid_block("alphas", lambda r: zero1, verts)
    # triangle_tags: n rows of 2*n values
    tri = " ".join(["9"] * (2 * n))
    grid_block("triangle_tags", lambda r: tri, n)
    # allowed_verts: single fixed entry
    out.append('%sallowed_verts' % i1)
    out.append('%s{' % i1)
    out.append('%s"10" "%s"' % (i2, " ".join(["-1"] * 10)))
    out.append('%s}' % i1)

    if blend and blend.get("mode") == "multiblend":
        w = blend["weights"]
        grid_block("multiblend",
                   lambda r: " ".join(_num(v) for c in range(verts) for v in w[r][c]),
                   verts)
        zero4 = " ".join(["0 0 0 0"] * verts)
        grid_block("alphablend", lambda r: zero4, verts)
        white3 = " ".join(["1 1 1"] * verts)
        for ch in range(4):
            grid_block("multiblend_color_%d" % ch, lambda r: white3, verts)

    out.append('%s}' % i)
    return "\n".join(out)


def _tile_solid(ids, x0, y0, x1, y1, ztop, thickness, samples, power, material,
                blend=None, tex_scale=0.25, lightmapscale=16, floor_z=None):
    """One displacement brush (a thin box with a dispinfo on its top face).

    Only the +Z (top) face carries the terrain material/displacement; the five
    hidden faces are nodraw'd so they cost nothing to render. If ``floor_z`` is
    given the brush extends down to it (solid terrain mass for vvis blocking +
    a sealed bottom) instead of being a thin slab."""
    sid = ids.next()
    zmax = ztop
    zmin = floor_z if floor_z is not None else ztop - thickness
    out = []
    out.append('\tsolid')
    out.append('\t{')
    out.append('\t\t"id" "%d"' % sid)
    for name, plane, ua, va in _box_faces(x0, y0, zmin, x1, y1, zmax, tex_scale):
        top = name == "top"
        out.append('\t\tside')
        out.append('\t\t{')
        out.append('\t\t\t"id" "%d"' % ids.next())
        out.append('\t\t\t"plane" "%s"' % _plane(*plane))
        out.append('\t\t\t"material" "%s"' % (material if top else NODRAW_MATERIAL))
        out.append('\t\t\t"uaxis" "%s"' % ua)
        out.append('\t\t\t"vaxis" "%s"' % va)
        out.append('\t\t\t"rotation" "0"')
        out.append('\t\t\t"lightmapscale" "%d"' % lightmapscale)
        out.append('\t\t\t"smoothing_groups" "0"')
        if top:
            out.append(_disp_block(power, (x0, y0, zmax), samples, ztop, blend))
        out.append('\t\t}')
    out.append('\t\teditor')
    out.append('\t\t{')
    out.append('\t\t\t"color" "0 180 0"')
    out.append('\t\t\t"visgroupshown" "1"')
    out.append('\t\t\t"visgroupautoshown" "1"')
    out.append('\t\t}')
    out.append('\t}')
    return "\n".join(out)


def _plain_box_solid(ids, x0, y0, z0, x1, y1, z1, top_material, side_material,
                     color="0 128 255", tex_scale=0.25):
    """An axis-aligned box brush with ``top_material`` on its +Z face and
    ``side_material`` on the other five. Used for water volumes (top = water,
    sides/bottom = nodraw)."""
    out = ['\tsolid', '\t{', '\t\t"id" "%d"' % ids.next()]
    for name, plane, ua, va in _box_faces(x0, y0, z0, x1, y1, z1, tex_scale):
        mat = top_material if name == "top" else side_material
        out += ['\t\tside', '\t\t{',
                '\t\t\t"id" "%d"' % ids.next(),
                '\t\t\t"plane" "%s"' % _plane(*plane),
                '\t\t\t"material" "%s"' % mat,
                '\t\t\t"uaxis" "%s"' % ua,
                '\t\t\t"vaxis" "%s"' % va,
                '\t\t\t"rotation" "0"',
                '\t\t\t"lightmapscale" "16"',
                '\t\t\t"smoothing_groups" "0"',
                '\t\t}']
    out += ['\t\teditor', '\t\t{', '\t\t\t"color" "%s"' % color,
            '\t\t\t"visgroupshown" "1"', '\t\t\t"visgroupautoshown" "1"', '\t\t}', '\t}']
    return "\n".join(out)


def _greedy_rects(cellset):
    """Cover a set of (cx,cy) cells with as few axis-aligned cell-rectangles as
    possible (greedy meshing). Returns [(x0,y0,x1,y1), ...] inclusive."""
    remaining = set(cellset)
    rects = []
    while remaining:
        cx, cy = min(remaining)                     # lowest x then y
        w = 1
        while (cx + w, cy) in remaining:
            w += 1
        h = 1
        while all((cx + i, cy + h) in remaining for i in range(w)):
            h += 1
        for j in range(h):
            for i in range(w):
                remaining.discard((cx + i, cy + j))
        rects.append((cx, cy, cx + w - 1, cy + h - 1))
    return rects


def build_water_solids(cells, cell_water, scale=1.0, recenter=True, ids=None,
                       water_material=WATER_MATERIAL, side_material=NODRAW_MATERIAL,
                       default_water=None, min_depth=64.0, margin=8.0, merge=True):
    """Swimmable water volumes from the cells that have water. With ``merge``
    (default), adjacent cells at the SAME water height are combined into one brush
    (greedy rectangles) — far fewer brushes and no per-cell surface seams. Each
    brush runs from below its lowest covered terrain up to the water surface.
    Returns (solid_strings, count). Uses the SAME recenter offsets as the terrain."""
    if not cell_water and default_water is None:
        return [], 0
    ids = ids or _Ids()
    x_off, y_off, z_off = _offsets(cells, recenter)

    # cell -> (water height, lowest terrain in that cell)
    info = {}
    for (cx, cy), grid in cells.items():
        h = cell_water.get((cx, cy), default_water)
        if h is not None:
            info[(cx, cy)] = (h, min(min(row) for row in grid))
    if not info:
        return [], 0

    # group cells by (rounded) water height so only equal surfaces merge
    groups = {}
    for cell, (h, _tmin) in info.items():
        groups.setdefault(round(h, 3), []).append(cell)

    out = []
    for hkey, group in sorted(groups.items()):
        rects = _greedy_rects(group) if merge else [(c[0], c[1], c[0], c[1]) for c in group]
        for (rx0, ry0, rx1, ry1) in rects:
            surface = (hkey - z_off) * scale
            tmin = min(info[(cx, cy)][1]
                       for cx in range(rx0, rx1 + 1) for cy in range(ry0, ry1 + 1)
                       if (cx, cy) in info)
            bottom = min(surface - min_depth, (tmin - z_off) * scale - margin)
            x0 = (rx0 * CELL_SIZE - x_off) * scale
            x1 = ((rx1 + 1) * CELL_SIZE - x_off) * scale
            y0 = (ry0 * CELL_SIZE - y_off) * scale
            y1 = ((ry1 + 1) * CELL_SIZE - y_off) * scale
            out.append(_plain_box_solid(ids, x0, y0, bottom, x1, y1, surface,
                                        water_material, side_material))
    return out, len(out)


SKYBOX_MATERIAL = "tools/toolsskybox"


def build_skybox_enclosure(cells, scale=1.0, recenter=True, ids=None, margin=1024.0,
                           ceiling=5000.0, thickness=32.0, vis_floor_depth=512.0,
                           skymat=SKYBOX_MATERIAL):
    """Six toolsskybox brushes forming a closed box around the whole map, so the
    map is SEALED (no leak) and vvis can compute real visleaves. Returns
    (solid_strings, count)."""
    if not cells:
        return [], 0
    ids = ids or _Ids()
    x_off, y_off, z_off = _offsets(cells, recenter)
    cxs = [c[0] for c in cells]
    cys = [c[1] for c in cells]
    gmin = min(min(min(row) for row in g) for g in cells.values())
    gmax = max(max(max(row) for row in g) for g in cells.values())
    wx0 = (min(cxs) * CELL_SIZE - x_off) * scale - margin
    wx1 = ((max(cxs) + 1) * CELL_SIZE - x_off) * scale + margin
    wy0 = (min(cys) * CELL_SIZE - y_off) * scale - margin
    wy1 = ((max(cys) + 1) * CELL_SIZE - y_off) * scale + margin
    wz0 = (gmin - z_off) * scale - vis_floor_depth - margin
    wz1 = (gmax - z_off) * scale + ceiling
    t = thickness
    boxes = [
        (wx0, wy0, wz0, wx1, wy1, wz0 + t),        # floor
        (wx0, wy0, wz1 - t, wx1, wy1, wz1),        # ceiling
        (wx0, wy0, wz0, wx0 + t, wy1, wz1),        # west
        (wx1 - t, wy0, wz0, wx1, wy1, wz1),        # east
        (wx0, wy0, wz0, wx1, wy0 + t, wz1),        # south
        (wx0, wy1 - t, wz0, wx1, wy1, wz1),        # north
    ]
    out = [_plain_box_solid(ids, *b, skymat, skymat, color="128 128 200") for b in boxes]
    return out, len(out)


def build_skybox_model(ids, model_path, model_bounds, sky_scale=16,
                       margin=1500.0, prescaled=True):
    """Place a baked terrain model as a 3D-skybox backdrop. Builds a small
    toolsskybox room (so the prop sits in a sealed leaf and renders as the sky),
    a sky_camera, and a prop_static at 1/sky_scale that aligns 1:1 with the real
    terrain. ``model_bounds`` = (minx,maxx,miny,maxy,minz,maxz) of the model in HU.

    If ``prescaled`` the model was already compiled at 1/sky_scale (via QC $scale),
    so the prop is placed at scale 1.0 — GMod's 3D-skybox pass ignores prop_static
    ``uniformscale``, which would otherwise leave the backdrop floating. Only an
    externally-supplied --skybox-model-file is placed with uniformscale (prescaled
    False), since we can't recompile it.
    Returns (world_solid_strings, entity_strings)."""
    mnx, mxx, mny, mxy, mnz, mxz = model_bounds
    inv = 1.0 / sky_scale
    # the scaled model occupies these half/extents around the prop origin; pad the
    # room generously (×1.15 + margin) so the model can't poke outside the box
    hx = max(abs(mnx), abs(mxx)) * inv * 1.15
    hy = max(abs(mny), abs(mxy)) * inv * 1.15
    z_lo, z_hi = mnz * inv, mxz * inv
    # park the skybox room well below the playable map, within the +/-16384 cube
    O = (0.0, 0.0, -9000.0)
    rx = hx + margin
    ry = hy + margin
    rz0 = O[2] + z_lo - margin
    rz1 = O[2] + z_hi + margin
    t = 32.0
    boxes = [
        (O[0] - rx, O[1] - ry, rz0 - t, O[0] + rx, O[1] + ry, rz0),     # floor
        (O[0] - rx, O[1] - ry, rz1, O[0] + rx, O[1] + ry, rz1 + t),     # ceiling
        (O[0] - rx - t, O[1] - ry, rz0, O[0] - rx, O[1] + ry, rz1),     # west
        (O[0] + rx, O[1] - ry, rz0, O[0] + rx + t, O[1] + ry, rz1),     # east
        (O[0] - rx, O[1] - ry - t, rz0, O[0] + rx, O[1] - ry, rz1),     # south
        (O[0] - rx, O[1] + ry, rz0, O[0] + rx, O[1] + ry + t, rz1),     # north
    ]
    solids = [_plain_box_solid(ids, *b, SKYBOX_MATERIAL, SKYBOX_MATERIAL,
                               color="120 130 200") for b in boxes]
    org = "%s %s %s" % tuple(_num(c) for c in O)
    prop_scale = 1.0 if prescaled else inv
    ents = [
        _ent(ids, "sky_camera", [("origin", org), ("scale", str(int(sky_scale))),
                                 ("fogenable", "0"), ("angles", "0 0 0")]),
        _prop_static(ids, model_path, O, (0.0, 0.0, 0.0), prop_scale),
    ]
    return solids, ents


def _water_lod_control(ids, cheap_start=1000, cheap_end=2000):
    """Required helper entity so water transitions cheap<->expensive correctly."""
    return ('entity\n{\n\t"id" "%d"\n\t"classname" "water_lod_control"\n'
            '\t"cheapwaterstartdistance" "%d"\n\t"cheapwaterenddistance" "%d"\n'
            '\t"origin" "0 0 0"\n}' % (ids.next(), cheap_start, cheap_end))


# --- outdoor lighting --------------------------------------------------------
# Daytime defaults: warm sun ~45 deg up, cool sky ambient.
DEFAULT_SUN_PITCH = -45         # degrees below horizontal (negative = downward)
DEFAULT_SUN_YAW = 200           # compass direction the light comes from
DEFAULT_SUN_LIGHT = "255 248 230 350"     # r g b brightness
DEFAULT_AMBIENT = "130 145 170 70"        # sky-fill r g b brightness
DEFAULT_FOG_COLOR = "190 200 215"
DEFAULT_FOG_START = 3000
DEFAULT_FOG_END = 14000


def _ent(ids, classname, kv):
    o = ['entity', '{', '\t"id" "%d"' % ids.next(), '\t"classname" "%s"' % classname]
    for k, v in kv:
        o.append('\t"%s" "%s"' % (k, v))
    o += ['\teditor', '{', '\t\t"color" "255 255 100"',
          '\t\t"visgroupshown" "1"', '\t\t"visgroupautoshown" "1"', '\t}', '}']
    return "\n".join(o)


def player_start_origin(cells, scale, recenter, height=96.0):
    """A safe spawn point: the surface height at the cell nearest the recentered
    region centre, lifted ``height`` HU into the air so the player stands on the
    ground (not buried in the displacement / vis-floor solid at world origin)."""
    if not cells:
        return (0.0, 0.0, 64.0)
    x_off, y_off, z_off = _offsets(cells, recenter)
    mid = GRID // 2
    best = None
    for (cx, cy), g in cells.items():
        wx = cx * CELL_SIZE + mid * VERTEX_SPACING
        wy = cy * CELL_SIZE + mid * VERTEX_SPACING
        d = (wx - x_off) ** 2 + (wy - y_off) ** 2
        if best is None or d < best[0]:
            best = (d, wx, wy, g[mid][mid])
    _, wx, wy, h = best
    return ((wx - x_off) * scale, (wy - y_off) * scale, (h - z_off) * scale + height)


def build_player_start(ids, origin):
    """info_player_start so GMod spawns the player on the map instead of at world
    origin (which sits inside the terrain/vis-floor solid → you see nothing)."""
    return [_ent(ids, "info_player_start", [
        ("origin", "%s %s %s" % (_num(origin[0]), _num(origin[1]), _num(origin[2]))),
        ("angles", "0 0 0"),
    ])]


def build_lighting_entities(ids, origin=(0, 0, 1024), sun_pitch=DEFAULT_SUN_PITCH,
                            sun_yaw=DEFAULT_SUN_YAW, sun_light=DEFAULT_SUN_LIGHT,
                            ambient=DEFAULT_AMBIENT, fog=True,
                            fog_color=DEFAULT_FOG_COLOR, fog_start=DEFAULT_FOG_START,
                            fog_end=DEFAULT_FOG_END):
    """A light_environment (sun + sky ambient), a shadow_control (matching sun
    direction), and optionally an env_fog_controller for outdoor haze. Returns a
    list of entity strings. Exactly one light_environment is what vrad needs to
    light the 2D sky."""
    org = "%s %s %s" % (_num(origin[0]), _num(origin[1]), _num(origin[2]))
    ang = "0 %s 0" % _num(sun_yaw)
    ents = []
    ents.append(_ent(ids, "light_environment", [
        ("origin", org), ("angles", ang), ("pitch", _num(sun_pitch)),
        ("_light", sun_light), ("_ambient", ambient),
        ("_lightHDR", "-1 -1 -1 1"), ("_lightscaleHDR", "1"),
        ("_ambientHDR", "-1 -1 -1 1"), ("_AmbientScaleHDR", "1"),
        ("SunSpreadAngle", "5"),
    ]))
    ents.append(_ent(ids, "shadow_control", [
        ("origin", org), ("angles", "%s %s 0" % (_num(-sun_pitch), _num(sun_yaw))),
        ("color", "100 100 110"), ("distance", "150"),
        ("disableallshadows", "0"), ("enableshadows", "1"),
    ]))
    if fog:
        ents.append(_ent(ids, "env_fog_controller", [
            ("origin", org), ("spawnflags", "1"), ("fogenable", "1"),
            ("fogcolor", fog_color), ("fogcolor2", fog_color),
            ("fogstart", _num(fog_start)), ("fogend", _num(fog_end)),
            ("fogmaxdensity", "1"), ("farz", "16384"), ("fogdir", "1 0 0"),
        ]))
    return ents


def _offsets(cells, recenter):
    """World-unit offsets subtracted before scaling. Centers the selected cell
    box on the origin in X/Y and drops the lowest vertex to ~Z=0, so even an
    off-center region (West Weald is ~cell -19) lands inside the Source cube."""
    if not recenter or not cells:
        return 0.0, 0.0, 0.0
    cxs = [c[0] for c in cells]
    cys = [c[1] for c in cells]
    x_off = (min(cxs) * CELL_SIZE + (max(cxs) + 1) * CELL_SIZE) / 2.0
    y_off = (min(cys) * CELL_SIZE + (max(cys) + 1) * CELL_SIZE) / 2.0
    z_off = min(min(min(row) for row in g) for g in cells.values())
    return x_off, y_off, z_off


def build_solids(cells, scale=1.0, power=4, thickness=16.0,
                 material=DEFAULT_MATERIAL, recenter=True, texturer=None, ids=None,
                 tex_scale=0.25, lightmapscale=16, outer_power=None, outer_margin=1,
                 vis_floor=False, vis_floor_depth=512.0):
    """Return (solid_strings, stats) for the given {(cx,cy): heightgrid} dict.

    If ``texturer`` is given, each tile's material and blend alphas come from it
    (Oblivion land textures); otherwise every tile uses ``material`` flat.

    ``outer_power`` (optional, < ``power``) drops cells within ``outer_margin`` of
    the selection's edge to a lower power so distant border terrain is cheaper
    (a coarse, automatic terrain LOD)."""
    for pw in (power, outer_power):
        if pw is None:
            continue
        if pw not in ALLOWED_POWERS:
            raise ValueError("power must be one of %s (Source max is 4)" % (ALLOWED_POWERS,))
        if INTERVALS % (1 << pw) != 0:
            raise ValueError("power %d does not divide %d cell intervals" % (pw, INTERVALS))
    if outer_power is not None and outer_power > power:
        raise ValueError("outer_power must be <= power")

    x_off, y_off, z_off = _offsets(cells, recenter)
    ids = ids or _Ids()
    solids = []
    minc = [float("inf")] * 3
    maxc = [float("-inf")] * 3

    # Optional: under each displacement, a SEPARATE solid nodraw brush down to one
    # shared floor. Regular (non-displacement) brushes are what vvis actually uses
    # for visibility, so this lets hills occlude; the shared floor also seals the
    # map bottom. (Thickening the displacement brush itself doesn't work — vvis
    # ignores displacement geometry.)
    floor_z = None
    vis_solids = []
    if vis_floor:
        gmin = min(min(min(row) for row in g) for g in cells.values())
        floor_z = (gmin - z_off) * scale - vis_floor_depth

    # bounds for the "outer ring" test
    cxs = [c[0] for c in cells]
    cys = [c[1] for c in cells]
    bx0, bx1, by0, by1 = min(cxs), max(cxs), min(cys), max(cys)

    def _cell_power(cx, cy):
        if outer_power is None:
            return power
        near_edge = (cx - bx0 < outer_margin or bx1 - cx < outer_margin or
                     cy - by0 < outer_margin or by1 - cy < outer_margin)
        return outer_power if near_edge else power

    for (cx, cy), grid in sorted(cells.items()):
        ox, oy = cx * CELL_SIZE, cy * CELL_SIZE
        pw = _cell_power(cx, cy)
        n = 1 << pw
        tiles = INTERVALS // n
        span = n * VERTEX_SPACING                   # game units covered by one tile
        for ti in range(tiles):                     # +Y (north)
            for tj in range(tiles):                 # +X (east)
                samples = [
                    [(grid[ti * n + r][tj * n + c] - z_off) * scale for c in range(n + 1)]
                    for r in range(n + 1)
                ]
                ztop = min(min(row) for row in samples)
                x0 = (ox + tj * span - x_off) * scale
                x1 = (ox + (tj + 1) * span - x_off) * scale
                y0 = (oy + ti * span - y_off) * scale
                y1 = (oy + (ti + 1) * span - y_off) * scale
                if texturer is not None:
                    mat, blend = texturer.tile(cx, cy, ti * n, tj * n, n)
                else:
                    mat, blend = material, None
                solids.append(
                    _tile_solid(ids, x0, y0, x1, y1, ztop, thickness, samples,
                                pw, mat, blend, tex_scale=tex_scale,
                                lightmapscale=lightmapscale)   # disp stays thin
                )
                zbot = ztop - thickness
                if floor_z is not None and zbot - floor_z > 1.0:
                    # separate solid nodraw vis-blocker filling under the tile
                    vis_solids.append(_plain_box_solid(
                        ids, x0, y0, floor_z, x1, y1, zbot,
                        NODRAW_MATERIAL, NODRAW_MATERIAL, color="80 80 80"))
                    zbot = floor_z
                for axis, lo, hi in ((0, x0, x1), (1, y0, y1), (2, zbot, ztop)):
                    minc[axis] = min(minc[axis], lo)
                    maxc[axis] = max(maxc[axis], hi)
                for row in samples:
                    maxc[2] = max(maxc[2], max(row))

    n_disp = len(solids)
    solids.extend(vis_solids)

    stats = {
        "cells": len(cells),
        "displacements": n_disp,
        "vis_brushes": len(vis_solids),
        "power": power,
        "outer_power": outer_power,
        "scale": scale,
        "recentered": recenter,
        "mins": tuple(minc) if solids else None,
        "maxs": tuple(maxc) if solids else None,
    }
    return solids, stats


_HEADER = """versioninfo
{
	"editorversion" "400"
	"editorbuild" "8000"
	"mapversion" "1"
	"formatversion" "100"
	"prefab" "0"
}
visgroups
{
}
viewsettings
{
	"bSnapToGrid" "1"
	"bShowGrid" "1"
	"nGridSpacing" "64"
}
"""

_FOOTER = """cameras
{
	"activecamera" "-1"
}
cordons
{
	"active" "0"
}
"""


# Prop-fade tuning. (start, end) in Hammer units: a prop begins fading at
# ``start`` and is gone by ``end``. None = never fade. Categories are matched
# against the model path (first substring hit wins). Tuned for the ~32k-HU West
# Weald diagonal; scale these if you convert a much larger/smaller box.
FADE_CATEGORY_RULES = (
    ("bush_", "bush"),
    ("tree_deciduous", "tree"), ("_tree_", "tree"), ("kvatchtree", "tree"),
    ("clutter_", "clutter"), ("furniture_", "clutter"),
    ("plants_", "plants"), ("flora", "plants"),
    ("rocks_", "rock"),
    ("architecture", "building"), ("dungeons", "building"),
)
DEFAULT_FADE_TABLE = {
    "clutter": (1200, 2000),
    "plants": (1000, 1800),
    "bush": (2500, 3800),
    "rock": (3500, 5200),
    "tree": (5000, 8000),
    "building": (11000, 14000),   # landmarks fade only at the fog wall
    "default": (3500, 5500),
}


def _fade_for(model_path, table, scale=1.0):
    low = model_path.lower()
    fade = table.get("default")
    for sub, cat in FADE_CATEGORY_RULES:
        if sub in low:
            fade = table.get(cat, table.get("default"))
            break
    if fade is None:
        return None
    return (fade[0] * scale, fade[1] * scale)


def _prop_static(ids, model, origin, angles, scale, fade=None):
    o = []
    o.append("entity")
    o.append("{")
    o.append('\t"id" "%d"' % ids.next())
    o.append('\t"classname" "prop_static"')
    o.append('\t"model" "%s"' % model)
    o.append('\t"origin" "%s %s %s"' % (_num(origin[0]), _num(origin[1]), _num(origin[2])))
    o.append('\t"angles" "%s %s %s"' % (_num(angles[0]), _num(angles[1]), _num(angles[2])))
    o.append('\t"uniformscale" "%s"' % _num(scale))
    o.append('\t"solid" "6"')
    o.append('\t"skin" "0"')
    if fade:
        o.append('\t"fademindist" "%s"' % _num(fade[0]))
        o.append('\t"fademaxdist" "%s"' % _num(fade[1]))
    else:
        o.append('\t"fademindist" "-1"')
    o.append('\t"fadescaledist" "0"')
    o.append('\t"disableshadows" "0"')
    o.append("\teditor")
    o.append("\t{")
    o.append('\t\t"color" "255 200 0"')
    o.append('\t\t"visgroupshown" "1"')
    o.append('\t\t"visgroupautoshown" "1"')
    o.append("\t}")
    o.append("}")
    return "\n".join(o)


def build_prop_entities(placements, model_map, ids, scale, offsets,
                        angle_sign=-1.0, yaw_offset=-90.0, model_scale=None,
                        prop_fade=False, fade_table=None, fade_scale=1.0,
                        skip_models=None):
    """Build prop_static entity strings from {(cx,cy): [placement]} dicts.
    ``model_map`` maps a base-object FormID -> compiled model path (or None to skip).
    ``model_scale`` optionally maps a FormID -> extra uniform-scale multiplier.
    If ``prop_fade``, each prop gets distance fade by category (``fade_table`` or
    the default). Returns (entity_strings, placed_count, skipped_count)."""
    from .props import placement_origin, refr_to_angles
    model_scale = model_scale or {}
    table = fade_table or DEFAULT_FADE_TABLE
    skip_models = skip_models or set()
    ents = []
    skipped = oob = 0
    for cell in sorted(placements):
        for p in placements[cell]:
            model = model_map.get(p["base"])
            if not model or model in skip_models:
                skipped += 1
                continue
            origin = placement_origin(p["pos"], offsets, scale)
            # Skip props parked outside the map cube — Oblivion stashes distant-LOD
            # building meshes (and disabled refs) at e.g. z=-30000, which would land
            # far underground / off-map. The real (non-parked) prop is placed normally.
            if any(abs(c) > MAX_COORD for c in origin):
                oob += 1
                continue
            angles = refr_to_angles(*p["rot"], sign=angle_sign, yaw_offset=yaw_offset)
            psc = p["scale"] * model_scale.get(p["base"], 1.0)
            fade = _fade_for(model, table, fade_scale) if prop_fade else None
            ents.append(_prop_static(ids, model, origin, angles, psc, fade=fade))
    return ents, len(ents), skipped + oob


def write_vmf(cells, out_path, scale=1.0, power=4, thickness=16.0,
              material=DEFAULT_MATERIAL, skyname=DEFAULT_SKY, recenter=True,
              texturer=None, placements=None, model_map=None, angle_sign=-1.0,
              yaw_offset=-90.0, tex_scale=0.25, model_scale=None,
              water=False, cell_water=None, water_material=WATER_MATERIAL,
              ws_default_water=None, lightmapscale=16, outer_power=None,
              outer_margin=1, prop_fade=False, fade_table=None,
              lighting=True, fog=True, sun_pitch=DEFAULT_SUN_PITCH,
              sun_yaw=DEFAULT_SUN_YAW, vis_floor=False, vis_floor_depth=512.0,
              seal_sky=False, fade_scale=1.0, skybox_model=None, skybox_bounds=None,
              sky_scale=16, skip_models=None, skybox_model_prescaled=True):
    """Generate a .vmf from {(cx,cy): heightgrid} and write it to ``out_path``.

    If ``placements`` + ``model_map`` are given, prop_static entities are emitted
    alongside the terrain. If ``water`` is set, a swimmable water volume is added
    per cell that has water (heights from ``cell_water``). Returns the stats dict
    from :func:`build_solids` (with "props" / "props_skipped" / "water" added).
    """
    ids = _Ids()
    solids, stats = build_solids(cells, scale=scale, power=power,
                                 thickness=thickness, material=material,
                                 recenter=recenter, texturer=texturer, ids=ids,
                                 tex_scale=tex_scale, lightmapscale=lightmapscale,
                                 outer_power=outer_power, outer_margin=outer_margin,
                                 vis_floor=vis_floor, vis_floor_depth=vis_floor_depth)
    water_solids = []
    stats["water"] = 0
    if water:
        water_solids, n_water = build_water_solids(
            cells, cell_water or {}, scale=scale, recenter=recenter, ids=ids,
            water_material=water_material, default_water=ws_default_water)
        stats["water"] = n_water
    sky_solids = []
    stats["skybox"] = 0
    if seal_sky:
        sky_solids, n_sky = build_skybox_enclosure(
            cells, scale=scale, recenter=recenter, ids=ids,
            vis_floor_depth=vis_floor_depth)
        stats["skybox"] = n_sky
    skybox_model_ents = []
    stats["skybox_model"] = False
    if skybox_model and skybox_bounds:
        room, skybox_model_ents = build_skybox_model(ids, skybox_model, skybox_bounds,
                                                     sky_scale=sky_scale,
                                                     prescaled=skybox_model_prescaled)
        sky_solids = sky_solids + room
        stats["skybox_model"] = True
    world_id = ids.next()
    parts = [_HEADER]
    parts.append("world")
    parts.append("{")
    parts.append('\t"id" "%d"' % world_id)
    parts.append('\t"mapversion" "1"')
    parts.append('\t"classname" "worldspawn"')
    parts.append('\t"skyname" "%s"' % skyname)
    parts.extend(solids)
    parts.extend(water_solids)
    parts.extend(sky_solids)
    parts.append("}")

    if water_solids:
        parts.append(_water_lod_control(ids))
    parts.extend(skybox_model_ents)              # sky_camera + backdrop prop

    spawn = player_start_origin(cells, scale, recenter)
    parts.extend(build_player_start(ids, spawn))
    stats["player_start"] = spawn

    stats["lighting"] = False
    if lighting:
        parts.extend(build_lighting_entities(ids, sun_pitch=sun_pitch, sun_yaw=sun_yaw,
                                             fog=fog))
        stats["lighting"] = True

    stats["props"] = 0
    stats["props_skipped"] = 0
    if placements and model_map:
        offsets = _offsets(cells, recenter)
        ents, placed, skipped = build_prop_entities(placements, model_map, ids, scale,
                                                    offsets, angle_sign=angle_sign,
                                                    yaw_offset=yaw_offset,
                                                    model_scale=model_scale,
                                                    prop_fade=prop_fade,
                                                    fade_table=fade_table,
                                                    fade_scale=fade_scale,
                                                    skip_models=skip_models)
        parts.extend(ents)
        stats["props"] = placed
        stats["props_skipped"] = skipped

    parts.append(_FOOTER)
    text = "\n".join(parts) + "\n"
    with open(out_path, "w", encoding="ascii") as f:
        f.write(text)
    return stats


def write_interior_vmf(out_path, placements, model_map, scale=1.0, lights=None,
                       light_bases=None, ambient=None, margin=768.0,
                       angle_sign=-1.0, yaw_offset=-90.0, model_scale=None,
                       skip_models=None, skyname=DEFAULT_SKY, skybox_room=False):
    """Generate a .vmf for ONE interior cell (a room): a sealed box sized to the
    placed geometry, the props, light entities from placed LIGH references, and a
    player start.

    The room is always wrapped in six sealing brushes. By default they use
    ``tools/toolsblack`` (a solid, light-tight enclosure). When ``skybox_room`` is
    True the wrap uses ``tools/toolsskybox`` instead, so the enclosure renders the
    2D sky on every face — for cells that are really open-air courtyards/exteriors.

    placements  : {key: [{base,pos,rot,scale}]} — every list is included.
    light_bases : LIGH FormID -> {"radius","color"} (from InteriorExtractor).
    ambient     : optional (r,g,b) cell ambient -> a dim central fill light.
    Returns a stats dict.
    """
    from .props import placement_origin
    all_p = [p for plist in placements.values() for p in plist]
    if not all_p:
        raise ValueError("interior has no object references")

    # recenter: XY midpoint -> 0,0; lowest Z -> 0 (so the floor sits near z=0)
    xs = [p["pos"][0] for p in all_p]
    ys = [p["pos"][1] for p in all_p]
    zs = [p["pos"][2] for p in all_p]
    offsets = ((min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0, min(zs))

    ids = _Ids()
    ents, placed, skipped = build_prop_entities(placements, model_map, ids, scale,
                                                offsets, angle_sign=angle_sign,
                                                yaw_offset=yaw_offset,
                                                model_scale=model_scale,
                                                skip_models=skip_models)

    # room shell from the recentered HU bounds of every placement origin
    origins = [placement_origin(p["pos"], offsets, scale) for p in all_p]
    hx0 = min(o[0] for o in origins) - margin
    hx1 = max(o[0] for o in origins) + margin
    hy0 = min(o[1] for o in origins) - margin
    hy1 = max(o[1] for o in origins) + margin
    hz0 = min(o[2] for o in origins) - margin
    hz1 = max(o[2] for o in origins) + margin
    t = 32.0
    boxes = [
        (hx0, hy0, hz0 - t, hx1, hy1, hz0),          # floor
        (hx0, hy0, hz1, hx1, hy1, hz1 + t),          # ceiling
        (hx0 - t, hy0, hz0, hx0, hy1, hz1),          # west
        (hx1, hy0, hz0, hx1 + t, hy1, hz1),          # east
        (hx0, hy0 - t, hz0, hx1, hy0, hz1),          # south
        (hx0, hy1, hz0, hx1, hy1 + t, hz1),          # north
    ]
    wall = SKYBOX_MATERIAL if skybox_room else "tools/toolsblack"
    wcol = "100 130 170" if skybox_room else "40 40 40"
    solids = [_plain_box_solid(ids, *b, wall, wall, color=wcol) for b in boxes]

    # lights: one per placed LIGH reference (colour/radius from the base record)
    light_ents = []
    if light_bases:
        for plist in placements.values():
            for p in plist:
                lb = light_bases.get(p["base"])
                if not lb:
                    continue
                o = placement_origin(p["pos"], offsets, scale)
                r, g, b = lb["color"]
                if r + g + b < 30:                  # near-black light -> warm default
                    r, g, b = 255, 214, 170
                bright = max(80, min(500, int(lb["radius"] * scale)))
                light_ents.append(_ent(ids, "light", [
                    ("origin", "%s %s %s" % (_num(o[0]), _num(o[1]), _num(o[2]))),
                    ("_light", "%d %d %d %d" % (r, g, b, bright)),
                    ("_lightHDR", "-1 -1 -1 1"), ("_lightscaleHDR", "1"),
                ]))
    cx, cy = (hx0 + hx1) / 2.0, (hy0 + hy1) / 2.0
    if not light_ents:
        amb = ambient or (255, 245, 230)
        light_ents.append(_ent(ids, "light", [
            ("origin", "%s %s %s" % (_num(cx), _num(cy), _num((hz0 + hz1) / 2.0))),
            ("_light", "%d %d %d 300" % amb[:3]),
            ("_lightHDR", "-1 -1 -1 1"), ("_lightscaleHDR", "1"),
        ]))

    spawn_z = min(o[2] for o in origins) + 64.0
    spawn = build_player_start(ids, (cx, cy, spawn_z))

    world_id = ids.next()
    parts = [_HEADER, "world", "{",
             '\t"id" "%d"' % world_id,
             '\t"mapversion" "1"',
             '\t"classname" "worldspawn"',
             '\t"skyname" "%s"' % skyname]
    parts.extend(solids)
    parts.append("}")
    parts.extend(light_ents)
    parts.extend(spawn)
    parts.extend(ents)
    parts.append(_FOOTER)
    with open(out_path, "w", encoding="ascii") as f:
        f.write("\n".join(parts) + "\n")
    return {"props": placed, "props_skipped": skipped, "lights": len(light_ents),
            "bounds": (hx0, hx1, hy0, hy1, hz0, hz1)}


# --- func_instance insertion into a host map ---------------------------------

import os as _os
import re as _re

_EMPTY_HOST = (_HEADER + "world\n{\n"
               '\t"id" "1"\n\t"mapversion" "1"\n\t"classname" "worldspawn"\n'
               '\t"skyname" "%s"\n}\n' % DEFAULT_SKY) + _FOOTER

# lane width for laying instances side by side along +X (Source +/-16384 world)
_INSTANCE_LANE = 8192.0
_INSTANCE_MARGIN = 1024.0


def _vmf_coord_bounds(text):
    """Axis-aligned bounds of every brush-plane vertex in a VMF string, as
    (minx,maxx,miny,maxy,minz,maxz), or None if the file has no brushwork."""
    xs = ys = zs = None
    lo = [None, None, None]
    hi = [None, None, None]
    for m in _re.finditer(r'"plane"\s+"([^"]+)"', text):
        for tri in _re.findall(r'\(([^)]+)\)', m.group(1)):
            p = tri.split()
            if len(p) != 3:
                continue
            for i in range(3):
                v = float(p[i])
                lo[i] = v if lo[i] is None else min(lo[i], v)
                hi[i] = v if hi[i] is None else max(hi[i], v)
    if lo[0] is None:
        return None
    return (lo[0], hi[0], lo[1], hi[1], lo[2], hi[2])


def _vmf_max_id(text):
    ids = [int(m.group(1)) for m in _re.finditer(r'"id"\s+"(\d+)"', text)]
    return max(ids) if ids else 0


def _existing_instances(text):
    """Return the list of func_instance origins already present in the host."""
    out = []
    for m in _re.finditer(r'"classname"\s+"func_instance".*?(?="classname"|\Z)',
                          text, _re.S):
        om = _re.search(r'"origin"\s+"(-?[\d.]+) (-?[\d.]+) (-?[\d.]+)"', m.group(0))
        if om:
            out.append(tuple(float(x) for x in om.groups()))
    return out


def _auto_instance_origin(host_text, inst_bounds):
    """Pick a free origin for a new instance: lay it in a lane just past the host's
    existing content (and any instances already placed), aligned so its own
    bounding box clears that edge. Returns (x, y, z)."""
    hb = _vmf_coord_bounds(host_text)
    host_max_x = hb[1] if hb else 0.0
    insts = _existing_instances(host_text)
    if insts:
        host_max_x = max(host_max_x, max(o[0] for o in insts) + _INSTANCE_LANE)
    if inst_bounds:
        # shift so the room's left edge clears host_max_x by the margin
        x = host_max_x + _INSTANCE_MARGIN - inst_bounds[0]
        z = -inst_bounds[4]                          # drop floor to z=0
    else:
        x = host_max_x + _INSTANCE_MARGIN
        z = 0.0
    return (x, 0.0, z)


def add_instance_to_vmf(host_path, instance_path, origin=None, angles=(0, 0, 0),
                        targetname=None):
    """Insert ``instance_path`` into ``host_path`` as a ``func_instance`` whose
    ``file`` is stored relative to the host (forward slashes). If ``host_path``
    doesn't exist a minimal empty host is created. When ``origin`` is None a free
    spot is found automatically beside the host's existing content. Returns
    ``(origin, rel_file)``."""
    if _os.path.isfile(host_path):
        with open(host_path, "r", encoding="ascii", errors="replace") as f:
            text = f.read()
    else:
        text = _EMPTY_HOST

    with open(instance_path, "r", encoding="ascii", errors="replace") as f:
        inst_bounds = _vmf_coord_bounds(f.read())

    if origin is None:
        origin = _auto_instance_origin(text, inst_bounds)

    rel = _os.path.relpath(_os.path.abspath(instance_path),
                           _os.path.dirname(_os.path.abspath(host_path)))
    rel = rel.replace("\\", "/")

    nid = _vmf_max_id(text)
    kv = ['entity', '{',
          '\t"id" "%d"' % (nid + 1),
          '\t"classname" "func_instance"',
          '\t"angles" "%s %s %s"' % (_num(angles[0]), _num(angles[1]), _num(angles[2])),
          '\t"file" "%s"' % rel,
          '\t"fixup_style" "0"',
          '\t"origin" "%s %s %s"' % (_num(origin[0]), _num(origin[1]), _num(origin[2]))]
    if targetname:
        kv.append('\t"targetname" "%s"' % targetname)
    kv += ['\teditor', '{', '\t\t"color" "0 180 0"',
           '\t\t"visgroupshown" "1"', '\t\t"visgroupautoshown" "1"', '\t}', '}']
    ent = "\n".join(kv)

    # insert before the trailing cameras/cordons footer if present, else append
    idx = text.rfind("\ncameras")
    if idx == -1:
        idx = text.rfind("\ncordon")
    if idx == -1:
        new = text.rstrip("\n") + "\n" + ent + "\n"
    else:
        new = text[:idx + 1] + ent + "\n" + text[idx + 1:]

    with open(host_path, "w", encoding="ascii") as f:
        f.write(new)
    return origin, rel
