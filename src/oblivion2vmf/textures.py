
from __future__ import annotations

import os
import re
import struct

from .dds import dds_to_vtf
from .land import INTERVALS

QUAD = INTERVALS // 2  # 16: a quadrant spans 17 vertices (0..16) along each axis

# Ground type -> representative RGB colour (tinted so quadrants read distinctly).
COLORS = {
    "grass":   (86, 122, 58),
    "dirt":    (120, 94, 64),
    "rock":    (122, 120, 112),
    "sand":    (194, 178, 128),
    "snow":    (236, 239, 245),
    "mud":     (84, 68, 50),
    "gravel":  (143, 138, 128),
    "road":    (108, 100, 90),
    "moss":    (74, 96, 56),
    "forest":  (78, 92, 56),
    "default": (120, 94, 64),
}

# Source $surfaceprop per ground type (footstep / impact sounds).
SURFACEPROP = {
    "grass": "grass", "dirt": "dirt", "rock": "rock", "sand": "sand",
    "snow": "snow", "mud": "dirt", "gravel": "gravel", "road": "dirt",
    "moss": "grass", "forest": "grass", "default": "dirt",
}

# Keyword -> ground type, checked in order (specific before generic). Matched
# against the LTEX ICON path and EditorID, lowercased.
_KEYWORDS = [
    (("snow", "ice", "frost"), "snow"),
    (("sand", "beach", "dune", "desert"), "sand"),
    (("gravel", "pebble"), "gravel"),
    (("rock", "stone", "cliff", "mountain", "boulder"), "rock"),
    (("road", "path", "trail", "cobble"), "road"),
    (("moss",), "moss"),
    (("leaf", "leaves", "forest"), "forest"),
    (("grass", "field", "meadow", "farm", "plain", "lawn", "lush"), "grass"),
    (("mud", "swamp", "bog"), "mud"),
    (("dirt", "ground", "earth", "soil", "terrain"), "dirt"),
]


def classify(name):
    """Map an LTEX ICON/EditorID string to a ground type."""
    s = (name or "").lower()
    for keys, ground in _KEYWORDS:
        if any(k in s for k in keys):
            return ground
    return "default"


def _clamp8(v):
    return 0 if v < 0 else 255 if v > 255 else int(v)


def _bgr888_level(w, h, rgb, noise):
    """One BGR888 mip level: dithered at full res, flat for smaller mips."""
    r, g, b = rgb
    px = bytearray(w * h * 3)
    i = 0
    for y in range(h):
        for x in range(w):
            if noise:
                n = ((x * 73856093) ^ (y * 19349663)) & 0xFF
                d = (n % (2 * noise + 1)) - noise
            else:
                d = 0
            px[i] = _clamp8(b + d)
            px[i + 1] = _clamp8(g + d)
            px[i + 2] = _clamp8(r + d)
            i += 3
    return bytes(px)


def write_vtf(path, rgb, size=64, noise=10, mips=True):
    """Write a flat-colour BGR888 VTF (v7.2) the engine loads directly. With
    ``mips`` (default) a full mip chain is written so the texture trilinearly
    minifies at distance (no NOMIP shimmer / VRAM waste)."""
    r, g, b = rgb
    levels = []                       # (w, h, bytes), largest first
    w = hgt = size
    lvl = 0
    while True:
        levels.append((w, hgt, _bgr888_level(w, hgt, rgb, noise if lvl == 0 else 0)))
        if not mips or (w == 1 and hgt == 1):
            break
        w = max(1, w // 2); hgt = max(1, hgt // 2); lvl += 1
    mipcount = len(levels)
    flags = 0 if mipcount > 1 else 0x0300            # clear NOMIP|NOLOD when we have mips

    h = bytearray()
    h += b"VTF\x00"                                  # signature
    h += struct.pack("<II", 7, 2)                    # version 7.2
    h += struct.pack("<I", 80)                       # headerSize
    h += struct.pack("<HH", size, size)              # width, height
    h += struct.pack("<I", flags)                    # flags
    h += struct.pack("<HH", 1, 0)                    # frames, firstFrame
    h += b"\x00" * 4                                  # padding0 (aligns reflectivity)
    h += struct.pack("<fff", r / 255.0, g / 255.0, b / 255.0)  # reflectivity
    h += b"\x00" * 4                                  # padding1
    h += struct.pack("<f", 1.0)                      # bumpmapScale
    h += struct.pack("<I", 3)                        # highResImageFormat = BGR888
    h += struct.pack("<B", mipcount)                 # mipmapCount
    h += struct.pack("<I", 0xFFFFFFFF)               # lowResImageFormat = NONE
    h += struct.pack("<BB", 0, 0)                    # lowResWidth, lowResHeight
    h += struct.pack("<H", 1)                        # depth (7.2)
    h += b"\x00" * (80 - len(h))                      # pad to headerSize
    assert len(h) == 80, len(h)

    with open(path, "wb") as f:
        f.write(h)
        for _, _, blk in reversed(levels):           # VTF stores smallest mip first
            f.write(blk)


def write_vtf_rgb(path, pixels, w, h):
    """Write a w*h BGR888 VTF (v7.2, single mip) from ``pixels`` — a flat list of
    (r,g,b) in row-major order, row 0 at the TOP. Used for baked top-down terrain
    textures. Dimensions should be powers of two for the engine to be happy."""
    body = bytearray(w * h * 3)
    for i, (r, g, b) in enumerate(pixels):
        body[i * 3] = _clamp8(b)
        body[i * 3 + 1] = _clamp8(g)
        body[i * 3 + 2] = _clamp8(r)
    hdr = bytearray(b"VTF\x00")
    hdr += struct.pack("<II", 7, 2)
    hdr += struct.pack("<I", 80)
    hdr += struct.pack("<HH", w, h)
    hdr += struct.pack("<I", 0x0300)                 # NOMIP|NOLOD
    hdr += struct.pack("<HH", 1, 0)
    hdr += b"\x00" * 4
    hdr += struct.pack("<fff", 0.5, 0.5, 0.5)
    hdr += b"\x00" * 4
    hdr += struct.pack("<f", 1.0)
    hdr += struct.pack("<I", 3)                      # BGR888
    hdr += struct.pack("<B", 1)                      # 1 mip
    hdr += struct.pack("<I", 0xFFFFFFFF)
    hdr += struct.pack("<BB", 0, 0)
    hdr += struct.pack("<H", 1)
    hdr += b"\x00" * (80 - len(hdr))
    with open(path, "wb") as f:
        f.write(hdr)
        f.write(body)


def write_water_material(materials_root, prefix="oblivion2vmf", name="water",
                         fog_rgb=(31, 54, 52), fog_end=1200):
    """Write a self-contained, swimmable Source water material into
    materials/<prefix>/: a cheap `Water` shader (top face), a flat normal map, and
    an opaque bottom material. `%compilewater` marks the brush volume as water so
    the player can swim. No reflection RTs/cubemaps required, so it renders without
    extra setup. Returns the material path 'prefix/name' for use as a face material."""
    d = os.path.join(materials_root, prefix)
    os.makedirs(d, exist_ok=True)
    # flat tangent-space normal (no perturbation) + an opaque sea-bottom texture
    write_vtf(os.path.join(d, name + "_normal.vtf"), (128, 128, 255), size=64, noise=0)
    write_vtf(os.path.join(d, name + "_bottom.vtf"),
              (fog_rgb[0] + 8, fog_rgb[1] + 12, fog_rgb[2] + 12), size=64)
    with open(os.path.join(d, name + "_bottom.vmt"), "w", encoding="ascii") as f:
        f.write('"LightmappedGeneric"\n{\n'
                '\t"$basetexture" "%s/%s_bottom"\n'
                '\t"$surfaceprop" "water"\n}\n' % (prefix, name))
    r, g, b = fog_rgb
    with open(os.path.join(d, name + ".vmt"), "w", encoding="ascii") as f:
        f.write(
            '"Water"\n{\n'
            '\t"$normalmap" "%s/%s_normal"\n'
            '\t"$bottommaterial" "%s/%s_bottom"\n'
            '\t"%%tooltexture" "%s/%s_bottom"\n'
            '\t"%%compilewater" "1"\n'
            '\t"$surfaceprop" "water"\n'
            '\t"$abovewater" "1"\n'
            '\t"$forceexpensive" "0"\n'
            '\t"$fogenable" "1"\n'
            '\t"$fogcolor" "{ %d %d %d }"\n'
            '\t"$fogstart" "0"\n'
            '\t"$fogend" "%d"\n'
            '\t"$scale" "[1 1]"\n}\n'
            % (prefix, name, prefix, name, prefix, name, r, g, b, fog_end))
    return "%s/%s" % (prefix, name)


_SINGLE_VMT = (
    '"LightmappedGeneric"\n{\n'
    '\t"$basetexture" "%s/tex_%s"\n'
    '\t"$surfaceprop" "%s"\n}\n'
)

_BLEND_VMT = (
    '"WorldVertexTransition"\n{\n'
    '\t"$basetexture" "%s/tex_%s"\n'
    '\t"$surfaceprop" "%s"\n'
    '\t"$basetexture2" "%s/tex_%s"\n'
    '\t"$surfaceprop2" "%s"\n}\n'
)


def _blend4_vmt(prefix, grounds):
    """Lightmapped_4WayBlend material. grounds = (g1,g2,g3,g4); channel 1 is the
    base (shown where upper weights are 0), 2/3/4 layer on via multiblend weights.
    Per-layer modifiers use the $textureN_* prefix (not $basetextureN_)."""
    g1, g2, g3, g4 = grounds
    L = []
    L.append('"Lightmapped_4WayBlend"')
    L.append("{")
    L.append('\t"$basetexture" "%s/tex_%s"' % (prefix, g1))
    L.append('\t"$surfaceprop" "%s"' % SURFACEPROP.get(g1, "dirt"))
    L.append('\t"$texture1_uvscale" "[1 1]"')   # keep all layers at the same tiling
    L.append('\t"$texture1_lumstart" "0.0"')
    L.append('\t"$texture1_lumend" "1.0"')
    for i, g in ((2, g2), (3, g3), (4, g4)):
        L.append('\t"$basetexture%d" "%s/tex_%s"' % (i, prefix, g))
        L.append('\t"$surfaceprop%d" "%s"' % (i, SURFACEPROP.get(g, "dirt")))
        L.append('\t"$texture%d_uvscale" "[1 1]"' % i)
        L.append('\t"$texture%d_lumstart" "0.0"' % i)
        L.append('\t"$texture%d_lumend" "1.0"' % i)
        L.append('\t"$texture%d_blendstart" "0.0"' % i)
        L.append('\t"$texture%d_blendend" "1.0"' % i)
    L.append("}")
    return "\n".join(L) + "\n"


def _tex_token(icon):
    """Stable token for a real LTEX texture path (relative to textures\\)."""
    s = icon.replace("/", "\\").lower()
    if s.startswith("textures\\"):
        s = s[len("textures\\"):]
    if s.endswith(".dds"):
        s = s[:-4]
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    return s or "default"


class MaterialLibrary:
    """Tracks the textures + blend combos used and writes the .vtf/.vmt files.

    A "token" identifies one texture layer. In real-texture mode (``source``
    given) a token is registered with its LTEX .dds path and transcoded to .vtf;
    otherwise it's a ground-type name painted as a flat colour. Material paths are
    relative to ``materials/`` (no extension)."""

    def __init__(self, prefix="oblivion2vmf", source=None):
        self.prefix = prefix
        self.source = source
        self.tokens = {}      # token -> (kind, value, ground)  kind in {"flat","tex"}
        self.singles = set()
        self.blends = set()
        self.blends4 = set()

    def _ensure(self, token):
        self.tokens.setdefault(token, ("flat", token, token))

    def register_tex(self, icon):
        """Register a real texture by its LTEX ICON path; returns its token."""
        token = _tex_token(icon)
        self.tokens[token] = ("tex", icon, classify(icon))
        return token

    def material_for(self, base, overlay):
        self._ensure(base)
        if overlay and overlay != base:
            self._ensure(overlay)
            self.blends.add((base, overlay))
            return "%s/blend_%s_%s" % (self.prefix, base, overlay)
        self.singles.add(base)
        return "%s/%s" % (self.prefix, base)

    def material_for_4way(self, tokens):
        tokens = tuple(tokens)
        for t in tokens:
            self._ensure(t)
        self.blends4.add(tokens)
        return "%s/blend4_%s" % (self.prefix, "_".join(tokens))

    def _surfaceprop(self, token):
        return SURFACEPROP.get(self.tokens.get(token, (None, None, "default"))[2], "dirt")

    def write(self, materials_root, size=512):    # match real land textures so flat
                                                  # fallbacks tile at the same world rate
        out_dir = os.path.join(materials_root, self.prefix)
        os.makedirs(out_dir, exist_ok=True)
        real = fallback = 0
        for token, (kind, value, ground) in sorted(self.tokens.items()):
            path = os.path.join(out_dir, "tex_%s.vtf" % token)
            vtf = None
            if kind == "tex" and self.source is not None:
                dds = self.source.get_landscape_texture(value)   # LTEX ICON -> textures\landscape\
                vtf = dds_to_vtf(dds) if dds else None
            if vtf is not None:
                with open(path, "wb") as f:
                    f.write(vtf)
                real += 1
            else:
                write_vtf(path, COLORS.get(ground, COLORS["default"]), size=size)
                fallback += 1
        for t in sorted(self.singles):
            with open(os.path.join(out_dir, "%s.vmt" % t), "w", encoding="ascii") as f:
                f.write(_SINGLE_VMT % (self.prefix, t, self._surfaceprop(t)))
        for base, over in sorted(self.blends):
            with open(os.path.join(out_dir, "blend_%s_%s.vmt" % (base, over)), "w",
                      encoding="ascii") as f:
                f.write(_BLEND_VMT % (self.prefix, base, self._surfaceprop(base),
                                      self.prefix, over, self._surfaceprop(over)))
        for tokens in sorted(self.blends4):
            with open(os.path.join(out_dir, "blend4_%s.vmt" % "_".join(tokens)), "w",
                      encoding="ascii") as f:
                f.write(_blend4_vmt(self.prefix, tokens))
        return {
            "textures": len(self.tokens),
            "real": real,
            "fallback": fallback,
            "single_materials": len(self.singles),
            "blend_materials": len(self.blends),
            "blend4_materials": len(self.blends4),
            "dir": out_dir,
        }


class Texturer:
    """Resolves, per displacement tile, the Source material and per-vertex blend
    data from the parsed Oblivion LTEX + per-cell quadrant texture data.

    ``tile()`` returns ``(material_path, blend)`` where ``blend`` is:
      * ``None``                                   -> single material, no blend
      * ``{"mode": "alpha", "alphas": grid}``      -> 2-way WorldVertexTransition
      * ``{"mode": "multiblend", "weights": grid}``-> 4-way Lightmapped_4WayBlend
    """

    def __init__(self, ltex, cell_textures, prefix="oblivion2vmf", fourway=False, source=None):
        self.ltex = ltex                      # {formid: {"icon","edid"}}
        self.cell_textures = cell_textures    # {(cx,cy): {quad: {"base","overs"}}}
        self.fourway = fourway
        self.real = source is not None
        self.lib = MaterialLibrary(prefix, source=source)
        self._default_icon = self._dominant_base_icon()

    def _dominant_base_icon(self):
        """The most common real base-texture ICON in the selection, used to
        substitute for any missing/null layer (so we never fall back to the
        finely-tiling flat placeholder when real textures are available)."""
        from collections import Counter
        counts = Counter()
        for quads in self.cell_textures.values():
            for q in quads.values():
                rec = self.ltex.get(q.get("base"))
                if rec and rec.get("icon"):
                    counts[rec["icon"]] += 1
        return counts.most_common(1)[0][0] if counts else None

    def _token(self, formid):
        """Material token for an LTEX FormID: a real-texture token when a texture
        source is available, else a flat ground-type token."""
        rec = self.ltex.get(formid) if formid else None
        icon = rec.get("icon") if rec else None
        if self.real:
            if icon:
                return self.lib.register_tex(icon)
            if self._default_icon:                 # missing layer -> dominant real texture
                return self.lib.register_tex(self._default_icon)
        return classify(icon or (rec.get("edid") if rec else "") or "") if rec else "default"

    def tile(self, cx, cy, r0, c0, n):
        """Resolve the tile whose SW vertex is cell-grid index (r0=row/+Y,
        c0=col/+X), spanning n intervals."""
        qrow = 0 if r0 < QUAD else 1
        qcol = 0 if c0 < QUAD else 1
        quad = qrow * 2 + qcol
        info = self.cell_textures.get((cx, cy), {}).get(quad)
        base_fid = info.get("base") if info else None
        overs = info.get("overs") if info else None   # [(formid, 17x17 grid), ...]
        base_g = self._token(base_fid)
        br, bc = qrow * QUAD, qcol * QUAD

        if not overs:
            return self.lib.material_for(base_g, None), None

        if self.fourway:
            return self._tile_4way(base_g, overs, r0, c0, n, br, bc)
        return self._tile_2way(base_g, overs[0], r0, c0, n, br, bc)

    def _tile_2way(self, base_g, over, r0, c0, n, br, bc):
        over_g = self._token(over[0])
        if over_g == base_g:
            return self.lib.material_for(base_g, None), None
        grid = over[1]
        # Canonicalise the pair so neighbouring quadrants that use the same two
        # textures (in either base/overlay role) share ONE material and blend
        # seamlessly across the boundary. material_for(a, b) puts `a` at alpha 0
        # and `b` at alpha 255, so when we swap base<->overlay we invert the alpha.
        if base_g <= over_g:
            material = self.lib.material_for(base_g, over_g)   # base@0, overlay@255
            invert = False
        else:
            material = self.lib.material_for(over_g, base_g)   # overlay@0, base@255
            invert = True
        alphas = [[(255 - grid[r0 - br + r][c0 - bc + c]) if invert
                   else grid[r0 - br + r][c0 - bc + c]
                   for c in range(n + 1)] for r in range(n + 1)]
        return material, {"mode": "alpha", "alphas": alphas}

    def _tile_4way(self, base_g, overs, r0, c0, n, br, bc):
        # Accumulate a per-vertex weight per UNIQUE texture token (base implicit,
        # fills the remainder), then assign the up-to-4 tokens in CANONICAL sorted
        # order so neighbouring quads with the same textures share one material and
        # blend seamlessly regardless of which layer was base/overlay.
        sel = overs[:3]
        over_tokens = [self._token(f) for f, _ in sel]
        uniq = []
        for t in [base_g] + over_tokens:
            if t not in uniq:
                uniq.append(t)
        canon = sorted(uniq[:4])
        idx = {t: i for i, t in enumerate(canon)}

        weights = []
        for r in range(n + 1):
            row = []
            for c in range(n + 1):
                ops = [sel[k][1][r0 - br + r][c0 - bc + c] / 255.0 for k in range(len(sel))]
                w = [0.0, 0.0, 0.0, 0.0]
                if base_g in idx:
                    w[idx[base_g]] += max(0.0, 1.0 - sum(ops))
                for k, t in enumerate(over_tokens):
                    if t in idx:
                        w[idx[t]] += ops[k]
                s = sum(w)
                if s > 0.0:
                    row.append((w[0] / s, w[1] / s, w[2] / s, w[3] / s))
                else:
                    row.append((1.0, 0.0, 0.0, 0.0))
            weights.append(row)

        canon4 = canon + [canon[-1]] * (4 - len(canon))   # pad to 4 channels
        material = self.lib.material_for_4way(tuple(canon4))
        return material, {"mode": "multiblend", "weights": weights}
