"""Minimal streaming parser for Oblivion (TES4) ESM/ESP plugin files.

Walks the worldspace -> cell-block -> cell -> land hierarchy and extracts
exterior-cell heightmaps. Only the records needed for terrain (WRLD / CELL /
LAND) are decoded; everything else is skipped.

Record header (Oblivion / TES4) = 20 bytes:
    char[4] type, uint32 dataSize, uint32 flags, uint32 formID, uint32 vcInfo
GRUP header = 20 bytes:
    "GRUP", uint32 groupSize (incl. header), char[4] label, int32 type, uint32 stamp
Subrecord = char[4] type, uint16 dataSize, data.  (XXXX overrides the next size.)

Reference: UESP "Oblivion Mod:Mod File Format".
"""
from __future__ import annotations

import mmap
import struct
import zlib
from dataclasses import dataclass, field

from .land import decode_vhgt

COMPRESSED = 0x00040000               # record flag: data is zlib-compressed
TAMRIEL_WORLDSPACE = 0x0000003C       # FormID of the Cyrodiil ("Tamriel") worldspace

# Base-object record types that carry a MODL (.nif mesh path) for visible statics.
BASE_MODEL_SIGS = frozenset({b"STAT", b"TREE", b"FLOR", b"FURN", b"ACTI",
                             b"CONT", b"DOOR", b"MSTT", b"LIGH"})

# GRUP group types we care about (UESP):
#   0 Top, 1 World Children, 2 Interior Cell Block, 3 Interior Cell Sub-Block,
#   4 Exterior Cell Block, 5 Exterior Cell Sub-Block, 6 Cell Children,
#   8 Persistent Children, 9 Cell Temporary Children (where LAND lives),
#   10 Visible-Distant Children.
_GT_TOP = 0
_GT_WORLD_CHILDREN = 1
_GT_CELL_CHILD_TYPES = (6, 8, 9, 10)   # label = parent cell FormID


def _u16(b, o):
    return struct.unpack_from("<H", b, o)[0]


def _u32(b, o):
    return struct.unpack_from("<I", b, o)[0]


def _i32(b, o):
    return struct.unpack_from("<i", b, o)[0]


def _zstr(b):
    return b.split(b"\x00", 1)[0].decode("latin-1", "replace")


def read_masters(path):
    """Return the list of master filenames a plugin depends on (from its TES4
    header MAST subrecords), in order."""
    with open(path, "rb") as f:
        head = f.read(24)
        if head[:4] != b"TES4":
            return []
        size = struct.unpack_from("<I", head, 4)[0]
        with open(path, "rb") as f2:
            f2.seek(20)
            body = f2.read(size)
    out = []
    pos = 0
    while pos + 6 <= len(body):
        sig = body[pos:pos + 4]
        sz = _u16(body, pos + 4)
        pos += 6
        if sig == b"MAST":
            out.append(_zstr(body[pos:pos + sz]))
        pos += sz
    return out


def _iter_items(buf, start, end):
    """Yield top-of-block items (records and groups) between ``start`` and ``end``."""
    pos = start
    while pos + 20 <= end:
        sig = bytes(buf[pos:pos + 4])
        if sig == b"GRUP":
            size = _u32(buf, pos + 4)                  # includes the 20-byte header
            label = bytes(buf[pos + 8:pos + 12])
            gtype = _i32(buf, pos + 12)
            yield ("GRUP", sig, label, gtype, pos + 20, pos + size)
            pos += size
        else:
            size = _u32(buf, pos + 4)
            flags = _u32(buf, pos + 8)
            formid = _u32(buf, pos + 12)
            data_start = pos + 20
            data_end = data_start + size
            yield ("REC", sig, flags, formid, data_start, data_end)
            pos = data_end


def _record_data(buf, flags, data_start, data_end):
    raw = bytes(buf[data_start:data_end])
    if flags & COMPRESSED:
        # first 4 bytes = decompressed size; remainder = zlib stream
        return zlib.decompress(raw[4:])
    return raw


def _iter_subrecords(data):
    pos, n = 0, len(data)
    override = None
    while pos + 6 <= n:
        sig = data[pos:pos + 4]
        size = _u16(data, pos + 4)
        pos += 6
        if sig == b"XXXX":                # next subrecord's real size (uint32)
            override = _u32(data, pos)
            pos += size
            continue
        if override is not None:
            size, override = override, None
        yield sig, data[pos:pos + size]
        pos += size


@dataclass
class TerrainExtractor:
    """Extract exterior-cell heightmaps for one worldspace.

    Attributes
    ----------
    target_ws : FormID of the worldspace to read (default Tamriel/Cyrodiil).
    bounds    : (minX, minY, maxX, maxY) inclusive cell-grid box, or None for all.
    list_only : if True, only collect the set of cell coordinates (no decoding).
    cells     : {(cx, cy): 33x33 height grid} for cells in bounds.
    cell_coords : set of every (cx, cy) that has a LAND record in the worldspace.
    """

    target_ws: int = TAMRIEL_WORLDSPACE
    bounds: object = None
    list_only: bool = False
    textures: bool = True
    models: bool = False
    water: bool = False
    cells: dict = field(default_factory=dict)
    cell_coords: set = field(default_factory=set)
    cell_textures: dict = field(default_factory=dict)   # (cx,cy) -> {quad: {"base","overs"}}
    ltex: dict = field(default_factory=dict)            # formid -> {"icon","edid"}
    placements: dict = field(default_factory=dict)      # (cx,cy) -> [{base,pos,rot,scale}]
    base_models: dict = field(default_factory=dict)     # base formid -> .nif path
    cell_water: dict = field(default_factory=dict)      # (cx,cy) -> water height (Obliv units)
    ws_default_water: object = None                     # worldspace default water height
    skipped: int = 0
    _cur_ws: object = None
    _last_cell: object = None
    # REFR placements keyed by the REFR's own FormID so later plugins (mods)
    # override or delete base-game references; flattened to ``placements`` by
    # finalize(). (cx,cy) cell context is stored per entry.
    _refrs: dict = field(default_factory=dict)

    # -- public API ---------------------------------------------------------
    def parse_file(self, path):
        with open(path, "rb") as f:
            buf = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            try:
                self._descend(buf, 0, len(buf), top=True)
            finally:
                buf.close()
        self.finalize()                # idempotent; rebuilds from accumulated _refrs
        return self

    def parse_bytes(self, data):
        self._descend(data, 0, len(data), top=True)
        self.finalize()
        return self

    # -- internals ----------------------------------------------------------
    def _in_bounds(self, cx, cy):
        if self.bounds is None:
            return True
        x0, y0, x1, y1 = self.bounds
        return x0 <= cx <= x1 and y0 <= cy <= y1

    def _descend(self, buf, start, end, top=False):
        for item in _iter_items(buf, start, end):
            if item[0] == "GRUP":
                _, _, label, gtype, gs, ge = item
                if top and gtype == _GT_TOP and not self._want_top(label):
                    continue   # skip top-level trees we don't need
                if gtype == _GT_WORLD_CHILDREN:
                    ws = _u32(label, 0)
                    if self.target_ws is not None and ws != self.target_ws:
                        continue   # different worldspace -> don't even descend
                    saved = self._cur_ws
                    self._cur_ws = ws
                    self._descend(buf, gs, ge)
                    self._cur_ws = saved
                else:
                    self._descend(buf, gs, ge)
            else:
                _, sig, flags, formid, ds, de = item
                if sig == b"WRLD" and self.water:
                    self._on_wrld(buf, flags, formid, ds, de)
                elif sig == b"CELL":
                    self._on_cell(buf, flags, ds, de)
                elif sig == b"LAND":
                    self._on_land(buf, flags, ds, de)
                elif sig == b"LTEX" and self.textures and not self.list_only:
                    self._on_ltex(buf, flags, formid, ds, de)
                elif sig == b"REFR" and self.models and not self.list_only:
                    self._on_refr(buf, flags, formid, ds, de)
                elif sig in BASE_MODEL_SIGS and self.models and not self.list_only:
                    self._on_base_model(buf, flags, formid, ds, de)

    def _want_top(self, label):
        if label == b"WRLD":
            return True
        if label == b"LTEX" and self.textures:
            return True
        if self.models and label in BASE_MODEL_SIGS:
            return True
        return False

    def _on_wrld(self, buf, flags, formid, ds, de):
        """Capture a worldspace's default water height (WNAM/DNAM); used for cells
        flagged as having water but without an explicit XCLW override."""
        if self.target_ws is not None and formid != self.target_ws:
            return
        try:
            data = _record_data(buf, flags, ds, de)
        except zlib.error:
            return
        for sig, payload in _iter_subrecords(data):
            if sig == b"DNAM" and len(payload) >= 8:
                # DNAM = [default land height][default water height] (2 floats)
                self.ws_default_water = struct.unpack_from("<f", payload, 4)[0]
                return

    def _on_cell(self, buf, flags, ds, de):
        try:
            data = _record_data(buf, flags, ds, de)
        except zlib.error:
            self._last_cell = None
            return
        coord = None
        cflags = 0
        water_h = None
        for sig, payload in _iter_subrecords(data):
            if sig == b"XCLC" and len(payload) >= 8:
                coord = (_i32(payload, 0), _i32(payload, 4))
            elif sig == b"DATA" and len(payload) >= 1:
                cflags = payload[0]
            elif sig == b"XCLW" and len(payload) >= 4:
                water_h = struct.unpack_from("<f", payload, 0)[0]
        self._last_cell = coord   # None for interior / persistent dummy cells
        # Record water for exterior cells flagged "has water" (0x02) or carrying an
        # explicit XCLW height. NaN XCLW (used by some cells) -> worldspace default.
        if self.water and coord is not None and ((cflags & 0x02) or water_h is not None):
            if water_h is None or water_h != water_h:        # None or NaN
                water_h = self.ws_default_water
            if water_h is not None:
                self.cell_water[coord] = water_h

    def _on_ltex(self, buf, flags, formid, ds, de):
        try:
            data = _record_data(buf, flags, ds, de)
        except zlib.error:
            return
        edid = icon = ""
        for sig, payload in _iter_subrecords(data):
            if sig == b"EDID":
                edid = _zstr(payload)
            elif sig == b"ICON":
                icon = _zstr(payload)
        self.ltex[formid] = {"edid": edid, "icon": icon}

    def _on_base_model(self, buf, flags, formid, ds, de):
        try:
            data = _record_data(buf, flags, ds, de)
        except zlib.error:
            return
        for sig, payload in _iter_subrecords(data):
            if sig == b"MODL":
                path = _zstr(payload)
                if path:
                    self.base_models[formid] = path
                return

    def _on_refr(self, buf, flags, formid, ds, de):
        if self.target_ws is not None and self._cur_ws != self.target_ws:
            return
        cell = self._last_cell
        if cell is None or not self._in_bounds(*cell):
            return
        # A later plugin can DELETE a base-game reference (record flag 0x20) to,
        # e.g., remove a closed-city wall. Drop any earlier placement of this FormID.
        if flags & 0x20:
            self._refrs.pop(formid, None)
            return
        try:
            data = _record_data(buf, flags, ds, de)
        except zlib.error:
            return
        base = pos = None
        rot = (0.0, 0.0, 0.0)
        scale = 1.0
        for sig, payload in _iter_subrecords(data):
            if sig == b"NAME" and len(payload) >= 4:
                base = _u32(payload, 0)
            elif sig == b"DATA" and len(payload) >= 24:
                v = struct.unpack_from("<6f", payload, 0)
                pos, rot = v[0:3], v[3:6]
            elif sig == b"XSCL" and len(payload) >= 4:
                scale = struct.unpack_from("<f", payload, 0)[0]
        if base and pos is not None:
            # keyed by REFR FormID -> later plugins override the same reference
            self._refrs[formid] = {"cell": cell, "base": base, "pos": pos,
                                   "rot": rot, "scale": scale}

    def finalize(self):
        """Flatten the FormID-keyed REFR table into ``placements`` ({cell:[...]}).
        Call once after parsing the whole load order."""
        self.placements = {}
        for p in self._refrs.values():
            self.placements.setdefault(p["cell"], []).append(
                {"base": p["base"], "pos": p["pos"], "rot": p["rot"], "scale": p["scale"]})
        return self

    def _on_land(self, buf, flags, ds, de):
        if self.target_ws is not None and self._cur_ws != self.target_ws:
            return
        cell = self._last_cell
        if cell is None:
            return
        self.cell_coords.add(cell)
        if self.list_only or not self._in_bounds(*cell):
            return
        try:
            data = _record_data(buf, flags, ds, de)
            height = None
            quads = {}            # quad -> {"base", "over", "_score"}
            pending = None        # (formid, quad) awaiting its VTXT
            for sig, payload in _iter_subrecords(data):
                if sig == b"VHGT":
                    height = decode_vhgt(payload)
                elif not self.textures:
                    continue
                elif sig == b"BTXT" and len(payload) >= 8:
                    fid = _u32(payload, 0)
                    if fid:                          # skip null (0x0) layers
                        quads.setdefault(payload[4], {})["base"] = fid
                elif sig == b"ATXT" and len(payload) >= 8:
                    fid = _u32(payload, 0)
                    pending = (fid, payload[4]) if fid else None
                elif sig == b"VTXT" and pending is not None:
                    fid, quad = pending
                    pending = None
                    self._add_overlay(quads, quad, fid, payload)
            if height is None:
                self.skipped += 1
                return
            self.cells[cell] = height
            if quads:
                self._finalize_quads(quads)
                self.cell_textures[cell] = quads
        except (zlib.error, ValueError, struct.error):
            self.skipped += 1

    # Max overlay layers kept per quadrant (base + up to 3 overlays = 4-way blend).
    MAX_OVERLAYS = 3

    @staticmethod
    def _add_overlay(quads, quad, fid, payload):
        """Decode a VTXT 17x17 alpha grid (0-255) and record it as one overlay
        layer of the quadrant, tagged with its coverage score."""
        grid = [[0] * 17 for _ in range(17)]
        score = 0.0
        for off in range(0, len(payload) - 7, 8):
            pos = _u16(payload, off)
            if 0 <= pos <= 288:
                op = struct.unpack_from("<f", payload, off + 4)[0]
                if op < 0.0:
                    op = 0.0
                elif op > 1.0:
                    op = 1.0
                grid[pos // 17][pos % 17] = int(round(op * 255))
                score += op
        quads.setdefault(quad, {}).setdefault("_overs", []).append((score, fid, grid))

    @classmethod
    def _finalize_quads(cls, quads):
        """Sort each quadrant's overlays by coverage and keep the top N as
        ``overs`` = [(formid, 17x17 grid), ...] (most-covering first)."""
        for q in quads.values():
            raw = q.pop("_overs", None)
            if raw:
                raw.sort(key=lambda t: t[0], reverse=True)
                q["overs"] = [(fid, grid) for _, fid, grid in raw[:cls.MAX_OVERLAYS]]

    # -- convenience --------------------------------------------------------
    def bbox(self):
        """(minX, minY, maxX, maxY) over every cell with a LAND, or None."""
        coords = self.cell_coords or set(self.cells)
        if not coords:
            return None
        xs = [c[0] for c in coords]
        ys = [c[1] for c in coords]
        return (min(xs), min(ys), max(xs), max(ys))


@dataclass
class InteriorExtractor:
    """Extract interior CELLs (rooms) and their object references.

    Interiors live under the top-level CELL group (not a worldspace): interior
    cell block (GRUP type 2) -> sub-block (3) -> CELL record + its children
    groups (6/8/9/10, label = the cell's FormID) holding the REFRs.

    interiors   : cell FormID -> {"edid", "full", "ambient"(r,g,b)|None}
    base_models : base FormID -> .nif path (same record types as the exterior)
    lights      : LIGH FormID -> {"radius", "color"(r,g,b)}
    placements  : cell FormID -> [{base,pos,rot,scale}]  (after finalize())
    """

    interiors: dict = field(default_factory=dict)
    base_models: dict = field(default_factory=dict)
    lights: dict = field(default_factory=dict)
    placements: dict = field(default_factory=dict)
    _refrs: dict = field(default_factory=dict)       # REFR FormID -> {...,"cell":fid}
    _cur_cell: object = None

    def parse_file(self, path):
        with open(path, "rb") as f:
            buf = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            try:
                self._descend(buf, 0, len(buf), top=True)
            finally:
                buf.close()
        self.finalize()
        return self

    def parse_bytes(self, data):
        self._descend(data, 0, len(data), top=True)
        self.finalize()
        return self

    def _descend(self, buf, start, end, top=False):
        for item in _iter_items(buf, start, end):
            if item[0] == "GRUP":
                _, _, label, gtype, gs, ge = item
                if top and gtype == _GT_TOP and not self._want_top(label):
                    continue
                if gtype in _GT_CELL_CHILD_TYPES:
                    saved = self._cur_cell
                    self._cur_cell = _u32(label, 0)
                    self._descend(buf, gs, ge)
                    self._cur_cell = saved
                else:
                    self._descend(buf, gs, ge)
            else:
                _, sig, flags, formid, ds, de = item
                if sig == b"CELL":
                    self._on_cell(buf, flags, formid, ds, de)
                elif sig == b"REFR":
                    self._on_refr(buf, flags, formid, ds, de)
                elif sig in BASE_MODEL_SIGS:
                    self._on_base(buf, flags, formid, ds, de)

    @staticmethod
    def _want_top(label):
        # interiors (CELL top) + every base type that can be referenced in them
        return label == b"CELL" or label in BASE_MODEL_SIGS

    def _on_cell(self, buf, flags, formid, ds, de):
        try:
            data = _record_data(buf, flags, ds, de)
        except zlib.error:
            return
        edid = full = ""
        cflags = 0
        ambient = None
        for sig, payload in _iter_subrecords(data):
            if sig == b"EDID":
                edid = _zstr(payload)
            elif sig == b"FULL":
                full = _zstr(payload)
            elif sig == b"DATA" and len(payload) >= 1:
                cflags = payload[0]
            elif sig == b"XCLL" and len(payload) >= 3:
                ambient = (payload[0], payload[1], payload[2])
        if cflags & 0x01:                       # interior flag
            self.interiors[formid] = {"edid": edid, "full": full, "ambient": ambient}

    def _on_refr(self, buf, flags, formid, ds, de):
        if self._cur_cell is None:
            return
        if flags & 0x20:                        # deleted by a later plugin
            self._refrs.pop(formid, None)
            return
        try:
            data = _record_data(buf, flags, ds, de)
        except zlib.error:
            return
        base = pos = None
        rot = (0.0, 0.0, 0.0)
        scale = 1.0
        for sig, payload in _iter_subrecords(data):
            if sig == b"NAME" and len(payload) >= 4:
                base = _u32(payload, 0)
            elif sig == b"DATA" and len(payload) >= 24:
                v = struct.unpack_from("<6f", payload, 0)
                pos, rot = v[0:3], v[3:6]
            elif sig == b"XSCL" and len(payload) >= 4:
                scale = struct.unpack_from("<f", payload, 0)[0]
        if base and pos is not None:
            self._refrs[formid] = {"cell": self._cur_cell, "base": base, "pos": pos,
                                   "rot": rot, "scale": scale}

    def _on_base(self, buf, flags, formid, ds, de):
        try:
            data = _record_data(buf, flags, ds, de)
        except zlib.error:
            return
        sig0 = bytes(buf[ds - 20:ds - 16])
        for sig, payload in _iter_subrecords(data):
            if sig == b"MODL":
                path = _zstr(payload)
                if path:
                    self.base_models[formid] = path
            elif sig == b"DATA" and sig0 == b"LIGH" and len(payload) >= 12:
                # LIGH DATA: time(i32), radius(u32), color(RGBA), flags(u32), ...
                radius = _u32(payload, 4)
                self.lights[formid] = {"radius": radius,
                                       "color": (payload[8], payload[9], payload[10])}

    def finalize(self):
        """Flatten the FormID-keyed REFR table into ``placements`` ({cellfid:[...]})."""
        self.placements = {}
        for p in self._refrs.values():
            self.placements.setdefault(p["cell"], []).append(
                {"base": p["base"], "pos": p["pos"], "rot": p["rot"], "scale": p["scale"]})
        return self

    def find(self, name):
        """Cell FormID by EDID (case-insensitive); exact match first, then substring.
        Returns (formid, info) or (None, None)."""
        low = name.lower()
        for fid, info in self.interiors.items():
            if info["edid"].lower() == low:
                return fid, info
        hits = [(fid, info) for fid, info in self.interiors.items()
                if low in info["edid"].lower() or low in info["full"].lower()]
        if len(hits) == 1:
            return hits[0]
        return None, None
