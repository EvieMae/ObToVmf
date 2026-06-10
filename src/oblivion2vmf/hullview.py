"""A dependency-free 3D model viewer + collision-hull box editor (Tkinter).

Renders a compiled model's reference SMD as a rotatable wireframe on a plain Tk
Canvas (software projection — no OpenGL/matplotlib needed) and lets you author
multiple axis-aligned box hulls overlaid on it. Coordinates are the SMD's own
units (Hammer units, i.e. already scaled), so the boxes you draw here map 1:1 to
what the build writes as ``$collisionmodel`` convex pieces.

Use :func:`open_hull_editor`; it calls ``on_save(hulls)`` with a list of
``[x0,y0,z0,x1,y1,z1]`` boxes when the user saves.
"""
from __future__ import annotations

import math
import os
import tkinter as tk
from tkinter import ttk

# 12 edges of a box given 8 corners ordered bottom 0-3 then top 4-7
_BOX_EDGES = [(0, 1), (1, 2), (2, 3), (3, 0),
              (4, 5), (5, 6), (6, 7), (7, 4),
              (0, 4), (1, 5), (2, 6), (3, 7)]

_MAX_DISPLAY_TRIS = 2500          # stride large meshes so rotation stays smooth


def read_smd_mesh(path):
    """Parse an SMD's ``triangles`` block -> (verts, tris). verts is a list of
    (x,y,z); tris is a list of (i0,i1,i2) into verts (deduped corners)."""
    verts, tris = [], []
    index = {}
    with open(path, "r", encoding="ascii", errors="replace") as f:
        lines = f.read().splitlines()
    i = 0
    n = len(lines)
    while i < n and lines[i].strip() != "triangles":
        i += 1
    i += 1
    while i < n:
        line = lines[i].strip()
        i += 1
        if line == "end" or not line:
            if line == "end":
                break
            continue
        # next 3 lines are the verts of this triangle; line was the material name
        tri = []
        for _ in range(3):
            if i >= n:
                break
            parts = lines[i].split()
            i += 1
            if len(parts) < 4:
                continue
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
            key = (round(x, 3), round(y, 3), round(z, 3))
            vi = index.get(key)
            if vi is None:
                vi = index[key] = len(verts)
                verts.append((x, y, z))
            tri.append(vi)
        if len(tri) == 3:
            tris.append(tuple(tri))
    return verts, tris


def read_smd_grouped(path):
    """Parse an SMD into per-material submeshes for textured display. Returns
    {material: {"points": [(x,y,z)], "tris": [(i,j,k)], "uvs": [(u,v)]}} where
    vertices are deduped per (position, uv) so texture seams stay sharp."""
    groups = {}
    index = {}                                   # material -> {(pos,uv): vi}
    with open(path, "r", encoding="ascii", errors="replace") as f:
        lines = f.read().splitlines()
    i = 0
    n = len(lines)
    while i < n and lines[i].strip() != "triangles":
        i += 1
    i += 1
    while i < n:
        mat = lines[i].strip()
        i += 1
        if mat == "end" or not mat:
            if mat == "end":
                break
            continue
        g = groups.setdefault(mat, {"points": [], "tris": [], "uvs": []})
        idx = index.setdefault(mat, {})
        tri = []
        for _ in range(3):
            if i >= n:
                break
            p = lines[i].split()
            i += 1
            if len(p) < 9:
                continue
            pos = (float(p[1]), float(p[2]), float(p[3]))
            uv = (float(p[7]), float(p[8]))
            key = (round(pos[0], 3), round(pos[1], 3), round(pos[2], 3),
                   round(uv[0], 4), round(uv[1], 4))
            vi = idx.get(key)
            if vi is None:
                vi = idx[key] = len(g["points"])
                g["points"].append(pos)
                g["uvs"].append(uv)
            tri.append(vi)
        if len(tri) == 3:
            g["tris"].append(tuple(tri))
    return groups


def mesh_bounds(verts):
    if not verts:
        return (0, 0, 0, 1, 1, 1)
    xs = [v[0] for v in verts]; ys = [v[1] for v in verts]; zs = [v[2] for v in verts]
    return (min(xs), min(ys), min(zs), max(xs), max(ys), max(zs))


class HullEditor(tk.Toplevel):
    def __init__(self, parent, title, verts, tris, hulls, on_save):
        super().__init__(parent)
        self.title("Hulls — " + title)
        self.geometry("960x640")
        self.verts = verts
        self.tris = tris
        self.on_save = on_save
        self.hulls = [list(map(float, b)) for b in (hulls or []) if len(b) == 6]
        self.bb = mesh_bounds(verts)
        cx = (self.bb[0] + self.bb[3]) / 2.0
        cy = (self.bb[1] + self.bb[4]) / 2.0
        cz = (self.bb[2] + self.bb[5]) / 2.0
        self.center = (cx, cy, cz)
        diag = math.dist(self.bb[:3], self.bb[3:]) or 1.0
        self.radius = diag / 2.0
        self.yaw = math.radians(35)
        self.pitch = math.radians(-20)
        self.user_zoom = 1.0
        self.pan = [0.0, 0.0]
        self.sel = None            # selected hull index
        self._drag = None

        # stride very dense meshes for display only
        step = max(1, len(self.tris) // _MAX_DISPLAY_TRIS)
        self._disp_tris = self.tris[::step]

        self._build_ui()
        self._redraw()

    # ---- layout ----
    def _build_ui(self):
        left = ttk.Frame(self)
        left.pack(side="left", fill="both", expand=True)
        self.canvas = tk.Canvas(left, bg="#101418", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<ButtonPress-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", lambda e: setattr(self, "_drag", None))
        self.canvas.bind("<ButtonPress-3>", self._on_press_pan)
        self.canvas.bind("<B3-Motion>", self._on_pan)
        self.canvas.bind("<MouseWheel>", self._on_wheel)
        self.canvas.bind("<Configure>", lambda e: self._redraw())

        right = ttk.Frame(self, width=300)
        right.pack(side="right", fill="y")
        right.pack_propagate(False)
        ttk.Label(right, text="Collision hull boxes", font=("", 10, "bold")).pack(
            anchor="w", padx=6, pady=(6, 2))
        self.box_list = tk.Listbox(right, height=10, font=("Consolas", 9))
        self.box_list.pack(fill="x", padx=6)
        self.box_list.bind("<<ListboxSelect>>", self._on_select)
        bb = ttk.Frame(right)
        bb.pack(fill="x", padx=6, pady=4)
        ttk.Button(bb, text="Add box", command=self._add_box).pack(side="left")
        ttk.Button(bb, text="Duplicate", command=self._dup_box).pack(side="left", padx=2)
        ttk.Button(bb, text="Remove", command=self._rm_box).pack(side="left")

        ed = ttk.LabelFrame(right, text="Selected box (min / max, model units)")
        ed.pack(fill="x", padx=6, pady=6)
        self.fields = []
        labels = ["X0", "Y0", "Z0", "X1", "Y1", "Z1"]
        self.fvars = [tk.StringVar() for _ in range(6)]
        for i, lab in enumerate(labels):
            r, c = divmod(i, 3)
            cell = ttk.Frame(ed)
            cell.grid(row=r, column=c, padx=3, pady=3)
            ttk.Label(cell, text=lab, width=3).pack(side="left")
            e = ttk.Entry(cell, textvariable=self.fvars[i], width=8)
            e.pack(side="left")
            self.fvars[i].trace_add("write", lambda *a: self._apply_fields())
        gb = ttk.Frame(right)
        gb.pack(fill="x", padx=6)
        ttk.Button(gb, text="Fit box to model", command=self._fit_box).pack(side="left")

        info = ttk.Label(right, justify="left", wraplength=280, foreground="#666",
                         text="Drag-L: rotate · Drag-R: pan · Wheel: zoom.\n"
                              "Boxes are convex collision hulls; stack several to "
                              "approximate a concave shape (e.g. an L room).")
        info.pack(anchor="w", padx=6, pady=6)

        bar = ttk.Frame(right)
        bar.pack(side="bottom", fill="x", padx=6, pady=8)
        ttk.Button(bar, text="Save hulls", command=self._save).pack(side="right")
        ttk.Button(bar, text="Cancel", command=self.destroy).pack(side="right", padx=4)
        self._refresh_list()

    # ---- 3D projection ----
    def _project(self, p):
        x = p[0] - self.center[0]
        y = p[1] - self.center[1]
        z = p[2] - self.center[2]
        cy, sy = math.cos(self.yaw), math.sin(self.yaw)
        x, y = x * cy - y * sy, x * sy + y * cy          # yaw about Z
        cp, sp = math.cos(self.pitch), math.sin(self.pitch)
        y, z = y * cp - z * sp, y * sp + z * cp          # pitch about X
        w = self.canvas.winfo_width() or 660
        h = self.canvas.winfo_height() or 640
        zoom = (min(w, h) * 0.42 / (self.radius or 1.0)) * self.user_zoom
        sx = w / 2 + self.pan[0] + x * zoom
        sy_ = h / 2 + self.pan[1] - z * zoom             # screen y up = +Z
        return sx, sy_

    def _redraw(self):
        c = self.canvas
        c.delete("all")
        # mesh wireframe
        for a, b, d in self._disp_tris:
            pa, pb, pd = (self._project(self.verts[a]), self._project(self.verts[b]),
                          self._project(self.verts[d]))
            c.create_line(pa[0], pa[1], pb[0], pb[1], fill="#3a4654")
            c.create_line(pb[0], pb[1], pd[0], pd[1], fill="#3a4654")
            c.create_line(pd[0], pd[1], pa[0], pa[1], fill="#3a4654")
        # hull boxes
        for idx, bx in enumerate(self.hulls):
            self._draw_box(bx, "#ff5d5d" if idx == self.sel else "#46c0ff")
        c.create_text(8, 8, anchor="nw", fill="#7a8694",
                      text="%d tris · %d hull(s)" % (len(self.tris), len(self.hulls)))

    def _box_corners(self, bx):
        x0, y0, z0, x1, y1, z1 = bx
        return [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
                (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]

    def _draw_box(self, bx, color):
        pts = [self._project(p) for p in self._box_corners(bx)]
        for a, b in _BOX_EDGES:
            self.canvas.create_line(pts[a][0], pts[a][1], pts[b][0], pts[b][1],
                                    fill=color, width=2)

    # ---- mouse ----
    def _on_press(self, e):
        self._drag = (e.x, e.y, "rot")

    def _on_drag(self, e):
        if not self._drag:
            return
        dx, dy = e.x - self._drag[0], e.y - self._drag[1]
        self.yaw += dx * 0.01
        self.pitch += dy * 0.01
        self.pitch = max(-1.55, min(1.55, self.pitch))
        self._drag = (e.x, e.y, "rot")
        self._redraw()

    def _on_press_pan(self, e):
        self._drag = (e.x, e.y, "pan")

    def _on_pan(self, e):
        if not self._drag:
            return
        self.pan[0] += e.x - self._drag[0]
        self.pan[1] += e.y - self._drag[1]
        self._drag = (e.x, e.y, "pan")
        self._redraw()

    def _on_wheel(self, e):
        self.user_zoom *= 1.1 if e.delta > 0 else 0.9
        self.user_zoom = max(0.1, min(20.0, self.user_zoom))
        self._redraw()

    # ---- box list / editing ----
    def _refresh_list(self):
        self.box_list.delete(0, "end")
        for i, b in enumerate(self.hulls):
            self.box_list.insert("end", "box %d  [%.0f %.0f %.0f]-[%.0f %.0f %.0f]"
                                 % (i, b[0], b[1], b[2], b[3], b[4], b[5]))
        if self.sel is not None and self.sel < len(self.hulls):
            self.box_list.selection_set(self.sel)
            self._load_fields()

    def _on_select(self, _e):
        s = self.box_list.curselection()
        self.sel = s[0] if s else None
        self._load_fields()
        self._redraw()

    def _load_fields(self):
        if self.sel is None or self.sel >= len(self.hulls):
            return
        self._loading = True
        for i, v in enumerate(self.hulls[self.sel]):
            self.fvars[i].set("%.1f" % v)
        self._loading = False

    def _apply_fields(self):
        if getattr(self, "_loading", False) or self.sel is None:
            return
        try:
            vals = [float(v.get()) for v in self.fvars]
        except ValueError:
            return
        # normalise so min<=max per axis
        for ax in range(3):
            lo, hi = sorted((vals[ax], vals[ax + 3]))
            vals[ax], vals[ax + 3] = lo, hi
        self.hulls[self.sel] = vals
        self._redraw()

    def _default_box(self):
        x0, y0, z0, x1, y1, z1 = self.bb
        mx = (x0 + x1) / 2; my = (y0 + y1) / 2
        return [x0, y0, z0, x1, y1, (z0 + z1) / 2] if (x1 - x0) else [mx, my, z0, mx + 32, my + 32, z1]

    def _add_box(self):
        self.hulls.append(self._default_box())
        self.sel = len(self.hulls) - 1
        self._refresh_list()
        self._redraw()

    def _dup_box(self):
        if self.sel is None:
            return
        self.hulls.append(list(self.hulls[self.sel]))
        self.sel = len(self.hulls) - 1
        self._refresh_list()
        self._redraw()

    def _rm_box(self):
        if self.sel is None:
            return
        del self.hulls[self.sel]
        self.sel = None
        self._refresh_list()
        self._redraw()

    def _fit_box(self):
        if self.sel is None:
            return
        self.hulls[self.sel] = list(self.bb)
        self._load_fields()
        self._refresh_list()
        self._redraw()

    def _save(self):
        self.on_save([list(b) for b in self.hulls])
        self.destroy()


def open_hull_editor(parent, title, smd_path, hulls, on_save):
    """Open the 3D hull editor for ``smd_path``. Raises FileNotFoundError if the
    SMD (compiled model source) is missing."""
    if not os.path.isfile(smd_path):
        raise FileNotFoundError(smd_path)
    verts, tris = read_smd_mesh(smd_path)
    HullEditor(parent, title, verts, tris, hulls, on_save)
