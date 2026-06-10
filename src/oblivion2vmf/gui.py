"""Tkinter GUI for oblivion2vmf.

Pick conversion options, map Oblivion tree species to existing GMod/Source tree
models, remember your last choices, and run the converter with streamed output.
Launch with:  python -m oblivion2vmf.gui   (or the oblivion2vmf-gui command)
"""
from __future__ import annotations

import json
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext

CONFIG_PATH = os.path.join(os.path.expanduser("~"), ".oblivion2vmf_gui.json")
_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # the src/ dir


def _load():
    try:
        with open(CONFIG_PATH, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save(cfg):
    try:
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
    except Exception:
        pass


class App:
    def __init__(self, root):
        self.root = root
        root.title("oblivion2vmf — Oblivion → Garry's Mod")
        self.cfg = _load()
        self.proc = None
        self.q = queue.Queue()
        self.v = {}                  # option name -> tk Var
        self.species_vars = {}       # species -> StringVar (model path)
        self.bsa_list = list(self.cfg.get("bsa", []))
        self.plugin_list = list(self.cfg.get("plugins", []))   # load order (--plugin)
        self.interiors = []          # [(edid, name, refs)] from last scan
        self.model_rows = {}         # nif path -> {collision,ramp,scale,surf,skip} vars
        self._build()
        self._poll()
        root.protocol("WM_DELETE_WINDOW", self._close)

    # ---- var/widget helpers ----
    def _sv(self, name, default=""):
        self.v[name] = tk.StringVar(value=str(self.cfg.get(name, default)))
        return self.v[name]

    def _bv(self, name, default=False):
        self.v[name] = tk.BooleanVar(value=bool(self.cfg.get(name, default)))
        return self.v[name]

    def _row(self, parent, r, label, name, default="", browse=None, width=46):
        ttk.Label(parent, text=label).grid(row=r, column=0, sticky="w", padx=4, pady=2)
        ttk.Entry(parent, textvariable=self._sv(name, default), width=width).grid(
            row=r, column=1, sticky="we", padx=4, pady=2)
        if browse:
            ttk.Button(parent, text="…", width=3,
                       command=lambda: self._browse(name, browse)).grid(row=r, column=2, padx=2)

    def _combo(self, parent, r, label, name, values, default):
        ttk.Label(parent, text=label).grid(row=r, column=0, sticky="w", padx=4, pady=2)
        self.v[name] = tk.StringVar(value=str(self.cfg.get(name, default)))
        ttk.Combobox(parent, textvariable=self.v[name], values=values, width=8,
                     state="readonly").grid(row=r, column=1, sticky="w", padx=4)

    def _browse(self, name, kind):
        p = (filedialog.askopenfilename() if kind == "open"
             else filedialog.asksaveasfilename(defaultextension=".vmf") if kind == "save"
             else filedialog.askdirectory())
        if p:
            self.v[name].set(p)

    # ---- layout ----
    def _build(self):
        nb = ttk.Notebook(self.root)
        nb.pack(fill="both", expand=True)
        self._tab_main(nb)
        self._tab_models(nb)
        self._tab_trees(nb)
        self._tab_interiors(nb)
        self._tab_model_edits(nb)

        bar = ttk.Frame(self.root)
        bar.pack(fill="x")
        self.run_btn = ttk.Button(bar, text="Run", command=self._run)
        self.run_btn.pack(side="left", padx=4, pady=4)
        ttk.Button(bar, text="Stop", command=self._stop).pack(side="left")
        self.v["copy_materials"] = tk.BooleanVar(value=bool(self.cfg.get("copy_materials", True)))
        ttk.Checkbutton(bar, text="Copy materials to GMod after run",
                        variable=self.v["copy_materials"]).pack(side="left", padx=8)

        self.log = scrolledtext.ScrolledText(self.root, height=14, bg="#111", fg="#ddd")
        self.log.pack(fill="both", expand=True, padx=4, pady=4)

    def _tab_main(self, nb):
        f = ttk.Frame(nb)
        nb.add(f, text="Input / Terrain")
        f.columnconfigure(1, weight=1)
        self._row(f, 0, "Oblivion.esm", "esm", browse="open")
        self._row(f, 1, "Output .vmf", "out", "build/test.vmf", browse="save")
        self._row(f, 2, "Cells (minX,minY,maxX,maxY)", "cells", "-19,-6,-9,2")
        self._row(f, 3, "Scale (Hammer units / Oblivion unit)", "scale", "0.5625")
        self._combo(f, 4, "Displacement power", "power", ["2", "3", "4"], "3")
        self._row(f, 5, "Terrain texture scale", "terrain_tex_scale", "0.25")
        cf = ttk.Frame(f)
        cf.grid(row=6, column=0, columnspan=3, sticky="w", pady=4)
        ttk.Checkbutton(cf, text="Land textures", variable=self._bv("textures", True)).pack(side="left", padx=4)
        ttk.Checkbutton(cf, text="4-way blend", variable=self._bv("fourway", False)).pack(side="left", padx=4)
        ttk.Checkbutton(cf, text="Water volumes", variable=self._bv("water", True)).pack(side="left", padx=4)
        ttk.Checkbutton(cf, text="Prop fade", variable=self._bv("prop_fade", True)).pack(side="left", padx=4)
        ttk.Checkbutton(cf, text="Lighting", variable=self._bv("lighting", True)).pack(side="left", padx=4)
        ttk.Checkbutton(cf, text="Fog", variable=self._bv("fog", True)).pack(side="left", padx=4)
        of = ttk.Frame(f)
        of.grid(row=12, column=0, columnspan=3, sticky="w", pady=2)
        ttk.Label(of, text="Terrain lightmapscale").pack(side="left", padx=4)
        ttk.Entry(of, textvariable=self._sv("terrain_lightmapscale", "32"), width=6).pack(side="left")
        ttk.Label(of, text="  Outer power (0=off)").pack(side="left", padx=4)
        ttk.Entry(of, textvariable=self._sv("outer_power", "0"), width=6).pack(side="left")
        vf = ttk.Frame(f)
        vf.grid(row=13, column=0, columnspan=3, sticky="w", pady=2)
        ttk.Checkbutton(vf, text="Vis floor (solid terrain)", variable=self._bv("vis_floor", True)).pack(side="left", padx=4)
        ttk.Checkbutton(vf, text="Seal skybox (vvis)", variable=self._bv("seal_sky", True)).pack(side="left", padx=4)

        bf = ttk.LabelFrame(f, text="BSA archives (add the Meshes + Textures BSAs)")
        bf.grid(row=7, column=0, columnspan=3, sticky="we", padx=4, pady=6)
        self.bsa_box = tk.Listbox(bf, height=3)
        self.bsa_box.pack(side="left", fill="x", expand=True, padx=4, pady=4)
        for b in self.bsa_list:
            self.bsa_box.insert("end", b)
        bb = ttk.Frame(bf)
        bb.pack(side="left", padx=4)
        ttk.Button(bb, text="Add", command=self._add_bsa).pack(fill="x")
        ttk.Button(bb, text="Remove", command=self._rm_bsa).pack(fill="x")

        pf = ttk.LabelFrame(f, text="Plugins / load order (mods on top of the ESM, in order — "
                                    "e.g. Open Cities Resources.esm then Open Cities Classic.esp)")
        pf.grid(row=8, column=0, columnspan=3, sticky="we", padx=4, pady=6)
        self.plugin_box = tk.Listbox(pf, height=3)
        self.plugin_box.pack(side="left", fill="x", expand=True, padx=4, pady=4)
        for p in self.plugin_list:
            self.plugin_box.insert("end", p)
        pb = ttk.Frame(pf)
        pb.pack(side="left", padx=4)
        ttk.Button(pb, text="Add", command=self._add_plugin).pack(fill="x")
        ttk.Button(pb, text="Remove", command=self._rm_plugin).pack(fill="x")
        ttk.Label(f, text="Loose Data dir (mod meshes, e.g. Open Cities '00 Core')").grid(
            row=9, column=0, sticky="w", padx=4, pady=2)
        ttk.Entry(f, textvariable=self._sv("data_dir", ""), width=46).grid(
            row=9, column=1, sticky="we", padx=4)
        ttk.Button(f, text="…", width=3,
                   command=lambda: self._browse("data_dir", "dir")).grid(row=9, column=2, padx=2)

    def _tab_models(self, nb):
        f = ttk.Frame(nb)
        nb.add(f, text="Models")
        f.columnconfigure(1, weight=1)
        ttk.Checkbutton(f, text="Convert models (props)", variable=self._bv("models", True)).grid(
            row=0, column=0, columnspan=2, sticky="w", padx=4, pady=2)
        ttk.Checkbutton(f, text="Skip compile (write .smd/.qc only)",
                        variable=self._bv("skip_compile", False)).grid(
            row=1, column=0, columnspan=2, sticky="w", padx=4)
        ttk.Checkbutton(f, text="Reuse model cache (skip unchanged models)",
                        variable=self._bv("cache", True)).grid(
            row=1, column=2, sticky="w", padx=4)
        self._combo(f, 2, "Collision", "collision",
                    ["auto", "acd", "full", "bbox", "ramp", "none"], "acd")
        self._combo(f, 12, "Ramp axis (rise)", "ramp_axis", ["+x", "-x", "+y", "-y"], "+x")
        self._row(f, 3, "Collision size (HU)", "collision_size", "400")
        self._row(f, 4, "ACD threshold", "acd_threshold", "0.08")
        self._row(f, 11, "ACD max hulls (-1=none)", "acd_max_hulls", "-1")
        self._row(f, 5, "Parallel jobs (0=auto)", "jobs", "0")
        self._combo(f, 6, "Rotation sign", "angle_sign", ["neg", "pos"], "neg")
        self._row(f, 7, "Yaw offset (deg)", "yaw_offset", "-90")
        self._row(f, 8, "studiomdl.exe (optional)", "studiomdl", browse="open")
        self._row(f, 9, "GMod garrysmod dir (optional)", "gamedir", browse="dir")
        # (Loose Data dir + Plugins/load-order live on the Input/Terrain tab)

    def _tab_trees(self, nb):
        f = ttk.Frame(nb)
        nb.add(f, text="Trees")
        f.columnconfigure(1, weight=1)
        self._row(f, 0, "Default tree model", "tree_default", "models/props_foliage/tree_pine04.mdl")
        self._row(f, 1, "Tree scale", "tree_scale", "1.0")
        ttk.Button(f, text="Scan species", command=self._scan).grid(row=2, column=0, sticky="w", padx=4, pady=4)
        ttk.Label(f, text="model per species (blank = use default)").grid(row=2, column=1, sticky="w")
        wrap = ttk.Frame(f)
        wrap.grid(row=3, column=0, columnspan=3, sticky="nswe", padx=4, pady=2)
        f.rowconfigure(3, weight=1)
        canvas = tk.Canvas(wrap, height=240, highlightthickness=0)
        sb = ttk.Scrollbar(wrap, orient="vertical", command=canvas.yview)
        self.species_frame = ttk.Frame(canvas)
        self.species_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=self.species_frame, anchor="nw")
        canvas.configure(yscrollcommand=sb.set)
        canvas.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        self._populate(self.cfg.get("species", []), self.cfg.get("tree_map", {}))

    def _tab_interiors(self, nb):
        f = ttk.Frame(nb)
        nb.add(f, text="Interiors")
        f.columnconfigure(0, weight=1)
        top = ttk.Frame(f)
        top.grid(row=0, column=0, sticky="we", padx=4, pady=4)
        ttk.Label(top, text="Filter").pack(side="left")
        self.v["int_filter"] = tk.StringVar(value=str(self.cfg.get("int_filter", "")))
        ttk.Entry(top, textvariable=self.v["int_filter"], width=22).pack(side="left", padx=4)
        ttk.Button(top, text="List interiors", command=self._scan_interiors).pack(side="left", padx=4)
        ttk.Button(top, text="Build selected room",
                   command=self._build_interior).pack(side="left", padx=4)

        opts = ttk.Frame(f)
        opts.grid(row=1, column=0, sticky="we", padx=4, pady=2)
        self.v["skybox_room"] = tk.BooleanVar(value=bool(self.cfg.get("skybox_room", False)))
        ttk.Checkbutton(opts, text="Skybox room (toolsskybox wrap, not sealed black)",
                        variable=self.v["skybox_room"]).pack(side="left")
        opts2 = ttk.Frame(f)
        opts2.grid(row=2, column=0, sticky="we", padx=4, pady=2)
        self.v["instance_into"] = tk.BooleanVar(value=bool(self.cfg.get("instance_into", False)))
        ttk.Checkbutton(opts2, text="Add as func_instance into host map:",
                        variable=self.v["instance_into"]).pack(side="left")
        self.v["instance_host"] = tk.StringVar(value=str(self.cfg.get("instance_host", "")))
        ttk.Entry(opts2, textvariable=self.v["instance_host"], width=40).pack(side="left", padx=4)
        ttk.Button(opts2, text="…", width=3, command=lambda: self._pick_host()).pack(side="left")

        wrap = ttk.Frame(f)
        wrap.grid(row=3, column=0, sticky="nswe", padx=4, pady=4)
        f.rowconfigure(3, weight=1)
        self.int_box = tk.Listbox(wrap, font=("Consolas", 9), activestyle="dotbox")
        sb = ttk.Scrollbar(wrap, orient="vertical", command=self.int_box.yview)
        self.int_box.configure(yscrollcommand=sb.set)
        self.int_box.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        self.int_box.bind("<Double-Button-1>", lambda e: self._build_interior())
        ttk.Label(f, text="Lists interior cells across your load order. Build uses the Models-tab "
                          "settings; output is <EDID>.vmf next to your main Output .vmf. "
                          "func_instance places the room beside the host's geometry and you "
                          "compile the HOST map.").grid(
            row=4, column=0, sticky="w", padx=4, pady=2)

    def _pick_host(self):
        p = filedialog.askopenfilename(filetypes=[("VMF", "*.vmf"), ("All", "*.*")])
        if p:
            self.v["instance_host"].set(p)

    def _scan_interiors(self):
        filt = self.v["int_filter"].get().strip()
        args = self._common_args() + ["--list-interiors"]
        if filt:
            args.append(filt)
        self._append("Listing interiors%s…\n" % (" matching '%s'" % filt if filt else ""))

        def work():
            try:
                r = subprocess.run(args, capture_output=True, text=True, env=self._env())
                self.q.put(("interiors", r.stdout))
                if r.returncode != 0:
                    self.q.put(("log", (r.stderr or "")[-1000:]))
            except Exception as e:
                self.q.put(("log", "list failed: %r\n" % e))
        threading.Thread(target=work, daemon=True).start()

    def _set_interiors(self, stdout):
        self.interiors = []
        self.int_box.delete(0, "end")
        for line in stdout.splitlines():
            m = re.match(r"\s+(\S+)\s+(.*?)\s+(\d+) refs\s+\(0x", line)
            if m:
                edid, name, refs = m.group(1), m.group(2).strip(), int(m.group(3))
                self.interiors.append((edid, name, refs))
                self.int_box.insert("end", "%-40s %5d refs   %s" % (edid, refs, name))
        self._append("Found %d interiors.\n" % len(self.interiors))

    def _build_interior(self):
        sel = self.int_box.curselection()
        if not sel:
            self._append("Pick an interior from the list first.\n")
            return
        edid = self.interiors[sel[0]][0]
        out_dir = os.path.dirname(os.path.abspath(self.v["out"].get())) or "."
        os.makedirs(out_dir, exist_ok=True)
        out = os.path.join(out_dir, edid + ".vmf")
        a = self._common_args() + ["--interior", edid, "--out", out,
                                    "--scale", self.v["scale"].get()]
        if self.v["skybox_room"].get():
            a.append("--skybox-room")
        if self.v["instance_into"].get():
            host = self.v["instance_host"].get().strip()
            if not host:
                self._append("Tick 'func_instance' but pick a host .vmf first "
                             "(or untick it).\n")
                return
            a += ["--instance-into", host]
        if self.v["models"].get():
            a.append("--models")
            a += self._model_args()
        self._collect()
        _save(self.cfg)
        self._append("\n$ " + " ".join(a) + "\n")
        self.run_btn.config(state="disabled")
        threading.Thread(target=self._exec, args=(a,), daemon=True).start()

    # ---- model edits (per-model overrides) ----
    _COLL_CHOICES = ["(global)", "auto", "acd", "full", "bbox", "ramp", "hulls", "none"]
    _AXIS_CHOICES = ["+x", "-x", "+y", "-y"]

    def _work_dir(self):
        """The models_src dir (compiled-model SMD sources + build cache), beside
        the Output .vmf — same path the CLI uses."""
        out_dir = os.path.dirname(os.path.abspath(self.v["out"].get())) or "."
        return os.path.join(out_dir, "models_src")

    def _tab_model_edits(self, nb):
        f = ttk.Frame(nb)
        nb.add(f, text="Model edits")
        f.columnconfigure(0, weight=1)

        top = ttk.Frame(f)
        top.grid(row=0, column=0, sticky="we", padx=4, pady=4)
        ttk.Label(top, text="Source").pack(side="left")
        self.v["model_src"] = tk.StringVar(value=str(self.cfg.get("model_src", "Exterior")))
        ttk.Combobox(top, textvariable=self.v["model_src"], width=10, state="readonly",
                     values=["Exterior", "Interior"]).pack(side="left", padx=4)
        ttk.Label(top, text="Filter").pack(side="left")
        self.v["model_filter"] = tk.StringVar(value=str(self.cfg.get("model_filter", "")))
        ttk.Entry(top, textvariable=self.v["model_filter"], width=18).pack(side="left", padx=4)
        ttk.Button(top, text="Scan models", command=self._scan_models).pack(side="left", padx=4)
        ttk.Button(top, text="Load compiled", command=self._load_compiled_models).pack(side="left", padx=4)
        ttk.Button(top, text="Save overrides", command=self._save_overrides).pack(side="left", padx=4)

        of = ttk.Frame(f)
        of.grid(row=1, column=0, sticky="we", padx=4, pady=2)
        ttk.Label(of, text="Overrides JSON").pack(side="left")
        self._sv("model_overrides", "")
        ttk.Entry(of, textvariable=self.v["model_overrides"], width=48).pack(side="left", padx=4)
        ttk.Button(of, text="…", width=3,
                   command=lambda: self._browse("model_overrides", "open")).pack(side="left")

        # column headers
        hdr = ttk.Frame(f)
        hdr.grid(row=2, column=0, sticky="we", padx=4)
        for c, (txt, w) in enumerate([("Model (.nif)", 44), ("n", 4), ("Collision", 10),
                                      ("Ramp", 5), ("Scale", 6), ("Surfaceprop", 12),
                                      ("Skip", 4), ("3D", 6)]):
            ttk.Label(hdr, text=txt, width=w, anchor="w").grid(row=0, column=c, padx=2)

        # scrollable rows area
        wrap = ttk.Frame(f)
        wrap.grid(row=3, column=0, sticky="nswe", padx=4, pady=4)
        f.rowconfigure(3, weight=1)
        self.me_canvas = tk.Canvas(wrap, highlightthickness=0)
        sb = ttk.Scrollbar(wrap, orient="vertical", command=self.me_canvas.yview)
        self.me_canvas.configure(yscrollcommand=sb.set)
        self.me_canvas.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        self.me_rows = ttk.Frame(self.me_canvas)
        self.me_win = self.me_canvas.create_window((0, 0), window=self.me_rows, anchor="nw")
        self.me_rows.bind("<Configure>", lambda e: self.me_canvas.configure(
            scrollregion=self.me_canvas.bbox("all")))
        self.me_canvas.bind("<Configure>", lambda e: self.me_canvas.itemconfig(
            self.me_win, width=e.width))

        ttk.Label(f, text="Scan lists every mesh placed in your selection (Exterior uses the "
                          "Main-tab Cells; Interior uses the room selected on the Interiors "
                          "tab). Set per-model collision/ramp/scale/surfaceprop or Skip, then "
                          "Save overrides. Builds auto-use this JSON.").grid(
            row=4, column=0, sticky="w", padx=4, pady=2)

    def _scan_models(self):
        a = self._common_args() + ["--list-models"]
        if self.v["model_src"].get() == "Interior":
            sel = self.int_box.curselection() if hasattr(self, "int_box") else ()
            if not sel:
                self._append("Interior source: select a room on the Interiors tab first.\n")
                return
            a += ["--interior", self.interiors[sel[0]][0]]
        else:
            a.append("--cells=" + self.v["cells"].get().replace(" ", ""))
        filt = self.v["model_filter"].get().strip()
        if filt:
            a.append(filt)
        self._append("Scanning models…\n")

        def work():
            try:
                r = subprocess.run(a, capture_output=True, text=True, env=self._env())
                self.q.put(("models", r.stdout))
                if r.returncode != 0:
                    self.q.put(("log", (r.stderr or "")[-1200:]))
            except Exception as e:
                self.q.put(("log", "model scan failed: %r\n" % e))
        threading.Thread(target=work, daemon=True).start()

    def _set_models(self, stdout):
        models = []
        for line in stdout.splitlines():
            m = re.match(r"\s+(\S+)\s+(\d+)\s+(mesh|tree)\s*$", line)
            if m:
                models.append((m.group(1), int(m.group(2))))
        self._populate_models(models, "scan")

    def _load_compiled_models(self):
        """Populate rows from every model currently compiled (the build cache in
        models_src) — no ESM scan needed."""
        cache = os.path.join(self._work_dir(), ".build_cache.json")
        if not os.path.isfile(cache):
            self._append("No build cache at %s — compile a build first "
                         "(or use Scan models).\n" % cache)
            return
        try:
            with open(cache, encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception as e:
            self._append("could not read build cache: %r\n" % e)
            return
        filt = self.v["model_filter"].get().strip().lower()
        models = [(modl, 0) for modl in sorted(data) if not filt or filt in modl.lower()]
        self._populate_models(models, "compiled")

    def _populate_models(self, models, source):
        for w in self.me_rows.winfo_children():
            w.destroy()
        self.model_rows = {}
        saved = self._read_overrides_file()
        for i, (modl, n) in enumerate(models):
            ov = saved.get(modl.lower(), {})
            tail = modl if len(modl) <= 44 else "…" + modl[-43:]
            ttk.Label(self.me_rows, text=tail, width=44, anchor="w").grid(
                row=i, column=0, padx=2, sticky="w")
            ttk.Label(self.me_rows, text=(str(n) if n else "·"), width=4,
                      anchor="e").grid(row=i, column=1, padx=2)
            coll = tk.StringVar(value=ov.get("collision") or "(global)")
            ttk.Combobox(self.me_rows, textvariable=coll, values=self._COLL_CHOICES,
                         width=9, state="readonly").grid(row=i, column=2, padx=2)
            ramp = tk.StringVar(value=ov.get("ramp_axis") or "+x")
            ttk.Combobox(self.me_rows, textvariable=ramp, values=self._AXIS_CHOICES,
                         width=4, state="readonly").grid(row=i, column=3, padx=2)
            scale = tk.StringVar(value=("" if ov.get("scale") in (None, 1.0)
                                        else str(ov.get("scale"))))
            ttk.Entry(self.me_rows, textvariable=scale, width=6).grid(row=i, column=4, padx=2)
            surf = tk.StringVar(value=ov.get("surfaceprop") or "")
            ttk.Entry(self.me_rows, textvariable=surf, width=12).grid(row=i, column=5, padx=2)
            skip = tk.BooleanVar(value=bool(ov.get("skip")))
            ttk.Checkbutton(self.me_rows, variable=skip).grid(row=i, column=6, padx=2)
            row = {"collision": coll, "ramp": ramp, "scale": scale, "surf": surf,
                   "skip": skip, "hulls": list(ov.get("hulls") or [])}
            self.model_rows[modl] = row
            ttk.Button(self.me_rows, text="Hulls…", width=6,
                       command=lambda m=modl: self._edit_hulls(m)).grid(row=i, column=7, padx=2)
        self._append("Loaded %d models (%s; %d with saved overrides).\n"
                     % (len(models), source,
                        sum(1 for k in self.model_rows if k.lower() in saved)))

    def _edit_hulls(self, modl):
        from . import hullview
        from .model import slugify
        smd = os.path.join(self._work_dir(), slugify(modl) + ".smd")
        if not os.path.isfile(smd):
            self._append("No compiled SMD for %s at %s — build it first "
                         "(needs models_src).\n" % (modl, smd))
            return
        row = self.model_rows[modl]

        def on_save(hulls):
            row["hulls"] = hulls
            if hulls:
                row["collision"].set("hulls")
            self._append("%s: %d hull box(es) set (remember to Save overrides).\n"
                         % (modl, len(hulls)))
        try:
            hullview.open_hull_editor(self.root, os.path.basename(modl), smd,
                                      row["hulls"], on_save)
        except Exception as e:
            self._append("could not open 3D editor: %r\n" % e)

    def _read_overrides_file(self):
        path = self.v["model_overrides"].get().strip()
        if path and os.path.isfile(path):
            try:
                with open(path, encoding="utf-8") as fh:
                    return {str(k).lower(): v for k, v in json.load(fh).items()}
            except Exception as e:
                self._append("could not read overrides JSON: %r\n" % e)
        return {}

    def _overrides_from_rows(self):
        """Collapse the editor rows into {nif_lower: settings}, dropping defaults."""
        out = {}
        for modl, r in self.model_rows.items():
            d = {}
            if r["collision"].get() not in ("(global)", ""):
                d["collision"] = r["collision"].get()
                if d["collision"] == "ramp":
                    d["ramp_axis"] = r["ramp"].get()
                if d["collision"] == "hulls" and r.get("hulls"):
                    d["hulls"] = [[round(float(c), 2) for c in b] for b in r["hulls"]]
            sc = r["scale"].get().strip()
            if sc:
                try:
                    if float(sc) != 1.0:
                        d["scale"] = float(sc)
                except ValueError:
                    pass
            if r["surf"].get().strip():
                d["surfaceprop"] = r["surf"].get().strip()
            if r["skip"].get():
                d["skip"] = True
            if d:
                out[modl.lower()] = d
        return out

    def _save_overrides(self):
        path = self.v["model_overrides"].get().strip()
        if not path:
            out_dir = os.path.dirname(os.path.abspath(self.v["out"].get())) or "."
            path = os.path.join(out_dir, "model_overrides.json")
            self.v["model_overrides"].set(path)
        # merge: keep entries for models not currently listed, overwrite listed ones
        merged = self._read_overrides_file()
        listed = {k.lower() for k in self.model_rows}
        merged = {k: v for k, v in merged.items() if k not in listed}
        merged.update(self._overrides_from_rows())
        try:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(merged, fh, indent=2, sort_keys=True)
            self._append("Saved %d model override(s) -> %s\n" % (len(merged), path))
        except Exception as e:
            self._append("save overrides failed: %r\n" % e)

    def _populate(self, species, mapping):
        for w in self.species_frame.winfo_children():
            w.destroy()
        self.species_vars = {}
        for i, sp in enumerate(sorted(species)):
            ttk.Label(self.species_frame, text=sp).grid(row=i, column=0, sticky="w", padx=2, pady=1)
            var = tk.StringVar(value=mapping.get(sp, ""))
            self.species_vars[sp] = var
            ttk.Entry(self.species_frame, textvariable=var, width=42).grid(row=i, column=1, sticky="we", padx=2)

    # ---- bsa ----
    def _add_bsa(self):
        for p in filedialog.askopenfilenames(filetypes=[("BSA", "*.bsa"), ("All", "*.*")]):
            self.bsa_list.append(p)
            self.bsa_box.insert("end", p)

    def _rm_bsa(self):
        for i in reversed(self.bsa_box.curselection()):
            self.bsa_box.delete(i)
            del self.bsa_list[i]

    def _add_plugin(self):
        for p in filedialog.askopenfilenames(
                filetypes=[("Plugin", "*.esp *.esm"), ("All", "*.*")]):
            self.plugin_list.append(p)
            self.plugin_box.insert("end", p)

    def _rm_plugin(self):
        for i in reversed(self.plugin_box.curselection()):
            self.plugin_box.delete(i)
            del self.plugin_list[i]

    # ---- run / scan ----
    def _common_args(self):
        """Args shared by every invocation: esm, load-order plugins, archives, data."""
        a = [sys.executable, "-m", "oblivion2vmf", "--esm", self.v["esm"].get()]
        for p in self.plugin_list:
            a += ["--plugin", p]
        for b in self.bsa_list:
            a += ["--bsa", b]
        dd = self.v["data_dir"].get().strip()
        if dd:
            a += ["--data-dir", dd]
        return a

    def _base_args(self):
        return self._common_args() + [
            "--cells=" + self.v["cells"].get().replace(" ", ""),
            "--out", self.v["out"].get()]

    def _env(self):
        env = os.environ.copy()
        env["PYTHONPATH"] = _SRC_DIR + os.pathsep + env.get("PYTHONPATH", "")
        return env

    def _scan(self):
        args = self._base_args() + ["--models", "--list-tree-species"]
        self._append("Scanning tree species…\n")

        def work():
            try:
                r = subprocess.run(args, capture_output=True, text=True, env=self._env())
                sp = [l.strip() for l in r.stdout.splitlines() if l.strip()]
                self.q.put(("species", sp))
                self.q.put(("log", "Found %d species.\n" % len(sp)))
                if r.returncode != 0:
                    self.q.put(("log", (r.stderr or "")[-1000:]))
            except Exception as e:
                self.q.put(("log", "scan failed: %r\n" % e))
        threading.Thread(target=work, daemon=True).start()

    def _model_args(self):
        """The NIF->MDL pipeline flags shared by terrain and interior builds
        (everything except --models itself and tree mapping)."""
        a = []
        if self.v["skip_compile"].get():
            a.append("--skip-compile")
        if not self.v["cache"].get():
            a.append("--no-cache")
        ov = self.v.get("model_overrides")
        if ov is not None and ov.get().strip():
            a += ["--model-overrides", ov.get().strip()]
        a += ["--collision", self.v["collision"].get(),
              "--ramp-axis", self.v["ramp_axis"].get(),
              "--collision-size", self.v["collision_size"].get(),
              "--acd-threshold", self.v["acd_threshold"].get(),
              "--acd-max-hulls", (self.v["acd_max_hulls"].get().strip() or "-1"),
              "--jobs", (self.v["jobs"].get().strip() or "0"),
              "--angle-sign", self.v["angle_sign"].get(),
              "--yaw-offset", self.v["yaw_offset"].get()]
        for name, flag in (("studiomdl", "--studiomdl"), ("gamedir", "--gamedir")):
            val = self.v[name].get().strip()
            if val:
                a += [flag, val]
        return a

    def _tree_map(self):
        tm = {}
        d = self.v["tree_default"].get().strip()
        if d:
            tm["_default"] = d
        for sp, var in self.species_vars.items():
            val = var.get().strip()
            if val:
                tm[sp] = val
        return tm

    def _run(self):
        a = self._base_args()
        a += ["--scale", self.v["scale"].get(), "--power", self.v["power"].get(),
              "--terrain-tex-scale", self.v["terrain_tex_scale"].get(),
              "--terrain-lightmapscale", (self.v["terrain_lightmapscale"].get().strip() or "32")]
        if not self.v["textures"].get():
            a.append("--no-textures")
        if self.v["fourway"].get():
            a.append("--fourway")
        if not self.v["water"].get():
            a.append("--no-water")
        if not self.v["prop_fade"].get():
            a.append("--no-prop-fade")
        if not self.v["lighting"].get():
            a.append("--no-lighting")
        if not self.v["fog"].get():
            a.append("--no-fog")
        op = self.v["outer_power"].get().strip()
        if op and op not in ("0", "off"):
            a += ["--outer-power", op]
        if self.v["vis_floor"].get():
            a.append("--vis-floor")
        if self.v["seal_sky"].get():
            a.append("--seal-sky")
        if self.v["models"].get():
            a.append("--models")
            a += self._model_args()
            a += ["--tree-scale", self.v["tree_scale"].get()]
            tm = self._tree_map()
            if tm:
                out_dir = os.path.dirname(os.path.abspath(self.v["out"].get())) or "."
                os.makedirs(out_dir, exist_ok=True)
                tmf = os.path.join(out_dir, "treemap.json")
                with open(tmf, "w", encoding="utf-8") as f:
                    json.dump(tm, f, indent=2)
                a += ["--tree-map-file", tmf]

        self._collect()
        _save(self.cfg)
        self._append("\n$ " + " ".join(a) + "\n")
        self.run_btn.config(state="disabled")
        threading.Thread(target=self._exec, args=(a,), daemon=True).start()

    def _exec(self, args):
        try:
            self.proc = subprocess.Popen(args, stdout=subprocess.PIPE,
                                         stderr=subprocess.STDOUT, text=True, env=self._env())
            for line in self.proc.stdout:
                self.q.put(("log", line))
            self.proc.wait()
            rc = self.proc.returncode
            self.q.put(("log", "\n[exit %d]\n" % rc))
            if rc == 0 and self.v["copy_materials"].get():
                self._copy_materials()
        except Exception as e:
            self.q.put(("log", "run failed: %r\n" % e))
        finally:
            self.proc = None
            self.q.put(("done", None))

    def _copy_materials(self):
        src = os.path.join(os.path.dirname(os.path.abspath(self.v["out"].get())), "materials")
        gamedir = self.v["gamedir"].get().strip()
        if not gamedir or not os.path.isdir(src):
            self.q.put(("log", "(material copy skipped: set GMod garrysmod dir)\n"))
            return
        try:
            shutil.copytree(src, os.path.join(gamedir, "materials"), dirs_exist_ok=True)
            self.q.put(("log", "Copied materials -> %s\\materials\n" % gamedir))
        except Exception as e:
            self.q.put(("log", "material copy failed: %r\n" % e))

    def _stop(self):
        if self.proc:
            self.proc.terminate()
            self.q.put(("log", "[stopped]\n"))

    # ---- log pump / config ----
    def _append(self, s):
        self.log.insert("end", s)
        self.log.see("end")

    def _poll(self):
        try:
            while True:
                kind, data = self.q.get_nowait()
                if kind == "log":
                    self._append(data)
                elif kind == "species":
                    self.cfg["species"] = data
                    self._populate(data, self.cfg.get("tree_map", {}))
                elif kind == "interiors":
                    self._set_interiors(data)
                elif kind == "models":
                    self._set_models(data)
                elif kind == "done":
                    self.run_btn.config(state="normal")
        except queue.Empty:
            pass
        self.root.after(120, self._poll)

    def _collect(self):
        for k, var in self.v.items():
            self.cfg[k] = var.get()
        self.cfg["bsa"] = list(self.bsa_list)
        self.cfg["plugins"] = list(self.plugin_list)
        self.cfg["tree_map"] = {sp: v.get().strip() for sp, v in self.species_vars.items() if v.get().strip()}

    def _close(self):
        self._collect()
        _save(self.cfg)
        self.root.destroy()


def main():
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
