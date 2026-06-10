"""PySide6 GUI for oblivion2vmf with an embedded 3D collision-hull editor.

A modern Qt port of the Tkinter ``gui`` module. The headline addition is the
**Model edits** tab: an inline pyvista (VTK) viewport that renders a compiled
model and gives every collision hull a real box widget — drag the face handles to
resize, drag the body to move. Boxes are saved as per-model ``hulls`` overrides
that the build turns into convex ``$collisionmodel`` pieces.

Run with:  python -m oblivion2vmf.qtgui
Falls back gracefully: the Tkinter ``gui`` module still works if Qt/pyvista
aren't installed.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import QProcess, QProcessEnvironment, Qt

from .gui import CONFIG_PATH, _SRC_DIR, _load, _save     # share config with the Tk GUI
from .hullview import read_smd_grouped, read_smd_mesh, mesh_bounds
from .model import slugify, hull_from_spec

_IMPORT_ERR = None
try:
    from pyvistaqt import QtInteractor
    import pyvista as pv
    _HAVE_3D = True
except Exception as _e:                                  # pragma: no cover
    QtInteractor = None
    pv = None
    _HAVE_3D = False
    _IMPORT_ERR = _e


# --------------------------------------------------------------------------- #
#  argument building (pure functions over a values dict)                      #
# --------------------------------------------------------------------------- #
def _common_args(v, plugins, bsas):
    a = [sys.executable, "-m", "oblivion2vmf", "--esm", v["esm"]]
    for p in plugins:
        a += ["--plugin", p]
    for b in bsas:
        a += ["--bsa", b]
    if v["data_dir"].strip():
        a += ["--data-dir", v["data_dir"].strip()]
    return a


def _model_args(v):
    a = []
    if v["skip_compile"]:
        a.append("--skip-compile")
    if not v["cache"]:
        a.append("--no-cache")
    if v["model_overrides"].strip():
        a += ["--model-overrides", v["model_overrides"].strip()]
    a += ["--collision", v["collision"], "--ramp-axis", v["ramp_axis"],
          "--collision-size", v["collision_size"],
          "--acd-threshold", v["acd_threshold"],
          "--acd-max-hulls", (v["acd_max_hulls"].strip() or "-1"),
          "--jobs", (v["jobs"].strip() or "0"),
          "--angle-sign", v["angle_sign"], "--yaw-offset", v["yaw_offset"]]
    for name, flag in (("studiomdl", "--studiomdl"), ("gamedir", "--gamedir")):
        if v[name].strip():
            a += [flag, v[name].strip()]
    return a


def _terrain_args(v, plugins, bsas):
    a = _common_args(v, plugins, bsas)
    a += ["--cells=" + v["cells"].replace(" ", ""), "--out", v["out"],
          "--scale", v["scale"], "--power", v["power"],
          "--terrain-tex-scale", v["terrain_tex_scale"],
          "--terrain-lightmapscale", (v["terrain_lightmapscale"].strip() or "32")]
    if not v["textures"]:
        a.append("--no-textures")
    if v["fourway"]:
        a.append("--fourway")
    if not v["water"]:
        a.append("--no-water")
    if not v["prop_fade"]:
        a.append("--no-prop-fade")
    if not v["lighting"]:
        a.append("--no-lighting")
    if not v["fog"]:
        a.append("--no-fog")
    op = v["outer_power"].strip()
    if op and op not in ("0", "off"):
        a += ["--outer-power", op]
    if v["vis_floor"]:
        a.append("--vis-floor")
    if v["seal_sky"]:
        a.append("--seal-sky")
    return a


# --------------------------------------------------------------------------- #
#  main window                                                                #
# --------------------------------------------------------------------------- #
class Main(QtWidgets.QMainWindow):
    acd_ready = QtCore.Signal(str, object)     # (model, parts|None) from CoACD worker

    def __init__(self):
        super().__init__()
        self.setWindowTitle("oblivion2vmf — Oblivion → Garry's Mod (Qt)")
        self.resize(1180, 820)
        self.cfg = _load()
        self.getters = {}          # name -> callable returning current value
        self.setters = {}          # name -> callable(value)
        self.bsa_list = list(self.cfg.get("bsa", []))
        self.plugin_list = list(self.cfg.get("plugins", []))
        self.species_edits = {}    # species -> QLineEdit
        self.interiors = []        # [(edid, name, refs)]
        self.model_rows = {}       # nif -> {collision,ramp,scale,surf,skip,hulls}
        self.proc = None
        self.cap = None            # capture QProcess
        self.cap_buf = ""
        self.cap_done = None
        self.boxes = []            # [{widget, bounds}] active hull widgets
        self.acd_actors = []       # named actors of the current ACD preview
        self.cur_model = None      # nif currently loaded in the 3D view
        self._undo_stack = []      # pre-mutation hull snapshots (Ctrl+Z)
        self._redo_stack = []      # undone snapshots (Ctrl+Y / Ctrl+Shift+Z)
        self.acd_ready.connect(self._on_acd_ready)

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        outer = QtWidgets.QVBoxLayout(central)
        self.tabs = QtWidgets.QTabWidget()
        outer.addWidget(self.tabs, 1)
        self._tab_main()
        self._tab_models()
        self._tab_trees()
        self._tab_interiors()
        self._tab_model_edits()

        bar = QtWidgets.QHBoxLayout()
        self.run_btn = QtWidgets.QPushButton("Run terrain build")
        self.run_btn.clicked.connect(self._run)
        bar.addWidget(self.run_btn)
        stop = QtWidgets.QPushButton("Stop")
        stop.clicked.connect(self._stop)
        bar.addWidget(stop)
        self.copy_mats = QtWidgets.QCheckBox("Copy materials to GMod after run")
        self.copy_mats.setChecked(bool(self.cfg.get("copy_materials", True)))
        bar.addWidget(self.copy_mats)
        bar.addStretch(1)
        outer.addLayout(bar)

        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(20000)
        self.log.setStyleSheet("background:#111;color:#ddd;font-family:Consolas;")
        outer.addWidget(self.log, 1)

        self._apply_dark()

        # hull-edit undo/redo (scoped to the 3D editor's hull state)
        QtGui.QShortcut(QtGui.QKeySequence.Undo, self, activated=self._undo)
        QtGui.QShortcut(QtGui.QKeySequence.Redo, self, activated=self._redo)
        QtGui.QShortcut(QtGui.QKeySequence("Ctrl+Y"), self, activated=self._redo)

    # ---- field registry helpers ----
    def _edit(self, name, default="", browse=None, width=None):
        e = QtWidgets.QLineEdit(str(self.cfg.get(name, default)))
        if width:
            e.setMaximumWidth(width)
        self.getters[name] = e.text
        self.setters[name] = e.setText
        if browse:
            w = QtWidgets.QWidget()
            h = QtWidgets.QHBoxLayout(w)
            h.setContentsMargins(0, 0, 0, 0)
            h.addWidget(e, 1)
            b = QtWidgets.QPushButton("…")
            b.setMaximumWidth(30)
            b.clicked.connect(lambda: self._browse(name, browse))
            h.addWidget(b)
            return w
        return e

    def _check(self, name, label, default=False):
        c = QtWidgets.QCheckBox(label)
        c.setChecked(bool(self.cfg.get(name, default)))
        self.getters[name] = c.isChecked
        self.setters[name] = c.setChecked
        return c

    def _combo(self, name, values, default):
        c = QtWidgets.QComboBox()
        c.addItems(values)
        cur = str(self.cfg.get(name, default))
        if cur in values:
            c.setCurrentText(cur)
        self.getters[name] = c.currentText
        self.setters[name] = c.setCurrentText
        return c

    def _browse(self, name, kind):
        if kind == "open":
            p, _ = QtWidgets.QFileDialog.getOpenFileName(self)
        elif kind == "save":
            p, _ = QtWidgets.QFileDialog.getSaveFileName(self, dir="build/test.vmf")
        else:
            p = QtWidgets.QFileDialog.getExistingDirectory(self)
        if p:
            self.setters[name](p)

    def _vals(self):
        return {k: g() for k, g in self.getters.items()}

    # ---- tabs ----
    def _tab_main(self):
        f = QtWidgets.QWidget()
        self.tabs.addTab(f, "Input / Terrain")
        g = QtWidgets.QFormLayout(f)
        g.addRow("Oblivion.esm", self._edit("esm", browse="open"))
        g.addRow("Output .vmf", self._edit("out", "build/test.vmf", browse="save"))
        g.addRow("Cells (minX,minY,maxX,maxY)", self._edit("cells", "-19,-6,-9,2"))
        g.addRow("Scale (HU / Oblivion unit)", self._edit("scale", "0.5625"))
        g.addRow("Displacement power", self._combo("power", ["2", "3", "4"], "3"))
        g.addRow("Terrain texture scale", self._edit("terrain_tex_scale", "0.25"))
        g.addRow("Terrain lightmapscale", self._edit("terrain_lightmapscale", "32"))
        g.addRow("Outer power (0=off)", self._edit("outer_power", "0"))

        row = QtWidgets.QHBoxLayout()
        for nm, lab, dv in [("textures", "Land textures", True), ("fourway", "4-way blend", False),
                            ("water", "Water", True), ("prop_fade", "Prop fade", True),
                            ("lighting", "Lighting", True), ("fog", "Fog", True),
                            ("vis_floor", "Vis floor", True), ("seal_sky", "Seal sky", True)]:
            row.addWidget(self._check(nm, lab, dv))
        row.addStretch(1)
        rw = QtWidgets.QWidget()
        rw.setLayout(row)
        g.addRow(rw)

        g.addRow(self._list_group("BSA archives (Meshes + Textures BSAs)", "bsa"))
        g.addRow(self._list_group("Plugins / load order (after the ESM, in order)", "plugin"))
        g.addRow("Loose Data dir", self._edit("data_dir", "", browse="dir"))

    def _list_group(self, title, which):
        box = QtWidgets.QGroupBox(title)
        h = QtWidgets.QHBoxLayout(box)
        lst = QtWidgets.QListWidget()
        lst.setMaximumHeight(72)
        items = self.bsa_list if which == "bsa" else self.plugin_list
        lst.addItems(items)
        h.addWidget(lst, 1)
        col = QtWidgets.QVBoxLayout()
        add = QtWidgets.QPushButton("Add")
        rm = QtWidgets.QPushButton("Remove")
        col.addWidget(add)
        col.addWidget(rm)
        h.addLayout(col)
        if which == "bsa":
            self.bsa_widget = lst
            add.clicked.connect(lambda: self._add_files(lst, items, "BSA (*.bsa)"))
        else:
            self.plugin_widget = lst
            add.clicked.connect(lambda: self._add_files(lst, items, "Plugins (*.esp *.esm)"))
        rm.clicked.connect(lambda: self._rm_selected(lst, items))
        return box

    def _add_files(self, lst, store, filt):
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(self, filter=filt)
        for p in paths:
            store.append(p)
            lst.addItem(p)

    def _rm_selected(self, lst, store):
        for it in lst.selectedItems():
            r = lst.row(it)
            lst.takeItem(r)
            del store[r]

    def _tab_models(self):
        f = QtWidgets.QWidget()
        self.tabs.addTab(f, "Models")
        g = QtWidgets.QFormLayout(f)
        g.addRow(self._check("models", "Convert models (props)", True))
        g.addRow(self._check("skip_compile", "Skip compile (.smd/.qc only)", False))
        g.addRow(self._check("cache", "Reuse model cache", True))
        g.addRow("Collision", self._combo("collision",
                 ["auto", "acd", "full", "bbox", "ramp", "hulls", "none"], "acd"))
        g.addRow("Ramp axis (rise)", self._combo("ramp_axis", ["+x", "-x", "+y", "-y"], "+x"))
        g.addRow("Collision size (HU)", self._edit("collision_size", "400"))
        g.addRow("ACD threshold", self._edit("acd_threshold", "0.08"))
        g.addRow("ACD max hulls (-1=none)", self._edit("acd_max_hulls", "-1"))
        g.addRow("Parallel jobs (0=auto)", self._edit("jobs", "0"))
        g.addRow("Rotation sign", self._combo("angle_sign", ["neg", "pos"], "neg"))
        g.addRow("Yaw offset (deg)", self._edit("yaw_offset", "-90"))
        g.addRow("studiomdl.exe", self._edit("studiomdl", browse="open"))
        g.addRow("GMod garrysmod dir", self._edit("gamedir", browse="dir"))

    def _tab_trees(self):
        f = QtWidgets.QWidget()
        self.tabs.addTab(f, "Trees")
        v = QtWidgets.QVBoxLayout(f)
        form = QtWidgets.QFormLayout()
        form.addRow("Default tree model",
                    self._edit("tree_default", "models/props_foliage/tree_pine04.mdl"))
        form.addRow("Tree scale", self._edit("tree_scale", "1.0"))
        v.addLayout(form)
        scan = QtWidgets.QPushButton("Scan species")
        scan.clicked.connect(self._scan_species)
        v.addWidget(scan)
        self.species_area = QtWidgets.QScrollArea()
        self.species_area.setWidgetResizable(True)
        self.species_host = QtWidgets.QWidget()
        self.species_form = QtWidgets.QFormLayout(self.species_host)
        self.species_area.setWidget(self.species_host)
        v.addWidget(self.species_area, 1)
        self._populate_species(self.cfg.get("species", []), self.cfg.get("tree_map", {}))

    def _tab_interiors(self):
        f = QtWidgets.QWidget()
        self.tabs.addTab(f, "Interiors")
        v = QtWidgets.QVBoxLayout(f)
        top = QtWidgets.QHBoxLayout()
        top.addWidget(QtWidgets.QLabel("Filter"))
        self.int_filter = QtWidgets.QLineEdit(str(self.cfg.get("int_filter", "")))
        top.addWidget(self.int_filter)
        b1 = QtWidgets.QPushButton("List interiors")
        b1.clicked.connect(self._scan_interiors)
        top.addWidget(b1)
        b2 = QtWidgets.QPushButton("Build selected room")
        b2.clicked.connect(self._build_interior)
        top.addWidget(b2)
        v.addLayout(top)
        self.skybox_room = QtWidgets.QCheckBox("Skybox room (toolsskybox wrap, not sealed black)")
        self.skybox_room.setChecked(bool(self.cfg.get("skybox_room", False)))
        v.addWidget(self.skybox_room)
        ih = QtWidgets.QHBoxLayout()
        self.instance_into = QtWidgets.QCheckBox("Add as func_instance into host map:")
        self.instance_into.setChecked(bool(self.cfg.get("instance_into", False)))
        ih.addWidget(self.instance_into)
        self.instance_host = QtWidgets.QLineEdit(str(self.cfg.get("instance_host", "")))
        ih.addWidget(self.instance_host, 1)
        hb = QtWidgets.QPushButton("…")
        hb.clicked.connect(lambda: self._pick_into(self.instance_host))
        ih.addWidget(hb)
        v.addLayout(ih)
        self.int_list = QtWidgets.QListWidget()
        self.int_list.itemDoubleClicked.connect(lambda *_: self._build_interior())
        v.addWidget(self.int_list, 1)

    def _pick_into(self, edit):
        p, _ = QtWidgets.QFileDialog.getOpenFileName(self, filter="VMF (*.vmf)")
        if p:
            edit.setText(p)

    # ---- model edits + 3D ----
    def _tab_model_edits(self):
        f = QtWidgets.QWidget()
        self.tabs.addTab(f, "Model edits / 3D")
        split = QtWidgets.QSplitter(Qt.Horizontal, f)
        QtWidgets.QVBoxLayout(f).addWidget(split)

        # left: controls + table
        left = QtWidgets.QWidget()
        lv = QtWidgets.QVBoxLayout(left)
        top = QtWidgets.QHBoxLayout()
        top.addWidget(QtWidgets.QLabel("Source"))
        self.model_src = QtWidgets.QComboBox()
        self.model_src.addItems(["Exterior", "Interior"])
        self.model_src.setCurrentText(str(self.cfg.get("model_src", "Exterior")))
        top.addWidget(self.model_src)
        self.model_filter = QtWidgets.QLineEdit(str(self.cfg.get("model_filter", "")))
        self.model_filter.setPlaceholderText("filter…")
        top.addWidget(self.model_filter, 1)
        for lab, fn in [("Scan", self._scan_models), ("Load compiled", self._load_compiled),
                        ("Save overrides", self._save_overrides), ("Export…", self._export_data)]:
            b = QtWidgets.QPushButton(lab)
            b.clicked.connect(fn)
            top.addWidget(b)
        lv.addLayout(top)

        oh = QtWidgets.QHBoxLayout()
        oh.addWidget(QtWidgets.QLabel("Overrides JSON"))
        self.ov_path = QtWidgets.QLineEdit(str(self.cfg.get("model_overrides", "")))
        oh.addWidget(self.ov_path, 1)
        ob = QtWidgets.QPushButton("…")
        ob.clicked.connect(lambda: self._pick_into(self.ov_path))
        oh.addWidget(ob)
        lv.addLayout(oh)
        # keep ov_path readable by _model_args via getters
        self.getters["model_overrides"] = self.ov_path.text
        self.setters["model_overrides"] = self.ov_path.setText

        self.table = QtWidgets.QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(
            ["Model (.nif)", "Collision", "Ramp", "Scale", "Surfaceprop", "Skip"])
        self.table.horizontalHeader().setStretchLastSection(False)
        self.table.setColumnWidth(0, 300)
        self.table.itemSelectionChanged.connect(self._on_model_selected)
        lv.addWidget(self.table, 1)
        split.addWidget(left)

        # right: 3D viewport + hull controls
        right = QtWidgets.QWidget()
        rv = QtWidgets.QVBoxLayout(right)
        self.plotter = None
        if _HAVE_3D:
            try:
                self.plotter = QtInteractor(right)
                rv.addWidget(self.plotter.interactor, 1)
                self._style_viewport()
                # terrain interaction: orbit = azimuth + elevation only (the camera
                # can never roll), wheel zooms, shift pans
                self.plotter.enable_terrain_style(mouse_wheel_zooms=True, shift_pans=True)
            except Exception as e:                       # missing/old GL drivers, etc.
                rv.addWidget(QtWidgets.QLabel(
                    "3D viewport failed to initialise (OpenGL?):\n%r\n\n"
                    "The rest of the editor still works; hulls can be set via the "
                    "overrides JSON." % (e,)))
        else:
            rv.addWidget(QtWidgets.QLabel("pyvista not available:\n%r" % (_IMPORT_ERR,)))
        # axis-snap camera views
        vb = QtWidgets.QHBoxLayout()
        vb.addWidget(QtWidgets.QLabel("View"))
        for lab, fn in [("Top", lambda: self._snap_view("xy")),
                        ("Front", lambda: self._snap_view("xz")),
                        ("Side", lambda: self._snap_view("yz")),
                        ("Iso", lambda: self._snap_view("iso"))]:
            b = QtWidgets.QPushButton(lab)
            b.setMaximumWidth(52)
            b.clicked.connect(fn)
            vb.addWidget(b)
        vb.addStretch(1)
        rv.addLayout(vb)

        hb = QtWidgets.QHBoxLayout()
        for lab, fn in [("Add box", lambda: self._add_hull("box")),
                        ("Add wedge", lambda: self._add_hull("wedge")),
                        ("Add trapezium", lambda: self._add_hull("trap")),
                        ("Remove", self._remove_box),
                        ("Fit to model", self._fit_box),
                        ("Save hulls → row", self._commit_hulls)]:
            b = QtWidgets.QPushButton(lab)
            b.clicked.connect(fn)
            hb.addWidget(b)
        rv.addLayout(hb)

        sb = QtWidgets.QHBoxLayout()
        sb.addWidget(QtWidgets.QLabel("Wedge rise axis"))
        self.wedge_axis = QtWidgets.QComboBox()
        self.wedge_axis.addItems(["+x", "-x", "+y", "-y"])
        sb.addWidget(self.wedge_axis)
        sb.addWidget(QtWidgets.QLabel("Trapezium top scale"))
        self.trap_top = QtWidgets.QDoubleSpinBox()
        self.trap_top.setRange(0.05, 1.0)
        self.trap_top.setSingleStep(0.05)
        self.trap_top.setValue(0.5)
        sb.addWidget(self.trap_top)
        sb.addStretch(1)
        rv.addLayout(sb)

        # CoACD convex-decomposition row
        ab = QtWidgets.QHBoxLayout()
        ab.addWidget(QtWidgets.QLabel("ACD threshold"))
        self.acd_thresh = QtWidgets.QDoubleSpinBox()
        self.acd_thresh.setRange(0.01, 1.0)
        self.acd_thresh.setSingleStep(0.01)
        self.acd_thresh.setDecimals(2)
        self.acd_thresh.setValue(float(self.cfg.get("acd_threshold", 0.08) or 0.08))
        ab.addWidget(self.acd_thresh)
        self.acd_btn = QtWidgets.QPushButton("Generate ACD")
        self.acd_btn.clicked.connect(self._generate_acd)
        ab.addWidget(self.acd_btn)
        self.acd_clear_btn = QtWidgets.QPushButton("Clear ACD")
        self.acd_clear_btn.clicked.connect(self._clear_acd)
        ab.addWidget(self.acd_clear_btn)
        ab.addStretch(1)
        rv.addLayout(ab)
        self.hull_info = QtWidgets.QLabel("Select a model row to load it. Drag box handles to "
                                          "resize, drag the body to move. Stack boxes for "
                                          "concave shapes.")
        self.hull_info.setWordWrap(True)
        rv.addWidget(self.hull_info)
        split.addWidget(right)
        split.setSizes([520, 660])

    # ---- environment / process ----
    def _env(self):
        env = QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONPATH", _SRC_DIR + os.pathsep + env.value("PYTHONPATH", ""))
        return env

    def _append(self, s):
        self.log.appendPlainText(s.rstrip("\n"))

    def _run_stream(self, args, on_finish=None):
        if self.proc is not None:
            self._append("(a build is already running)")
            return
        self._append("\n$ " + " ".join(args))
        self.run_btn.setEnabled(False)
        self.proc = QProcess(self)
        self.proc.setProcessEnvironment(self._env())
        self.proc.setProcessChannelMode(QProcess.MergedChannels)
        self.proc.readyReadStandardOutput.connect(
            lambda: self._append(bytes(self.proc.readAllStandardOutput()).decode("utf-8", "replace")))
        self.proc.finished.connect(lambda code, _s: self._on_finish(code, on_finish))
        self.proc.start(args[0], args[1:])

    def _on_finish(self, code, cb):
        self._append("[exit %d]" % code)
        self.proc = None
        self.run_btn.setEnabled(True)
        if cb:
            cb(code)

    def _stop(self):
        if self.proc:
            self.proc.kill()
            self._append("[stopped]")

    def _capture(self, args, done):
        """Run args, collect all output, call done(text) on finish."""
        self.cap = QProcess(self)
        self.cap.setProcessEnvironment(self._env())
        self.cap.setProcessChannelMode(QProcess.MergedChannels)
        self.cap_buf = ""
        self.cap.readyReadStandardOutput.connect(
            lambda: setattr(self, "cap_buf",
                            self.cap_buf + bytes(self.cap.readAllStandardOutput()).decode("utf-8", "replace")))
        self.cap.finished.connect(lambda *_: done(self.cap_buf))
        self.cap.start(args[0], args[1:])

    # ---- runs ----
    def _run(self):
        v = self._vals()
        a = _terrain_args(v, self.plugin_list, self.bsa_list)
        if v["models"]:
            a.append("--models")
            a += _model_args(v)
            a += ["--tree-scale", v["tree_scale"]]
            tm = self._tree_map()
            if tm:
                out_dir = os.path.dirname(os.path.abspath(v["out"])) or "."
                os.makedirs(out_dir, exist_ok=True)
                tmf = os.path.join(out_dir, "treemap.json")
                with open(tmf, "w", encoding="utf-8") as fh:
                    json.dump(tm, fh, indent=2)
                a += ["--tree-map-file", tmf]
        self._save_cfg()
        self._run_stream(a, on_finish=self._maybe_copy_materials)

    def _maybe_copy_materials(self, code):
        if code != 0 or not self.copy_mats.isChecked():
            return
        import shutil
        v = self._vals()
        src = os.path.join(os.path.dirname(os.path.abspath(v["out"])), "materials")
        gamedir = v["gamedir"].strip()
        if gamedir and os.path.isdir(src):
            try:
                shutil.copytree(src, os.path.join(gamedir, "materials"), dirs_exist_ok=True)
                self._append("Copied materials -> %s\\materials" % gamedir)
            except Exception as e:
                self._append("material copy failed: %r" % e)

    def _scan_species(self):
        v = self._vals()
        a = _terrain_args(v, self.plugin_list, self.bsa_list) + ["--models", "--list-tree-species"]
        self._append("Scanning species…")
        self._capture(a, self._set_species)

    def _set_species(self, text):
        # the CLI prints one species path per line (contains a path separator)
        sp = [l.strip() for l in text.splitlines()
              if l.strip() and ("\\" in l or "/" in l)]
        self.cfg["species"] = sp
        self._populate_species(sp, self.cfg.get("tree_map", {}))
        self._append("Found %d species." % len(sp))

    def _populate_species(self, species, mapping):
        while self.species_form.rowCount():
            self.species_form.removeRow(0)
        self.species_edits = {}
        for sp in sorted(species):
            e = QtWidgets.QLineEdit(mapping.get(sp, ""))
            self.species_edits[sp] = e
            self.species_form.addRow(sp, e)

    def _tree_map(self):
        tm = {}
        d = self.getters["tree_default"]().strip()
        if d:
            tm["_default"] = d
        for sp, e in self.species_edits.items():
            if e.text().strip():
                tm[sp] = e.text().strip()
        return tm

    def _scan_interiors(self):
        v = self._vals()
        a = _common_args(v, self.plugin_list, self.bsa_list) + ["--list-interiors"]
        if self.int_filter.text().strip():
            a.append(self.int_filter.text().strip())
        self._append("Listing interiors…")
        self._capture(a, self._set_interiors)

    def _set_interiors(self, text):
        import re
        self.interiors = []
        self.int_list.clear()
        for line in text.splitlines():
            m = re.match(r"\s+(\S+)\s+(.*?)\s+(\d+) refs\s+\(0x", line)
            if m:
                edid, name, refs = m.group(1), m.group(2).strip(), int(m.group(3))
                self.interiors.append((edid, name, refs))
                self.int_list.addItem("%-40s %5d refs   %s" % (edid, refs, name))
        self._append("Found %d interiors." % len(self.interiors))

    def _build_interior(self):
        row = self.int_list.currentRow()
        if row < 0:
            self._append("Pick an interior from the list first.")
            return
        v = self._vals()
        edid = self.interiors[row][0]
        out_dir = os.path.dirname(os.path.abspath(v["out"])) or "."
        os.makedirs(out_dir, exist_ok=True)
        out = os.path.join(out_dir, edid + ".vmf")
        a = _common_args(v, self.plugin_list, self.bsa_list) + \
            ["--interior", edid, "--out", out, "--scale", v["scale"]]
        if self.skybox_room.isChecked():
            a.append("--skybox-room")
        if self.instance_into.isChecked():
            host = self.instance_host.text().strip()
            if not host:
                self._append("Tick func_instance but pick a host .vmf first.")
                return
            a += ["--instance-into", host]
        if v["models"]:
            a.append("--models")
            a += _model_args(v)
        self._save_cfg()
        self._run_stream(a)

    # ---- model table ----
    def _scan_models(self):
        v = self._vals()
        a = _common_args(v, self.plugin_list, self.bsa_list) + ["--list-models"]
        if self.model_src.currentText() == "Interior":
            row = self.int_list.currentRow()
            if row < 0:
                self._append("Interior source: select a room on the Interiors tab first.")
                return
            a += ["--interior", self.interiors[row][0]]
        else:
            a.append("--cells=" + v["cells"].replace(" ", ""))
        if self.model_filter.text().strip():
            a.append(self.model_filter.text().strip())
        self._append("Scanning models…")
        self._capture(a, self._set_models_text)

    def _set_models_text(self, text):
        import re
        models = []
        for line in text.splitlines():
            m = re.match(r"\s+(\S+)\s+(\d+)\s+(mesh|tree)\s*$", line)
            if m:
                models.append((m.group(1), int(m.group(2))))
        self._fill_table(models, "scan")

    def _load_compiled(self):
        cache = os.path.join(self._work_dir(), ".build_cache.json")
        if not os.path.isfile(cache):
            self._append("No build cache at %s — compile a build first." % cache)
            return
        try:
            with open(cache, encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception as e:
            self._append("could not read build cache: %r" % e)
            return
        filt = self.model_filter.text().strip().lower()
        models = [(m, 0) for m in sorted(data) if not filt or filt in m.lower()]
        self._fill_table(models, "compiled")

    def _work_dir(self):
        out_dir = os.path.dirname(os.path.abspath(self.getters["out"]())) or "."
        return os.path.join(out_dir, "models_src")

    def _read_overrides_file(self):
        path = self.ov_path.text().strip()
        if path and os.path.isfile(path):
            try:
                with open(path, encoding="utf-8") as fh:
                    return {str(k).lower(): val for k, val in json.load(fh).items()}
            except Exception as e:
                self._append("could not read overrides JSON: %r" % e)
        return {}

    def _fill_table(self, models, source):
        saved = self._read_overrides_file()
        self.model_rows = {}
        self.table.setRowCount(len(models))
        for i, (modl, _n) in enumerate(models):
            ov = saved.get(modl.lower(), {})
            it = QtWidgets.QTableWidgetItem(modl)
            it.setFlags(it.flags() & ~Qt.ItemIsEditable)
            it.setData(Qt.UserRole, modl)
            self.table.setItem(i, 0, it)
            coll = QtWidgets.QComboBox()
            coll.addItems(["(global)", "auto", "acd", "full", "bbox", "ramp", "hulls", "none"])
            coll.setCurrentText(ov.get("collision") or "(global)")
            self.table.setCellWidget(i, 1, coll)
            ramp = QtWidgets.QComboBox()
            ramp.addItems(["+x", "-x", "+y", "-y"])
            ramp.setCurrentText(ov.get("ramp_axis") or "+x")
            self.table.setCellWidget(i, 2, ramp)
            scale = QtWidgets.QLineEdit("" if ov.get("scale") in (None, 1.0) else str(ov.get("scale")))
            self.table.setCellWidget(i, 3, scale)
            surf = QtWidgets.QLineEdit(ov.get("surfaceprop") or "")
            self.table.setCellWidget(i, 4, surf)
            skip = QtWidgets.QCheckBox()
            skip.setChecked(bool(ov.get("skip")))
            self.table.setCellWidget(i, 5, skip)
            self.model_rows[modl] = {"collision": coll, "ramp": ramp, "scale": scale,
                                     "surf": surf, "skip": skip,
                                     "hulls": list(ov.get("hulls") or []),
                                     "acd_parts": ov.get("acd_parts")}
        self._append("Loaded %d models (%s; %d with saved overrides)."
                     % (len(models), source,
                        sum(1 for k in self.model_rows if k.lower() in saved)))

    def _selected_model(self):
        it = self.table.item(self.table.currentRow(), 0)
        return it.data(Qt.UserRole) if it else None

    def _on_model_selected(self):
        modl = self._selected_model()
        if not modl or modl == self.cur_model or self.plotter is None:
            return
        self._load_model_3d(modl)

    # ---- 3D viewport ----
    def _style_viewport(self):
        """Dark-blue gradient background; grid added per-model in _grid()."""
        try:
            self.plotter.set_background("#0b1b3a", top="#06101f")   # dark blue gradient
        except Exception:
            self.plotter.set_background("#0b1b3a")

    def _grid(self):
        try:
            self.plotter.show_grid(color="#33507f", grid="back", location="outer",
                                   ticks="outside")
        except Exception:
            pass

    def _load_model_3d(self, modl):
        self.cur_model = modl
        self._undo_stack.clear()   # undo history is per-model
        self._redo_stack.clear()
        self._clear_boxes()
        self.acd_actors = []
        self.plotter.clear()
        self._style_viewport()
        smd = os.path.join(self._work_dir(), slugify(modl) + ".smd")
        if not os.path.isfile(smd):
            self.hull_info.setText("No compiled SMD for this model at %s — build it first." % smd)
            self.plotter.render()
            return
        groups = read_smd_grouped(smd)
        all_pts, ntri, ntex = [], 0, 0
        for mat, g in groups.items():
            if not g["tris"]:
                continue
            pts = np.array(g["points"], dtype=float)
            faces = np.hstack([[3, *t] for t in g["tris"]]).astype(np.int64)
            mesh = pv.PolyData(pts, faces)
            all_pts.append(pts)
            ntri += len(g["tris"])
            tex = self._load_texture(mat)
            if tex is not None:
                mesh.active_texture_coordinates = np.array(g["uvs"], dtype=float)
                self.plotter.add_mesh(mesh, texture=tex, name="m_%d" % ntex)
                ntex += 1
            else:
                self.plotter.add_mesh(mesh, color="#8d99ae", name="flat_%d" % len(all_pts))
        verts = np.vstack(all_pts) if all_pts else np.zeros((0, 3))
        self.bb = mesh_bounds([tuple(p) for p in verts]) if len(verts) else (0, 0, 0, 64, 64, 64)
        for h in self.model_rows[modl]["hulls"]:
            s = self._hull_spec(h)
            if len(s["bounds"]) == 6:
                self._spawn_box(self._to_vtk_bounds(s["bounds"]), s["type"],
                                s["axis"], s["top_scale"])
        if self.model_rows[modl].get("acd_parts"):
            self._show_acd(self.model_rows[modl]["acd_parts"])
        self._grid()
        self.plotter.reset_camera()
        self.plotter.render()
        self.hull_info.setText("%s — %d tri(s), %d/%d material(s) textured, %d hull(s)."
                               % (os.path.basename(modl), ntri, ntex, len(groups), len(self.boxes)))

    # ---- CoACD generation / preview ----
    def _generate_acd(self):
        if self.plotter is None or self.cur_model is None:
            self._append("Load a model row first.")
            return
        from . import acd
        if not acd.available():
            self._append("CoACD not installed — run: pip install coacd")
            return
        smd = os.path.join(self._work_dir(), slugify(self.cur_model) + ".smd")
        verts, tris = read_smd_mesh(smd)
        if not tris:
            self._append("No mesh geometry to decompose.")
            return
        thr = float(self.acd_thresh.value())
        modl = self.cur_model
        self.acd_btn.setEnabled(False)
        self._append("Generating ACD for %s (threshold %.2f)…" % (os.path.basename(modl), thr))
        subs = [{"verts": verts, "tris": tris}]

        def work():
            try:
                parts = acd.decompose_isolated(subs, timeout=120, threshold=thr, max_hulls=-1)
            except Exception as e:
                self._append("ACD failed: %r" % e)
                parts = None
            self.acd_ready.emit(modl, parts)
        import threading
        threading.Thread(target=work, daemon=True).start()

    def _on_acd_ready(self, modl, parts):
        self.acd_btn.setEnabled(True)
        if not parts:
            self._append("ACD produced no parts (mesh too complex/degenerate, or timed out).")
            return
        if modl in self.model_rows:
            self.model_rows[modl]["acd_parts"] = parts
            self.model_rows[modl]["collision"].setCurrentText("acd")
        if modl == self.cur_model and self.plotter is not None:
            self._show_acd(parts)
            self.plotter.render()
        self._append("ACD: %d convex part(s) for %s → row set to 'acd' (Save overrides to "
                     "bake exactly these). " % (len(parts), os.path.basename(modl)))

    _ACD_COLORS = ["#ff7043", "#ffca28", "#9ccc65", "#26c6da", "#ab47bc",
                   "#ec407a", "#7e57c2", "#5c6bc0", "#66bb6a", "#ffa726"]

    def _show_acd(self, parts):
        self._clear_acd_actors()
        self.acd_actors = []
        for i, (pv_, pf) in enumerate(parts):
            try:
                pts = np.array(pv_, dtype=float)
                faces = np.hstack([[3, *f] for f in pf]).astype(np.int64)
                mesh = pv.PolyData(pts, faces)
                a = self.plotter.add_mesh(mesh, color=self._ACD_COLORS[i % len(self._ACD_COLORS)],
                                          opacity=0.45, show_edges=True, edge_color="#0b1b3a",
                                          name="acd_%d" % i)
                self.acd_actors.append("acd_%d" % i)
            except Exception:
                pass

    def _clear_acd_actors(self):
        for name in getattr(self, "acd_actors", []):
            try:
                self.plotter.remove_actor(name)
            except Exception:
                pass
        self.acd_actors = []

    def _clear_acd(self):
        self._clear_acd_actors()
        if self.cur_model in self.model_rows:
            self.model_rows[self.cur_model]["acd_parts"] = None
        if self.plotter is not None:
            self.plotter.render()
        self._append("Cleared ACD preview/parts for the current model.")

    # ---- export ----
    def _export_data(self):
        """Write a shareable JSON bundle of the current model overrides + context."""
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export model data", "oblivion2vmf_export.json", "JSON (*.json)")
        if not path:
            return
        bundle = {
            "tool": "oblivion2vmf",
            "source": self.model_src.currentText(),
            "filter": self.model_filter.text(),
            "cells": self.getters["cells"](),
            "models": list(self.model_rows),
            "overrides": self._overrides_from_rows(),
        }
        try:
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(bundle, fh, indent=2, sort_keys=True)
            n = len(bundle["overrides"])
            nacd = sum(1 for r in self.model_rows.values() if r.get("acd_parts"))
            self._append("Exported %d model(s), %d override(s), %d with ACD parts → %s\n"
                         "Send that file back to share your collision setup."
                         % (len(self.model_rows), n, nacd, path))
        except Exception as e:
            self._append("export failed: %r" % e)

    def _load_texture(self, material):
        """Decode the model material's VTF (from the build's materials dir, then
        the gamedir) to a pyvista Texture, or None to fall back to flat colour."""
        from .model import MODEL_PREFIX
        from .vtf import vtf_to_rgb
        cache = getattr(self, "_tex_cache", None)
        if cache is None:
            cache = self._tex_cache = {}
        if material in cache:
            return cache[material]
        rel = os.path.join("materials", "models", MODEL_PREFIX, material + ".vtf")
        out_dir = os.path.dirname(os.path.abspath(self.getters["out"]()))
        candidates = [os.path.join(out_dir, rel)]
        gd = self.getters["gamedir"]().strip()
        if gd:
            candidates.append(os.path.join(gd, rel))
        tex = None
        for path in candidates:
            if os.path.isfile(path):
                rgb = vtf_to_rgb(path)
                if rgb is not None:
                    tex = pv.Texture(np.flipud(rgb).copy())   # VTK texture origin = bottom-left
                    break
        cache[material] = tex
        return tex

    @staticmethod
    def _to_vtk_bounds(b):
        # hull [x0,y0,z0,x1,y1,z1] -> vtk bounds (xmin,xmax,ymin,ymax,zmin,zmax)
        return (b[0], b[3], b[1], b[4], b[2], b[5])

    @staticmethod
    def _from_vtk_bounds(vb):
        return [vb[0], vb[2], vb[4], vb[1], vb[3], vb[5]]

    def _snap_view(self, which):
        if self.plotter is None:
            return
        {"xy": self.plotter.view_xy, "xz": self.plotter.view_xz,
         "yz": self.plotter.view_yz, "iso": self.plotter.view_isometric}[which]()
        self.plotter.render()

    _SHAPE_COLORS = {"box": "#46c0ff", "wedge": "#ffd54f", "trap": "#80cbc4"}

    @staticmethod
    def _hull_spec(entry):
        """Normalize a stored hull (legacy 6-float list or shape dict)."""
        if isinstance(entry, dict):
            return {"type": (entry.get("type") or "box").lower(),
                    "bounds": [float(c) for c in entry.get("bounds", [])[:6]],
                    "axis": entry.get("axis", "+x"),
                    "top_scale": float(entry.get("top_scale", 0.5))}
        return {"type": "box", "bounds": [float(c) for c in entry[:6]],
                "axis": "+x", "top_scale": 0.5}

    def _spawn_box(self, vtk_bounds, shape="box", axis="+x", top_scale=0.5):
        entry = {"widget": None, "bounds": vtk_bounds, "type": shape, "axis": axis,
                 "top_scale": top_scale, "name": "hullprev_%d" % len(self.boxes)}

        def cb(box, widget):
            nb = tuple(box.bounds)
            if nb != tuple(entry["bounds"]):   # one snapshot per completed drag
                self._push_undo()              # entry still holds pre-drag bounds
                entry["bounds"] = nb
            self._update_hull_preview(entry)
        w = self.plotter.add_box_widget(cb, bounds=vtk_bounds, rotation_enabled=False,
                                        pass_widget=True,
                                        color=self._SHAPE_COLORS.get(shape, "#46c0ff"))
        entry["widget"] = w
        self.boxes.append(entry)
        self._update_hull_preview(entry)

    def _update_hull_preview(self, entry):
        """Draw the actual wedge/trapezium inside its box-gizmo envelope (the box
        widget itself is the shape for type 'box')."""
        if entry["type"] == "box":
            return
        part = hull_from_spec({"type": entry["type"],
                               "bounds": self._from_vtk_bounds(entry["bounds"]),
                               "axis": entry["axis"], "top_scale": entry["top_scale"]})
        if part is None:
            return
        verts, faces = part
        mesh = pv.PolyData(np.array(verts, dtype=float),
                           np.hstack([[3, *f] for f in faces]).astype(np.int64))
        self.plotter.add_mesh(mesh, name=entry["name"], opacity=0.45, show_edges=True,
                              color=self._SHAPE_COLORS.get(entry["type"], "#80cbc4"))

    def _default_box_bounds(self):
        x0, y0, z0, x1, y1, z1 = getattr(self, "bb", (0, 0, 0, 64, 64, 64))
        return (x0, x1, y0, y1, z0, (z0 + z1) / 2.0)

    def _add_hull(self, shape):
        if self.plotter is None or self.cur_model is None:
            return
        self._push_undo()
        self._spawn_box(self._default_box_bounds(), shape,
                        axis=self.wedge_axis.currentText(),
                        top_scale=float(self.trap_top.value()))
        self.plotter.render()

    def _remove_box(self):
        if self.plotter is not None and self.boxes:
            self._push_undo()
            gone = self.boxes.pop()                # drop the most recently added
            try:
                self.plotter.remove_actor(gone["name"])
            except Exception:
                pass
            self._respawn_boxes()

    def _fit_box(self):
        if self.plotter is not None and self.boxes and hasattr(self, "bb"):
            self._push_undo()
            self.boxes[-1]["bounds"] = (self.bb[0], self.bb[3], self.bb[1],
                                        self.bb[4], self.bb[2], self.bb[5])
            self._respawn_boxes()

    def _clear_boxes(self):
        if _HAVE_3D and self.plotter is not None:
            try:
                self.plotter.clear_box_widgets()
            except Exception:
                pass
            for e in self.boxes:
                try:
                    self.plotter.remove_actor(e["name"])
                except Exception:
                    pass
        self.boxes = []

    def _respawn_boxes(self):
        """Rebuild all widgets + previews from current state (after add/remove)."""
        specs = [(e["bounds"], e["type"], e["axis"], e["top_scale"]) for e in self.boxes]
        self._clear_boxes()
        for vb, t, ax, ts in specs:
            self._spawn_box(vb, t, ax, ts)
        self.plotter.render()

    # ---- hull undo/redo ----
    def _hull_state(self):
        """Deep, widget-free snapshot of self.boxes (only plain scalar/list data
        survives, so snapshots stay JSON-safe and never pin VTK objects)."""
        snap = []
        for e in self.boxes:
            d = {}
            for k, v in e.items():
                if k in ("widget", "actor", "name"):
                    continue
                if isinstance(v, (list, tuple)):
                    d[k] = list(v)
                elif isinstance(v, (str, int, float, bool)):
                    d[k] = v
            snap.append(d)
        return snap

    def _push_undo(self):
        """Record the pre-mutation hull state; any new edit invalidates redo."""
        self._undo_stack.append(self._hull_state())
        if len(self._undo_stack) > 50:
            del self._undo_stack[0]
        self._redo_stack.clear()

    def _restore_state(self, state):
        """Rebuild self.boxes from a snapshot — data-only when no plotter is up."""
        self._clear_boxes()
        for s in state:
            bounds = tuple(s.get("bounds", ()))
            if self.plotter is not None:
                self._spawn_box(bounds, s.get("type", "box"), s.get("axis", "+x"),
                                s.get("top_scale", 0.5))
            else:
                e = dict(s)
                e["bounds"] = bounds
                e.setdefault("type", "box")
                e.setdefault("axis", "+x")
                e.setdefault("top_scale", 0.5)
                e["widget"] = None
                e["name"] = "hullprev_%d" % len(self.boxes)
                self.boxes.append(e)
        if self.plotter is not None:
            self.plotter.render()

    @staticmethod
    def _focused_text_editor():
        """Window shortcuts outrank widget keys; hand undo/redo back to text fields."""
        fw = QtWidgets.QApplication.focusWidget()
        if isinstance(fw, (QtWidgets.QLineEdit, QtWidgets.QPlainTextEdit,
                           QtWidgets.QTextEdit)):
            return fw
        return None

    def _undo(self):
        fw = self._focused_text_editor()
        if fw is not None:
            fw.undo()
            return
        if not self._undo_stack or self.cur_model is None:
            return
        self._redo_stack.append(self._hull_state())
        self._restore_state(self._undo_stack.pop())
        self._append("undo: %d hull(s)" % len(self.boxes))

    def _redo(self):
        fw = self._focused_text_editor()
        if fw is not None:
            fw.redo()
            return
        if not self._redo_stack or self.cur_model is None:
            return
        self._undo_stack.append(self._hull_state())
        self._restore_state(self._redo_stack.pop())
        self._append("redo: %d hull(s)" % len(self.boxes))

    def _commit_hulls(self):
        if self.cur_model is None:
            return
        hulls = []
        for e in self.boxes:
            b = [round(float(c), 2) for c in self._from_vtk_bounds(e["bounds"])]
            if e["type"] == "box":
                hulls.append(b)                    # legacy compact form
            elif e["type"] == "wedge":
                hulls.append({"type": "wedge", "bounds": b, "axis": e["axis"]})
            else:
                hulls.append({"type": "trap", "bounds": b,
                              "top_scale": round(float(e["top_scale"]), 3)})
        self.model_rows[self.cur_model]["hulls"] = hulls
        if hulls:
            self.model_rows[self.cur_model]["collision"].setCurrentText("hulls")
        self._append("%s: %d hull(s) set (now Save overrides)." % (self.cur_model, len(hulls)))

    # ---- overrides persistence ----
    def _overrides_from_rows(self):
        out = {}
        for modl, r in self.model_rows.items():
            d = {}
            coll = r["collision"].currentText()
            if coll not in ("(global)", ""):
                d["collision"] = coll
                if coll == "ramp":
                    d["ramp_axis"] = r["ramp"].currentText()
                if coll == "hulls" and r.get("hulls"):
                    # boxes are compact 6-float lists; wedges/trapeziums are dicts
                    # (already rounded at commit time)
                    d["hulls"] = [h if isinstance(h, dict)
                                  else [round(float(c), 2) for c in h]
                                  for h in r["hulls"]]
                if coll == "acd" and r.get("acd_parts"):
                    # store the exact previewed convex parts so the build bakes
                    # them verbatim instead of recomputing CoACD
                    d["acd_parts"] = [[[[round(float(c), 3) for c in v] for v in pv_],
                                       [[int(i) for i in f] for f in pf]]
                                      for pv_, pf in r["acd_parts"]]
            sc = r["scale"].text().strip()
            if sc:
                try:
                    if float(sc) != 1.0:
                        d["scale"] = float(sc)
                except ValueError:
                    pass
            if r["surf"].text().strip():
                d["surfaceprop"] = r["surf"].text().strip()
            if r["skip"].isChecked():
                d["skip"] = True
            if d:
                out[modl.lower()] = d
        return out

    def _save_overrides(self):
        path = self.ov_path.text().strip()
        if not path:
            out_dir = os.path.dirname(os.path.abspath(self.getters["out"]())) or "."
            path = os.path.join(out_dir, "model_overrides.json")
            self.ov_path.setText(path)
        merged = self._read_overrides_file()
        listed = {k.lower() for k in self.model_rows}
        merged = {k: val for k, val in merged.items() if k not in listed}
        merged.update(self._overrides_from_rows())
        try:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(merged, fh, indent=2, sort_keys=True)
            self._append("Saved %d model override(s) -> %s" % (len(merged), path))
        except Exception as e:
            self._append("save overrides failed: %r" % e)

    # ---- config ----
    def _save_cfg(self):
        for k, g in self.getters.items():
            self.cfg[k] = g()
        self.cfg["bsa"] = list(self.bsa_list)
        self.cfg["plugins"] = list(self.plugin_list)
        self.cfg["copy_materials"] = self.copy_mats.isChecked()
        self.cfg["int_filter"] = self.int_filter.text()
        self.cfg["skybox_room"] = self.skybox_room.isChecked()
        self.cfg["instance_into"] = self.instance_into.isChecked()
        self.cfg["instance_host"] = self.instance_host.text()
        self.cfg["model_src"] = self.model_src.currentText()
        self.cfg["model_filter"] = self.model_filter.text()
        self.cfg["model_overrides"] = self.ov_path.text()
        self.cfg["tree_map"] = self._tree_map()
        _save(self.cfg)

    def closeEvent(self, e):
        self._save_cfg()
        super().closeEvent(e)

    def _apply_dark(self):
        self.setStyleSheet(
            "QWidget { background:#23272e; color:#d6dae0; }"
            "QLineEdit, QComboBox, QPlainTextEdit, QListWidget, QTableWidget, QGroupBox {"
            " background:#1b1f25; border:1px solid #333a44; }"
            "QPushButton { background:#2d3640; padding:4px 10px; border:1px solid #3a424d;"
            " border-radius:3px; }"
            "QPushButton:hover { background:#39424f; }"
            "QTabBar::tab { background:#1b1f25; padding:6px 12px; }"
            "QTabBar::tab:selected { background:#2d3640; }")


def main():
    if not _HAVE_3D:
        sys.stderr.write("pyvista/PySide6 not fully available: %r\n" % (_IMPORT_ERR,))
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    win = Main()
    win.show()
    app.exec()


if __name__ == "__main__":
    main()
