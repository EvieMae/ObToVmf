"""Command-line interface for oblivion2vmf (terrain milestone)."""
from __future__ import annotations

import argparse
import os
import sys

from . import __version__
from .esm import TerrainExtractor, read_masters
from .regions import REGIONS, WORLDSPACES
from .textures import Texturer
from .vmf import (
    ALLOWED_POWERS,
    DEFAULT_MATERIAL,
    MAX_COORD,
    MAX_MAP_DISPINFO,
    REAL_WORLD_SCALE,
    WATER_MATERIAL,
    write_vmf,
)

# Common install locations for Oblivion.esm (original + 2025 Remastered).
_DEFAULT_ESM_HINTS = [
    r"C:\Program Files (x86)\Steam\steamapps\common\Oblivion Remastered\OblivionRemastered\Content\Dev\ObvData\Data\Oblivion.esm",
    r"C:\Program Files (x86)\Steam\steamapps\common\Oblivion\Data\Oblivion.esm",
]


def _parse_ws(value):
    if value is None:
        return WORLDSPACES["tamriel"]
    v = value.strip().lower()
    if v in WORLDSPACES:
        return WORLDSPACES[v]
    return int(v, 0)   # accepts 0x3c, 60, etc.


def _find_default_esm():
    for p in _DEFAULT_ESM_HINTS:
        if os.path.isfile(p):
            return p
    return None


def _load_order(esm, plugins):
    """[esm, *plugins] with each plugin existence-checked."""
    order = [esm]
    for p in (plugins or []):
        if not os.path.isfile(p):
            raise SystemExit("Plugin not found: %s" % p)
        order.append(p)
    return order


def _parse_load_order(order, **kwargs):
    """Parse a whole load order into one extractor (later plugins override/add),
    then finalize. Warns if a plugin's masters don't match the load-order prefix
    (which would misalign FormIDs)."""
    ex = TerrainExtractor(**kwargs)
    names = [os.path.basename(p).lower() for p in order]
    for i, path in enumerate(order):
        if i:
            masters = [m.lower() for m in read_masters(path)]
            if masters != names[:i]:
                print("[warn] %s expects masters %s but load order prefix is %s. "
                      "Pass plugins in the correct order (masters first) or FormIDs "
                      "may misalign." % (os.path.basename(path), masters, names[:i]))
        ex.parse_file(path)
    ex.finalize()
    return ex


def build_parser():
    p = argparse.ArgumentParser(
        prog="oblivion2vmf",
        description="Convert Oblivion (TES4) terrain into a Source/Hammer .vmf "
                    "displacement map for Garry's Mod (terrain-only milestone).",
    )
    p.add_argument("--esm", help="path to Oblivion.esm (defaults to a Steam install if found)")
    p.add_argument("--plugin", action="append", default=[], metavar="PLUGIN.esp",
                   help="additional plugin (mod .esm/.esp) to load ON TOP of the ESM, "
                        "in load order (repeatable). Later plugins override/add records "
                        "(e.g. Open Cities). Remember loose mod meshes need --data-dir too.")
    p.add_argument("--worldspace", help="worldspace name or FormID (default: tamriel / 0x3C)")

    sel = p.add_mutually_exclusive_group()
    sel.add_argument("--cells", metavar="minX,minY,maxX,maxY",
                     help="inclusive exterior-cell bounding box to convert")
    sel.add_argument("--region", choices=sorted(REGIONS),
                     help="named region (APPROXIMATE bounding box; verify with --list-cells)")
    sel.add_argument("--list-cells", action="store_true",
                     help="list the exterior cells present in the worldspace and exit")
    sel.add_argument("--list-interiors", nargs="?", const="", metavar="FILTER",
                     help="list interior cells (rooms) across the load order, optionally "
                          "filtered by substring (EDID or display name), then exit")
    sel.add_argument("--interior", metavar="EDID",
                     help="convert ONE interior cell (room) to a VMF: a sealed room shell "
                          "+ its placed objects as props + lights from placed LIGH refs. "
                          "Use --list-interiors to find the EDID. Needs --bsa/--data-dir.")
    p.add_argument("--skybox-room", action="store_true",
                   help="(interiors) wrap the room in tools/toolsskybox instead of the "
                        "default sealed tools/toolsblack, so the enclosure renders the 2D "
                        "sky on every face (for open-air courtyard cells).")
    p.add_argument("--instance-into", metavar="HOST.vmf",
                   help="(interiors) after writing the room, add it to this host .vmf as a "
                        "func_instance (relative file path), auto-placed in a free spot "
                        "beside the host's existing geometry. The host is created if absent.")
    p.add_argument("--instance-origin", metavar="X,Y,Z",
                   help="(interiors) explicit origin for --instance-into instead of "
                        "auto-placement, in Hammer units, e.g. 0,0,0")

    p.add_argument("--out", default="terrain.vmf", help="output .vmf path (default: terrain.vmf)")
    p.add_argument("--scale", type=float, default=REAL_WORLD_SCALE,
                   help="Oblivion-unit -> Hammer-unit factor (default %(default)s = "
                        "player-accurate real-world; use 1.0 for raw 1:1)")
    p.add_argument("--power", type=int, default=3, choices=ALLOWED_POWERS,
                   help="displacement power (3=4x4 disps/cell [default], 4=2x2 [coarser "
                        "vis leaves], 2=8x8 [finest vis, most brushes])")
    p.add_argument("--material", default=DEFAULT_MATERIAL,
                   help="fallback flat material when --no-textures is used")
    p.add_argument("--terrain-tex-scale", type=float, default=0.25,
                   help="displacement texture UV scale (world units/texel); lower = "
                        "finer/smaller terrain texture features (default 0.25)")
    p.add_argument("--terrain-lightmapscale", type=int, default=32,
                   help="lightmap luxel size for terrain/water faces; higher = faster "
                        "vrad + smaller bsp, coarser terrain shadows (default 32)")
    p.add_argument("--outer-power", type=int, default=None, choices=ALLOWED_POWERS,
                   help="use this lower displacement power for edge cells (coarse "
                        "terrain LOD for distant border land); default: same as --power")
    p.add_argument("--outer-margin", type=int, default=1,
                   help="how many cells deep the --outer-power border ring is (default 1)")
    p.add_argument("--vis-floor", action="store_true",
                   help="extend every displacement brush down to one shared floor so "
                        "the terrain is solid mass (hills occlude -> vvis can cull) and "
                        "the bottom is sealed. NOTE: makes vvis MUCH slower and only "
                        "helps if the map is also sealed on the sides/top (see --seal-sky)")
    p.add_argument("--vis-floor-depth", type=float, default=512.0,
                   help="how far below the lowest terrain the --vis-floor sits (HU, default 512)")
    p.add_argument("--seal-sky", action="store_true",
                   help="wrap the map in a toolsskybox box so it's SEALED (no leak) and "
                        "vvis can actually cull. Use together with --vis-floor; on its own "
                        "--vis-floor doesn't help because an unsealed map floods to one leaf")
    p.add_argument("--skybox-model", action="store_true",
                   help="bake the region (+margin) into ONE low-poly terrain model with a "
                        "baked top-down texture and place it as a 3D-skybox backdrop, so "
                        "distant terrain shows past the fog (a model dodges the displacement "
                        "cap + leaks). Auto-enables --seal-sky. Needs --bsa/--data-dir.")
    p.add_argument("--skybox-margin", type=int, default=6,
                   help="cells of surrounding terrain to include in the skybox backdrop "
                        "beyond the playable box (default 6)")
    p.add_argument("--sky-scale", type=int, default=16,
                   help="3D skybox scale factor (default 16; the engine standard)")
    p.add_argument("--skybox-model-file",
                   help="use a pre-built model (path like models/oblivion2vmf/x.mdl, already "
                        "in the gamedir) as the 3D-skybox backdrop instead of the auto-baked "
                        "terrain — e.g. one you decimated/baked in Blender from --export-obj")
    p.add_argument("--skybox-obj", metavar="PATH.obj",
                   help="compile an OBJ (e.g. your edited --export-obj output) to a model "
                        "and place it as the 3D-skybox backdrop. Materials become flat "
                        "UnlitGeneric from the .mtl colours. Auto-enables --seal-sky.")
    p.add_argument("--export-obj", metavar="PATH.obj",
                   help="export the region (+--skybox-margin) terrain + props to a Wavefront "
                        "OBJ+MTL (in the map's coordinate frame) for Blender, then exit. "
                        "Decimate/bake it and feed it back via --skybox-model-file.")
    p.add_argument("--obj-large-only", action="store_true",
                   help="--export-obj: keep only large props (buildings/towers/walls, "
                        ">=600 HU); skip rocks/bushes/clutter for a light horizon model")
    p.add_argument("--obj-min-prop-size", type=float, default=0.0,
                   help="--export-obj: skip props whose world size (HU) is below this "
                        "(0 = keep all; overrides --obj-large-only's 600)")
    p.add_argument("--max-prop-size", type=float, default=4000.0,
                   help="skip placing props whose model is bigger than this (HU); catches "
                        "Oblivion distant-LOD landmark meshes like the XXXL flags (real "
                        "props max ~2400 HU). 0 = no cap")
    p.add_argument("--no-prop-fade", dest="prop_fade", action="store_false",
                   help="don't distance-fade props. By default clutter/plants/bushes/"
                        "rocks/trees fade out at range (big perf win on open terrain); "
                        "buildings fade only at the fog wall")
    p.add_argument("--fade-scale", type=float, default=1.0,
                   help="multiply all prop fade distances (default 1.0). Lower = props "
                        "cull closer = more FPS (e.g. 0.6); higher = props persist farther")
    p.add_argument("--no-textures", dest="textures", action="store_false",
                   help="skip Oblivion land textures; use the flat --material instead")
    p.add_argument("--fourway", action="store_true",
                   help="use Lightmapped_4WayBlend (up to 4 textures/displacement) instead "
                        "of 2-way. NOTE: requires a CS:GO-branch vbsp or current GMod tools "
                        "to compile; stock Source SDK 2013 vbsp silently drops the blend data.")
    p.add_argument("--no-water", dest="water", action="store_false",
                   help="don't add water volumes (by default a swimmable water brush "
                        "is added per cell that has water, at its Oblivion water height)")
    p.add_argument("--water-material", default=None,
                   help="face material for water tops (default: a self-contained "
                        "generated 'oblivion2vmf/water'; pass e.g. nature/water_canals_water2 "
                        "to use an existing GMod/CSS water material instead)")
    p.add_argument("--water-height", type=float, default=None,
                   help="force this Oblivion water height for ALL selected cells "
                        "(overrides per-cell XCLW; useful for a single flat sea level)")
    p.add_argument("--no-lighting", dest="lighting", action="store_false",
                   help="don't add outdoor lighting (by default a light_environment "
                        "sun + sky ambient + shadow_control are added; without it the "
                        "map is fullbright)")
    p.add_argument("--no-fog", dest="fog", action="store_false",
                   help="don't add the outdoor env_fog_controller")
    p.add_argument("--sun-pitch", type=float, default=-45.0,
                   help="sun elevation in degrees below horizontal (default -45)")
    p.add_argument("--sun-yaw", type=float, default=200.0,
                   help="compass direction the sunlight comes from (default 200)")
    p.add_argument("--no-recenter", dest="recenter", action="store_false",
                   help="keep original Oblivion world coordinates (by default the "
                        "selection is recentered on the origin so it fits the map)")
    p.add_argument("--max-cells", type=int, default=512,
                   help="abort if more cells than this are selected (safety guard)")

    mg = p.add_argument_group("models (props)")
    mg.add_argument("--models", action="store_true",
                    help="extract REFR placements and convert Oblivion NIF meshes to prop_static")
    mg.add_argument("--data-dir", action="append", default=[],
                    help="folder with loose meshes\\/textures\\ (repeatable; e.g. a mod "
                         "folder like 'Open Cities Classic\\00 Core'). Searched before BSAs.")
    mg.add_argument("--bsa", action="append", default=[], metavar="ARCHIVE.bsa",
                    help="Oblivion BSA to read meshes from (repeatable)")
    mg.add_argument("--studiomdl", help="path to studiomdl.exe (else auto-detect Source SDK 2013)")
    mg.add_argument("--gamedir", help="GMod garrysmod dir with gameinfo.txt (else auto-detect)")
    mg.add_argument("--skip-compile", action="store_true",
                    help="write .smd/.qc but don't run studiomdl")
    mg.add_argument("--collision",
                    choices=["auto", "acd", "havok", "custom", "full", "bbox", "ramp",
                             "none"],
                    default="auto",
                    help="prop collision: 'auto' (solid small props, big buildings "
                         "non-solid so you pass through), 'acd' (walk-in collision via "
                         "CoACD convex decomposition of the Havok shell, needs "
                         "`pip install coacd`), 'havok' (EXACT collision: Bethesda's "
                         "Havok shell with coplanar faces -> one convex prism per "
                         "wall/floor, no approximation, no extra deps), 'full' (all "
                         "solid, per-triangle), 'bbox' (one box hull per prop), 'ramp' "
                         "(one wedge rising z0->z1 along --ramp-axis), 'custom' (use "
                         "the collision defined per-model in --model-overrides — "
                         "editor hulls / baked ACD parts; models without an entry "
                         "fall back to 'auto'), 'none'. Requires recompile.")
    mg.add_argument("--ramp-axis", choices=["+x", "-x", "+y", "-y"], default="+x",
                    help="for --collision ramp: model-local direction the wedge rises "
                         "(default +x). Rotate the prop in-world to aim it.")
    mg.add_argument("--collision-size", type=float, default=400.0,
                    help="props wider than this (HU) count as 'big' for auto/acd (default 400)")
    mg.add_argument("--acd-threshold", type=float, default=0.08,
                    help="CoACD concavity threshold for --collision acd; lower = more "
                         "hulls/finer/slower (default 0.08)")
    mg.add_argument("--acd-max-hulls", type=int, default=-1,
                    help="cap convex hulls per prop for --collision acd (-1 = no cap); "
                         "lower = cheaper physics for complex buildings (default -1)")
    mg.add_argument("--acd-jobs", type=int, default=0,
                    help="max concurrent CoACD decompositions (each an isolated "
                         "subprocess; 0 = auto min(jobs, 6)). Raise it (e.g. 24) on a "
                         "many-core machine to speed up the collision phase")
    mg.add_argument("--model-overrides", metavar="FILE.json",
                    help="JSON of PER-MODEL build settings keyed by .nif path "
                         "(lowercased), e.g. {\"architecture/anvil/house01.nif\": "
                         "{\"collision\":\"ramp\",\"ramp_axis\":\"+x\",\"scale\":1.0,"
                         "\"surfaceprop\":\"wood\",\"skip\":false}}. Overrides the global "
                         "--collision/--ramp-axis/--scale for that mesh; 'skip' drops it. "
                         "Edit this in the GUI 'Model edits' tab. Use --list-models to "
                         "enumerate the meshes in your selection.")
    mg.add_argument("--list-models", nargs="?", const="", metavar="FILTER",
                    help="list the unique .nif meshes placed in the current selection "
                         "(--cells/--region or --interior), optionally filtered by "
                         "substring, then exit. Feeds the model editor.")
    mg.add_argument("--no-model-lods", dest="model_lods", action="store_false",
                    help="don't generate decimated $lod stages for props. By default "
                         "props over ~200 tris get 2 LODs (vertex-clustering decimation) "
                         "so distant buildings/rocks cost far fewer polys")
    mg.add_argument("--jobs", "-j", type=int, default=0,
                    help="parallel worker threads for model compilation "
                         "(0 = auto: min(8, CPU count); 1 = serial). studiomdl/CoACD "
                         "release the GIL so this gives a near-linear speedup")
    mg.add_argument("--no-cache", dest="cache", action="store_false",
                    help="don't reuse cached compiled models. By default a model is "
                         "rebuilt only if its NIF or build settings (scale/flip/"
                         "collision) changed, so re-runs are near-instant")
    mg.add_argument("--rebuild-cache", action="store_true",
                    help="reconstruct the model cache from the .mdl already compiled "
                         "in the gamedir (re-hash NIFs, NO recompile), then exit. "
                         "Recovers a lost/mismatched cache fast. Needs --esm/--bsa/--gamedir "
                         "and the same model settings you build with")
    mg.add_argument("--no-trees", dest="trees", action="store_false",
                    help="don't place generated trees for SpeedTree (.spt) placements")
    mg.add_argument("--tree-model",
                    help="use an existing GMod/Source tree .mdl for ALL tree placements "
                         "(e.g. models/props_foliage/tree_pine04.mdl) instead of generating "
                         "trees; the model must already be in your GMod content")
    mg.add_argument("--tree-scale", type=float, default=1.0,
                    help="extra uniform-scale multiplier for tree models (default 1.0)")
    mg.add_argument("--tree-map-file",
                    help="JSON {sptName: modelPath, _default: modelPath} mapping tree "
                         "species to existing .mdl models (overrides --tree-model)")
    mg.add_argument("--list-tree-species", action="store_true",
                    help="list tree species (.spt) placed in the cell box, then exit")
    mg.add_argument("--flip-winding", action=argparse.BooleanOptionalAction, default=False,
                    help="reverse model triangle winding (if props are inside-out)")
    mg.add_argument("--flip-v", action=argparse.BooleanOptionalAction, default=True,
                    help="flip model texture V (NIF top-left -> Source bottom-left)")
    mg.add_argument("--angle-sign", choices=["neg", "pos"], default="neg",
                    help="Oblivion rotation sign: 'neg' (default) or 'pos' (rebuilds .vmf only)")
    mg.add_argument("--yaw-offset", type=float, default=-90.0,
                    help="constant yaw correction (degrees) added to every prop "
                         "(default -90; rebuilds .vmf only)")

    p.add_argument("--version", action="version", version="%%(prog)s %s" % __version__)
    return p


def _parse_cells(text):
    parts = text.replace(" ", "").split(",")
    if len(parts) != 4:
        raise SystemExit("--cells must be minX,minY,maxX,maxY")
    x0, y0, x1, y1 = (int(v) for v in parts)
    return (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))


def main(argv=None):
    args = build_parser().parse_args(argv)

    esm = args.esm or _find_default_esm()
    if not esm:
        raise SystemExit(
            "Oblivion.esm not found. Pass --esm <path>. Typical Remastered path:\n  "
            + _DEFAULT_ESM_HINTS[0]
        )
    if not os.path.isfile(esm):
        raise SystemExit("ESM not found: %s" % esm)

    ws = _parse_ws(args.worldspace)

    # --- interiors (rooms) -------------------------------------------------
    if args.list_interiors is not None:
        return _list_interiors(args, esm)
    if args.list_models is not None:
        return _list_models(args, esm, ws)
    if args.interior:
        return _build_interior(args, esm)

    # --- list mode -------------------------------------------------------
    if args.list_cells:
        ex = _parse_load_order(_load_order(esm, args.plugin), target_ws=ws, list_only=True)
        coords = sorted(ex.cell_coords)
        if not coords:
            print("No exterior cells found in worldspace 0x%X." % ws)
            return 0
        bbox = ex.bbox()
        print("Worldspace 0x%X: %d exterior cells with terrain." % (ws, len(coords)))
        print("Cell bounding box: minX=%d minY=%d maxX=%d maxY=%d" % bbox)
        print("Tip: pick a sub-box with --cells minX,minY,maxX,maxY")
        return 0

    # --- determine bounds ------------------------------------------------
    if args.region:
        bounds = REGIONS[args.region]
        print("[note] --region %s uses an APPROXIMATE cell box %s." % (args.region, bounds))
        print("       Verify/refine against the UESP map and --list-cells.")
    elif args.cells:
        bounds = _parse_cells(args.cells)
    else:
        raise SystemExit("Choose one of --cells, --region, or --list-cells.")

    if args.list_tree_species:
        ex = _parse_load_order(_load_order(esm, args.plugin),
                               target_ws=ws, bounds=bounds, models=True)
        species = set()
        for plist in ex.placements.values():
            for p in plist:
                modl = ex.base_models.get(p["base"], "")
                if modl.lower().endswith(".spt"):
                    species.add(modl.lstrip("\\").rsplit(".", 1)[0])
        for s in sorted(species):
            print(s)
        return 0

    if args.rebuild_cache:
        return _rebuild_cache(args, ws, bounds, esm)

    if args.export_obj:
        return _export_obj(args, ws, bounds, esm)

    n_cells = (bounds[2] - bounds[0] + 1) * (bounds[3] - bounds[1] + 1)
    if n_cells > args.max_cells:
        raise SystemExit(
            "Selected box spans up to %d cells (> --max-cells %d). Narrow the box "
            "or raise --max-cells." % (n_cells, args.max_cells)
        )

    if args.models and not args.data_dir and not args.bsa:
        raise SystemExit("--models needs --data-dir <extracted Data> and/or --bsa <archive> "
                         "to read the meshes.")

    # --- extract ---------------------------------------------------------
    order = _load_order(esm, args.plugin)
    if args.plugin:
        print("Reading load order: %s ..." % ", ".join(os.path.basename(p) for p in order))
    else:
        print("Reading %s ..." % esm)
    ex = _parse_load_order(order, target_ws=ws, bounds=bounds,
                           textures=args.textures, models=args.models, water=args.water)
    if not ex.cells:
        raise SystemExit(
            "No terrain cells found in box %s of worldspace 0x%X. "
            "Run --list-cells to see what's available." % (bounds, ws)
        )
    if ex.skipped:
        print("[warn] skipped %d malformed/undecodable LAND record(s)." % ex.skipped)

    # A shared data source (BSAs / loose Data) feeds both real terrain textures
    # and the model pipeline.
    source = None
    if args.bsa or args.data_dir:
        from .bsa import DataSource
        source = DataSource(data_dir=args.data_dir, bsa_paths=args.bsa)

    texturer = (Texturer(ex.ltex, ex.cell_textures, fourway=args.fourway, source=source)
                if args.textures else None)
    if args.fourway and not args.textures:
        print("[note] --fourway has no effect with --no-textures.")

    out_parent = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_parent, exist_ok=True)

    # --- water -----------------------------------------------------------
    water_material = args.water_material
    cell_water = ex.cell_water
    if args.water and args.water_height is not None:        # force a flat sea level
        cell_water = {c: args.water_height for c in ex.cells}
    if args.water and water_material is None:               # self-contained material
        from .textures import write_water_material
        water_material = write_water_material(os.path.join(out_parent, "materials"))

    # --- models (props) --------------------------------------------------
    placements = model_map = model_scale = None
    skip_models = set()
    if args.models:
        placements, model_map, model_scale = _run_models(args, ex, out_parent, source)
        if model_map and args.max_prop_size:
            skip_models = _oversized_models(args, model_map, args.max_prop_size)
            if skip_models:
                print("  oversized props: %d model(s) skipped (>%g HU, e.g. distant-LOD "
                      "flags)" % (len(skip_models), args.max_prop_size))

    # --- 3D skybox backdrop model ----------------------------------------
    skybox_model = skybox_bounds = None
    skybox_prescaled = True       # generated models are compiled at 1/sky_scale
    if args.skybox_obj:
        skybox_model, skybox_bounds = _compile_skybox_obj(args, out_parent)
        if skybox_model:
            args.seal_sky = True
    elif args.skybox_model_file:
        skybox_model = args.skybox_model_file
        skybox_bounds = _skybox_region_bounds(args, ws, esm, bounds, ex)
        skybox_prescaled = False  # external full-scale model -> uniformscale it
        args.seal_sky = True
    elif args.skybox_model:
        skybox_model, skybox_bounds = _bake_skybox(args, ws, esm, bounds, ex, source,
                                                   out_parent)
        if skybox_model:
            args.seal_sky = True       # 3D skybox needs the map sealed to show

    # --- write -----------------------------------------------------------
    stats = write_vmf(ex.cells, args.out, scale=args.scale,
                      power=args.power, material=args.material,
                      recenter=args.recenter, texturer=texturer,
                      placements=placements, model_map=model_map,
                      angle_sign=(-1.0 if args.angle_sign == "neg" else 1.0),
                      yaw_offset=args.yaw_offset, tex_scale=args.terrain_tex_scale,
                      model_scale=model_scale, water=args.water, cell_water=cell_water,
                      water_material=(water_material or WATER_MATERIAL),
                      ws_default_water=ex.ws_default_water,
                      lightmapscale=args.terrain_lightmapscale,
                      outer_power=args.outer_power, outer_margin=args.outer_margin,
                      prop_fade=args.prop_fade, lighting=args.lighting, fog=args.fog,
                      sun_pitch=args.sun_pitch, sun_yaw=args.sun_yaw,
                      vis_floor=args.vis_floor, vis_floor_depth=args.vis_floor_depth,
                      seal_sky=args.seal_sky, fade_scale=args.fade_scale,
                      skybox_model=skybox_model, skybox_bounds=skybox_bounds,
                      sky_scale=args.sky_scale, skip_models=skip_models,
                      skybox_model_prescaled=skybox_prescaled)
    print("Wrote %s" % args.out)
    print("  cells:          %d" % stats["cells"])
    print("  recentered:     %s" % ("yes" if stats["recentered"] else "no (original coords)"))
    print("  displacements:  %d (power %d, %d per cell)"
          % (stats["displacements"], stats["power"], (32 // (1 << stats["power"])) ** 2))
    if stats["mins"]:
        mn, mx = stats["mins"], stats["maxs"]
        print("  extent (HU):    x[%d..%d] y[%d..%d] z[%d..%d]"
              % (mn[0], mx[0], mn[1], mx[1], mn[2], mx[2]))
        _warn_limits(stats)

    if args.water:
        print("  water:          %d volume(s)" % stats.get("water", 0))
    if args.lighting:
        print("  lighting:       sun + ambient%s" % (" + fog" if args.fog else ""))
    if stats.get("skybox_model"):
        print("  3D skybox:      baked terrain backdrop (%dx scale)" % args.sky_scale)
    if args.vis_floor or args.seal_sky:
        print("  vis:            %s%s"
              % (("%d under-displacement brushes" % stats.get("vis_brushes", 0))
                 if args.vis_floor else "",
                 (" + sealed skybox (%d brushes)" % stats.get("skybox", 0)) if args.seal_sky else ""))

    if args.models:
        print("  props:          %d placed (%d skipped: no converted model)"
              % (stats.get("props", 0), stats.get("props_skipped", 0)))

    if texturer is not None:
        out_dir = os.path.dirname(os.path.abspath(args.out))
        mats_root = os.path.join(out_dir, "materials")
        mstats = texturer.lib.write(mats_root)
        print("  terrain tex:    %d (%d real .dds, %d flat) | %d single, %d 2-way, %d 4-way mats"
              % (mstats["textures"], mstats["real"], mstats["fallback"],
                 mstats["single_materials"], mstats["blend_materials"], mstats["blend4_materials"]))
        print("  materials dir:  %s" % mstats["dir"])
        print("  -> copy the 'materials' folder into your GarrysMod\\garrysmod\\ "
              "(or pack it into the .bsp with bspzip) so the textures load.")
        if args.fourway:
            print("[warn] --fourway uses the Lightmapped_4WayBlend shader, which only "
                  "exists on the CS:GO branch / current (Nov-2025+) GMod tools. If your "
                  "GMod's renderer doesn't have that shader the terrain loads the WHITE "
                  "error material and the whole ground appears completely white. With "
                  "stock Source SDK 2013 vbsp the blend data is also dropped. If you see "
                  "white terrain, rebuild WITHOUT --fourway (standard WorldVertexTransition "
                  "2-way blend, renders on every Source engine).")
    return 0


def _rebuild_cache(args, ws, bounds, esm):
    """Reconstruct the model build-cache from the .mdl already in the gamedir,
    without recompiling. Re-hashes each placed NIF and records a cache entry if its
    compiled .mdl exists. Exits 0."""
    from . import model as modelmod
    if not args.bsa and not args.data_dir:
        raise SystemExit("--rebuild-cache needs --bsa/--data-dir to read NIFs.")
    gamedir = modelmod.find_gamedir(args.gamedir)
    if not gamedir:
        raise SystemExit("--rebuild-cache needs --gamedir (GMod garrysmod dir) to "
                         "find the compiled .mdl files.")
    from .bsa import DataSource
    source = DataSource(data_dir=args.data_dir, bsa_paths=args.bsa)
    order = _load_order(esm, args.plugin)
    print("Reading %s ..." % ", ".join(os.path.basename(p) for p in order))
    ex = _parse_load_order(order, target_ws=ws, bounds=bounds, models=True)
    work_dir = os.path.join(os.path.dirname(os.path.abspath(args.out)), "models_src")
    print("Rebuilding cache from compiled .mdl in %s ..." % gamedir)
    modelmod.build_models(
        ex.base_models, ex.placements, source, work_dir,
        scale=args.scale, gamedir=gamedir, compile_models=False, materials_root=None,
        flip_winding=args.flip_winding, flip_v=args.flip_v,
        collision=args.collision, collision_size=args.collision_size,
        acd_threshold=args.acd_threshold, acd_max_hulls=args.acd_max_hulls,
        ramp_axis=args.ramp_axis, model_overrides=_load_model_overrides(args.model_overrides),
        trees=False, jobs=(args.jobs or None), use_cache=True, cache_rebuild=True)
    print("Cache rebuilt -> %s" % os.path.join(work_dir, ".build_cache.json"))
    return 0


def _export_obj(args, ws, bounds, esm):
    """Export region+margin terrain + props to OBJ+MTL for Blender, then exit."""
    from .obj import export_scene_obj
    from .vmf import _offsets
    if not args.bsa and not args.data_dir:
        raise SystemExit("--export-obj needs --bsa/--data-dir to read prop meshes.")
    from .bsa import DataSource
    source = DataSource(data_dir=args.data_dir, bsa_paths=args.bsa)
    m = args.skybox_margin
    big = (bounds[0] - m, bounds[1] - m, bounds[2] + m, bounds[3] + m)
    print("Extracting region %s (+%d margin) for OBJ export ..." % (str(big), m))
    ex = _parse_load_order(_load_order(esm, args.plugin), target_ws=ws, bounds=big,
                           textures=True, models=True)
    if not ex.cells:
        raise SystemExit("No terrain in the export region.")
    playable = {c: g for c, g in ex.cells.items()
                if bounds[0] <= c[0] <= bounds[2] and bounds[1] <= c[1] <= bounds[3]}
    offs = _offsets(playable or ex.cells, args.recenter)
    out = args.export_obj
    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    min_size = args.obj_min_prop_size or (600.0 if args.obj_large_only else 0.0)
    export_scene_obj(ex.cells, ex.placements, ex.base_models, ex.ltex, ex.cell_textures,
                     source, args.scale, offs, out,
                     angle_sign=(-1.0 if args.angle_sign == "neg" else 1.0),
                     yaw_offset=args.yaw_offset, min_prop_size=min_size,
                     max_prop_size=args.max_prop_size)
    print("Wrote %s (+ .mtl). Decimate/bake it in Blender, compile the result to a "
          ".mdl, then build with --skybox-model-file <that.mdl>." % out)
    return 0


def _oversized_models(args, model_map, max_size):
    """Set of model paths whose compiled .mdl bounding box exceeds max_size (HU) —
    i.e. distant-LOD landmark meshes (XXXL flags etc.) we shouldn't place as props."""
    from . import model as modelmod
    gamedir = modelmod.find_gamedir(args.gamedir)
    if not gamedir:
        return set()
    out = set()
    for mp in set(model_map.values()):
        bb = _read_mdl_bbox(os.path.join(gamedir, mp.replace("/", os.sep)))
        if bb and max(bb[1] - bb[0], bb[3] - bb[2], bb[5] - bb[4]) > max_size:
            out.add(mp)
    return out


def _read_mdl_bbox(mdl_full_path):
    """Read a compiled .mdl's bounding box (hull_min/hull_max) from its studiohdr.
    Returns (minx,maxx,miny,maxy,minz,maxz) or None."""
    import struct
    try:
        with open(mdl_full_path, "rb") as f:
            head = f.read(132)
        if head[:4] != b"IDST":
            return None
        hmin = struct.unpack_from("<3f", head, 104)        # hull_min
        hmax = struct.unpack_from("<3f", head, 116)        # hull_max
        return (hmin[0], hmax[0], hmin[1], hmax[1], hmin[2], hmax[2])
    except Exception:
        return None


def _skybox_region_bounds(args, ws, esm, bounds, real_ex):
    """HU bounding box for sizing the skybox room, from the region+margin terrain
    extent (padded for buildings). The model can be larger than the +/-16384 cube,
    which makes its compiled .mdl bbox clamped/unreliable, so we use the region
    extent the model was exported from instead."""
    from .vmf import _offsets
    m = args.skybox_margin
    big = (bounds[0] - m, bounds[1] - m, bounds[2] + m, bounds[3] + m)
    ex = _parse_load_order(_load_order(esm, args.plugin), target_ws=ws, bounds=big)
    cells = ex.cells or real_ex.cells
    from .land import CELL_SIZE
    x_off, y_off, z_off = _offsets(real_ex.cells, args.recenter)
    s = args.scale
    cxs = [c[0] for c in cells]
    cys = [c[1] for c in cells]
    zs = [v for g in cells.values() for row in g for v in row]
    pad = 3000.0    # HU headroom so tall buildings on the terrain don't poke out
    return ((min(cxs) * CELL_SIZE - x_off) * s, ((max(cxs) + 1) * CELL_SIZE - x_off) * s,
            (min(cys) * CELL_SIZE - y_off) * s, ((max(cys) + 1) * CELL_SIZE - y_off) * s,
            (min(zs) - z_off) * s, (max(zs) - z_off) * s + pad)


def _compile_skybox_obj(args, out_parent):
    """Compile an OBJ to a model and return (model_path, HU bounds) for the skybox."""
    from . import model as modelmod
    from .skybox import generate_skybox_from_obj
    if not os.path.isfile(args.skybox_obj):
        raise SystemExit("--skybox-obj file not found: %s" % args.skybox_obj)
    studiomdl = modelmod.find_studiomdl(args.studiomdl)
    gamedir = modelmod.find_gamedir(args.gamedir)
    compile_models = not args.skip_compile and bool(studiomdl and gamedir)
    if not compile_models:
        print("[warn] studiomdl/gamedir not found; skybox OBJ won't compile.")
    work_dir = os.path.join(out_parent, "models_src")
    os.makedirs(work_dir, exist_ok=True)
    print("Compiling skybox OBJ %s ..." % args.skybox_obj)
    return generate_skybox_from_obj(args.skybox_obj, work_dir,
                                    os.path.join(out_parent, "materials"),
                                    studiomdl, gamedir, compile_models=compile_models,
                                    sky_scale=args.sky_scale)


def _bake_skybox(args, ws, esm, bounds, real_ex, source, out_parent):
    """Extract the region + margin, bake it to one low-poly terrain model with a
    baked texture, and compile it. Returns (model_path, bounds_hu) or (None, None)."""
    from . import model as modelmod
    from .skybox import generate_skybox_terrain
    from .vmf import _offsets
    if source is None:
        print("[warn] --skybox-model needs --bsa/--data-dir; skipping.")
        return None, None
    m = args.skybox_margin
    big = (bounds[0] - m, bounds[1] - m, bounds[2] + m, bounds[3] + m)
    print("Baking skybox terrain over cells %s (+%d margin) ..." % (str(big), m))
    sky_ex = _parse_load_order(_load_order(esm, args.plugin), target_ws=ws,
                               bounds=big, textures=True, models=False)
    if not sky_ex.cells:
        print("[warn] no terrain for the skybox region; skipping.")
        return None, None
    studiomdl = modelmod.find_studiomdl(args.studiomdl)
    gamedir = modelmod.find_gamedir(args.gamedir)
    compile_models = not args.skip_compile and bool(studiomdl and gamedir)
    offs = _offsets(real_ex.cells, args.recenter)        # align to the real terrain
    work_dir = os.path.join(out_parent, "models_src")
    os.makedirs(work_dir, exist_ok=True)
    return generate_skybox_terrain(
        sky_ex.cells, sky_ex.ltex, sky_ex.cell_textures, args.scale, offs, work_dir,
        os.path.join(out_parent, "materials"), studiomdl, gamedir,
        compile_models=compile_models, sky_scale=args.sky_scale)


def _parse_interiors(order):
    """Parse a whole load order into one InteriorExtractor (later plugins
    override/add, same FormID-prefix rules as the terrain path)."""
    from .esm import InteriorExtractor
    ex = InteriorExtractor()
    for path in order:
        ex.parse_file(path)
    ex.finalize()
    return ex


def _list_interiors(args, esm):
    order = _load_order(esm, args.plugin)
    print("Reading load order: %s ..." % ", ".join(os.path.basename(p) for p in order))
    ex = _parse_interiors(order)
    filt = (args.list_interiors or "").lower()
    rows = []
    for fid, info in ex.interiors.items():
        if filt and filt not in info["edid"].lower() and filt not in info["full"].lower():
            continue
        rows.append((info["edid"] or "(no edid)", info["full"],
                     len(ex.placements.get(fid, [])), fid))
    rows.sort()
    for edid, full, nrefs, fid in rows:
        print("  %-42s %-34s %5d refs  (0x%08X)" % (edid, full[:34], nrefs, fid))
    print("%d interior cells%s." % (len(rows), " match '%s'" % filt if filt else ""))
    print("Build one with: --interior <EDID> --out build/room.vmf")
    return 0


def _selection_models(args, esm, ws):
    """(placements, base_models) for the current selection — interior or exterior.
    Shared by --list-models and the model editor."""
    order = _load_order(esm, args.plugin)
    if args.interior:
        ex = _parse_interiors(order)
        fid, _info = ex.find(args.interior)
        if fid is None:
            raise SystemExit("Interior '%s' not found (try --list-interiors)." % args.interior)
        return {(0, 0): ex.placements.get(fid, [])}, ex.base_models
    if args.region:
        bounds = REGIONS[args.region]
    elif args.cells:
        bounds = _parse_cells(args.cells)
    else:
        raise SystemExit("--list-models needs --cells/--region or --interior.")
    ex = _parse_load_order(order, target_ws=ws, bounds=bounds, models=True)
    return ex.placements, ex.base_models


def _list_models(args, esm, ws):
    placements, base_models = _selection_models(args, esm, ws)
    counts = {}
    for plist in placements.values():
        for p in plist:
            modl = base_models.get(p["base"])
            if modl:
                counts[modl] = counts.get(modl, 0) + 1
    filt = (args.list_models or "").lower()
    n = 0
    for modl, c in sorted(counts.items()):
        if filt and filt not in modl.lower():
            continue
        kind = "tree" if modl.lower().endswith(".spt") else "mesh"
        print("  %-62s %5d  %s" % (modl, c, kind))
        n += 1
    print("%d models%s. Edit per-model settings in the GUI 'Model edits' tab "
          "(saved to a --model-overrides JSON)." % (n, " match '%s'" % filt if filt else ""))
    return 0


def _load_model_overrides(path):
    """Read a --model-overrides JSON into {nif_path_lower: settings}. Tolerant of a
    missing file (returns {}). Keys are lowercased so lookups are case-insensitive."""
    if not path:
        return None
    if not os.path.isfile(path):
        print("[warn] --model-overrides file not found: %s (ignoring)" % path)
        return None
    import json
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    return {str(k).lower(): v for k, v in raw.items()}


class _InteriorShim:
    """Quacks like TerrainExtractor for _run_models: one cell of placements."""
    def __init__(self, refs, base_models):
        self.placements = {(0, 0): refs}
        self.base_models = base_models


def _build_interior(args, esm):
    """Convert one interior cell to a sealed-room VMF using the prop pipeline."""
    from .vmf import write_interior_vmf
    if args.models and not args.data_dir and not args.bsa:
        raise SystemExit("--interior needs --bsa/--data-dir to read the room meshes.")
    order = _load_order(esm, args.plugin)
    print("Reading load order: %s ..." % ", ".join(os.path.basename(p) for p in order))
    ex = _parse_interiors(order)
    fid, info = ex.find(args.interior)
    if fid is None:
        raise SystemExit("Interior '%s' not found (or ambiguous). Try --list-interiors %s"
                         % (args.interior, args.interior))
    refs = ex.placements.get(fid, [])
    print("Interior %s (%s): %d refs  (0x%08X)"
          % (info["edid"], info["full"] or "-", len(refs), fid))
    if not refs:
        raise SystemExit("That interior has no object references.")

    source = None
    if args.bsa or args.data_dir:
        from .bsa import DataSource
        source = DataSource(data_dir=args.data_dir, bsa_paths=args.bsa)

    out_parent = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_parent, exist_ok=True)

    # Interiors are CLOSED shells you stand INSIDE, so the usual modes seal you out:
    # 'full' convex-hulls the connected room into one solid block, and 'acd'/CoACD
    # decomposes the *enclosed volume* of a closed mesh -> also solid. 'havok'
    # extrudes each wall surface into a THIN slab, leaving the interior empty
    # (walk-in by construction) -> the right default for rooms. Explicit modes win.
    if args.collision == "auto":
        args.collision = "havok"
        print("[note] interior: using --collision havok (each wall -> a thin solid "
              "slab so you can stand inside the room). 'full'/'acd' would seal a "
              "closed room solid; pass --collision none for walk-through.")
    elif args.collision in ("acd", "full"):
        print("[warn] --collision %s on a CLOSED interior room usually SEALS it "
              "(you can't enter): 'full' convex-hulls the room solid; 'acd' fills "
              "the enclosed volume. Use --collision havok for walk-in rooms."
              % args.collision)
    elif args.collision == "custom":
        print("[note] interior: using the overrides' per-model collision; models "
              "WITHOUT an override entry fall back to havok (walk-in).")

    shim = _InteriorShim(refs, ex.base_models)
    placements = model_map = model_scale = None
    skip_models = set()
    if args.models:
        placements, model_map, model_scale = _run_models(args, shim, out_parent, source)
        if model_map and args.max_prop_size:
            skip_models = _oversized_models(args, model_map, args.max_prop_size)
    else:
        placements, model_map = shim.placements, {}

    stats = write_interior_vmf(args.out, placements, model_map, scale=args.scale,
                               light_bases=ex.lights, ambient=info.get("ambient"),
                               angle_sign=(-1.0 if args.angle_sign == "neg" else 1.0),
                               yaw_offset=args.yaw_offset, model_scale=model_scale,
                               skip_models=skip_models, skybox_room=args.skybox_room)
    b = stats["bounds"]
    print("  props:   %d placed (%d skipped)" % (stats["props"], stats["props_skipped"]))
    print("  lights:  %d" % stats["lights"])
    print("  room:    %.0f x %.0f x %.0f HU"
          % (b[1] - b[0], b[3] - b[2], b[5] - b[4]))

    # The .mdl props compiled straight into the gamedir, but their .vmt/.vtf
    # materials were written next to the .vmf. Without these in GarrysMod the
    # props render as a pink/black checkerboard. Auto-copy them into the gamedir
    # (the user's own local assets) and always print the reminder as a fallback.
    mats_dir = os.path.join(out_parent, "materials")
    if args.models and os.path.isdir(mats_dir):
        copied = False
        if not args.skip_compile:
            from . import model as modelmod
            gamedir = modelmod.find_gamedir(args.gamedir)
            if gamedir:
                n = _copy_tree_into(mats_dir, os.path.join(gamedir, "materials"))
                print("  materials:      %d file(s) copied -> %s\\materials\\"
                      % (n, gamedir))
                copied = True
        if not copied:
            print("  materials dir:  %s" % mats_dir)
            print("  -> copy the 'materials' folder into your GarrysMod\\garrysmod\\ "
                  "so the prop textures load (else they show pink/black checkerboard).")

    print("Wrote %s%s" % (args.out, "  (skybox room)" if args.skybox_room else ""))

    if args.instance_into:
        from .vmf import add_instance_to_vmf
        origin = None
        if args.instance_origin:
            try:
                origin = tuple(float(v) for v in args.instance_origin.split(","))
                if len(origin) != 3:
                    raise ValueError
            except ValueError:
                raise SystemExit("--instance-origin must be X,Y,Z (3 numbers)")
        o, rel = add_instance_to_vmf(args.instance_into, args.out, origin=origin,
                                     targetname=info["edid"])
        print("  instance:       added func_instance \"%s\" at (%.0f %.0f %.0f) in %s"
              % (rel, o[0], o[1], o[2], args.instance_into))
        print("Compile the HOST map: ./scripts/compile_map.ps1 -Map %s" % args.instance_into)
    else:
        print("Compile: ./scripts/compile_map.ps1 -Map %s" % args.out)
    return 0


def _copy_tree_into(src, dst):
    """Recursively copy ``src``'s contents into ``dst`` (merging into any existing
    tree, overwriting files). Returns the number of files copied."""
    import shutil
    n = 0
    for root, _dirs, files in os.walk(src):
        rel = os.path.relpath(root, src)
        target = dst if rel == "." else os.path.join(dst, rel)
        os.makedirs(target, exist_ok=True)
        for fn in files:
            shutil.copy2(os.path.join(root, fn), os.path.join(target, fn))
            n += 1
    return n


def _run_models(args, ex, out_parent, source):
    """Convert placed NIF meshes to .mdl props. Returns (placements, model_map)."""
    from . import model as modelmod

    if not ex.placements:
        print("[note] no object placements (REFR) found in this cell box.")
        return {}, {}, {}

    tree_map = None
    if args.tree_map_file:
        import json
        with open(args.tree_map_file, encoding="utf-8") as f:
            tree_map = json.load(f)

    studiomdl = gamedir = None
    compile_models = not args.skip_compile
    if compile_models:
        studiomdl = modelmod.find_studiomdl(args.studiomdl)
        gamedir = modelmod.find_gamedir(args.gamedir)
        if not studiomdl or not gamedir:
            print("[warn] studiomdl/gamedir not found (studiomdl=%s, gamedir=%s); "
                  "writing .smd/.qc only. Pass --studiomdl/--gamedir, or compile the "
                  "generated .qc yourself." % (studiomdl, gamedir))
            compile_models = False

    work_dir = os.path.join(out_parent, "models_src")
    model_map, mstats = modelmod.build_models(
        ex.base_models, ex.placements, source, work_dir,
        scale=args.scale, studiomdl=studiomdl, gamedir=gamedir,
        compile_models=compile_models,
        materials_root=os.path.join(out_parent, "materials"),
        flip_winding=args.flip_winding, flip_v=args.flip_v,
        collision=args.collision, collision_size=args.collision_size,
        acd_threshold=args.acd_threshold, acd_max_hulls=args.acd_max_hulls,
        acd_jobs=(args.acd_jobs or None), ramp_axis=args.ramp_axis,
        model_overrides=_load_model_overrides(args.model_overrides),
        custom_fallback=("havok" if args.interior else "auto"),
        model_lods=args.model_lods,
        trees=args.trees,
        tree_model=args.tree_model, tree_scale=args.tree_scale, tree_map=tree_map,
        jobs=(args.jobs or None), use_cache=args.cache)

    print("  meshes:         %d unique, %d converted (%d reused from cache), %d failed, %d tree species"
          % (mstats["unique_meshes"], mstats["converted"], mstats.get("cached", 0),
             mstats["failed"], mstats["trees"]))
    if mstats.get("skipped"):
        print("  model edits:    %d mesh(es) skipped via --model-overrides" % mstats["skipped"])
    print("  model textures: %d converted, %d fell back to placeholder"
          % (mstats["textures"], mstats["textures_failed"]))
    print("  model sources:  %s" % work_dir)
    if compile_models and gamedir:
        print("  compiled .mdl -> %s\\models\\oblivion2vmf\\" % gamedir)
    else:
        print("  -> compile the .qc files in models_src with studiomdl to produce the .mdl props.")
    for modl, reason in mstats["failures"][:8]:
        print("    - skip %s: %s" % (modl, reason))
    if len(mstats["failures"]) > 8:
        print("    ... and %d more" % (len(mstats["failures"]) - 8))
    return ex.placements, model_map, mstats.get("model_scale", {})


def _warn_limits(stats):
    if stats["displacements"] > MAX_MAP_DISPINFO:
        print("[warn] %d displacements exceeds MAX_MAP_DISPINFO (%d). "
              "Reduce the cell box or lower --power; the map will not compile."
              % (stats["displacements"], MAX_MAP_DISPINFO))
    mn, mx = stats["mins"], stats["maxs"]
    worst = max(abs(mn[0]), abs(mx[0]), abs(mn[1]), abs(mx[1]), abs(mn[2]), abs(mx[2]))
    if worst > MAX_COORD:
        suggested = (MAX_COORD / worst) * stats["scale"] * 0.95
        print("[warn] geometry reaches +/-%d HU, beyond the +/-%d Source world "
              "boundary. Re-run with --scale ~%.3f or fewer cells (tiling needed "
              "for the full region)." % (int(worst), MAX_COORD, suggested))


if __name__ == "__main__":
    sys.exit(main())
