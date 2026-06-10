# oblivion2vmf

Convert **Oblivion 2006 (TES4) terrain** into a **Source / Hammer `.vmf`** displacement
map for **Garry's Mod**.

It reads landscape heightmaps **and land textures** straight out of
`Oblivion.esm` and emits textured displacement brushes:

```
Oblivion.esm ─┬─ per-cell LAND/VHGT heightmap -> displacement brushes ->
              └─ per-quadrant BTXT/ATXT textures -> blend materials -> terrain.vmf + materials/
```



## Requirements

- Python 3.9+ (standard library only - no pip dependencies for the converter).
- A copy of Oblivion you own. `Oblivion.esm` lives at:
  - **Remastered (2025):** `…\Oblivion Remastered\OblivionRemastered\Content\Dev\ObvData\Data\Oblivion.esm`
  - **Original (2006):** `…\steamapps\common\Oblivion\Data\Oblivion.esm`
  - (The Remaster ships the original data, so both work identically here.)

## GUI

A  GUI wraps everything - pick options, manage BSAs, map tree species to
models, and run with a live log; it **remembers your last choices** (saved to
`~/.oblivion2vmf_gui.json`):

```powershell
python -m oblivion2vmf.gui        # or: oblivion2vmf-gui  (after pip install -e .)
```



## Usage (CLI)

No install needed - run it as a module from the repo root:

```powershell
# 1. See what cells exist (find the West Weald cells on your own ESM)
python -m oblivion2vmf --esm "<path>\Oblivion.esm" --list-cells

# 2. Convert a small test box first (a 2x2-cell patch)
#    NOTE: use --cells=... (with the equals sign) so negative coords aren't
#    mistaken for flags.
python -m oblivion2vmf --esm "<path>\Oblivion.esm" --cells=-19,-2,-18,-1 --out test.vmf

# 3. Approximate West Weald box (verify/refine against the UESP map!)
python -m oblivion2vmf --esm "<path>\Oblivion.esm" --region west-weald --out westweald.vmf
```

The default scale is **0.5625** (player-accurate: a 6-ft Oblivion NPC becomes a
6-ft GMod player). Pass `--scale 1.0` for raw 1:1 (everything ~1.8× bigger).

By default the selection is **recentered on the origin** so it fits the Source
map cube no matter where the region sits in Cyrodiil (West Weald is ~cell −19, so
without recentering it would land ~44k units off-origin and outside the map).
Pass `--no-recenter` to keep original Oblivion world coordinates.

Or install the console script:

```powershell
pip install -e .
oblivion2vmf --esm "<path>\Oblivion.esm" --list-cells
```

### Options

| Flag | Default | Meaning |
|---|---|---|
| `--esm PATH` | autodetect Steam | path to `Oblivion.esm` |
| `--list-cells` | - | print exterior cells + bounding box, then exit |
| `--cells=minX,minY,maxX,maxY` | - | inclusive cell box (use `=` for negative coords) |
| `--region west-weald` | - | approximate named-region box (verify it!) |
| `--worldspace` | `tamriel` (`0x3C`) | worldspace name or FormID |
| `--scale` | `0.5625` | Oblivion-unit → Hammer-unit factor. `0.5625` = player-accurate real-world (default; 6-ft NPC = 6-ft player); `1.0` = raw 1:1 (~10-ft players) |
| `--power` | `4` | displacement power: 4 → 4 disps/cell, 3 → 16, 2 → 64 |
| `--no-textures` | (textures on) | skip land textures; use a flat `--material` instead |
| `--fourway` | (2-way) | use `Lightmapped_4WayBlend` (4 textures/disp). **Needs a CS:GO `vbsp` or current GMod tools** - see Textures |
| `--material` | `dev/dev_measuregeneric01b` | flat fallback material when `--no-textures` |
| `--no-recenter` | (recenter on) | keep original Oblivion world coordinates |
| `--out` | `terrain.vmf` | output path (parent dirs are created) |
| `--max-cells` | `512` | safety guard against huge selections |
| `--models` | (off) | extract `REFR` placements → NIF→`.mdl` `prop_static` (see Models) |
| `--plugin` | - | extra mod `.esm`/`.esp` to load on top, in load order (repeatable) |
| `--bsa` / `--data-dir` | - | where to read `.nif` meshes from (archive / loose, both repeatable) |
| `--studiomdl` / `--gamedir` | autodetect | model compiler + GMod game dir |
| `--skip-compile` | (compile) | emit `.smd`/`.qc` only |
| `--collision` | `auto` | `auto`/`acd`/`full`/`none` (see Models → Collision) |
| `--collision-size` | `400` | HU width above which a prop counts as a "building" |
| `--jobs` / `-j` | `0` (auto) | parallel worker threads for model compile (0 = min(8, CPU); 1 = serial) |
| `--no-cache` | (cache on) | don't reuse cached compiled models (force rebuild all) |
| `--rebuild-cache` | - | rebuild the cache from already-compiled `.mdl` (no recompile), then exit |
| `--no-water` | (water on) | don't add water volumes |
| `--water-material` | generated | water face material; pass e.g. `nature/water_canals_water2` for an existing one |
| `--water-height` | per-cell | force one flat Oblivion water height for all cells (overrides `XCLW`) |
| `--no-lighting` | (lighting on) | don't add sun/ambient/shadows (map stays fullbright) |
| `--no-fog` | (fog on) | don't add the outdoor `env_fog_controller` |
| `--sun-pitch` / `--sun-yaw` | `-45` / `200` | sun elevation / compass direction |
| `--no-trees` | (trees on) | don't place generated trees for SpeedTree placements |
| `--terrain-tex-scale` | `0.25` | displacement texture UV scale (lower = finer) |
| `--no-prop-fade` | (fade on) | don't distance-fade props (see Optimization) |
| `--fade-scale` | `1.0` | multiply all prop fade distances (lower = cull closer = more FPS) |
| `--terrain-lightmapscale` | `32` | luxel size for terrain/water faces (higher = faster/coarser) |
| `--outer-power` | = `--power` | lower displacement power for edge cells (coarse terrain LOD) |
| `--outer-margin` | `1` | depth (cells) of the `--outer-power` border ring |
| `--vis-floor` | (off) | extend displacements down to a solid floor (vvis occlusion) |
| `--seal-sky` | (off) | wrap the map in a toolsskybox box so vvis isn't leaking |
| `--skybox-model` | (off) | bake the region+margin into one low-poly model used as a 3D-skybox backdrop |
| `--skybox-margin` / `--sky-scale` | `6` / `16` | backdrop margin (cells) / 3D-skybox scale |
| `--skybox-model-file` | - | use a pre-built `.mdl` (e.g. from Blender) as the 3D-skybox backdrop |
| `--export-obj PATH` | - | export region+margin terrain+props to OBJ+MTL for Blender, then exit |
| `--acd-max-hulls` | `-1` | cap convex collision hulls per prop (`-1` = no cap) |
| `--acd-jobs` | `0` (auto) | max concurrent CoACD subprocesses (raise on many-core, e.g. 24) |
| `--no-model-lods` | (LODs on) | don't generate decimated `$lod` stages for props |

## Water (`--no-water` to disable)

On by default. Oblivion stores water per exterior cell (the `XCLW` height + the
"has water" flag, with a worldspace default from `WRLD/DNAM`). For each watered
cell the tool emits one **swimmable water volume** - a brush spanning the cell
footprint, running from just below the cell's lowest terrain up to the water
surface, with a `%compilewater` material on top and `nodraw` on the other faces.
Adjacent cells of the same body share an exact surface height, so the planes line
up across cell borders. A `water_lod_control` entity is added automatically.

The water material is **self-contained** (generated into
`materials/oblivion2vmf/water*`: a cheap `Water` shader + flat normal + bottom
material - no cubemap/RT setup needed). Use `--water-material nature/water_canals_water2`
(or any GMod/CSS water) if you prefer a fancier reflective one, or `--water-height H`
to force a single flat sea level across the whole selection.


## Mods / load order (`--plugin`)

Load mod plugins on top of `Oblivion.esm` with `--plugin` (repeatable, **in load
order** - masters first). Later plugins **override, add, and delete** records by
FormID, just like the game: a mod that opens the cities (e.g. **Open Cities**) adds
its building placements to the Tamriel worldspace and removes the closed-city
shell, and both show up in your box. The tool reads each plugin's master list and
warns if you pass them out of order (which would misalign FormIDs).

Mods usually ship **loose** `meshes/`/`textures/` - point `--data-dir` at the mod
folder (repeatable, searched before the BSAs) so those models/textures resolve.

```powershell
python -m oblivion2vmf --esm Oblivion.esm `
  --plugin "...\Open Cities Classic\00 Core\Open Cities Resources.esm" `
  --plugin "...\Open Cities Classic\00 Core\Open Cities Classic.esp" `
  --data-dir "...\Open Cities Classic\00 Core" `
  --bsa "Oblivion - Meshes.bsa" --bsa "Oblivion - Textures - Compressed.bsa" `
  --cells=-19,-6,-9,2 --out build/test.vmf --fourway --models --collision acd `
  --tree-map-file treemap.json
```

## Lighting (`--no-lighting` to disable)

On by default. Without it the map is **fullbright**. The tool adds:
- a **`light_environment`** - the sun + sky ambient that lights the 2D sky
  (warm sun ~45° up, cool sky fill). One per map, which is what `vrad` needs.
- a **`shadow_control`** matching the sun direction (dynamic prop shadows).
- an **`env_fog_controller`** for outdoor haze (`--no-fog` to skip), tuned so the
  far cutoff sits past where props fade.

Tune with `--sun-pitch` (elevation below horizontal) and `--sun-yaw` (compass
direction). Lighting is baked by `vrad`, so a re-run that only changes these needs
the map recompiled, but **not** the models (use the cache). Colors/brightness live
in `DEFAULT_SUN_LIGHT` / `DEFAULT_AMBIENT` in `vmf.py`.

Water volumes for adjacent cells at the same height are **merged into single
brushes** (greedy rectangles), so a lake is one brush with no per-cell seams.

## Scale & engine limits (important)

Source maps are hard-capped to a **±16384-unit cube**. One Oblivion cell is 4096
units. So:

- At `--scale 1.0`, only ~**4 cells/axis** fit before you hit the wall.
- At `--scale 0.5625` (real-world), ~**14 cells/axis** fit.
- The full West Weald region is bigger than that → it must be **tiled** into
  multiple maps (a later milestone). The CLI prints a warning with a suggested
  scale when your selection overflows.

`MAX_MAP_DISPINFO` is 2048, so at `--power 4` you can fit ≤ 512 cells per map.

## Textures

By default the converter reads each cell's land textures and writes a
**self-contained** material set next to the `.vmf`, under `materials/oblivion2vmf/`:

- **With a Textures BSA** (pass it via `--bsa`), each cell's real Oblivion land
  texture (`LTEX`) is transcoded `.dds`→`.vtf` and used directly - the terrain
  looks like actual Cyrodiil ground.
- **Without one**, each land texture is classified into a *ground type* (grass,
  dirt, rock, …) and painted as a generated flat-tinted `.vtf` (legal, no ripped
  assets, renders on any GMod client).
- Where a cell quadrant has an overlay layer, the displacement uses a
  `WorldVertexTransition` 2-way blend driven by per-vertex `alphas` taken from
  Oblivion's VTXT opacity data (`$basetexture` = base at alpha 0, `$basetexture2`
  = overlay at alpha 255). Extra overlay layers beyond the dominant one are
  dropped (Source displacements blend two textures at a time).

Use `--no-textures` to skip all of this and get a flat dev-textured map instead.

## Models / props (`--models`)

`--models` extracts every object placement (`REFR`) in the selected cells,
resolves each to its `.nif` mesh (`STAT`/`TREE`/`FLOR`/… → `MODL`), converts the
mesh to a Source static prop, and emits a `prop_static` at the scaled, recentered
position with the orientation converted from Oblivion's rotation.

Pipeline: `Oblivion.esm REFR → .nif (from BSA/loose) → PyFFI geometry → .smd + .qc → studiomdl → .mdl → prop_static`.

```powershell
pip install -e ".[models]"   # installs PyFFI (NIF parser)

oblivion2vmf --esm "<path>\Oblivion.esm" --cells=-19,-1,-18,-1 --out build/test.vmf ^
    --models --bsa "<path>\Oblivion - Meshes.bsa" ^
              --bsa "<path>\Oblivion - Textures - Compressed.bsa"
```

- **Mesh + texture source:** pass `--bsa <archive>` once per BSA - you need the
  **Meshes** BSA (for `.nif`) and the **Textures** BSA (for `.dds`). Or use
  `--data-dir <Data>` for loose extracted files.
- **Compiling:** by default it runs `studiomdl` (auto-detected from GMod or
  *Source SDK Base 2013 Multiplayer*; override with `--studiomdl`/`--gamedir`),
  writing `.mdl`s into `garrysmod\models\oblivion2vmf\`. Use `--skip-compile` to
  just emit `.smd`/`.qc` (in `models_src/`) and compile them yourself.
- **Textures:** each model's real Oblivion `.dds` is transcoded to `.vtf`
  (DXT blocks copied, no re-encode) with a generated `.vmt`. Textures that can't
  be found/transcoded fall back to a flat placeholder. Copy the `materials/`
  folder into `garrysmod\` like the terrain materials.
- **Collision** (`--collision`): collision is built from the NIF's **own Havok
  shape** (the clean structural shell Bethesda authored) when present, else the
  render mesh. studiomdl can only make *convex* pieces, so a hollow building
  becomes a solid hull unless decomposed. Modes: `auto` (default) - small props
  solid, big props (> `--collision-size`, 400 HU) left non-solid so you aren't
  walled out; **`acd`** - big props are convex-**decomposed** (CoACD, `pip install
  coacd`) into thin wall pieces so you're blocked by walls but walk through
  doorways and stand on floors; `full` - everything solid; `none` - walk through
  all. Terrain displacements are always solid, so you never fall through the ground.
- **Speed (`--jobs`)**: model compile is the slow part, and each mesh
  (BSA read → NIF parse → CoACD → `studiomdl`) is independent. By default it runs
  on `min(8, CPU count)` worker threads - `studiomdl` (a subprocess) and CoACD (a
  C++ extension) both release the GIL, so the speedup is near-linear. Use `-j 1`
  to force serial (e.g. for clean, ordered logs), or a higher `-j N` to push it.
- **Caching (`--no-cache` to disable)**: each compiled model is cached
  (`<out_dir>/models_src/.build_cache.json`) keyed by a hash of its NIF bytes plus
  the build settings that affect output (scale, flip, collision mode/size/
  threshold). On a re-run a model is rebuilt **only** if its NIF or those settings
  changed and its `.mdl` is still in the gamedir - so the first build takes minutes
  but subsequent builds are near-instant. Change a collision flag and only the
  affected models rebuild. The CLI prints `N converted (M reused from cache)`.
  The cache is written **incrementally** (each model flushes as it finishes, under
  a lock), so killing a long first build mid-way is safe - the next run reuses
  whatever already compiled.
- **Recovering a lost cache (`--rebuild-cache`)**: if the cache is deleted or its
  signature no longer matches (e.g. you bumped a default), you don't have to redo
  the ~40-min compile. `--rebuild-cache` re-hashes the placed NIFs and writes a
  cache entry for every mesh whose `.mdl` is still in the gamedir - ~30s instead of
  40 min. **Pass the same model flags you build with** (e.g. `--collision acd`), so
  the rebuilt signatures match your real builds:

  ```powershell
  python -m oblivion2vmf --esm Oblivion.esm --bsa "Oblivion - Meshes.bsa" `
    --cells=-19,-6,-9,2 --out build/test.vmf --collision acd --rebuild-cache
  ```

⚠️ **Two "verify in-game" knobs** (CLI flags, defaults shown):
1. **Rotation** - `--angle-sign neg` (default); try `--angle-sign pos` if props
   face the wrong way. This only rewrites the `.vmf`, so re-run with
   `--skip-compile` to flip it fast (no model recompile).
2. **Winding / UV** - `--flip-winding` (default off) if props are inside-out;
   `--no-flip-v` if textures look upside-down. (These DO require a recompile.)

PyFFI parses real Oblivion meshes robustly, but the long tail of NIF variants
(skinned, exotic blocks) may fail to convert - those props are logged and skipped,
not fatal.



### 4-way blend (`--fourway`)

By default a displacement blends **two** textures (base + dominant overlay). Where
an Oblivion quadrant has several overlays, the 2-way limit forces hard seams
between quadrants with different bases. `--fourway` instead emits
`Lightmapped_4WayBlend` materials with up to **four** textures per displacement
(base + 3 overlays), written as CS:GO-style `multiblend` per-vertex weights in the
`dispinfo` block - far fewer seams.

⚠️ **Compile requirement.** This needs a compiler that understands 4-way blend
data: a **CS:GO-branch `vbsp`**, or **current (Nov 2025+) Garry's Mod tools**
(which added `Lightmapped_4WayBlend` "with tools support"; you also need CS:S/HL2
VPKs mounted). **Stock Source SDK 2013 `vbsp` silently drops the blend data** - the
map still compiles, but tiles show only their base texture. The CLI prints a
warning, and the format has known GMod rendering bugs (dynamic lights/flashlight
can break the blend). If 4-way comes out flat in-game, your compiler doesn't
support it - use the default 2-way.

## Compiling for Garry's Mod

1. Open the `.vmf` in **Hammer / Hammer++** (Source SDK Base 2013 Multiplayer).
2. Compile with `vbsp` / `vvis` / `vrad` from that branch (GMod needs the 2013
   static-prop lump version).
3. Copy the `.bsp` to `GarrysMod/garrysmod/maps/` (or pack with `gmad`).
4. **Copy the generated `materials/` folder into `GarrysMod/garrysmod/`** (so you
   get `garrysmod/materials/oblivion2vmf/…`), or pack it into the `.bsp` with
   `bspzip`. Without this the terrain shows the missing-texture checkerboard.

