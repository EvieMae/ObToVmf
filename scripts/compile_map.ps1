<#
.SYNOPSIS
  Copy generated materials/models into the GMod gamedir and compile a .vmf to .bsp.

.EXAMPLE
  ./scripts/compile_map.ps1 -Map build/test.vmf                 # prop-combine ON (default)
  ./scripts/compile_map.ps1 -Map build/test.vmf -Fast           # quick vvis/vrad
  ./scripts/compile_map.ps1 -Map build/test.vmf -NoPropCombine  # disable prop-combine

.NOTES
  Static prop combine (-StaticPropCombine) is ON by default — it merges nearby
  same-model props into batched meshes (big FPS win, fewer draw calls). GMod's vbsp
  supports it. If it errors on your tools, pass -NoPropCombine.
#>
param(
    [string]$Map = "build/test.vmf",
    [string]$Bin = "C:\Program Files (x86)\Steam\steamapps\common\GarrysMod\bin",
    [string]$Game = "C:\Program Files (x86)\Steam\steamapps\common\GarrysMod\garrysmod",
    [switch]$Fast,
    [switch]$NoPropCombine,
    [switch]$SkipMaterialCopy
)

$ErrorActionPreference = "Stop"
$proj = Split-Path -Parent $PSScriptRoot
$mapPath = if ([System.IO.Path]::IsPathRooted($Map)) { $Map } else { Join-Path $proj $Map }
if (-not (Test-Path $mapPath)) { throw "VMF not found: $mapPath" }

# 1. copy generated materials + models into the gamedir so vbsp/the engine find them
if (-not $SkipMaterialCopy) {
    $buildDir = Split-Path -Parent $mapPath
    $mats = Join-Path $buildDir "materials"
    if (Test-Path $mats) {
        Write-Host "Copying materials -> $Game\materials"
        Copy-Item "$mats\*" (Join-Path $Game "materials") -Recurse -Force
    }
}

# 2. compile
$vbsp = Join-Path $Bin "vbsp.exe"
$vvis = Join-Path $Bin "vvis.exe"
$vrad = Join-Path $Bin "vrad.exe"

# Static prop combine merges nearby same-model props into batched meshes -> far
# fewer draw calls (big FPS win on prop-heavy maps). On by default; -NoPropCombine
# to skip. MinInstances 2 = combine groups of 2+.
$vbspArgs = @("-game", $Game)
if (-not $NoPropCombine) { $vbspArgs += @("-StaticPropCombine", "-StaticPropCombine_MinInstances", "2") }
$vbspArgs += $mapPath
Write-Host "vbsp $($vbspArgs -join ' ')"
& $vbsp @vbspArgs
if ($LASTEXITCODE -ne 0) { throw "vbsp failed (exit $LASTEXITCODE) - no .bsp written. Loading GMod now would show a STALE/white map. Fix the error above and re-run." }

$visArgs = @("-game", $Game)
if ($Fast) { $visArgs += "-fast" }
$visArgs += $mapPath
Write-Host "vvis $($visArgs -join ' ')"
& $vvis @visArgs
if ($LASTEXITCODE -ne 0) { throw "vvis failed (exit $LASTEXITCODE)." }

$radArgs = @("-game", $Game, "-StaticPropLighting", "-StaticPropPolys", "-textureshadows")
if ($Fast) { $radArgs += "-fast" }
$radArgs += $mapPath
Write-Host "vrad $($radArgs -join ' ')"
& $vrad @radArgs
if ($LASTEXITCODE -ne 0) { throw "vrad failed (exit $LASTEXITCODE) - map has no lighting (renders fullbright/white)." }

# 3. drop the bsp into maps/
$bsp = [System.IO.Path]::ChangeExtension($mapPath, ".bsp")
$mapsDir = Join-Path $Game "maps"
Copy-Item $bsp $mapsDir -Force
Write-Host "Done -> $(Join-Path $mapsDir (Split-Path -Leaf $bsp))"
