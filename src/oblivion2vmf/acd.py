"""Approximate convex decomposition for accurate prop collision (optional).

studiomdl can only build collision from convex pieces, and its own decomposition
just convex-hulls each connected component — so a hollow building becomes a solid
block. CoACD splits a concave mesh into many convex parts; we then nudge each part
slightly toward its own centroid so studiomdl doesn't weld them back together,
yielding real walk-in collision (blocked by walls, open doorways).

Optional: install with `pip install coacd`. If absent, callers fall back.
"""
from __future__ import annotations

try:
    import coacd
    import numpy as np
    coacd.set_log_level("error")
    HAVE_COACD = True
except Exception:                       # pragma: no cover - depends on optional dep
    HAVE_COACD = False


def available():
    return HAVE_COACD


def decompose(submeshes, threshold=0.08, max_hulls=-1, shrink=0.03):
    """Decompose the merged submeshes into convex parts (model-local coords).

    Returns a list of (verts, faces). ``shrink`` insets each part toward its
    centroid (fraction) so adjacent parts don't weld in studiomdl.

    ``preprocess_mode="off"`` is essential for buildings: Oblivion meshes are open
    shells, and CoACD's default voxel preprocess would FILL the interior, producing
    collision that walls you out. "off" decomposes the actual surface into thin,
    wall-shaped convex pieces you can walk between.
    """
    if not HAVE_COACD:
        raise RuntimeError("coacd not installed (pip install coacd)")
    verts, faces = [], []
    for s in submeshes:
        base = len(verts)
        verts.extend(s["verts"])
        for a, b, c in s["tris"]:
            faces.append((base + a, base + b, base + c))
    if not faces:
        return []

    mesh = coacd.Mesh(np.asarray(verts, dtype=np.float64),
                      np.asarray(faces, dtype=np.int32))
    parts = coacd.run_coacd(
        mesh, threshold=threshold, max_convex_hull=max_hulls,
        preprocess_mode="off",
        mcts_iterations=20, mcts_max_depth=2, mcts_nodes=10, merge=True)

    out = []
    for pv, pf in parts:
        v = np.asarray(pv, dtype=np.float64)
        c = v.mean(axis=0)
        v = c + (v - c) * (1.0 - shrink)
        out.append((v.tolist(), [tuple(int(i) for i in f) for f in pf]))
    return out


def decompose_isolated(submeshes, timeout=180, **kwargs):
    """Run :func:`decompose` in a separate process so a native CoACD crash on a
    degenerate mesh can't take down the build. Returns the parts list, or None if
    the worker crashed/timed out (caller should then fall back to non-solid)."""
    import os
    import pickle
    import subprocess
    import sys
    import tempfile

    pkg_parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = dict(os.environ)
    env["PYTHONPATH"] = pkg_parent + os.pathsep + env.get("PYTHONPATH", "")
    d = tempfile.mkdtemp(prefix="acd_")
    in_path = os.path.join(d, "in.pkl")
    out_path = os.path.join(d, "out.pkl")
    try:
        with open(in_path, "wb") as f:
            pickle.dump((submeshes, kwargs), f)
        try:
            p = subprocess.run([sys.executable, "-m", "oblivion2vmf.acd_worker",
                                in_path, out_path],
                               capture_output=True, timeout=timeout, env=env)
        except Exception:
            return None
        if p.returncode != 0 or not os.path.exists(out_path):
            return None
        with open(out_path, "rb") as f:
            return pickle.load(f)
    finally:
        for fp in (in_path, out_path):
            try:
                os.remove(fp)
            except OSError:
                pass
        try:
            os.rmdir(d)
        except OSError:
            pass
