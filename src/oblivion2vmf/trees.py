"""Resolve real Oblivion foliage textures for tree (.spt) placements.

Oblivion trees are SpeedTree, which we can't render, but their leaf and bark
textures ship in the Textures BSA under textures\\trees\\leaves\\ and
textures\\trees\\branches\\. The shipped names differ from the .spt's internal
refs, so we catalogue them and match by the .spt's *name* (e.g.
"TreeEnglishOakForest01SU" -> textures\\trees\\leaves\\treeenglishoakleavessu.dds).
"""
from __future__ import annotations

import re


def _norm(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())


class TreeTextures:
    """Catalogue of leaf/bark textures from a DataSource's BSAs, with name match."""

    def __init__(self, source):
        self.leaf = {}   # core -> {season: relpath}
        self.bark = {}   # core -> relpath
        for b in getattr(source, "bsas", []):
            try:
                files = b.list_files()
            except Exception:
                files = []
            for f in files:
                low = f.lower()
                leaf = f.split("\\")[-1].rsplit(".", 1)[0].lower()
                if "trees\\leaves" in low and "leaves" in leaf:
                    core, _, season = leaf.partition("leaves")
                    if core:
                        self.leaf.setdefault(core, {})[season] = f
                elif "trees\\branches" in low and "bark" in leaf and not leaf.endswith("_n"):
                    core = leaf.partition("bark")[0]
                    # prefer the plain '<core>bark' over moss/variant spellings
                    if core and (core not in self.bark or leaf == core + "bark"):
                        self.bark[core] = f

    def _best_core(self, cat, n):
        best = None
        for core in cat:
            if core and n.startswith(core) and (best is None or len(core) > len(best)):
                best = core
        return best

    def match(self, spt_name):
        """Return (leaf_relpath_or_None, bark_relpath_or_None) for a .spt name."""
        n = _norm(spt_name)
        want = "su" if n.endswith("su") else "fa" if n.endswith("fa") else ""
        leaf = None
        core = self._best_core(self.leaf, n)
        if core:
            s = self.leaf[core]
            leaf = s.get(want) or s.get("su") or s.get("") or next(iter(s.values()))
        bcore = self._best_core(self.bark, n)
        bark = self.bark.get(bcore) if bcore else None
        return leaf, bark
