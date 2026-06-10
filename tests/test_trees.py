from oblivion2vmf.trees import TreeTextures


class _FakeBsa:
    def __init__(self, files):
        self._files = files

    def list_files(self):
        return self._files


class _FakeSource:
    def __init__(self, files):
        self.bsas = [_FakeBsa(files)]


FILES = [
    "textures\\trees\\leaves\\treeenglishoakleavessu.dds",
    "textures\\trees\\leaves\\treeenglishoakleavesfa.dds",
    "textures\\trees\\leaves\\shrubazalealeavessu.dds",
    "textures\\trees\\branches\\treeenglishoakbark.dds",
    "textures\\trees\\branches\\treeenglishoakbark_n.dds",      # normal map, ignored
    "textures\\trees\\branches\\treeenglishoakbarkmoss.dds",    # variant, not preferred
    "textures\\landscape\\terrainhdgrass01su.dds",             # unrelated
]


def test_catalog_built():
    cat = TreeTextures(_FakeSource(FILES))
    assert "treeenglishoak" in cat.leaf
    assert cat.leaf["treeenglishoak"]["su"].endswith("treeenglishoakleavessu.dds")
    assert cat.bark["treeenglishoak"].endswith("treeenglishoakbark.dds")   # plain, not _n/moss


def test_match_oak_summer():
    cat = TreeTextures(_FakeSource(FILES))
    leaf, bark = cat.match("TreeEnglishOakForest01SU")
    assert leaf.endswith("treeenglishoakleavessu.dds")           # SU season chosen
    assert bark.endswith("treeenglishoakbark.dds")


def test_match_prefers_season():
    cat = TreeTextures(_FakeSource(FILES))
    # an FA-suffixed name should pick the fall leaf
    leaf, _ = cat.match("TreeEnglishOakForestFA")
    assert leaf.endswith("treeenglishoakleavesfa.dds")


def test_match_shrub_and_unknown():
    cat = TreeTextures(_FakeSource(FILES))
    leaf, _ = cat.match("ShrubAzaleaSU")
    assert leaf.endswith("shrubazalealeavessu.dds")
    assert cat.match("TreeNonexistentSpecies") == (None, None)


def test_no_source_empty():
    cat = TreeTextures(_FakeSource([]))
    assert cat.match("TreeEnglishOakForest01SU") == (None, None)
