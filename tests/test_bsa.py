import struct
import zlib

from oblivion2vmf.bsa import Bsa, DataSource, tes_hash


def build_bsa(folder, filename, filedata, compressed=False):
    """Minimal single-folder/single-file Oblivion BSA (v103)."""
    flags = 0x1 | 0x2 | (0x4 if compressed else 0)
    folder_b = folder.encode("latin-1")
    file_b = filename.encode("latin-1")
    tfolder = len(folder_b) + 1
    tfile = len(file_b) + 1

    if compressed:
        blob = struct.pack("<I", len(filedata)) + zlib.compress(filedata)
    else:
        blob = filedata
    size_field = len(blob)

    header = bytearray(b"BSA\x00")
    header += struct.pack("<IIIIIIII", 103, 36, flags, 1, 1, tfolder, tfile, 0)
    assert len(header) == 36

    perfolder_start = 36 + 16
    bz = bytes([tfolder]) + folder_b + b"\x00"
    filerec_start = perfolder_start + len(bz)
    nameblock_start = filerec_start + 16
    filedata_start = nameblock_start + tfile

    folder_rec = struct.pack("<QII", tes_hash(folder), 1, perfolder_start + tfile)
    file_rec = struct.pack("<QII", tes_hash(filename), size_field, filedata_start)
    name_block = file_b + b"\x00"

    return bytes(header) + folder_rec + bz + file_rec + name_block + blob


def test_tes_hash_deterministic_and_distinct():
    assert tes_hash("meshes\\test") == tes_hash("meshes/test")   # slash normalised
    assert tes_hash("foo.nif") == tes_hash("FOO.NIF")            # case normalised
    assert tes_hash("foo.nif") != tes_hash("bar.nif")
    assert isinstance(tes_hash("foo.nif"), int)


def test_bsa_extract_uncompressed(tmp_path):
    data = b"hello nif bytes" * 4
    p = tmp_path / "a.bsa"
    p.write_bytes(build_bsa("meshes\\test", "foo.nif", data))
    b = Bsa(str(p))
    assert b.extract("meshes\\test\\foo.nif") == data
    assert b.extract("meshes/test/foo.nif") == data        # slash form
    assert b.extract("meshes\\test\\missing.nif") is None


def test_bsa_extract_compressed(tmp_path):
    data = b"compress me " * 50
    p = tmp_path / "c.bsa"
    p.write_bytes(build_bsa("meshes\\architecture", "house.nif", data, compressed=True))
    b = Bsa(str(p))
    assert b.extract("meshes\\architecture\\house.nif") == data


def test_datasource_loose_priority(tmp_path):
    # loose file wins over BSA
    loose = tmp_path / "data"
    (loose / "meshes" / "clutter").mkdir(parents=True)
    (loose / "meshes" / "clutter" / "rock.nif").write_bytes(b"LOOSE")
    bsa = tmp_path / "z.bsa"
    bsa.write_bytes(build_bsa("meshes\\clutter", "rock.nif", b"ARCHIVE"))
    src = DataSource(data_dir=str(loose), bsa_paths=[str(bsa)])
    assert src.get_mesh("clutter\\rock.nif") == b"LOOSE"
    assert src.get_mesh("clutter\\missing.nif") is None


def test_datasource_bsa_fallback(tmp_path):
    bsa = tmp_path / "z.bsa"
    bsa.write_bytes(build_bsa("meshes\\clutter", "rock.nif", b"ARCHIVE"))
    src = DataSource(data_dir=None, bsa_paths=[str(bsa)])
    assert src.get_mesh("clutter\\rock.nif") == b"ARCHIVE"
