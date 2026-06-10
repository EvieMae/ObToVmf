"""Read Oblivion BSA archives (version 103) to extract meshes/textures by path.

Layout + hash per UESP "Oblivion Mod:BSA File Format" / "Hash Calculation".
A DataSource abstracts over loose extracted files and one or more BSAs.
"""
from __future__ import annotations

import mmap
import os
import struct
import zlib

_FLAG_DIR_NAMES = 0x1
_FLAG_FILE_NAMES = 0x2
_FLAG_COMPRESSED = 0x4
_FLAG_EMBED_NAMES = 0x100   # full path prefixed on file data (absent in stock BSAs)
_SIZE_MASK = 0x3FFFFFFF
_SIZE_INVERT = 0x40000000


def tes_hash(name):
    """Oblivion folder/file name hash (uint64). Pass a folder path (no ext) or a
    leaf file name. Name is lowercased and '/'->'\\' normalised by the caller's
    convention; we normalise here too."""
    name = name.lower().replace("/", "\\")
    root, ext = os.path.splitext(name)
    chars = [c & 0xFF for c in root.encode("latin-1", "replace")]
    if not chars:
        return 0
    mask = 0xFFFFFFFF
    hash1 = (chars[-1]
             | ((chars[-2] << 8) if len(chars) > 2 else 0)
             | (len(chars) << 16)
             | (chars[0] << 24)) & mask
    if ext == ".kf":
        hash1 |= 0x80
    elif ext == ".nif":
        hash1 |= 0x8000
    elif ext == ".dds":
        hash1 |= 0x8080
    elif ext == ".wav":
        hash1 |= 0x80000000
    hash1 &= mask
    hash2 = 0
    for c in chars[1:-2]:
        hash2 = ((hash2 * 0x1003F) + c) & mask
    hash3 = 0
    for c in ext.encode("latin-1", "replace"):
        hash3 = ((hash3 * 0x1003F) + c) & mask
    hash2 = (hash2 + hash3) & mask
    return ((hash2 << 32) + hash1) & 0xFFFFFFFFFFFFFFFF


class Bsa:
    def __init__(self, path):
        self.path = path
        self._index = {}        # (folder_hash, file_hash) -> (offset, size, compressed)
        self._embed_names = False
        self._parse()

    def _parse(self):
        with open(self.path, "rb") as f:
            buf = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            try:
                self._parse_buf(buf)
            finally:
                buf.close()

    def _parse_buf(self, buf):
        if bytes(buf[0:4]) != b"BSA\x00":
            raise ValueError("not a BSA: %s" % self.path)
        version = struct.unpack_from("<I", buf, 4)[0]
        if version != 103:
            raise ValueError("unsupported BSA version %d (need 103)" % version)
        flags = struct.unpack_from("<I", buf, 12)[0]
        folder_count = struct.unpack_from("<I", buf, 16)[0]
        total_file_name_len = struct.unpack_from("<I", buf, 28)[0]
        default_comp = bool(flags & _FLAG_COMPRESSED)
        self._embed_names = bool(flags & _FLAG_EMBED_NAMES)

        # folder records
        folders = []
        pos = 36
        for _ in range(folder_count):
            fhash = struct.unpack_from("<Q", buf, pos)[0]
            fcount = struct.unpack_from("<I", buf, pos + 8)[0]
            foffset = struct.unpack_from("<I", buf, pos + 12)[0]
            folders.append((fhash, fcount, foffset))
            pos += 16

        for fhash, fcount, foffset in folders:
            p = foffset - total_file_name_len      # stored offset includes total file-name length
            if flags & _FLAG_DIR_NAMES:
                namelen = buf[p]
                p += 1 + namelen                   # bzstring: 1 len byte (incl. null) + bytes
            for _ in range(fcount):
                xhash = struct.unpack_from("<Q", buf, p)[0]
                rawsize = struct.unpack_from("<I", buf, p + 8)[0]
                offset = struct.unpack_from("<I", buf, p + 12)[0]
                size = rawsize & _SIZE_MASK
                compressed = default_comp ^ bool(rawsize & _SIZE_INVERT)
                self._index[(fhash, xhash)] = (offset, size, compressed)
                p += 16

    def list_files(self):
        """Return all file paths ('folder\\file') in the archive (needs the
        dir-names + file-names flags, which Oblivion BSAs have)."""
        with open(self.path, "rb") as f:
            buf = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            try:
                flags = struct.unpack_from("<I", buf, 12)[0]
                folder_count = struct.unpack_from("<I", buf, 16)[0]
                total_file_name_len = struct.unpack_from("<I", buf, 28)[0]
                if not (flags & _FLAG_DIR_NAMES) or not (flags & _FLAG_FILE_NAMES):
                    return []
                folders = []
                pos = 36
                for _ in range(folder_count):
                    cnt = struct.unpack_from("<I", buf, pos + 8)[0]
                    off = struct.unpack_from("<I", buf, pos + 12)[0]
                    folders.append((cnt, off))
                    pos += 16
                fol_names, counts, end = [], [], 0
                for cnt, off in folders:
                    p = off - total_file_name_len
                    nlen = buf[p]
                    p += 1
                    fol_names.append(bytes(buf[p:p + nlen - 1]).decode("latin-1"))
                    counts.append(cnt)
                    p += nlen - 1 + cnt * 16
                    end = max(end, p)
                names = [n for n in bytes(buf[end:end + total_file_name_len]).split(b"\x00") if n]
                out, i = [], 0
                for fol, cnt in zip(fol_names, counts):
                    for _ in range(cnt):
                        if i < len(names):
                            out.append(fol + "\\" + names[i].decode("latin-1"))
                            i += 1
                return out
            finally:
                buf.close()

    def extract(self, rel_path):
        """Return file bytes for e.g. 'meshes\\architecture\\foo.nif', or None."""
        norm = rel_path.replace("/", "\\")
        folder, _, leaf = norm.rpartition("\\")
        key = (tes_hash(folder), tes_hash(leaf))
        rec = self._index.get(key)
        if rec is None:
            return None
        offset, size, compressed = rec
        with open(self.path, "rb") as f:
            f.seek(offset)
            blob = f.read(size)
        # NOTE: Oblivion v103 file data starts directly with the (optional
        # uint32 originalSize +) payload. The 0x100 archive flag is NOT an
        # embedded-name prefix here (that's Skyrim/v104), so we never skip a
        # bstring -- doing so corrupts the zlib stream.
        if compressed:
            return zlib.decompress(blob[4:])       # uint32 originalSize prefix + zlib
        return blob


class DataSource:
    """Resolve a mesh by its MODL path (relative to Data\\Meshes\\) from loose
    extracted files and/or BSAs. Loose files take priority."""

    def __init__(self, data_dir=None, bsa_paths=()):
        # data_dir may be a single path or a list of loose-file roots (mod folders
        # take priority in the order given, before the BSAs).
        if data_dir is None:
            self.data_dirs = []
        elif isinstance(data_dir, (list, tuple)):
            self.data_dirs = [d for d in data_dir if d]
        else:
            self.data_dirs = [data_dir]
        self.bsas = []
        for p in bsa_paths:
            self.bsas.append(Bsa(p))

    @property
    def data_dir(self):
        return self.data_dirs[0] if self.data_dirs else None

    def get_mesh(self, modl_path):
        rel = "meshes\\" + modl_path.replace("/", "\\").lstrip("\\")
        return self._get(rel)

    def get_texture(self, tex_path):
        rel = tex_path.replace("/", "\\").lstrip("\\")
        if not rel.lower().startswith("textures\\"):
            rel = "textures\\" + rel
        return self._get(rel)

    def get_landscape_texture(self, icon):
        """Resolve an LTEX ICON, which is a bare name relative to textures\\landscape\\."""
        icon = icon.replace("/", "\\").lstrip("\\")
        return self.get_texture("landscape\\" + icon) or self.get_texture(icon)

    def _get(self, rel):
        for d in self.data_dirs:
            fp = os.path.join(d, rel.replace("\\", os.sep))
            if os.path.isfile(fp):
                with open(fp, "rb") as f:
                    return f.read()
        for b in self.bsas:
            data = b.extract(rel)
            if data is not None:
                return data
        return None
