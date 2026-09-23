"""Retarget a versioned libm dependency in an ELF shared object.

vLLM's _moe_C.abi3.so was built against a newer glibc and its .gnu.version_r section declares
that it needs version GLIBC_2.29 from libm.so.6. The loader rejects the library on a glibc 2.28
host purely on that declaration, before any symbol is looked up, which is why LD_PRELOAD cannot
help. One symbol is involved, log2, and the 2.29 revision differs from the original only in
errno and exception handling, not in the value returned.

So the fix is to point that dependency at a version this host does define. The Vernaux entry
naming GLIBC_2.29 is rewritten to name GLIBC_2.2.5 instead, which leaves the version index
untouched: every symbol that referenced the old entry now resolves against log2@GLIBC_2.2.5.

usage: downgrade_verneed.py <library> <from-version> <to-version>
"""

from __future__ import annotations

import shutil
import struct
import sys


def elf_hash(name: bytes) -> int:
    h = 0
    for byte in name:
        h = ((h << 4) + byte) & 0xFFFFFFFFFFFFFFFF
        g = h & 0xF0000000
        if g:
            h ^= g >> 24
        h &= ~g & 0xFFFFFFFFFFFFFFFF
    return h & 0xFFFFFFFF


def sections(blob: bytes) -> dict[str, tuple[int, int]]:
    """name -> (file offset, size), 64-bit little-endian ELF only."""
    if blob[:4] != b"\x7fELF" or blob[4] != 2 or blob[5] != 1:
        raise SystemExit("not a 64-bit little-endian ELF")
    e_shoff, = struct.unpack_from("<Q", blob, 0x28)
    e_shentsize, e_shnum, e_shstrndx = struct.unpack_from("<HHH", blob, 0x3A)
    raw = []
    for i in range(e_shnum):
        base = e_shoff + i * e_shentsize
        sh_name, = struct.unpack_from("<I", blob, base)
        sh_offset, sh_size = struct.unpack_from("<QQ", blob, base + 0x18)
        raw.append((sh_name, sh_offset, sh_size))
    str_off = raw[e_shstrndx][1]
    out = {}
    for sh_name, sh_offset, sh_size in raw:
        end = blob.index(b"\0", str_off + sh_name)
        out[blob[str_off + sh_name:end].decode()] = (sh_offset, sh_size)
    return out


def find_in_dynstr(blob: bytes, dynstr: tuple[int, int], want: bytes) -> int:
    off, size = dynstr
    table = blob[off:off + size]
    idx = table.find(b"\0" + want + b"\0")
    if idx < 0:
        raise SystemExit(f"{want.decode()} is not in .dynstr; cannot retarget onto it")
    return idx + 1


def main() -> None:
    if len(sys.argv) != 4:
        raise SystemExit(__doc__)
    path, old, new = sys.argv[1], sys.argv[2].encode(), sys.argv[3].encode()

    blob = bytearray(open(path, "rb").read())
    secs = sections(bytes(blob))
    if ".gnu.version_r" not in secs:
        raise SystemExit("no .gnu.version_r section")
    verneed_off, _ = secs[".gnu.version_r"]
    dynstr = secs[".dynstr"]
    new_name_off = find_in_dynstr(bytes(blob), dynstr, new)

    def dynstr_at(offset: int) -> bytes:
        base = dynstr[0] + offset
        return bytes(blob[base:blob.index(b"\0", base)])

    patched = []
    cursor = verneed_off
    while True:
        vn_cnt, vn_file, vn_aux, vn_next = struct.unpack_from("<HIII", blob, cursor + 2)
        aux = cursor + vn_aux
        for _ in range(vn_cnt):
            vna_hash, _flags, vna_other, vna_name, vna_next = \
                struct.unpack_from("<IHHII", blob, aux)
            if dynstr_at(vna_name) == old:
                struct.pack_into("<I", blob, aux, elf_hash(new))
                struct.pack_into("<I", blob, aux + 8, new_name_off)
                patched.append((dynstr_at(vn_file).decode(), vna_other))
            if not vna_next:
                break
            aux += vna_next
        if not vn_next:
            break
        cursor += vn_next

    if not patched:
        raise SystemExit(f"no dependency on {old.decode()} found; nothing to do")

    shutil.copy(path, path + ".orig")
    open(path, "wb").write(blob)
    for library, index in patched:
        print(f"{path}: need {old.decode()} from {library} "
              f"(version index {index}) -> {new.decode()}")
    print(f"original saved as {path}.orig")


if __name__ == "__main__":
    main()
