# Vendored utf8proc

ScamGuard uses utf8proc for NFC normalization before native tokenization. These
four upstream files are unmodified copies from the official JuliaStrings repo:

- Repository: https://github.com/JuliaStrings/utf8proc
- Release: https://github.com/JuliaStrings/utf8proc/releases/tag/v2.10.0
- Tag: `v2.10.0`
- Commit: `a1b99daa2a3393884220264c927a48ba1251a9c6`
- Unicode data version: 16.0.0
- Imported: 2026-10-10
- Source URL prefix: `https://raw.githubusercontent.com/JuliaStrings/utf8proc/a1b99daa2a3393884220264c927a48ba1251a9c6/`

The release/tag commit was verified against the official repository. The MIT
license and Unicode data license notices are retained in `LICENSE.md`; include
that file when redistributing the native library. `utf8proc.c` includes
`utf8proc_data.c`, so only the former is compiled as a source file. The local
CMake target is static and position independent for desktop, iOS, and Android.

## SHA-256 of unmodified upstream files

| File | SHA-256 |
| --- | --- |
| `utf8proc.c` | `1f36d14e69e4a5f0658635496f5a9b1f76c418dc36841d2c39d61ab61a258c5d` |
| `utf8proc.h` | `d947b38124a9f218631b537dd7a367f6a2a6a992c72caa46b3f3e7c81e6138ee` |
| `utf8proc_data.c` | `6aa28d305d65af54260a3521278715c49c4e2445c9c017e8ebf38da3735422e5` |
| `LICENSE.md` | `3b510150d34f248a221bb88e1d811238d6c6c18b51231822c42974c39bb07256` |

`checksums.sha256` can be checked from this directory with `shasum -a 256 -c checksums.sha256`.

The ScamGuard helper calls `utf8proc_map` with explicit byte length and
`UTF8PROC_STABLE | UTF8PROC_COMPOSE`. It preserves embedded NULs, rejects invalid
UTF-8, and does not enable compatibility normalization, case folding, or removal
of marks/control characters. This intentionally changes protocol 4 token IDs
where NFC composition is needed; protocol 5 artifacts require fresh calibration.
