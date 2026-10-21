# Eagle 4 materials in mlx2

This project includes or is derived from Eagle 4 technology developed by
Eagle AI Corp. It is not an official Eagle AI Corp project unless expressly
stated otherwise. mlx2 is not affiliated with, certified by, sponsored by,
endorsed by, or an official project of Eagle AI Corp, and its modifications
have not been reviewed or approved by Eagle AI Corp.

## Which files

These files contain, port, or are derived from Eagle 4 and are Eagle Files:

- `src/mlx2/runtime/drafters/eagle4.py`
- `src/mlx2/runtime/drafters/eagle4_config.py`
- `src/mlx2/adapters/eagle4.py`

Each carries `SPDX-License-Identifier: LicenseRef-Eagle-4-Source-Available`.
They are **not** covered by mlx2's Apache License 2.0: the root `LICENSE`
applies to the rest of mlx2 only. Every other mlx2 file, including the
scheduler, verifier, caches and the dispatch entry in
`src/mlx2/adapters/external_drafter_formats.py`, remains Apache-2.0.

## License: source-available, not open source

The Eagle Files are made available under the **Eagle 4 Source-Available
License** in [`LICENSE`](LICENSE). It is **not an OSI-approved open-source
license**. It limits Commercial Use and Competing Offerings; read it before
any use beyond research, evaluation, education or personal noncommercial use.
Commercial-license inquiries go to Eagle AI Corp at licensing@eagle.ai.

Downloading, cloning, forking, copying, accessing, modifying, using, hosting,
distributing, or otherwise exercising any licensed right in the Eagle Files
constitutes acceptance of the Eagle 4 Source-Available License and the
incorporated Eagle Trademark Policy. Mere viewing or inspection does not
itself constitute trademark use.

The **Eagle Trademark Policy** is `TRADEMARKS.md` in the Eagle 4 repository:
<https://github.com/eagle-hq/EAGLE-4/blob/main/TRADEMARKS.md> (the version
read for this port is at revision `75436712e7415d352cf9f17d070019d07410786f`:
<https://github.com/eagle-hq/EAGLE-4/blob/75436712e7415d352cf9f17d070019d07410786f/TRADEMARKS.md>).

## Files in this directory

- [`LICENSE`](LICENSE): the Eagle 4 Source-Available License, unmodified.
- [`NOTICE`](NOTICE): the Eagle 4 NOTICE file, unmodified.
- [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) and [`LICENSES/`](LICENSES/):
  Eagle 4's third-party notices and license texts, unmodified. The EAGLE-4
  draft layer that mlx2 ports (`eagle_harness/sglang/models/llama_eagle4.py`)
  derives from SGLang (Copyright 2023-2024 SGLang Team) and SafeAILab EAGLE
  (Copyright 2025 SafeAI Lab (SAIL)), both Apache-2.0. mlx2 does not include
  the Eagle 4 kernels or native library those notices also cover.
- [`MODIFICATIONS.md`](MODIFICATIONS.md): the modification notice and change
  log mapping each material modification to the affected files.

## Trademarks

EAGLE, Eagle 4, EAGLE-Harness, and EAGLE Efficient LM Head are trademarks or
claimed trademarks of Eagle AI Corp. mlx2 refers to them only to identify the
origin of the files above and the checkpoint format they load.
