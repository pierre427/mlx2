"""Check (and optionally repair) packs whose tokenizer.json lost the declared split rule.

Run after every conversion; exits 1 while any pack is inconsistent.

    python scripts/check_tokenizer_pretokenize.py ~/mlx-models/* [--fix]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mlx2.runtime.tokenizer_integrity import check_pack, repair_pack  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("packs", nargs="+", type=Path)
    parser.add_argument("--fix", action="store_true", help="rewrite the regex in tokenizer.json")
    args = parser.parse_args(argv)
    bad = 0
    for pack in args.packs:
        if not (pack / "tokenizer_config.json").is_file():
            continue
        report = (repair_pack if args.fix else check_pack)(pack)
        status = report["status"]
        # A declared rule with no tokenizer.json to carry it is a broken pack.
        bad += status in {"mismatch", "unsupported", "missing"}
        print(f"{status:12s} {pack}" + (f"  ({report['reason']})" if "reason" in report else ""))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
