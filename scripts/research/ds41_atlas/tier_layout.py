"""Print the resolved reduced-tier layout for a Source policy JSON (CPU only).

usage: tier_layout.py POLICY.json [--json]

POLICY.json is either a full execution policy or just the
``deepseek_source_experts`` object. The same pure resolver the store uses
(deepseek_v41_source_tiers.resolve via q1_on_miss.resolve_layout) prints
per-layer budgets, expert counts per tier, rank tiers and the layout
identity that enters the approximate-operation salt.
"""
import sys as _tree_sys  # mlx2 worktree bootstrap: import this tree's src
from pathlib import Path as _TreePath

_tree_src = str(_TreePath(__file__).resolve().parents[3] / "src")
if _tree_src not in _tree_sys.path[:1]:
    _tree_sys.path.insert(0, _tree_src)
del _tree_sys, _TreePath, _tree_src
import argparse
import json

from mlx2.adapters import deepseek_v41_source_lfu as lfu
from mlx2.adapters import deepseek_v41_source_q1_on_miss as q1m
from mlx2.adapters import deepseek_v41_source_tiers as tiers

SOURCE = 'c30fd25e3cc79ba7044686fa3f9a7a1c0f514f77ee6293d2f5f62bc409e02ddb'


def resolved(policy):
    expert = policy.get('deepseek_source_experts', policy)
    expert = lfu.parse_policy(expert)
    mode = next((k for k in q1m.KEYS if k in expert), None)
    if mode is None:
        raise SystemExit('policy selects neither q1_on_miss nor tiers')
    settings = expert[mode]
    usage = None
    if mode == q1m.TIERS_KEY:
        usage, _ = tiers.load_usage(settings['usage_map'], SOURCE)
    return mode, q1m.resolve_layout(settings, mode, budget_bytes=expert['budget_bytes'], usage=usage)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('policy'); parser.add_argument('--json', action='store_true')
    args = parser.parse_args()
    mode, layout = resolved(json.loads(open(args.policy).read()))
    if args.json:
        print(json.dumps({k: v for k, v in layout.items() if k != 'map'} | {'mode': mode}, indent=1))
    else:
        print('mode=' + mode)
        print(tiers.summary(layout))


if __name__ == '__main__':
    main()
