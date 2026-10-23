"""Prompt lookup as an external-draft / AST tree-tile proposal backend.

Original mlx2 implementation.  A model-free drafter: proposals are copies of
earlier spans of the request's own committed history (prompt plus output)
that follow the latest n-gram, the indexed prompt-lookup oracle of
``runtime/prompt_lookup.py``.  On the tree-tile route it fills a tile with
several distinct copy continuations (most specific n-gram, most recent site
first), shared prefixes folded once.  Wrapping another drafter (``base``)
grafts the copy path onto that drafter's tree: the copy branch first when a
specific n-gram matched, then the base drafter's paths in their own order.

Every proposal is a deterministic point mass; the target verifies exactly.
The per-lane index lives here, not in the draft cache: the draft cache stays
empty so recovery snapshots never copy an index.  Each call extends the
index of the matching earlier history incrementally (committed history only
grows), bounded by a small LRU.
"""

from __future__ import annotations

from types import SimpleNamespace

from ..prompt_lookup import IndexedPromptLookup

_INDEX_LIMIT = 32
_TAIL_CHECK = 32


class PromptLookupDraftModel:
    """External-draft backend over the request's own committed history."""

    requires_processor_histories = True
    supports_logits_processors = True
    proposal_distribution = "deterministic_point_mass"
    tree_topology_aware = True
    # Grafting reorders complete paths as the node budget changes. Its first
    # N rows need not reproduce a separately constructed N-node graft.
    tree_budget_prefix_invariant = False

    def __init__(self, target, *, num_draft, ngram_min=2, ngram_max=6, base=None,
                 copy_first_ngram=3):
        if type(num_draft) is not int or not 1 <= num_draft <= 15:
            raise ValueError("prompt lookup num_draft must be an integer from 1 to 15")
        if not 1 <= int(ngram_min) <= int(ngram_max):
            raise ValueError("prompt lookup ngram bounds are invalid")
        self.base = base
        if callable(getattr(base, "tree_codebooks", None)):
            # The base lattice's cached codebooks (DFlash2); every other
            # base capability (lattice prelaunch included) stays off here.
            self.tree_codebooks = base.tree_codebooks
        args = getattr(target, "speculative_args", None) or target.args
        layers = getattr(target, "layers", None)
        if layers is None:
            layers = target.model.layers
        if base is not None:
            self.config = base.config
            self.requires_context_tokens = bool(getattr(base, "requires_context_tokens", False))
        else:
            # Taps are never read; the last layer keeps the capture cheapest.
            # A zero draft geometry: memory admission reserves no draft KV,
            # no draft transient and no draft layers for a model-free drafter.
            self.config = SimpleNamespace(
                block_size=num_draft + 1,
                vocab_size=int(args.vocab_size),
                target_layer_ids=(len(layers) - 1,),
                hidden_size=0,
                intermediate_size=0,
                num_hidden_layers=0,
                num_key_value_heads=0,
                head_dim=0,
                feature_width=int(args.hidden_size),
            )
            self.requires_context_tokens = False
        self.num_draft = int(num_draft)
        self.ngram_min, self.ngram_max = int(ngram_min), int(ngram_max)
        self.copy_first_ngram = int(copy_first_ngram)
        self._indexes: list[IndexedPromptLookup] = []
        self.stats = {"tree_calls": 0, "copy_paths": 0, "copy_first": 0,
                      "index_rebuilds": 0, "index_extends": 0}

    # -- receipts ----------------------------------------------------------
    @property
    def receipt_kind(self):
        return (getattr(self.base, "receipt_kind", "external_draft")
                if self.base is not None else "external_prompt_lookup")

    @property
    def receipt_settings(self):
        settings = {
            "kind": "prompt_lookup",
            "ngram_min": self.ngram_min,
            "ngram_max": self.ngram_max,
            "num_draft": self.num_draft,
            "qualified": False,
        }
        if self.base is not None:
            base = dict(getattr(self.base, "receipt_settings", {"kind": type(self.base).__name__}))
            base["graft"] = settings
            return base
        return settings

    # -- cache contract (empty: the index is drafter-owned) ------------------
    def make_cache(self):
        return list(self.base.make_cache()) if self.base is not None else []

    def batch_caches(self, rows):
        if self.base is not None:
            return self.base.batch_caches(rows)
        return [list(row) for row in rows]

    def append_context(self, hidden, cache, *, context_tokens=None):
        if self.base is not None:
            kwargs = {"context_tokens": context_tokens} if self.requires_context_tokens else {}
            return self.base.append_context(hidden, cache, **kwargs)
        return None

    # -- index -----------------------------------------------------------------
    def _index(self, tokens):
        """The index over ``tokens``, extending an earlier prefix's index."""

        n = len(tokens)
        for position, index in enumerate(self._indexes):
            m = len(index.tokens)
            # A common suffix is only a cheap rejection filter. Independent
            # requests can share that suffix while having different n-gram
            # sites earlier in their histories; reuse needs an exact prefix.
            if (
                m <= n
                and index.tokens[-_TAIL_CHECK:] == tokens[max(0, m - _TAIL_CHECK):m]
                and index.tokens == (tokens if m == n else tokens[:m])
            ):
                for token in tokens[m:]:
                    index.observe(token)
                self._indexes.append(self._indexes.pop(position))
                self.stats["index_extends"] += 1
                return index
        index = IndexedPromptLookup(
            tokens, ngram_min=self.ngram_min, ngram_max=self.ngram_max,
            hot_segments=0, reject_ttl=0,
        )
        self._indexes.append(index)
        if len(self._indexes) > _INDEX_LIMIT:
            self._indexes.pop(0)
        self.stats["index_rebuilds"] += 1
        return index

    def copy_paths(self, history, anchor, depth, limit):
        """Distinct copy continuations after ``history + [anchor]``.

        Longest matching n-gram first, most recent site first.  Returns
        ``(paths, best_ngram)``.
        """

        tokens = [int(token) for token in history] + [int(anchor)]
        index = self._index(tokens)
        paths, seen, best = [], set(), 0
        for size in range(min(self.ngram_max, len(tokens)), self.ngram_min - 1, -1):
            key = tuple(tokens[-size:])
            for start in reversed(index.index[size].get(key, ())):
                if start + size >= len(tokens):
                    continue
                path = tuple(tokens[start + size:start + size + depth])
                if not path or path in seen:
                    continue
                seen.add(path)
                paths.append(path)
                best = max(best, size)
                if len(paths) >= limit:
                    return paths, best
        return paths, best

    # -- proposals -------------------------------------------------------------
    def draft_distributions(self, anchors, hidden, cache, proposal_length, rngs, temperatures,
                            *, processor_histories=None, **kwargs):
        from ..proposal_providers import PointMassLaw

        if self.base is not None:
            return self.base.draft_distributions(
                anchors, hidden, cache, proposal_length, rngs, temperatures,
                processor_histories=processor_histories, **kwargs,
            )
        histories = processor_histories or [[] for _ in anchors]
        tokens = []
        for history, anchor in zip(histories, anchors):
            paths, _ = self.copy_paths(history or [], anchor, int(proposal_length), 1)
            tokens.append(list(paths[0]) if paths else [])
        return tokens, [[PointMassLaw(token) for token in row] for row in tokens]

    def propose_tree(self, anchors, hidden, cache, max_nodes, *, processor_histories=None,
                     forbidden_token_ids=None, topology=None, mark=None, **options):
        from ..ast_tree_proposals import paths_to_tree, pattern_for_budget
        from .dflash_tree import tree_paths

        anchors = [int(anchor) for anchor in anchors]
        histories = processor_histories or [[] for _ in anchors]
        forbidden = forbidden_token_ids or [() for _ in anchors]
        max_nodes = int(max_nodes)
        pattern = pattern_for_budget(topology, max_nodes)
        # A graft's exact-copy branch uses its available tile budget; the
        # base drafter's default topology must not silently truncate it.
        depth = min(max_nodes if self.base is not None else max(pattern, default=1),
                    self.num_draft)
        base_trees = None
        if self.base is not None:
            base_options = dict(options)
            if getattr(self.base, "tree_topology_aware", False):
                base_options["topology"] = topology
            if getattr(self.base, "requires_processor_histories", False):
                base_options["processor_histories"] = processor_histories
            base_trees = self.base.propose_tree(
                anchors, hidden, cache, max_nodes, forbidden_token_ids=forbidden,
                mark=mark, **base_options,
            )
        self.stats["tree_calls"] += 1
        result = []
        for row, (history, anchor) in enumerate(zip(histories, anchors)):
            limit = 1 if base_trees is not None else max(1, len(pattern))
            copies, ngram = self.copy_paths(history or [], anchor, depth, limit)
            self.stats["copy_paths"] += len(copies)
            if base_trees is None:
                result.append(paths_to_tree(copies, max_nodes, forbidden_root=forbidden[row]))
                continue
            tokens, parents = base_trees[row]
            node_paths = tree_paths(list(parents))
            interior = {int(parent) for parent in parents}
            leaves = [path for node, path in enumerate(node_paths) if node not in interior]
            base_paths = [tuple(tokens[i] for i in path) for path in leaves]
            if copies and ngram >= self.copy_first_ngram:
                self.stats["copy_first"] += 1
                paths = copies + base_paths
            else:
                paths = base_paths + copies
            result.append(paths_to_tree(paths, max_nodes, forbidden_root=forbidden[row]))
        return result


def load_prompt_lookup_drafter(record, target, *, base=None):
    return PromptLookupDraftModel(
        target,
        num_draft=int(record["args"].block_size) - 1,
        ngram_min=int(record.get("ngram_min", 2)),
        ngram_max=int(record.get("ngram_max", 6)),
        base=base,
    )


__all__ = ["PromptLookupDraftModel", "load_prompt_lookup_drafter"]
