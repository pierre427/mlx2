"""Opt-in match-aware PLD tree budgets, with bounded cost-control defaults.

No learned probabilities are invented. Each proposed token is copied from
committed history and verified by the target. The existing request-local cost
controller prices completed target transactions; this module prices no GPU
work from Python graph-building time. See provenance/nanbeige43-tree.json.
"""

from ..runtime.ast_tree_proposals import paths_to_tree
from ..runtime.drafters.prompt_lookup_draft import PromptLookupDraftModel


class NanbeigeAdaptivePromptLookup(PromptLookupDraftModel):
    """Use one deep unambiguous copy, or distribute the budget over copies."""
    def __init__(self, target, *, num_draft=15, ngram_min=2, ngram_max=6,
                 min_chain_ngram=4, max_branches=4):
        super().__init__(target, num_draft=num_draft, ngram_min=ngram_min, ngram_max=ngram_max)
        if type(min_chain_ngram) is not int or not ngram_min <= min_chain_ngram <= ngram_max:
            raise ValueError("adaptive PLD chain ngram must fit the search bounds")
        if type(max_branches) is not int or not 1 <= max_branches <= 4:
            raise ValueError("adaptive PLD branches must be in 1..4")
        self.min_chain_ngram = min_chain_ngram
        self.max_branches = max_branches
        self.adaptive_stats = {"chain": 0, "branches": 0, "no_match": 0,
                               "requested_nodes": 0, "proposed_nodes": 0}

    @property
    def receipt_settings(self):
        return {**super().receipt_settings, "adaptive_copy_budget": True,
                "min_chain_ngram": self.min_chain_ngram,
                "max_branches": self.max_branches,
                "budget_counters": dict(self.adaptive_stats)}

    def propose_tree(self, anchors, hidden, cache, max_nodes, *, processor_histories=None,
                     forbidden_token_ids=None, topology=None, mark=None, **options):
        # Explicit topologies remain authoritative. "drafter" opts into this
        # provider's policy and is what the adapter should select for auto.
        if topology not in (None, "drafter"):
            return super().propose_tree(anchors, hidden, cache, max_nodes,
                processor_histories=processor_histories, forbidden_token_ids=forbidden_token_ids,
                topology=topology, mark=mark, **options)
        anchors = [int(token) for token in anchors]
        histories = processor_histories or [[] for _ in anchors]
        forbidden = forbidden_token_ids or [() for _ in anchors]
        budget = min(int(max_nodes), self.num_draft)
        if budget < 1:
            raise ValueError("adaptive PLD requires a positive node budget")
        result = []
        for history, anchor, blocked in zip(histories, anchors, forbidden, strict=True):
            copies, ngram = self.copy_paths(history, anchor, budget, self.max_branches)
            self.adaptive_stats["requested_nodes"] += budget
            if not copies:
                selected = []
                self.adaptive_stats["no_match"] += 1
            elif ngram >= self.min_chain_ngram and len({tuple(c[:min(3, len(c))]) for c in copies}) == 1:
                selected = [copies[0][:budget]]
                self.adaptive_stats["chain"] += 1
            else:
                # Every branch gets at least one useful successor. Depth is
                # bounded by its actual copy, never filled with fake nodes.
                branches = min(len(copies), budget, self.max_branches)
                base, extra = divmod(budget, branches)
                selected = [copy[:base+(index < extra)] for index, copy in enumerate(copies[:branches])]
                self.adaptive_stats["branches"] += 1
            tokens, parents = paths_to_tree(selected, budget, forbidden_root=blocked)
            self.adaptive_stats["proposed_nodes"] += len(tokens)
            result.append((tokens, parents))
        self.stats["tree_calls"] += 1
        return result


def cost_control_defaults():
    """Exploration is bounded and charged; the common controller owns it."""
    return {"enabled": True, "ordinary_samples": 2, "speculative_samples": 2,
            "loss_decisions": 1, "loss_ratio": 0.97, "settle_rounds": 0,
            "reversible": True, "reprobe_tokens": 32, "reprobe_max_tokens": 256,
            "pipelined_park": True}


def load_adaptive_prompt_lookup(record, target):
    """Adapter loader with the same call contract as the existing PLD loader."""
    return NanbeigeAdaptivePromptLookup(target, num_draft=int(record["args"].block_size)-1,
        ngram_min=int(record.get("ngram_min", 2)), ngram_max=int(record.get("ngram_max", 6)))
