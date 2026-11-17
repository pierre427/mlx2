# SPDX-License-Identifier: LicenseRef-Pierre-Lamy-Calculator-NC-ND-1.0
# Copyright (c) 2026 Pierre Lamy
# Terms: provenance/LICENSE-calculator.txt; scope: provenance/LICENSE-calculator-scope.json
"""Transactional resident-calculator state for exact external draft verification.

The target applies the same device processor to every reachable verify prefix.
Draft probes use detached device state; verifier ledgers commit only used rows.
No parser output, gate bit or result token crosses to Python here. Tree15 uses
the selected-path row-wise verifier. Progressive private AST branches remain
unsupported: this processor deliberately does not claim the history-pure branch
contract and that executor records its primary-chain fallback.
"""
from __future__ import annotations

from .amoeba_calc_resident import ResidentCalculatorProcessor


class SpeculativeResidentCalculatorProcessor(ResidentCalculatorProcessor):
    speculative_protocol = 'resident-calculator-exact-verify-v1'
    progressive_branch_safe = False
    history_pure = False

    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self._speculative_lane=None
        self._position=self.prompt_length-1
        self._verify_pending=False
        self._detached=False

    def bind_speculative_lane(self,uid):
        uid=int(uid)
        if self._speculative_lane not in (None,uid):
            raise ValueError('resident calculator processor cannot be shared between speculative lanes')
        self._speculative_lane=uid

    def _binding_valid(self):
        # V5 reads no activation probe: the admitted lane, complete bytes and
        # checked arithmetic own its gate, not a stale packed-forward row.
        return self._speculative_lane is not None and self.tiles.route=='external_draft'

    def __call__(self,tokens,logits):
        size=int(tokens.size)
        if not self._binding_valid():
            raise ValueError('resident speculative calculator has no admitted lane binding')
        if size!=self._position+1:
            raise ValueError('resident speculative calculator requires consecutive target prefixes')
        published=self.entry
        if self._verify_pending or self._detached:
            self.entry={'state':self.state,'seen':[0,0,0,0],'finished':False}
        try:
            value=super().__call__(tokens,logits)
            self._position=size
            return value
        finally:
            self.entry=published

    def verify_ledger(self):
        # MLX arrays are immutable values; subsequent kernel calls replace the
        # array, so retaining its reference is an exact device-state snapshot.
        return self.state,self.first,self._position

    def begin_verify_ledger(self):
        if self._verify_pending:
            raise RuntimeError('resident calculator verify transaction is already open')
        self._verify_pending=True

    def restore_verify_ledger(self,ledger):
        self.state,self.first,self._position=ledger

    def commit_verify_ledger(self,ledger):
        self.restore_verify_ledger(ledger)
        self._verify_pending=False
        self.entry['state']=self.state

    def __deepcopy__(self,memo):
        # Share immutable vocabulary/request/device arrays and the installed
        # tile set. Never copy its lock or register a provisional counter row.
        existing=memo.get(id(self))
        if existing is not None:return existing
        clone=object.__new__(type(self));memo[id(self)]=clone
        clone.__dict__=dict(self.__dict__)
        clone.entry={'state':self.state,'seen':list(self.entry['seen']),'finished':False}
        clone._detached=True
        return clone

    def restore_from_recovery(self,saved):
        if type(saved) is not type(self) or saved._speculative_lane!=self._speculative_lane:
            raise ValueError('resident calculator recovery changed lane identity')
        self.restore_verify_ledger(saved.verify_ledger())
        self._verify_pending=saved._verify_pending
        self.entry['state']=self.state

    def probe(self,tokens,logits):
        """Replay only the provisional suffix on disposable device state.

        DFlash calls one isolated probe for each proposal position. Starting
        every probe from the live state without replay would repeatedly force
        the first result digit. Replay uses neutral rows because draft probes
        are proposal guidance; exact target verification remains authoritative.
        """
        import mlx.core as mx
        clone=self.__deepcopy__({});size=int(tokens.size)
        if size<=clone._position:
            raise ValueError('resident calculator draft probe precedes its committed prefix')
        # A malicious or incompatible drafter must not replay an unbounded
        # transcript. Current external draft windows are far smaller than 256.
        if size-clone._position>256:
            raise ValueError('resident calculator provisional suffix exceeds 256 tokens')
        neutral=mx.zeros_like(logits)
        for end in range(clone._position+1,size):
            clone(tokens[:end],neutral)
        return clone(tokens,logits)


__all__=['SpeculativeResidentCalculatorProcessor']
