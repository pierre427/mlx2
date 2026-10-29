"""Bounded, untimed generic AST provider/commit trace; numerical code untouched.

The outer owned functional probe persists the yielded dictionary in finally.
Only16 greedy IDs cross the GPU boundary; full logits are never copied here.
"""
from contextlib import contextmanager
import numpy as np


def _cache_metadata(cache):
    rows, history = [], None
    for index, row in enumerate(cache):
        rows.append(dict(layer=index, offset=int(row.offset), ratio=int(row.ratio),
                         source=bool(row.source), pool_count=int(row._pool_count),
                         partial_shape=None if row.buf_kv is None else list(row.buf_kv.shape)))
        if row.hash_tail is not None:
            history = np.asarray(row.hash_tail).tolist()
    return dict(layers=rows, hash_tail=history)


class _TracedTransaction:
    def __init__(self, inner, entry, cache, check_live):
        self.inner, self.entry, self.cache, self.check_live = inner, entry, cache, check_live

    @property
    def provider_protocol(self):
        return self.inner.provider_protocol

    @property
    def provider_revision_token(self):
        return self.inner.provider_revision_token

    @property
    def closed(self):
        return self.inner.closed

    def _event(self, operation, paths):
        if operation != "abort":
            self.check_live()
        if len(self.entry["events"]) >= 64 and operation != "abort":
            raise ValueError("AST provider trace exceeded64 transaction events")
        inverse = self.entry["caller_to_canonical"]
        clean = [list(path) for path in paths]
        # Trace only legal integer metadata, without changing target validation.
        canonical = [[inverse[i] for i in path] if all(type(i) is int and 0 <= i < 16 for i in path)
                     else None for path in clean]
        event = dict(operation=operation, caller_paths=clean, canonical_paths=canonical,
                     consumed_tokens=[[self.entry["inputs"][i] for i in path] if mapped is not None else None
                                      for path, mapped in zip(clean, canonical)],
                     expected_frontier_offsets=[self.entry["base_offset"]+len(path) for path in clean],
                     greedy_after_path=[self.entry["greedy_tokens"][path[-1]] if path and mapped is not None else None
                                        for path, mapped in zip(clean, canonical)],
                     closed_before=self.closed)
        if len(self.entry["events"]) < 64:
            self.entry["events"].append(event)
        else:
            self.entry["cleanup_abort"] = event
        return event

    def prepared_cache(self, path):
        event = self._event("prepared_cache", [path])
        try:
            result = self.inner.prepared_cache(path)
            event.update(result_cache=_cache_metadata(result), closed_after=self.closed)
            return result
        except BaseException as error:
            event["error"] = dict(type=type(error).__name__, message=str(error))
            raise

    def commit_paths(self, paths):
        event = self._event("commit_paths", paths)
        try:
            result = self.inner.commit_paths(paths)
            event.update(result_caches=[_cache_metadata(row) for row in result],
                         original_cache_after=_cache_metadata(self.cache), closed_after=self.closed)
            return result
        except BaseException as error:
            event.update(error=dict(type=type(error).__name__, message=str(error)), closed_after=self.closed)
            raise

    def abort(self):
        event = self._event("abort", [])
        try:
            result = self.inner.abort()
            event.update(original_cache_after=_cache_metadata(self.cache), closed_after=self.closed)
            return result
        except BaseException as error:
            event["error"] = dict(type=type(error).__name__, message=str(error))
            raise


@contextmanager
def instrument(provider, check_live, *, max_tiles=1):
    if not callable(check_live) or max_tiles != 1:
        raise ValueError("AST provider trace requires a live callback and exactlyone tile maximum")
    import mlx.core as mx
    from mlx2.adapters.deepseek_v41_ast_target import _canonical_rows, _paths
    original = provider.forward
    owned_attribute = "forward" in provider.__dict__
    original_attribute = provider.__dict__.get("forward")
    report = dict(schema="mlx2.ds41-ast-provider-trace.v1", maximum_tiles=1,
                  numerical_source_unchanged=True, qualified=False, performance_claim=False,
                  full_logits_host_copies=0, tiles=[])
    def forward(model, inputs, parents, cache, layers, *, cached=True):
        check_live()
        if report["tiles"]:
            raise ValueError("AST provider trace permits exactlyone bounded target tile")
        canonical = _canonical_rows(parents)
        inverse = [canonical.index(i) for i in range(16)]
        entry = dict(inputs=list(inputs), parents=list(parents),
                     canonical_to_caller=list(canonical), caller_to_canonical=inverse,
                     canonical_inputs=[inputs[i] for i in canonical],
                     caller_ancestry=[list(path) for path in _paths(parents)],
                     base_offset=int(cache[0].offset), base_cache=_cache_metadata(cache),
                     capture_layers=list(layers), events=[], stage="forward")
        report["tiles"].append(entry)
        transaction = None
        try:
            logits, features, transaction = original(model, inputs, parents, cache, layers, cached=cached)
            greedy = (mx.argmax(logits, axis=-1) if isinstance(logits, mx.array)
                      else np.argmax(np.asarray(logits), axis=-1))
            entry.update(greedy_tokens=np.asarray(greedy).reshape(-1).tolist(),
                         logits_shape=list(logits.shape),
                         features_shape=None if features is None else list(features.shape),
                         stage="returned", provider_revision_token=transaction.provider_revision_token)
            if len(entry["greedy_tokens"]) != 16:
                raise ValueError("AST provider trace expected16 greedy outputs")
            return logits, features, _TracedTransaction(transaction, entry, cache, check_live)
        except BaseException as error:
            entry.update(stage="failed", error=dict(type=type(error).__name__, message=str(error)))
            if transaction is not None and not transaction.closed:
                transaction.abort()
            raise
    provider.forward = forward
    try:
        yield report
    finally:
        if owned_attribute:
            provider.forward = original_attribute
        else:
            del provider.forward
