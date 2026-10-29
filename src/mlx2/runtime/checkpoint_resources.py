# SPDX-License-Identifier: MIT
"""Optional adapter resource hints; independent of numeric APCv2 validity.

No model dispatch, device work or physical leases live here. An adapter may
predict/preload resources after a valid prefix restore. Missing, stale or bad
hints cost that optimization only; normal lazy loading remains authoritative.
"""
import json

from .expert_residency_hints import BINDING_FIELDS, ExpertResidencyHints, prefix_token_digest


def resource_binding(adapter):
    try:
        hook = getattr(adapter, 'checkpoint_resource_binding', None)
        if not callable(hook):
            return None
        value = dict(hook())
        if set(value) != set(BINDING_FIELDS):
            return None
        if any(type(v) is not str or not 1 <= len(v) <= 512 for v in value.values()):
            return None
        return tuple(sorted(value.items()))
    except Exception:
        return None


def publication_hints(adapter, prompt_cache, tokens):
    """Return only a source-bound, host-only hint for this exact prefix."""
    try:
        hook = getattr(adapter, 'checkpoint_resource_hints', None)
        binding = resource_binding(adapter)
        if not callable(hook) or binding is None:
            return None, dict(action='not_supported')
        hint = hook(prompt_cache, tokens)
        if hint is None:
            return None, dict(action='declined', reason='no_resource_inventory')
        if type(hint) is not ExpertResidencyHints:
            return None, dict(action='declined', reason='invalid_hint_type')
        hint = hint.for_store(tokens)
        if hint is None or hint.for_restore(len(tokens), binding) is None:
            return None, dict(action='declined', reason='resource_or_prefix_binding_mismatch')
        return hint, dict(action='attached', experts=len(hint.experts), metadata_bytes=hint.nbytes,
                          phase=hint.phase, coverage=hint.coverage,
                          truncated=getattr(hint, 'truncated', False), physical_buffers_owned=False)
    except Exception as error:
        return None, dict(action='declined', reason='resource_hint_error', error_type=type(error).__name__)


def restore_resources(adapter, lookup, tokens):
    """Use a hint only after numeric lookup succeeds; never alter its hit."""
    if not getattr(lookup, 'hit', False) or not getattr(lookup, 'cache', None):
        return dict(action='not_applicable')
    hint = getattr(lookup, 'expert_hints', None)
    if hint is None:
        return dict(action='declined', reason='no_expert_hint')
    try:
        binding = resource_binding(adapter)
        hook = getattr(adapter, 'restore_checkpoint_resources', None)
        if binding is None or not callable(hook):
            return dict(action='not_supported')
        if type(hint) is not ExpertResidencyHints:
            return dict(action='declined', reason='invalid_hint_type')
        covered = getattr(lookup, 'cached_tokens', None)
        hint = hint.for_restore(covered, binding)
        if hint is None:
            return dict(action='declined', reason='resource_binding_mismatch')
        # A shorter restore may legitimately originate in a different longer
        # branch. APCv2 validated that stored observation; do not relabel its
        # original digest or claim its full expert set describes the new path.
        if covered == hint.covered_tokens and prefix_token_digest(tokens[:covered]) != hint.token_digest:
            return dict(action='declined', reason='prefix_digest_mismatch')
        result = hook(hint, lookup.cache)
        if result is None:
            result = dict(action='declined', reason='adapter_returned_no_result')
        if type(result) is not dict:
            return dict(action='declined', reason='invalid_resource_telemetry')
        encoded = json.dumps(result, allow_nan=False, separators=(',', ':'))
        if len(encoded.encode()) > 16 << 10:
            return dict(action='declined', reason='resource_telemetry_too_large')
        return dict(action='adapter', hint_experts=len(hint.experts), phase=hint.phase,
                    coverage=hint.coverage, resource=result, numeric_hit_unchanged=True)
    except Exception as error:
        return dict(action='declined', reason='resource_prepare_error', error_type=type(error).__name__,
                    numeric_hit_unchanged=True)
