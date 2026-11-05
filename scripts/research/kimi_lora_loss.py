"""Auxiliary held-out assistant-token loss; never a task-success oracle."""
from __future__ import annotations
import math


def validate_loss_rows(rows):
    if not rows:
        raise ValueError('heldout loss set is empty')
    for row in rows:
        ids, mask = row.get('input_ids'), row.get('loss_mask')
        if (row.get('split') not in {'validation', 'dev'} or row.get('evaluation_only') is not True
                or row.get('never_train') is not True):
            raise ValueError('development loss requires explicit held-out validation rows')
        if (not row.get('id') or not isinstance(ids, list) or not isinstance(mask, list)
                or not 2 <= len(ids) <= 4096 or len(ids) != len(mask)
                or any(type(token) is not int or not 0 <= token < 163840 for token in ids)
                or any(type(value) not in (int, float, bool) or value not in (0, 1) for value in mask)
                or not sum(mask[1:])):
            raise ValueError('invalid complete heldout loss row')


def evaluate_loss(model, rows, *, chunk_size=128, guard=lambda: None):
    import mlx.core as mx
    import mlx.nn as nn
    validate_loss_rows(rows)
    if type(chunk_size) is not int or chunk_size < 1:
        raise ValueError('positive loss chunk size required')
    results = []
    for row in rows:
        cache = model.make_cache()
        ids, mask = row['input_ids'], row['loss_mask']
        total, targets = 0.0, 0
        for start in range(0, len(ids) - 1, chunk_size):
            end = min(start + chunk_size, len(ids) - 1)
            logits = model(mx.array([ids[start:end]], mx.int32), cache=cache)
            if logits.shape[:2] != (1, end - start):
                raise ValueError('loss forward must retain every causal logit row')
            selected = mx.array([mask[start + 1:end + 1]], mx.float32)
            losses = nn.losses.cross_entropy(logits.astype(mx.float32),
                mx.array([ids[start + 1:end + 1]], mx.int32), reduction='none')
            value = (losses * selected).sum()
            mx.eval(value, cache)
            total += float(value.item())
            targets += sum(mask[start + 1:end + 1])
            guard()
        if not math.isfinite(total):
            raise ValueError('heldout loss is nonfinite')
        results.append({'id': row['id'], 'source': row.get('source'), 'split_group': row.get('split_group'),
                        'assistant_tokens': int(targets), 'nll_sum': total, 'mean_nll': total / targets})
    token_count = sum(row['assistant_tokens'] for row in results)
    nll = sum(row['nll_sum'] for row in results) / token_count
    return {'rows': results, 'examples': len(results), 'assistant_tokens': token_count,
            'token_mean_nll': nll, 'token_perplexity': math.exp(min(nll, 700)),
            'example_mean_nll': sum(row['mean_nll'] for row in results) / len(results),
            'purpose': 'auxiliary teacher-forced heldout likelihood, not task correctness'}
