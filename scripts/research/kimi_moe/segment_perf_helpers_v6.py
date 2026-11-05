"""Fenced component phases plus same-clock serving metrics, not production qualification."""
import hashlib
import json
import math
import time
from collections import Counter


def delta(before, after):
    if any(after.get(k, 0) < v for k, v in before.items()):
        raise RuntimeError('Counter reset during direct phase')
    return dict(Counter(after) - Counter(before))


def serving_metrics(response, expected_prompt, completion=128):
    usage, receipt = response['usage'], response['mlx2']
    first, elapsed = receipt['ttft_seconds'], receipt['elapsed_seconds']
    if not (usage['prompt_tokens'] == expected_prompt and usage['completion_tokens'] == completion
            and receipt['cached_tokens'] == 0 and math.isfinite(first) and math.isfinite(elapsed) and 0 < first < elapsed):
        raise RuntimeError('Invalid cold fixed-budget timing receipt')
    return {'prompt_tokens': expected_prompt, 'completion_tokens': completion,
            'cold_prefill_to_first_token_seconds': first,
            'cold_prompt_tokens_per_first_token_second': expected_prompt / first,
            'decode_after_first_seconds': elapsed-first,
            'decode_after_first_tokens_per_second': (completion-1)/(elapsed-first),
            'elapsed_job_seconds': elapsed,
            'scope': 'Same job-start clock. TTFT includes prompt work, scheduling and first token; not pure PP. Decode includes serving/host overhead.'}


def direct_samples(adapter, frames, budget, engage, policy, repeats=3, on_sample=None):
    import mlx.core as mx
    records = []
    for frame in frames:
        prompt = adapter.prompt_tokens(frame['body'])
        if len(prompt) != frame['prompt_tokens']:
            raise RuntimeError('Component prompt identity mismatch')
        for repetition in range(-1, repeats):
            cache = adapter.model.make_cache()
            before = dict(budget.COUNTS)
            start = time.perf_counter()
            for offset in range(0, len(prompt), 128):
                logits = adapter.model(mx.array([prompt[offset:offset+128]], mx.int32), cache=cache)
                mx.eval(logits, cache)
            prefill_seconds = time.perf_counter()-start
            middle = dict(budget.COUNTS)
            prefill_routes = {'normal': delta(before, middle), 'extra': {}}
            if prefill_routes['normal'].get('router_rows') != 26 * len(prompt) or not engage(policy, prefill_routes)['passed']:
                raise RuntimeError('Component PP engagement failure')
            eos = mx.array(sorted(adapter.tokenizer.eos_token_ids), mx.int32)
            eos_mask = (mx.arange(logits.shape[-1])[:, None] == eos[None, :]).any(-1)
            def sample(value, mask=eos_mask):
                return int(mx.argmax(mx.where(mask, -mx.inf, value[0,-1])).item())
            output = [sample(logits)]
            start = time.perf_counter()
            for _ in range(127):
                logits = adapter.model(mx.array([[output[-1]]], mx.int32), cache=cache)
                mx.eval(logits, cache)
                output.append(sample(logits))
            decode_seconds = time.perf_counter()-start
            decode_routes = {'normal': delta(middle, dict(budget.COUNTS)), 'extra': {}}
            if decode_routes['normal'].get('router_rows') != 26 * 127 or not engage(policy, decode_routes)['passed']:
                raise RuntimeError('Component decode engagement failure')
            records.append({'frame':frame['name'], 'repetition':repetition, 'warmup':repetition==-1,
                            'prompt_tokens':len(prompt), 'prefill_step':128, 'decode_forward_steps':127, 'eos_suppressed_min_tokens':128,
                            'prefill_seconds':prefill_seconds, 'prefill_tokens_per_second':len(prompt)/prefill_seconds,
                            'decode_seconds':decode_seconds, 'decode_forward_tokens_per_second':127/decode_seconds,
                            'prefill_routes':prefill_routes, 'decode_routes':decode_routes,
                            'output_token_ids':output,
                            'prompt_token_ids_sha256':hashlib.sha256(json.dumps(prompt).encode()).hexdigest(),
                            'scope':'Direct adapter, fresh KV per sample, no APC. All PP chunks and decode steps GPU-evaluated before phase timer stops. Sampling included in decode; first sample per frame excluded warmup. EOS suppressed to bound equal128 output tokens; not semantic grading.'})
            if on_sample is not None:
                on_sample(records)
    return records
