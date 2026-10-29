"""One owned Source repair smoke: ordinary reference, then one HTTP server.

This is a bounded functional diagnostic, never qualification or a performance
campaign. ``prepare`` freezes source/artifact/policy inputs without loading a
model; ``run`` requires that fresh plan, obtains CPG and paired host locks,
and launches ``unit``. Do not run until the repair source has been reviewed.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import signal
import socket
import subprocess
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(Path(__file__).parent))
from mlx2.desktop_campaign_worker import atomic, check_ownership
from mlx2.desktop_runner import digest, source_manifest

HELLO = {'messages': [{'role': 'user', 'content': 'Hello'}], 'enable_thinking': False}
PLAN_FILES = frozenset({'spec.json', 'source.json', 'policy.json'})


def read_object(path):
    raw = path.read_bytes()
    if len(raw) > 2 << 20:
        raise ValueError('bounded JSON input exceeds2MiB')
    return json.loads(raw)


def baseline_policy(policy):
    # Original Source precision and its ordinary reference graph, before any
    # completion profile is selected. Expert derivatives use separate units.
    return {}


def validate_replay_receipts(policy, status, cold, warm):
    """Distinguish repeated ordinary execution from an actual APCv2 hit."""
    reusable = policy.get('deepseek_source_prefill_partition', 'adaptive_v1') == 'canonical_m1_v1'
    reuse = status.get('settings', {}).get('apcv2_reuse', {})
    if not reusable and reuse != {'enabled': False, 'reason': 'adaptive_prefill_partitions_not_qualified'}:
        raise RuntimeError('adaptive Source must explicitly disable APCv2 reuse')
    expected_route = 'external_draft' if policy.get('draft_model') else 'ordinary'
    for label, result in (('cold', cold), ('warm', warm)):
        receipt = result['receipt']
        cached = receipt.get('cached_tokens')
        expected_hit = reusable and label == 'warm'
        if (receipt.get('cache') != 'apcv2' or receipt.get('route') != expected_route
                or not receipt.get('route_receipt') or type(cached) is not int
                or (cached <= 0 if expected_hit else cached != 0)):
            raise RuntimeError(label + ' route/APCv2 receipt does not match the requested operation')
    return 'apcv2_cold_warm' if reusable else 'independent_repeated_execution_apcv2_disabled'


def server_command(spec, output):
    width = int(spec['policy'].get('deepseek_physical_batch', 'b1')[1:])
    route = '--external-draft' if spec['policy'].get('draft_model') else '--ordinary'
    return [sys.executable, '-m', 'mlx2.server', '--model', spec['model'], '--host', '127.0.0.1',
            '--port', str(spec['port']), '--fabric', 'off', route, '--max-lanes', str(width),
            '--max-inflight', str(max(1, width)), '--max-context', str(spec['max_context']),
            '--prefill-step', str(spec['prefill_step']), '--default-max-tokens', '32',
            '--cache-bytes', str(spec['cache_bytes']), '--cache-dir', str(output / 'apcv2'),
            '--execution-policy', str(output / 'policy.json'), '--gpu-keep-warm-seconds', '0',
            '--residency-keep-seconds', '0', '--generation-event-timeout-seconds', '600']


def prepare(args):
    output = args.output.resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError('refusing nonempty evidence directory')
    if not args.model or not args.policy:
        raise ValueError('prepare requires exact --model and --policy paths')
    from mlx2.adapters.deepseek_v41_source import source_policy
    from mlx2.adapters.deepseek_v41_source_artifact import inspect_artifact
    policy = read_object(args.policy)
    if not isinstance(policy, dict):
        raise TypeError('execution policy must be an object')
    from mlx2.serving import SERVER_OWNED_EXECUTION_POLICY_KEYS
    kernel = {key: value for key, value in policy.items() if key not in SERVER_OWNED_EXECUTION_POLICY_KEYS}
    if 'deepseek_source_experts' in kernel:
        raise ValueError('completion smoke uses the original Source expert provider; LFU/hot-map derivatives require a separate unit')
    policy.setdefault('moe_expert_streaming', {'enabled': True, 'cache_gib': 24})
    source_policy(kernel)  # fail incompatible combinations before any lease
    artifact = inspect_artifact(args.model)
    optional = read_object(args.request_json) if args.request_json else []
    optional = [optional] if isinstance(optional, dict) else optional
    if not isinstance(optional, list) or len(optional) > 8 or any(not isinstance(row, dict) for row in optional):
        raise ValueError('optional request JSON must be an object or a list of at most8objects')
    from mlx2.server import validate_request
    for body in optional:
        maximum = body.get('max_tokens', 32)
        if type(maximum) is not int or not 1 <= maximum <= 128 or body.get('n', 1) != 1:
            raise ValueError('repair requests require max_tokens1..128 and one output per member')
        validate_request(body, structured_thinking=True, allow_buffered_tool_stream=True,
                         constrained_tool_grammar=policy.get('constrained_tool_grammar', False))
    width = int(kernel.get('deepseek_physical_batch', 'b1')[1:])
    if width > 1 and optional and len(optional) != width:
        raise ValueError('a batch profile optional group must contain exactly its physical width')
    files = source_manifest(ROOT)
    spec = {'schema': 'mlx2.v41.serving-completion-smoke.v1', 'out': str(output), 'job_id': output.name,
            'model': str(args.model.resolve()), 'checkpoint': args.model.name + '@' + artifact['identity']['fingerprint'],
            'fingerprint': artifact['identity']['fingerprint'], 'policy': kernel, 'server_policy': policy,
            'optional_requests': optional, 'source_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
            'source_snapshot_sha256': digest(files), 'mcp_url': args.mcp_url,
            'port': args.port, 'max_context': args.max_context, 'prefill_step': args.prefill_step,
            'cache_bytes': args.cache_bytes, 'deadline_seconds': args.deadline_seconds,
            'thermal_max_c': args.thermal_max_c, 'qualification': False, 'performance_claim': False}
    output.mkdir(parents=True, exist_ok=True, mode=0o700)
    atomic(output / 'source.json', {'files': files, 'source_revision': spec['source_revision'], 'source_snapshot_sha256': digest(files)})
    atomic(output / 'policy.json', policy)
    atomic(output / 'spec.json', spec)
    return spec


def verify_plan(output):
    if not output.is_dir() or {path.name for path in output.iterdir()} != PLAN_FILES:
        raise ValueError('run requires exactly a fresh prepared plan; refusing prior evidence or missing inputs')
    spec = read_object(output / 'spec.json')
    if spec['out'] != str(output.resolve()) or spec['server_policy'] != read_object(output / 'policy.json'):
        raise ValueError('prepared output/policy binding changed')
    if digest(source_manifest(ROOT)) != spec['source_snapshot_sha256']:
        raise ValueError('source changed after preparation; use a new evidence directory after review')
    return spec


def _finite_nonempty(result):
    return bool((result.get('text') or '').strip() or result.get('tool_calls'))


def _calls_from_sse(path):
    calls = {}
    for line in path.read_text().splitlines():
        if not line.startswith('data: ') or line[6:] == '[DONE]':
            continue
        event = json.loads(line[6:])
        for choice in event.get('choices', ()):
            for call in (choice.get('delta') or {}).get('tool_calls', ()):
                target = calls.setdefault(call['index'], {'id': call.get('id'), 'name': '', 'arguments': ''})
                function = call.get('function') or {}
                target['name'] += function.get('name', '')
                target['arguments'] += function.get('arguments', '')
    return list(calls.values())


def unit(output):
    spec = read_object(output / 'spec.json')
    report = {**spec, 'passed': False, 'scope': 'bounded repair smoke; no qualified/default/performance claim', 'phases': []}
    adapter = server = log = mx = None
    cache = logits = model = None
    stop = threading.Event()
    sampler = None
    thermal_errors = []

    def phase(name, **values):
        report['phases'].append({'name': name, 'at': time.time(), **values})
        atomic(output / 'results.json', report)

    def signal_handler(number, frame):
        raise KeyboardInterrupt('owned unit interrupted: ' + str(number))

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    from probe_deepseek_v41_gpu import environment

    def thermal_samples():
        while not stop.wait(5):
            try:
                with (output / 'thermal-samples.jsonl').open('a') as stream:
                    stream.write(json.dumps(environment(), allow_nan=False) + '\n')
            except Exception as error:  # noqa: BLE001 - sampler failures fail the final receipt
                thermal_errors.append(str(error))
                return

    try:
        report['ownership'] = check_ownership(spec)
        if digest(source_manifest(ROOT)) != spec['source_snapshot_sha256']:
            raise RuntimeError('source changed before unit')
        from runpy import run_path
        guards = run_path(str(ROOT / 'scripts/run_with_gpu_locks.py'))
        foreign = guards['foreign_model_processes']()
        if foreign:
            raise RuntimeError('foreign model process before ordinary reference: ' + repr(foreign[:5]))
        report['before'] = environment()
        maximum = (report['before'].get('thermal') or {}).get('die_max_c')
        if maximum is None or not math.isfinite(maximum) or maximum > spec['thermal_max_c']:
            raise RuntimeError('initial thermal admission did not pass')
        if 'CPU_Speed_Limit' in report['before']['thermal_pressure']:
            raise RuntimeError('host reports thermal throttling')
        sampler = threading.Thread(target=thermal_samples, daemon=True)
        sampler.start()
        import mlx.core as mx

        from mlx2.adapters.deepseek_v41_source import DeepSeekV41SourceAdapter
        from mlx2.adapters.deepseek_v41_source_artifact import inspect_artifact
        from mlx2.runtime.streamed_load import WeightStreamRequest
        from mlx2.serving import preload_admission_limit_bytes
        artifact = inspect_artifact(spec['model'])
        if artifact['identity']['fingerprint'] != spec['fingerprint']:
            raise RuntimeError('Source checkpoint identity changed')
        mx.set_default_device(mx.gpu)
        mx.set_cache_limit(128 << 20)
        baseline = baseline_policy(spec['policy'])
        streaming = WeightStreamRequest(mode='moe_experts', budget_bytes=24 << 30,
                                       admission_limit_bytes=preload_admission_limit_bytes(), bypass_page_cache=True)
        adapter = DeepSeekV41SourceAdapter(spec['model'], execution_policy=baseline, weight_streaming=streaming)
        model = adapter.model
        cache = model.make_cache()
        prompt = adapter.prompt_tokens(HELLO)
        logits = model.prefill_forward(mx.array([prompt], mx.uint32), cache)
        mx.eval(logits, [row.state for row in cache])
        generated = []
        for _ in range(8):
            check_ownership(spec)
            row = logits[0, -1]
            if not bool(mx.all(mx.isfinite(row)).item()):
                raise RuntimeError('ordinary baseline logits are nonfinite')
            token = int(mx.argmax(row).item())
            if token in adapter.tokenizer.eos_token_ids:
                break
            generated.append(token)
            logits = model(mx.array([[token]], mx.uint32), cache)
            mx.eval(logits, [entry.state for entry in cache])
        text = adapter.tokenizer.decode(generated)
        if not text.strip():
            raise RuntimeError('ordinary Hello reference produced no visible text')
        phase('ordinary_reference', checkpoint=Path(spec['model']).name + '@' + adapter.identity['fingerprint'],
              artifact=adapter.identity, policy=baseline, request=HELLO, token_ids=generated, text=text, finite_logits=True,
              route='ordinary', qualification=False)
        baseline_tokens = list(generated)
        model = cache = logits = None
        adapter.close()
        adapter = None
        gc.collect()
        mx.synchronize()
        mx.clear_cache()
        if mx.get_active_memory() > 1 << 30:
            raise RuntimeError('ordinary reference retained model allocations before server')
        phase('ordinary_reference_released', active_mlx_bytes=mx.get_active_memory())
        check_ownership(spec)
        if guards['foreign_model_processes']():
            raise RuntimeError('foreign model process before loopback server')
        with socket.socket() as sock:
            if sock.connect_ex(('127.0.0.1', spec['port'])) == 0:
                raise RuntimeError('loopback port is occupied')
        from mlx2.desktop_campaign_process import spawn
        command = server_command(spec, output)
        log = (output / 'server.log').open('w')
        server = spawn(command, cwd=ROOT, env={**os.environ, 'PYTHONPATH': str(ROOT / 'src')},
                       stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        from probe_deepseek_v4_http import get_json, stream_chat
        base = 'http://127.0.0.1:' + str(spec['port'])
        deadline = time.monotonic() + 600
        while True:
            check_ownership(spec)
            if server.poll() is not None:
                raise RuntimeError('server exited before readiness')
            try:
                status = get_json(base, '/v1/status')
                if status.get('healthy'):
                    break
            except (OSError, ValueError):
                pass
            if time.monotonic() >= deadline:
                raise TimeoutError('server readiness deadline')
            time.sleep(1)
        if status.get('base_artifact', status.get('artifact')) != spec['fingerprint']:
            raise RuntimeError('server base artifact differs from frozen Source checkpoint')
        phase('server_ready', command=command, status=status)
        model_id = get_json(base, '/v1/models')['data'][0]['id']

        def request(body, name):
            check_ownership(spec)
            body = {'model': model_id, 'temperature': 0, 'max_tokens': 32, 'logprobs': True, **body, 'stream': True}
            if type(body['max_tokens']) is not int or not 1 <= body['max_tokens'] <= 128:
                raise ValueError('repair requests require max_tokens1..128')
            path = output / (name + '.sse')
            result = stream_chat(base, body, path, timeout=600)
            result['tool_calls'] = _calls_from_sse(path)
            result.pop('seconds', None)  # this harness makes no latency claim
            result.pop('timing_scope', None)
            if not _finite_nonempty(result) or not result['token_ids']:
                raise RuntimeError(name + ' has no finite/nonempty generated output')
            if any(not math.isfinite(value) for value in _logprobs(path)):
                raise RuntimeError(name + ' has nonfinite generated logprobs')
            atomic(output / (name + '.json'), result)
            return result

        cold = request(HELLO, 'cold')
        warm = request(HELLO, 'warm')
        replay_kind = validate_replay_receipts(spec['policy'], status, cold, warm)
        if cold['token_ids'] != warm['token_ids']:
            first = next((i for i, pair in enumerate(zip(cold['token_ids'], warm['token_ids'])) if pair[0] != pair[1]), min(len(cold['token_ids']), len(warm['token_ids'])))
            report['first_selected_divergence'] = {'position': first, 'shared_prefix_length': first,
                'scope': 'unclassified; no post-divergence logit attribution or numerical-equivalence verdict'}
            raise RuntimeError('cold/warm selected token divergence requires bounded localization')
        phase('cold_warm', replay_kind=replay_kind, exact_tokens=True,
              cold_receipt=cold['receipt'], warm_receipt=warm['receipt'])
        phase('ordinary_profile_ab', reference_token_ids=baseline_tokens,
              profile_token_ids=cold['token_ids'][:len(baseline_tokens)],
              exact_selected_prefix=baseline_tokens == cold['token_ids'][:len(baseline_tokens)],
              scope='bounded ordinary-reference versus selected-profile greeting; no numerical-equivalence or performance claim')
        requests = spec['optional_requests']
        width = int(spec['policy'].get('deepseek_physical_batch', 'b1')[1:])
        if width > 1 and requests:
            cohort = {'id': 'repair-' + output.name, 'size': width}
            requests = [{**body, 'batch_cohort': cohort, 'skip_writing_prefix_cache': True} for body in requests]
            with ThreadPoolExecutor(max_workers=width) as pool:
                optional = list(pool.map(lambda pair: request(pair[1], 'optional-' + str(pair[0])), enumerate(requests)))
        else:
            optional = [request(body, 'optional-' + str(index)) for index, body in enumerate(requests)]
        phase('optional_requests', count=len(optional), receipts=[row['receipt'] for row in optional])
        report['status_after'] = get_json(base, '/v1/status')
        efficiency = (report['status_after'].get('execution') or {}).get('scheduled_kv_calls') or {}
        if spec['policy'].get('deepseek_ced') == 'dependency_exact' and requests:
            if efficiency.get('ced_skipped_layer_rows', 0) <= 0:
                raise RuntimeError('bounded CED request did not exercise dependency pruning')
        if spec['policy'].get('deepseek_kv_storage') == 'native_packed':
            if efficiency.get('native_packed_local_rows', 0) <= 0:
                raise RuntimeError('selected native-packed route did not store packed rows')
        batch = (report['status_after'].get('execution') or {}).get('scheduled_batch_calls') or {}
        report['batch_probe'] = {'configured_width': width, 'requested_optional_group': bool(width > 1 and requests),
                                 'observed_counters': batch}
        if width > 1 and requests and (batch.get('maximum_live_requests', 0) < width or batch.get('joint_ffn_calls', 0) <= 0):
            raise RuntimeError('requested batch group did not execute the configured joint FFN width')
        if digest(source_manifest(ROOT)) != spec['source_snapshot_sha256']:
            raise RuntimeError('source changed during unit')
        check_ownership(spec)
        if thermal_errors:
            raise RuntimeError('thermal sampler failed')
        report['passed'] = True
    except BaseException as error:  # noqa: BLE001 - record failure, clean owned resources and return nonzero
        report['error'] = {'type': type(error).__name__, 'message': str(error), 'traceback': traceback.format_exc()}
    finally:
        report['cleanup'] = {}
        if server is not None:
            try:
                server.stop(timeout=30)
                report['cleanup']['server_descendants_stopped'] = True
            except BaseException as error:  # noqa: BLE001 - continue cleanup and fail the receipt
                report['cleanup']['server_error'] = str(error)
                report['passed'] = False
        if log is not None:
            log.close()
        model = cache = logits = None
        if adapter is not None:
            try:
                adapter.close()
            except BaseException as error:  # noqa: BLE001 - continue cleanup and fail the receipt
                report['cleanup']['adapter_error'] = str(error)
                report['passed'] = False
        adapter = None
        if mx is not None:
            try:
                mx.synchronize()
                gc.collect()
                mx.clear_cache()
                report['cleanup']['active_mlx_bytes'] = mx.get_active_memory()
            except BaseException as error:  # noqa: BLE001 - continue cleanup and fail the receipt
                report['cleanup']['mlx_error'] = str(error)
                report['passed'] = False
        stop.set()
        if sampler:
            sampler.join(timeout=15)
        try:
            report['after'] = environment()
        except BaseException as error:  # noqa: BLE001 - final observation failure must remain visible
            report['cleanup']['environment_error'] = str(error)
            report['passed'] = False
        if thermal_errors:
            report['thermal_sampler_errors'] = thermal_errors
            report['passed'] = False
        report['finished_at'] = time.time()
        atomic(output / 'results.json', report)
    return 0 if report['passed'] else 1


def _logprobs(path):
    for line in path.read_text().splitlines():
        if line.startswith('data: ') and line[6:] != '[DONE]':
            for choice in json.loads(line[6:]).get('choices', ()):
                for token in (choice.get('logprobs') or {}).get('content', ()):
                    yield token['logprob']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'run', 'unit'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', type=Path)
    parser.add_argument('--policy', type=Path)
    parser.add_argument('--request-json', type=Path)
    parser.add_argument('--mcp-url', default='http://127.0.0.1:8766/mcp')
    parser.add_argument('--port', type=int, default=8291)
    parser.add_argument('--max-context', type=int, default=4096)
    parser.add_argument('--prefill-step', type=int, default=128)
    parser.add_argument('--cache-bytes', type=int, default=512 << 20)
    parser.add_argument('--deadline-seconds', type=int, default=1800)
    parser.add_argument('--thermal-max-c', type=float, default=60)
    args = parser.parse_args()
    if not (1024 <= args.port <= 65535 and 128 <= args.max_context <= 65536
            and 1 <= args.prefill_step <= 1024 and 0 < args.cache_bytes <= 8 << 30
            and 60 <= args.deadline_seconds <= 3600 and math.isfinite(args.thermal_max_c)
            and 40 <= args.thermal_max_c <= 75):
        parser.error('bounded port/context/prefill/cache/deadline/thermal settings are invalid')
    output = args.output.resolve()
    if args.mode == 'prepare':
        prepare(args)
        return 0
    if args.mode == 'unit':
        return unit(output)
    spec = verify_plan(output)
    from mlx2.desktop_campaign_lease import DesktopCPGLease
    from mlx2.desktop_campaign_process import spawn
    with DesktopCPGLease(output, spec['job_id'], mcp_url=spec['mcp_url'], checkpoint=spec['checkpoint'], source_revision=spec['source_revision']) as lease:
        command = [sys.executable, str(ROOT / 'scripts/run_with_gpu_locks.py'), '--session', lease.owner['cpg_session'],
                   '--label', spec['job_id'], '--mcp-url', spec['mcp_url'], '--desktop-cpg-lease', str(lease.path),
                   '--receipt', str(output / 'paired-locks.json'), '--', sys.executable, '-u', str(Path(__file__).resolve()),
                   'unit', '--output', str(output)]
        child = spawn(command, cwd=ROOT, env={**os.environ, 'PYTHONPATH': str(ROOT / 'src')}, start_new_session=True)
        deadline = time.monotonic() + spec['deadline_seconds']
        try:
            while child.poll() is None:
                lease.check()
                if time.monotonic() >= deadline:
                    raise TimeoutError('bounded Source repair ownership window expired')
                time.sleep(1)
            lease.success = child.returncode == 0
            return child.returncode
        finally:
            child.stop(timeout=35)


if __name__ == '__main__':
    raise SystemExit(main())
