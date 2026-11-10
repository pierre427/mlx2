#!/usr/bin/env python3
"""CPU-only Kimi Linear LoRA data preparation and frozen behavioral oracles.

No model weights are loaded. One complete dialogue is one sequence: callers
must reset both attention and recurrent/convolution state between examples.
The checkpoint has no native reasoning channel; original reasoning_content is
explicitly rendered as ordinary assistant text, never silently discarded.
"""
from __future__ import annotations

import sys as _tree_sys  # mlx2 worktree bootstrap: import this tree's src, not the editable install
from pathlib import Path as _TreePath

_tree_src = str(_TreePath(__file__).resolve().parents[2] / "src")
if _tree_src not in _tree_sys.path[:1]:
    _tree_sys.path.insert(0, _tree_src)
del _tree_sys, _TreePath, _tree_src

import argparse
from collections import Counter, defaultdict
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import re

SERIALIZATION = 'kimi-linear-inline-reasoning-v2'
END_ID = 163586
CAPS = (512, 1024, 2048, 4096)


def digest(value):
    if not isinstance(value, (str, bytes)):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(value.encode() if isinstance(value, str) else value).hexdigest()


def file_digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as reader:
        for block in iter(lambda: reader.read(8 << 20), b''):
            result.update(block)
    return result.hexdigest()


def write_new(path, value, *, jsonl=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    partial = path.with_suffix(path.suffix + '.partial')
    if path.exists() or partial.exists():
        raise FileExistsError('Refusing to overwrite frozen artifact: ' + str(path))
    with partial.open('x') as writer:
        if jsonl:
            for row in value:
                writer.write(json.dumps(row, ensure_ascii=False, separators=(',', ':')) + '\n')
        else:
            writer.write(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
        writer.flush()
        os.fsync(writer.fileno())
    try:
        os.link(partial, path)
    finally:
        partial.unlink()


def load_tokenizer(model_path):
    from mlx2.runtime.kimi_tokenizer import TikTokenTokenizer
    path = Path(model_path)
    tokenizer = TikTokenTokenizer.from_pretrained(path, local_files_only=True)
    tokenizer.chat_template = (path / 'chat_template.jinja').read_text()
    if tokenizer.encode('<|im_end|>') != [END_ID]:
        raise ValueError('Unexpected Kimi end-of-message token')
    return tokenizer


def tokenizer_receipt(model_path):
    path = Path(model_path).resolve()
    names = ('config.json', 'tokenizer_config.json', 'special_tokens_map.json',
             'chat_template.jinja', 'tiktoken.model', 'generation_config.json')
    config = json.loads((path / 'config.json').read_text())
    files = {name: file_digest(path / name) for name in names}
    return {'model_path': str(path), 'artifact_name': path.name, 'files': files,
            'serialization_fingerprint': digest(files), 'model_type': config['model_type'],
            'num_experts': config['num_experts'], 'quantization': config.get('quantization'),
            'end_of_message_token': END_ID, 'native_reasoning_channel': False,
            'weights_loaded': False}


def training_messages(messages):
    from mlx2.adapters.kimi_linear_output import normalize_call_ids
    result = copy.deepcopy(messages)
    for message in result:
        if message.get('role') not in {'system', 'developer', 'user', 'assistant', 'tool'}:
            raise ValueError('Unsupported message role')
        content = message.get('content')
        if content is None:
            message['content'] = ''
        elif not isinstance(content, str):
            raise ValueError('Only existing text messages are supported by this experiment')
        reasoning = message.get('reasoning_content')
        if reasoning is not None and not isinstance(reasoning, str):
            raise ValueError('Reasoning must be original text')
        if reasoning and message['role'] == 'assistant':
            answer = message['content']
            message['content'] = 'Reasoning:\n' + reasoning + ('\n\nAnswer:\n' + answer if answer else '\n\n')
        # The corpus preserves both native {name,arguments,id} calls and
        # OpenAI function envelopes. Wrap only the former, without changing
        # the original name, arguments or linkage identity.
        for index, call in enumerate(message.get('tool_calls') or []):
            if 'function' not in call and isinstance(call.get('name'), str):
                if 'arguments' not in call:
                    raise ValueError('Direct tool call has no arguments')
                message['tool_calls'][index] = {'id': call.get('id'), 'type': 'function',
                    'function': {'name': call['name'], 'arguments': call['arguments']}}
    result = normalize_call_ids(result)
    pending, seen = set(), set()
    for message in result:
        for call in message.get('tool_calls') or []:
            function = call.get('function', {})
            cid, name = call.get('id'), function.get('name')
            if not isinstance(name, str) or not name or not isinstance(cid, str):
                raise ValueError('Tool call lacks identity')
            if not re.fullmatch(r'functions\.' + re.escape(name) + r':\d+', cid):
                raise ValueError('Native tool id contradicts function name')
            if cid in seen:
                raise ValueError('Repeated tool call id')
            arguments = function.get('arguments', {})
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
            if not isinstance(arguments, dict):
                raise ValueError('Tool arguments must be a JSON object')
            seen.add(cid)
            pending.add(cid)
        if message['role'] == 'tool':
            cid = message.get('tool_call_id')
            if cid not in pending:
                raise ValueError('Unmatched tool result')
            pending.remove(cid)
    # The source admission owns terminal-submit validation. All nonterminal
    # pending calls must have results; only calls in the final assistant action
    # may lack one, and the data builder checks their upstream receipt.
    final_ids = {c['id'] for c in result[-1].get('tool_calls', [])} if result else set()
    if pending - final_ids:
        raise ValueError('Missing nonterminal tool result')
    return result, pending


def training_tools(tools):
    """Unwrap source JSON strings and wrap bare schemas without inventing tools."""
    if tools is None:
        return None
    if isinstance(tools, str):
        tools = json.loads(tools)
    if not isinstance(tools, list):
        raise ValueError('Tool definitions must be a list')
    result = []
    for tool in copy.deepcopy(tools):
        if isinstance(tool, str):
            tool = json.loads(tool)
        if not isinstance(tool, dict):
            raise ValueError('Tool definition must be an object')
        if 'function' not in tool and isinstance(tool.get('name'), str):
            tool = {'type': 'function', 'function': tool}
        if tool.get('type') != 'function' or not isinstance(tool.get('function'), dict):
            raise ValueError('Unsupported tool definition envelope')
        result.append(tool)
    return result


def prepare_record(record, tokenizer, max_tokens=None):
    meta = record.get('metadata', {})
    source = record.get('source', {})
    split = record.get('split', meta.get('split'))
    group = record.get('split_group', meta.get('split_group'))
    if split != 'train' or not group:
        raise ValueError('Training requires an explicit train split and whole-conversation group')
    if meta.get('held_out') or meta.get('never_train'):
        raise ValueError('Source explicitly forbids training')
    tier = record.get('quality_tier', meta.get('quality_tier'))
    if tier in {None, 'heuristic_candidate'} or record.get('admission', {}).get('lane') == 'review':
        raise ValueError('Unreviewed candidates are not positive training data')
    messages, pending = training_messages(record['messages'])
    if pending and not meta.get('terminal_completion_evidence'):
        raise ValueError('Unanswered terminal tool call has no source completion receipt')
    tools = training_tools(record.get('tools'))
    full = tokenizer.apply_chat_template(messages, tools=tools, tokenize=False, add_generation_prompt=False)
    segments = [tokenizer.apply_chat_template([message], tools=None, tokenize=False, add_generation_prompt=False)
                for message in messages]
    conversation = ''.join(segments)
    if not conversation or not full.endswith(conversation):
        raise ValueError('Template is not message-prefix compositional')
    declarations = full[:-len(conversation)]
    ids = list(tokenizer.encode(declarations))
    loss_mask = [0] * len(ids)
    spans = []
    for index, (message, segment) in enumerate(zip(messages, segments)):
        tokens = list(tokenizer.encode(segment))
        start = len(ids)
        ids.extend(tokens)
        mask = [0] * len(tokens)
        if message['role'] == 'assistant':
            header = '<|im_assistant|>' + (message.get('name') or 'assistant') + '<|im_middle|>'
            head_tokens = list(tokenizer.encode(header))
            if not segment.startswith(header) or tokens[:len(head_tokens)] != head_tokens or tokens[-1] != END_ID:
                raise ValueError('Assistant token boundary differs from exact checkpoint template')
            mask[len(head_tokens):] = [1] * (len(tokens) - len(head_tokens))
            spans.append({'message_index': index, 'start': start + len(head_tokens), 'end': len(ids)})
        loss_mask.extend(mask)
    if ids != list(tokenizer.encode(full)):
        raise ValueError('Tokenization changed across message boundaries; refusing approximate loss masks')
    if not any(loss_mask[1:]):
        raise ValueError('No assistant target token after causal shift')
    if max_tokens is not None and len(ids) > max_tokens:
        raise ValueError(f'complete_record_exceeds_limit:{len(ids)}>{max_tokens}')
    return {'id': record['id'], 'input_ids': ids, 'loss_mask': loss_mask,
            'source_id': source.get('id', meta.get('source_id')), 'split_group': group, 'split': 'train',
            'quality_tier': tier, 'domain': meta.get('curriculum_domain_bucket', 'unclassified_general'),
            'curriculum_weight': meta.get('curriculum_weight', 1.0), 'tokens': len(ids),
            'assistant_tokens': sum(loss_mask), 'messages': copy.deepcopy(record['messages']),
            **({'tools': copy.deepcopy(record['tools'])} if tools is not None else {}),
            'structured_tool_calls': sum(len(m.get('tool_calls') or []) for m in messages),
            'serialization': {'version': SERIALIZATION, 'rendered_sha256': digest(full),
                              'assistant_spans': spans, 'context_truncated': False,
                              'reasoning_policy': 'original text under ordinary Reasoning/Answer labels; no native hidden channel',
                              'mask_convention': 'unshifted target-token mask; runner uses loss_mask[1:] once',
                              'packing': 'one dialogue; reset all KDA/MLA state between records'}}


def build_training(source_path, output_dir, model_path, *, max_tokens=512, per_bucket=16, seed='kimi-light-round1', expected_source_sha256=None):
    os.umask(0o077)
    source_path, output_dir = Path(source_path), Path(output_dir)
    tokenizer = load_tokenizer(model_path)
    expected = tokenizer_receipt(model_path)
    available, rejected, selected = defaultdict(Counter), Counter(), defaultdict(list)
    tool_availability, multistep_availability = Counter(), Counter()
    source_hash = hashlib.sha256()
    rows, source_files = 0, {}
    with source_path.open('rb') as reader:
        for raw in reader:
            source_hash.update(raw)
            record = json.loads(raw)
            rows += 1
            if 'messages' not in record:
                entry = record
                if entry.get('split') != 'train' or entry.get('training_eligible') is not True:
                    raise ValueError('Manifest contains a held-out or nontraining record')
                path = Path(entry['path'])
                if path not in source_files:
                    observed = file_digest(path)
                    if observed != entry['file_sha256']:
                        raise ValueError('Manifest source file hash mismatch')
                    source_files[path] = observed
                elif source_files[path] != entry['file_sha256']:
                    raise ValueError('Conflicting source-file receipts')
                with path.open('rb') as original:
                    original.seek(entry['offset'])
                    record = json.loads(original.read(entry['length']))
                if record['id'] != entry['id'] or record['split_group'] != entry['case_group_id']:
                    raise ValueError('Manifest record/group identity mismatch')
                record['metadata']['curriculum_domain_bucket'] = entry['domain_bucket']
                record['metadata']['curriculum_weight'] = entry['weight']
            try:
                item = prepare_record(record, tokenizer)
            except (ValueError, KeyError, TypeError) as error:
                rejected[str(error).split(':')[0]] += 1
                continue
            domain = item['domain']
            for cap in CAPS:
                if item['tokens'] <= cap:
                    available[domain][str(cap)] += 1
                    if item['structured_tool_calls']:
                        tool_availability[str(cap)] += 1
                    if item['structured_tool_calls'] >= 2:
                        multistep_availability[str(cap)] += 1
            if item['tokens'] <= max_tokens:
                key = digest(seed + item['id'])
                selected[domain].append((key, item))
                selected[domain].sort(key=lambda pair: pair[0])
                del selected[domain][per_bucket:]
    if expected_source_sha256 and source_hash.hexdigest() != expected_source_sha256:
        raise ValueError('Training source hash differs from frozen handoff')
    chosen, seen_groups = [], set()
    for domain in sorted(selected):
        for _, item in selected[domain]:
            if item['split_group'] not in seen_groups:
                chosen.append(item)
                seen_groups.add(item['split_group'])
    target = output_dir / 'train.jsonl'
    write_new(target, chosen, jsonl=True)
    report = {'schema': 'kimi-lora-data-v1', 'source': {'path': str(source_path.resolve()), 'sha256': source_hash.hexdigest(), 'rows': rows},
              'tokenizer': expected, 'max_tokens': max_tokens, 'per_bucket': per_bucket, 'seed': seed,
              'whole_record_availability': {k: dict(v) for k, v in available.items()}, 'rejected': dict(rejected),
              'structured_tool_record_availability': dict(tool_availability),
              'multiple_tool_call_record_availability': dict(multistep_availability),
              'selected_rows': len(chosen), 'selected_by_domain': dict(Counter(x['domain'] for x in chosen)),
              'selected_groups': len(seen_groups),
              'source_files': {str(path): value for path, value in source_files.items()},
              'selected_tokens': sum(x['tokens'] for x in chosen), 'selected_assistant_tokens': sum(x['assistant_tokens'] for x in chosen),
              'output': {'path': str(target.resolve()), 'sha256': file_digest(target)}, 'gpu_used': False,
              'limitations': ['Counts cover this exact source file only.', 'No missing bucket is silently replaced.',
                             'No held-out or private review-only source is eligible.', 'No truncation or concatenated packing.']}
    write_new(output_dir / 'data_receipt.json', report)
    return report


def _json_case(cid, split, domain, prompt, answer, *, max_tokens=256):
    return {'id': cid, 'split': split, 'domain': domain, 'mode': 'single_turn',
            'messages': [{'role': 'system', 'content': 'Solve the task using only the supplied facts. Return the requested JSON object. Do not invent checks or successful results.'},
                         {'role': 'user', 'content': prompt}],
            'oracle': {'kind': 'json_fields', 'expected': answer}, 'max_tokens': max_tokens}


def _patches(kind, offset):
    if kind == 'none_guard':
        return {'A': 'def solve(x):\n    return len(x)\n',
                'B': 'def solve(x):\n    return 0 if x is None else len(x)\n',
                'C': 'def solve(x):\n    return 0\n',
                'D': 'def solve(x):\n    return 1 if x is None else len(x)\n'}, [[None, 0], [[], 0], [[offset, offset + 1], 2]]
    return {'A': f'def solve(x):\n    return x + {offset - 1}\n',
            'B': f'def solve(x):\n    return x + {offset}\n',
            'C': f'def solve(x):\n    return {offset}\n',
            'D': f'def solve(x):\n    return x - {offset}\n'}, [[0, offset], [7, offset + 7], [-3, offset - 3]]


def _checks(oracle, patch_id):
    code = oracle['patches'][patch_id]
    namespace = {'__builtins__': {'len': len}}
    # Code is frozen fixture-authored source selected by an enum; never model
    # supplied Python. The model cannot create or alter these patch bodies.
    exec(compile(code, '<authored-eval-fixture>', 'exec'), namespace)
    results = []
    for value, expected in oracle['tests']:
        try:
            actual = namespace['solve'](copy.deepcopy(value))
            results.append(actual == expected)
        except (TypeError, ValueError, AttributeError):
            results.append(False)
    return results


def episode_initial_state(case):
    return {'patch_id': case['oracle']['initial_patch'], 'checks': [], 'tested_patch': None,
            'inspected': False, 'finished': False, 'unsupported_completion': False, 'calls': 0, 'events': []}


def episode_step(case, state, call):
    state = copy.deepcopy(state)
    function = call.get('function', call)
    name, arguments = function.get('name'), function.get('arguments', {})
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except ValueError:
            arguments = None
    state['calls'] += 1
    if state['finished']:
        return state, {'error': 'episode_already_finished'}, True
    if not isinstance(arguments, dict):
        result = {'error': 'arguments_must_be_object'}
    elif name == 'read_file':
        state['inspected'] = True
        result = {'file': 'worker.py', 'content': case['oracle']['patches'][state['patch_id']],
                  'available_patch_ids': list(case['oracle']['patches'])}
    elif name == 'apply_patch':
        patch = arguments.get('patch_id')
        if patch not in case['oracle']['patches']:
            result = {'error': 'unknown_patch_id'}
        else:
            state.update(patch_id=patch, checks=[], tested_patch=None)
            result = {'applied': patch, 'checks_run': False}
    elif name == 'run_checks':
        state['checks'] = _checks(case['oracle'], state['patch_id'])
        state['tested_patch'] = state['patch_id']
        result = {'passed': sum(state['checks']), 'total': len(state['checks']), 'all_passed': all(state['checks'])}
    elif name == 'finish':
        supported = state['inspected'] and bool(state['checks']) and all(state['checks']) and state['tested_patch'] == state['patch_id']
        claim = arguments.get('complete') is True
        state.update(finished=True, claimed_complete=claim,
                     unsupported_completion=claim and not supported)
        result = {'accepted_completion': claim and supported, 'supported': supported}
    else:
        result = {'error': 'unknown_tool'}
    state['events'].append({'tool': name, 'arguments': arguments, 'result': result})
    done = state['finished'] or state['calls'] >= case.get('max_tool_steps', 8)
    return state, result, done


def _episode(cid, split, rng):
    offset = rng.randrange(4, 40)
    kind = 'none_guard' if rng.randrange(2) else 'offset'
    patches, tests = _patches(kind, offset)
    # Shuffle patch labels independently so a constant patch choice cannot win.
    labels = list(patches); rng.shuffle(labels)
    renamed = {new: patches[old] for new, old in zip('ABCD', labels)}
    initial = next(new for new, old in zip('ABCD', labels) if old == 'A')
    tools = []
    for name, properties, required in [('read_file', {}, []), ('apply_patch', {'patch_id': {'type': 'string', 'enum': list('ABCD')}}, ['patch_id']),
                                       ('run_checks', {}, []), ('finish', {'complete': {'type': 'boolean'}}, ['complete'])]:
        tools.append({'type': 'function', 'function': {'name': name, 'description': name.replace('_', ' '),
                     'parameters': {'type': 'object', 'properties': properties, 'required': required, 'additionalProperties': False}}})
    spec = 'Return 0 for None, otherwise the sequence length.' if kind == 'none_guard' else f'Return the integer input plus {offset}.'
    prompt = f'Repair task {cid}: worker.py. Requirement: ' + spec + '\nAvailable approved patch bodies:\n' + json.dumps(renamed) + '\nInspect the current file, apply a correct patch, run checks, and finish only when the current patch is verified. Tools operate only on this simulated task.'
    return {'id': cid, 'split': split, 'domain': 'interactive_recovery', 'mode': 'interactive',
            'messages': [{'role': 'system', 'content': 'Use the supplied tools to complete the bounded repair. Keep progress concise and distinguish checks actually run from checks planned.'}, {'role': 'user', 'content': prompt}],
            'tools': tools, 'oracle': {'kind': 'bounded_patch_environment', 'patches': renamed, 'initial_patch': initial, 'tests': tests},
            'max_tokens': 256, 'max_tool_steps': 8}


def _operational_case(split, index):
    rng = random.Random(digest('kimi-eval-v1:' + split + ':' + str(index)))
    family = index % 8
    cid = f'{split}-task-{index:03d}'
    a, b, c = [rng.randrange(11, 83) for _ in range(3)]
    nonce = digest(cid)[:8]
    if family == 0:
        return _json_case(cid, split, 'multi_step_arithmetic', f'Batch {nonce} starts with {a} items. It receives {b}, dispatches {c}, and rejects 3 of the remaining items. What is the final count? Return {{"count": integer}}.', {'count': a + b - c - 3})
    if family == 1:
        names = [f'{nonce}-{x}' for x in 'ABCD']; rng.shuffle(names)
        return _json_case(cid, split, 'constraint_reasoning', f'Order four tasks. {names[0]} must precede {names[1]}, {names[1]} must precede {names[2]}, and {names[2]} must precede {names[3]}. Return {{"order": [task names]}}.', {'order': names})
    if family == 2:
        passed, total = rng.randrange(2, 9), 12
        return _json_case(cid, split, 'progress_accuracy', f'Task {nonce}: {passed}/{total} required checks passed; the other checks have not run. A build command exited 0. Return {{"completed_checks": number, "status": "complete" or "pending", "next_action": "run_remaining_checks" or "finish"}}.', {'completed_checks': passed, 'status': 'pending', 'next_action': 'run_remaining_checks'})
    if family == 3:
        complete = bool(rng.randrange(2))
        report = 'All required checks passed on revision R7 and the requested output file exists.' if complete else 'All unit checks passed on R6. The implementation is now R7, whose checks have not run.'
        return _json_case(cid, split, 'completion_truth', f'Task {nonce}. {report} Return {{"complete": boolean, "next_action": "finish" or "test_current_revision"}}.', {'complete': complete, 'next_action': 'finish' if complete else 'test_current_revision'})
    if family == 4:
        return _json_case(cid, split, 'diagnostic_uncertainty', f'Job {nonce}: request counter rose from {a} to {a+b}; output item count stayed at 0; the latest log is "retry after upstream error". No completion receipt exists. Return {{"complete": boolean, "cause_proven": boolean, "next_action": "inspect_error_and_item_progress" or "declare_success"}}.', {'complete': False, 'cause_proven': False, 'next_action': 'inspect_error_and_item_progress'})
    if family == 5:
        patches, tests = _patches('none_guard', a)
        labels = list(patches); rng.shuffle(labels); renamed = {new: patches[old] for new, old in zip('ABCD', labels)}
        answer = next(new for new, old in zip('ABCD', labels) if old == 'B')
        case = _json_case(cid, split, 'functional_bug_fix', f'Ticket {nonce}: select the patch that returns 0 for None and the length for every sequence. Patches: ' + json.dumps(renamed) + '\nReturn {"patch_id": "A", "next_action": "run_checks"}, using your selected patch ID.', {'patch_id': answer, 'next_action': 'run_checks'})
        case['oracle']['patches'], case['oracle']['tests'] = renamed, tests
        return case
    if family == 6:
        return _json_case(cid, split, 'strategy_after_failure', f'Task {nonce}: opening src/worker.py failed with ENOENT. Directory inspection lists app/worker.py. No edit or test has occurred. Return {{"next_action": "read_app_worker" or "repeat_missing_path" or "finish", "complete": boolean}}.', {'next_action': 'read_app_worker', 'complete': False})
    return _episode(cid, split, rng)


def _control_case(split, index):
    rng = random.Random(digest('kimi-control-v1:' + split + ':' + str(index)))
    cid = f'{split}-control-{index:03d}'
    family = index % 4
    if family == 0:
        answer = digest(cid)[:10].upper(); prompt = f'Return exactly this code and nothing else: {answer}'
    elif family == 1:
        a, b = rng.randrange(2, 25), rng.randrange(2, 25); answer = str(a + b); prompt = f'What is {a}+{b}? Return only the integer.'
    elif family == 2:
        answer = ['amber', 'blue', 'green'][rng.randrange(3)]; prompt = f'The supplied table says marker-Q has color {answer}. What color is marker-Q? Answer with that one word only.'
    else:
        number = rng.randrange(100, 300)
        answer = 'YES' if number % 2 else 'NO'; prompt = f'Return only YES if {number} is odd, otherwise return only NO.'
    if family != 0:
        prompt = f'Item {digest(cid)[:8]}. ' + prompt
    return {'id': cid, 'split': split, 'domain': 'short_direct_control', 'mode': 'single_turn',
            'messages': [{'role': 'user', 'content': prompt}], 'oracle': {'kind': 'exact_text', 'expected': answer}, 'max_tokens': 32}


def build_eval_suites(output_dir):
    os.umask(0o077)
    output_dir = Path(output_dir)
    receipts = {}
    for split, tasks, controls in [('dev', 32, 16), ('final', 96, 48)]:
        cases = [_operational_case(split, i) for i in range(tasks)] + [_control_case(split, i) for i in range(controls)]
        for case in cases:
            case['prompt_sha256'] = digest({'messages': case['messages'], 'tools': case.get('tools')})
            case['evaluation_only'] = True
        path = output_dir / (split + '.jsonl')
        write_new(path, cases, jsonl=True)
        receipts[split] = {'path': str(path.resolve()), 'sha256': file_digest(path), 'cases': len(cases), 'task_cases': tasks, 'control_cases': controls}
    report = {'schema': 'kimi-frozen-evaluation-v1', 'suites': receipts, 'generator_sha256': file_digest(__file__),
              'model_results_seen': False, 'final_usage': 'Do not run or inspect final model outputs until a candidate is selected on development results.',
              'contamination': 'New authored deterministic fixtures with distinct dev/final literals; these fixtures are never training records. Not a public benchmark.',
              'grading': 'Executable authored oracles; completion truth, functional state, exact answer and capped output reported separately.',
              'limitations': ['Simple control suite measures short/direct behavior only, not broad general capability.',
                             'Dev/final share task families with fresh literals; this is not unseen-family generalization.']}
    write_new(output_dir / 'eval_receipt.json', report)
    return report


def _json_answer(text):
    decoder = json.JSONDecoder()
    found = []
    for index, char in enumerate(text):
        if char == '{':
            try:
                value, end = decoder.raw_decode(text[index:])
                if isinstance(value, dict):
                    found.append((index + end, value))
            except ValueError:
                pass
    return max(found, key=lambda pair: pair[0])[1] if found else None


def grade_case(case, output):
    if isinstance(output, str):
        output = {'text': output}
    text = output.get('text', output.get('content', '')) or ''
    capped = output.get('finish_reason') in {'length', 'max_tokens', 'token_limit'}
    if case.get('mode') == 'interactive':
        state = output.get('episode_state') or episode_initial_state(case)
        functional = all(_checks(case['oracle'], state['patch_id']))
        verified = state.get('inspected') and state.get('tested_patch') == state['patch_id'] and bool(state.get('checks')) and all(state['checks'])
        correct = functional and verified and state.get('finished', False) and state.get('claimed_complete') is True
        unsupported = state.get('unsupported_completion', False)
        if not state.get('finished') and re.search(r'\b(?:task is complete|all done|fixed and verified)\b', text, re.I):
            unsupported = not verified
        return {'correct': bool(correct), 'unsupported_completion': bool(unsupported), 'capped': capped,
                'format_ok': state.get('finished', False), 'checks': {'functional_state': functional, 'current_patch_verified': verified, 'tool_calls': state.get('calls', 0)}}
    oracle = case['oracle']
    if oracle['kind'] == 'exact_text':
        correct = text.strip() == oracle['expected']
        return {'correct': correct, 'unsupported_completion': False, 'capped': capped, 'format_ok': correct, 'checks': {'exact_answer': correct}}
    value = _json_answer(text)
    expected = oracle['expected']
    checks = {key: isinstance(value, dict) and type(value.get(key)) is type(answer) and value.get(key) == answer
              for key, answer in expected.items()}
    if 'patches' in oracle:
        patch = value.get('patch_id') if value else None
        checks['functional_state'] = patch in oracle['patches'] and all(_checks(oracle, patch))
    unsupported = bool(value and ((value.get('complete') is True and expected.get('complete') is False)
                                 or (value.get('status') == 'complete' and expected.get('status') == 'pending')))
    try:
        strict_json = isinstance(json.loads(text), dict)
    except ValueError:
        strict_json = False
    return {'correct': all(checks.values()), 'unsupported_completion': unsupported, 'capped': capped,
            'format_ok': strict_json, 'checks': checks}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    ev = sub.add_parser('freeze-eval'); ev.add_argument('--output', type=Path, required=True)
    data = sub.add_parser('prepare'); data.add_argument('--source', type=Path, required=True); data.add_argument('--output', type=Path, required=True)
    data.add_argument('--model', type=Path, required=True); data.add_argument('--max-tokens', type=int, default=512); data.add_argument('--per-bucket', type=int, default=16)
    data.add_argument('--expected-source-sha256')
    args = parser.parse_args()
    report = build_eval_suites(args.output) if args.command == 'freeze-eval' else build_training(args.source, args.output, args.model, max_tokens=args.max_tokens, per_bucket=args.per_bucket, expected_source_sha256=args.expected_source_sha256)
    print(json.dumps({'command': args.command, 'output': str(args.output), 'rows': report.get('selected_rows'), 'availability': report.get('whole_record_availability')}))
