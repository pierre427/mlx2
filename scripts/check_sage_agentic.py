#!/usr/bin/env python3
"""Bounded, source-bound Sage orchestration/verification feature probe.

A dedicated ownership unit starts the server. This probe never starts a model
itself and refuses inference without matching paired GPU locks and CPG lease.
This is feature smoke evidence, not a scientific-quality benchmark or a claim
that a previous frozen qualification covered new request options.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from types import SimpleNamespace

from mlx2.agentic_chat import (
    AGENTIC_DEFAULTS,
    SPAWN_DESCRIPTION,
    SPAWN_PARAMETERS,
    validate_spawn_tasks,
)
from mlx2.desktop_agents import Agents
from mlx2.serving import runtime_identity


def probe(url, output, *, source_revision, expected_checkpoint, timeout=1800):
    output.mkdir(parents=True, exist_ok=False)
    calls = []
    def request(path, body=None, *, expected=200):
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(url.rstrip('/') + path, data=data,
            headers={"Content-Type": "application/json"} if data else {})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                code = response.status; raw = response.read(8_000_001)
        except urllib.error.HTTPError as error:
            code = error.code; raw = error.read(8_000_001)
        if len(raw) > 8_000_000: raise ValueError('Response exceeds probe evidence limit')
        value = json.loads(raw)
        calls.append({"path": path, "request": body, "status": code, "response": value})
        (output / 'http.json').write_text(json.dumps(calls, indent=2) + '\n')
        if code != expected: raise AssertionError(f'{path}: expected {expected}, received {code}')
        return value
    status = request('/v1/status')
    fingerprint = status.get('artifact')
    if isinstance(fingerprint, dict): fingerprint = fingerprint.get('fingerprint')
    checkpoint = str(status['model']) + '@sha256:' + str(fingerprint)
    report = {"schema": "mlx2.sage-agentic-feature-smoke.v1", "checkpoint": checkpoint,
        "source_revision": source_revision, "server_advertised_source_revision": (status.get('runtime') or {}).get('source_revision'),
        "scope": "new verification/template/orchestration features only; prior qualification unchanged",
        "profile": AGENTIC_DEFAULTS, "started": time.time(), "passed": False, "checks": []}
    def check(name, condition):
        report['checks'].append({"name": name, "passed": bool(condition)})
        if not condition: raise AssertionError(name)
    manager = None
    try:
        local_revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=Path(__file__).resolve().parents[1], text=True).strip()
        local_runtime = runtime_identity()
        report['server_runtime'] = status.get('build_runtime') or status.get('runtime')
        check('exact_checkpoint', checkpoint == expected_checkpoint)
        check('exact_source_revision', local_revision == source_revision)
        check('exact_runtime_source_hash', report['server_runtime']['source_sha256'] == local_runtime['source_sha256'])
        check('256k_server_context', status.get('max_context', (status.get('settings') or {}).get('max_context', 0)) >= 262144)
        check('verification_advertised', 'verification' in (status.get('chat_template_options') or {}))
        system = {"role": "system", "content": "Solve the assigned task. Check evidence and report uncertainty. Never invent tool results."}
        messages = [system, {"role": "user", "content": "Return OK."}]
        base = {"messages": messages, "enable_thinking": True}
        plain = request('/tokenize', base)
        off = request('/tokenize', {**base, "chat_template_kwargs": {"verification": False}})
        on = request('/tokenize', {**base, "chat_template_kwargs": {"verification": True}})
        rendered = request('/apply-template', {**base, "chat_template_kwargs": {"verification": True}})
        from mlx2.agentic_chat import VERIFICATION_TRIGGER
        check('exact_verification_trigger', json.dumps(rendered).count(VERIFICATION_TRIGGER) == 1)
        # The endpoint may expose count only. Exact byte/token prompt parity is
        # also covered by the model-template CPU test; do not invent token IDs.
        check('verification_off_unchanged', plain == off)
        check('verification_changes_prompt', plain != on)
        for invalid in ({"messages": messages[1:], "chat_template_kwargs": {"verification": True}},
                        {**base, "chat_template_kwargs": {"verification": "true"}},
                        {**base, "enable_thinking": False, "chat_template_kwargs": {"verification": True}}):
            request('/v1/chat/completions', {**invalid, "max_tokens": 32}, expected=400)
        tool = {"type": "function", "function": {"name": "spawn_subagent", "description": SPAWN_DESCRIPTION, "parameters": SPAWN_PARAMETERS}}
        prompt = ('Use spawn_subagent exactly once for these two independent tasks. '
            'Task one: from context "ALPHA-73: 17 + 24", calculate the sum and include ALPHA-73 in the answer. '
            'Task two: from context "BETA-29: 90 - 32", calculate the difference and include BETA-29 in the answer. '
            'Give each child only its own context. Expected output: identifier and integer. Do not solve them yourself.')
        body = {"messages": [system, {"role": "user", "content": prompt}], "stream": False,
            "max_tokens": 65536, "thinking_budget": 32768, "context_limit": 262144,
            "enable_thinking": True, "chat_template_kwargs": {"verification": True},
            "temperature": 0, "tools": [tool], "tool_choice": {"type": "function", "function": {"name": "spawn_subagent"}}}
        response = request('/v1/chat/completions', body)
        message = response['choices'][0]['message']; tool_calls = message.get('tool_calls') or []
        check('canonical_spawn_called_once', len(tool_calls) == 1 and tool_calls[0]['function']['name'] == 'spawn_subagent')
        tasks = validate_spawn_tasks(json.loads(tool_calls[0]['function']['arguments'])['tasks'])
        check('two_tasks', len(tasks) == 2)
        check('isolated_task_context', 'BETA-29' not in json.dumps(tasks[0]) and 'ALPHA-73' not in json.dumps(tasks[1]))
        target = {"url": url, "model": status['model'], "checkpoint": checkpoint,
                  "source_revision": source_revision, "context_limit": 262144,
                  "output_tokens": 65536, "chat_options": {**AGENTIC_DEFAULTS, "verification": True}}
        controller = SimpleNamespace(home=output, preferences={}, dispatch=lambda selection: target)
        manager = Agents(controller)
        receipt = manager.dispatch({"command": "spawn_subagent", "target": {"url": url}, "tasks": tasks})
        job = manager.jobs[receipt['job_id']]; job['thread'].join(timeout)
        check('children_finished', job['status'] == 'completed')
        results = job['results']
        check('ordered_child_answers', all(r['status'] == 'completed' and identifier in r['answer']
            and re.search(r'(?<![0-9])' + re.escape(answer) + r'(?![0-9])', r['answer'])
            for r, identifier, answer in zip(results, ('ALPHA-73', 'BETA-29'), ('41', '58'))))
        history = [*body['messages'], message, {"role": "tool", "tool_call_id": tool_calls[0]['id'], "content": json.dumps(results)}]
        final = request('/v1/chat/completions', {**body, "messages": history, "tool_choice": "none"})
        content = final['choices'][0]['message'].get('content') or ''
        check('parent_synthesizes_tool_results', all(x in content for x in ('ALPHA-73', 'BETA-29'))
            and all(re.search(r'(?<![0-9])' + x + r'(?![0-9])', content) for x in ('41', '58')))
        report['passed'] = True
    finally:
        if manager: manager.close()
        report['finished'] = time.time()
        (output / 'feature-smoke.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--url', required=True); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--source-revision', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--session', required=True); p.add_argument('--label', required=True)
    p.add_argument('--cpg-task', required=True); p.add_argument('--cpg-owner-lock', type=Path, required=True)
    p.add_argument('--generation', type=int, required=True)
    args = p.parse_args()
    from run_single_model_smoke_ladder import validate_campaign_ownership
    validate_campaign_ownership(args.session, args.label, cpg_task=args.cpg_task,
        cpg_owner_lock=args.cpg_owner_lock, cpg_generation=args.generation)
    result = probe(args.url, args.output, source_revision=args.source_revision, expected_checkpoint=args.checkpoint)
    print(json.dumps({k: result[k] for k in ('checkpoint', 'source_revision', 'passed', 'checks')}))


if __name__ == '__main__': main()
