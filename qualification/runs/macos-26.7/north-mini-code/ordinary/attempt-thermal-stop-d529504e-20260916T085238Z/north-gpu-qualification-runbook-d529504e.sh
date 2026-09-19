#!/bin/zsh
set -euo pipefail

# Reviewed no-touch runbook. Do not execute without the root coordinator's
# exclusive GPU grant. North Mini Code is ordinary-only: this script makes no
# optimized-arm, alternating-arm, MTP, DFlash, EAGLE, or PLD claim.

REPO=~/Desktop/mlx2
MODEL=~/mlx-models/North-Mini-Code-1.0-mlx-4bit
EXPECTED_SOURCE=d529504ee25593c8bbd051aea6a163fea081b1912bb5c79eb3465a6501b1d6cd
EXPECTED_QUALIFIER=8d32f490519f91e0e4b96f0e6b60d05b088aba41930b1b5df24eb0292fcfa8c1
EXPECTED_MANIFEST=b4cae832cfe3e783e2596458a3cdc31eac8884252da72a3504bc8c68392f8ff6
EXPECTED_ARTIFACT=6c88a4d3a5387abe2d97ac3d5eb70e484fb33615bb0c0f2ff8e7ef22b442f2e4
EXPECTED_NATIVE=30335939714520c3306cfa58597100aae08bbeabdc60a4bc8c35c32c607450a8
THERMAL_BARRIER_LAUNCHER=/private/tmp/mlx2-north-d529504e-20260916/qualify-serving-thermal-barrier.py
EXPECTED_THERMAL_BARRIER=e12ea8a08d8ccba2d3084b09f8a1a2a2d8eae2feb10b2464f6a713b51f76ceba
PORT=8288
TAG=${EXPECTED_SOURCE[1,8]}-${EXPECTED_QUALIFIER[1,8]}
RUNROOT=$REPO/qualification/runs/macos-26.7/north-mini-code/ordinary
STATE=/tmp/mlx2-qualification/north-server-$TAG.json
CACHE=/tmp/mlx2-qualification/north-ordinary-apc-$TAG
QUALIFIED_CACHE=/tmp/mlx2-qualification/north-qualified-apc-$TAG
LOG=$RUNROOT/server-$TAG.log
WATCHLOG=$RUNROOT/watchdog-$TAG.log
STATUS0=$RUNROOT/status-initial-$TAG.json
STAGED_RECEIPT=$RUNROOT/route-qualification.candidate-$TAG.json
RECEIPT=$RUNROOT/route-qualification.json
QUALIFIED_STATUS=$RUNROOT/status-qualified-$TAG.json
GENERATED_MANIFEST=/tmp/mlx2-qualification/north-four-model-$TAG.generated.json
BOUND_ACTIVATOR=/tmp/mlx2-qualification/north-bound-activator-$TAG.py
PROMPTS=/tmp/mlx2-qualification/north-context-prompts-$TAG
CONTEXT_REPORT=$RUNROOT/context-ladder-$TAG.json
BATCH_REPORT=$RUNROOT/batch-stress-b20-$TAG.json
SPOMIN_REPORT=$RUNROOT/spomin-20x20-$TAG.json
FAILURE=$RUNROOT/watchdog-failure-$TAG.txt
THERMAL_BARRIER=$RUNROOT/thermal-barrier-$TAG.json
WATCHDOG_PID=

[[ ${MLX2_GPU_GRANT:-} == north-exclusive ]] || {
  print -u2 -- 'Set MLX2_GPU_GRANT=north-exclusive only after the root grants the North GPU window.'
  exit 69
}

cd $REPO
mkdir -p $RUNROOT /tmp/mlx2-qualification

source_hash() {
  PYTHONPATH=src .venv/bin/python - <<'PY'
import hashlib
from pathlib import Path
root = Path('src/mlx2')
digest = hashlib.sha256()
for path in sorted(root.rglob('*.py')):
    digest.update(str(path.relative_to(root)).encode())
    digest.update(path.read_bytes())
print(digest.hexdigest())
PY
}

stop_owned() {
  if [[ -f $STATE ]]; then
    PYTHONPATH=scripts .venv/bin/python - $STATE <<'PY'
from pathlib import Path
import sys
from activate_qualification_arm import stop_owned
stop_owned(Path(sys.argv[1]))
PY
  fi
}

port_clear() {
  ! /usr/sbin/lsof -nP -iTCP:$PORT -sTCP:LISTEN >/dev/null 2>&1
}

wait_port_down() {
  local deadline=$(( $(/bin/date +%s) + 60 ))
  while ! port_clear; do
    (( $(/bin/date +%s) < deadline )) || {
      print -u2 -- "port $PORT remained bound after owned server shutdown"
      return 1
    }
    /bin/sleep 1
  done
}

assert_new_owned_listener() {
  local launched_after=$1
  PYTHONPATH=src .venv/bin/python - $STATE $PORT $launched_after <<'PY'
import json, subprocess, sys
from pathlib import Path

state_path, port, launched_after = Path(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3])
state = json.loads(state_path.read_text())
owned_pid = int(state['pid'])
assert float(state['started_at']) >= launched_after, (state['started_at'], launched_after)
assert state['cwd'] == str(Path.cwd().resolve())
assert '-m mlx2.server' in state['command']
assert f'--port {port}' in state['command']
listeners = subprocess.run(
    ['/usr/sbin/lsof', '-nP', f'-iTCP:{port}', '-sTCP:LISTEN', '-t'],
    capture_output=True, text=True, check=True,
).stdout.split()
assert listeners == [str(owned_pid)], (listeners, owned_pid)
actual = subprocess.run(
    ['/bin/ps', '-p', str(owned_pid), '-o', 'command='],
    capture_output=True, text=True, check=True,
).stdout.strip()
assert actual == state['command'], (actual, state['command'])
PY
}

swap_zero() {
  /usr/sbin/sysctl vm.swapusage | /usr/bin/grep -q 'used = 0.00M'
}

thermal_nominal() {
  local therm
  therm=$(/usr/bin/pmset -g therm)
  [[ $therm == *'No thermal warning'* && $therm == *'No performance warning'* && $therm == *'No CPU power status'* ]]
}

capture_health() {
  local health http_status
  health=$(/usr/bin/curl -fsS --max-time 30 http://127.0.0.1:$PORT/health)
  http_status=$(/usr/bin/curl -fsS --max-time 30 http://127.0.0.1:$PORT/v1/status)
  {
    /bin/date -u '+%Y-%m-%dT%H:%M:%SZ'
    /usr/sbin/sysctl vm.swapusage
    /usr/bin/pmset -g therm
    print -r -- $health
    print -r -- $http_status
  } >>$WATCHLOG 2>&1
  HEALTH_JSON=$health STATUS_JSON=$http_status PYTHONPATH=src .venv/bin/python - <<'PY'
import json, os
health = json.loads(os.environ['HEALTH_JSON'])
status = json.loads(os.environ['STATUS_JSON'])
assert health.get('status') == 'ok'
assert status.get('healthy') is True
PY
}

validate_thermal_barrier() {
  local sidecar=$1 profile=$2
  SIDECAR=$sidecar PROFILE=$profile EXPECTED_SOURCE=$EXPECTED_SOURCE \
  EXPECTED_QUALIFIER=$EXPECTED_QUALIFIER EXPECTED_MANIFEST=$EXPECTED_MANIFEST \
  PYTHONPATH=scripts .venv/bin/python - <<'PY'
import json, os
from pathlib import Path
from run_qualification_matrix import thermally_stable

p = Path(os.environ['SIDECAR'])
assert p.is_file(), p
r = json.loads(p.read_text())
assert r['schema'] == 'mlx2.qualifier-pre-long-thermal-barrier.v1'
assert r['gate_fired'] == 1
assert r['expected_source_sha256'] == os.environ['EXPECTED_SOURCE']
assert r['expected_profile'] == os.environ['PROFILE']
assert r['qualifier_sha256'] == os.environ['EXPECTED_QUALIFIER']
assert r['manifest_sha256'] == os.environ['EXPECTED_MANIFEST']
policy = r['policy']
tail = r['stable_tail']
assert int(policy['required_thermal_state']) == 0
assert int(policy['sample_interval_seconds']) == 15
assert len(tail) == int(policy['consecutive_samples']) == 3
assert all(sample['thermal_state'] == 0 for sample in tail)
assert all(thermally_stable(sample, policy) for sample in tail)
temperatures = [sample['virtual_temperature_c'] for sample in tail]
assert max(temperatures) - min(temperatures) <= float(policy['max_temperature_delta_c'])
for name in ('status_before', 'status_after'):
    status = r[name]
    assert status['healthy'] and status['error'] is None and status['state'] == 'ready'
    assert status['inflight'] == 0 and status['queue_depth'] == 0
    assert status['runtime']['source_sha256'] == os.environ['EXPECTED_SOURCE']
    assert status['profile'] == os.environ['PROFILE']
assert r['status_before']['runtime'] == r['status_after']['runtime']
assert r['status_before']['artifact'] == r['status_after']['artifact']
assert r['status_before']['settings'] == r['status_after']['settings']
PY
}

watchdog() {
  while true; do
    if [[ $(source_hash) != $EXPECTED_SOURCE ]] || ! swap_zero || ! thermal_nominal; then
      /bin/date -u '+watchdog failure %Y-%m-%dT%H:%M:%SZ' >$FAILURE
      stop_owned || true
      return 1
    fi
    if [[ -f $STATE ]]; then
      local started now
      started=$(PYTHONPATH=src .venv/bin/python - $STATE <<'PY'
import json, sys
print(int(json.load(open(sys.argv[1]))['started_at']))
PY
)
      now=$(/bin/date +%s)
      if ! capture_health && (( now - started > 1200 )); then
        /bin/date -u '+watchdog unhealthy server %Y-%m-%dT%H:%M:%SZ' >$FAILURE
        stop_owned || true
        return 1
      fi
    else
      /bin/date -u '+watchdog: no harness-owned server %Y-%m-%dT%H:%M:%SZ' >>$WATCHLOG
    fi
    /bin/sleep 900
  done
}

cleanup() {
  if [[ -n ${WATCHDOG_PID:-} ]]; then
    /bin/kill $WATCHDOG_PID >/dev/null 2>&1 || true
    wait $WATCHDOG_PID >/dev/null 2>&1 || true
  fi
  stop_owned || true
}
trap cleanup EXIT INT TERM

# Fail closed before any model allocation.
[[ $(source_hash) == $EXPECTED_SOURCE ]]
[[ $(/usr/bin/shasum -a 256 scripts/qualify_serving.py | /usr/bin/awk '{print $1}') == $EXPECTED_QUALIFIER ]]
[[ $(/usr/bin/shasum -a 256 qualification/four-model-experiments.json | /usr/bin/awk '{print $1}') == $EXPECTED_MANIFEST ]]
[[ $(/usr/bin/shasum -a 256 $THERMAL_BARRIER_LAUNCHER | /usr/bin/awk '{print $1}') == $EXPECTED_THERMAL_BARRIER ]]
[[ $(/usr/bin/sw_vers -productVersion) == 26.7 ]]
[[ $(/usr/bin/sw_vers -buildVersion) == 25G229 ]]
port_clear
swap_zero
thermal_nominal

PYTHONPATH=src .venv/bin/python - $MODEL $EXPECTED_ARTIFACT <<'PY'
import sys
from mlx2.adapters.north_mini_code import inspect_artifact
artifact = inspect_artifact(sys.argv[1])
assert artifact['identity']['fingerprint'] == sys.argv[2]
assert len(artifact['weight_map']) == 1226
assert len(artifact['identity']['files']) == 4
assert artifact['global_layers'] == 13
assert artifact['sliding_layers'] == 36
assert artifact['sliding_window'] == 4096
assert artifact['supports_native_mtp'] is False
assert artifact['qualification'] == 'pending'
PY

# Candidate server. No execution policy is accepted for this family.
rm -f $FAILURE
rm -rf $CACHE
LAUNCHED_AFTER=$(/bin/date +%s)
.venv/bin/python scripts/activate_qualification_arm.py \
  --state $STATE --log $LOG --cwd $REPO -- \
  /usr/bin/env PYTHONPATH=src .venv/bin/python -u -m mlx2.server \
  --model $MODEL --host 127.0.0.1 --port $PORT --max-context 500000 \
  --max-lanes 20 --max-inflight 40 --cache-bytes 17179869184 \
  --cache-dir $CACHE --ordinary --qualification-mode

for attempt in {1..240}; do
  /usr/bin/curl -fsS --max-time 5 http://127.0.0.1:$PORT/health >/dev/null 2>&1 && break
  /bin/sleep 5
done
assert_new_owned_listener $LAUNCHED_AFTER
/usr/bin/curl -fsS --max-time 30 http://127.0.0.1:$PORT/v1/status >$STATUS0

PYTHONPATH=src .venv/bin/python - $STATUS0 $EXPECTED_SOURCE $EXPECTED_NATIVE $EXPECTED_ARTIFACT <<'PY'
import json, sys
s = json.load(open(sys.argv[1]))
assert s['healthy'] and s['inflight'] == 0 and s['queue_depth'] == 0
assert s['runtime']['source_sha256'] == sys.argv[2]
assert s['runtime']['mlx_native_sha256'] == sys.argv[3]
assert s['artifact'] == sys.argv[4]
assert s['profile'] == 'north-mini-code-apcv2-ordinary'
assert s['qualification'] == 'candidate'
assert s['route_receipt'] == 'candidate_validation'
assert s['settings']['max_context'] == 500000
assert s['http'] == {'max_request_bytes': 16065536}
assert s['settings']['max_lanes'] == 20
assert s['settings']['max_inflight'] == 40
assert s['settings']['cache_bytes'] == 17179869184
assert s['settings']['disk_cache'] is True
assert s['settings']['mtp'] is False
assert s['settings']['speculation'] == 'ordinary'
assert s['settings']['execution_policy'] == {
  'persistent': True, 'num_draft': 0, 'rate_gate': False,
  'prefill_step_size': 2048, 'segment_aware_live_tip': False,
  'segment_aware_cohort_size': 20,
}
assert s['execution']['cache_layout'] == 'north-mini-code-layer-segments-v1'
assert s['execution']['global_layers'] == 13
assert s['execution']['sliding_layers'] == 36
assert s['apcv2']['cow']['active_leases'] == 0
PY

watchdog &
WATCHDOG_PID=$!

rm -f $THERMAL_BARRIER
EXPECTED_SOURCE_SHA=$EXPECTED_SOURCE EXPECTED_PROFILE=north-mini-code-apcv2-ordinary \
THERMAL_BARRIER_SIDECAR=$THERMAL_BARRIER PYTHONPATH=src:scripts .venv/bin/python -u $THERMAL_BARRIER_LAUNCHER \
  --url http://127.0.0.1:$PORT --output $STAGED_RECEIPT \
  --timeout 7200 --quiescence-timeout 30

# The candidate is unusable unless the pre-long gate fired, recorded three
# 15-second state-0 samples, and reasserted idle identity around that barrier.
validate_thermal_barrier $THERMAL_BARRIER north-mini-code-apcv2-ordinary

PYTHONPATH=src .venv/bin/python - $STAGED_RECEIPT $STATUS0 $EXPECTED_QUALIFIER <<'PY'
import json, sys
receipt = json.load(open(sys.argv[1]))
status = json.load(open(sys.argv[2]))
assert receipt['schema'] == 'mlx2.serving-qualification.v1'
assert receipt['passed'] is True
assert receipt['runtime'] == status['runtime']
assert receipt['artifact'] == status['artifact']
assert receipt['settings'] == status['settings']
assert receipt['qualification_harness'] == {
  'schema': 'mlx2.qualification-harness.v1',
  'name': 'scripts/qualify_serving.py',
  'sha256': sys.argv[3],
}
required = {
 'cold_text','warm_prefix','stream','tools','reasoning','stop','batch','context',
 'cancel','recovery','unit_tests','hermes_client','sampling_controls','logprobs',
 'cache_leases','runtime_stable','mixed_warm','quiescence','apcv2_reuse','apcv2_stores',
}
assert all(receipt['checks'].get(name, {}).get('passed') is True for name in required)
assert receipt['final_status']['apcv2']['cow']['active_leases'] == 0
assert receipt['quiescence']['passed'] is True
PY

# Prove the staged immutable receipt is accepted by the production selector
# before it may replace the canonical receipt.
stop_owned
wait_port_down
swap_zero
thermal_nominal
rm -rf $QUALIFIED_CACHE
LAUNCHED_AFTER=$(/bin/date +%s)
.venv/bin/python scripts/activate_qualification_arm.py \
  --state $STATE --log $LOG --cwd $REPO -- \
  /usr/bin/env PYTHONPATH=src .venv/bin/python -u -m mlx2.server \
  --model $MODEL --host 127.0.0.1 --port $PORT --max-context 500000 \
  --max-lanes 20 --max-inflight 40 --cache-bytes 17179869184 \
  --cache-dir $QUALIFIED_CACHE --ordinary --qualification $STAGED_RECEIPT

for attempt in {1..240}; do
  /usr/bin/curl -fsS --max-time 5 http://127.0.0.1:$PORT/health >/dev/null 2>&1 && break
  /bin/sleep 5
done
assert_new_owned_listener $LAUNCHED_AFTER
/usr/bin/curl -fsS --max-time 30 http://127.0.0.1:$PORT/v1/status >$QUALIFIED_STATUS
PYTHONPATH=src .venv/bin/python - $QUALIFIED_STATUS $EXPECTED_SOURCE $EXPECTED_ARTIFACT <<'PY'
import json, sys
s = json.load(open(sys.argv[1]))
assert s['healthy'] and s['qualification'] == 'qualified'
assert s['runtime']['source_sha256'] == sys.argv[2]
assert s['artifact'] == sys.argv[3]
assert s['profile'] == 'north-mini-code-apcv2-ordinary'
assert s['http'] == {'max_request_bytes': 16065536}
assert s['route_receipt'] == (
  'model=cohere2_moe:1.0-ordinary;profile=north-mini-code-apcv2-ordinary;'
  'requested=apc_v2,continuous_batch,layered_cache,prefix_reuse,reasoning,'
  'streaming,text,tools;fidelity=numerically_bounded'
)
assert s['apcv2']['cow']['active_leases'] == 0
PY
stop_owned
wait_port_down
port_clear
swap_zero
thermal_nominal

# Only a completely verified result that the production selector accepted may
# replace the canonical receipt. A failed/interrupted candidate remains tagged.
PYTHONPATH=src .venv/bin/python - \
  $STAGED_RECEIPT $RECEIPT $STATUS0 $QUALIFIED_STATUS $EXPECTED_QUALIFIER <<'PY'
import json, os, sys
import hashlib
from pathlib import Path
source, destination, status_path, selected_status_path = map(Path, sys.argv[1:5])
expected_qualifier = sys.argv[5]
data = source.read_bytes()
receipt = json.loads(data)
status = json.loads(status_path.read_bytes())
selected = json.loads(selected_status_path.read_bytes())
required = {
 'cold_text','warm_prefix','stream','tools','reasoning','stop','batch','context',
 'cancel','recovery','unit_tests','hermes_client','sampling_controls','logprobs',
 'cache_leases','runtime_stable','mixed_warm','quiescence','apcv2_reuse','apcv2_stores',
}
assert receipt['schema'] == 'mlx2.serving-qualification.v1'
assert receipt['passed'] is True
assert receipt['runtime'] == status['runtime']
assert receipt['artifact'] == status['artifact']
assert receipt['settings'] == status['settings']
assert receipt['qualification_harness'] == {
  'schema': 'mlx2.qualification-harness.v1',
  'name': 'scripts/qualify_serving.py',
  'sha256': expected_qualifier,
}
assert all(receipt['checks'].get(name, {}).get('passed') is True for name in required)
assert receipt['quiescence']['passed'] is True
assert receipt['final_status']['apcv2']['cow']['active_leases'] == 0
assert selected['qualification'] == 'qualified'
assert selected['runtime'] == receipt['runtime']
assert selected['artifact'] == receipt['artifact']
assert selected['settings'] == receipt['settings']
assert selected['route_receipt'] == (
  'model=cohere2_moe:1.0-ordinary;profile=north-mini-code-apcv2-ordinary;'
  'requested=apc_v2,continuous_batch,layered_cache,prefix_reuse,reasoning,'
  'streaming,text,tools;fidelity=numerically_bounded'
)
temporary = destination.with_name(destination.name + '.publishing')
with temporary.open('wb') as stream:
    stream.write(data)
    stream.flush()
    os.fsync(stream.fileno())
os.replace(temporary, destination)
directory = os.open(destination.parent, os.O_RDONLY)
try:
    os.fsync(directory)
finally:
    os.close(directory)
assert destination.read_bytes() == data
assert hashlib.sha256(destination.read_bytes()).digest() == hashlib.sha256(data).digest()
PY

# Create a North-only temporary manifest from the canonical receipt-bound
# definition so this lane cannot activate any of the other three models.
cat >$BOUND_ACTIVATOR <<'PY'
#!/usr/bin/env python3
"""Restart a harness-owned arm across an observed port-down barrier."""
import json
from pathlib import Path
import subprocess
import sys
import time

REPO = Path('~/Desktop/mlx2')
sys.path.insert(0, str(REPO / 'scripts'))
from activate_qualification_arm import process_command, stop_owned

args = sys.argv[1:]
state = Path(args[args.index('--state') + 1])
server = args[args.index('--') + 1:]
port = int(server[server.index('--port') + 1])

def listener_pids():
    result = subprocess.run(
        ['/usr/sbin/lsof', '-nP', f'-iTCP:{port}', '-sTCP:LISTEN', '-t'],
        capture_output=True, text=True,
    )
    return result.stdout.split()

stop_owned(state)
deadline = time.monotonic() + 60
while listener_pids():
    if time.monotonic() >= deadline:
        raise RuntimeError(f'port {port} remained bound after owned server shutdown')
    time.sleep(1)

launched_after = time.time()
subprocess.run(
    [sys.executable, str(REPO / 'scripts' / 'activate_qualification_arm.py'), *args],
    cwd=REPO, check=True,
)
deadline = time.monotonic() + 60
while True:
    record = json.loads(state.read_text())
    owned_pid = int(record['pid'])
    listeners = listener_pids()
    if listeners == [str(owned_pid)]:
        break
    if listeners and listeners != [str(owned_pid)]:
        raise RuntimeError(f'port {port} listener {listeners} is not newly owned PID {owned_pid}')
    if time.monotonic() >= deadline:
        raise RuntimeError(f'new owned PID {owned_pid} did not bind port {port}')
    time.sleep(1)

assert float(record['started_at']) >= launched_after
assert record['cwd'] == str(REPO.resolve())
assert process_command(owned_pid) == record['command']
assert '-m mlx2.server' in record['command']
assert f'--port {port}' in record['command']
print(owned_pid)
PY
chmod 700 $BOUND_ACTIVATOR

PYTHONPATH=src .venv/bin/python - \
  qualification/four-model-experiments.json $GENERATED_MANIFEST $RECEIPT $PORT $STATE $LOG $QUALIFIED_CACHE $BOUND_ACTIVATOR <<'PY'
import json, sys
from pathlib import Path
source, output, receipt, port, state, log, cache, bound_activator = sys.argv[1:]
manifest = json.loads(Path(source).read_text())
north = next(model for model in manifest['models'] if model['name'] == 'north-mini-code')
assert north['arm_order_claim'] == 'none' and len(north['arms']) == 1
manifest['models'] = [north]
arm = north['arms'][0]
arm['url'] = f'http://127.0.0.1:{port}'
command = arm['activate_command']
assert command[1] == '~/Desktop/mlx2/scripts/activate_qualification_arm.py'
command[1] = bound_activator
def replace_after(flag, value):
    command[command.index(flag) + 1] = value
replace_after('--state', state)
replace_after('--log', log)
replace_after('--port', port)
replace_after('--cache-dir', cache)
command.remove('--qualification-mode')
command += ['--qualification', receipt]
arm['qualification_receipt'] = {
  'path': receipt,
  'required_checks': ['apcv2_reuse','apcv2_stores','quiescence','cache_leases','runtime_stable'],
  'scope': 'North ordinary-only full serving receipt; exact runtime, artifact, settings and approved harness verified before execution',
}
arm['status_requirements'] += [
  {'path':'qualification','equals':'qualified'},
  {'path':'runtime.source_sha256','equals':'d529504ee25593c8bbd051aea6a163fea081b1912bb5c79eb3465a6501b1d6cd'},
  {'path':'runtime.mlx_native_sha256','equals':'30335939714520c3306cfa58597100aae08bbeabdc60a4bc8c35c32c607450a8'},
  {'path':'artifact','equals':'6c88a4d3a5387abe2d97ac3d5eb70e484fb33615bb0c0f2ff8e7ef22b442f2e4'},
]
Path(output).parent.mkdir(parents=True, exist_ok=True)
Path(output).write_text(json.dumps(manifest, indent=2) + '\n')
PY

# CPU/tokenizer-only calibration, then the authorized live experiment sequence.
.venv/bin/python scripts/build_context_prompts.py \
  --manifest $GENERATED_MANIFEST \
  --output-manifest $GENERATED_MANIFEST.prompts.json \
  --prompt-dir $PROMPTS

.venv/bin/python scripts/run_qualification_matrix.py \
  --manifest $GENERATED_MANIFEST.prompts.json --suite context \
  --output $CONTEXT_REPORT --resume

.venv/bin/python scripts/run_qualification_matrix.py \
  --manifest $GENERATED_MANIFEST.prompts.json --suite batch_stress \
  --output $BATCH_REPORT --resume

.venv/bin/python scripts/run_spomin_20x20.py \
  --manifest $GENERATED_MANIFEST.prompts.json --model north-mini-code \
  --output $SPOMIN_REPORT --resume

# Final unload criteria: request state quiesced, PID gone, port unbound, swap zero,
# thermal state nominal, source/qualifier still frozen, no watchdog failure.
if /usr/bin/curl -fsS --max-time 30 http://127.0.0.1:$PORT/v1/status >/tmp/north-final-status-$TAG.json 2>/dev/null; then
  PYTHONPATH=src .venv/bin/python - /tmp/north-final-status-$TAG.json <<'PY'
import json, sys
s=json.load(open(sys.argv[1]))
assert s['inflight'] == 0 and s['queue_depth'] == 0
assert s['apcv2']['cow']['active_leases'] == 0
PY
fi
stop_owned
wait_port_down
swap_zero
thermal_nominal
[[ $(source_hash) == $EXPECTED_SOURCE ]]
[[ $(/usr/bin/shasum -a 256 scripts/qualify_serving.py | /usr/bin/awk '{print $1}') == $EXPECTED_QUALIFIER ]]
[[ $(/usr/bin/shasum -a 256 $THERMAL_BARRIER_LAUNCHER | /usr/bin/awk '{print $1}') == $EXPECTED_THERMAL_BARRIER ]]
validate_thermal_barrier $THERMAL_BARRIER north-mini-code-apcv2-ordinary
[[ ! -e $FAILURE ]]

echo "North ordinary-only qualification sequence complete: $RUNROOT"
