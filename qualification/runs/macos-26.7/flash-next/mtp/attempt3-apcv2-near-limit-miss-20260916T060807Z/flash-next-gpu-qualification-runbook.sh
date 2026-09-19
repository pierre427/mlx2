#!/bin/bash
set -euo pipefail

ROOT=~/Desktop/mlx2
MODEL=~/mlx-models/Qwen3.8-Flash-Next-MLX-4bit-MTP
POLICY="$ROOT/qualification/policies/flash-next-mtp2.json"
EXPECTED_SOURCE_SHA=84a08d12ee5434553b6dae330b0828cfbb5f34231090ff59142bc284b5220967
EXPECTED_NATIVE_SHA=30335939714520c3306cfa58597100aae08bbeabdc60a4bc8c35c32c607450a8
EXPECTED_QUALIFIER_SHA=8d32f490519f91e0e4b96f0e6b60d05b088aba41930b1b5df24eb0292fcfa8c1
EXPECTED_MANIFEST_SHA=b4cae832cfe3e783e2596458a3cdc31eac8884252da72a3504bc8c68392f8ff6
EXPECTED_POLICY_SHA=900c06e74b4624e15f4a7d8dee7c26ffe35aca3f3428010960c612d7cc68a48f
EXPECTED_ARTIFACT=77a35a9692b49e736bd0daa708d8b14c71c05abb03f81f8c707a1fc1e372bb25
SHORT=${EXPECTED_SOURCE_SHA:0:8}
PORT=8285
RUNROOT="$ROOT/qualification/runs/macos-26.7/flash-next"

cd "$ROOT"
mkdir -p "$RUNROOT"
if [[ "${MLX2_GPU_GRANT:-}" != "flash-next-exclusive" ]]; then
  echo 'Set MLX2_GPU_GRANT=flash-next-exclusive only after root grants the Flash-Next GPU window.' >&2
  exit 69
fi

ACTIVE_PID=""
ACTIVE_MONITOR=""
LAST_STOPPED_PID=""
cleanup_active() {
  if [[ -n "$ACTIVE_MONITOR" ]]; then /bin/kill -TERM "$ACTIVE_MONITOR" 2>/dev/null || true; fi
  if [[ -n "$ACTIVE_PID" ]]; then
    /bin/kill -TERM "$ACTIVE_PID" 2>/dev/null || true
    for _ in $(seq 1 300); do /bin/kill -0 "$ACTIVE_PID" 2>/dev/null || break; /bin/sleep 1; done
  fi
}
trap cleanup_active EXIT INT TERM

runtime_json=$(PYTHONPATH=src .venv/bin/python - <<'PY'
import json
from mlx2.serving import runtime_identity
print(json.dumps(runtime_identity(), sort_keys=True))
PY
)
RUNTIME_JSON="$runtime_json" EXPECTED_SOURCE_SHA="$EXPECTED_SOURCE_SHA" EXPECTED_NATIVE_SHA="$EXPECTED_NATIVE_SHA" .venv/bin/python - <<'PY'
import json, os
r=json.loads(os.environ['RUNTIME_JSON'])
assert r['source_sha256']==os.environ['EXPECTED_SOURCE_SHA'], r
assert r['mlx_native_sha256']==os.environ['EXPECTED_NATIVE_SHA'], r
assert r['macos']=='26.7', r
assert r['python']=='3.12.13', r
assert r['mlx']=='0.32.2.dev20260915+2a817ad94', r
print(json.dumps(r, indent=2, sort_keys=True))
PY

ACTUAL_QUALIFIER_SHA=$(/usr/bin/shasum -a 256 scripts/qualify_serving.py | /usr/bin/awk '{print $1}')
ACTUAL_MANIFEST_SHA=$(/usr/bin/shasum -a 256 qualification/four-model-experiments.json | /usr/bin/awk '{print $1}')
ACTUAL_POLICY_SHA=$(/usr/bin/shasum -a 256 qualification/policies/flash-next-mtp2.json | /usr/bin/awk '{print $1}')
[[ "$ACTUAL_QUALIFIER_SHA" == "$EXPECTED_QUALIFIER_SHA" ]] || { echo 'qualification harness hash changed' >&2; exit 77; }
[[ "$ACTUAL_MANIFEST_SHA" == "$EXPECTED_MANIFEST_SHA" ]] || { echo 'qualification manifest hash changed' >&2; exit 78; }
[[ "$ACTUAL_POLICY_SHA" == "$EXPECTED_POLICY_SHA" ]] || { echo 'Flash-Next policy hash changed' >&2; exit 79; }
PYTHONPATH=src EXPECTED_ARTIFACT="$EXPECTED_ARTIFACT" MODEL="$MODEL" .venv/bin/python - <<'PY'
import os
from pathlib import Path
from mlx2.adapters.flash_next import artifact_identity
identity=artifact_identity(Path(os.environ['MODEL']))
assert identity['fingerprint']==os.environ['EXPECTED_ARTIFACT'], identity
assert (Path(os.environ['MODEL'])/'ple_rows.bin').is_file()
print(identity['fingerprint'])
PY
/usr/bin/shasum -a 256 scripts/qualify_serving.py qualification/four-model-experiments.json qualification/policies/flash-next-mtp2.json >"$RUNROOT/preflight-tool-hashes-$SHORT.txt"
/usr/sbin/sysctl vm.swapusage | tee "$RUNROOT/preflight-swap-$SHORT.txt"
/usr/bin/pmset -g therm | tee "$RUNROOT/preflight-thermal-$SHORT.txt"
PYTHONPATH=scripts .venv/bin/python - <<'PY' | tee "$RUNROOT/preflight-thermal-probe-$SHORT.json"
import json
from run_qualification_matrix import sample_thermal, thermally_stable
s=sample_thermal(); print(json.dumps(s, sort_keys=True)); assert thermally_stable(s, {})
PY
if ! /usr/sbin/sysctl vm.swapusage | /usr/bin/grep -q 'used = 0.00M'; then echo 'swap is nonzero' >&2; exit 70; fi
if /usr/sbin/lsof -nP -iTCP:"$PORT" -sTCP:LISTEN | /usr/bin/grep -q LISTEN; then echo "qualification port $PORT is occupied" >&2; exit 71; fi

wait_ready() {
  local pid=$1 port=$2
  for _ in $(seq 1 1800); do
    /bin/kill -0 "$pid" 2>/dev/null || return 1
    if /usr/bin/curl -fsS "http://127.0.0.1:$port/health" >/dev/null 2>&1; then
      assert_listener_pid "$port" "$pid"
      return 0
    fi
    /bin/sleep 1
  done
  return 1
}

assert_listener_pid() {
  local port=$1 expected_pid=$2
  local listeners
  listeners=$(/usr/sbin/lsof -nP -t -iTCP:"$port" -sTCP:LISTEN 2>/dev/null | /usr/bin/sort -u || true)
  [[ "$listeners" == "$expected_pid" ]] || {
    echo "port $port listener identity mismatch: expected PID $expected_pid, observed ${listeners:-none}" >&2
    exit 83
  }
}

port_down_pid_barrier() {
  local port=$1 stopped_pid=${2:-}
  for _ in $(seq 1 300); do
    local listeners
    listeners=$(/usr/sbin/lsof -nP -t -iTCP:"$port" -sTCP:LISTEN 2>/dev/null | /usr/bin/sort -u || true)
    if [[ -z "$listeners" ]] && { [[ -z "$stopped_pid" ]] || ! /bin/kill -0 "$stopped_pid" 2>/dev/null; }; then
      # Require two consecutive port-down samples so a selector teardown cannot
      # be mistaken for the next qualification server becoming ready.
      /bin/sleep 1
      listeners=$(/usr/sbin/lsof -nP -t -iTCP:"$port" -sTCP:LISTEN 2>/dev/null | /usr/bin/sort -u || true)
      if [[ -z "$listeners" ]] && { [[ -z "$stopped_pid" ]] || ! /bin/kill -0 "$stopped_pid" 2>/dev/null; }; then return 0; fi
    fi
    /bin/sleep 1
  done
  echo "port/PID handoff barrier failed for port $port, stopped PID ${stopped_pid:-none}" >&2
  exit 84
}

start_monitor() {
  local pid=$1 port=$2 out=$3
  (
    while /bin/kill -0 "$pid" 2>/dev/null; do
      /bin/date -u '+%Y-%m-%dT%H:%M:%SZ'
      /usr/sbin/sysctl vm.swapusage
      /usr/bin/pmset -g therm
      if ! /usr/sbin/sysctl vm.swapusage | /usr/bin/grep -q 'used = 0.00M'; then
        echo 'STOP: swap became nonzero'; /bin/kill -TERM "$pid"; exit 80
      fi
      if ! PYTHONPATH=scripts .venv/bin/python - <<'PY'
import json
from run_qualification_matrix import sample_thermal, thermally_stable
s=sample_thermal(); stable=thermally_stable(s,{}); print(json.dumps({'sample':s,'stable':stable},sort_keys=True)); assert stable
PY
      then echo 'STOP: thermal envelope breached'; /bin/kill -TERM "$pid"; exit 81; fi
      if ! /usr/bin/curl -fsS "http://127.0.0.1:$port/v1/status" | EXPECTED_SOURCE_SHA="$EXPECTED_SOURCE_SHA" .venv/bin/python -c 'import json,os,sys; s=json.load(sys.stdin); assert s["healthy"] and s["runtime"]["source_sha256"]==os.environ["EXPECTED_SOURCE_SHA"] and s["error"] is None; print(json.dumps(s,sort_keys=True))'; then
        echo 'STOP: server unhealthy or identity changed'; /bin/kill -TERM "$pid"; exit 82
      fi
      /bin/sleep 900
    done
  ) >>"$out" 2>&1 &
  echo $!
}

validate_status() {
  local path=$1 profile=$2 mtp=$3
  STATUS_PATH="$path" PROFILE="$profile" MTP="$mtp" EXPECTED_ARTIFACT="$EXPECTED_ARTIFACT" \
  EXPECTED_SOURCE_SHA="$EXPECTED_SOURCE_SHA" EXPECTED_NATIVE_SHA="$EXPECTED_NATIVE_SHA" POLICY="$POLICY" MODEL="$MODEL" .venv/bin/python - <<'PY'
import json, os
s=json.load(open(os.environ['STATUS_PATH']))
mtp=os.environ['MTP']=='true'
policy=json.load(open(os.environ['POLICY']))
assert s['healthy'] and s['error'] is None and s['inflight']==0 and s['queue_depth']==0, s
assert s['runtime']['source_sha256']==os.environ['EXPECTED_SOURCE_SHA'], s['runtime']
assert s['runtime']['mlx_native_sha256']==os.environ['EXPECTED_NATIVE_SHA'], s['runtime']
assert s['runtime']['macos']=='26.7', s['runtime']
assert s['artifact']==os.environ['EXPECTED_ARTIFACT'], s['artifact']
assert s['profile']==os.environ['PROFILE'], s['profile']
settings=s['settings']
assert settings['max_context']==262144 and settings['max_lanes']==20 and settings['max_inflight']==40
assert settings['prefill_step']==2048 and settings['cache_bytes']==17179869184 and settings['disk_cache'] is True
assert settings['mtp'] is mtp and settings['speculation']==('self_mtp' if mtp else 'ordinary')
assert settings['execution_policy']=={
    'persistent': True,
    'num_draft': 2,
    'rate_gate': False,
    'prefill_step_size': 2048,
    'segment_aware_live_tip': True,
    'segment_aware_cohort_size': 20,
    'segment_aware_async_qsa_promotion': True,
    'prefetch_known_tail_ple': True,
}
assert s['execution']['policy']==policy
env=settings['environment']
assert env['MLX_QWEN4_PLE_NVME']==os.environ['MODEL']+'/ple_rows.bin'
for name in ('MLX_QWEN4_PLE_COMPILE','MLX_QWEN4_QSA_POOLED_KEY_CACHE','MLX_QWEN4_QSA_SCATTER_CHOSEN','MLX_QWEN4_FUSED_GDN_DECODE','MLX_QWEN4_EAGER_DISPATCH','MLX_QWEN4_MOE_FUSED_GATE_UP'):
    assert env[name]=='1', (name,env.get(name))
assert env['MLX_QWEN4_FUSED_EXPERT_KERNEL']=='auto'
if mtp:
    assert env['MLX_QWEN4_FUSED_GDN_VERIFY']=='1'
    assert env['MLX_LM_SHARED_QSA_SUFFIX']=='auto'
    assert env['MLX_QWEN4_QSA_INDEXED']=='auto'
apc=s['apcv2']
assert apc['version']==2 and apc['cow']['active_leases']==0
assert apc['layout_name']=='qwen4-exp-layer-segments-v1'
segments=apc['layer_segments']
assert segments['schema']=='apcv2.layer-segments.v1'
assert segments['layers']>=0 and segments['segments']>=0
PY
}

validate_candidate() {
  local report=$1 profile=$2 mtp=$3
  REPORT="$report" PROFILE="$profile" MTP="$mtp" EXPECTED_ARTIFACT="$EXPECTED_ARTIFACT" \
  EXPECTED_SOURCE_SHA="$EXPECTED_SOURCE_SHA" EXPECTED_NATIVE_SHA="$EXPECTED_NATIVE_SHA" \
  EXPECTED_QUALIFIER_SHA="$EXPECTED_QUALIFIER_SHA" POLICY="$POLICY" .venv/bin/python - <<'PY'
import json, os
from pathlib import Path
r=json.loads(Path(os.environ['REPORT']).read_text())
mtp=os.environ['MTP']=='true'
assert r['schema']=='mlx2.serving-qualification.v1' and r['passed'] is True
assert r['qualification_harness']=={'schema':'mlx2.qualification-harness.v1','name':'scripts/qualify_serving.py','sha256':os.environ['EXPECTED_QUALIFIER_SHA']}
assert r['runtime']['source_sha256']==os.environ['EXPECTED_SOURCE_SHA']
assert r['runtime']['mlx_native_sha256']==os.environ['EXPECTED_NATIVE_SHA']
assert r['runtime']['macos']=='26.7' and r['artifact']==os.environ['EXPECTED_ARTIFACT']
s=r['settings']; assert s['max_context']==262144 and s['max_lanes']==20 and s['max_inflight']==40
assert s['prefill_step']==2048 and s['cache_bytes']==17179869184 and s['disk_cache'] is True
assert s['mtp'] is mtp and s['speculation']==('self_mtp' if mtp else 'ordinary')
policy=json.load(open(os.environ['POLICY']))
assert s['execution_policy']=={
    'persistent': True,
    'num_draft': 2,
    'rate_gate': False,
    'prefill_step_size': 2048,
    'segment_aware_live_tip': True,
    'segment_aware_cohort_size': 20,
    'segment_aware_async_qsa_promotion': True,
    'prefetch_known_tail_ple': True,
}
checks=r['checks']
required={'cold_text','warm_prefix','stream','tools','reasoning','stop','batch','context','cancel','recovery','unit_tests','hermes_client','sampling_controls','logprobs','cache_leases','runtime_stable','mixed_warm','quiescence','apcv2_reuse','apcv2_stores'}
features={'file_backed_ple','compiled_ple','pooled_qsa','scatter_qsa','fused_gdn_decode','eager_dispatch','fused_moe'}
if mtp:
    required |= {'mtp_execution','mtp_segmented_attention','mtp_transaction_branches','mtp_committed_cycles','mtp_true_batching','mtp_zero_full_prefix_materializations','mtp_zero_physical_b2','mtp_zero_failures'}
    features |= {'fused_gdn_verify','shared_qsa','async_promotion','known_tail_prefetch','indexed_qsa','private_delta'}
required |= {'feature_'+name for name in features}
assert not (required-set(checks)), required-set(checks)
assert all(checks[name]['passed'] is True for name in required)
assert all(row['passed'] is True for row in checks.values())
assert all(r['feature_observations'][name]>0 for name in features), r['feature_observations']
assert r['quiescence']['passed'] is True and r['quiescence']['timed_out'] is False
f=r['final_status']; assert f['healthy'] and f['error'] is None
assert f['inflight']==0 and f['queue_depth']==0 and f['profile']==os.environ['PROFILE']
assert f['runtime']==r['runtime'] and f['artifact']==r['artifact'] and f['settings']==s
assert f['execution']['policy']==policy
apc=f['apcv2']; assert apc['version']==2 and apc['hits']>0 and apc['stores']>0 and apc['cow']['active_leases']==0
assert apc['layout_name']=='qwen4-exp-layer-segments-v1'
assert apc['layer_segments']['layers']>0 and apc['layer_segments']['segments']>0
tables=f['execution']['ple_tables']; assert tables and sum(x['lookups'] for x in tables)>0
assert sum(x['rows'] for x in tables)>0 and sum(x['unique_rows'] for x in tables)>0
pc=f['execution']['ple_compile']; assert pc['enabled'] is True
assert pc['counts']['builds']>0 and pc['counts']['hits']>0
assert all(pc['counts'][name]==0 for name in ('fallbacks','overflow','skips'))
if mtp:
    seg=f['execution']['segmented_mtp']
    assert seg['segmented_attention_calls']>0 and seg['transaction_branches']>0
    assert seg['committed_cycles']>0 and seg['true_batched_engaged']>0 and seg['batched_target_forwards']>0
    assert seg['full_prefix_materializations']==0 and seg['physical_b2_formations']==0 and seg['failures']==0
PY
}

capture_performance() {
  local report=$1 output=$2 arm=$3
  REPORT="$report" OUTPUT="$output" ARM="$arm" .venv/bin/python - <<'PY'
import json, os
from pathlib import Path
r=json.loads(Path(os.environ['REPORT']).read_text())
b=r['checks']['batch']['evidence']; c=r['checks']['context']['evidence']
def receipt(response): return response['mlx2']
cold,warm=receipt(c['cold']),receipt(c['warm'])
rows=b['request_receipts']
out={
  'schema':'mlx2.route-performance-summary.v1', 'arm':os.environ['ARM'],
  'runtime':r['runtime'], 'artifact':r['artifact'], 'settings':r['settings'],
  'batch':{'requests':len(rows),'completion_tokens':sum(x['completion_tokens'] for x in rows),'wall_seconds':b['wall_seconds'],'aggregate_tokens_per_second':b['aggregate_tokens_per_second'],'observed_widths':b['observed_widths'],'request_completion_tokens_per_second':[x['completion_tokens']/x['elapsed_seconds'] for x in rows]},
  'near_limit_context':{'prompt_tokens':cold['prompt_tokens'],'cold_cached_tokens':cold['cached_tokens'],'cold_ttft_seconds':cold['ttft_seconds'],'cold_prompt_tokens_per_second':cold['prompt_tokens']/cold['ttft_seconds'],'warm_cached_tokens':warm['cached_tokens'],'warm_ttft_seconds':warm['ttft_seconds'],'warm_elapsed_seconds':warm['elapsed_seconds']},
  'feature_observations':r['feature_observations'],
}
Path(os.environ['OUTPUT']).write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
print(json.dumps(out,indent=2,sort_keys=True))
PY
}

unload() {
  local pid=$1 monitor=$2 port=$3 out=$4
  /bin/kill -TERM "$pid"
  for _ in $(seq 1 300); do /bin/kill -0 "$pid" 2>/dev/null || break; /bin/sleep 1; done
  if /bin/kill -0 "$pid" 2>/dev/null; then echo "PID $pid did not unload" >&2; exit 72; fi
  wait "$pid" || true
  /bin/kill -TERM "$monitor" 2>/dev/null || true
  wait "$monitor" 2>/dev/null || true
  ACTIVE_PID=""; ACTIVE_MONITOR=""
  if /usr/sbin/lsof -nP -iTCP:"$port" -sTCP:LISTEN | /usr/bin/grep -q LISTEN; then echo "port $port remains bound" >&2; exit 73; fi
  LAST_STOPPED_PID="$pid"
  port_down_pid_barrier "$port" "$LAST_STOPPED_PID"
  /usr/sbin/sysctl vm.swapusage | tee "$out/final-swap-$SHORT.txt"
  /usr/bin/pmset -g therm | tee "$out/final-thermal-$SHORT.txt"
  if ! /usr/sbin/sysctl vm.swapusage | /usr/bin/grep -q 'used = 0.00M'; then echo 'swap grew' >&2; exit 74; fi
}

publish_receipt() {
  local candidate=$1 canonical=$2
  CANDIDATE="$candidate" CANONICAL="$canonical" .venv/bin/python - <<'PY'
import hashlib, json, os
from pathlib import Path
src=Path(os.environ['CANDIDATE']); dst=Path(os.environ['CANONICAL']); payload=src.read_bytes()
assert json.loads(payload)['passed'] is True
dst.parent.mkdir(parents=True,exist_ok=True); tmp=dst.with_name('.'+dst.name+'.tmp')
with tmp.open('wb') as handle: handle.write(payload); handle.flush(); os.fsync(handle.fileno())
os.replace(tmp,dst)
directory_fd=os.open(dst.parent,os.O_RDONLY)
try: os.fsync(directory_fd)
finally: os.close(directory_fd)
assert dst.read_bytes()==payload
print(json.dumps({'canonical':str(dst),'sha256':hashlib.sha256(payload).hexdigest()},sort_keys=True))
PY
}

validate_canonical_publication() {
  local candidate=$1 canonical=$2 profile=$3 mtp=$4 selected_status=$5
  /usr/bin/cmp -s "$candidate" "$canonical" || {
    echo "canonical receipt differs from the selector-accepted candidate" >&2
    exit 85
  }
  validate_candidate "$canonical" "$profile" "$mtp"
  CANDIDATE="$candidate" SELECTED_STATUS="$selected_status" PROFILE="$profile" MTP="$mtp" .venv/bin/python - <<'PY'
import json, os
candidate=json.load(open(os.environ['CANDIDATE']))
selected=json.load(open(os.environ['SELECTED_STATUS']))
mtp=os.environ['MTP']=='true'
requested=['apc_v2','continuous_batch','layered_cache']
if mtp: requested += ['mtp']
requested += ['prefix_reuse','reasoning']
if mtp: requested += ['segmented_mtp']
requested += ['streaming','text','tools']
expected=(
    'model=qwen4_exp:flash-next;profile='+os.environ['PROFILE']+
    ';requested='+','.join(requested)+';fidelity=numerically_bounded'
)
assert selected['healthy'] and selected['qualification']=='qualified'
assert selected['runtime']==candidate['runtime']
assert selected['artifact']==candidate['artifact']
assert selected['settings']==candidate['settings']
assert selected['route_receipt']==expected, selected['route_receipt']
PY
}

prove_selector_acceptance() {
  local receipt=$1 profile=$2 mtp=$3 out=$4
  local cache="$out/selector-cache-$SHORT"
  rm -rf "$cache"; mkdir -p "$cache"
  port_down_pid_barrier "$PORT" "$LAST_STOPPED_PID"
  local -a command=(.venv/bin/python -u -m mlx2.server --model "$MODEL" --host 127.0.0.1 --port "$PORT" --max-context 262144 --max-lanes 20 --max-inflight 40 --cache-bytes 17179869184 --cache-dir "$cache" --execution-policy "$POLICY" --qualification "$receipt")
  if [[ "$mtp" == false ]]; then command+=(--ordinary); fi
  nohup env PYTHONPATH=src "${command[@]}" >"$out/selector-$SHORT.log" 2>&1 &
  local pid=$!; ACTIVE_PID="$pid"; echo "$pid" >"$out/selector-$SHORT.pid"
  wait_ready "$pid" "$PORT"
  assert_listener_pid "$PORT" "$pid"
  /usr/bin/curl -fsS "http://127.0.0.1:$PORT/v1/status" >"$out/selector-status-$SHORT.json"
  validate_status "$out/selector-status-$SHORT.json" "$profile" "$mtp"
  STATUS_PATH="$out/selector-status-$SHORT.json" PROFILE="$profile" MTP="$mtp" .venv/bin/python - <<'PY'
import json, os
s=json.load(open(os.environ['STATUS_PATH']))
mtp=os.environ.get('MTP')=='true'
requested=['apc_v2','continuous_batch','layered_cache']
if mtp: requested += ['mtp']
requested += ['prefix_reuse','reasoning']
if mtp: requested += ['segmented_mtp']
requested += ['streaming','text','tools']
expected=(
    'model=qwen4_exp:flash-next;profile='+os.environ['PROFILE']+
    ';requested='+','.join(requested)+';fidelity=numerically_bounded'
)
assert s['qualification']=='qualified' and s['route_receipt']==expected
PY
  local monitor; monitor=$(start_monitor "$pid" "$PORT" "$out/selector-monitor-$SHORT.log"); ACTIVE_MONITOR="$monitor"
  unload "$pid" "$monitor" "$PORT" "$out"
}

run_arm() {
  local arm=$1 profile=$2 mtp=$3
  local out="$RUNROOT/$arm" cache="$RUNROOT/$arm/cache-$SHORT"
  mkdir -p "$out" "$cache"
  port_down_pid_barrier "$PORT" "$LAST_STOPPED_PID"
  if find "$cache" -mindepth 1 -print -quit | /usr/bin/grep -q .; then echo "$arm cache is not fresh" >&2; exit 75; fi
  local -a command=(.venv/bin/python -u -m mlx2.server --model "$MODEL" --host 127.0.0.1 --port "$PORT" --max-context 262144 --max-lanes 20 --max-inflight 40 --cache-bytes 17179869184 --cache-dir "$cache" --execution-policy "$POLICY" --qualification-mode)
  if [[ "$mtp" == false ]]; then command+=(--ordinary); fi
  nohup env PYTHONPATH=src "${command[@]}" >"$out/server-$SHORT.log" 2>&1 &
  local pid=$!; ACTIVE_PID="$pid"; echo "$pid" >"$out/server-$SHORT.pid"
  wait_ready "$pid" "$PORT"
  assert_listener_pid "$PORT" "$pid"
  /usr/bin/curl -fsS "http://127.0.0.1:$PORT/v1/status" >"$out/status-initial-$SHORT.json"
  validate_status "$out/status-initial-$SHORT.json" "$profile" "$mtp"
  local monitor; monitor=$(start_monitor "$pid" "$PORT" "$out/monitor-$SHORT.log"); ACTIVE_MONITOR="$monitor"
  local -a qualify=(PYTHONPATH=src .venv/bin/python -u scripts/qualify_serving.py --url "http://127.0.0.1:$PORT" --timeout 7200 --quiescence-timeout 30 --output "$out/route-qualification-$SHORT.json" --require-feature file_backed_ple --require-feature compiled_ple --require-feature pooled_qsa --require-feature scatter_qsa --require-feature fused_gdn_decode --require-feature eager_dispatch --require-feature fused_moe)
  if [[ "$mtp" == true ]]; then qualify+=(--require-feature fused_gdn_verify --require-feature shared_qsa --require-feature async_promotion --require-feature known_tail_prefetch --require-feature indexed_qsa --require-feature private_delta); fi
  /usr/bin/env "${qualify[@]}" 2>&1 | tee "$out/qualify-$SHORT.log"
  validate_candidate "$out/route-qualification-$SHORT.json" "$profile" "$mtp"
  capture_performance "$out/route-qualification-$SHORT.json" "$out/performance-$SHORT.json" "$arm" | tee "$out/performance-$SHORT.log"
  unload "$pid" "$monitor" "$PORT" "$out"
  prove_selector_acceptance "$out/route-qualification-$SHORT.json" "$profile" "$mtp" "$out"
  publish_receipt "$out/route-qualification-$SHORT.json" "$out/route-qualification.json" | tee "$out/publication-$SHORT.json"
  validate_canonical_publication "$out/route-qualification-$SHORT.json" "$out/route-qualification.json" "$profile" "$mtp" "$out/selector-status-$SHORT.json"
}

# MTP is the default serving route and runs first. Each arm uses a fresh APCv2
# directory, unloads fully, proves production-selector acceptance, and publishes
# only after all exact identity, composition, PLE, quiescence and performance
# evidence is captured.
run_arm mtp flash-next-apcv2-mtp2 true
run_arm ordinary flash-next-apcv2-ordinary false
