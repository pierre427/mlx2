#!/usr/bin/env bash
# run_queue.sh - serial runner for the 14 splash GPU job specs in this directory.
#
#   Usage: bash run_queue.sh [--dry-run] [--only NN[,NN...]] [--list]
#                            [--force-stale-lock] [--no-lease] [--full]
#                            [--timeout-mult N]
#
# Runs one job at a time, never in parallel, never imports mlx itself.
# Written for bash 3.2 (stock /bin/bash on macOS): no associative arrays,
# no `timeout` binary, no GNU-only flags.
#
# Ordering rationale (see the final report): grouped by model to minimise
# model loads, correctness-gate jobs before perf jobs inside each group,
# cheap/short gate jobs first so a broken tree is found in minutes not hours.

set -u
set -o pipefail
# NOT set -e: the queue is continue-on-failure by design.

QUEUE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_ROOT="$QUEUE_DIR/logs"
PY="~/Desktop/mlx2/.venv/bin/python"
MDIR="~/mlx-models"
LOCK_DIRS="${LOCK_DIRS:-/tmp/gpu.lock /Users/Shared/mlxuag/gpu.lock}"
QUEUE_PORTS="8285 8295 8296 8297 8298 8309 8311 8313 8314 8412 8413"

DRY_RUN=0
ONLY=""
LIST=0
FORCE_STALE=0
TAKE_LEASE=1
QUEUE_FULL=0
TIMEOUT_MULT=3
# Hard ceiling on how long ONE spec may hold the GPU lease, independent of its
# own watchdog.  The interleaving contract promised to the peer sessions is
# ~55 min; a 3x watchdog on a 50 min estimate is 2.5h, so the watchdog alone
# cannot honour it.  This is the binding limit.
LEASE_CEILING="${LEASE_CEILING:-3300}"

# ---------------------------------------------------------------- job table
# Parallel indexed arrays, index order == execution order.
J_ID=()      ; J_SLUG=()  ; J_WT=()   ; J_MODELS=()
J_EST=()     ; J_KIND=()  ; J_EXCL=() ; J_REQ=()

add_job() {   # id slug worktree "models" est_min kind excl "required rel paths"
  J_ID+=("$1"); J_SLUG+=("$2"); J_WT+=("$3"); J_MODELS+=("$4")
  J_EST+=("$5"); J_KIND+=("$6"); J_EXCL+=("$7"); J_REQ+=("$8")
}

W() { echo "/private/tmp/mlx2-splash-$1"; }

NORTH="$MDIR/North-Mini-Code-1.0-mlx-4bit"
FLASH="$MDIR/Qwen3.8-Flash-Next-MLX-4bit-MTP"
HYBRID="$MDIR/Qwen3.6-35B-A3B-uncensored-heretic-Native-MTP-Preserved-oQ4e-mtp"
MUSE="$MDIR/Muse-Glimmer-30B-mlx-4bit"
MUSE_DRAFT="$MDIR/Muse-Glimmer-30B-DFlash2"

#        id  slug                      worktree                            models                    est kind   excl required-paths
add_job  11 "http-security"        "$(W 11-http-security)"        "$NORTH"                    5  gate  no  "scripts/sdk_smoke.py src/mlx2/server.py"
add_job  09 "host-memory-signals"  "$(W 09-host-memory-signals)"  "$NORTH"                    6  gate  no  "src/mlx2/server.py"
add_job  14 "agent-launcher"       "$(W 14-agent-launcher)"       "$NORTH"                    8  gate  no  "src/mlx2/clients.py src/mlx2/cli.py src/mlx2/server.py"
add_job  13 "progress-tokenize"    "$(W 13-progress-tokenize)"    "$NORTH $FLASH"            15  gate  no  "src/mlx2/server.py"
add_job  12 "tool-grammar"         "$(W 12-tool-grammar)"         "$FLASH"                   20  gate  no  "scripts/bench_tool_grammar.py"
add_job  01 "gpu-accept-count"     "$(W 01-gpu-accept-count)"     "$FLASH"                   45  both  yes "scripts/bench_gpu_accept_count.py scripts/qualify_qwen4_gdn_replay_model.py scripts/benchmark_serving.py"
add_job  08 "suspend-replay"       "$(W 08-suspend-replay)"       "$HYBRID"                  30  gate  no  "scripts/bench_suspend_replay.py"
add_job  06 "junction-snapshots"   "$(W 06-junction-snapshots)"   "$HYBRID"                  30  both  yes "scripts/bench_06_junction_snapshots.py"
add_job  07 "rolling-prefill-ckpt" "$(W 07-rolling-prefill-ckpt)" "$HYBRID"                  35  both  yes "scripts/bench_rolling_prefill_ckpt.py"
add_job  10 "srpt-prefill"         "$(W 10-srpt-prefill)"         "$HYBRID"                  35  perf  yes "scripts/bench_srpt_prefill.py"
add_job  04 "gdn-double-buffer"    "$(W 04-gdn-double-buffer)"    "$MUSE $MUSE_DRAFT"        15  gate  yes "scripts/bench_04_gdn_double_buffer.py scripts/qualify_serving.py"
add_job  02 "dflash-pair-select"   "$(W 02-dflash-pair-select)"   "$MUSE $MUSE_DRAFT"        25  both  yes "scripts/bench_dflash_pair_select.py qualification/policies/muse-dflash2.json"
add_job  05 "gpu-topk-accept"      "$(W 05-gpu-topk-accept)"      "$MUSE $MUSE_DRAFT"        25  perf  yes "scripts/bench_gpu_topk_accept.py tests/test_gpu_topk_accept.py qualification/policies/muse-dflash2.json"
add_job  03 "dflash-long-block"    "$(W 03-dflash-long-block)"    "$MUSE $MUSE_DRAFT"        50  perf  yes "scripts/bench_external_block_size.py tests/test_dflash_long_block.py qualification/policies/muse-dflash2.json"

# -------------------------------------------------------------- small utils
say()  { printf '%s\n' "$*"; }
note() { printf '[runner] %s\n' "$*"; }
ts()   { date +%Y-%m-%dT%H:%M:%S; }

idx_of() {  # job id -> array index, or empty
  local want="$1" i=0
  while [ "$i" -lt "${#J_ID[@]}" ]; do
    [ "${J_ID[$i]}" = "$want" ] && { echo "$i"; return 0; }
    i=$((i + 1))
  done
  return 1
}

kill_tree() {  # signal pid  (recursive; macOS has no `kill -- -pgid` for us here)
  local sig="$1" pid="$2" c
  for c in $(pgrep -P "$pid" 2>/dev/null); do kill_tree "$sig" "$c"; done
  kill "-$sig" "$pid" 2>/dev/null
}

port_listener() { lsof -nP -iTCP:"$1" -sTCP:LISTEN -t 2>/dev/null | head -5; }

# True only if $1 descends from this runner process.  Guards sweep_ports.
is_our_descendant() {
  local pid="$1" guard=0
  while [ -n "$pid" ] && [ "$pid" -gt 1 ] && [ "$guard" -lt 30 ]; do
    [ "$pid" = "$$" ] && return 0
    pid="$(ps -o ppid= -p "$pid" 2>/dev/null | tr -d ' ')"
    guard=$((guard + 1))
  done
  return 1
}

sweep_ports() {  # kill anything OF OURS still holding a queue port
  # Was: kill every listener on a QUEUE_PORT.  With per-spec leasing the peer
  # sessions now interleave on this machine and reuse the same port numbers --
  # observed live 2026-09-20 02:41, when mlx2-e0 came straight in behind our
  # release and bound 8296, a port in QUEUE_PORTS.  The old sweep would have
  # killed another session's server between our specs.  Only ever reap our own
  # descendants; anything else is logged and left alone.
  local p pids q
  for p in $QUEUE_PORTS; do
    pids="$(port_listener "$p")"
    [ -n "$pids" ] || continue
    for q in $pids; do
      if is_our_descendant "$q"; then
        note "sweeping OUR stray listener on port $p (pid $q)"
        kill "$q" 2>/dev/null; sleep 2; kill -9 "$q" 2>/dev/null
      else
        note "port $p has a FOREIGN listener (pid $q) - leaving it alone (not ours)"
      fi
    done
  done
}

# ------------------------------------------------------- GPU-holder guard
# Repo convention (scripts/qualify_qwen4_gdn_replay_model.py,
# scripts/bench_qwen4_gdn_replay_gpu.py, scripts/run_qwen36_overnight.py,
# docs/experiments/QWEN4-GDN-REPLAY-GPU-2026-09-17.md): a GPU holder writes an
# owner.json receipt into BOTH /tmp/gpu.lock/ and /Users/Shared/mlxuag/gpu.lock/,
# carrying at least {"pid", "label", "command", "started"}. A receipt whose pid
# is no longer alive is stale. There is no other lock file and no serve pidfile
# in the tree, so we additionally look for a live `mlx2.server` process and for
# listeners on the ports this queue uses.
json_field() {  # file key -> value (string or number), no jq dependency
  sed -n 's/.*"'"$2"'"[[:space:]]*:[[:space:]]*"\{0,1\}\([^",}]*\)"\{0,1\}.*/\1/p' "$1" | head -1
}

gpu_guard() {  # 0 = free, 1 = held, 2 = stale-only
  local busy=0 stale=0 d f pid label owner pids p
  for d in $LOCK_DIRS; do
    f="$d/owner.json"
    [ -f "$f" ] || continue
    pid="$(json_field "$f" pid)"
    label="$(json_field "$f" label)"
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
      say "  HELD   $f  pid=$pid label=${label:-?}  (process is alive)"
      busy=1
    else
      say "  STALE  $f  pid=${pid:-?} label=${label:-?}  (no such process)"
      stale=1
    fi
  done
  pids="$(pgrep -f 'mlx2\.server' 2>/dev/null | tr '\n' ' ')"
  if [ -n "${pids// /}" ]; then
    say "  HELD   a live mlx2.server process is running (pids: $pids)"
    busy=1
  fi
  for p in $QUEUE_PORTS; do
    owner="$(port_listener "$p")"
    if [ -n "$owner" ]; then
      say "  HELD   port $p already has a listener (pids: $(echo $owner | tr '\n' ' '))"
      busy=1
    fi
  done
  [ "$busy" -eq 1 ] && return 1
  [ "$stale" -eq 1 ] && return 2
  return 0
}

# ----------------------------------------------------------- per-job lease
# REPAIR 2026-09-20: the previous runner wrote ONE owner.json at batch start
# with `mkdir -p` and held /Users/Shared/mlxuag/gpu.lock across all 14 specs,
# starving the other session on this machine for hours.  Two bugs:
#   (a) batch-scoped instead of spec-scoped, and
#   (b) `mkdir -p` succeeds when the directory already exists, so it STOMPED a
#       live peer lease instead of waiting for it.
# The lock IS the directory (cpg_job.py and gpuq.sh both treat it that way and
# poll `[ -d $LOCK ]`), so a bare `mkdir` is the atomic acquire primitive.
LEASE_HELD=""           # lock dirs this process currently owns
LEASE_LABEL=""
LEASE_POLL="${LEASE_POLL:-10}"    # seconds between politeness polls
YIELD_GAP="${YIELD_GAP:-20}"      # inter-spec gap so a fast poller can get in
YIELD_FILE="$QUEUE_DIR/YIELD"     # touch to hand a peer the next slot

# ------------------------------------------------------- waiter registry
# Agreed cross-session wire format (mlx2-c2 has it live):
#   dir:  /Users/Shared/mlxuag/gpu.lock.waiters/
#   file: <session>.<label>.<pid>.<unix-ts>      e.g.
#         mlx2-76.splash-03-dflash-long-block.4711.1789883778
# Timestamp is LAST so it parses as ${f##*.} even when the label holds dots;
# pid is second-to-last and is used for liveness, which beats waiting out the
# age timeout when a session dies without firing its trap.
# The GPU then goes first-to-ASK rather than first-to-poll.
WAITERS_DIR="${WAITERS_DIR:-/Users/Shared/mlxuag/gpu.lock.waiters}"
SESSION_NAME="${SESSION_NAME:-mlx2-76}"
WAITER_FILE=""
WAITER_TS=""
WAITER_STALE_AGE="${WAITER_STALE_AGE:-5400}"   # 90 min > our 50 min job ceiling

WAITER_MTIME_STALE="${WAITER_MTIME_STALE:-300}"   # agreed heartbeat threshold
WAITER_DEFER_CAP="${WAITER_DEFER_CAP:-1200}"      # max 20 min deferred to ONE entry
WAITER_DEFER_SINCE=""     # when we first started deferring to $WAITER_DEFER_WHO
WAITER_DEFER_WHO=""

_mtime() { stat -f %m "$1" 2>/dev/null; }

waiter_register() {  # label -- register BEFORE polling.  Re-registering after a
                     # lost race MUST keep the original timestamp, or losing a
                     # race sends us to the back of the queue.
  local label="$1"
  mkdir -p "$WAITERS_DIR" 2>/dev/null || {
    note "waiter: cannot create $WAITERS_DIR - continuing without registration"
    WAITER_FILE=""; [ -n "$WAITER_TS" ] || WAITER_TS="$(date +%s)"; return 0
  }
  [ -n "$WAITER_TS" ] || WAITER_TS="$(date +%s)"     # keep the ORIGINAL ts
  WAITER_FILE="$WAITERS_DIR/$SESSION_NAME.$label.$$.$WAITER_TS"
  : > "$WAITER_FILE" 2>/dev/null || WAITER_FILE=""
  note "waiter: REGISTERED $SESSION_NAME.$label.$$.$WAITER_TS at $(ts)"
}

waiter_heartbeat() {  # "I still want the GPU" -- stronger than "my pid exists"
  if [ -n "$WAITER_FILE" ]; then
    [ -e "$WAITER_FILE" ] || : > "$WAITER_FILE" 2>/dev/null   # someone reaped it
    touch "$WAITER_FILE" 2>/dev/null
  fi
}

waiter_remove() {
  if [ -n "$WAITER_FILE" ] && [ -e "$WAITER_FILE" ]; then
    rm -f "$WAITER_FILE"
    note "waiter: removed $(basename "$WAITER_FILE")"
  fi
  WAITER_FILE=""; WAITER_TS=""
  WAITER_DEFER_SINCE=""; WAITER_DEFER_WHO=""
}

# 0 = clear to attempt the lock; 1 = an earlier, live, heartbeating waiter exists.
# Three independent staleness signals, ANY of which clears an entry:
#   (a) kill -0 fails            -> the session is gone
#   (b) mtime older than 300s    -> the session stopped saying it still wants it
#   (c) filename ts older than 90m -> age fallback
waiter_clear_to_go() {
  local f base rest tsv pidv age now mt mage blocked=0 who="" whots=0
  [ -d "$WAITERS_DIR" ] || return 0
  now="$(date +%s)"
  for f in "$WAITERS_DIR"/*; do
    [ -e "$f" ] || continue
    [ -n "$WAITER_FILE" ] && [ "$f" = "$WAITER_FILE" ] && continue
    base="$(basename "$f")"
    tsv="${base##*.}"
    rest="${base%.*}"; pidv="${rest##*.}"
    case "$tsv"  in ""|*[!0-9]*) continue ;; esac   # unparseable never blocks
    case "$pidv" in ""|*[!0-9]*) continue ;; esac
    if ! kill -0 "$pidv" 2>/dev/null; then
      note "waiter: reaped STALE $base (reason: dead pid $pidv)"
      rm -f "$f" 2>/dev/null; continue
    fi
    mt="$(_mtime "$f")"; [ -n "$mt" ] || mt="$tsv"
    mage=$((now - mt))
    if [ "$mage" -gt "$WAITER_MTIME_STALE" ]; then
      # Expected for a peer that keeps its waiter registered while it HOLDS the
      # lock (mlx2-c2 does this): it stops heartbeating, so it ages out here.
      note "waiter: reaped STALE $base (reason: no heartbeat for ${mage}s > ${WAITER_MTIME_STALE}s)"
      rm -f "$f" 2>/dev/null; continue
    fi
    age=$((now - tsv))
    if [ "$age" -gt "$WAITER_STALE_AGE" ]; then
      note "waiter: reaped STALE $base (reason: age ${age}s > ${WAITER_STALE_AGE}s)"
      rm -f "$f" 2>/dev/null; continue
    fi
    if [ -n "$WAITER_TS" ] && [ "$tsv" -lt "$WAITER_TS" ]; then
      blocked=1
      if [ "$whots" -eq 0 ] || [ "$tsv" -lt "$whots" ]; then who="$base"; whots="$tsv"; fi
    fi
  done
  if [ "$blocked" -eq 0 ]; then
    WAITER_DEFER_SINCE=""; WAITER_DEFER_WHO=""
    return 0
  fi
  # Cap how long we defer to ONE entry.  A live pid proves the process exists,
  # not that it still wants the GPU; deferring forever to a stuck entry is a
  # worse failure than occasionally taking a slot out of order.
  if [ "$WAITER_DEFER_WHO" != "$who" ]; then
    WAITER_DEFER_WHO="$who"; WAITER_DEFER_SINCE="$now"
  fi
  local deferred=$((now - WAITER_DEFER_SINCE))
  if [ "$deferred" -ge "$WAITER_DEFER_CAP" ]; then
    note "waiter: OVERRIDE - bypassing $who after deferring ${deferred}s (cap ${WAITER_DEFER_CAP}s); it is live and heartbeating but has not taken the GPU"
    WAITER_DEFER_SINCE="$now"     # re-arm so we do not spin on the override
    return 0
  fi
  note "waiter: YIELDING to $who (asked $((WAITER_TS - whots))s before us; deferred ${deferred}s of ${WAITER_DEFER_CAP}s cap)"
  return 1
}

lease_peer_alive() {  # dir -> 0 if a LIVE peer owns it
  local d="$1" f="$d/owner.json" pid
  [ -d "$d" ] || return 1
  [ -f "$f" ] || return 0        # mid-acquire: treat as held
  pid="$(json_field "$f" pid)"
  [ -n "$pid" ] || return 0
  kill -0 "$pid" 2>/dev/null && return 0
  return 1                       # receipt present, pid dead => stale
}

lease_describe() {
  local f="$1/owner.json"
  if [ -f "$f" ]; then
    printf 'pid=%s label=%s started=%s' \
      "$(json_field "$f" pid)" "$(json_field "$f" label)" "$(json_field "$f" started)"
  else
    printf '<no receipt>'
  fi
}

lease_write_receipt() {
  cat > "$1/owner.json" <<EOF
{
 "agent": "gpu-queue-runner",
 "agent_id": "gpu-queue-runner-$$",
 "session": "$SESSION_NAME",
 "label": "$2",
 "command": ["bash", "$QUEUE_DIR/run_queue_v2.sh"],
 "pid": $$,
 "log": "$RUN_LOG_DIR",
 "started": "$(ts)"
}
EOF
}

# lease_acquire is a thin wrapper whose ONLY job is to guarantee that the
# waiter file is removed however the wait ends.
#
# Incident on mlx2-e0 (2026-09-20 ~02:13): their runner removed its waiter only
# on the success path, so when a per-spec retry budget expired while job 03 held
# the lock, the spec was abandoned and the waiter file survived with a LIVE pid
# and the earliest timestamp -- invisible to dead-pid reaping and blocking every
# participant until the 90-minute age-out.  "Remove on acquire plus an EXIT
# trap" is NOT sufficient: a waiter can be abandoned by a process that keeps
# running.  So the removal lives here, on the single path every exit takes,
# rather than being repeated at each return site where one will be missed.
lease_acquire() {  # label
  local label="$1" rc=0
  [ "$TAKE_LEASE" -eq 1 ] || { note "lease: --no-lease, skipping acquire for $label"; return 0; }
  _lease_wait_loop "$label"; rc=$?
  waiter_remove          # acquired, failed, aborted, deferred -- all land here
  if [ "$rc" -ne 0 ]; then
    note "lease: ERROR - wait loop for $label ended WITHOUT the lease (rc=$rc)"
  fi
  return "$rc"
}

_lease_wait_loop() {  # label -- waits indefinitely by design; see note below
  local label="$1" d got waited=0 want=0 have=0 announced=""
  # (1) operator hand-off file, checked before we even register
  while [ -e "$YIELD_FILE" ]; do
    [ "$announced" = yield ] || { note "lease: YIELD file present ($YIELD_FILE) - holding off"; announced=yield; }
    sleep "$LEASE_POLL"; waited=$((waited + LEASE_POLL))
  done
  # (2) announce our intent BEFORE polling, so first-to-ask wins
  waiter_register "$label"
  # DELIBERATELY UNBOUNDED.  With the ordering rule a turn is guaranteed, so
  # waiting is correct; a per-spec retry budget that expires into "skip this
  # spec" is the mlx2-e0 failure above.  If this ever needs a bound it must end
  # the RUN loudly, never silently abandon a spec.
  while :; do
    waiter_heartbeat
    if [ -e "$YIELD_FILE" ]; then sleep "$LEASE_POLL"; continue; fi
    if ! waiter_clear_to_go; then sleep "$LEASE_POLL"; waited=$((waited + LEASE_POLL)); continue; fi
    got=""
    for d in $LOCK_DIRS; do
      if lease_peer_alive "$d"; then
        if [ "$announced" != "$d" ]; then
          note "lease: WAITING on $d held by $(lease_describe "$d")  [job $label, waited ${waited}s]"
          announced="$d"
        fi
        break
      fi
      if [ -d "$d" ]; then
        note "lease: reclaiming STALE $d ($(lease_describe "$d"))"
        rm -rf "$d"
      fi
      if mkdir "$d" 2>/dev/null; then
        lease_write_receipt "$d" "$label"; got="$got $d"
      else
        break                       # lost the race between check and mkdir
      fi
    done
    want=0; have=0
    for d in $LOCK_DIRS; do want=$((want + 1)); done
    for d in $got;        do have=$((have + 1)); done
    if [ "$want" -eq "$have" ]; then
      LEASE_HELD="$got"; LEASE_LABEL="$label"
      note "lease: ACQUIRED for $label at $(ts) after ${waited}s ->$got"
      # The waiter is dropped by lease_acquire, strictly AFTER this acquire --
      # never before, since removing it first would order intent rather than
      # acquisition and surrender our queue position in the gap.
      return 0
    fi
    for d in $got; do rm -rf "$d"; done   # never hold a partial lease
    waiter_register "$label"              # lost the race: keep the ORIGINAL ts
    sleep "$LEASE_POLL"; waited=$((waited + LEASE_POLL))
  done
}

lease_release() {
  local d f
  for d in $LEASE_HELD; do
    f="$d/owner.json"
    if [ -f "$f" ] && [ "$(json_field "$f" pid)" = "$$" ]; then
      rm -rf "$d"; note "lease: RELEASED $d for ${LEASE_LABEL:-?} at $(ts)"
    elif [ -d "$d" ]; then
      note "lease: NOT releasing $d - receipt is no longer ours ($(lease_describe "$d"))"
    fi
  done
  LEASE_HELD=""; LEASE_LABEL=""
  waiter_remove          # same cleanup path: failure, timeout and Ctrl-C too
}

# ======================================================= gate enforcement ==
# REPAIR 2026-09-20: the previous runner PRINTED each job's expected values and
# returned 0 regardless (job_11 marked "ok" in 15s with five of ten assertions
# wrong).  A gate that is printed but never compared is worse than no gate: it
# manufactures a confident false pass.  Every job now funnels its observations
# through these helpers, and gate_verdict is the job's return value.
G_TOTAL=0; G_FAIL=0
gate_reset() { G_TOTAL=0; G_FAIL=0; }
_g_pass() { G_TOTAL=$((G_TOTAL + 1)); echo "GATE PASS  $1"; }
_g_fail() { G_TOTAL=$((G_TOTAL + 1)); G_FAIL=$((G_FAIL + 1)); echo "GATE FAIL  $1"; }
gate_eq() {  # label actual expected
  if [ "$2" = "$3" ]; then _g_pass "$1: got '$2' (== '$3')"
  else _g_fail "$1: got '$2' WANT '$3'"; fi
}
gate_ne() {  # label actual notexpected
  if [ "$2" != "$3" ]; then _g_pass "$1: got '$2' (!= '$3')"
  else _g_fail "$1: got '$2' WANT anything but '$3'"; fi
}
gate_true() {  # label description ; uses rc of the preceding command via $?
  if [ "$2" -eq 0 ]; then _g_pass "$1"; else _g_fail "$1 (rc=$2)"; fi
}
gate_verdict() {
  echo "### GATE SUMMARY: $((G_TOTAL - G_FAIL))/$G_TOTAL passed, $G_FAIL failed"
  [ "$G_FAIL" -eq 0 ]
}
# Run the spec-gate evaluator for a job over the artifacts it produced.
gate_eval() {  # job_id artifact_dir [extra args...]
  local jid="$1"; shift
  echo "### spec-gate evaluation (gates.py $jid)"
  "$PY" "$QUEUE_DIR/gates.py" "$jid" "$@"
  local grc=$?
  if [ "$grc" -eq 0 ]; then _g_pass "spec gates for job $jid"
  else _g_fail "spec gates for job $jid (gates.py rc=$grc)"; fi
  return 0
}

# --------------------------------------------------- server start/stop helpers
# REPAIR: the tree under test must be the one that is measured.  Matching
# commit 7d41714 ("Pin every scripts/ server spawner to its own checkout's
# src"): the job worktree's absolute src goes FIRST on the child's PYTHONPATH,
# ahead of anything inherited.  A relative `PYTHONPATH=src` is one stray `cd`
# away from silently measuring the main checkout.
JOB_WT=""          # set by the driver before each job_NN call
SERVER_PID=""
SERVER_LOG=""

jp() {  # absolute PYTHONPATH for the job under test
  printf '%s' "$JOB_WT/src${PYTHONPATH:+:$PYTHONPATH}"
}
# Every in-job python call goes through this instead of `jpy`.
jpy() { PYTHONPATH="$(jp)" "$PY" "$@"; }

assert_tree_under_test() {  # hard refusal if mlx2 does not resolve into the worktree
  local got
  got="$(PYTHONPATH="$(jp)" "$PY" -c 'import mlx2,sys; sys.stdout.write(mlx2.__file__)' 2>/dev/null)"
  echo "### tree under test: mlx2.__file__ = ${got:-<import failed>}"
  case "$got" in
    "$JOB_WT"/src/mlx2/*) _g_pass "PYTHONPATH pin: mlx2 resolves inside $JOB_WT" ;;
    *) _g_fail "PYTHONPATH pin: mlx2 resolved to '${got:-<import failed>}', NOT under $JOB_WT"
       return 1 ;;
  esac
  return 0
}

start_server() {  # logfile args...
  local slog="$1"; shift
  SERVER_LOG="$slog"
  PYTHONPATH="$(jp)" "$PY" -u -m mlx2.server "$@" > "$slog" 2>&1 &
  SERVER_PID=$!
  note "server pid $SERVER_PID  log $slog  PYTHONPATH=$(jp)"
}

# REPAIR (M3 fix 2, then hardened 2026-09-20 after mlx2-c2's finding):
# /v1/status reported state="ready" about TWO SECONDS into a 27B load, and
# /health -- our first fix -- is still only a STATUS PROBE.  A status field is
# a claim; the only proof that the engine will accept work is work it accepted.
# So readiness is now a real completion that must return 200, retried through
# 503/429 with a bounded timeout.  The readiness probe and the operation about
# to be measured are the same operation; anything weaker is an audit that
# passes while the thing it audits has not happened.
wait_ready() {  # url [max_seconds]
  local url="$1" max="${2:-900}" waited=0 hbase code attempts=0 t0 probe body
  t0="$(date +%s)"
  hbase="$(printf '%s' "$url" | sed -E 's#(https?://[^/]+).*#\1#')"
  # phase 1 - transport and /health (cheap, weeds out "not listening yet")
  while [ "$waited" -lt "$max" ]; do
    if curl -sf "$url" >/dev/null 2>&1 && curl -sf "$hbase/health" >/dev/null 2>&1; then break; fi
    if [ -n "$SERVER_PID" ] && ! kill -0 "$SERVER_PID" 2>/dev/null; then
      note "FAIL server exited before becoming ready"
      [ -n "$SERVER_LOG" ] && tail -30 "$SERVER_LOG"
      return 1
    fi
    sleep 5; waited=$((waited + 5))
  done
  if [ "$waited" -ge "$max" ]; then
    note "FAIL timed out (${max}s) waiting for $url / $hbase/health"; return 1
  fi
  # phase 2 - PROVE it by doing the thing.  A key-gated server answers 401/403
  # before touching the engine, which is as far as an unauthenticated probe can
  # see; that is accepted and noted rather than retried to the timeout.
  resolve_served_id "$hbase" >/dev/null 2>&1
  probe='{"model":"'"${SERVED_ID:-x}"'","messages":[{"role":"user","content":"ready?"}],"max_tokens":1,"temperature":0}'
  while [ "$waited" -lt "$max" ]; do
    attempts=$((attempts + 1))
    body="$(mktemp)"
    code="$(curl -s -o "$body" -w '%{http_code}' --max-time 120 \
      "$hbase/v1/chat/completions" -H 'Content-Type: application/json' -d "$probe" 2>/dev/null)"
    case "$code" in
      200)
        rm -f "$body"
        note "ready: $hbase answered a real completion 200 after $(( $(date +%s) - t0 ))s, ${attempts} probe attempt(s)"
        record_route "$hbase"        # always record which path we are about to measure
        return 0 ;;
      401|403)
        rm -f "$body"
        note "ready: $hbase is key-gated (HTTP $code on the probe) after $(( $(date +%s) - t0 ))s, ${attempts} attempt(s) - engine reachability not further provable unauthenticated"
        record_route "$hbase"
        return 0 ;;
      503|429|000|500|502|504)
        note "not ready yet: probe HTTP $code (attempt $attempts, ${waited}s elapsed) - $(head -c 120 "$body" 2>/dev/null)"
        rm -f "$body" ;;
      *)
        note "readiness probe returned HTTP $code (attempt $attempts): $(head -c 200 "$body" 2>/dev/null)"
        rm -f "$body"
        note "ready: $hbase is answering requests (non-retryable status) after $(( $(date +%s) - t0 ))s"
        return 0 ;;
    esac
    if [ -n "$SERVER_PID" ] && ! kill -0 "$SERVER_PID" 2>/dev/null; then
      note "FAIL server exited during readiness probing after ${attempts} attempt(s)"
      [ -n "$SERVER_LOG" ] && tail -30 "$SERVER_LOG"
      return 1
    fi
    sleep 5; waited=$((waited + 5))
  done
  note "FAIL not ready after ${max}s and ${attempts} completion probe attempt(s) - last HTTP $code"
  [ -n "$SERVER_LOG" ] && tail -20 "$SERVER_LOG"
  return 1
}

# Route assertion.  mlx2-e0 had an arm LABELLED "fixed MTP" silently measure
# ordinary decode after a default-route change landed on main: the run
# completed, the numbers were real, and they described a different code path
# than the label claimed.  main records the resolved route and how it was
# chosen, so an arm can prove which path it is on instead of assuming.
# Always recorded; asserted (fail-closed) whenever the label claims a route.
RESOLVED_ROUTE=""
RESOLVED_ROUTE_SRC=""
record_route() {  # base_url [expected_route]
  local base="$1" expect="${2:-}" st
  st="$(curl -sf "$base/v1/status" 2>/dev/null)"
  # /v1/status has no single top-level route; the value appears nested and a
  # greedy sed picks the LAST match (it reported segmented_self_mtp for a
  # native_mtp arm and produced a false "MISLABELLED").  Collect the distinct
  # values instead; the AUTHORITATIVE per-request assertion lives in gates.py
  # against each arm's smoke receipt (mlx2.route), which is unambiguous.
  RESOLVED_ROUTE="$(printf '%s' "$st" | "$PY" -c '
import sys,json,re
b=sys.stdin.read()
vals=[]
for v in re.findall(r"\"route\"\s*:\s*\"([^\"]+)\"", b):
    if v not in vals: vals.append(v)
print(",".join(vals))' 2>/dev/null)"
  RESOLVED_ROUTE_SRC="$(printf '%s' "$st" | "$PY" -c '
import sys,re
m=re.findall(r"\"route_selection_source\"\s*:\s*\"([^\"]+)\"", sys.stdin.read())
print(m[0] if m else "")' 2>/dev/null)"
  echo "### resolved route: ${RESOLVED_ROUTE:-<none reported>} (source: ${RESOLVED_ROUTE_SRC:-<none>})"
  if [ -n "$expect" ]; then
    # Equivalence classes: segmented_self_mtp / continuous_batched_self_mtp are
    # VARIANTS of the native self-MTP route, not different routes.
    local okroute=1
    case "$expect" in
      native_mtp) case "$RESOLVED_ROUTE" in *native_mtp*|*self_mtp*) okroute=0 ;; esac ;;
      ordinary)   case "$RESOLVED_ROUTE" in
                    *self_mtp*|*external_draft*|*prompt_lookup*) okroute=1 ;;
                    *ordinary*) okroute=0 ;;
                  esac ;;
      *)          case "$RESOLVED_ROUTE" in *"$expect"*) okroute=0 ;; esac ;;
    esac
    if [ -z "$RESOLVED_ROUTE" ]; then
      note "route advisory: /v1/status exposed no route field (asserted from the receipt in gates.py instead)"
    elif [ "$okroute" -eq 0 ]; then
      _g_pass "route is '$RESOLVED_ROUTE' as the arm claims ('$expect', source ${RESOLVED_ROUTE_SRC:-?})"
    else
      _g_fail "route MISLABELLED: arm claims '$expect' but the server resolved '$RESOLVED_ROUTE' (source ${RESOLVED_ROUTE_SRC:-?}) - every number from this arm describes a different code path"
    fi
  fi
}

# REPAIR (M3 fix 1): the server matches the served model id EXACTLY (6
# ResourceNotFound sites on main), and every spec body sends {"model":"x"},
# which 404s.  The served id is the model dirname, reported by GET /v1/models.
# Bodies are rewritten at request time so the specs stay untouched.
SERVED_ID=""
resolve_served_id() {  # base_url
  # Never clobber a good id with a failed lookup: on a key-gated server
  # /v1/models answers 401, and wiping SERVED_ID there would send the next
  # request back to {"model":"x"} and reintroduce the very 404s this fixes
  # (job 11 run B resolves its id in run A and reuses it).
  local found
  found="$(curl -sf "$1/v1/models" 2>/dev/null \
    | sed -n 's/.*"id"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' | head -1)"
  if [ -z "$found" ]; then
    note "WARN could not resolve served model id from $1/v1/models (keeping '${SERVED_ID:-<unset>}')"
    return 1
  fi
  SERVED_ID="$found"
  note "served model id: $SERVED_ID"
  return 0
}
# Rewrite a request body's "model" field to the live served id.
mb() {  # json_body  -> body with "model" replaced by $SERVED_ID
  if [ -z "$SERVED_ID" ]; then printf '%s' "$1"; return 0; fi
  printf '%s' "$1" | sed 's|"model"[[:space:]]*:[[:space:]]*"[^"]*"|"model": "'"$SERVED_ID"'"|'
}

# REPAIR (M3 fix 3): a bare `wait` waits for EVERY background child of the
# shell, which includes the server started by start_server -- so job_09 hung
# until its 1200s timeout (observed: TIMEOUT at 1213s).  Collect the curl pids
# and wait only on those.
wait_pids() { local p; for p in "$@"; do wait "$p" 2>/dev/null; done; }
stop_server() {
  [ -n "$SERVER_PID" ] || return 0
  kill_tree TERM "$SERVER_PID"
  local i=0
  while kill -0 "$SERVER_PID" 2>/dev/null && [ "$i" -lt 60 ]; do sleep 1; i=$((i + 1)); done
  kill_tree KILL "$SERVER_PID" 2>/dev/null
  wait "$SERVER_PID" 2>/dev/null
  note "server stopped"
  SERVER_PID=""
}

# ============================================================== job bodies ==
# Each job_NN runs with cwd = its worktree. stdout/stderr go to its log file.

job_11() {
  local OUT="$QUEUE_DIR/11-out"; rm -rf "$OUT"; mkdir -p "$OUT"
  gate_reset
  assert_tree_under_test || return 1
  local KEYF="$OUT/api.key"; printf 'sk-smoke-11\n' > "$KEYF"; chmod 600 "$KEYF"
  local B='{"model":"x","messages":[{"role":"user","content":"Say hi."}],"max_tokens":16}'
  local BS='{"model":"x","messages":[{"role":"user","content":"Say hi."}],"max_tokens":16,"stream":true}'
  local MB='{"model":"x","max_tokens":16,"messages":[{"role":"user","content":"Say hi."}]}'
  local a1 a2 a3 a4 a5 b1 b2 b3 b4 b5

  note "Run A: default flags (gate on: Host/Origin, no key)"
  start_server "$OUT/a.log" --model "$NORTH" --host 127.0.0.1 --port 8311 --max-context 32768 --qualification-mode
  wait_ready http://127.0.0.1:8311/health || { stop_server; _g_fail "run A server never became ready"; gate_verdict; return 1; }
  # REPAIR: the server matches the served id exactly; {"model":"x"} 404s.
  resolve_served_id http://127.0.0.1:8311
  echo "served_id=$SERVED_ID" > "$OUT/served_id.txt"
  a1=$(curl -s -o "$OUT/a1.body" -w '%{http_code}' http://127.0.0.1:8311/v1/chat/completions -H 'Content-Type: application/json' -d "$(mb "$B")")
  a2=$(curl -s -o /dev/null -w '%{http_code}' http://localhost:8311/v1/chat/completions -H 'Content-Type: application/json' -d "$(mb "$B")")
  a3=$(curl -s -o /dev/null -w '%{http_code}' -N http://127.0.0.1:8311/v1/chat/completions -H 'Content-Type: application/json' -d "$(mb "$BS")")
  a4=$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8311/v1/models -H 'Host: evil.example:8311')
  a5=$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8311/v1/chat/completions -H 'Origin: http://evil.example' -H 'Content-Type: application/json' -d "$(mb "$B")")
  printf 'A1 %s\nA2 %s\nA3 %s\nA4 %s\nA5 %s\n' "$a1" "$a2" "$a3" "$a4" "$a5" | tee "$OUT/codes-a.txt"
  jpy scripts/sdk_smoke.py --help >/dev/null 2>&1
  gate_true "sdk_smoke.py importable" $?
  stop_server

  note "Run B: API key on"
  start_server "$OUT/b.log" --model "$NORTH" --host 127.0.0.1 --port 8311 --max-context 32768 --qualification-mode --api-key-file "$KEYF"
  wait_ready http://127.0.0.1:8311/health || { stop_server; _g_fail "run B server never became ready"; gate_verdict; return 1; }
  b1=$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8311/v1/chat/completions -H 'Content-Type: application/json' -d "$(mb "$B")")
  b2=$(curl -s -o "$OUT/b2.body" -w '%{http_code}' http://127.0.0.1:8311/v1/chat/completions -H 'Authorization: Bearer sk-smoke-11' -H 'Content-Type: application/json' -d "$(mb "$B")")
  b3=$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8311/v1/messages -H 'x-api-key: sk-smoke-11' -H 'anthropic-version: 2023-06-01' -H 'Content-Type: application/json' -d "$(mb "$MB")")
  b4=$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8311/health)
  b5=$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8311/metrics)
  printf 'B1 %s\nB2 %s\nB3 %s\nB4 %s\nB5 %s\n' "$b1" "$b2" "$b3" "$b4" "$b5" | tee "$OUT/codes-b.txt"
  stop_server

  # ---- the gates the old runner only printed ----
  echo "--- gate check: expect A1/A2/A3=200 A4=421 A5=403 B1=401 B2=200 B3=200 B4=200 B5=401"
  gate_eq "A1 plain client"            "$a1" 200
  gate_eq "A2 localhost Host"          "$a2" 200
  gate_eq "A3 streaming"               "$a3" 200
  gate_eq "A4 evil Host -> 421"        "$a4" 421
  gate_eq "A5 evil Origin -> 403"      "$a5" 403
  gate_eq "B1 no key -> 401"           "$b1" 401
  gate_eq "B2 bearer key -> 200"       "$b2" 200
  gate_eq "B3 x-api-key messages -> 200" "$b3" 200
  gate_eq "B4 /health unauthenticated" "$b4" 200
  gate_eq "B5 /metrics -> 401"         "$b5" 401
  gate_eq "a.log tracebacks" "$(grep -c Traceback "$OUT/a.log")" 0
  gate_eq "b.log tracebacks" "$(grep -c Traceback "$OUT/b.log")" 0
  gate_eval 11 "$OUT"
  gate_verdict
}

job_09() {
  local OUT="$QUEUE_DIR/09-out"; rm -rf "$OUT"; mkdir -p "$OUT"
  gate_reset
  assert_tree_under_test || return 1
  printf '{"host_memory_signals": {"enabled": true, "fall_after_seconds": 5.0}}\n' > "$OUT/policy.json"
  local B='{"model":"x","messages":[{"role":"user","content":"Count from 1 to 20."}],"max_tokens":64,"temperature":0}'
  local arm code pids i

  for arm in a b; do
    if [ "$arm" = a ]; then
      note "Run A: default (feature off)"
      start_server "$OUT/a.log" --model "$NORTH" --host 127.0.0.1 --port 8309 --max-context 32768 --qualification-mode
    else
      note "Run B: host_memory_signals enabled"
      start_server "$OUT/b.log" --model "$NORTH" --host 127.0.0.1 --port 8309 --max-context 32768 --qualification-mode --execution-policy "$OUT/policy.json"
    fi
    if ! wait_ready http://127.0.0.1:8309/health; then
      stop_server; _g_fail "run $arm server never became ready"; gate_verdict; return 1
    fi
    resolve_served_id http://127.0.0.1:8309
    code=$(curl -s -o "$OUT/$arm.json" -w '%{http_code}' http://127.0.0.1:8309/v1/chat/completions -H 'Content-Type: application/json' -d "$(mb "$B")")
    gate_eq "run $arm primary request" "$code" 200
    # REPAIR (M3 fix 3): a bare `wait` here waited on the SERVER pid too, which
    # is why job 09 hung to its 1200s timeout. Wait only on the curl pids.
    pids=""
    for i in 1 2 3 4; do
      curl -s -o "$OUT/$arm.conc$i.json" -w "%{http_code}\n" http://127.0.0.1:8309/v1/chat/completions \
        -H 'Content-Type: application/json' -d "$(mb "$B")" > "$OUT/$arm.conc$i.code" &
      pids="$pids $!"
    done
    wait_pids $pids
    for i in 1 2 3 4; do
      gate_eq "run $arm concurrent request $i" "$(tr -d '[:space:]' < "$OUT/$arm.conc$i.code")" 200
    done
    [ "$arm" = b ] && sleep 2
    curl -s http://127.0.0.1:8309/v1/status  > "$OUT/$arm.status.json"
    curl -s http://127.0.0.1:8309/metrics    > "$OUT/$arm.metrics.txt"
    stop_server
    gate_eq "$arm.log tracebacks" "$(grep -c Traceback "$OUT/$arm.log")" 0
  done

  echo "--- kernel reference for the level gate:"
  sysctl -n kern.memorystatus_vm_pressure_level 2>/dev/null > "$OUT/kern.pressure"
  sysctl -n hw.memsize > "$OUT/kern.memsize"
  cat "$OUT/kern.pressure" "$OUT/kern.memsize"
  gate_eval 09 "$OUT"
  gate_verdict
}

job_14() {
  local OUT="$QUEUE_DIR/14-out"
  rm -rf "$OUT"; mkdir -p "$OUT"
  gate_reset
  assert_tree_under_test || return 1
  local STATE="$OUT/state"
  local ask='{"model":"x","max_tokens":256,"thinking":{"type":"enabled","budget_tokens":128},"messages":[{"role":"user","content":"What is 17*3? Think briefly."}]}'
  local code

  note "Run A: --api-state-dir, first start (key gets created)"
  start_server "$OUT/a.log" --model "$NORTH" --host 127.0.0.1 --port 8314 --max-context 32768 --qualification-mode --api-state-dir "$STATE"
  wait_ready http://127.0.0.1:8314/health || { stop_server; _g_fail "run A server never became ready"; gate_verdict; return 1; }
  resolve_served_id http://127.0.0.1:8314
  ls -l "$STATE/reasoning-signing.key" | tee "$OUT/keyls.txt"
  stat -f '%Sp %z' "$STATE/reasoning-signing.key" > "$OUT/keystat.txt" 2>/dev/null
  cat "$OUT/keystat.txt"
  curl -s http://127.0.0.1:8314/v1/status | "$PY" -c 'import json,sys; print(json.load(sys.stdin)["reasoning_signing"])' | tee "$OUT/a.signing"
  code=$(curl -s -o "$OUT/a.msg.json" -w '%{http_code}' http://127.0.0.1:8314/v1/messages -H 'anthropic-version: 2023-06-01' -H 'Content-Type: application/json' -d "$(mb "$ask")")
  gate_eq "run A /v1/messages" "$code" 200
  stop_server

  # Build turn 2 exactly as the spec does (turn2.json), plus two harness-only
  # variants the VERDICT depends on:
  #   turn2fix.json  - identical but budget_tokens < max_tokens.  The spec's own
  #     turn 2 sets budget_tokens == max_tokens == 64, which the API rejects with
  #     400 "thinking budget_tokens must be less than max_tokens" BEFORE any
  #     signature is verified.  That is a SPEC defect: run as written, gate 4
  #     can never be evaluated either way.
  #   turn2tamper.json - budget fixed AND one character of the signature
  #     corrupted.  This is the real negative control: if a TAMPERED signature
  #     is accepted with the rejected counter still 0, the server is not
  #     verifying inbound signatures at all, which would make gate 4 vacuous.
  "$PY" - "$OUT" "$SERVED_ID" <<'PY'
import json, sys
out, served = sys.argv[1], (sys.argv[2] or "x")
first = json.load(open(f"{out}/a.msg.json"))
if "content" not in first:
    print(f"RUN A DID NOT RETURN A MESSAGE: {json.dumps(first)[:400]}")
    sys.exit(2)
blocks = [b for b in first["content"] if b["type"] in ("thinking", "text")]
if not any(b["type"] == "thinking" and b.get("signature") for b in blocks):
    print(f"RUN A HAS NO SIGNED THINKING BLOCK: {json.dumps(first)[:400]}")
    sys.exit(3)
def body(budget, blks):
    return {"model": served, "max_tokens": 64,
            "thinking": {"type": "enabled", "budget_tokens": budget},
            "messages": [{"role": "user", "content": "What is 17*3? Think briefly."},
                         {"role": "assistant", "content": blks},
                         {"role": "user", "content": "Now add 1."}]}
json.dump(body(64, blocks), open(f"{out}/turn2.json", "w"))          # spec-literal
json.dump(body(32, blocks), open(f"{out}/turn2fix.json", "w"))       # budget < max
tampered = json.loads(json.dumps(blocks))
for b in tampered:
    if b["type"] == "thinking" and b.get("signature"):
        sig = b["signature"]
        b["signature"] = sig[:-4] + ("AAAA" if not sig.endswith("AAAA") else "BBBB")
json.dump(body(32, tampered), open(f"{out}/turn2tamper.json", "w"))
print("turn2 variants written; signed block present")
PY
  local t2rc=$?
  gate_true "run A produced a signed thinking block (gate 3)" "$t2rc"
  [ "$t2rc" -eq 0 ] || { gate_eval 14 "$OUT"; gate_verdict; return 1; }

  note "Run B: restart on the same state dir"
  start_server "$OUT/b.log" --model "$NORTH" --host 127.0.0.1 --port 8314 --max-context 32768 --qualification-mode --api-state-dir "$STATE"
  wait_ready http://127.0.0.1:8314/health || { stop_server; _g_fail "run B server never became ready"; gate_verdict; return 1; }
  curl -s http://127.0.0.1:8314/v1/status | "$PY" -c 'import json,sys; print(json.load(sys.stdin)["reasoning_signing"])' | tee "$OUT/b.signing"
  # spec-literal turn 2 (budget_tokens == max_tokens): recorded to prove the spec defect
  code=$(curl -s -o "$OUT/b.spec.msg.json" -w '%{http_code}' http://127.0.0.1:8314/v1/messages -H 'anthropic-version: 2023-06-01' -H 'Content-Type: application/json' -d @"$OUT/turn2.json")
  echo "$code" > "$OUT/b.spec.code"; note "turn2 as the spec writes it -> HTTP $code"
  # corrected turn 2 (budget < max): this is what actually exercises gate 4
  code=$(curl -s -o "$OUT/b.msg.json" -w '%{http_code}' http://127.0.0.1:8314/v1/messages -H 'anthropic-version: 2023-06-01' -H 'Content-Type: application/json' -d @"$OUT/turn2fix.json")
  echo "$code" > "$OUT/b.code"
  gate_eq "run B replay of A's signature (gate 4)" "$code" 200
  curl -s http://127.0.0.1:8314/metrics | grep -i reasoning_signature | tee "$OUT/b.metrics"
  # tamper arm: does the server verify inbound signatures AT ALL?
  code=$(curl -s -o "$OUT/b.tamper.msg.json" -w '%{http_code}' http://127.0.0.1:8314/v1/messages -H 'anthropic-version: 2023-06-01' -H 'Content-Type: application/json' -d @"$OUT/turn2tamper.json")
  echo "$code" > "$OUT/b.tamper.code"; note "tampered signature -> HTTP $code"
  curl -s http://127.0.0.1:8314/metrics | grep -i reasoning_signature | tee "$OUT/b.tamper.metrics"
  jpy - <<'PY' | tee "$OUT/discover.txt"
from mlx2 import clients
m, c, v = clients.discover("http://127.0.0.1:8314", "local")
print("discover", m, c, v)
argv, env = clients.command("codex", "codex", "http://127.0.0.1:8314", m, c, "local", {})
print(argv)
PY
  stop_server

  note "Run C: control, --reasoning-signing-ephemeral"
  start_server "$OUT/c.log" --model "$NORTH" --host 127.0.0.1 --port 8314 --max-context 32768 --qualification-mode --api-state-dir "$STATE" --reasoning-signing-ephemeral
  wait_ready http://127.0.0.1:8314/health || { stop_server; _g_fail "run C server never became ready"; gate_verdict; return 1; }
  curl -s http://127.0.0.1:8314/v1/status | "$PY" -c 'import json,sys; print(json.load(sys.stdin)["reasoning_signing"])' | tee "$OUT/c.signing"
  code=$(curl -s -o "$OUT/c.msg.json" -w '%{http_code}' http://127.0.0.1:8314/v1/messages -H 'anthropic-version: 2023-06-01' -H 'Content-Type: application/json' -d @"$OUT/turn2fix.json")
  echo "$code" > "$OUT/c.code"; note "run C replay under an EPHEMERAL key -> HTTP $code"
  curl -s http://127.0.0.1:8314/metrics | grep -i reasoning_signature | tee "$OUT/c.metrics"
  stop_server
  gate_eq "a.log tracebacks" "$(grep -c Traceback "$OUT/a.log")" 0
  gate_eq "b.log tracebacks" "$(grep -c Traceback "$OUT/b.log")" 0
  gate_eq "c.log tracebacks" "$(grep -c Traceback "$OUT/c.log")" 0
  gate_eval 14 "$OUT"
  gate_verdict
}

job_13() {
  gate_reset
  assert_tree_under_test || return 1
  local OUT="$QUEUE_DIR/13-out"; mkdir -p "$OUT"
  local SMOKE="$QUEUE_DIR/13-progress-tokenize-smoke.py"
  local rc=0 M N
  for M in "$NORTH" "$FLASH"; do
    N="$(basename "$M")"
    note "model $N"
    start_server "$OUT/$N.log" --model "$M" --host 127.0.0.1 --port 8313 --max-context 32768 --qualification-mode
    wait_ready http://127.0.0.1:8313/health || { stop_server; rc=1; continue; }
    jpy "$SMOKE" http://127.0.0.1:8313 "$M" | tee "$OUT/$N.json" || rc=1
    jpy "$SMOKE" http://127.0.0.1:8313 "$M" | tee "$OUT/$N.warm.json" || rc=1
    curl -s http://127.0.0.1:8313/metrics | grep -E 'route="(tokenize|apply_template)"' | head -4
    stop_server
  done
  gate_true "job 13 commands all exited 0" "$rc"
  gate_eval 13 "$OUT"
  gate_verdict
}

job_12() {
  gate_reset
  assert_tree_under_test || return 1
  local S="$QUEUE_DIR/12-out"; mkdir -p "$S"
  echo '{"constrained_tool_grammar": true}' > "$S/policy-A.json"
  echo '{"constrained_tool_grammar": true, "constrained_tool_grammar_auto": true, "tool_grammar_streaming": true}' > "$S/policy-B.json"
  local rc=0 MODEL TAG

  run_arm_12() {  # tag model arm port policy
    local tag="$1" model="$2" arm="$3" port="$4" pol="$5"
    note "$tag arm $arm on port $port"
    start_server "$S/server-$tag-$arm.log" --model "$model" --port "$port" --qualification-mode --execution-policy "$pol"
    wait_ready "http://127.0.0.1:$port/health" || { stop_server; return 1; }
    jpy scripts/bench_tool_grammar.py --base "http://127.0.0.1:$port" \
      --repeats 3 --label "$arm" --output "$S/$tag-$arm.json" || rc=1
    curl -s "http://127.0.0.1:$port/metrics" | grep -E 'tool_calls|structured_output' > "$S/$tag-$arm.metrics"
    if [ "$arm" = "B" ] && [ "$tag" = "flashnext" ]; then
      curl -sN "http://127.0.0.1:$port/v1/responses" -H 'Content-Type: application/json' \
        -d '{"model":"'"$model"'","input":"Weather in Toronto in celsius? Use the tool.","stream":true,"store":false,"tools":[{"type":"function","name":"get_weather","strict":true,"parameters":{"type":"object","properties":{"city":{"type":"string"},"unit":{"type":"string","enum":["celsius","fahrenheit"]}},"required":["city","unit"],"additionalProperties":false}}]}' \
        > "$S/responses-B.sse"
    fi
    stop_server
    return 0
  }

  run_arm_12 flashnext "$FLASH" A 8412 "$S/policy-A.json" || rc=1
  run_arm_12 flashnext "$FLASH" B 8413 "$S/policy-B.json" || rc=1
  if [ "$QUEUE_FULL" -eq 1 ]; then
    run_arm_12 north "$NORTH" A 8412 "$S/policy-A.json" || rc=1
    run_arm_12 north "$NORTH" B 8413 "$S/policy-B.json" || rc=1
    run_arm_12 muse  "$MUSE"  A 8412 "$S/policy-A.json" || rc=1
    run_arm_12 muse  "$MUSE"  B 8413 "$S/policy-B.json" || rc=1
  else
    note "north/muse arms skipped (pass --full to include them; spec marks them 'if time allows')"
  fi
  gate_true "job 12 commands all exited 0" "$rc"
  gate_eval 12 "$S"
  gate_verdict
}

job_01() {
  gate_reset
  assert_tree_under_test || return 1
  local OUT="qualification/runs/splash-01-gpu-accept-count-$(date +%Y%m%d)"
  mkdir -p "$OUT"
  local rc=0

  note "step 1: kernel equality + micro A/B (no model load)"
  jpy scripts/bench_gpu_accept_count.py --device gpu \
    --widths 2,3,4,5,8,17 --rows 4 --repeats 200 --output "$OUT/kernel_ab.json" || { rc=1; note "GATE A/B FAILED"; }

  note "step 2: model-bound exactness (template then dynamic arm)"
  jpy scripts/qualify_qwen4_gdn_replay_model.py --model "$FLASH" \
    --widths 3,4,8 --repeats 3 --output "$OUT/replay_template.json" || { rc=1; note "template arm FAILED"; }
  jpy scripts/qualify_qwen4_gdn_replay_model.py --model "$FLASH" \
    --widths 3,4,8 --repeats 3 --dynamic-accept --output "$OUT/replay_dynamic.json" || { rc=1; note "GATE C/D FAILED"; }

  note "step 3: serving smoke + A/B (policy key on vs off)"
  echo '{"fused_gdn_dynamic_accept": true}' > "$OUT/policy-on.json"
  local arm
  for arm in off on; do
    # SPEC FIX: the spec writes one command with a literal '[--execution-policy ...]';
    # bracketed optional args are not runnable, so the two arms are split here.
    if [ "$arm" = on ]; then
      start_server "$OUT/server-on.log" --model "$FLASH" --port 8285 --max-lanes 4 --max-inflight 8 \
        --qualification-mode --execution-policy "$OUT/policy-on.json"
    else
      start_server "$OUT/server-off.log" --model "$FLASH" --port 8285 --max-lanes 4 --max-inflight 8 \
        --qualification-mode
    fi
    wait_ready http://127.0.0.1:8285/v1/status || { stop_server; rc=1; continue; }
    local i p
    i=0
    for p in "Explain the Metal unified memory model in one paragraph." \
             "Write a Python function that merges two sorted lists." \
             "List the first ten prime numbers separated by commas."; do
      i=$((i + 1))
      curl -s http://127.0.0.1:8285/v1/chat/completions -H 'Content-Type: application/json' \
        -d '{"messages":[{"role":"user","content":"'"$p"'"}],"max_tokens":256,"temperature":0}' \
        > "$OUT/greedy-$arm-p$i.json"
    done
    cpids=""
    for i in 1 2 3 4; do
      curl -s http://127.0.0.1:8285/v1/chat/completions -H 'Content-Type: application/json' \
        -d '{"messages":[{"role":"user","content":"Write a Python function that merges two sorted lists."}],"max_tokens":256,"temperature":0}' \
        > "$OUT/conc-$arm-$i.json" &
      cpids="$cpids $!"
    done
    wait_pids $cpids
    curl -s http://127.0.0.1:8285/v1/status > "$OUT/status-$arm.json"
    curl -s http://127.0.0.1:8285/metrics  > "$OUT/metrics-$arm.txt"
    jpy scripts/benchmark_serving.py --url http://127.0.0.1:8285 \
      --rounds 3 --widths 1 4 --max-tokens 160 --output "$OUT/bench-$arm.json" || rc=1
    stop_server
  done
  note "compare $OUT/greedy-on-*.json vs greedy-off-*.json byte-for-byte (gate E)"
  gate_true "job 01 commands all exited 0" "$rc"
  gate_eval 01 "$OUT"
  gate_verdict
}

job_08() {
  gate_reset
  assert_tree_under_test || return 1
  local OUT="qualification/runs/suspend-replay-$(date +%Y%m%d)"; mkdir -p "$OUT"
  echo '{}' > "$OUT/ord-off.json"
  echo '{"memory_preemption": {"enabled": true, "stall_seconds": 60, "on_pressure": true}}' > "$OUT/ord-on.json"
  echo '{"num_draft": 2}' > "$OUT/mtp-off.json"
  echo '{"num_draft": 2, "memory_preemption": {"enabled": true, "stall_seconds": 60, "on_pressure": true}}' > "$OUT/mtp-on.json"
  local rc=0 ARM FLAG
  # SPEC FIX: the spec states the arm loop in prose; it is expanded here.
  for ARM in ord-off ord-on mtp-off mtp-on; do
    case "$ARM" in ord-*) FLAG=--ordinary ;; mtp-*) FLAG=--native-mtp ;; esac
    note "arm $ARM ($FLAG)"
    start_server "$OUT/$ARM.server.log" --model "$HYBRID" --host 127.0.0.1 --port 8298 \
      --max-context 32768 --max-lanes 4 --max-inflight 8 \
      --execution-policy "$OUT/$ARM.json" --qualification-mode "$FLAG"
    wait_ready http://127.0.0.1:8298/v1/status || { stop_server; rc=1; continue; }
    curl -s http://127.0.0.1:8298/v1/chat/completions -H 'Content-Type: application/json' \
      -d '{"messages":[{"role":"user","content":"Say hello in five words."}],"temperature":0,"max_tokens":16,"enable_thinking":false}' \
      > "$OUT/$ARM.smoke.json"
    jpy scripts/bench_suspend_replay.py --url http://127.0.0.1:8298 \
      --output "$OUT/$ARM.bench.json" --prompt-repeats 400 --max-tokens 64 --fault-after 8 || rc=1
    curl -s http://127.0.0.1:8298/v1/status > "$OUT/$ARM.status.json"
    curl -s http://127.0.0.1:8298/metrics  > "$OUT/$ARM.metrics.txt"
    if [ "$ARM" = ord-on ]; then
      curl -s -X POST http://127.0.0.1:8298/v1/admin/quiesce -H 'Content-Type: application/json' \
        -d '{"suspend": false}' > "$OUT/quiesce.json"
      curl -s http://127.0.0.1:8298/v1/status > "$OUT/quiesce-status.json"
      curl -s -X POST http://127.0.0.1:8298/v1/admin/resume > "$OUT/resume.json"
    fi
    stop_server
  done
  gate_true "job 08 commands all exited 0" "$rc"
  gate_eval 08 "$OUT"
  gate_verdict
}

job_06() {
  gate_reset
  assert_tree_under_test || return 1
  local OUT="qualification/runs/junction-snapshots-$(date +%Y%m%d)"; mkdir -p "$OUT"
  echo '{"num_draft": 2}' > "$OUT/mtp-off.json"
  echo '{"num_draft": 2, "apc_junction_checkpoints": true}' > "$OUT/mtp-on.json"
  echo '{}' > "$OUT/ord-off.json"
  echo '{"apc_junction_checkpoints": true}' > "$OUT/ord-on.json"
  local rc=0 ARM FLAG
  # SPEC FIX: arm loop expanded from prose.
  for ARM in mtp-off mtp-on ord-off ord-on; do
    case "$ARM" in ord-*) FLAG=--ordinary ;; mtp-*) FLAG=--native-mtp ;; esac
    note "arm $ARM ($FLAG)"
    start_server "$OUT/$ARM.server.log" --model "$HYBRID" --host 127.0.0.1 --port 8297 \
      --max-context 65536 --max-lanes 4 --max-inflight 8 \
      --execution-policy "$OUT/$ARM.json" --qualification-mode "$FLAG"
    wait_ready http://127.0.0.1:8297/v1/status || { stop_server; rc=1; continue; }
    curl -s http://127.0.0.1:8297/v1/chat/completions -H 'Content-Type: application/json' \
      -d '{"messages":[{"role":"user","content":"Say hello in five words."}],"temperature":0,"max_tokens":16,"enable_thinking":false}' \
      > "$OUT/$ARM.smoke.json"
    jpy scripts/bench_06_junction_snapshots.py --url http://127.0.0.1:8297 \
      --output "$OUT/$ARM.bench.json" --system-repeats 1200 --turns 5 --max-tokens 16 || rc=1
    curl -s http://127.0.0.1:8297/v1/status > "$OUT/$ARM.status.json"
    stop_server
  done
  gate_true "job 06 commands all exited 0" "$rc"
  gate_eval 06 "$OUT"
  gate_verdict
}

job_07() {
  gate_reset
  assert_tree_under_test || return 1
  local OUT="qualification/runs/rolling-prefill-ckpt-$(date +%Y%m%d)"; mkdir -p "$OUT"
  echo '{"num_draft": 2}' > "$OUT/mtp-off.json"
  echo '{"num_draft": 2, "apc_rolling_checkpoints": {"interval_tokens": 4096}}' > "$OUT/mtp-on.json"
  echo '{}' > "$OUT/ord-off.json"
  echo '{"apc_rolling_checkpoints": {"interval_tokens": 4096}}' > "$OUT/ord-on.json"
  local rc=0 ARM FLAG
  # SPEC FIX: arm loop expanded from prose.
  for ARM in mtp-off mtp-on ord-off ord-on; do
    case "$ARM" in ord-*) FLAG=--ordinary ;; mtp-*) FLAG=--native-mtp ;; esac
    note "arm $ARM ($FLAG)"
    start_server "$OUT/$ARM.server.log" --model "$HYBRID" --host 127.0.0.1 --port 8298 \
      --max-context 65536 --max-lanes 4 --max-inflight 8 \
      --execution-policy "$OUT/$ARM.json" --qualification-mode "$FLAG"
    wait_ready http://127.0.0.1:8298/v1/status || { stop_server; rc=1; continue; }
    curl -s http://127.0.0.1:8298/v1/chat/completions -H 'Content-Type: application/json' \
      -d '{"messages":[{"role":"user","content":"Say hello in five words."}],"temperature":0,"max_tokens":16,"enable_thinking":false}' \
      > "$OUT/$ARM.smoke.json"
    jpy scripts/bench_rolling_prefill_ckpt.py --url http://127.0.0.1:8298 \
      --output "$OUT/$ARM.bench.json" --repeats 2400 --cancel-after 6 --stagger 4 || rc=1
    curl -s http://127.0.0.1:8298/v1/status > "$OUT/$ARM.status.json"
    stop_server
  done
  note "if the off-arm cancel_retry shows no cancellation, rerun both arms of that route with --repeats 4800"
  gate_true "job 07 commands all exited 0" "$rc"
  gate_eval 07 "$OUT"
  gate_verdict
}

job_10() {
  gate_reset
  assert_tree_under_test || return 1
  local OUT="qualification/runs/srpt-prefill-$(date +%Y%m%d)"; mkdir -p "$OUT"
  echo '{"num_draft": 2}' > "$OUT/mtp-off.json"
  echo '{"num_draft": 2, "prefill_scheduling": {"order": "srpt", "max_bypass": 3, "one_slice_contention": true}}' > "$OUT/mtp-on.json"
  echo '{}' > "$OUT/ord-off.json"
  echo '{"prefill_scheduling": {"order": "srpt", "max_bypass": 3, "one_slice_contention": true}}' > "$OUT/ord-on.json"
  local rc=0 ARM FLAG
  # SPEC FIX: arm loop expanded from prose.
  for ARM in mtp-off mtp-on ord-off ord-on; do
    case "$ARM" in ord-*) FLAG=--ordinary ;; mtp-*) FLAG=--native-mtp ;; esac
    note "arm $ARM ($FLAG)"
    start_server "$OUT/$ARM.server.log" --model "$HYBRID" --host 127.0.0.1 --port 8297 \
      --max-context 65536 --max-lanes 4 --max-inflight 8 \
      --execution-policy "$OUT/$ARM.json" --qualification-mode "$FLAG"
    wait_ready http://127.0.0.1:8297/v1/status || { stop_server; rc=1; continue; }
    curl -s http://127.0.0.1:8297/v1/chat/completions -H 'Content-Type: application/json' \
      -d '{"messages":[{"role":"user","content":"Say hello in five words."}],"temperature":0,"max_tokens":16,"enable_thinking":false}' \
      > "$OUT/$ARM.smoke.json"
    jpy scripts/bench_srpt_prefill.py --url http://127.0.0.1:8297 \
      --output "$OUT/$ARM.bench.json" --long-repeats 900 --shorts 6 --short-delay 1.0 --short-interval 0.5 --max-tokens 32 || rc=1
    jpy scripts/bench_srpt_prefill.py --url http://127.0.0.1:8297 \
      --output "$OUT/$ARM.bench2.json" --long-repeats 900 --shorts 6 --short-delay 1.0 --short-interval 0.5 --max-tokens 32 || rc=1
    # HARNESS FIX: spec 10 gates 1 and 4 are read out of /v1/status (settings
    # echo, prefill_scheduling_bypasses/_bypass_forced/_one_slice_clamps), and
    # the original job body never captured it, so those gates were unevaluable.
    curl -s http://127.0.0.1:8297/v1/status > "$OUT/$ARM.status.json"
    curl -s http://127.0.0.1:8297/metrics  > "$OUT/$ARM.metrics.txt"
    stop_server
  done
  gate_true "job 10 commands all exited 0" "$rc"
  gate_eval 10 "$OUT"
  gate_verdict
}

job_04() {
  gate_reset
  assert_tree_under_test || return 1
  local rc=0
  note "step 1: in-process A/B (correctness + snapshot cost)"
  jpy scripts/bench_04_gdn_double_buffer.py \
    --prompt-tokens 4096 --max-tokens 128 --width 4 \
    --output "$QUEUE_DIR/04-result.json" || rc=1

  note "step 2: serving smoke (default mode)"
  printf '{"draft_model": "%s", "num_draft": 4}\n' "$MUSE_DRAFT" > /tmp/muse-dflash2-policy.json
  start_server "$QUEUE_DIR/logs-04-server.log" --model "$MUSE" --port 8285 --external-draft \
    --execution-policy /tmp/muse-dflash2-policy.json --max-lanes 4 --max-inflight 8 --qualification-mode
  wait_ready http://127.0.0.1:8285/v1/status || { stop_server; return 1; }
  local req='{"messages":[{"role":"user","content":"Write a Python function that merges two sorted lists."}],"max_tokens":200,"temperature":0}'
  curl -s localhost:8285/v1/chat/completions -H 'content-type: application/json' -d "$req" | "$PY" -m json.tool | head -40
  curl -s localhost:8285/metrics | grep -E 'external_speculative|recovery'
  # same prompt again -> APC hit path
  curl -s localhost:8285/v1/chat/completions -H 'content-type: application/json' -d "$req" | "$PY" -m json.tool | head -10
  jpy scripts/qualify_serving.py --url http://127.0.0.1:8285 \
    --output "$QUEUE_DIR/04-qualify.json" \
    --require-feature external_draft --require-feature segmented_transaction || rc=1
  stop_server

  note "step 3 (optional default decision): repeat the curl with MLX_LM_EXTERNAL_ROUND_COW=1"
  MLX_LM_EXTERNAL_ROUND_COW=1 jpy -u -m mlx2.server --model "$MUSE" --port 8285 \
    --external-draft --execution-policy /tmp/muse-dflash2-policy.json --max-lanes 4 --max-inflight 8 \
    --qualification-mode > "$QUEUE_DIR/logs-04-server-cow.log" 2>&1 &
  SERVER_PID=$!
  if wait_ready http://127.0.0.1:8285/v1/status; then
    curl -s localhost:8285/v1/chat/completions -H 'content-type: application/json' -d "$req" | "$PY" -m json.tool | head -20
    curl -s localhost:8285/metrics | grep -E 'external_cow'
  else
    rc=1
  fi
  stop_server
  gate_true "job 04 commands all exited 0" "$rc"
  gate_eval 04 "$QUEUE_DIR"
  gate_verdict
}

job_02() {
  gate_reset
  assert_tree_under_test || return 1
  local rc=0
  local Q2="$QUEUE_DIR/02-out"; rm -rf "$Q2"; mkdir -p "$Q2"
  note "step 1: A/B bench (host vs batched)"
  jpy scripts/bench_dflash_pair_select.py \
    --num-draft 4 --widths 1,4,8 --trials 20 --max-tokens 128 --out /tmp/pair-select-k4.json || rc=1
  jpy scripts/bench_dflash_pair_select.py \
    --num-draft 7 --widths 1,8 --trials 20 --max-tokens 128 --out /tmp/pair-select-k7.json || rc=1

  note "step 2: serving smoke (feature on, real route)"
  cat > /tmp/muse-dflash2-batched.json <<J
{"draft_model": "$MUSE_DRAFT", "num_draft": 4, "pairwise_selection": "batched"}
J
  # HARNESS FIX: spec 02 step 2 omits --qualification-mode, which this tree now
  # requires ("server.py: error: provide --qualification or explicitly run
  # --qualification-mode"), so the smoke could not start as written.
  start_server "$QUEUE_DIR/logs-02-server.log" --model "$MUSE" --host 127.0.0.1 --port 8295 \
    --max-context 32768 --max-lanes 8 --execution-policy /tmp/muse-dflash2-batched.json --external-draft --qualification-mode
  wait_ready http://127.0.0.1:8295/v1/status || { stop_server; _g_fail "02 serving smoke server never became ready"; gate_verdict; return 1; }
  record_route http://127.0.0.1:8295 external_draft
  resolve_served_id http://127.0.0.1:8295
  curl -s -o "$Q2/poem.json" -w 'poem %{http_code}\n' 127.0.0.1:8295/v1/chat/completions -H 'content-type: application/json' \
    -d "$(mb '{"model":"x","messages":[{"role":"user","content":"Write a short poem about rivers."}],"max_tokens":96,"temperature":0.8}')" | tee "$Q2/poem.code"
  curl -s -o "$Q2/math.json" -w 'math %{http_code}\n' 127.0.0.1:8295/v1/chat/completions -H 'content-type: application/json' \
    -d "$(mb '{"model":"x","messages":[{"role":"user","content":"What is 17*23?"}],"max_tokens":64,"temperature":0}')" | tee "$Q2/math.code"
  curl -s -o "$Q2/tool.json" -w 'tool %{http_code}\n' 127.0.0.1:8295/v1/chat/completions -H 'content-type: application/json' \
    -d "$(mb '{"model":"x","messages":[{"role":"user","content":"Weather in Paris?"}],"tools":[{"type":"function","function":{"name":"get_weather","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}],"max_tokens":96}')" | tee "$Q2/tool.code"
  curl -s 127.0.0.1:8295/metrics > "$Q2/metrics.txt"; grep pairwise_selection "$Q2/metrics.txt"
  curl -s 127.0.0.1:8295/v1/status > "$Q2/status.json"; "$PY" -m json.tool < "$Q2/status.json" | grep -i pairwise
  stop_server
  gate_true "job 02 commands all exited 0" "$rc"
  gate_eval 02 "$Q2"
  gate_verdict
}

job_05() {
  gate_reset
  assert_tree_under_test || return 1
  local S="$QUEUE_DIR/05-out"; mkdir -p "$S"
  local rc=0
  note "step 1: micro (no model) device vs host verifier"
  jpy scripts/bench_gpu_topk_accept.py micro --batch 4 --depth 4 --vocab 202048 --k 32 || rc=1
  jpy scripts/bench_gpu_topk_accept.py micro --batch 1 --depth 4 --vocab 202048 --k 32 || rc=1
  jpy -m pytest tests/test_gpu_topk_accept.py -q -o addopts="" || rc=1

  "$PY" - "$S" <<'PY'
import json, sys
s = sys.argv[1].rstrip("/")
base = json.load(open("qualification/policies/muse-dflash2.json"))
json.dump(base, open(f"{s}/05-off.json", "w"))
json.dump({**base, "gpu_acceptance": {"enabled": True, "top_k": 32}}, open(f"{s}/05-on.json", "w"))
PY

  note "step 2: serve A/B (policy off vs on)"
  local arm
  for arm in off on; do
    start_server "$S/05-server-$arm.log" --model "$MUSE" --port 8285 --external-draft \
      --execution-policy "$S/05-$arm.json" --max-lanes 4 --qualification-mode
    wait_ready http://127.0.0.1:8285/health || { stop_server; rc=1; continue; }
    jpy scripts/bench_gpu_topk_accept.py serve --label "$arm" --requests 16 --max-tokens 256 \
      | tee "$S/05-serve-$arm.json" || rc=1
    curl -s http://127.0.0.1:8285/v1/chat/completions -H 'Content-Type: application/json' \
      -d '{"messages":[{"role":"user","content":"Say hi"}],"max_tokens":16,"temperature":0}' | tee "$S/05-greedy-$arm.json"
    curl -s http://127.0.0.1:8285/v1/chat/completions -H 'Content-Type: application/json' \
      -d '{"messages":[{"role":"user","content":"Say hi"}],"max_tokens":16,"temperature":1.0,"logprobs":true,"top_logprobs":2}' | tee "$S/05-logprobs-$arm.json"
    curl -s http://127.0.0.1:8285/v1/status | tee "$S/05-status-$arm.json" >/dev/null
    stop_server
  done
  gate_true "job 05 commands all exited 0" "$rc"
  gate_eval 05 "$S"
  gate_verdict
}

job_03() {
  gate_reset
  assert_tree_under_test || return 1
  local rc=0
  note "step 0: pre-flight CPU test"
  jpy -m pytest tests/test_dflash_long_block.py -q || rc=1

  local OUT="qualification/runs/dflash-long-block-$(date +%Y%m%d)"; mkdir -p "$OUT"
  note "step 1a: natural-stop sweep"
  jpy -u scripts/bench_external_block_size.py \
    --num-draft 2,3,4,5,6,7,8,10,12,15 --batch 1,4 --temps 0,1.0 \
    --max-tokens 256 --repeats 1 --out "$OUT/sweep.json" 2>&1 | tee "$OUT/sweep.log" || rc=1

  # SPEC FIX: the spec's 1b command contains the literal placeholder
  # `<best-from-1a>`, which is both a shell redirect and an unparseable K.
  # Unattended we confirm the fixed finalists; K* from 1a must be confirmed by
  # hand afterwards if it is not already in this list.
  local CONFIRM_K="${QUEUE_03_CONFIRM_K:-4,7,15}"
  note "step 1b: fixed-length confirmation on K=$CONFIRM_K (override with QUEUE_03_CONFIRM_K)"
  jpy -u scripts/bench_external_block_size.py \
    --num-draft "$CONFIRM_K" --batch 1,4 --temps 0,1.0 \
    --max-tokens 384 --repeats 3 --ignore-eos --out "$OUT/confirm.json" 2>&1 | tee "$OUT/confirm.log" || rc=1

  note "step 2: serving-path correctness smoke at K=15"
  printf '{"draft_model": "%s", "num_draft": 15}\n' "$MUSE_DRAFT" > /tmp/muse-dflash2-k15.json
  start_server "$OUT/serve-k15.server.log" --model "$MUSE" --port 8296 --max-lanes 4 \
    --execution-policy /tmp/muse-dflash2-k15.json --external-draft --qualification-mode
  wait_ready http://127.0.0.1:8296/v1/status || { stop_server; return 1; }
  curl -s localhost:8296/v1/chat/completions -H 'Content-Type: application/json' \
    -d '{"messages":[{"role":"user","content":"Count from 1 to 40 separated by commas."}],"max_tokens":200,"temperature":0}' | tee "$OUT/serve-k15.json"
  local i
  local cpids=""
  for i in 1 2 3 4; do
    curl -s localhost:8296/v1/chat/completions -H 'Content-Type: application/json' \
      -d '{"messages":[{"role":"user","content":"Write a haiku about rivers."}],"max_tokens":120,"temperature":1.0,"top_p":0.95}' > "$OUT/serve-k15-s$i.json" &
    cpids="$cpids $!"
  done
  wait_pids $cpids
  curl -s localhost:8296/v1/status > "$OUT/status-k15.json"
  stop_server
  gate_true "job 03 commands all exited 0" "$rc"
  gate_eval 03 "$OUT"
  gate_verdict
}

# ---------------------------------------------------- harness self-check
# 2026-09-20: building v2 spliced out the whole gate-enforcement block -- the
# runner ran a spec with NO gates at all and only failed because bash errors
# on an undefined function.  A refactor silently removing the checks is the
# same failure family as a check that never checks, so the harness now asserts
# its own preconditions before it can take a lease.  Fail closed, at startup.
harness_self_check() {
  local missing="" fn
  for fn in gate_reset gate_eq gate_ne gate_true gate_verdict gate_eval \
            _g_pass _g_fail wait_ready start_server stop_server resolve_served_id \
            mb wait_pids assert_tree_under_test record_route \
            lease_acquire lease_release waiter_register waiter_remove \
            waiter_clear_to_go waiter_heartbeat is_our_descendant sweep_ports; do
    type "$fn" >/dev/null 2>&1 || missing="$missing $fn"
  done
  if [ -n "$missing" ]; then
    say "FATAL: harness is missing required helper(s):$missing"
    say "Refusing to run - a runner without its gate helpers reports nothing useful."
    exit 4
  fi
  [ -f "$QUEUE_DIR/gates.py" ] || { say "FATAL: $QUEUE_DIR/gates.py not found"; exit 4; }
  say "harness self-check: all gate/lease helpers present, gates.py found"
}

# ============================================================ arg parsing ==
while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run)           DRY_RUN=1 ;;
    --list)              LIST=1 ;;
    --only)              ONLY="$2"; shift ;;
    --only=*)            ONLY="${1#--only=}" ;;
    --force-stale-lock)  FORCE_STALE=1 ;;
    --no-lease)          TAKE_LEASE=0 ;;
    --full)              QUEUE_FULL=1 ;;
    --timeout-mult)      TIMEOUT_MULT="$2"; shift ;;
    -h|--help)
      sed -n '2,12p' "$0"; exit 0 ;;
    *) say "unknown option: $1"; exit 2 ;;
  esac
  shift
done

selected() {  # job id -> 0 if it should run
  local id="$1" tok
  [ -z "$ONLY" ] && return 0
  for tok in $(printf '%s' "$ONLY" | tr ',' ' '); do
    [ "$tok" = "$id" ] && return 0
  done
  return 1
}

# ============================================================== validation ==
validate_job() {  # index -> 0 ok, 1 problem (prints findings)
  local i="$1" bad=0 wt="${J_WT[$i]}" p m
  if [ ! -d "$wt" ]; then say "    MISSING worktree $wt"; return 1; fi
  for p in ${J_REQ[$i]}; do
    if [ -e "$wt/$p" ]; then say "    ok   $wt/$p"; else say "    MISSING $wt/$p"; bad=1; fi
  done
  for m in ${J_MODELS[$i]}; do
    if [ -d "$m" ]; then say "    ok   model $m"; else say "    MISSING model $m"; bad=1; fi
  done
  return "$bad"
}

print_body() {  # index
  local id="${J_ID[$1]}"
  say "    --- commands (function job_$id, cwd=${J_WT[$1]}) ---"
  declare -f "job_$id" | sed -n '3,$p' | sed 's/^/    /'
}

total_est=0
i=0
while [ "$i" -lt "${#J_ID[@]}" ]; do
  selected "${J_ID[$i]}" && total_est=$((total_est + J_EST[i]))
  i=$((i + 1))
done

if [ "$LIST" -eq 1 ]; then
  say "pos  job  slug                   model(s)                                   est   kind  exclusive"
  i=0
  while [ "$i" -lt "${#J_ID[@]}" ]; do
    printf '%3d  %s   %-22s %-42s %3dm  %-5s %s\n' "$((i + 1))" "${J_ID[$i]}" "${J_SLUG[$i]}" \
      "$(for m in ${J_MODELS[$i]}; do basename "$m"; done | tr '\n' '+' | sed 's/+$//')" \
      "${J_EST[$i]}" "${J_KIND[$i]}" "${J_EXCL[$i]}"
    i=$((i + 1))
  done
  say ""
  say "total estimate: ${total_est} min"
  exit 0
fi

# =================================================================== dry run ==
if [ "$DRY_RUN" -eq 1 ]; then
  say "=== run_queue.sh --dry-run  ($(ts)) ==="
  say "queue dir : $QUEUE_DIR"
  say "log root  : $LOG_ROOT/<timestamp>/"
  say "python    : $PY $([ -x "$PY" ] && echo '(ok)' || echo '(MISSING)')"
  say "bash      : $BASH_VERSION"
  say 'timeout   : no `timeout` binary used; per-job watchdog polls kill -0 and kills the job tree'
  say "            watchdog budget = ${TIMEOUT_MULT}x estimate (floor 20m)"
  say ""
  say "--- GPU holder guard ---"
  gpu_guard; guard_rc=$?
  case "$guard_rc" in
    0) say "  GPU is FREE (no live receipt, no mlx2.server, no listener on queue ports)" ;;
    1) say "  -> a real run would WAIT for this peer at its first per-spec acquire" ;;
    2) say "  -> only stale receipts; a real run would reclaim them at the first acquire" ;;
  esac
  say ""
  problems=0
  i=0
  while [ "$i" -lt "${#J_ID[@]}" ]; do
    if selected "${J_ID[$i]}"; then
      say "[$((i + 1))/${#J_ID[@]}] job ${J_ID[$i]}-${J_SLUG[$i]}  (${J_EST[$i]} min, ${J_KIND[$i]}, exclusive=${J_EXCL[$i]})"
      say "    worktree : ${J_WT[$i]}"
      say "    log      : $LOG_ROOT/<timestamp>/${J_ID[$i]}-${J_SLUG[$i]}.log"
      tmo=$((J_EST[i] * 60 * TIMEOUT_MULT)); [ "$tmo" -lt 1200 ] && tmo=1200
      say "    timeout  : ${tmo}s"
      validate_job "$i" || problems=$((problems + 1))
      print_body "$i"
      say ""
    fi
    i=$((i + 1))
  done
  say "=== dry run summary ==="
  say "jobs selected : $(if [ -z "$ONLY" ]; then echo "${#J_ID[@]} (all)"; else echo "$ONLY"; fi)"
  say "path problems : $problems job(s)"
  say "wall estimate : ${total_est} min (~$((total_est / 60))h$((total_est % 60))m) plus load/teardown slack"
  if [ "$problems" -gt 0 ]; then exit 1; fi
  exit 0
fi

# ================================================================== real run ==
RUN_STAMP="$(date +%Y%m%d-%H%M%S)"
RUN_LOG_DIR="$LOG_ROOT/$RUN_STAMP"
mkdir -p "$RUN_LOG_DIR"
rm -f "$LOG_ROOT/latest"; ln -s "$RUN_STAMP" "$LOG_ROOT/latest" 2>/dev/null

# The lease is now taken PER SPEC, not per batch, so a peer holding the GPU at
# batch start is no longer a reason to refuse: we wait for it per job.  The
# guard runs once for the record.
harness_self_check
say "=== GPU holder guard (informational; the lease is taken per spec) ==="
gpu_guard; guard_rc=$?
case "$guard_rc" in
  0) say "  GPU is free right now." ;;
  1) say "  GPU is currently held by a peer. Per-spec acquire will WAIT for it." ;;
  2) say "  stale receipt(s) only; per-spec acquire will reclaim them." ;;
esac

cleanup() {
  stop_server
  sweep_ports
  lease_release     # must never leave a stale receipt behind
}
trap 'say ""; note "interrupted"; cleanup; exit 130' INT TERM
trap 'cleanup' EXIT

R_ID=(); R_STATUS=(); R_SECS=(); R_LOG=()
FIRST_SPEC=1
i=0
while [ "$i" -lt "${#J_ID[@]}" ]; do
  id="${J_ID[$i]}"
  if ! selected "$id"; then i=$((i + 1)); continue; fi
  slug="${J_SLUG[$i]}"; wt="${J_WT[$i]}"
  log="$RUN_LOG_DIR/$id-$slug.log"
  tmo=$((J_EST[i] * 60 * TIMEOUT_MULT)); [ "$tmo" -lt 1200 ] && tmo=1200

  say ""
  say "=== [$id-$slug] start $(ts)  est ${J_EST[$i]}m  timeout ${tmo}s ==="
  # Identify the tree under test unambiguously: main has moved to 9f24c82
  # while the feature branches are still based on 8bde1dd, so "which tree"
  # is part of the evidence, not a detail.
  wt_branch="$(git -C "$wt" rev-parse --abbrev-ref HEAD 2>/dev/null || echo '?')"
  wt_head="$(git -C "$wt" rev-parse --short HEAD 2>/dev/null || echo '?')"
  wt_base="$(git -C "$wt" merge-base HEAD main 2>/dev/null | cut -c1-7 || echo '?')"
  {
    echo "### job $id-$slug"
    echo "### started $(ts)"
    echo "### worktree $wt"
    echo "### branch   $wt_branch @ $wt_head  (merge-base with main: $wt_base)"
    echo "### models ${J_MODELS[$i]}"
    echo "### runner  repaired 2026-09-20 (real gate enforcement, per-spec lease)"
  } > "$log"
  say "    tree: $wt_branch @ $wt_head (base $wt_base)"

  vout="$(validate_job "$i" 2>&1)"
  echo "$vout" >> "$log"
  if echo "$vout" | grep -q MISSING; then
    say "  SKIPPED: missing paths (see log)"
    R_ID+=("$id-$slug"); R_STATUS+=("skipped/missing-paths"); R_SECS+=(0); R_LOG+=("$log")
    i=$((i + 1)); continue
  fi

  # ---- per-spec lease: acquire, run, release on EVERY exit path ----
  # NB: no pipeline here.  `lease_acquire | tee` would run the function in a
  # SUBSHELL, so LEASE_HELD would be set in the child and lost in the parent --
  # the lease would then never be released and the peer queue would starve,
  # which is the exact failure this repair exists to remove.  Redirection alone
  # keeps the function in the current shell.
  # INTER-SPEC YIELD GAP: release-then-instantly-reacquire means the lock is
  # never observed free, so peers polling every few seconds never win a gap.
  # Sleep before even asking, so a fast poller can take it and our own acquire
  # then waits on them through the normal foreign-lease path.
  if [ "$FIRST_SPEC" -eq 0 ] && [ "$TAKE_LEASE" -eq 1 ]; then
    note "lease: yielding ${YIELD_GAP}s before next spec ($id-$slug)"
    echo "### [runner] yielding ${YIELD_GAP}s before next spec at $(ts)" >> "$log"
    sleep "$YIELD_GAP"
  fi
  FIRST_SPEC=0
  lease_want_t0=$(date +%s)
  lease_acquire "splash-$id-$slug" >> "$log" 2>&1
  grep -E "lease: (WAITING|ACQUIRED|reclaiming|yielding)|waiter: (REGISTERED|YIELDING|discarding)" "$log" | tail -4
  lease_t0=$(date +%s)
  echo "### lease wanted-at $(date -r "$lease_want_t0" +%Y-%m-%dT%H:%M:%S) acquired-at $(date -r "$lease_t0" +%Y-%m-%dT%H:%M:%S) queued-for $((lease_t0 - lease_want_t0))s" >> "$log"

  t0=$(date +%s)
  JOB_WT="$wt"
  ( cd "$wt" && JOB_WT="$wt" && "job_$id" ) >> "$log" 2>&1 &
  jpid=$!
  waited=0; rc=0
  while kill -0 "$jpid" 2>/dev/null; do
    if [ "$waited" -ge "$tmo" ]; then
      echo "### [runner] TIMEOUT after ${tmo}s - killing job tree" >> "$log"
      kill_tree TERM "$jpid"; sleep 10; kill_tree KILL "$jpid"
      wait "$jpid" 2>/dev/null
      rc=124
      break
    fi
    # Lease ceiling: enforced even when the job's own watchdog is far longer,
    # so one spec can never breach the interleaving contract unattended.
    if [ "$TAKE_LEASE" -eq 1 ] && [ $(( $(date +%s) - lease_t0 )) -ge "$LEASE_CEILING" ]; then
      echo "### [runner] LEASE CEILING ${LEASE_CEILING}s reached - killing job tree and releasing" >> "$log"
      note "LEASE CEILING: $id-$slug held the GPU ${LEASE_CEILING}s; killing and releasing"
      kill_tree TERM "$jpid"; sleep 10; kill_tree KILL "$jpid"
      wait "$jpid" 2>/dev/null
      rc=125
      break
    fi
    sleep 5; waited=$((waited + 5))
  done
  if [ "$rc" -ne 124 ]; then wait "$jpid"; rc=$?; fi
  t1=$(date +%s); dur=$((t1 - t0))

  sweep_ports
  # Release in the SAME path as the timeout kill, so a dead job never leaves a
  # receipt behind and the peer queue can interleave between specs.
  lease_release >> "$log" 2>&1      # same subshell hazard: redirect, never pipe
  grep -E "lease: (RELEASED|NOT releasing)|waiter: removed" "$log" | tail -3
  lease_t1=$(date +%s); lease_dur=$((lease_t1 - lease_t0))
  # Emitted for EVERY spec -- ok, FAILED and TIMEOUT alike -- because this line
  # is what makes the interleaving contract with the peer sessions auditable.
  echo "### lease held ${lease_dur}s for $id-$slug (wanted $(date -r "$lease_want_t0" +%H:%M:%S), acquired $(date -r "$lease_t0" +%H:%M:%S), released $(date -r "$lease_t1" +%H:%M:%S), queued $((lease_t0 - lease_want_t0))s)" >> "$log"
  say "    lease: queued $((lease_t0 - lease_want_t0))s, held ${lease_dur}s"
  if [ "$lease_dur" -gt 3300 ]; then
    note "WARNING: $id-$slug held the GPU lease for ${lease_dur}s (>55m budget)"
  fi
  case "$rc" in
    0)   st="ok" ;;
    124) st="TIMEOUT" ;;
    125) st="LEASE-CEILING" ;;
    *)   st="FAILED(rc=$rc)" ;;
  esac
  echo "### finished $(ts) status=$st duration=${dur}s" >> "$log"
  say "=== [$id-$slug] $st in $((dur / 60))m$((dur % 60))s ==="
  R_ID+=("$id-$slug"); R_STATUS+=("$st"); R_SECS+=("$dur"); R_LOG+=("$log")
  i=$((i + 1))
done

say ""
say "================================ SUMMARY ================================"
printf '%-28s %-16s %10s  %s\n' "JOB" "STATUS" "DURATION" "LOG"
grand=0; nfail=0
i=0
while [ "$i" -lt "${#R_ID[@]}" ]; do
  printf '%-28s %-16s %7dm%02ds  %s\n' "${R_ID[$i]}" "${R_STATUS[$i]}" \
    "$((R_SECS[i] / 60))" "$((R_SECS[i] % 60))" "${R_LOG[$i]}"
  grand=$((grand + R_SECS[i]))
  [ "${R_STATUS[$i]}" = "ok" ] || nfail=$((nfail + 1))
  i=$((i + 1))
done
say "-------------------------------------------------------------------------"
printf '%-28s %-16s %7dm%02ds  %s\n' "TOTAL (${#R_ID[@]} jobs)" "$nfail not ok" "$((grand / 60))" "$((grand % 60))" "$RUN_LOG_DIR"
say "========================================================================="
[ "$nfail" -eq 0 ] || exit 1
exit 0
