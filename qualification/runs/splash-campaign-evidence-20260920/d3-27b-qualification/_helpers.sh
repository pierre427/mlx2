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
