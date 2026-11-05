"""Build one real32K Q&A history, then compare continuing policy branches.

The history is actual model-generated dialogue, not repeated filler. The
same frozen history seeds every continuation; earlier generated assertions
are context, not a correctness oracle. User-provided design facts stay exact.
"""
import argparse
import gc
import hashlib
import json
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/research"))
import kimi_expert_budget_probe as budget

from mlx2.desktop_campaign_process import spawn
from mlx2.desktop_campaign_worker import atomic, check_ownership

SYSTEM = ("Discuss reliable distributed payment processing as an engineering tutor. "
          "Keep user-provided design constraints authoritative; correct earlier assistant mistakes when relevant. "
          "Explain mechanisms, worked examples, failure cases and tradeoffs. Do not produce hidden reasoning.")
SEED = ("We are designing a fictional payment platform. Record these agreed constraints for the whole discussion: "
        "currency CAD; three availability zones; idempotency records retained for72hours; maximum6charge attempts "
        "per logical payment; ledger RPO zero; recovery-time objective15minutes; downstream event delivery at least once. "
        "A duplicate identical payment request within the retention window must return its committed result without "
        "another charge. The same key with a different payload must be rejected. Secrets must never enter logs. "
        "Give a concrete overview of how the payment API, ledger, idempotency records and transactional outbox fit together.")
QUESTIONS = [
    "Explain atomic recording of payment outcome and idempotency key, including concurrent retries.",
    "Distinguish client timeout, failed payment, unknown outcome and successful payment.",
    "Design a retry budget with exponential backoff and jitter under our constraints.",
    "Explain how a transactional outbox relates committed ledger changes to event publication.",
    "Show how an idempotent consumer handles a duplicate settlement event and a duplicate email effect.",
    "Explain write skew with a concrete payment invariant and a suitable isolation choice.",
    "Compare serializable isolation, explicit locking and uniqueness constraints for deduplication.",
    "Analyze a network partition and the difference between ledger consistency and API availability.",
    "Explain leader election, quorum intersection, fencing and stale-leader writes.",
    "Design dead-letter handling and safe operator replay without duplicate charges.",
    "Explain the guarantee boundary of exactly-once processing across an external bank API.",
    "Discuss partial refunds, reversal events and reconciliation after delayed settlement.",
    "Design an auditable append-only ledger, distinguishing append-only semantics from cryptographic tamper evidence.",
    "Design object-level authorization and authentication for payment and refund endpoints.",
    "Explain why HTTPS, SQL parameterization, CSRF protections and authorization solve different problems.",
    "Discuss short-lived credentials, key rotation, revocation and the limits of repairing earlier exposure.",
    "Design backpressure, bounded queues, admission control and circuit breakers during an outage.",
    "Explain a cache invalidation and versioning strategy that cannot double-charge a customer.",
    "Design monitoring with SLIs, SLOs and useful alerts for user-visible payment correctness.",
    "Plan a canary rollout and rollback when ledger and API schema versions differ.",
    "Explain late events, event time, processing time, watermarks and settlement reports.",
    "Discuss schema evolution, compatibility, replay and data lineage for financial event streams.",
    "Design disaster recovery with backups, write-ahead logs, replication and a zero-loss objective.",
    "Walk through an entire duplicate-charge incident investigation and how to prevent recurrence.",
]
ANGLES = [
    "Use a concrete worked example, explicit invariants, and two failure cases. Aim for250–350words.",
    "Stress-test your earlier design with a crash between each pair of steps; explain recovery. Aim for250–350words.",
    "Compare two implementation alternatives with an example and a material limitation. Aim for250–350words.",
    "Review what could go wrong under overload, delayed messages and a failover. Aim for250–350words.",
    "Explain how an operator would validate and safely change this mechanism in production. Aim for250–350words.",
    "Give a precise counterexample to an overstrong guarantee, then repair the design. Aim for250–350words.",
]
PROBES = [
    "Recall the exact agreed currency, zone count, idempotency retention, charge-attempt limit, ledger RPO and recovery-time objective. Then explain one consequence of combining these constraints.",
    "A client retries an identical request60hours after its payment committed. Under our agreed retention policy, what should happen, and what must remain atomic to prevent a second charge?",
    "A different payload arrives with the same idempotency key. What does our agreement require, and why is returning the earlier result insufficient?",
    "The broker redelivers a settlement event. What delivery semantics did we agree, and how should the consumer avoid a duplicate external effect?",
    "An operator proposes allowing8charge attempts because retrying is harmless. Does that fit our agreement? Explain the difference between idempotent effects and retry resource budgets.",
    "After a failover, both sides claim to be ledger leader. Explain fencing and quorum checks, and how the ledger RPO and recovery-time objective constrain the response.",
    "A teammate says an append-only database table automatically gives cryptographic integrity and that revoking a leaked encryption key recovers earlier confidentiality. Audit both claims.",
    "Two transactions independently approve withdrawals after reading the same invariant snapshot and together violate the balance floor. Diagnose the isolation problem and one valid prevention.",
    "The outbox row is committed but event publication times out. Walk through safe replay, settlement-consumer deduplication and user-visible status without promising impossible end-to-end exactly-once behavior.",
    "Produce a concise final design review: restate the six numerical/design constraints accurately, identify three important remaining failure modes, and say how we would test each.",
]


def token_counter():
    from mlx2.runtime.kimi_tokenizer import TikTokenTokenizer
    tokenizer = TikTokenTokenizer.from_pretrained(budget.MODEL, local_files_only=True)
    tokenizer.chat_template = (budget.MODEL / "chat_template.jinja").read_text()
    def count(messages):
        ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        if hasattr(ids, "keys"): ids = ids["input_ids"]
        return len(ids)
    return count


def server(spec):
    from mlx2 import server as module
    original = module.main
    def main():
        sys.argv[sys.argv.index("--max-context") + 1] = "65536"
        original()
    module.main = main
    budget.server(spec)


def unit(output):
    spec = json.loads((output / "spec.json").read_text())
    receipt = {**spec, "ownership": check_ownership(spec), "before": budget.environment(), "turns": [], "completed": False}
    running = log = adapter = None
    try:
        budget.install_probe()
        import mlx.core as mx

        from mlx2.adapters.kimi_linear import KimiLinearAdapter
        mx.set_default_device(mx.gpu); mx.set_wired_limit(mx.device_info()["max_recommended_working_set_size"])
        adapter = KimiLinearAdapter(spec["model"])
        receipt["ordinary_reference_smoke"] = budget.generate(adapter, "What is2+2? Reply with the integer only.", {"mode": "fixed", "k": 8}, limit=32)
        smoke = receipt["ordinary_reference_smoke"]
        if not smoke["completed"] or smoke["text"].replace("<|im_end|>", "").strip().strip(".") != "4":
            raise RuntimeError("ordinary smoke failed")
        adapter.close(); adapter = None; gc.collect(); mx.clear_cache()
        count = token_counter()
        log = (output / "server.log").open("w")
        running = spawn([sys.executable, "-u", str(Path(__file__).resolve()), "server", "--output", str(output)],
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True, cwd=ROOT)
        deadline = time.monotonic() + 300
        while True:
            if running.poll() is not None: raise RuntimeError("server exited")
            try:
                status = budget.http(spec["port"], "/v1/status", timeout=3)
                if status.get("state") == "ready": break
            except (OSError, TimeoutError): pass
            if time.monotonic() > deadline: raise TimeoutError("readiness")
            time.sleep(.5)
        receipt["initial_status"] = status
        def ask(messages, maximum=1024):
            response = budget.http(spec["port"], "/v1/chat/completions", {"model": budget.MODEL.name,
                "messages": messages, "temperature": 0, "enable_thinking": False,
                "stream": False, "max_tokens": maximum, "session_id": spec["job_id"]})
            if response["choices"][0]["finish_reason"] == "length":
                response = budget.http(spec["port"], "/v1/chat/completions", {"model": budget.MODEL.name,
                    "messages": messages, "temperature": 0, "enable_thinking": False,
                    "stream": False, "max_tokens": 1536, "session_id": spec["job_id"]})
            if response["usage"]["prompt_tokens"] != count(messages):
                raise RuntimeError("CPU/server prompt count mismatch")
            return response
        if spec.get("frozen"):
            frozen = json.loads(Path(spec["frozen"]).read_text())
            messages = frozen["messages"]
        else:
            messages = [{"role": "system", "content": SYSTEM}]
            for index in range(160):
                question = SEED if index == 0 else QUESTIONS[(index-1)%len(QUESTIONS)] + " " + ANGLES[((index-1)//len(QUESTIONS))%len(ANGLES)]
                messages.append({"role": "user", "content": question})
                response = ask(messages)
                text = response["choices"][0]["message"]["content"]
                messages.append({"role": "assistant", "content": text})
                row = {"phase": "build", "turn": index, "question": question, "response": response}
                atomic(output / f"build-{index:03d}.json", row)
                receipt["turns"].append({"phase": "build", "turn": index, "prompt_tokens": response["usage"]["prompt_tokens"], "output_tokens": response["usage"]["completion_tokens"]})
                atomic(output / "receipt.json", receipt)
                next_width = count(messages + [{"role": "user", "content": PROBES[0]}])
                print(json.dumps({"build_turn": index, "next_probe_tokens": next_width}), flush=True)
                if next_width >= 32768: break
            else: raise RuntimeError("real dialogue did not reach32K within bound")
            frozen = {"messages": messages, "first_probe_tokens": next_width,
                "source_revision": spec["source_revision"], "checkpoint": spec["checkpoint"],
                "scope": "Actual generated dialogue; earlier assistant assertions are not ground truth."}
        atomic(output / "frozen.json", frozen)
        receipt["frozen_sha256"] = hashlib.sha256(json.dumps(frozen["messages"], sort_keys=True).encode()).hexdigest()
        receipt["initial_probe_tokens"] = count(messages + [{"role": "user", "content": PROBES[0]}])
        for index, question in enumerate(PROBES):
            messages.append({"role": "user", "content": question})
            response = ask(messages, maximum=768)
            row = {"phase": "probe", "turn": index, "question": question, "response": response}
            atomic(output / f"probe-{index:02d}.json", row)
            messages.append({"role": "assistant", "content": response["choices"][0]["message"]["content"]})
            receipt["turns"].append({"phase": "probe", "turn": index, "prompt_tokens": response["usage"]["prompt_tokens"], "output_tokens": response["usage"]["completion_tokens"]})
            atomic(output / "receipt.json", receipt)
            print(json.dumps({"probe": index, "prompt_tokens": response["usage"]["prompt_tokens"]}), flush=True)
        receipt["final_status"] = budget.http(spec["port"], "/v1/status")
        receipt["completed"] = True
    finally:
        if adapter is not None: adapter.close()
        if running is not None:
            running.stop(timeout=25); receipt["server_stopped"] = not running.live()
        if log is not None: log.close()
        receipt["after"] = budget.environment(); atomic(output / "receipt.json", receipt)


def run(args):
    raise RuntimeError("Retired unsafe multi-unit lifecycle. Use the one-unit parents in scripts/research/kimi_moe; see README.md there.")


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("mode",choices=("run","unit","server"));p.add_argument("--output",type=Path,required=True);p.add_argument("--arm",default="default8");p.add_argument("--frozen",type=Path);p.add_argument("--suspended-by",type=int);args=p.parse_args()
    def interrupted(number,frame):raise KeyboardInterrupt(str(number))
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    if args.mode=="run":return run(args)
    spec=json.loads((args.output/"spec.json").read_text())
    return server(spec) if args.mode=="server" else unit(args.output)


if __name__=="__main__":main()
