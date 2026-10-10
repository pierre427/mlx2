# Qualification producer and auditors

This directory contains reusable scripts for collecting a model's functional
qualification ladder, evaluating its correctness verdict, and assessing
performance separately. The scripts do not qualify a model by themselves; a
result applies only to the exact source, runtime, artifact, settings, and host
recorded in its evidence.

`thermal_ladder.py` drives a running server and writes raw request and serving
evidence. `qualification_verdict.py` checks functional requirements without
using throughput as a correctness signal. `performance_assessment.py` compares
two ladders only when their identities, cells, and measurement conditions
match. `run_profile.py` supervises a local server process. `dloop_ab.py`
collects the DLoop versus fixed-depth comparison, and `dloop_qualification.py`
audits those reports and their state-continuation checks.

`thermal_ladder.py` reads its small, portable admission policy from the
adjacent `thermal-policy.json` and binds that file's digest into its reports.
Private model matrices, prompts, launch commands, host paths, and run artifacts
are not inputs to this public producer.

Run the CPU-only contract tests with:

```bash
python -m pytest -q qualification/runs/qualify-1010-correctness/test_*.py \
  tests/test_qualification_runfiles_cpu.py
```

These tests exercise schema validation and mocked launch behavior. They do not
load a model, access a GPU, establish qualification, or support performance
claims. For operational gates and evidence requirements, see
[`docs/QUALIFICATION.md`](../../../docs/QUALIFICATION.md).
