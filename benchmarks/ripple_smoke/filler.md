# Qualifying a model

> **Policy (2026-10-29, AGENTS.md "Working rules"):** a full preflight is never
> required (an overnight job runs it daily), and thermal gating is only for
> publishable statistics. Features and models are evaluated by trying and
> testing them (engagement, correctness, near-tie classification, an A/B); if
> they pass, ask the user whether they should become defaults. The full
> procedure below remains available when the user wants publishable numbers.

This is the procedure for qualifying a model route for serving. The current
correctness verdict and context-ladder producers are in
qualification/runs/qualify-1010-correctness/. Older qualify-* run directories
preserve their campaign-specific scripts and evidence; use them as historical
records, not as the current policy or a mutable source template.

## Repair-first campaign order

While planned defect repairs are open or changing, use targeted functional
probes, route receipts and focused tests. Finish and review all intended
repairs, then run focused regressions. Ask the user explicitly before a full
preflight; that approved run is the final publication gate. Moving to another
model does not finish a repair campaign. In an isolated worktree sharing the
primary virtualenv, set `PYTHONPATH="$PWD/src"` so validation exercises this
worktree. Documentation builds do not require GPU qualification.

The October AST tile route, EAGLE-3, DSpark and Muse assistant integrations
are implemented and unqualified unless an exact matching receipt establishes
otherwise. Model-matrix `P`/`P*` cells are exploratory smoke verdicts, not
qualification. An `ast_*` cell can serve a chain after dtype gating; bind the
concrete route, artifact fingerprint, source revision, drafter and policy.
Offline physical-target capability qualification and direct tensor probes do
not qualify the served AST route. See [AST.md](AST.md) and
[RESULTS.md](RESULTS.md).

## What qualification means

Qualifying a route answers three questions, in order:

1. **Does it load and decode properly?** Preflight and the ordinary route's
   smoke.
2. **Do the default options work?** Smoke on every route the model ships,
   plus the MTP handoff gates for MTP routes, where handoff is on by default.
3. **Does the route preserve expected behavior across context and cache
   conditions?** The context ladder completes all three repetitions in every
   cell, every needle is found, warm requests hit APCv2, and cold/warm answers
   agree. Throughput is assessed separately and cannot qualify or disqualify a
   route.

A route is **qualified** when all three hold on a pinned snapshot, on the
host that will serve it. A passing unit test, a smoke run on its own or a
fast benchmark does not qualify anything.

Qualification is a statement of confidence, **not permission to run**. An
unqualified model can still be loaded and served; we just cannot say for
sure that it runs properly, and its status must say so (see
[AGENTS.md](../AGENTS.md)). The server has three modes:
