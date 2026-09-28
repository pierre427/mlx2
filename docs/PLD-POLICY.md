# Prompt lookup policy

Status: **cliff-aware span is selected for the prompt-lookup policy; other
experimental candidates remain disabled**.

Prompt lookup remains an explicit route for adapters that declare the
capability and have a qualified verifier and exact rollback path. Ordinary
decode remains the reference and fallback. Selecting this policy does not
select prompt lookup globally or claim that proposals were observed on a
particular request.

The selected policy enables `cliff_aware_span`. It explicitly disables
deferred admission, cost-aware admission, recent-source indexing, rotating
replay, and batched verify. External retrieval segments remain request-scoped
and are not configured by the default policy.

## Qualification summary

The same Qwen3.5-9B-4bit artifact was tested on an M3 Pro Mac15,7 and a
Mac17,6 host. The M3 ran a counterbalanced 21-cell candidate matrix and the
complete 35-check serving qualifier. The second host ran the same matrix plus
four cooldown-controlled ordinary/current-PLD/cliff blocks.

On the M3, cliff-aware span measured 1.106x current-PLD throughput for the
copy-positive case, 1.101x for periodic text, 1.000x for novel text, 1.008x
for an approximately 5K-token copy case, and 25.7 versus 24.3 tokens/s for
atomic B=2. The full qualifier passed 35/35 checks with 53 observed lookup
cycles, 693 proposed tokens, 96 accepted tokens, 52 rollbacks, and no
checkpoint failures, restores, or full rebuilds.

Across the first three thermal-zero and swap-stable confirmation blocks on
Mac17,6, paired medians were 1.098x, 1.136x, 1.004x, 1.026x, and 1.152x for
the same copy, periodic, novel, long-copy, and B=2 workloads. Cliff-aware span
was the only disabled candidate to clear observed engagement,
parity-with-current-PLD, recovery safety, and repeated performance gates on
both hosts.

Deferred admission and cost-aware admission lost materially. Recent-source
indexing bounded the host index but did not show served speed value. External
retrieval helped only when explicitly supplied for the matching request.
Batched verify and rotating replay did not engage on this hybrid
recurrent-cache model and therefore remain disabled rather than being treated
as qualified from implementation or CPU-test coverage alone.

One boundary remains: an alphabet/retrieval prompt diverged from ordinary
decode on the Qwen3.5 PLD target path even when PLD proposed zero tokens. Every
PLD policy arm followed the same baseline path. Cliff-aware span did not
introduce the divergence, but it does not resolve it; prompt lookup is not the
global ordinary-decode replacement.
