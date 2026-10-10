# RC2 qualification status — 2026-10-10

This is a checkpoint for the RC2 qualification campaign. It reports
functional qualification evidence, not performance, route selection, or routine
production use. Qualification applies only to the exact tested runtime,
artifact, settings, and host identity; this public summary is not a portable
qualification receipt and does not certify another checkout or host.

## Decision routes

Four typed decision routes—Clef, Decision2, JEV, and the 27B decision route—each
passed all 13 required functional gates on the internal validation snapshot.
The checks covered two connections, a 4 MiB request-body limit, the normal
context-truncation default, and text, `noul`, choice, and score request paths. This establishes functional
qualification for those exact tested routes and settings. It does not select
them as defaults or establish that production traffic has used them.

## Vision and media routes

| Route/profile | Evidence and current boundary |
| --- | --- |
| SmolVLM, LFM2.5-VL and Qwen2.5-VL ordinary decode | Three complete passes at the tested 4K context, one lane, 256-token prefill, and 1 GiB cache profile. The evidence is limited to that profile. |
| Gemma 4 media | Media qualification passed. The generic Hermes literal-prompt probe remains unqualified, so this does not establish that probe's behavior. |
| Gemma 3n media | Not qualified: the cold continuous-batch reference mismatched, and video-logit parity failed. |
| MiniCPM-O media | Not qualified: image feature/token parity failed. Audio parity was exact, but the aggregate gate remains coupled and therefore does not pass. |

These results describe the tested cases only. A pass for one modality or probe
does not override a failed or still-unqualified gate for the route as a whole.

## DLoop and performance work

The public DLoop harness now rejects rows with no decode tokens instead of
dividing by zero. This is a CPU-tested harness correctness fix; it is not model
qualification or a performance result. A matched-Q4 width-one run is in
progress. Width two remains pending the width-one audit and a fresh admission
check.

Controlled performance runs and thermal ladders have not started. The release remains
`0.1.0rc1`; `0.1.0rc2` is pending completion and review of the remaining
qualification and release gates.

## State terminology

- **Implemented** means the code path exists.
- **Qualified** means the required evidence passed for the exact bound test
  identity and settings described above.
- **Selected** means a route has been explicitly chosen for a deployment; the
  results here do not make that choice.
- **Observed-used** means execution evidence shows a mechanism actually ran.
  Qualification traces can demonstrate the exercised mechanisms; they do not
  establish routine production use.
