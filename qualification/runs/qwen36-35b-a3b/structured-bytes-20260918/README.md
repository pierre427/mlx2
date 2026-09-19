# Structured output: byte-fallback pieces on GPU (2026-09-18)

Source `4eb54e9`, Qwen3.6-35B-A3B (MTP-preserved oQ4e artifact), ordinary route,
`--qualification-mode`, 32K context, 4 lanes. Mechanism evidence, not a route
qualification.

| Probe | Result |
|---|---|
| Raw grammar requiring U+10FFFF (reachable only through 4 byte pieces) | exact `=\U0010ffff=`, 7 tokens, engine `automaton` |
| Emoji class `[U+1F600-U+1F64F]{3}` | three in-class characters |
| Greedy `json_object` with an astral character in a free string | valid JSON containing U+1F70A |
| Sampled `json_object` (temp 0.8, top_p 0.95, 4 seeds) | 4/4 valid, `tail_mass_bound` 0.0 |
| Thinking enabled + `json_object` | 338 deferred tokens, then valid JSON with U+1F732 |
| Server | healthy, 0 structured-output failures |

Before this change each of the first, third, fourth and fifth probes could only
end in a 502 (`no valid token continuation`) once the model reached the
character.

**Separate observation (engine-independent):** the strict-schema control
(`population: integer`, temp 0) ran `2161000000…` to the token cap. Replayed
with `MLX2_STRUCTURED_AUTOMATON=0` the scanner engine produces the same text,
and both engines' masks at that prefix are identical (434 tokens, `}` and
whitespace admitted). JSON Schema `integer` is unbounded, and the greedy model
loops on `0`. A digit cap in the compiled pattern would bound it; not changed
here.
