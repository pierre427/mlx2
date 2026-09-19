# OpenRouter Muse Glimmer port review

Date: 2026-09-15. Independent reviewer: the main Muse port agent. All candidate review was CPU/static only. No candidate was merged, loaded, GPU-tested, qualified, or deployed.

## Outcome

The cheap models produced useful architecture inventories and portions of an ordinary-decode port, but none completed the modern serving contract. After one bounded repair round, GLM was the strongest Muse candidate. DeepSeek's repair introduced a model-shape validation error that rejects the supplied target. These are patch-quality assessments, not general model rankings.

Grades: **C** means reusable partial work with integration blockers; **D** means a startup blocker or a largely incomplete serving integration. Neither means ready to adopt.

| Candidate | First pass | Repaired | Cumulative API cost for this Muse run | Decision |
|---|---|---|---:|---|
| GLM-5.3-Flash | D: wrong artifact binder prevents startup | C: substantial repair, tools/qualification and parser gaps remain | $0.168046973 | Mine selected metadata/parser test ideas; do not merge |
| DeepSeek-v4.1-Flash | C: useful ordinary port, incomplete channels/tools | D: new validator rejects actual Muse config | $0.181158690 | Keep architecture inventory; do not merge |
| MiMo-v2.5 | D: adapter/quantization/descriptor incomplete | D: attributes fixed, output parser import now fails | $0.201862774 | Reuse inventory only; do not merge |

Costs are recorded provider usage from each run's final summary, including that run's repair responses. The orchestrator's aggregate ledger additionally covers Qwen tasks and failed infrastructure attempts. A turn-limit result is a frozen unfinished candidate, not a successful completion.

## Conditions and fairness

Each candidate received the same isolated baseline, selected unified source, artifact metadata, a bounded CPU-port task, and an AST syntax tool. The broker did not offer real execution of Python tests or actual model files. Models honestly stating “AST checked, not executed” receive credit for that distinction. Generated tests called CPU-only were inspected before execution; tests that could import/evaluate MLX on its default device were excluded. Reviewer checks installed an import hook rejecting every `mlx` import. No generated tensor code was executed.

The initial broker omitted the actual Muse chat template, tokenizer configuration and ATEM parser source. Thus native protocol implementation was underspecified on the first pass. Repair feedback supplied explicit recipient-channel examples and described ATEM function/parameter semantics, but did not provide the complete ATEM grammar fixture. Missing template access limits what can fairly be attributed to model capability. Candidates could still flag the missing input and reject unsupported tools; inventing a grammar or claiming an unused parser was integrated remains an implementation defect. A better pilot needs complete protocol fixtures and a guarded CPU execution tool.

The metadata broker listed `tensor_key_count: 0` for single-file draft artifacts. That proves the index-based inventory was empty, not that weights were absent. The main implementation had access to actual local files and tokenizer, so its result and duration are not a controlled speed comparison. The main project's adapter API also evolved during the experiment; missing post-snapshot methods are integration debt, not counted as first-pass mistakes.

Frozen first-pass source trees remain in `/tmp/mlx2-openrouter/{glm,deepseek41,mimo}-muse-first-pass`. Repaired trees, feedback, logs and reviewer test artifacts remain alongside them. Those temporary paths are evidence locations, not production dependencies.

## GLM: concrete defects and repairs

Seven first-pass defect groups were recorded:

1. FlashNext artifact identity unconditionally accesses `ple_rows.bin`, which Muse lacks. Reproduced `FileNotFoundError`. **Fixed** by a Muse binder supporting indexed and single-file weights.
2. Qwen output parser emits native Muse recipient headers and private reasoning as answer text. **Partly fixed** by a native channel parser; ordinary reasoning/answer transitions and character splits now pass.
3. Tools and reasoning capability absent despite the supplied qualification gate requiring both. **Still blocked**: tools remain rejected; native parsing alone does not complete tool/template semantics or the qualification contract.
4. CPU-labeled tensor tests never selected the CPU device. **Fixed in written tests** with `mx.set_default_device(mx.cpu)` before allocations. Reviewer still excluded tensor execution.
5. Test incorrectly bounded rotating cache storage immediately after long initial prefill. **Fixed** by removing that false invariant and explicitly deferring an oracle test.
6. Source MIT license mislabeled Apache. **Partly fixed** in code/provenance with the full notice; unchanged port and provenance documentation still makes contradictory Apache claims.
7. “Drafter weights absent” inferred from zero indexed keys. **Partly fixed** in adapter comments; unchanged documentation still asserts absence.

Repair verification: **31 passed, 2 failed** in the extracted pure-Python test subset. The two failures are contradictory single-file identity expectations (config is included in the actual file list) and a call to nonexistent `MuseGlimmerAdapter.configure_environment` (the helper is module-level). Tests requiring MLX were excluded; an initial selection included two config imports that the import guard blocked before execution, then they were removed from the pure subset.

Two additional independent runtime defects were reproduced after repair:

- Plain-completion `STOP` split across character chunks is emitted and ignored because the non-chat branch never holds partial stop prefixes.
- `to=user<|message|>a<|python_tag|>b<|eot|>` emits the unknown special token when delivered in one chunk, but raises when delivered character by character. Structural validation depends on chunk boundaries.

Remaining work: native tool rendering/parsing, complete reasoning-effort semantics, qualification-compatible descriptor/registry integration, chunk-invariant stops/parser validation, two failing tests, and consistent provenance/gap documentation. No performance or cache-state qualification was established.

## DeepSeek: concrete defects and repairs

Seven first-pass defect groups were recorded:

1. Generic output parser leaks native channel syntax/reasoning. **Partly fixed** with a dedicated Muse channel parser.
2. Tools ignored and qualification-required capabilities omitted. **Partly fixed** by rejecting tools and declaring reasoning; the tool and qualification gap remains.
3. CPU-labeled tensor tests do not enforce CPU device. **Unchanged**; no reviewer tensor execution.
4. Artifact validation is weak and single-file targets unsupported. **Partly fixed** with an import-safe validator, but the repair introduces the critical defect below.
5. Quantization lookup misses conventional `quantization`-only artifacts and module overrides. **Partly fixed** by a resolver and override predicate; complete mixed namespace/override behavior is not established.
6. MIT notice lacks the complete permission text. **Unchanged**.
7. DFlash2 weight-absence inference is too strong and paired rollback requirements incomplete. **Unchanged documentation**.

The new config validator requires `hidden_size == num_attention_heads * head_dim`. The actual supplied Muse config has hidden size **6656**, **32** query heads, head dimension **128** (projected attention width **4096**). Muse deliberately projects between these dimensions. Calling the candidate config parser on the supplied Q4 metadata raises:

```
ArtifactError: hidden_size 6656 != num_attention_heads*head_dim 4096
```

This prevents target startup before weights are loaded. It is an architecture error, not a missing dependency or GPU qualification issue.

Three safe descriptor/metadata tests pass, but none exercises this new actual-config validator. Independent parser probes also reproduce split `STOP` leakage through the answer and plain text completions failing with “truncated header before <|message|>”. The adapter unconditionally constructs a chat parser even for completions.

Remaining work: remove the invalid shape constraint with an actual-config regression, native tools and reasoning-effort mapping, correct completion and split-stop behavior, strengthened target-role validation, CPU-safe tests, complete license notice, and updated gap documentation. No performance or state-cache qualification was established.

## MiMo: concrete defects and repairs

Eight first-pass defect groups were recorded:

1. Adapter lacks descriptor, layout and profile name. **Partly fixed**: attributes now exist, but `profile_name(True)` silently returns the ordinary profile instead of rejecting unsupported native MTP.
2. Descriptor lacks APCv2/layered/batch/stream declarations. **Fixed** as declarations; no qualification is fabricated. Tools/reasoning remain absent and the supplied qualification route remains incompatible.
3. Packed Q4 loading has no quantization step. **Partly fixed** with `nn.quantize`; only `quantization_config` is read and mixed per-module overrides remain unsupported.
4. Generic Qwen channels/tools used for Muse. **Not fixed**. The final adapter imports `MuseGlimmerChannelParser`, but the repaired module defines `MuseGlimmerOutputParser`. Calling `output_parser` on an uninitialized adapter object reproduces `ImportError` with no model load.
5. CPU tests allocate MLX tensors and some “tiny” configurations retain large default dimensions. **Fixed** by replacing tests with pure static/metadata checks.
6. Tests contradict qualification and artifact identity behavior. **Partly fixed**: identity assertion removed; two route tests still expect success without any qualified profile.
7. Artifact metadata fields falsely described as unavailable. **Fixed** in revised metadata checks and documentation; zero indexed draft keys now marked unverified.
8. Optimization inventory confuses Qwen mechanisms with Muse. **Partly fixed**: a new section correctly explains inapplicability and paired DFlash2 state, but earlier sections still call PLE “page-local evidence” and list QSA/GDN/PLE as missing Muse enhancements.

The complete repaired pure-Python suite ran under the MLX import guard: **43 passed, 3 failed**. Failures are the two unqualified route expectations and a missing Muse entry in the retained notice. Many passing tests merely check that names occur in source, so this count does not establish integration correctness.

Independent parser probes found further defects behind the immediate import failure:

- A whole native example emits `assistant answer` as content and stores reasoning in a property instead of emitting the required `reasoning_content` events.
- Character-by-character input emits the entire native syntax and reasoning as answer text; partial markers are not retained.
- A string stop is converted into a tuple of individual characters.
- The alleged ATEM implementation uses invented `<|function|>` / `<|parameter|>` sentinels. Actual `<atem:function_calls>` / `<atem:invoke>` / `<atem:parameter>` input returns no tool call. It also has no schema or declared-tool validation. Complete ATEM source was absent from the broker input, but claiming support for an invented unverified protocol is still incorrect.

Remaining work: repair the adapter/parser import and actually bind the native parser, implement chunk-safe channel/stop behavior and event delivery, supply and validate real tool grammar/template fixtures, normalize tool history, map native reasoning strength, strengthen metadata-before-import checks and quantization variants, fix three failed tests and reconcile contradictory documentation. This is substantial functional work, not just GPU qualification.

## What was useful

Mechanical source mining, import relocation, architecture inventories, descriptor scaffolding, artifact inventories and ordinary-route limitation lists were useful outputs. GLM's repair also produced a reasonably substantial native-channel test set and metadata fail-closed checks.

The work still requiring architectural judgment included projected attention dimensions, native Muse tool/recipient semantics, the difference between declared and qualified capabilities, full target/draft/RNG state transactions, and exact ragged rollback of rotating layers. The cheap candidates did not deliver a DFlash2 executor or new cache-state mechanism. Their lack of a qualification receipt was correct; “implemented enough to qualify” was not achieved.

For reference, the independently implemented main port includes native tool/channel parsing and import-safe metadata tests; it remains explicitly unqualified and has had no Muse GPU test. DFlash2 weights exist locally: missing target-tap/executor/paired-state/rotating-rollback integration must be completed before speculation can be selected. Acquisition is not the blocker.

## Economics including review

Bounded reviewer wall-time slices, measured from tool/message timestamps rather than an active-work stopwatch:

- GLM + DeepSeek first pass: 21:34:54–21:40:59 UTC, **6m05s**.
- MiMo first pass: 21:47:56–21:49:09, **1m13s**.
- GLM + DeepSeek repair review: 21:51:56–21:54:04, **2m08s**.
- MiMo preliminary repair read: 21:56:58–21:57:17, **19s**.
- MiMo frozen repair review: 22:04:14–22:05:09, **55s**.
- Total bounded review slices: **10m40s**, excluding provider waits and report drafting/closeout.

These slices exclude waiting on providers/repair jobs; they include review and focused probes, not every orchestration or reporting action. They are agent wall time, not billed human hours or known frontier token cost.

The main Muse port, registry and peer review occupied approximately 21:13:25–21:34:05 (**20m40s**), with broader scope and richer execution access. It is not a matched control. Candidate patch cost is small, but none reached the main CPU-port acceptance level and substantial coding remains. Therefore this run demonstrates cheap scaffolding, not demonstrated end-to-end dollar savings.

A defensible break-even comparison is:

`external API cost + frontier review/fix cost < frontier direct implementation cost`.

The two frontier dollar terms are unavailable. They must remain unknown rather than be inferred from subscription price or invented token rates. Provider elapsed time, reviewer oversight time, repair time, API charges, and remaining integration work are separate quantities. Infrastructure failures and exhausted repair budgets affect the observed outcome and should be reported separately from code quality.

## Frozen adapter identities

SHA256 of reviewed repaired adapter files:

- GLM `adapters/muse.py`: `0e66d9a21088617b6e13dda6050c37a6e4d686021e0f39b7d8b75a29d2d5482b`
- DeepSeek `adapters/muse_glimmer.py`: `2ed920db8ad725f5300eb23cb7fc5ac2e931679228f596768820aa06a6203d27`
- MiMo `adapters/muse_glimmer.py`: `f3a8f9a3abcef5281dea5ab602d1efb5440fdf3859f5f065d02057f299cff1fc`

Reviewer evidence: `/tmp/mlx2-openrouter/reviews/{glm,deepseek41,mimo}-muse-repaired/`; feedback files: `/tmp/mlx2-openrouter/feedback-{glm,deepseek41,mimo}-muse.txt`. First-pass source hashes and unchanged/repaired file comparisons were recorded in the agent execution history. No candidate code was adopted in the main project.
