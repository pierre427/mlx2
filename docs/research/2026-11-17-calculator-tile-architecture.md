# The calculator tile: design and implementation

This describes the implementation on the calculator evaluation branch, including the explicitly versioned LaTeX repair. It is an experimental, unqualified feature. The explanation describes how it works; the running evaluations will determine how reliably it helps and what it costs.

**Update: selectable CPU, Metal, and GEMM arithmetic.** The implementation now has `--calculator-backend cpu|gpu|gemm` in the evaluation runner. The preferred default for new version-3/4 tile configurations is `gpu`, as requested. Older version-1/2 manifests retain their CPU behavior. The original CPU implementation remains the reference. The GPU implementation is being tested; implemented code is not a qualification claim.

In GPU mode, the CPU parses text into a deferred operation plan instead of evaluating the expression with `Fraction`. Exact multi-digit integer arithmetic and rational intermediates are evaluated by custom Metal kernels. Numerical intermediate buffers remain on the GPU between operations. The CPU currently walks the syntax plan to enqueue those operations and formats the final returned digits. In `gemm` mode, eligible literal multiplications use a matrix-based digit convolution followed by GPU carry propagation; other arithmetic still uses Metal. Neither GPU strategy silently falls back to CPU arithmetic.

The GEMM representation splits an operand into decimal digits. Each matrix product term is at most 9 × 9, and a dot product contains at most 60 such terms. Its maximum sum is 4,860, within the exact integer range of float32. Carry propagation then converts those digit sums into the final number using integer operations. This is different from converting a 12-digit number directly into a float32 value and hoping multiplication will be accurate.

A GPU cue prefilter is also being tested. It keeps a fixed 48-token device window, recognizes arithmetic answer cues, and combines those cues with the learned gate. The LaTeX separator has a separate symbolic check. A negative prefilter result avoids materializing and decoding the history on the CPU. Invalid pieces and possible Unicode whitespace at the cue boundary conservatively permit the full parser. The first call and active answer plans retain the full checks.

The current hybrid controller still reads a small trigger flag on the CPU. That is a synchronization point even on unified memory. It therefore reduces Python text work, but does not yet provide a completely GPU-resident parser, answer-plan controller, or generation loop. The handoff below distinguishes the numerical backend from the host responsibilities that remain.

**The central idea.** A language model is good at producing plausible text. It can interpret a question, describe a method, and write an arithmetic expression. However, its next-token predictions can still produce the wrong digits. The calculator tile gives the generation process access to ordinary exact arithmetic at a recognized answer position. The model continues to handle language and reasoning; the calculator handles the arithmetic expression it can actually read.

For example, the model might write `109 × 71 =`. The tile can read that expression, calculate `7739`, and make the decoder emit the tokens spelling that answer. The model then continues writing with those answer tokens included in its conversation history.

This does not replace the whole model or permanently change its weights. It is an extra component attached to the inference process. No network calculator service or second language model is needed.

**The parts of the system.** There are five main parts:

1. A small learned detector reads an internal model activation and estimates whether an arithmetic answer belongs at this point.
2. A parser examines a short piece of recent text and tries to recognize a complete supported expression.
3. An exact arithmetic evaluator calculates the expression's value.
4. A token controller guides the decoder through the answer's digits.
5. Metadata, checks, counters, and event records identify the component and show whether it actually ran.

The learned detector recognizes a situation. The parser identifies the calculation. The evaluator obtains the result. The token controller puts that result into the generated text.

**Which work runs on the GPU and which runs on the CPU.** In the selected `gpu` arithmetic mode, the language model, learned detector, cue-window filter, exact numerical arithmetic, and output-logit mask execute on the GPU. Text parsing, syntax-plan construction, operation dispatch, final digit-to-text formatting, tokenization, and answer-plan bookkeeping still execute on the CPU. Unified memory removes a separate PCIe copy, but does not remove synchronization or Python overhead.

The handoff for one calculation is:

1. **GPU:** run the model and detector; update the fixed token-history window and cue flag.
2. **GPU → CPU:** read the small combined trigger flag. If it is false, skip history conversion and text parsing.
3. **CPU:** for a possible trigger, decode history and parse a complete supported expression into an operation tree. This constructs the recipe without calculating its numerical answer.
4. **GPU:** execute that recipe using exact digit/rational Metal kernels. Arithmetic and rational reduction are separate kernel stages, with numerical intermediates in GPU arrays. Eligible `gemm` multiplications use digit convolution first.
5. **GPU → CPU:** return the final digit packet or refusal status. The CPU formats and tokenizes the answer and maintains the emission plan.
6. **GPU:** mask the next-token logits according to that plan, then continue model generation.

Selecting `cpu` replaces step 4 with Python exact integer/rational arithmetic. It is a useful independent reference and may be faster for these tiny calculations; speed is an experimental question. The original M3 pilot also ran model inference on the CPU. Current M5 GPU arithmetic tests use GPU inference and GPU numerical evaluation.

The filter includes every vocabulary piece containing `=` or `\hline`, including merged tokens such as `=$` or `=\)`. It is a conservative admission filter: the strict parser still decides whether the complete context is a valid calculation. A lookup hit alone never authorizes an answer.

**Where the learned detector attaches.** A transformer consists of a sequence of layers. Each layer includes attention, which combines information from different token positions, and an MLP, which transforms the representation at a position. An activation is the vector of numbers carrying the model's working representation at that point.

The tile attaches a wrapper around an MLP. The wrapper lets the ordinary MLP run and observes its input. For this calculator's exact output modes, the observation does not alter the MLP's output. This is a small observation point inside the model, rather than a replacement neural layer.

The existing Qwen3-0.6B tile reads layer index 4, the fifth layer when counting from one, in a model with 28 layers. Specifically, it observes the normalized input to that layer's MLP. Calling this the model's thoughts would be misleading: it is a numerical representation from which a detector learned to recognize useful patterns.

**How the learned detector makes its decision.** Its main learned parameter is one vector of weights, approximately one weight for each activation dimension. It multiplies those weights by the observed activation values and adds a bias:

`score = w₁x₁ + w₂x₂ + … + wₙxₙ + bias`

The implementation turns that score into a probability-like value with a logistic function and compares it with a threshold. A high enough value opens the neural gate. The gate is evaluated using a specified float32, row-wise arithmetic rule, so its calculation does not change merely because a request shares a batch with other requests.

The supplied Qwen3-0.6B artifact uses a threshold of 0.5 and a bias of approximately −3.7343. The learned weight vector is stored in a small safetensors file. This artifact is only a few kilobytes; it does not contain another arithmetic language model.

The detector was fitted using examples of answer positions and negative examples, including number-heavy material that should not trigger a calculator. The language model's original weights remain frozen during this process. The existing builder chooses detector settings using validation data. Our new evaluation does not retrain it on the evaluation problems.

A gate is not a guarantee that a calculation is appropriate. It can miss a valid opportunity or open at an unsuitable position. That is why the next part checks the actual text independently.

**What the text parser checks.** During generation, the calculator controller decodes a short recent token window. Legacy versions inspect 48 tokens. Version 4 uses a configurable 256-token parser window and retains a separate 48-token GPU cue window. It parses a complete LaTeX candidate under a 4096-character raw safety bound, then applies the 160-character expression limit to canonical syntax. Formatting macros therefore do not consume the semantic expression budget. A token is a piece of text, so 48 tokens is not necessarily 48 characters or words.

The parser looks for supported arithmetic and an answer cue. An expression ending in `=` is a clear example. The code also supports certain word operators and question forms. A direct question form is restricted to its initial answer position; it is not allowed to repeatedly reopen the calculator throughout a later answer.

The parser is deliberately conservative. A number may instead be a date, version, telephone number, identifier, time, range, or amount with incompatible units. The implementation contains boundary checks and refusal rules for these cases. It also tries to avoid evaluating only the tail of a longer unsupported calculation. If a text window cuts through an operand, that is a reason to refuse rather than silently calculate from the surviving digits.

An unsuccessful parse normally leaves generation with the original model. It does not mean the model is stopped or the whole user request fails. Refusing an unsupported serving configuration, such as combining this route with unaudited speculation, is a separate load-time or route-admission decision.

**Expressions and exact arithmetic.** The first parser version primarily handles a single arithmetic operation. Version 2 adds whole expressions: parentheses, precedence, unary minus, and bounded integer powers. Thus `2 + 3 * 4` is 14, whereas `(2 + 3) * 4` is 20.

The explicit CPU reference evaluator uses Python's `Fraction` representation. It keeps a value as an exact ratio of integers. For example, decimal `0.1` becomes exactly one tenth. Multiplying it by `0.2` gives exactly one fiftieth, which can be rendered as `0.02`. The arithmetic does not depend on the language model's floating-point estimates of the answer.

That also explains why larger operands are an important test. The arithmetic algorithm can multiply 12-digit integers without learning examples of those particular numbers. However, the surrounding system still has to read the complete expression, recognize the answer position, and emit the full result. Exact arithmetic alone does not guarantee end-to-end accuracy.

This calculator is bounded. Version 2's expression machinery limits numerator and denominator sizes to 60 decimal digits and exponent magnitudes to 64, with additional checks before exponentiation. Division by zero is refused. A quotient with a non-terminating decimal expansion is also refused by the current output renderer. It does not invent a rounded answer to one third. It is not a general symbolic algebra engine or a complete solution to advanced mathematics.

**What the LaTeX repair changes.** Models often write mathematics using LaTeX commands, such as `\times`, and sometimes lay multiplication out vertically:

```latex
\begin{array}{r}
109 \\
\times 71 \\
\hline
```

The earlier parser did not recognize that result position. The new explicit `calc-parse-v3` version recognizes supported LaTeX operator commands and a narrow two-row multiplication-array grammar. The grammar requires both operands, the multiplication operator, the row separators, and the horizontal result separator. It refuses extra columns, missing operands, additional operand rows, and a result that is already written.

The reproduction exposed a second problem: the existing learned detector stayed closed at this unfamiliar separator. Version 3 therefore permits this complete, unambiguous column grammar to identify its own arithmetic answer slot. This is a specific symbolic gate, not a reduction in the neural threshold for all requests. It still requires valid request-row/forward binding and respects explicit gate overrides used by experimental controls. Its activations receive a separate `latex_column_open` counter.

Older parser versions retain their behavior. The repaired artifact explicitly selects version 3; the implementation's default parser has not been changed. In the targeted reproduction, the repaired route changes `109 × 71` from the model's incorrect `7661` to `7739`. That is one functional reproduction, rather than a general quality claim.

**Version 4 LaTeX support.** The bounded frontend adds fractions and nested fractions, matched sized grouping, integer powers, literal factorials/binomials, and finite sums/products. Exact roots and integer modulo are available in CPU reference mode; the GPU route explicitly refuses those operations rather than moving their numerical evaluation to the CPU. Transcendentals, arbitrary symbolic algebra, integrals, matrices, and proofs remain unsupported. See the [capability matrix](2026-11-17-calculator-latex.md) for precise domains and limits.

Formatting is removed only after a complete expression has been parsed. Unknown macros fail parsing. The raw window bound and evidence that an operand may have been cut are retained, so normalization cannot turn a truncated tail into a trusted full formula.

**How the answer reaches the output.** Language models normally produce a score for each possible next token. These scores are called logits. A decoder chooses a token from them, and the model then predicts the following token.

The calculator's exact modes operate at this choice point. They do not simply paste an answer into the final response after generation has finished.

In `force` mode, the controller tokenizes the canonical answer string. If the answer is `7739`, its tokenizer may represent that as several tokens. For each step of that plan, the controller masks the logits so only the next planned token remains eligible. Version 3 emits a newline after a completed cued equation. Version 4 preserves units and ordinary continuations instead: after the complete result it blocks digit extensions and arithmetic operators that would change that right-hand value. Spaces/tabs keep this boundary check alive; a newline can begin a new formula. Forcing a newline unconditionally had changed correct word-problem reasoning. The column-array form preserves its own row formatting. After the planned output is emitted, ordinary generation resumes.

In `constrain` mode, the controller instead allows any eligible token that keeps the emitted text a prefix of the correct answer string. This permits different tokenizations of the same numerical answer. The model still chooses among those allowed tokens.

In either mode, the emitted answer becomes part of the ordinary token history. The model reads it on the following forward pass, and its normal KV cache develops from those actual tokens. The controller checks that the output still matches its plan. If a higher-priority output constraint has already banned the required tokens, the plan aborts rather than silently overriding that constraint.

There is also a boundary guard after the result. It prevents the model from immediately extending a correct number with extra digits—for example, turning `7739` into `77390` or adding a decimal extension through the next punctuation step. It is a local guard, not a promise that the model will never contradict the result later in its prose.

**Why tokenization and request identity matter.** The controller must use the same tokenizer as the model. A token ID has meaning only in that tokenizer's vocabulary. The implementation builds and caches information about tokens that can start or continue a number. Our MLX-LM integration uncovered a wrapper compatibility issue here, which was repaired by using the underlying tokenizer object.

Batched generation adds another issue: a detector result for one request must never be applied to another request. The generator binds the controller to a batch row and the corresponding forward pass. Ambiguous or stale binding causes the calculator to remain closed. This is why integration involves more than adding a regular expression to a text stream.

Each request gets a fresh answer-plan controller. The installed tile set can retain its immutable parameters and aggregate counters, while individual requests do not share partially emitted numerical answers.

**What is stored in a tile artifact.** A manifest is a small JSON description of the tile. It identifies the source checkpoint, number of layers, read and write layer indices, detector bias and threshold, parser version, output mode, and parameter file. The parameter file contains the learned probe weights.

File digests are checked when loading. The loader checks that the artifact matches the model fingerprint and layer count. These checks reduce the chance of accidentally applying a detector built for one model to another model with different activations.

The write-layer field is part of the shared tile contract. In the exact `force` and `constrain` calculator modes, the answer is enforced at the logits processor; that field should not be interpreted as a neural arithmetic circuit being inserted there. A separate experimental `push` mode can modify hidden activations, but that approximate research mode is not allowed in serving. It is not the mechanism being evaluated here.

**How this fits into Amoeba.** Amoeba uses a common tile contract for optional capabilities. A learned neural box executes selected parts of a model. A deterministic calculator supplies a small exact operation. They can have a shared lifecycle and identity system without performing the same kind of computation.

The runtime code is model-agnostic. An adapter supplies the model's layer/MLP layout and activation function, while the artifact supplies the checkpoint-specific detector. The existing 27B serving adapter has deterministic-tile integration. The current 0.6B evaluation uses the same tile runtime attached to stock MLX-LM's MLPs and logits processing. That direct evaluation is distinct from claiming a generally qualified 0.6B server route.

The supported route is ordinary autoregressive decoding. Speculative verification and the interaction of deterministic writes with those verification paths have not been audited; unsupported combinations are refused. Tile manifests and parser identities are exposed in route/numerics metadata so experimental configurations can be distinguished.

**What the calculator cannot supply by itself.** Consider a story that mentions a bus route number, a date, three quantities, and a price. The arithmetic engine does not understand which quantities belong in the answer. The language model must interpret the story and express the correct calculation. If it writes the wrong expression, the calculator can evaluate that wrong expression perfectly.

This is why the evaluation separates several questions:

- Can the arithmetic parser and evaluator handle large and unfamiliar operands exactly?
- Does the complete model-plus-tile system produce a correct final answer?
- Can the small model extract the right expression from a noisy question?
- How often does the tile actually activate, and what does it change?
- What happens on a standardized word-problem test such as GSM8K?

A gain on bare multiplication does not automatically answer the other questions. Likewise, a GSM8K miss does not by itself identify an arithmetic failure: the model may have chosen the wrong operations, lost track of a condition, exceeded its output budget, or failed to expose a supported expression.

**Where throughput overhead comes from.** The learned gate adds a small vector calculation at its read layer. The controller also decodes recent tokens and attempts a parse during generation. When a complete candidate appears, checking the gate can require synchronizing a result from the accelerator to the CPU. A calculator activation adds answer planning and output-token masking. Building tokenizer lookup information has a one-time cost that can affect an early request.

On ordinary non-math text, the calculator should stay closed. The vector observation, text decoding, and parser checks can still cost time. The running throughput test therefore compares the completely unwrapped stock model with the wrapped calculator route, warms both paths, alternates their order, uses equal output-token budgets, and records die temperatures. We must measure this cost; the size of the probe file is not enough to predict it.

**How we can tell whether it really engaged.** The runtime records candidate cues, opened and closed gates, parse-refusal reasons, created answer plans, written tokens, completed plans, and aborted plans. Version 3 also counts symbolic column-layout openings. Per-request events record the position, recognized expression, computed result, and answer text.

These records distinguish a selected tile from a tile that actually wrote tokens. They also let us inspect why two runs differ. A final accuracy score is useful, but the event records explain whether the calculator repaired a numerical step or whether the model changed its reasoning after seeing the corrected answer.

The design's useful promise is narrow and testable: when the generation process exposes a supported arithmetic expression at a valid answer slot, the tile can replace an uncertain digit prediction with an exact calculation. The broader questions—reliable extraction, benchmark gains, and worthwhile speed overhead—are what the current experiments are intended to establish.

Implementation references: `src/mlx2/runtime/amoeba_det_tiles.py` (parser, evaluator, detector, controller, artifacts and receipts); `research/amoeba/det_calc.py` and `scripts/amoeba_det_calc_cpu.py` (original detector-building experiments); `scripts/research/amoeba_calc_eval.py` (current stock-versus-calculator evaluation); `tests/test_amoeba_det_tiles_cpu.py` (focused behavior and refusal tests).


## Explicit whole-request calculation and counting (opt-in follow-up)

`complete_literal_requests` adds a strict symbolic gate for requests that consist entirely of a literal mathematical expression, a quoted character count, a numeric-list count, or a numeric-list sum. It reads the actual user question, not an evaluator expression field or expected answer. The CPU checks the complete grammar and builds syntax. Ambiguous prose and general word problems are refused by this gate and continue through ordinary model reasoning. This option is off unless selected in the artifact.

For an accepted expression, numerical work uses the selected arithmetic backend. In the GPU route, exact rational arithmetic remains in the existing Metal kernels. Character/list counts use GPU integer comparisons and reduction; list sums use the exact rational Metal engine. The GPU counting route currently supports ASCII characters only and refuses Unicode instead of silently using the CPU. The CPU reference can count Unicode code points. Counting defaults to case insensitive, with an explicit case-sensitive form. Bounded explicit `Count ... =` and `Sum [...] =` lines can also trigger during generated reasoning. This does not provide a learned general semantic counting ability.

When the entire user request has been solved, the processor emits its exact answer and an end token. That prevents the neural continuation from replacing the computed value with a rounded number or the last operand. This is a deterministic calculator request route, not evidence that the 0.6B model learned those operations. It is separate from the original result-cue mechanism and is reported separately in experiments. The gate preserves physical row binding, explicit control overrides, and upstream token masks.

## GPU fence stall and benchmark recovery

The offline batched harness inherited `MLX_METAL_FAST_SYNCH=1` without the production generation-thread stream binding. The runtime already documents a cross-stream GPU fence spin hazard for that combination. The hard1000 run stopped at 856 saved pairs; a retry stalled in the same batch and the host was rebooted. A later 30-second native watchdog captured a wait inside model-forward evaluation, before calculator execution. Disabling the cue filter and synchronizing callbacks did not resolve it.

The harness now pins `MLX_METAL_FAST_SYNCH=0` before MLX/MLX2 initialization and binds MLX-LM's actual generation stream before creating model/cache state. The blocked four-question batch then completed with the original calculator and GPU cue filter enabled. This is strong evidence for the fence-hazard diagnosis, not a GPU instruction trace proving which fence spun. The watchdog and per-arm journals preserve progress if an extension blocks. Production serving defaults are unchanged. Any throughput quoted for the earlier synchronization profile is historical and does not measure this repaired harness.

## Resident calculator

`calc-parse-v5-resident` moves the calculation pipeline into the tile. The CPU translates the current user turn to bytes at the request boundary and prepares a vocabulary table when it first sees a tokenizer. On the generation path, the GPU checks recent token bytes for a cue, reads a complete supported expression, evaluates it, builds the answer's token sequence and applies it to the model's scores. The host dispatches kernels and moves requests and responses through the API. It can read counters when producing a receipt.

The vocabulary table keeps each token's exact bytes. This lets one GPU check recognize an equals sign, trailing delimiter or LaTeX `\hline` even when the tokenizer splits it across several pieces. The parser sees the full candidate and translates supported LaTeX formatting before checking the size and depth of the expression. The device program fits at most 256 instructions and consumes at most 4,096 input bytes.

The evaluator tracks integers and reduced fractions using decimal digits. It can cancel an intermediate fraction before formatting the final answer. The gate can open once every part of the candidate has been validated and the exact result can be expressed in the supported output format. Unknown syntax, division by zero, out-of-range operations and nonterminating final decimal results produce no answer plan. The processor leaves token scores unchanged on refusal.

Gemma 4 and Qwen use the same parser and evaluator. A model adapter supplies the tokenizer, binds the current request to the proper row and checks that the manifest names the loaded checkpoint. The resident manifest needs no learned detection weights. A word problem still depends on the model expressing the right calculation. The tile can then evaluate a complete expression it recognizes in the model's output.

During DFlash verification, each provisional prefix has its own calculator state. The verifier commits the state associated with the accepted prefix so a discarded proposal cannot consume an answer digit or open a gate in a live request.

[Resident calculator one-pager](calculator-tile-explained-resident.pdf) · [Gemma integration](2026-11-17-calculator-gemma4.md) · [Speculative decoding integration](2026-11-17-calculator-speculative.md)

License terms for covered original calculator documentation appear in [Pierre Lamy Calculator NC-ND 1.0](../../provenance/LICENSE-calculator.txt) and the [scope and earlier grants](../../provenance/LICENSE-calculator-scope.json).
