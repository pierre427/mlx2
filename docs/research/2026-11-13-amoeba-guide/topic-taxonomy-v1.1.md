# Amoeba topic taxonomy v1.1

Schema `mlx2.amoeba-topics.v1`, from `docs/research/amoeba-topics/taxonomy.v1.json` on branch `claude/amoeba-topics-20261114`. The probe corpus manifest was built at mlx2 `663f64efd`.

**Size:** 8,299 nodes: 379 subject topics and 7,919 languages (all of ISO 639-3). **232 topics are tested**, meaning they have probe texts in the corpus (11518 texts, split into fit and held halves).

**Rules:** IDs are permanent dotted paths that are never renumbered or reused. Adding a node bumps the minor version. Nodes are never removed: a retired node is marked deprecated and points to its replacement. Moving a node means deprecating it and adding a new ID. A model need not have a cluster for every topic; unsplit topics stay in the core as tagged, core-resident knowledge.

**Source schemes:** cip, open_syllabus_galaxy, iso639_3, linguist, msc, arxiv, acm. Open Syllabus Galaxy has no public field list or API, so its source scheme, US CIP 2020, is used instead.

`form.*` nodes describe a text's form and genre, not its subject. `skill.*` nodes are in-context skills, not knowledge.


## Programming languages (`prog`)

- `prog.assembly` Assembly — **tested**, 48 probes
  - `prog.assembly.arm64` ARM64 assembly · linguist Assembly — **tested**, 42 probes
  - `prog.assembly.x86` x86 assembly · linguist Assembly — 34 probes (not tested)
- `prog.build` Build and infrastructure languages
  - `prog.build.bazel` Bazel / Starlark · linguist Starlark — 1 probes (not tested)
  - `prog.build.cmake` CMake · linguist CMake — **tested**, 48 probes
  - `prog.build.dockerfile` Dockerfile · linguist Dockerfile — 29 probes (not tested)
  - `prog.build.hcl` HCL / Terraform · linguist HCL — 3 probes (not tested)
  - `prog.build.make` Makefile · linguist Makefile — **tested**, 48 probes
  - `prog.build.nix` Nix · linguist Nix
- `prog.c` C · linguist C — **tested**, 48 probes
- `prog.coq` Coq / Rocq · linguist Coq
- `prog.cpp` C++ · linguist C++ — **tested**, 48 probes
- `prog.csharp` C# · linguist C# — 2 probes (not tested)
- `prog.dart` Dart · linguist Dart
- `prog.data` Data and configuration formats
  - `prog.data.csv` CSV / TSV · linguist CSV — **tested**, 48 probes
  - `prog.data.ini` INI / conf · linguist INI — **tested**, 48 probes
  - `prog.data.json` JSON · linguist JSON — **tested**, 48 probes
  - `prog.data.protobuf` Protocol Buffers · linguist Protocol Buffer — **tested**, 48 probes
  - `prog.data.toml` TOML · linguist TOML — **tested**, 48 probes
  - `prog.data.xml` XML / plist · linguist XML — **tested**, 48 probes
  - `prog.data.yaml` YAML · linguist YAML — **tested**, 48 probes
- `prog.elixir` Elixir · linguist Elixir
- `prog.erlang` Erlang · linguist Erlang
- `prog.fortran` Fortran · linguist Fortran — 35 probes (not tested)
- `prog.fsharp` F# · linguist F#
- `prog.go` Go · linguist Go — **tested**, 48 probes
- `prog.gpu` GPU kernel languages
  - `prog.gpu.cuda` CUDA · linguist Cuda — **tested**, 48 probes
  - `prog.gpu.glsl` GLSL · linguist GLSL
  - `prog.gpu.hlsl` HLSL · linguist HLSL
  - `prog.gpu.metal` Metal Shading Language · linguist Metal — 39 probes (not tested)
  - `prog.gpu.triton` Triton (Python DSL) — 1 probes (not tested)
  - `prog.gpu.wgsl` WGSL · linguist WGSL
- `prog.haskell` Haskell · linguist Haskell
- `prog.java` Java · linguist Java — **tested**, 48 probes
- `prog.javascript` JavaScript · linguist JavaScript — **tested**, 48 probes
- `prog.julia` Julia · linguist Julia
- `prog.kotlin` Kotlin · linguist Kotlin
- `prog.lean` Lean · linguist Lean — **tested**, 48 probes
- `prog.lisp` Lisp family
  - `prog.lisp.clojure` Clojure · linguist Clojure
  - `prog.lisp.common_lisp` Common Lisp · linguist Common Lisp — 4 probes (not tested)
  - `prog.lisp.emacs_lisp` Emacs Lisp · linguist Emacs Lisp — 5 probes (not tested)
  - `prog.lisp.scheme` Scheme / Racket · linguist Scheme — 2 probes (not tested)
- `prog.lua` Lua · linguist Lua — **tested**, 48 probes
- `prog.markup` Documentation markup
  - `prog.markup.asciidoc` AsciiDoc · linguist AsciiDoc — **tested**, 48 probes
  - `prog.markup.latex` LaTeX / TeX · linguist TeX — 2 probes (not tested)
  - `prog.markup.man` roff / man pages · linguist Roff — **tested**, 48 probes
  - `prog.markup.markdown` Markdown · linguist Markdown — **tested**, 48 probes
  - `prog.markup.rst` reStructuredText · linguist reStructuredText — **tested**, 48 probes
- `prog.matlab` MATLAB / Octave · linguist MATLAB
- `prog.nim` Nim · linguist Nim
- `prog.objc` Objective-C · linguist Objective-C — 11 probes (not tested)
- `prog.ocaml` OCaml · linguist OCaml
- `prog.perl` Perl · linguist Perl — **tested**, 48 probes
  - `prog.perl.pod` Perl POD documentation · linguist Pod — **tested**, 48 probes
- `prog.php` PHP · linguist PHP — 6 probes (not tested)
- `prog.prolog` Prolog · linguist Prolog
- `prog.python` Python · linguist Python — **tested**, 48 probes
  - `prog.python.cython` Cython · linguist Cython — **tested**, 48 probes
  - `prog.python.ml_frameworks` Python ML frameworks (PyTorch, MLX, JAX)
  - `prog.python.packaging` Python packaging (setup, pyproject) — 6 probes (not tested)
  - `prog.python.scientific` Scientific Python (NumPy, SciPy, pandas) — **tested**, 48 probes
  - `prog.python.stdlib` Python standard library — **tested**, 48 probes
  - `prog.python.testing` Python tests (pytest, unittest) — **tested**, 48 probes
  - `prog.python.typing` Python typing and stubs · linguist Python — **tested**, 48 probes
  - `prog.python.web` Python web frameworks — **tested**, 48 probes
- `prog.r` R · linguist R — 2 probes (not tested)
- `prog.regex` Regular expressions
- `prog.ruby` Ruby · linguist Ruby — **tested**, 48 probes
- `prog.rust` Rust · linguist Rust — **tested**, 48 probes
- `prog.scala` Scala · linguist Scala
- `prog.shell` Shell scripting · linguist Shell — **tested**, 48 probes
  - `prog.shell.bash` Bash · linguist Shell — **tested**, 48 probes
  - `prog.shell.batch` Windows batch · linguist Batchfile — **tested**, 48 probes
  - `prog.shell.fish` fish · linguist fish — **tested**, 42 probes
  - `prog.shell.powershell` PowerShell · linguist PowerShell — **tested**, 48 probes
  - `prog.shell.sh` POSIX sh · linguist Shell — **tested**, 48 probes
  - `prog.shell.zsh` Zsh · linguist Shell — **tested**, 38 probes
- `prog.sql` SQL · linguist SQL — **tested**, 48 probes
- `prog.swift` Swift · linguist Swift
- `prog.tcl` Tcl · linguist Tcl — 2 probes (not tested)
- `prog.typescript` TypeScript · linguist TypeScript — **tested**, 40 probes
- `prog.web` Web markup and styling
  - `prog.web.css` CSS · linguist CSS — **tested**, 48 probes
  - `prog.web.html` HTML · linguist HTML — **tested**, 48 probes
  - `prog.web.scss` SCSS / Sass · linguist SCSS — 1 probes (not tested)
  - `prog.web.templates` Web templates (Jinja, ERB, Handlebars) · linguist Jinja — **tested**, 48 probes
- `prog.zig` Zig · linguist Zig

## Software engineering practice (`swe`) — **tested**, 48 probes

- `swe.api_design` API design
- `swe.build_systems` Build systems and compilation
- `swe.ci` Continuous integration
- `swe.cli_design` Command-line interface design
- `swe.code_review` Code review
- `swe.concurrency` Concurrency and parallel programming
- `swe.databases` Databases in practice
- `swe.debugging` Debugging and tracebacks
- `swe.devops` DevOps
  - `swe.devops.containers` Containers
  - `swe.devops.deployment` Deployment and services (launchd, systemd) — **tested**, 48 probes
- `swe.documentation` Technical documentation
- `swe.git` Git and version control — **tested**, 48 probes
  - `swe.git.branching` Branches, merges, rebases — **tested**, 48 probes
  - `swe.git.commits` Commit messages and history — **tested**, 48 probes
- `swe.licensing` Software licences — **tested**, 48 probes
- `swe.networking` Networking and HTTP
- `swe.os` Operating systems in practice
  - `swe.os.linux` Linux
  - `swe.os.macos` macOS
  - `swe.os.unix_tools` Unix command-line tools — **tested**, 48 probes
- `swe.packaging` Packaging and dependency management — **tested**, 48 probes
- `swe.performance` Profiling and performance engineering — **tested**, 48 probes
- `swe.refactoring` Refactoring and code design
- `swe.repo_management` Repository management (issues, PRs, releases) — **tested**, 48 probes
- `swe.security` Software security — 34 probes (not tested)
- `swe.testing` Testing and test design — **tested**, 41 probes
- `swe.web_dev` Web development

## Machine learning and AI (`ml`)

- `ml.deep_learning` Deep learning architectures
  - `ml.deep_learning.attention` Attention mechanisms — **tested**, 48 probes
  - `ml.deep_learning.diffusion` Diffusion models — **tested**, 48 probes
  - `ml.deep_learning.embeddings` Embeddings and retrieval · arxiv cs.IR — **tested**, 48 probes
  - `ml.deep_learning.linear_attention` Linear attention and state-space models (GDN, Mamba) — **tested**, 48 probes
  - `ml.deep_learning.moe` Mixture of experts — **tested**, 48 probes
  - `ml.deep_learning.speech` Speech and audio models · arxiv eess.AS — **tested**, 48 probes
  - `ml.deep_learning.transformers` Transformers
  - `ml.deep_learning.vision` Vision models · arxiv cs.CV — **tested**, 48 probes
- `ml.foundations` ML foundations (losses, optimisation, generalisation) · arxiv stat.ML — **tested**, 48 probes
- `ml.frameworks` ML frameworks
  - `ml.frameworks.huggingface` Hugging Face transformers — **tested**, 48 probes
  - `ml.frameworks.jax` JAX
  - `ml.frameworks.llama_cpp` llama.cpp — 17 probes (not tested)
  - `ml.frameworks.mlx` MLX — **tested**, 48 probes
  - `ml.frameworks.mlx2` mlx2 — **tested**, 48 probes
  - `ml.frameworks.pytorch` PyTorch — **tested**, 48 probes
  - `ml.frameworks.vllm` vLLM / SGLang — **tested**, 48 probes
- `ml.hardware` ML hardware
  - `ml.hardware.ane` Apple Neural Engine — **tested**, 41 probes
  - `ml.hardware.apple_silicon` Apple silicon GPU — **tested**, 47 probes
  - `ml.hardware.nvidia` NVIDIA GPUs — 35 probes (not tested)
- `ml.inference` LLM inference engineering
  - `ml.inference.batching` Batching and scheduling (continuous batching) — **tested**, 48 probes
  - `ml.inference.benchmarking` Inference benchmarking — **tested**, 48 probes
  - `ml.inference.conversion` Model conversion and checkpoints — **tested**, 48 probes
  - `ml.inference.decode` Decode — **tested**, 48 probes
  - `ml.inference.distributed` Distributed inference (tensor/pipeline parallel) — **tested**, 48 probes
  - `ml.inference.kernels` Inference kernels
    - `ml.inference.kernels.attention_kernels` Attention kernels (flash, paged, SDPA) — **tested**, 48 probes
    - `ml.inference.kernels.cuda_kernels` CUDA kernels — 22 probes (not tested)
    - `ml.inference.kernels.matmul` Matmul / GEMV kernels — **tested**, 48 probes
    - `ml.inference.kernels.metal_kernels` Metal kernels — **tested**, 48 probes
  - `ml.inference.kv_cache` KV cache — **tested**, 48 probes
  - `ml.inference.long_context` Long context (RoPE scaling, sparse attention) — **tested**, 48 probes
  - `ml.inference.memory` Memory management (unified memory, footprint, swap) — **tested**, 48 probes
  - `ml.inference.prefill` Prefill — **tested**, 48 probes
  - `ml.inference.prefix_cache` Prefix caching (APC) — **tested**, 48 probes
  - `ml.inference.quantization` Quantisation — **tested**, 48 probes
    - `ml.inference.quantization.affine` Affine / group quantisation (q4, q8) — **tested**, 48 probes
    - `ml.inference.quantization.gguf` GGUF / llama.cpp formats — 28 probes (not tested)
    - `ml.inference.quantization.microscaling` Microscaling formats (MXFP4/8, NVFP4) — **tested**, 34 probes
    - `ml.inference.quantization.trellis` Trellis / codebook quantisation (EXL3, QTIP) — 20 probes (not tested)
  - `ml.inference.sampling` Sampling (temperature, top-k, top-p) — **tested**, 48 probes
  - `ml.inference.serving` Serving and APIs
    - `ml.inference.serving.openai_api` OpenAI-compatible API — 16 probes (not tested)
    - `ml.inference.serving.streaming` Streaming responses — 8 probes (not tested)
  - `ml.inference.speculative` Speculative decoding
    - `ml.inference.speculative.draft_model` Draft-model speculation — **tested**, 48 probes
    - `ml.inference.speculative.mtp` Multi-token prediction heads — **tested**, 48 probes
    - `ml.inference.speculative.prompt_lookup` Prompt lookup decoding — 15 probes (not tested)
    - `ml.inference.speculative.tree` Tree speculation — **tested**, 48 probes
- `ml.interpretability` Interpretability
  - `ml.interpretability.neurons` Neuron and feature analysis — 18 probes (not tested)
  - `ml.interpretability.probing` Probing and representation analysis — **tested**, 48 probes
  - `ml.interpretability.pruning` Pruning, masking and model surgery — **tested**, 39 probes
  - `ml.interpretability.sae` Sparse autoencoders
- `ml.llm` Large language models
  - `ml.llm.agents` Agents and harnesses — **tested**, 48 probes
  - `ml.llm.alignment` RLHF and alignment
  - `ml.llm.evaluation` Evaluation and benchmarks — 12 probes (not tested)
  - `ml.llm.finetuning` Fine-tuning (SFT, LoRA) — **tested**, 48 probes
  - `ml.llm.pretraining` Pretraining
  - `ml.llm.prompting` Prompting and chat templates — 16 probes (not tested)
  - `ml.llm.reasoning` Reasoning and chain of thought — **tested**, 48 probes
  - `ml.llm.tokenization` Tokenisation — **tested**, 48 probes
  - `ml.llm.tool_use` Tool use and function calling — **tested**, 48 probes
- `ml.nlp` Natural language processing (pre-LLM and general) · arxiv cs.CL — **tested**, 48 probes

## Computer science (theory and systems) (`cs`)

- `cs.algorithms` Algorithms and data structures · arxiv cs.DS — **tested**, 48 probes
- `cs.complexity` Computational complexity · arxiv cs.CC — **tested**, 48 probes
- `cs.cryptography` Cryptography · arxiv cs.CR — **tested**, 48 probes
- `cs.databases` Database theory · arxiv cs.DB — **tested**, 48 probes
- `cs.distributed` Distributed computing · arxiv cs.DC — **tested**, 48 probes
- `cs.formal_methods` Formal methods and proof assistants · arxiv cs.LO — **tested**, 48 probes
- `cs.game_theory` Algorithmic game theory · arxiv cs.GT — **tested**, 48 probes
- `cs.graphics` Computer graphics · arxiv cs.GR — **tested**, 48 probes
- `cs.hci` Human-computer interaction · arxiv cs.HC — **tested**, 48 probes
- `cs.information_theory` Information theory · arxiv cs.IT — **tested**, 48 probes
- `cs.networks` Computer networks · arxiv cs.NI — **tested**, 48 probes
- `cs.pl_theory` Programming-language theory · arxiv cs.PL — **tested**, 48 probes
- `cs.robotics` Robotics · arxiv cs.RO — **tested**, 48 probes
- `cs.systems` Operating systems and architecture · arxiv cs.OS — **tested**, 48 probes

## Mathematics (`math`) — **tested**, 48 probes

- `math.algebra` Algebra · arxiv math.RA, msc 08 — **tested**, 48 probes
  - `math.algebra.category_theory` Category theory · arxiv math.CT, msc 18 — **tested**, 48 probes
  - `math.algebra.group_theory` Group theory · arxiv math.GR, msc 20 — **tested**, 48 probes
  - `math.algebra.linear_algebra` Linear algebra · msc 15 — **tested**, 48 probes
  - `math.algebra.operator_algebras` Operator algebras · arxiv math.OA, msc 46L — **tested**, 48 probes
  - `math.algebra.quantum_algebra` Quantum algebra · arxiv math.QA — **tested**, 48 probes
  - `math.algebra.representation_theory` Representation theory · arxiv math.RT — **tested**, 48 probes
  - `math.algebra.ring_theory` Rings, fields and commutative algebra · arxiv math.AC, msc 13 — **tested**, 48 probes
- `math.analysis` Analysis
  - `math.analysis.calculus` Calculus — **tested**, 48 probes
  - `math.analysis.functional_analysis` Functional analysis · arxiv math.FA, msc 46 — **tested**, 48 probes
  - `math.analysis.harmonic` Harmonic analysis · msc 42 — **tested**, 44 probes
  - `math.analysis.ode` Ordinary differential equations · msc 34 — **tested**, 44 probes
  - `math.analysis.pde` Partial differential equations · arxiv math.AP, msc 35 — **tested**, 48 probes
  - `math.analysis.real_complex` Real and complex analysis · arxiv math.CA, msc 26 — **tested**, 48 probes
- `math.arithmetic` Elementary arithmetic
  - `math.arithmetic.addition` Addition — **tested**, 48 probes
  - `math.arithmetic.multiplication` Multiplication — **tested**, 48 probes
- `math.combinatorics` Combinatorics · arxiv math.CO, msc 05 — **tested**, 48 probes
- `math.dynamical_systems` Dynamical systems and ergodic theory · arxiv math.DS, msc 37 — **tested**, 48 probes
- `math.geometry` Geometry
  - `math.geometry.algebraic_geometry` Algebraic and complex geometry · arxiv math.AG, msc 14 — **tested**, 48 probes
  - `math.geometry.differential_geometry` Differential geometry · arxiv math.DG, msc 53 — **tested**, 48 probes
  - `math.geometry.euclidean` Euclidean geometry · msc 51 — **tested**, 45 probes
  - `math.geometry.metric_geometry` Convex and metric geometry · arxiv math.MG, msc 52 — **tested**, 48 probes
  - `math.geometry.symplectic` Symplectic geometry · arxiv math.SG, msc 53D — **tested**, 48 probes
- `math.logic` Mathematical logic and foundations · arxiv math.LO, msc 03 — **tested**, 48 probes
- `math.mathematical_physics` Mathematical physics · arxiv math-ph, msc 81 — **tested**, 48 probes
- `math.number_theory` Number theory · arxiv math.NT, msc 11 — **tested**, 48 probes
- `math.numerical_analysis` Numerical analysis · arxiv math.NA, msc 65 — **tested**, 48 probes
- `math.optimization` Optimisation and control · arxiv math.OC, msc 90 — **tested**, 48 probes
- `math.probability` Probability and statistical mechanics · arxiv math.PR, msc 60 — **tested**, 48 probes
- `math.statistics` Statistics · arxiv math.ST, msc 62 — **tested**, 48 probes
- `math.theoretical_cs` Theoretical computer science (as mathematics) · msc 68 — **tested**, 48 probes
- `math.topology` Topology · arxiv math.GT, msc 54 — **tested**, 48 probes
  - `math.topology.algebraic_topology` Algebraic topology · arxiv math.AT, msc 55 — **tested**, 48 probes
- `math.word_problems` Word problems (grade school) — **tested**, 48 probes

## Natural sciences (`sci`)

- `sci.astronomy` Astronomy and astrophysics · arxiv astro-ph, cip 40.02 — **tested**, 48 probes
  - `sci.astronomy.cosmology` Cosmology · arxiv astro-ph.CO — **tested**, 48 probes
  - `sci.astronomy.galaxies` Galaxies · arxiv astro-ph.GA — **tested**, 48 probes
  - `sci.astronomy.high_energy` High-energy astrophysics · arxiv astro-ph.HE — **tested**, 48 probes
  - `sci.astronomy.instrumentation` Astronomical instrumentation · arxiv astro-ph.IM — **tested**, 48 probes
  - `sci.astronomy.planetary` Earth and planetary astrophysics · arxiv astro-ph.EP — **tested**, 48 probes
  - `sci.astronomy.stellar` Solar and stellar astrophysics · arxiv astro-ph.SR — **tested**, 48 probes
- `sci.biology` Biology · arxiv q-bio, cip 26, os_field Biology — **tested**, 48 probes
  - `sci.biology.botany` Botany
  - `sci.biology.cell` Cell biology · arxiv q-bio.CB — **tested**, 48 probes
  - `sci.biology.ecology` Ecology
  - `sci.biology.evolution` Evolution · arxiv q-bio.PE — **tested**, 48 probes
  - `sci.biology.genetics` Genetics and genomics · arxiv q-bio.GN — **tested**, 48 probes
  - `sci.biology.immunology` Immunology — **tested**, 48 probes
  - `sci.biology.microbiology` Microbiology — **tested**, 48 probes
  - `sci.biology.molecular` Molecular biology · arxiv q-bio.BM — **tested**, 48 probes
  - `sci.biology.neuroscience` Neuroscience · arxiv q-bio.NC — **tested**, 48 probes
  - `sci.biology.zoology` Zoology
- `sci.chemistry` Chemistry · cip 40.05, os_field Chemistry
  - `sci.chemistry.analytical` Analytical chemistry
  - `sci.chemistry.biochemistry` Biochemistry · cip 26.02
  - `sci.chemistry.inorganic` Inorganic chemistry
  - `sci.chemistry.organic` Organic chemistry
  - `sci.chemistry.physical` Physical chemistry — **tested**, 48 probes
- `sci.earth` Earth sciences · cip 40.06
  - `sci.earth.geology` Geology — **tested**, 48 probes
  - `sci.earth.meteorology` Meteorology and climate · cip 40.04 — **tested**, 48 probes
  - `sci.earth.oceanography` Oceanography
- `sci.materials` Materials science · cip 40.10 — **tested**, 48 probes
- `sci.medicine` Medicine and health · cip 51, os_field Medicine
  - `sci.medicine.cardiology` Cardiology — **tested**, 48 probes
  - `sci.medicine.clinical` Clinical medicine and trials — **tested**, 48 probes
  - `sci.medicine.epidemiology` Epidemiology and public health — **tested**, 48 probes
  - `sci.medicine.infectious_disease` Infectious disease — **tested**, 48 probes
  - `sci.medicine.nursing` Nursing · cip 51.38, os_field Nursing
  - `sci.medicine.oncology` Oncology — **tested**, 48 probes
  - `sci.medicine.pharmacology` Pharmacology — **tested**, 48 probes
- `sci.physics` Physics · arxiv physics, cip 40.08, os_field Physics — **tested**, 48 probes
  - `sci.physics.atomic` Atomic and molecular physics · arxiv physics.atom-ph — **tested**, 48 probes
  - `sci.physics.classical_mechanics` Classical mechanics · arxiv physics.class-ph — **tested**, 48 probes
  - `sci.physics.condensed_matter` Condensed matter · arxiv cond-mat — **tested**, 48 probes
  - `sci.physics.electromagnetism` Electromagnetism
  - `sci.physics.fluids` Fluid dynamics · arxiv physics.flu-dyn — **tested**, 48 probes
  - `sci.physics.high_energy_theory` High-energy theory (fields, strings) · arxiv hep-th — **tested**, 48 probes
  - `sci.physics.nonlinear` Nonlinear dynamics and chaos · arxiv nlin — **tested**, 48 probes
  - `sci.physics.nuclear` Nuclear physics · arxiv nucl-th — **tested**, 48 probes
  - `sci.physics.optics` Optics · arxiv physics.optics — **tested**, 48 probes
  - `sci.physics.particle` Particle physics · arxiv hep-ph — **tested**, 48 probes
  - `sci.physics.plasma` Plasma physics · arxiv physics.plasm-ph — **tested**, 48 probes
  - `sci.physics.quantum` Quantum mechanics · arxiv quant-ph — **tested**, 48 probes
  - `sci.physics.relativity` Relativity and gravitation · arxiv gr-qc — **tested**, 48 probes
  - `sci.physics.thermodynamics` Thermodynamics and statistical physics · arxiv cond-mat.stat-mech — **tested**, 48 probes
  - `sci.physics.units` Physical units and measurement — **tested**, 48 probes

## Engineering (`eng`)

- `eng.aerospace` Aerospace engineering · cip 14.02
- `eng.biomedical` Biomedical engineering · cip 14.05
- `eng.chemical` Chemical engineering · cip 14.07
- `eng.civil` Civil engineering · cip 14.08 — 23 probes (not tested)
- `eng.electrical` Electrical and electronic engineering · arxiv eess, cip 14.10 — **tested**, 48 probes
- `eng.industrial` Industrial engineering · cip 14.35
- `eng.mechanical` Mechanical engineering · cip 14.19
- `eng.signal_processing` Signal processing · arxiv eess.SP — **tested**, 48 probes

## Humanities (`hum`)

- `hum.art_history` Art and art history · cip 50.07, os_field Fine Arts
- `hum.classics` Classics · cip 16.12, os_field Classics
- `hum.film` Film and media · cip 50.06, os_field Film and Photography
- `hum.history` History · cip 54, os_field History — **tested**, 48 probes
  - `hum.history.ancient` Ancient history
  - `hum.history.medieval` Medieval history
  - `hum.history.military` Military history
  - `hum.history.modern` Modern history
- `hum.linguistics` Linguistics · arxiv cs.CL, cip 16.01
- `hum.literature` Literature · cip 23, os_field English Literature
  - `hum.literature.criticism` Literary criticism
  - `hum.literature.drama` Drama
  - `hum.literature.fiction` Fiction
  - `hum.literature.poetry` Poetry
- `hum.music` Music · cip 50.09, os_field Music
- `hum.philosophy` Philosophy · cip 38.01, os_field Philosophy
  - `hum.philosophy.epistemology` Epistemology
  - `hum.philosophy.ethics` Ethics
  - `hum.philosophy.metaphysics` Metaphysics
  - `hum.philosophy.political` Political philosophy
- `hum.religion` Religion and theology · cip 38.02, os_field Religion

## Social sciences (`soc`)

- `soc.anthropology` Anthropology · cip 45.02, os_field Anthropology
- `soc.business` Business and management · cip 52, os_field Business
  - `soc.business.marketing` Marketing · cip 52.14, os_field Marketing
- `soc.communication` Communication and journalism · cip 09, os_field Journalism
- `soc.economics` Economics · arxiv econ, cip 45.06, os_field Economics — **tested**, 48 probes
  - `soc.economics.accounting` Accounting · cip 52.03, os_field Accounting
  - `soc.economics.finance` Finance · arxiv q-fin, cip 52.08 — **tested**, 48 probes
- `soc.education` Education · cip 13, os_field Education — **tested**, 48 probes
- `soc.geography` Geography · cip 45.07, os_field Geography
  - `soc.geography.countries` Countries and places
  - `soc.geography.world_capitals` World capitals
- `soc.law` Law · cip 22, os_field Law
  - `soc.law.constitutional` Constitutional law
  - `soc.law.contracts` Contract law
  - `soc.law.criminal` Criminal law
  - `soc.law.us_legislation` US legislation (bills, statutes) — **tested**, 48 probes
- `soc.political_science` Political science · cip 45.10, os_field Political Science
- `soc.psychology` Psychology · cip 42, os_field Psychology
- `soc.sociology` Sociology · cip 45.11, os_field Sociology — **tested**, 48 probes

## Everyday life and practical knowledge (`life`)

- `life.cooking` Cooking and food
- `life.games` Games
- `life.health_wellness` Health and wellness
- `life.home_diy` Home and DIY
- `life.personal_finance` Personal finance
- `life.sports` Sports
- `life.travel` Travel

## Text form and genre (not subject matter) (`form`)

- `form.academic_register` Academic register
- `form.dialogue` Dialogue and chat turns
- `form.digits` Digits and numerals
- `form.instructions` Instructions and how-tos
- `form.legal_register` Legal register
- `form.list` Lists and enumerations
- `form.markdown_structure` Markdown structure
  - `form.markdown_structure.blockquote` Block quotes
  - `form.markdown_structure.headings` Headings and bold
  - `form.markdown_structure.inline_code` Inline code and fences
  - `form.markdown_structure.tables` Tables (pipes)
- `form.narrative` Narrative
  - `form.narrative.childrens_story` Children's stories — **tested**, 48 probes
  - `form.narrative.story_opening` Story openings — **tested**, 48 probes
- `form.qa` Question and answer
- `form.subword` Sub-word and orthographic pieces

## In-context skills (`skill`)

- `skill.arithmetic_reasoning` Multi-step arithmetic reasoning
- `skill.copying` Copying and repetition
- `skill.in_context_recall` In-context recall
- `skill.translation` Translation

## Natural languages (`lang`)

All ISO 639-3 codes are present, by type: living 7,078, extinct 602, historical 215, constructed 24. The full list is in the JSON. Languages with probe texts (FLORES+ devtest, macOS .lproj strings, local text):

| ID | Language | Probes | Tested |
|---|---|---|---|
| `lang.ara` | Arabic | 48 | yes |
| `lang.cat` | Catalan | 48 | yes |
| `lang.ces` | Czech | 48 | yes |
| `lang.dan` | Danish | 48 | yes |
| `lang.deu` | German | 48 | yes |
| `lang.ell` | Modern Greek (1453-) | 48 | yes |
| `lang.eng` | English | 48 | yes |
| `lang.fin` | Finnish | 48 | yes |
| `lang.fra` | French | 48 | yes |
| `lang.heb` | Hebrew | 48 | yes |
| `lang.hin` | Hindi | 48 | yes |
| `lang.hrv` | Croatian | 48 | yes |
| `lang.hun` | Hungarian | 48 | yes |
| `lang.ind` | Indonesian | 48 | yes |
| `lang.ita` | Italian | 48 | yes |
| `lang.jpn` | Japanese | 48 | yes |
| `lang.kor` | Korean | 48 | yes |
| `lang.msa` | Malay (macrolanguage) | 48 | yes |
| `lang.nld` | Dutch | 48 | yes |
| `lang.nob` | Norwegian Bokmål | 1 |  |
| `lang.nor` | Norwegian | 48 | yes |
| `lang.pol` | Polish | 48 | yes |
| `lang.por` | Portuguese | 48 | yes |
| `lang.ron` | Romanian | 48 | yes |
| `lang.rus` | Russian | 48 | yes |
| `lang.slk` | Slovak | 48 | yes |
| `lang.slv` | Slovenian | 48 | yes |
| `lang.spa` | Spanish | 48 | yes |
| `lang.swe` | Swedish | 48 | yes |
| `lang.tam` | Tamil | 1 |  |
| `lang.tha` | Thai | 48 | yes |
| `lang.tur` | Turkish | 48 | yes |
| `lang.ukr` | Ukrainian | 48 | yes |
| `lang.vie` | Vietnamese | 48 | yes |
| `lang.yue` | Yue Chinese | 10 |  |
| `lang.zho` | Chinese | 48 | yes |

## Aliases

- `arithmetic` → `math.arithmetic.addition`
- `billsum` → `soc.law.us_legislation`
- `capitals` → `soc.geography.world_capitals`
- `capitals_fresh` → `soc.geography.world_capitals`
- `capitals_phrasing` → `soc.geography.world_capitals`
- `english` → `lang.eng`
- `french` → `lang.fra`
- `gsm8k` → `math.word_problems`
- `recall` → `skill.in_context_recall`
- `repo_markdown` → `prog.markup.markdown`
- `tinystories_train` → `form.narrative.childrens_story`

## Earlier task labels mapped onto the taxonomy

- `capitals` → `['soc.geography.world_capitals']`
- `capitals_fresh` → `['soc.geography.world_capitals']`
- `capitals_phrasing` → `['soc.geography.world_capitals']`
- `arithmetic` → `['math.arithmetic.addition']`
- `french` → `['lang.fra', 'skill.translation']`
- `recall` → `['skill.in_context_recall']`
- `english` → `['lang.eng']`
- `tinystories_train` → `['form.narrative.childrens_story', 'lang.eng']`
- `repo_markdown` → `['prog.markup.markdown']`
- `donor_generated` → `['lang.eng']`
