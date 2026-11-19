#!/usr/bin/env python3
"""Build the amoeba cross-model topic taxonomy (schema mlx2.amoeba-topics.v1).

Design: docs/research/amoeba-topics/DESIGN.md. The curated outline below plus every ISO 639-3 code
(from a local pycountry copy of Debian iso-codes) become docs/research/amoeba-topics/taxonomy.v1.json.

Outline syntax: one node per line, two spaces of indent per depth level:
    <segment> | <name> | key=value;key=value      (refs and aliases are optional)
The node ID is the dotted path of segments. IDs are permanent: the builder loads the previous file and
refuses to write if any earlier ID is missing (deprecate with status=deprecated instead).

Refs: csf (NIST CSF 2.0 code), see (cross-link to an overlapping node elsewhere), cip (CIP 2020), arxiv, msc (MSC 2020), linguist (GitHub Linguist name), acm (ACM CCS 2012 top
concept), iso (ISO 639-3), os_field (Open Syllabus field name, only where the name is known to match),
ddc (Dewey Decimal class number, numbers only), usenet (Usenet newsgroup or hierarchy name),
aka (comma-separated aliases, e.g. legacy task labels).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/research/amoeba-topics/taxonomy.v1.json"
SCHEMA = "mlx2.amoeba-topics.v1"
VERSION = "1.5"
ISO_DEFAULT = ROOT / "docs/research/amoeba-topics/sources/iso639-3.json"   # pycountry 26.2.16 copy (LGPL-2.1)

OUTLINE = r"""
prog | Programming languages | acm=Software and its engineering;ddc=005.13;usenet=comp.lang.*
  python | Python | linguist=Python;usenet=comp.lang.python
    stdlib | Python standard library
    scientific | Scientific Python (NumPy, SciPy, pandas)
    ml_frameworks | Python ML frameworks (PyTorch, MLX, JAX)
    web | Python web frameworks
    typing | Python typing and stubs | linguist=Python
    testing | Python tests (pytest, unittest)
    packaging | Python packaging (setup, pyproject)
    cython | Cython | linguist=Cython
  c | C | linguist=C;usenet=comp.lang.c
  cpp | C++ | linguist=C++;usenet=comp.lang.c++
  objc | Objective-C | linguist=Objective-C
  swift | Swift | linguist=Swift
  rust | Rust | linguist=Rust
  go | Go | linguist=Go
  java | Java | linguist=Java;usenet=comp.lang.java.programmer
  kotlin | Kotlin | linguist=Kotlin
  scala | Scala | linguist=Scala
  csharp | C# | linguist=C#
  fsharp | F# | linguist=F#
  javascript | JavaScript | linguist=JavaScript;usenet=comp.lang.javascript
  typescript | TypeScript | linguist=TypeScript
  ruby | Ruby | linguist=Ruby;usenet=comp.lang.ruby
  perl | Perl | linguist=Perl;usenet=comp.lang.perl.misc
    pod | Perl POD documentation | linguist=Pod
  php | PHP | linguist=PHP
  lua | Lua | linguist=Lua
  r | R | linguist=R
  julia | Julia | linguist=Julia
  matlab | MATLAB / Octave | linguist=MATLAB
  fortran | Fortran | linguist=Fortran;usenet=comp.lang.fortran
  haskell | Haskell | linguist=Haskell;usenet=comp.lang.haskell
  ocaml | OCaml | linguist=OCaml
  lisp | Lisp family | usenet=comp.lang.lisp
    common_lisp | Common Lisp | linguist=Common Lisp
    emacs_lisp | Emacs Lisp | linguist=Emacs Lisp
    scheme | Scheme / Racket | linguist=Scheme
    clojure | Clojure | linguist=Clojure
  erlang | Erlang | linguist=Erlang
  elixir | Elixir | linguist=Elixir
  dart | Dart | linguist=Dart
  zig | Zig | linguist=Zig
  nim | Nim | linguist=Nim
  tcl | Tcl | linguist=Tcl;usenet=comp.lang.tcl
  lean | Lean | linguist=Lean
  coq | Coq / Rocq | linguist=Coq
  prolog | Prolog | linguist=Prolog;usenet=comp.lang.prolog
  assembly | Assembly | usenet=comp.lang.asm.x86
    arm64 | ARM64 assembly | linguist=Assembly
    x86 | x86 assembly | linguist=Assembly
  gpu | GPU kernel languages
    metal | Metal Shading Language | linguist=Metal
    cuda | CUDA | linguist=Cuda
    glsl | GLSL | linguist=GLSL
    hlsl | HLSL | linguist=HLSL
    wgsl | WGSL | linguist=WGSL
    triton | Triton (Python DSL)
  shell | Shell scripting | linguist=Shell;usenet=comp.unix.shell
    sh | POSIX sh | linguist=Shell
    bash | Bash | linguist=Shell
    zsh | Zsh | linguist=Shell
    fish | fish | linguist=fish
    powershell | PowerShell | linguist=PowerShell
    batch | Windows batch | linguist=Batchfile
  sql | SQL | linguist=SQL;usenet=comp.databases
  web | Web markup and styling
    html | HTML | linguist=HTML
    css | CSS | linguist=CSS
    scss | SCSS / Sass | linguist=SCSS
    templates | Web templates (Jinja, ERB, Handlebars) | linguist=Jinja
  markup | Documentation markup
    markdown | Markdown | linguist=Markdown;aka=repo_markdown
    rst | reStructuredText | linguist=reStructuredText
    latex | LaTeX / TeX | linguist=TeX
    asciidoc | AsciiDoc | linguist=AsciiDoc
    man | roff / man pages | linguist=Roff
  data | Data and configuration formats
    json | JSON | linguist=JSON
    yaml | YAML | linguist=YAML
    toml | TOML | linguist=TOML
    xml | XML / plist | linguist=XML
    csv | CSV / TSV | linguist=CSV
    ini | INI / conf | linguist=INI
    protobuf | Protocol Buffers | linguist=Protocol Buffer
  build | Build and infrastructure languages
    make | Makefile | linguist=Makefile
    cmake | CMake | linguist=CMake
    bazel | Bazel / Starlark | linguist=Starlark
    dockerfile | Dockerfile | linguist=Dockerfile
    nix | Nix | linguist=Nix
    hcl | HCL / Terraform | linguist=HCL
  regex | Regular expressions
swe | Software engineering practice | acm=Software and its engineering;ddc=005.1;usenet=comp.software-eng
  git | Git and version control
    commits | Commit messages and history
    branching | Branches, merges, rebases
  repo_management | Repository management (issues, PRs, releases)
  code_review | Code review
  testing | Testing and test design
  debugging | Debugging and tracebacks
  ci | Continuous integration
  packaging | Packaging and dependency management
  build_systems | Build systems and compilation
  documentation | Technical documentation
  refactoring | Refactoring and code design
  api_design | API design
  concurrency | Concurrency and parallel programming
  performance | Profiling and performance engineering
  security | Software security | ddc=005.8;usenet=comp.security.misc
  licensing | Software licences
  devops | DevOps
    containers | Containers
    deployment | Deployment and services (launchd, systemd)
  os | Operating systems in practice | usenet=comp.os.*
    unix_tools | Unix command-line tools
    macos | macOS
    linux | Linux
  databases | Databases in practice
  networking | Networking and HTTP
  web_dev | Web development
  cli_design | Command-line interface design
ml | Machine learning and AI | arxiv=cs.LG;acm=Computing methodologies;ddc=006.3;usenet=comp.ai
  foundations | ML foundations (losses, optimisation, generalisation) | arxiv=stat.ML
  deep_learning | Deep learning architectures
    transformers | Transformers
    attention | Attention mechanisms
    moe | Mixture of experts
    linear_attention | Linear attention and state-space models (GDN, Mamba)
    diffusion | Diffusion models
    vision | Vision models | arxiv=cs.CV
    speech | Speech and audio models | arxiv=eess.AS
    embeddings | Embeddings and retrieval | arxiv=cs.IR
  nlp | Natural language processing (pre-LLM and general) | arxiv=cs.CL;since=1.1
  llm | Large language models
    pretraining | Pretraining
    finetuning | Fine-tuning (SFT, LoRA)
    alignment | RLHF and alignment
    prompting | Prompting and chat templates
    tokenization | Tokenisation
    evaluation | Evaluation and benchmarks
    reasoning | Reasoning and chain of thought
    tool_use | Tool use and function calling
    agents | Agents and harnesses
  interpretability | Interpretability
    neurons | Neuron and feature analysis
    pruning | Pruning, masking and model surgery
    probing | Probing and representation analysis
    sae | Sparse autoencoders
  inference | LLM inference engineering
    kv_cache | KV cache
      paging | Paged KV cache and block tables | since=1.4
      kv_quantization | KV cache quantisation | since=1.4
      eviction | KV eviction and compression (H2O, SnapKV, sliding windows) | since=1.4
      offload | KV offload and tiered caches (CPU, disk, LMCache) | since=1.4
    prefix_cache | Prefix caching (APC)
    quantization | Quantisation
      affine | Affine / group quantisation (q4, q8)
      microscaling | Microscaling formats (MXFP4/8, NVFP4)
      trellis | Trellis / codebook quantisation (EXL3, QTIP)
      gguf | GGUF / llama.cpp formats
      awq | Activation-aware weight quantisation (AWQ) | since=1.4
      gptq | GPTQ and second-order post-training quantisation | since=1.4
      fp8 | FP8 weights and activations | since=1.4
      weight_activation | Weight-activation quantisation (W8A8, W4A8, SmoothQuant) | since=1.4
    kernels | Inference kernels
      matmul | Matmul / GEMV kernels
      attention_kernels | Attention kernels (flash, paged, SDPA)
      metal_kernels | Metal kernels
      cuda_kernels | CUDA kernels
      triton_kernels | Triton kernels | since=1.4;see=prog.gpu.triton
      moe_kernels | MoE kernels (grouped GEMM, routing, gather) | since=1.4
      fused | Kernel fusion | since=1.4
    speculative | Speculative decoding
      draft_model | Draft-model speculation
      mtp | Multi-token prediction heads
      tree | Tree speculation
      prompt_lookup | Prompt lookup decoding
      eagle | Feature-level drafters (EAGLE, Medusa heads) | since=1.4
      verification | Verification and acceptance (rejection sampling, acceptance length) | since=1.4
    batching | Batching and scheduling (continuous batching)
      chunked_prefill | Chunked prefill | since=1.4
      disaggregation | Prefill/decode disaggregation | since=1.4
      preemption | Preemption, swapping and fairness | since=1.4
    serving | Serving and APIs
      openai_api | OpenAI-compatible API
      streaming | Streaming responses
      multi_lora | Multi-LoRA serving | since=1.4
      structured_output | Structured output and constrained decoding | since=1.4
    sampling | Sampling (temperature, top-k, top-p)
    long_context | Long context (RoPE scaling, sparse attention)
    memory | Memory management (unified memory, footprint, swap)
    distributed | Distributed inference (tensor/pipeline parallel)
    conversion | Model conversion and checkpoints
    benchmarking | Inference benchmarking
    prefill | Prefill
    decode | Decode
    compilation | Graph capture and compilation (CUDA graphs, torch.compile) | since=1.4
    moe_serving | MoE serving (expert parallelism, expert offload) | since=1.4
    multimodal | Multimodal inference (vision/audio encoders in serving) | since=1.4
  training | LLM training engineering | since=1.4
    parallelism | Training parallelism (data, tensor, pipeline, FSDP, ZeRO) | since=1.4
    mixed_precision | Mixed precision and low-precision training | since=1.4
    optimizers | Optimisers and schedules (Adam, Muon) | since=1.4
    checkpointing | Checkpointing and activation recomputation | since=1.4
    data_pipelines | Training data pipelines | since=1.4
  frameworks | ML frameworks
    mlx | MLX
    mlx2 | mlx2
    pytorch | PyTorch
    jax | JAX
    huggingface | Hugging Face transformers
    llama_cpp | llama.cpp
    vllm | vLLM / SGLang
  hardware | ML hardware
    apple_silicon | Apple silicon GPU
    nvidia | NVIDIA GPUs
    ane | Apple Neural Engine
    amd | AMD GPUs (ROCm, HIP) | since=1.4
cs | Computer science (theory and systems) | cip=11;acm=Theory of computation;ddc=004;usenet=comp.theory
  algorithms | Algorithms and data structures | arxiv=cs.DS
  information_theory | Information theory | arxiv=cs.IT;since=1.1
  robotics | Robotics | arxiv=cs.RO;since=1.1
  complexity | Computational complexity | arxiv=cs.CC
  cryptography | Cryptography | arxiv=cs.CR;usenet=sci.crypt
  pl_theory | Programming-language theory | arxiv=cs.PL
  formal_methods | Formal methods and proof assistants | arxiv=cs.LO
  systems | Operating systems and architecture | arxiv=cs.OS
  networks | Computer networks | arxiv=cs.NI
  databases | Database theory | arxiv=cs.DB
  graphics | Computer graphics | arxiv=cs.GR
  hci | Human-computer interaction | arxiv=cs.HC
  distributed | Distributed computing | arxiv=cs.DC
  game_theory | Algorithmic game theory | arxiv=cs.GT
math | Mathematics | cip=27;arxiv=math;os_field=Mathematics;ddc=510;usenet=sci.math
  arithmetic | Elementary arithmetic | ddc=513;usenet=sci.math
    addition | Addition | aka=arithmetic
    multiplication | Multiplication
  word_problems | Word problems (grade school) | aka=gsm8k
  number_theory | Number theory | msc=11;arxiv=math.NT
  algebra | Algebra | msc=08;arxiv=math.RA
    linear_algebra | Linear algebra | msc=15
    group_theory | Group theory | msc=20;arxiv=math.GR
    ring_theory | Rings, fields and commutative algebra | msc=13;arxiv=math.AC
    representation_theory | Representation theory | arxiv=math.RT
    quantum_algebra | Quantum algebra | arxiv=math.QA;since=1.1
    category_theory | Category theory | msc=18;arxiv=math.CT
    operator_algebras | Operator algebras | msc=46L;arxiv=math.OA
  geometry | Geometry
    algebraic_geometry | Algebraic and complex geometry | msc=14;arxiv=math.AG
    symplectic | Symplectic geometry | msc=53D;arxiv=math.SG;since=1.1
    differential_geometry | Differential geometry | msc=53;arxiv=math.DG
    metric_geometry | Convex and metric geometry | msc=52;arxiv=math.MG
    euclidean | Euclidean geometry | msc=51
  topology | Topology | msc=54;arxiv=math.GT
    algebraic_topology | Algebraic topology | msc=55;arxiv=math.AT
  analysis | Analysis
    real_complex | Real and complex analysis | msc=26;arxiv=math.CA
    functional_analysis | Functional analysis | msc=46;arxiv=math.FA
    pde | Partial differential equations | msc=35;arxiv=math.AP
    ode | Ordinary differential equations | msc=34
    calculus | Calculus
    harmonic | Harmonic analysis | msc=42
  dynamical_systems | Dynamical systems and ergodic theory | msc=37;arxiv=math.DS
  combinatorics | Combinatorics | msc=05;arxiv=math.CO
  logic | Mathematical logic and foundations | msc=03;arxiv=math.LO;ddc=511.3;usenet=sci.logic;see=hum.philosophy.logic
  probability | Probability and statistical mechanics | msc=60;arxiv=math.PR;ddc=519.2
  statistics | Statistics | msc=62;arxiv=math.ST;ddc=519.5;usenet=sci.stat.math
  numerical_analysis | Numerical analysis | msc=65;arxiv=math.NA
  optimization | Optimisation and control | msc=90;arxiv=math.OC
  mathematical_physics | Mathematical physics | msc=81;arxiv=math-ph
  theoretical_cs | Theoretical computer science (as mathematics) | msc=68
sci | Natural sciences | cip=40;ddc=500;usenet=sci.*
  physics | Physics | cip=40.08;arxiv=physics;os_field=Physics;ddc=530;usenet=sci.physics
    classical_mechanics | Classical mechanics | arxiv=physics.class-ph
    electromagnetism | Electromagnetism
    thermodynamics | Thermodynamics and statistical physics | arxiv=cond-mat.stat-mech
    quantum | Quantum mechanics | arxiv=quant-ph
    relativity | Relativity and gravitation | arxiv=gr-qc
    particle | Particle physics | arxiv=hep-ph
    nuclear | Nuclear physics | arxiv=nucl-th
    condensed_matter | Condensed matter | arxiv=cond-mat
    optics | Optics | arxiv=physics.optics
    fluids | Fluid dynamics | arxiv=physics.flu-dyn
    units | Physical units and measurement
    atomic | Atomic and molecular physics | arxiv=physics.atom-ph;since=1.1
    plasma | Plasma physics | arxiv=physics.plasm-ph;since=1.1
    high_energy_theory | High-energy theory (fields, strings) | arxiv=hep-th;since=1.1
    nonlinear | Nonlinear dynamics and chaos | arxiv=nlin;since=1.1
  astronomy | Astronomy and astrophysics | cip=40.02;arxiv=astro-ph;ddc=520;usenet=sci.astro
    galaxies | Galaxies | arxiv=astro-ph.GA;since=1.1
    cosmology | Cosmology | arxiv=astro-ph.CO;since=1.1
    stellar | Solar and stellar astrophysics | arxiv=astro-ph.SR;since=1.1
    high_energy | High-energy astrophysics | arxiv=astro-ph.HE;since=1.1
    planetary | Earth and planetary astrophysics | arxiv=astro-ph.EP;since=1.1
    instrumentation | Astronomical instrumentation | arxiv=astro-ph.IM;since=1.1
  chemistry | Chemistry | cip=40.05;os_field=Chemistry;ddc=540;usenet=sci.chem
    organic | Organic chemistry
    inorganic | Inorganic chemistry
    physical | Physical chemistry
    analytical | Analytical chemistry
    biochemistry | Biochemistry | cip=26.02
  biology | Biology | cip=26;arxiv=q-bio;os_field=Biology;ddc=570;usenet=sci.bio.*
    molecular | Molecular biology | arxiv=q-bio.BM
    cell | Cell biology | arxiv=q-bio.CB
    genetics | Genetics and genomics | arxiv=q-bio.GN
    evolution | Evolution | arxiv=q-bio.PE
    ecology | Ecology
    microbiology | Microbiology
    neuroscience | Neuroscience | arxiv=q-bio.NC
    immunology | Immunology
    botany | Botany
    zoology | Zoology
  medicine | Medicine and health | cip=51;os_field=Medicine;ddc=610;usenet=sci.med
    clinical | Clinical medicine and trials
    pharmacology | Pharmacology
    epidemiology | Epidemiology and public health
    oncology | Oncology
    cardiology | Cardiology
    infectious_disease | Infectious disease
    nursing | Nursing | cip=51.38;os_field=Nursing
  earth | Earth sciences | cip=40.06;ddc=550;usenet=sci.geo.*
    geology | Geology
    meteorology | Meteorology and climate | cip=40.04
    oceanography | Oceanography
  materials | Materials science | cip=40.10
eng | Engineering | cip=14;os_field=Engineering;ddc=620;usenet=sci.engr
  electrical | Electrical and electronic engineering | cip=14.10;arxiv=eess;ddc=621.3;usenet=sci.electronics.*
  mechanical | Mechanical engineering | cip=14.19
  civil | Civil engineering | cip=14.08
  chemical | Chemical engineering | cip=14.07
  aerospace | Aerospace engineering | cip=14.02
  biomedical | Biomedical engineering | cip=14.05
  industrial | Industrial engineering | cip=14.35
  signal_processing | Signal processing | arxiv=eess.SP
hum | Humanities | cip=24
  history | History | cip=54;os_field=History;ddc=900;usenet=soc.history
    ancient | Ancient history
    medieval | Medieval history
    modern | Modern history
    military | Military history
  philosophy | Philosophy | cip=38.01;os_field=Philosophy;ddc=100;usenet=talk.philosophy.misc
    ethics | Ethics | ddc=170
    epistemology | Epistemology | ddc=121
    logic | Philosophical logic and reasoning | ddc=160;usenet=sci.logic;see=math.logic;since=1.3
      argumentation | Argumentation and persuasion | ddc=168;usenet=talk.philosophy.misc;since=1.3
    metaphysics | Metaphysics | ddc=110
    political | Political philosophy
  religion | Religion and theology | cip=38.02;os_field=Religion;ddc=200;usenet=soc.religion.*
  literature | Literature | cip=23;os_field=English Literature;ddc=800;usenet=rec.arts.books
    fiction | Fiction
    poetry | Poetry | ddc=808.1;usenet=rec.arts.poems
    drama | Drama
    criticism | Literary criticism
  linguistics | Linguistics | cip=16.01;arxiv=cs.CL;ddc=410;usenet=sci.lang
  art_history | Art and art history | cip=50.07;os_field=Fine Arts;ddc=700
  music | Music | cip=50.09;os_field=Music;ddc=780;usenet=rec.music.*
  film | Film and media | cip=50.06;os_field=Film and Photography;ddc=791.43;usenet=rec.arts.movies.*
  classics | Classics | cip=16.12;os_field=Classics
soc | Social sciences | cip=45;ddc=300;usenet=soc.*
  economics | Economics | cip=45.06;arxiv=econ;os_field=Economics;ddc=330;usenet=sci.econ
    finance | Finance | cip=52.08;arxiv=q-fin
    accounting | Accounting | cip=52.03;os_field=Accounting
  business | Business and management | cip=52;os_field=Business
    marketing | Marketing | cip=52.14;os_field=Marketing
  psychology | Psychology | cip=42;os_field=Psychology;ddc=150;usenet=sci.psychology.*
  sociology | Sociology | cip=45.11;os_field=Sociology
  anthropology | Anthropology | cip=45.02;os_field=Anthropology
  political_science | Political science | cip=45.10;os_field=Political Science;ddc=320;usenet=talk.politics.*
  law | Law | cip=22;os_field=Law;ddc=340;usenet=misc.legal
    us_legislation | US legislation (bills, statutes) | aka=billsum
    constitutional | Constitutional law
    contracts | Contract law
    criminal | Criminal law
  education | Education | cip=13;os_field=Education;ddc=370;usenet=misc.education
  geography | Geography | cip=45.07;os_field=Geography;ddc=910
    world_capitals | World capitals | aka=capitals,capitals_fresh,capitals_phrasing
    countries | Countries and places
  communication | Communication and journalism | cip=09;os_field=Journalism;ddc=070;usenet=alt.journalism
life | Everyday life and practical knowledge
  cooking | Cooking and food | ddc=641.5;usenet=rec.food.cooking
  sports | Sports | ddc=796;usenet=rec.sport.*
  travel | Travel | ddc=910.4;usenet=rec.travel.*
  health_wellness | Health and wellness
  personal_finance | Personal finance
  home_diy | Home and DIY
  games | Games | ddc=793;usenet=rec.games.*
form | Text form and genre (not subject matter)
  narrative | Narrative
    childrens_story | Children's stories | aka=tinystories_train;usenet=rec.arts.books.childrens
    story_opening | Story openings
  dialogue | Dialogue and chat turns
    socratic | Socratic dialogue | ddc=183;usenet=talk.philosophy.misc;see=hum.philosophy.logic;since=1.3
  debate | Debate | ddc=808.53;usenet=talk.*;see=hum.philosophy.logic.argumentation;since=1.3
  encyclopedic | Encyclopedic and expository reference prose | ddc=030;since=1.3
  textbook | Textbook exposition | ddc=371.32;usenet=misc.education;see=soc.education;since=1.3
  audience | Intended audience and reading level (register, not subject) | since=1.3
    young_children | Young children (pre-school to age 7) | ddc=372.21;since=1.3
    children | Children (primary school) | ddc=372;usenet=k12.*;since=1.3
    middle_school | Middle school | ddc=373;usenet=k12.*;since=1.3
    high_school | High school | ddc=373;usenet=k12.*;since=1.3
    college | College and general adult | ddc=378;since=1.3
    researchers | Researchers and specialists | since=1.3
  instructions | Instructions and how-tos
  qa | Question and answer
  list | Lists and enumerations
  markdown_structure | Markdown structure
    headings | Headings and bold
    tables | Tables (pipes)
    inline_code | Inline code and fences
    blockquote | Block quotes
  digits | Digits and numerals
  subword | Sub-word and orthographic pieces
  legal_register | Legal register | ddc=340
  academic_register | Academic register
skill | In-context skills
  in_context_recall | In-context recall | aka=recall
  translation | Translation
  arithmetic_reasoning | Multi-step arithmetic reasoning
  copying | Copying and repetition
  deduction | Deductive reasoning and logic puzzles | ddc=160;see=hum.philosophy.logic;since=1.3
  entailment | Textual entailment and inference | see=hum.philosophy.logic;since=1.3
sec | Cybersecurity | cip=11.10;acm=Security and privacy;since=1.2;ddc=005.8;usenet=comp.security.*
  csf | NIST Cybersecurity Framework 2.0 functions | csf=CSF2.0;since=1.2
    govern | Govern | csf=GV;since=1.2
      context | Organizational context | csf=GV.OC;since=1.2
      risk_strategy | Risk management strategy | csf=GV.RM;since=1.2
      roles | Roles, responsibilities and authorities | csf=GV.RR;since=1.2
      policy | Security policy | csf=GV.PO;since=1.2
      oversight | Oversight | csf=GV.OV;since=1.2
      supply_chain | Cybersecurity supply-chain risk management | csf=GV.SC;since=1.2
    identify | Identify | csf=ID;since=1.2
      asset_management | Asset management | csf=ID.AM;since=1.2
      risk_assessment | Risk assessment | csf=ID.RA;since=1.2
      improvement | Improvement | csf=ID.IM;since=1.2
    protect | Protect | csf=PR;since=1.2
      access_control | Identity management, authentication and access control | csf=PR.AA;since=1.2
      awareness_training | Awareness and training | csf=PR.AT;since=1.2
      data_security | Data security | csf=PR.DS;since=1.2
      platform_security | Platform security | csf=PR.PS;since=1.2
      infrastructure_resilience | Technology infrastructure resilience | csf=PR.IR;since=1.2
    detect | Detect | csf=DE;since=1.2
      continuous_monitoring | Continuous monitoring | csf=DE.CM;since=1.2
      adverse_event_analysis | Adverse event analysis | csf=DE.AE;since=1.2
    respond | Respond | csf=RS;since=1.2
      incident_management | Incident management | csf=RS.MA;since=1.2
      incident_analysis | Incident analysis | csf=RS.AN;since=1.2
      reporting | Incident response reporting and communication | csf=RS.CO;since=1.2
      mitigation | Incident mitigation | csf=RS.MI;since=1.2
    recover | Recover | csf=RC;since=1.2
      recovery_plan | Incident recovery plan execution | csf=RC.RP;since=1.2
      recovery_communication | Incident recovery communication | csf=RC.CO;since=1.2
  cti | Cyber threat intelligence | since=1.2
    attack_techniques | MITRE ATT&CK tactics and techniques | since=1.2
    threat_actors | Threat-actor profiling | since=1.2
    indicators | Indicators of compromise (IOCs) | since=1.2
    stix_taxii | STIX / TAXII | since=1.2
    tlp | Traffic Light Protocol and sharing | since=1.2
    lifecycle | Intelligence lifecycle and requirements | since=1.2
    diamond_model | Diamond model and kill chain analysis | since=1.2
    report_writing | CTI report writing and dissemination | since=1.5
    analytic_tradecraft | Analytic tradecraft (structured analytic techniques, estimative language) | see=intel;since=1.5
    threat_actor_comms | Threat-actor communications (forums, leaks, chat) | since=1.5
  vuln | Vulnerability management and patching | csf=ID.RA;since=1.2
    cve | CVE records and advisories | since=1.2
    cwe | CWE weakness classes | since=1.2
    scoring | CVSS / EPSS scoring | since=1.2
    kev | Known exploited vulnerabilities (CISA KEV) | since=1.2
    patching | Patch management and cadence | csf=PR.PS;since=1.2
    sbom | SBOM and dependency inventory | csf=ID.AM;since=1.2
  offensive | Offensive security | since=1.2
    red_team | Red teaming and adversary emulation | since=1.2
    pentest | Penetration testing methodology | since=1.2
  defensive | Blue team and detection | csf=DE;since=1.2
    soc | Security operations (SOC) | since=1.2
    detection_engineering | Detection engineering (Sigma, YARA) | since=1.2
    siem | SIEM and log analysis | csf=DE.CM;since=1.2
    threat_hunting | Threat hunting | since=1.5
  dfir | Digital forensics and incident response | csf=RS.AN;since=1.2
    forensics | Digital forensics | since=1.2
    malware_analysis | Malware analysis (descriptive) | since=1.2
    logs | Log and event forensics | see=sec.defensive.siem;since=1.5
  grc | Governance, risk and compliance | csf=GV;since=1.2
    iso27001 | ISO/IEC 27001 | since=1.2
    soc2 | SOC 2 | since=1.2
    nist_800_53 | NIST SP 800-53 controls | since=1.2
    pci_dss | PCI DSS | since=1.2
    privacy | Privacy and data protection (GDPR) | since=1.2
    risk_register | Risk registers | since=1.2
    audit | Security audit | since=1.2
  domain | Security domains | since=1.2
    appsec | Application security and secure SDLC | see=swe.security;since=1.2
    cloud | Cloud security | since=1.2
    network | Network security | see=swe.networking;since=1.2
    iam | Identity and access management | csf=PR.AA;since=1.2
    crypto | Applied cryptography | see=cs.cryptography;since=1.2
    ot_ics | OT / ICS security | since=1.2
    email_phishing | Email security and phishing | since=1.2
    supply_chain | Software supply-chain security | see=swe.packaging;since=1.2
    ml_security | ML and LLM security (adversarial ML, prompt injection) | see=ml.llm.alignment;since=1.5
  re | Reverse engineering | see=sec.dfir.malware_analysis;since=1.5
    binary | Binary reverse engineering (disassembly, decompilation) | see=prog.assembly;since=1.5
    firmware | Firmware and embedded reverse engineering | since=1.5
    mobile | Mobile app reverse engineering (Android, iOS) | since=1.5
  exploit | Exploit development | see=sec.offensive;since=1.5
    memory_corruption | Memory-corruption exploitation (overflows, UAF, ROP) | since=1.5
    kernel | Kernel exploitation | see=swe.os;since=1.5
  ctf | Capture-the-flag writeups | since=1.5
agent | Model-harness interaction: reasoning traces, tool calls, agent loops (the text itself, not writing about it) | see=ml.llm.agents;since=1.5
  reasoning | Reasoning traces and chain of thought | see=ml.llm.reasoning;since=1.5
    cot | Worked step-by-step solutions (visible chain of thought) | see=skill.arithmetic_reasoning;since=1.5
    thinking_trace | Long thinking traces (think blocks, exploration, backtracking) | since=1.5
    self_correction | Self-correction and verification moves | since=1.5
  tool_use | Tool use and function calling | see=ml.llm.tool_use;since=1.5
    definitions | Tool and function schemas (JSON Schema, OpenAPI, MCP tool lists) | since=1.5
    calls | Tool-call emission formats (JSON calls, tool_call tags, parallel calls) | since=1.5
    results | Tool results and observations returned to the model | since=1.5
    error_recovery | Tool errors, retries and recovery | since=1.5
    decision | Deciding whether to call a tool, answer directly or ask | since=1.5
  harness | Agent harness dynamics | see=ml.llm.agents;since=1.5
    loop | Agent loop turns (thought, action, observation) | since=1.5
    shell | Shell transcripts (commands and their output) | see=prog.shell;since=1.5
    file_edits | File edits, diffs and patches | see=swe.git;since=1.5
    planning | Plans, task lists and decomposition | since=1.5
    status | Progress and status updates, final reports | since=1.5
dialog | Model-user dialogue dynamics | see=form.dialogue;since=1.5
  multiturn | Multi-turn chat dynamics (follow-ups, topic shifts, corrections) | since=1.5
  refusal | Refusals, safe completions and boundary-setting | see=ml.llm.alignment;since=1.5
  clarification | Clarifying questions and ambiguity handling | since=1.5
  system_prompt | System prompts, personas and operator instructions | see=ml.llm.prompting;since=1.5
  instruction_following | Format and constraint instructions (length, style, output format) | since=1.5
intel | Intelligence studies (non-cyber) | see=sec.cti;since=1.5
  doctrine | Intelligence doctrine and the intelligence cycle | since=1.5
  osint | Open-source intelligence | since=1.5
  counterintelligence | Counterintelligence | since=1.5
news | News journalism (general register, any subject) | see=soc.communication;since=1.5
  general | General news reporting | since=1.5
lang | Natural languages (ISO 639-3) | cip=16;ddc=400
"""

# Labels used by earlier amoeba runs (task names, atlas corpus sources) -> topic IDs. A label may map to
# several topics; the first is primary.
LEGACY = {"capitals": ["soc.geography.world_capitals"], "capitals_fresh": ["soc.geography.world_capitals"],
          "capitals_phrasing": ["soc.geography.world_capitals"], "arithmetic": ["math.arithmetic.addition"],
          "french": ["lang.fra", "skill.translation"], "recall": ["skill.in_context_recall"],
          "english": ["lang.eng"], "tinystories_train": ["form.narrative.childrens_story", "lang.eng"],
          "repo_markdown": ["prog.markup.markdown"], "donor_generated": ["lang.eng"]}

ISO_TYPE = {"L": "living", "E": "extinct", "H": "historical", "C": "constructed", "A": "ancient", "S": "special"}
ISO_SCOPE = {"I": "individual", "M": "macrolanguage", "S": "special"}
NODE_DEFAULTS = {"aliases": [], "refs": {}, "since": "1.0", "status": "active", "replaced_by": None}
SEG = re.compile(r"^[a-z0-9_]+$")


def parse_outline(text):
    nodes, stack = [], []
    for raw in text.splitlines():
        if not raw.strip():
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        if indent % 2:
            raise SystemExit(f"odd indent: {raw!r}")
        depth = indent // 2
        parts = [p.strip() for p in raw.strip().split("|")]
        seg, name = parts[0], parts[1]
        refs = {}
        if len(parts) > 2 and parts[2]:
            for kv in parts[2].split(";"):
                k, v = kv.split("=", 1)
                refs[k.strip()] = v.strip()
        if not SEG.match(seg):
            raise SystemExit(f"bad segment {seg!r}")
        stack = stack[:depth]
        if len(stack) != depth:
            raise SystemExit(f"indent jump at {raw!r}")
        parent = ".".join(stack) if stack else None
        stack.append(seg)
        aliases = [a for a in refs.pop("aka", "").split(",") if a]
        since = refs.pop("since", "1.0")
        nodes.append({"id": ".".join(stack), "name": name, "parent": parent, "depth": depth,
                      "aliases": aliases, "refs": refs, "since": since})
    return nodes


# Dewey 420-490 class numbers and Usenet soc.culture / national hierarchies for major languages (since 1.3)
LANG_REFS = {
    "eng": {"ddc": "420"}, "deu": {"ddc": "430", "usenet": "soc.culture.german,de.*"},
    "fra": {"ddc": "440", "usenet": "soc.culture.french,fr.*"}, "ita": {"ddc": "450", "usenet": "soc.culture.italian,it.*"},
    "spa": {"ddc": "460", "usenet": "es.*"}, "por": {"ddc": "469"}, "lat": {"ddc": "470"}, "grc": {"ddc": "480"},
    "rus": {"ddc": "491.7", "usenet": "soc.culture.russian"}, "zho": {"ddc": "495.1", "usenet": "soc.culture.china"},
    "jpn": {"ddc": "495.6", "usenet": "soc.culture.japan,fj.*"}, "kor": {"ddc": "495.7", "usenet": "soc.culture.korean"},
    "ara": {"ddc": "492.7"},
}


def iso_nodes(path):
    rows = json.loads(Path(path).read_text())["639-3"]
    out = []
    macro_names = {}
    for r in rows:
        if r.get("scope") == "S":
            continue
        code = r["alpha_3"]
        ref = {"iso639_3": code}
        if r.get("alpha_2"):
            ref["iso639_1"] = r["alpha_2"]
        ref.update(LANG_REFS.get(code, {}))
        out.append({"id": f"lang.{code}", "name": r["name"],
                    "parent": "lang", "depth": 1, "aliases": [], "refs": ref,
                    "lang_type": ISO_TYPE.get(r.get("type"), r.get("type")),
                    "lang_scope": ISO_SCOPE.get(r.get("scope"), r.get("scope"))})
        if r.get("scope") == "M":
            macro_names[code] = r["name"]
    for n in out:
        if n["id"] == "lang.eng":
            n["aliases"] = ["english"]
        if n["id"] == "lang.fra":
            n["aliases"] = ["french"]
    return out, len(rows)


def build(iso_path):
    nodes = parse_outline(OUTLINE)
    langs, n_iso = iso_nodes(iso_path)
    nodes += langs
    ids = [n["id"] for n in nodes]
    dup = {i for i in ids if ids.count(i) > 1}
    if dup:
        raise SystemExit(f"duplicate ids: {sorted(dup)[:10]}")
    kids = {}
    for n in nodes:
        if n["parent"]:
            kids.setdefault(n["parent"], []).append(n["id"])
    for n in nodes:
        n["kind"] = "branch" if n["id"] in kids else "leaf"
        n.setdefault("since", "1.0")
        n["status"] = "active"
        n["replaced_by"] = None
    alias_map = {}
    for n in nodes:
        for a in n["aliases"]:
            if a in alias_map:
                raise SystemExit(f"alias {a} on {alias_map[a]} and {n['id']}")
            alias_map[a] = n["id"]
    return nodes, alias_map, n_iso


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--iso", default=str(ISO_DEFAULT))
    ap.add_argument("--check", action="store_true", help="verify the file on disk is what the builder makes")
    args = ap.parse_args()
    out = Path(args.out)
    nodes, alias_map, n_iso = build(args.iso)
    if out.exists():
        prev = json.loads(out.read_text())
        prev_ids = {n["id"] for n in prev["nodes"]}
        now_ids = {n["id"] for n in nodes}
        gone = sorted(prev_ids - now_ids)
        if gone:
            raise SystemExit(f"refusing: {len(gone)} existing topic IDs would disappear (deprecate instead): {gone[:10]}")
    iso_sha = hashlib.sha256(Path(args.iso).read_bytes()).hexdigest()
    doc = {
        "schema": SCHEMA, "version": VERSION,
        "rules": {"ids": "dotted [a-z0-9_] segments; permanent, never renumbered or reused",
                  "add": "new node -> minor version bump", "remove": "never; set status=deprecated and replaced_by",
                  "move": "deprecate + new id"},
        "sources": {
            "cip": "US Dept. of Education CIP 2020 (public domain); Open Syllabus fields derive from it",
            "open_syllabus_galaxy": "checked 2026-11-14: no public field list, download or API; not scraped",
            "iso639_3": {"file": "pycountry 26.2.16 databases/iso639-3.json (Debian iso-codes, LGPL-2.1)",
                         "sha256": iso_sha, "codes": n_iso},
            "linguist": "GitHub Linguist language names (MIT), names only",
            "msc": "MSC 2020 top-level codes cited (CC BY-NC-SA 4.0)",
            "arxiv": "arXiv category taxonomy codes",
            "csf": "NIST Cybersecurity Framework 2.0 function/category codes (NIST CSWP 29, US government work)",
            "ddc": "Dewey Decimal Classification class numbers only (DDC is OCLC's; no schedule captions copied); since 1.3",
            "usenet": "Usenet Big-8 / k12 / alt newsgroup and hierarchy names; since 1.3",
            "see": "refs.see cross-links a node to an overlapping node in another branch (no inheritance)", "acm": "ACM CCS 2012 top-level concepts"},
        "node_defaults": NODE_DEFAULTS,
        "alias_map": dict(sorted(alias_map.items())),
        "legacy_task_labels": LEGACY,
        "counts": {"nodes": len(nodes), "leaves": sum(n["kind"] == "leaf" for n in nodes),
                   "lang": sum(n["id"].startswith("lang.") for n in nodes),
                   "non_lang": sum(not n["id"].startswith("lang") for n in nodes)},
        "nodes": nodes,
    }
    def slim(n):   # omit fields equal to the declared defaults (doc["node_defaults"])
        return {k: v for k, v in n.items() if not (k in NODE_DEFAULTS and v == NODE_DEFAULTS[k])}
    nodes_txt = ",\n".join(json.dumps(slim(n), ensure_ascii=False, sort_keys=True) for n in doc.pop("nodes"))
    head = json.dumps(doc, indent=1, ensure_ascii=False)
    text = head[:-2] + ',\n "nodes": [\n' + nodes_txt + "\n]\n}\n"
    if args.check:
        same = out.exists() and out.read_text() == text
        print("taxonomy up to date" if same else "taxonomy differs from builder output")
        sys.exit(0 if same else 1)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    print(f"wrote {out}: {doc['counts']}")


if __name__ == "__main__":
    main()
