"""Source-label -> aMoEba taxonomy maps for the 2026-11-18 category corpora (Pierre 2026-11-18: flesh out every
probed category from public datasets; survey in amoeba-corpus surveys/category-sources-20261118.md).

A source label (StackExchange site + tags, arXiv category, LibreTexts shelf, peS2o field of study) only nominates
candidate categories. Final assignment and filtering is by per-category term log-odds (amoeba_category_mine.py),
so these maps can be generous. Medical labels are dropped outright outside sci.medicine (box spec).
"""
from __future__ import annotations

import re

# No date cut by default: these are probe corpora, not training data (Pierre 2026-11-18). --created-before sets one.
CUTOFF = "9999-12-31"

# StackExchange site -> default category. Sites already used by subject-boxes-99a59a894 keep their mapping.
SE_SITE = {
    # math / stats / theory
    "math": "math", "mathoverflow": "math", "stats": "math.statistics", "scicomp": "math.numerical_analysis",
    "or": "math.optimization", "cstheory": "math.theoretical_cs", "cs": "cs", "hsm": "hum.history",
    "matheducators": "soc.education", "cseducators": "soc.education", "proofassistants": "cs.formal_methods",
    # physics / engineering
    "physics": "sci.physics", "quantumcomputing": "sci.physics.quantum", "electronics": "eng.electrical",
    "ham": "eng.electrical", "arduino": "eng.electrical", "dsp": "eng.signal_processing",
    "engineering": "eng.mechanical", "mechanics": "eng.mechanical", "3dprinting": "eng.mechanical",
    "materials": "sci.materials", "robotics": "cs.robotics", "aviation": "eng.aerospace", "space": "eng.aerospace",
    # software
    "softwareengineering": "swe", "codereview": "swe.code_review", "unix": "swe.os.unix_tools",
    "superuser": "swe.os", "askubuntu": "swe.os.linux", "serverfault": "swe.devops", "devops": "swe.devops",
    "dba": "swe.databases", "networkengineering": "cs.networks", "computergraphics": "cs.graphics",
    "gamedev": "cs.graphics", "tex": "prog.markup.latex", "stackoverflow": "prog",
    # security
    "security": "sec", "crypto": "cs.cryptography", "reverseengineering": "sec.dfir.malware_analysis",
    # humanities / social / life
    "history": "hum.history", "genealogy": "hum.history", "movies": "hum.film", "anime": "hum.film",
    "scifi": "hum.film", "christianity": "hum.religion", "judaism": "hum.religion", "islam": "hum.religion",
    "hinduism": "hum.religion", "buddhism": "hum.religion", "hermeneutics": "hum.religion",
    "mythology": "hum.religion", "economics": "soc.economics", "quant": "soc.economics.finance",
    "money": "life.personal_finance", "pm": "soc.business", "workplace": "soc.business",
    "freelancing": "soc.business", "startups": "soc.business", "academia": "soc.education",
    "interpersonal": "soc.communication", "sports": "life.sports", "bicycles": "life.sports",
    "martialarts": "life.sports", "outdoors": "life.sports", "travel": "life.travel", "expatriates": "life.travel",
    "diy": "life.home_diy", "woodworking": "life.home_diy", "gardening": "life.home_diy",
    "bioinformatics": "sci.biology.genetics", "gis": "soc.geography",
}
SE_SKIP = {"datascience", "ai", "mathematica", "parenting", "photo", "sustainability"}   # off-target / covered

# Tag rules: (regex over the '|'-joined tag string, category). The first match per rule set adds a candidate; all
# matching rules are added, and the site default stays a candidate for the parent box.
_T = [
    # math leaves (math.SE plain tags and MathOverflow arXiv-style tags)
    (r"\b(linear-algebra|matrices|eigenvalues-eigenvectors|vector-spaces|determinant|ra\.rings-and-algebras-linear)\b", "math.algebra.linear_algebra"),
    (r"\b(group-theory|finite-groups|gr\.group-theory|abelian-groups|sylow-theory|lie-groups)\b", "math.algebra.group_theory"),
    (r"\b(ring-theory|commutative-algebra|ac\.commutative-algebra|ra\.rings-and-algebras|ideals|field-theory|galois-theory|modules)\b", "math.algebra.ring_theory"),
    (r"\b(representation-theory|rt\.representation-theory|lie-algebras)\b", "math.algebra.representation_theory"),
    (r"\b(category-theory|ct\.category-theory|homological-algebra|higher-category-theory)\b", "math.algebra.category_theory"),
    (r"\b(operator-algebras|oa\.operator-algebras|c-star-algebras|von-neumann-algebras)\b", "math.algebra.operator_algebras"),
    (r"\b(quantum-groups|qa\.quantum-algebra|hopf-algebras)\b", "math.algebra.quantum_algebra"),
    (r"\b(abstract-algebra)\b", "math.algebra"),
    (r"\b(algebraic-geometry|ag\.algebraic-geometry|schemes|algebraic-curves|sheaf-theory)\b", "math.geometry.algebraic_geometry"),
    (r"\b(symplectic-geometry|sg\.symplectic-geometry)\b", "math.geometry.symplectic"),
    (r"\b(differential-geometry|dg\.differential-geometry|riemannian-geometry|manifolds|smooth-manifolds)\b", "math.geometry.differential_geometry"),
    (r"\b(metric-geometry|mg\.metric-geometry|convex-geometry|discrete-geometry)\b", "math.geometry.metric_geometry"),
    (r"\b(euclidean-geometry|triangles|circles|geometry|analytic-geometry|trigonometry|polygons)\b", "math.geometry.euclidean"),
    (r"\b(algebraic-topology|at\.algebraic-topology|homotopy-theory|homology-cohomology|fundamental-groups)\b", "math.topology.algebraic_topology"),
    (r"\b(general-topology|gn\.general-topology|gt\.geometric-topology|knot-theory|low-dimensional-topology)\b", "math.topology"),
    (r"\b(real-analysis|complex-analysis|cv\.complex-variables|ca\.classical-analysis-and-odes|measure-theory|lebesgue-integral|sequences-and-series)\b", "math.analysis.real_complex"),
    (r"\b(functional-analysis|fa\.functional-analysis|banach-spaces|hilbert-spaces|operator-theory|spectral-theory|sp\.spectral-theory)\b", "math.analysis.functional_analysis"),
    (r"\b(pde|partial-differential-equations|ap\.analysis-of-pdes|heat-equation|wave-equation)\b", "math.analysis.pde"),
    (r"\b(ordinary-differential-equations|ode|ca\.classical-analysis-and-odes)\b", "math.analysis.ode"),
    (r"\b(calculus|integration|derivatives|limits|multivariable-calculus|definite-integrals|improper-integrals)\b", "math.analysis.calculus"),
    (r"\b(harmonic-analysis|fourier-analysis|fourier-series|fourier-transform|wavelets)\b", "math.analysis.harmonic"),
    (r"\b(dynamical-systems|ds\.dynamical-systems|ergodic-theory|chaos-theory|bifurcation)\b", "math.dynamical_systems"),
    (r"\b(combinatorics|co\.combinatorics|graph-theory|enumerative-combinatorics|extremal-combinatorics|discrete-mathematics|ramsey-theory)\b", "math.combinatorics"),
    (r"\b(logic|lo\.logic|set-theory|model-theory|first-order-logic|propositional-calculus|proof-theory|computability|ordinals|cardinals)\b", "math.logic"),
    (r"\b(number-theory|nt\.number-theory|elementary-number-theory|prime-numbers|analytic-number-theory|algebraic-number-theory|modular-arithmetic|diophantine-equations|modular-forms)\b", "math.number_theory"),
    (r"\b(probability|pr\.probability|probability-theory|stochastic-processes|probability-distributions|random-variables|martingales|brownian-motion|markov-chains)\b", "math.probability"),
    (r"\b(statistics|st\.statistics|statistical-inference|hypothesis-testing|regression|bayesian|estimation)\b", "math.statistics"),
    (r"\b(numerical-methods|na\.numerical-analysis|numerical-linear-algebra|finite-element-method|numerical-integration)\b", "math.numerical_analysis"),
    (r"\b(optimization|oc\.optimization-and-control|convex-optimization|linear-programming|nonlinear-optimization|integer-programming|optimal-control|calculus-of-variations)\b", "math.optimization"),
    (r"\b(mathematical-physics|mp\.mathematical-physics)\b", "math.mathematical_physics"),
    (r"\b(computational-complexity|automata|formal-languages|turing-machines|computer-science|cc\.complexity-theory)\b", "math.theoretical_cs"),
    (r"\b(arithmetic)\b", "math.arithmetic"),
    # physics leaves (physics.SE)
    (r"\b(newtonian-mechanics|classical-mechanics|lagrangian-formalism|hamiltonian-formalism|rotational-dynamics|kinematics|forces|friction|momentum|projectile)\b", "sci.physics.classical_mechanics"),
    (r"\b(electromagnetism|electrostatics|magnetic-fields|electric-circuits|maxwell-equations|electromagnetic-radiation|electric-fields|induction)\b", "sci.physics.electromagnetism"),
    (r"\b(thermodynamics|statistical-mechanics|entropy|temperature|heat|heat-conduction|ideal-gas)\b", "sci.physics.thermodynamics"),
    (r"\b(quantum-mechanics|quantum-information|quantum-entanglement|wavefunction|schroedinger-equation|hilbert-space|quantum-spin|operators|quantum-computing|qiskit|quantum-gate|quantum-circuit)\b", "sci.physics.quantum"),
    (r"\b(special-relativity|general-relativity|spacetime|black-holes|metric-tensor|gravitational-waves|differential-geometry)\b", "sci.physics.relativity"),
    (r"\b(particle-physics|standard-model|quarks|neutrinos|higgs|particle-detectors|large-hadron-collider)\b", "sci.physics.particle"),
    (r"\b(nuclear-physics|radioactivity|fission|fusion|nuclear-engineering)\b", "sci.physics.nuclear"),
    (r"\b(condensed-matter|solid-state-physics|superconductivity|semiconductor-physics|crystals|band-structure|phonons)\b", "sci.physics.condensed_matter"),
    (r"\b(optics|lasers|refraction|reflection|diffraction|interference|lenses|polarization)\b", "sci.physics.optics"),
    (r"\b(fluid-dynamics|aerodynamics|fluid-statics|viscosity|turbulence|navier-stokes|bernoulli-equation)\b", "sci.physics.fluids"),
    (r"\b(units|dimensional-analysis|si-units|unit-conversion|metrology)\b", "sci.physics.units"),
    (r"\b(atomic-physics|atoms|spectroscopy|hydrogen|orbitals|electrons)\b", "sci.physics.atomic"),
    (r"\b(plasma-physics|plasma)\b", "sci.physics.plasma"),
    (r"\b(quantum-field-theory|string-theory|gauge-theory|renormalization|supersymmetry|feynman-diagrams|path-integral)\b", "sci.physics.high_energy_theory"),
    (r"\b(non-linear-systems|chaos-theory|solitons|nonlinear-dynamics)\b", "sci.physics.nonlinear"),
    # engineering.SE / mechanics.SE
    (r"\b(civil-engineering|structural-engineering|structural-analysis|concrete|steel|bridges|geotechnical-engineering|beam|construction|foundations)\b", "eng.civil"),
    (r"\b(chemical-engineering|process-engineering|distillation|reactor|heat-exchanger|chemical-process)\b", "eng.chemical"),
    (r"\b(electrical-engineering|power-electronics|power-engineering|motors|transformer|circuits)\b", "eng.electrical"),
    (r"\b(industrial-engineering|manufacturing-engineering|lean|six-sigma|quality-control|operations-research|supply-chain)\b", "eng.industrial"),
    (r"\b(biomedical-engineering|biomechanics|prosthetics)\b", "eng.biomedical"),
    (r"\b(mechanical-engineering|machine-design|gears|bearings|stress|fatigue|cad|solidworks|hydraulics|pneumatics|engine|transmission)\b", "eng.mechanical"),
    (r"\b(signal-processing|filters|fft|dsp)\b", "eng.signal_processing"),
    (r"\b(control-engineering|control-theory|pid-control)\b", "eng.electrical"),
    # history
    (r"\b(ancient-history|ancient-rome|ancient-greece|ancient-egypt|roman-empire|classical-antiquity|bronze-age|iron-age|byzantine-empire)\b", "hum.history.ancient"),
    (r"\b(medieval|middle-ages|crusades|feudalism|vikings|renaissance|early-modern|holy-roman-empire)\b", "hum.history.medieval"),
    (r"\b(20th-century|19th-century|world-war-one|world-war-two|cold-war|modern|industrial-revolution|soviet-union|nazi-germany)\b", "hum.history.modern"),
    (r"\b(military|warfare|navy|army|battles|weapons|military-history|tactics|strategy|air-force)\b", "hum.history.military"),
    # economics / business
    (r"\b(finance|financial-markets|options|derivatives|stocks|bonds|portfolio|asset-pricing|interest-rates|volatility|black-scholes|risk-management)\b", "soc.economics.finance"),
    (r"\b(accounting|tax|bookkeeping|gaap|financial-statements|depreciation)\b", "soc.economics.accounting"),
    (r"\b(marketing|advertising|branding|sales)\b", "soc.business.marketing"),
    # software engineering / OS (SO, softwareengineering, unix, superuser, serverfault)
    (r"\b(git|github|gitlab|git-merge|git-rebase|version-control|mercurial|svn)\b", "swe.git"),
    (r"\b(git-commit|commit-message|git-log)\b", "swe.git.commits"),
    (r"\b(git-branch|branching-and-merging|git-flow)\b", "swe.git.branching"),
    (r"\b(code-review|pull-request)\b", "swe.code_review"),
    (r"\b(unit-testing|testing|tdd|pytest|junit|mocking|integration-testing|test-automation|selenium)\b", "swe.testing"),
    (r"\b(debugging|gdb|lldb|stack-trace|segmentation-fault|valgrind|exception|traceback)\b", "swe.debugging"),
    (r"\b(continuous-integration|jenkins|github-actions|gitlab-ci|travis-ci|ci-cd|circleci)\b", "swe.ci"),
    (r"\b(pip|npm|maven|gradle|packaging|dependency-management|conda|cargo|nuget|composer-php|setuptools|pyproject\.toml|poetry)\b", "swe.packaging"),
    (r"\b(makefile|cmake|build|compilation|linker|bazel|msbuild|autotools|ninja)\b", "swe.build_systems"),
    (r"\b(documentation|docstring|sphinx|javadoc|doxygen)\b", "swe.documentation"),
    (r"\b(refactoring|design-patterns|object-oriented-design|solid-principles|clean-code|code-smell)\b", "swe.refactoring"),
    (r"\b(api-design|rest|restful-api|graphql|grpc|openapi|swagger)\b", "swe.api_design"),
    (r"\b(multithreading|concurrency|async-await|asynchronous|locking|mutex|parallel-processing|thread-safety|race-condition|deadlock)\b", "swe.concurrency"),
    (r"\b(performance|optimization|profiling|benchmarking|memory-leaks|caching)\b", "swe.performance"),
    (r"\b(security|sql-injection|xss|csrf|authentication|encryption|oauth|jwt)\b", "swe.security"),
    (r"\b(licensing|gpl|mit-license|open-source|copyright|apache-license)\b", "swe.licensing"),
    (r"\b(docker|kubernetes|containers|docker-compose|podman|helm)\b", "swe.devops.containers"),
    (r"\b(deployment|ansible|terraform|aws|nginx|load-balancing|cloud|heroku|systemd)\b", "swe.devops.deployment"),
    (r"\b(command-line|coreutils|grep|sed|awk|find|xargs|text-processing|pipe|ssh|rsync|tmux)\b", "swe.os.unix_tools"),
    (r"\b(macos|osx|homebrew|launchd|xcode|applescript)\b", "swe.os.macos"),
    (r"\b(linux|ubuntu|debian|kernel|linux-kernel|centos|fedora|arch-linux|apt|systemd|bootloader|grub)\b", "swe.os.linux"),
    (r"\b(database|sql|postgresql|mysql|sqlite|indexing|query-optimization|transactions|nosql|mongodb|redis)\b", "swe.databases"),
    (r"\b(networking|http|tcp|dns|tls|ssl|ip|routing|firewall|vpn|sockets|websocket)\b", "swe.networking"),
    (r"\b(web-development|frontend|reactjs|angular|vue\.js|django|flask|ruby-on-rails|express|spring-boot|asp\.net)\b", "swe.web_dev"),
    (r"\b(command-line-interface|argparse|click|cli|getopt|command-line-arguments)\b", "swe.cli_design"),
    # cs
    (r"\b(algorithms|data-structures|graphs|sorting|dynamic-programming|time-complexity|algorithm-analysis|trees|hash-tables|shortest-path)\b", "cs.algorithms"),
    (r"\b(complexity-theory|np-complete|np-hard|computational-complexity|p-vs-np|reductions)\b", "cs.complexity"),
    (r"\b(information-theory|entropy|coding-theory|compression|huffman)\b", "cs.information_theory"),
    (r"\b(programming-languages|type-theory|lambda-calculus|semantics|type-systems|compilers|parsing)\b", "cs.pl_theory"),
    (r"\b(formal-methods|formal-verification|model-checking|coq|lean|isabelle|agda|hoare-logic|proof-assistants|tla)\b", "cs.formal_methods"),
    (r"\b(operating-systems|computer-architecture|cpu|cache|virtual-memory|memory-management|scheduling|interrupts)\b", "cs.systems"),
    (r"\b(distributed-systems|consensus|paxos|raft|distributed-computing|mapreduce)\b", "cs.distributed"),
    (r"\b(game-theory|mechanism-design|nash-equilibrium|auctions)\b", "cs.game_theory"),
    (r"\b(opengl|shaders|rendering|ray-tracing|vulkan|directx|3d|graphics)\b", "cs.graphics"),
    (r"\b(user-interface|usability|human-computer-interaction|accessibility)\b", "cs.hci"),
    # security (security.SE, crypto.SE)
    (r"\b(penetration-test|exploit|metasploit|vulnerability-scanners|red-team|kali-linux|burp-suite|nmap|privilege-escalation)\b", "sec.offensive.pentest"),
    (r"\b(red-team|social-engineering|phishing-simulation)\b", "sec.offensive.red_team"),
    (r"\b(forensics|incident-response|memory-forensics|disk-forensics|log-analysis)\b", "sec.dfir.forensics"),
    (r"\b(malware|malware-analysis|virus|ransomware|trojan|rootkits|disassembly|ida|ghidra|unpacking|obfuscation|x86|deobfuscation|debuggers|binary-analysis)\b", "sec.dfir.malware_analysis"),
    (r"\b(ids|intrusion-detection|siem|splunk|monitoring|logging|snort|suricata)\b", "sec.defensive.siem"),
    (r"\b(threat-modeling|threat-intelligence|apt|attack-vector)\b", "sec.cti"),
    (r"\b(vulnerability|cve|patching|zero-day|vulnerability-management|cvss)\b", "sec.vuln"),
    (r"\b(compliance|pci-dss|iso27001|gdpr|hipaa|audit|risk-management|policy|nist|soc2)\b", "sec.grc"),
    (r"\b(pci-dss)\b", "sec.grc.pci_dss"), (r"\b(iso27001|iso-27001)\b", "sec.grc.iso27001"),
    (r"\b(gdpr|privacy|anonymity|tracking)\b", "sec.grc.privacy"),
    (r"\b(web-application|xss|sql-injection|csrf|owasp|appsec|content-security-policy|cookies)\b", "sec.domain.appsec"),
    (r"\b(aws|azure|cloud-computing|google-cloud-platform|cloud-security|s3)\b", "sec.domain.cloud"),
    (r"\b(network|firewalls|vpn|wifi|wpa2|router|network-scanners|tls|dns|ddos)\b", "sec.domain.network"),
    (r"\b(authentication|passwords|multi-factor|2fa|oauth|saml|access-control|authorization|kerberos|active-directory|ldap)\b", "sec.domain.iam"),
    (r"\b(encryption|aes|rsa|elliptic-curves|hash|block-cipher|public-key|signature|tls|key-exchange|diffie-hellman|random-number-generator|cryptanalysis)\b", "sec.domain.crypto"),
    (r"\b(scada|ics|plc|industrial-control-systems)\b", "sec.domain.ot_ics"),
    (r"\b(email|phishing|spam|spf|dkim|dmarc)\b", "sec.domain.email_phishing"),
    (r"\b(supply-chain|dependencies|package-managers|code-signing|sbom)\b", "sec.domain.supply_chain"),
]
TAG_RULES = [(re.compile(p), c) for p, c in _T]

# StackOverflow language tags -> prog.* (exact tag match, first listed wins as primary).
SO_LANG = {
    "python": "prog.python", "python-3.x": "prog.python", "pandas": "prog.python.scientific", "numpy": "prog.python.scientific",
    "scipy": "prog.python.scientific", "matplotlib": "prog.python.scientific", "pytorch": "prog.python.ml_frameworks",
    "tensorflow": "prog.python.ml_frameworks", "jax": "prog.python.ml_frameworks", "django": "prog.python.web",
    "flask": "prog.python.web", "fastapi": "prog.python.web", "python-typing": "prog.python.typing", "mypy": "prog.python.typing",
    "pytest": "prog.python.testing", "python-unittest": "prog.python.testing", "setuptools": "prog.python.packaging",
    "pip": "prog.python.packaging", "python-packaging": "prog.python.packaging", "cython": "prog.python.cython",
    "c": "prog.c", "c++": "prog.cpp", "objective-c": "prog.objc", "swift": "prog.swift", "rust": "prog.rust",
    "go": "prog.go", "java": "prog.java", "kotlin": "prog.kotlin", "scala": "prog.scala", "c#": "prog.csharp",
    "f#": "prog.fsharp", "javascript": "prog.javascript", "typescript": "prog.typescript", "ruby": "prog.ruby",
    "perl": "prog.perl", "php": "prog.php", "lua": "prog.lua", "r": "prog.r", "julia": "prog.julia",
    "matlab": "prog.matlab", "octave": "prog.matlab", "fortran": "prog.fortran", "haskell": "prog.haskell",
    "ocaml": "prog.ocaml", "common-lisp": "prog.lisp.common_lisp", "elisp": "prog.lisp.emacs_lisp",
    "emacs": "prog.lisp.emacs_lisp", "scheme": "prog.lisp.scheme", "racket": "prog.lisp.scheme",
    "clojure": "prog.lisp.clojure", "erlang": "prog.erlang", "elixir": "prog.elixir", "dart": "prog.dart",
    "flutter": "prog.dart", "zig": "prog.zig", "nim-lang": "prog.nim", "tcl": "prog.tcl", "lean": "prog.lean",
    "coq": "prog.coq", "prolog": "prog.prolog", "arm64": "prog.assembly.arm64", "arm": "prog.assembly.arm64",
    "x86": "prog.assembly.x86", "x86-64": "prog.assembly.x86", "assembly": "prog.assembly", "nasm": "prog.assembly.x86",
    "metal": "prog.gpu.metal", "cuda": "prog.gpu.cuda", "glsl": "prog.gpu.glsl", "hlsl": "prog.gpu.hlsl",
    "wgsl": "prog.gpu.wgsl", "triton": "prog.gpu.triton", "sh": "prog.shell.sh", "bash": "prog.shell.bash",
    "zsh": "prog.shell.zsh", "fish": "prog.shell.fish", "powershell": "prog.shell.powershell",
    "batch-file": "prog.shell.batch", "cmd": "prog.shell.batch", "sql": "prog.sql", "html": "prog.web.html",
    "css": "prog.web.css", "sass": "prog.web.scss", "scss": "prog.web.scss", "jinja2": "prog.web.templates",
    "handlebars.js": "prog.web.templates", "mustache": "prog.web.templates", "twig": "prog.web.templates",
    "markdown": "prog.markup.markdown", "restructuredtext": "prog.markup.rst", "latex": "prog.markup.latex",
    "asciidoc": "prog.markup.asciidoc", "manpage": "prog.markup.man", "json": "prog.data.json", "yaml": "prog.data.yaml",
    "toml": "prog.data.toml", "xml": "prog.data.xml", "csv": "prog.data.csv", "ini": "prog.data.ini",
    "protocol-buffers": "prog.data.protobuf", "makefile": "prog.build.make", "cmake": "prog.build.cmake",
    "bazel": "prog.build.bazel", "dockerfile": "prog.build.dockerfile", "nix": "prog.build.nix",
    "terraform": "prog.build.hcl", "hcl": "prog.build.hcl", "regex": "prog.regex", "pod": "prog.perl.pod",
}

# arXiv category -> node. Primary category decides; cross-lists only add candidates.
ARXIV = {
    "math.NT": "math.number_theory", "math.AG": "math.geometry.algebraic_geometry", "math.GR": "math.algebra.group_theory",
    "math.RA": "math.algebra.ring_theory", "math.AC": "math.algebra.ring_theory", "math.RT": "math.algebra.representation_theory",
    "math.CT": "math.algebra.category_theory", "math.OA": "math.algebra.operator_algebras", "math.QA": "math.algebra.quantum_algebra",
    "math.KT": "math.algebra", "math.DG": "math.geometry.differential_geometry", "math.SG": "math.geometry.symplectic",
    "math.MG": "math.geometry.metric_geometry", "math.GT": "math.topology", "math.GN": "math.topology",
    "math.AT": "math.topology.algebraic_topology", "math.CA": "math.analysis.real_complex", "math.CV": "math.analysis.real_complex",
    "math.FA": "math.analysis.functional_analysis", "math.SP": "math.analysis.functional_analysis", "math.AP": "math.analysis.pde",
    "math.DS": "math.dynamical_systems", "math.CO": "math.combinatorics", "math.LO": "math.logic", "math.PR": "math.probability",
    "math.ST": "math.statistics", "stat.TH": "math.statistics", "stat.ME": "math.statistics", "stat.AP": "math.statistics",
    "stat.CO": "math.statistics", "math.NA": "math.numerical_analysis", "cs.NA": "math.numerical_analysis",
    "math.OC": "math.optimization", "math-ph": "math.mathematical_physics", "math.MP": "math.mathematical_physics",
    "math.IT": "cs.information_theory", "cs.IT": "cs.information_theory", "math.GM": "math", "math.HO": "math",
    "cs.CC": "cs.complexity", "cs.FL": "math.theoretical_cs", "cs.DS": "cs.algorithms", "cs.DM": "math.combinatorics",
    "cs.LO": "cs.formal_methods", "cs.PL": "cs.pl_theory", "cs.CR": "sec", "cs.NI": "cs.networks", "cs.DB": "cs.databases",
    "cs.GR": "cs.graphics", "cs.HC": "cs.hci", "cs.DC": "cs.distributed", "cs.GT": "cs.game_theory", "cs.RO": "cs.robotics",
    "cs.OS": "cs.systems", "cs.AR": "cs.systems", "cs.SE": "swe", "cs.CG": "math.geometry.metric_geometry",
    "hep-th": "sci.physics.high_energy_theory", "hep-ph": "sci.physics.particle", "hep-ex": "sci.physics.particle",
    "hep-lat": "sci.physics.particle", "nucl-th": "sci.physics.nuclear", "nucl-ex": "sci.physics.nuclear",
    "quant-ph": "sci.physics.quantum", "gr-qc": "sci.physics.relativity", "physics.optics": "sci.physics.optics",
    "physics.flu-dyn": "sci.physics.fluids", "physics.plasm-ph": "sci.physics.plasma", "physics.atom-ph": "sci.physics.atomic",
    "physics.atm-clus": "sci.physics.atomic", "physics.class-ph": "sci.physics.classical_mechanics",
    "physics.gen-ph": "sci.physics", "physics.app-ph": "sci.physics", "physics.ins-det": "sci.physics.particle",
    "physics.acc-ph": "sci.physics.particle", "physics.comp-ph": "sci.physics", "physics.chem-ph": "sci.chemistry.physical",
    "physics.ed-ph": "soc.education", "physics.hist-ph": "hum.history", "physics.geo-ph": "sci.earth.geology",
    "physics.ao-ph": "sci.earth.meteorology", "nlin.CD": "sci.physics.nonlinear", "nlin.PS": "sci.physics.nonlinear",
    "nlin.SI": "sci.physics.nonlinear", "nlin.AO": "sci.physics.nonlinear", "nlin.CG": "sci.physics.nonlinear",
    "cond-mat.mtrl-sci": "sci.materials", "astro-ph.GA": "sci.astronomy.galaxies", "astro-ph.CO": "sci.astronomy.cosmology",
    "astro-ph.SR": "sci.astronomy.stellar", "astro-ph.HE": "sci.astronomy.high_energy", "astro-ph.EP": "sci.astronomy.planetary",
    "astro-ph.IM": "sci.astronomy.instrumentation", "astro-ph": "sci.astronomy", "eess.SP": "eng.signal_processing",
    "eess.AS": "eng.signal_processing", "eess.IV": "eng.signal_processing", "eess.SY": "eng.electrical",
    "cs.SY": "eng.electrical", "econ.EM": "soc.economics", "econ.GN": "soc.economics", "econ.TH": "soc.economics",
    "q-fin.CP": "soc.economics.finance", "q-fin.EC": "soc.economics", "q-fin.GN": "soc.economics.finance",
    "q-fin.MF": "soc.economics.finance", "q-fin.PM": "soc.economics.finance", "q-fin.PR": "soc.economics.finance",
    "q-fin.RM": "soc.economics.finance", "q-fin.ST": "soc.economics.finance", "q-fin.TR": "soc.economics.finance",
    "q-bio.NC": "sci.biology.neuroscience", "q-bio.PE": "sci.biology.evolution", "q-bio.GN": "sci.biology.genetics",
    "q-bio.MN": "sci.biology.molecular", "q-bio.BM": "sci.biology.molecular", "q-bio.SC": "sci.biology.cell",
    "q-bio.CB": "sci.biology.cell",
}
ARXIV_SKIP_PREFIX = ("physics.med-ph", "q-bio.TO", "q-bio.QM")      # medical: never outside sci.medicine


def arxiv_node(cat):
    if cat.startswith(ARXIV_SKIP_PREFIX):
        return None
    if cat in ARXIV:
        return ARXIV[cat]
    if cat.startswith("cond-mat"):
        return "sci.physics.condensed_matter"
    if cat.startswith("astro-ph"):
        return "sci.astronomy"
    return None


# LibreTexts subdomain + bookshelf -> node (sections of the same book share a group). Courses/ pages are
# subdomain-level candidates only. med is dropped (medical); espanol is not English.
LIBRE_SUB = {"math": "math", "stats": "math.statistics", "phys": "sci.physics", "chem": "sci.chemistry",
             "bio": "sci.biology", "geo": "sci.earth", "eng": "eng", "biz": "soc.business", "socialsci": "soc",
             "human": "hum", "workforce": "eng"}
LIBRE_SHELF = {
    "math/Abstract_and_Geometric_Algebra": "math.algebra", "math/Algebra": "math.algebra", "math/Analysis": "math.analysis",
    "math/Applied_Mathematics": "math.numerical_analysis", "math/Arithmetic_and_Basic_Math": "math.arithmetic",
    "math/Calculus": "math.analysis.calculus", "math/Combinatorics_and_Discrete_Mathematics": "math.combinatorics",
    "math/Differential_Equations": "math.analysis.ode", "math/Geometry": "math.geometry.euclidean",
    "math/Linear_Algebra": "math.algebra.linear_algebra", "math/Mathematical_Logic_and_Proof": "math.logic",
    "math/PreAlgebra": "math.arithmetic", "math/Precalculus": "math.algebra",
    "math/Scientific_Computing_Simulations_and_Modeling": "math.numerical_analysis",
    "stats/Probability_Theory": "math.probability", "stats/Applied_Statistics": "math.statistics",
    "stats/Introductory_Statistics": "math.statistics",
    "phys/Quantum_Mechanics": "sci.physics.quantum", "phys/Relativity": "sci.physics.relativity",
    "phys/Mathematical_Physics_and_Pedagogy": "math.mathematical_physics", "phys/Astronomy__Cosmology": "sci.astronomy",
    "phys/Thermodynamics_and_Statistical_Mechanics": "sci.physics.thermodynamics", "phys/Optics": "sci.physics.optics",
    "phys/Electricity_and_Magnetism": "sci.physics.electromagnetism", "phys/Classical_Mechanics": "sci.physics.classical_mechanics",
    "phys/Nuclear_and_Particle_Physics": "sci.physics.nuclear", "phys/Condensed_Matter": "sci.physics.condensed_matter",
    "eng/Aerospace_Engineering": "eng.aerospace", "eng/Biological_Engineering": "eng.biomedical",
    "eng/Chemical_Engineering": "eng.chemical", "eng/Electrical_Engineering": "eng.electrical",
    "eng/Mechanical_Engineering": "eng.mechanical", "eng/Civil_Engineering": "eng.civil",
    "eng/Industrial_and_Systems_Engineering": "eng.industrial", "eng/Materials_Science": "sci.materials",
    "eng/Computer_Science": "cs", "workforce/Construction": "eng.civil", "workforce/Manufacturing": "eng.industrial",
    "workforce/Electronics_Technology": "eng.electrical", "workforce/HVAC_and_Power_Plant_Operations": "eng.mechanical",
    "workforce/Drafting_and_Design_Technology": "eng.mechanical", "workforce/Water_Systems_Technology": "eng.civil",
    "workforce/Information_Technology": "swe",
    "biz/Accounting": "soc.economics.accounting", "biz/Finance": "soc.economics.finance", "biz/Business": "soc.business",
    "biz/Management": "soc.business", "biz/Marketing": "soc.business.marketing", "biz/Civil_Law": "soc.law",
    "biz/Constitutional_Law": "soc.law.constitutional", "biz/Personal_Finance": "life.personal_finance",
    "socialsci/Anthropology": "soc.anthropology", "socialsci/Communication": "soc.communication",
    "socialsci/Economics": "soc.economics", "socialsci/Education_and_Professional_Development": "soc.education",
    "socialsci/Early_Childhood_Education": "soc.education", "socialsci/Psychology": "soc.psychology",
    "socialsci/Sociology": "soc.sociology", "socialsci/Political_Science_and_Civics": "soc.political_science",
    "socialsci/Geography_(Human)": "soc.geography", "socialsci/Linguistics": "hum.linguistics",
    "human/History": "hum.history", "human/Religious_Studies": "hum.religion", "human/Philosophy": "hum.philosophy",
    "human/Literature_and_Literacy": "hum.literature", "human/Music": "hum.music", "human/Art": "hum.art_history",
    "human/Theater_Film_and_Storytelling": "hum.film", "geo/Geology": "sci.earth.geology",
    "geo/Oceanography": "sci.earth.oceanography", "bio/Ecology": "sci.biology.ecology", "bio/Botany": "sci.biology.botany",
    "bio/Microbiology": "sci.biology.microbiology", "bio/Genetics": "sci.biology.genetics",
    "bio/Cell_and_Molecular_Biology": "sci.biology.molecular", "bio/Evolutionary_Developmental_Biology": "sci.biology.evolution",
}

# peS2o s2fieldsofstudy -> candidate nodes. Papers with Medicine among their fields are dropped (box spec).
PES2O_FIELD = {
    "Engineering": ["eng"], "Materials Science": ["sci.materials", "eng.chemical", "eng.mechanical"],
    "Economics": ["soc.economics"], "Business": ["soc.business", "soc.economics"], "History": ["hum.history"],
    "Education": ["soc.education"], "Sociology": ["soc.sociology", "soc.anthropology", "soc.communication"],
    "Political Science": ["soc.political_science"], "Philosophy": ["hum.philosophy", "hum.religion"],
    "Art": ["hum.art_history", "hum.film", "hum.music"], "Geography": ["soc.geography", "life.travel"],
    "Mathematics": ["math"], "Physics": ["sci.physics"], "Computer Science": ["cs", "swe", "sec"],
    "Psychology": ["soc.psychology"], "Linguistics": ["hum.linguistics"], "Law": ["soc.law"],
    "Chemistry": ["sci.chemistry", "eng.chemical"], "Environmental Science": ["sci.earth", "eng.civil"],
    "Geology": ["sci.earth.geology"], "Biology": ["sci.biology"], "Agricultural and Food Sciences": ["life.cooking"],
}
PES2O_DROP = {"Medicine"}


def tag_candidates(tags):
    """'<a><b>' or '|a|b|' tag string -> set of candidate nodes from TAG_RULES."""
    s = "|" + "|".join(t for t in re.split(r"[<>|]", tags or "") if t) + "|"
    return {c for rx, c in TAG_RULES if rx.search(s)}
