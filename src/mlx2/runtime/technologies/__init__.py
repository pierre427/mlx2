"""One reference implementation per model technology.

Each module in this package owns one technology (linear recurrence, sparse
selection, MLA, hyper-connections, MoE routing, ...) and follows the same
shape:

* a frozen ``Spec`` dataclass carrying every per-model parameter the math
  needs, with ``Spec.from_config(config, **overrides)`` deriving it from the
  checkpoint config where the config says enough;
* a pure-MLX ``reference`` path: the ordinary-decode source of truth, written
  once and parameterised by the spec (never by model name);
* optional fast paths registered in a :class:`dispatch.KernelRegistry`; each
  declares which specs and shapes it admits and returns a refusal reason
  otherwise, so the dispatcher falls back to the reference or fails closed
  and records a receipt either way.

Model files and adapters build a spec and call the technology; they do not
re-implement its math. Legitimate model differences are spec fields;
historical numerics differences that a model's identity depends on are spec
fields too, named for what they do, until a reviewed change retires them.
"""
