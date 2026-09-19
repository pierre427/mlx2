# AGENTS.md — mlx2

`mlx2` is a clean-generation inference runtime. Keep its production core small
and mine `mlx-lm-unified` as a source of proven mechanisms, not as a tree to
copy wholesale.

## Working rules

- APCv2 is the only prefix-cache engine. Do not introduce legacy APC, a legacy
  serving backend, or a fallback to a replaced product.
- Deliver working model-serving slices using the current proven mechanisms;
  migrate other model families onto these contracts.
- Add a provenance entry before or with every mined implementation. Record the
  source repository revision, paths, license, modifications, and validation.
- Put model-specific tensor math behind adapters. Do not branch on model names
  in the scheduler, cache, or request lifecycle.
- Keep `implemented`, `qualified`, `selected`, and `observed-used` as distinct
  states. A route is selectable only when its required capabilities are both
  declared and qualified with evidence.
- Emit a route receipt and fail closed when a requested capability is absent.
- Preserve an ordinary-decode reference path for every model family.
- Keep state operations revision-bound. Approximate state may only be published
  through an explicitly qualified approximate operation.
- Never add an Apple copyright header to original project code.
- Do not publish to or interact with `ml-explore/mlx` or `ml-explore/mlx-lm`.
