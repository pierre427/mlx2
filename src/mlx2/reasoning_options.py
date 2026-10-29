# SPDX-License-Identifier: MIT
"""Shared public thinking controls, before a model assigns effort semantics."""


def normalize_reasoning_options(request):
    result = dict(request)
    template = result.get("chat_template_kwargs")
    if template is None:
        template = {}
    if not isinstance(template, dict):
        raise ValueError("chat_template_kwargs must be an object")  # noqa: TRY004 - public request refusal contract
    for name in ("enable_thinking", "think", "reasoning_effort"):
        if result.get(name) is None:
            result.pop(name, None)
    toggles = []
    for name, value in (
        ("enable_thinking", result.get("enable_thinking")),
        ("think", result.pop("think", None)),
        ("chat_template_kwargs.enable_thinking", template.get("enable_thinking")),
    ):
        if value is not None:
            if type(value) is not bool:
                raise ValueError(f"{name} must be boolean")
            toggles.append(value)
    effort = result.get("reasoning_effort")
    if effort is not None and not (
        (type(effort) is int and 1 <= effort <= 100)
        or (type(effort) is str and effort in {
            "none", "minimal", "low", "medium", "high", "xhigh", "max", "ultra"
        })
    ):
        raise ValueError("invalid reasoning_effort: use a named level or integer1..100")
    if toggles:
        if any(value != toggles[0] for value in toggles):
            raise ValueError("thinking toggles disagree")
        result["enable_thinking"] = toggles[0]
    elif effort is not None:
        result["enable_thinking"] = effort != "none"
    return result
