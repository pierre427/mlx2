# SPDX-License-Identifier: MIT
"""Source policy admission for scheduled DSpark chains and lockstep verify rows.

Opt-in only; absent keys leave every existing profile and identity unchanged.
"""
SCHEDULE_KEYS = ("deepseek_dspark_verify_schedule", "deepseek_dspark_verify_costs", "deepseek_dspark_draft_cost")
LOCKSTEP = "lockstep_multirow_v1"


def split_chain_policy(value, *, embedded):
    """Remove chain keys the shared native parser does not own; validate shape."""
    from .deepseek_v41_dspark_schedule import validate
    chain = {key: value.pop(key) for key in SCHEDULE_KEYS if key in value}
    if chain and not embedded:
        raise ValueError("DSpark verify scheduling requires draft_model=embedded_dspark")
    if chain:
        validate(chain.get("deepseek_dspark_verify_schedule", "fixed"),
                 chain.get("deepseek_dspark_verify_costs"), chain.get("deepseek_dspark_draft_cost"))
        if "deepseek_dspark_verify_costs" in chain:
            chain["deepseek_dspark_verify_costs"] = [float(v) for v in chain["deepseek_dspark_verify_costs"]]
    if value.get("deepseek_verify_rows") == LOCKSTEP:
        chain["deepseek_verify_rows"] = value.pop("deepseek_verify_rows")
    return chain


def validate_chain_policy(policy):
    if policy.get("deepseek_verify_rows", "serial") != LOCKSTEP:
        return
    if policy.get("draft_model") != "embedded_dspark" and not policy.get("ast_prompt_lookup"):
        raise ValueError("lockstep_multirow_v1 verifies a B1 speculative chain; select a draft route")
    if policy.get("deepseek_verify_commit", "replay") != "local_frontier":
        raise ValueError("lockstep_multirow_v1 requires deepseek_verify_commit=local_frontier")
    for name, reference in (("deepseek_kv_storage", "reference"), ("deepseek_ced", "reference"),
                            ("deepseek_physical_batch", "b1"), ("deepseek_vision", "reference"),
                            ("deepseek_source_dspark_ast", "off"), ("deepseek_source_dspark_prepare", "off"),
                            ("deepseek_source_resident_gather", "reference"),
                            ("deepseek_source_resident_prefill", "reference"),
                            ("deepseek_source_row_packet", "reference"),
                            ("deepseek_source_ffn_input", "reference")):
        if policy.get(name, reference) != reference:
            raise ValueError("lockstep_multirow_v1 is unsupported with " + name)
    try:
        import greenlet  # noqa: F401 - row scheduler; not a declared base dependency
    except ImportError:
        raise ValueError("lockstep_multirow_v1 requires the greenlet package") from None
