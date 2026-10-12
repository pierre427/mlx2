"""Model-owned routed-MoE contracts; shared kernels only validate them."""

from .moe_routing_contract import MoeRoutingContract

QWEN4_E512_K10_ROUTING = MoeRoutingContract(
    owner="qwen4-exp-e512-k10-v1", expert_count=512, top_k=10,
    normalize_topk=True, selection_law="precise_softmax_argpartition",
    tie_order="stable_ascending_expert",
    slot_order="ascending_selected_probability", activation_dtype="bfloat16",
    projection_layouts=frozenset({"fused_gate_up", "split_gate_up"}),
    quantization="affine_q4_group64",
    expert_kernel_layout="qmv_down_non512_intermediate",
    shared_expert_form="dense_swiglu_sigmoid_gate",
    allowed_row_consumers=frozenset({"row_exact", "batch_decode", "verify"}),
)

QWEN36_E256_K8_ROUTING = MoeRoutingContract(
    owner="qwen3.6-35b-a3b-e256-k8-v1", expert_count=256, top_k=8,
    normalize_topk=True, selection_law="precise_softmax_argpartition",
    tie_order="stable_ascending_expert",
    slot_order="ascending_selected_probability", activation_dtype="bfloat16",
    projection_layouts=frozenset({"split_gate_up"}),
    quantization="affine_q4_group64",
    expert_kernel_layout="qmv_fast_down_h2048_i512",
    shared_expert_form="dense_swiglu_sigmoid_gate",
    allowed_row_consumers=frozenset({"row_exact", "batch_decode", "verify"}),
)
