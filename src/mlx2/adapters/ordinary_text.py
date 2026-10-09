"""Small serving surface for unqualified ordinary text model candidates."""

from __future__ import annotations


class OrdinaryTextAdapter:
    default_route = "ordinary"

    @staticmethod
    def spomin_backend(model, prompt_cache):
        return None

    def approximate_kv_operations(self):
        return {}

    def profile_name(self, mtp):
        if mtp:
            raise ValueError("this adapter has no MTP route")
        return self.profile

    def execution_config(self, *, max_lanes, prefill_step):
        return {
            "persistent": True,
            "num_draft": 0,
            "rate_gate": False,
            "prefill_step_size": prefill_step,
            "segment_aware_live_tip": False,
            "segment_aware_cohort_size": max_lanes,
        }

    # No reasoning is declared and the parser is content-only, so the prompt
    # must not open a think channel.  TokenizerWrapper defaults
    # enable_thinking to the tokenizer's has_thinking, which rendered Agnes 3
    # Flash's "<think>" generation prompt and leaked its reasoning (and a
    # literal "</think>") into the answer, with no way to turn it off.
    def prompt_tokens(self, request):
        if "messages" in request:
            return self.tokenizer.apply_chat_template(
                request["messages"], add_generation_prompt=True, tokenize=True,
                enable_thinking=False,
            )
        return self.tokenizer.encode(request["prompt"], add_special_tokens=False)

    def render_prompt(self, request):
        if "messages" in request:
            return self.tokenizer.apply_chat_template(
                request["messages"], add_generation_prompt=True, tokenize=False,
                enable_thinking=False,
            )
        return request["prompt"]

    def output_parser(self, request):
        from ..output import OutputParser, callable_tools

        if callable_tools(request) or request.get("enable_thinking"):
            raise ValueError("tools and reasoning are not declared for this adapter")
        return OutputParser(chat="messages" in request, thinking=False, tools=None,
                            stops=request.get("stop", ()))

    def tool_constraint(self, request):
        from ..output import callable_tools

        if callable_tools(request):
            raise ValueError("tools are not declared for this adapter")
        return None

    def diagnostics(self):
        return {"route": "ordinary", "qualification": "pending"}

    def close(self):
        # A no-op unless the adapter claimed process globals at construction.
        from .process_globals import release

        release(self)
