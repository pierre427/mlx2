import CoreFoundation
import Foundation

// Per-turn execution budget. Model requests never bypass the absolute ceiling.
struct ChatToolBudget {
    static let name = "mlx2_tool_budget"
    static let maximum = 128
    private(set) var limit = 16
    private(set) var requests = 0
    static var schema: JSON {
        JSON(["type": "function", "function": ["name": name,
            "description": "Plan the remaining tool calls for this user turn. Request additional_calls plus a short reason; the harness adds 25% padding (at least 4), capped at 128 total calls. At a budget boundary only this tool is available: request more calls or answer from completed results. At most 4 budget requests per turn. Before each round, assess whether session compaction is needed.",
            "parameters": ["type": "object", "properties": ["additional_calls": ["type": "integer", "minimum": 1, "maximum": 96], "reason": ["type": "string", "maxLength": 1024]],
                "required": ["additional_calls", "reason"], "additionalProperties": false]]])
    }
    func canExtend(calls: Int) -> Bool { calls < Self.maximum && requests < 4 }
    mutating func request(_ arguments: [String: Any], calls: Int) -> JSON {
        guard canExtend(calls: calls) else { return JSON(["error": "The absolute tool budget or four planning requests have been exhausted. Answer using completed results."]) }
        requests += 1
        guard Set(arguments.keys) == Set(["additional_calls", "reason"]),
              let number = arguments["additional_calls"] as? NSNumber, CFGetTypeID(number) != CFBooleanGetTypeID(),
              number.doubleValue.isFinite, number.doubleValue.rounded(.towardZero) == number.doubleValue,
              (1...96).contains(number.intValue), let reason = arguments["reason"] as? String,
              !reason.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty, reason.utf8.count <= 1024 else {
            return JSON(["error": "Provide additional_calls (integer 1–96) and a nonempty reason of at most 1024 UTF-8 bytes."])
        }
        let requested = number.intValue
        let padding = max(4, (requested + 3) / 4)
        limit = min(Self.maximum, calls + requested + padding)
        return JSON(["result": ["total_call_limit": limit, "calls_used": calls, "calls_remaining": max(0, limit - calls),
            "requested_additional_calls": requested, "padding": min(padding, max(0, limit - calls - requested)),
            "hard_ceiling": Self.maximum, "budget_requests_remaining": 4 - requests,
            "next_step": "Review context usage. Compact through session tools if needed; otherwise continue the plan within this budget."]])
    }
    func guidance(calls: Int, negotiating: Bool, previousContext: Int?, contextLimit: Int?) -> String {
        "\n\n[Harness round checkpoint: Before choosing your next action, decide whether context compaction is needed. Use the session tools to obtain the exact archive revision/digest and compact when needed; preserve recent turns and exact originals. Do not claim a compaction happened without its receipt. " +
        (contextLimit.map { "Configured context window: \($0). " } ?? "") +
        "The turn starts with 16 tool calls and has a hard ceiling of \(Self.maximum). Current counts are included with newly returned results. mlx2_tool_budget is already declared locally; do not search the backend catalog for it or add a namespace prefix. " +
        (negotiating ? "The current tool budget is reached. Other tools are paused. Call mlx2_tool_budget to request additional_calls with a reason, or answer the original user using completed results. " : "For multi-step work, use mlx2_tool_budget to plan the remaining calls with padding. For capability questions, answer from relevant discovery metadata without running controls to demonstrate them. ") +
        "When delegation is enabled, ordinary subagents use mlx2__agent_delegate with goals and no target: separate contexts on this same running engine's lane pool. Creation returns a job_id asynchronously: read mlx2__agent_result for that exact job_id to get the child report; repeating agent_delegate does not wait for a child. Use mlx2_tool_call with the exact tool name and arguments if its schema is not declared. Report the completed child result to the user and stop once the goal is met. Use mlx2__agent_launch only for an explicitly requested different model, quant, LoRA or separately loaded instance; never as a fallback for a delegation error. " +
        "Research belongs in fresh-context subagents on the same running engine. Establish a plan before research, track observed search and source-reading progress, and synthesize citations from fetched pages. Search snippets alone are not source verification. Read the child result rather than repeating research requests. For show/find a picture or photo, discover mlx2__world_image_search and put the subject in that tool's query; display returned URLs with Markdown image syntax. If the user asks for several images, request that count (up to the search limit), return each distinct image on its own Markdown image line with attribution, and state any shortfall. Source URLs render directly in chat; fetching into the 3D texture store is optional. Use media generation only when the user asks to generate or edit media, and respect configured providers. " +
        "For work that needs substantial tool output, choose result_max_tokens and chunk_tokens with mlx2__tool_result_policy (discover its schema or invoke through mlx2_tool_call). Oversized results return a result_id; use mlx2__tool_result_status for a bounded child assessment rather than importing full raw outputs. Do useful work while an assessment is queued; avoid repeated status polling.]"
    }
}
