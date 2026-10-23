import Foundation

// Polling leaves the desktop command pipe available for cancellation and UI work.
enum SubagentToolWaiter {
    static func wait(sessionID: String, jobID: String,
                     request: (String, [String: Any]) async throws -> JSON) async throws -> JSON {
        do {
            let deadline = Date().addingTimeInterval(1800)
            while Date() < deadline {
                try Task.checkCancellation()
                let state = try await request("tool_call", ["session_id": sessionID,
                    "name": "mlx2__agent_result", "arguments": ["job_id": jobID], "approved": false])
                let status = state["result"]["status"].text
                if status == "completed" {
                    guard let results = state["result"]["results"].raw as? [[String: Any]], !results.isEmpty else {
                        throw DesktopError.message("Subagent batch has no structured results.")
                    }
                    return JSON(["result": results, "job_id": jobID])
                }
                if ["failed", "cancelled", "interrupted"].contains(status) {
                    throw DesktopError.message("Subagent batch " + status + "; inspect its retained receipt in Agents.")
                }
                try await Task.sleep(for: .milliseconds(500))
            }
            throw DesktopError.message("Subagent batch exceeded its 30 minute budget.")
        } catch {
            _ = try? await request("tool_call", ["session_id": sessionID,
                "name": "mlx2__agent_cancel", "arguments": ["job_id": jobID], "approved": true])
            throw error
        }
    }
}
