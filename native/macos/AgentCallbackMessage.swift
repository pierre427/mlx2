import SwiftUI

// The complete callback remains in wire history. Only its agent-authored report
// and execution outcome belong in the conversation's visual presentation.
struct AgentCallbackMessage: View {
    let payload: JSON
    @State private var session: AgentSessionSelection?
    static func summary(_ payload: JSON) -> String {
        var parts = ["Subagent · " + payload["status"].text]
        if let synthesis = payload["synthesis"].string, !synthesis.isEmpty { parts.append(synthesis) }
        for worker in payload["workers"].array {
            let report = worker["report"].string ?? worker["report"]["content"].string ?? ""
            if !report.isEmpty && report != payload["synthesis"].string { parts.append(report) }
            if let cause = worker["failure"]["cause"].string { parts.append(cause) }
            else if let error = worker["error"].string { parts.append(error) }
        }
        if parts.count == 1, let error = payload["error"].string { parts.append(error) }
        return parts.joined(separator: "\n\n")
    }
    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            ChatMarkdownView(text: Self.summary(payload))
            if let worker = payload["workers"].array.first, let id = worker["worker_id"].string ?? worker["id"].string,
               let job = payload["job_id"].string {
                Button("Open Agent Session", systemImage: "arrow.up.right.square") {
                    session = AgentSessionSelection(jobID: job, workerID: id)
                }.font(.caption)
            }
        }.sheet(item: $session) { selection in AgentSessionView(jobID: selection.jobID, workerID: selection.workerID) }
    }
}
