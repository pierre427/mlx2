import SwiftUI

enum ChildTools {
    static let names = ["web_search", "fetch_page", "memory_search", "memory_get", "message_post", "message_read", "message_ack", "skills_list", "skill_view", "skill_manage"]
    static func requested(in job: JSON) -> Set<String> {
        Set(job["tool_requests"].array.flatMap { $0["tools"].array.compactMap(\.string) }).intersection(Set(names))
    }
    static func grantable(in job: JSON) -> [String] {
        requested(in: job).subtracting(Set(job["tools"].array.compactMap(\.string))).sorted()
    }
}

struct ChildToolSelectionView: View {
    @Binding var selection: Set<String>
    var available = ChildTools.names
    var body: some View {
        VStack(alignment: .leading, spacing: 5) {
            ForEach(available, id: \.self) { name in
                Toggle(name, isOn: Binding(get: { selection.contains(name) }, set: { enabled in
                    if enabled { selection.insert(name) } else { selection.remove(name) }
                })).font(.caption.monospaced())
            }
        }
    }
}

struct AgentJobControls: View {
    @EnvironmentObject var model: DesktopModel
    let job: JSON
    @State private var grant = Set<String>()
    @State private var message = ""
    private var available: [String] { ChildTools.grantable(in: job) }
    private var resumable: Bool { job["workers"].array.count == 1 && ["completed", "failed", "cancelled", "interrupted"].contains(job["status"].text) }
    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Granted tools: " + (job["tools"].array.compactMap(\.string).joined(separator: ", ").isEmpty ? "none" : job["tools"].array.compactMap(\.string).joined(separator: ", "))).font(.caption).foregroundStyle(.secondary)
            if !job["tool_requests"].array.isEmpty {
                DisclosureGroup("Requested tools") {
                    ForEach(Array(job["tool_requests"].array.enumerated()), id: \.offset) { _, request in
                        Text(request["tools"].array.compactMap(\.string).joined(separator: ", ")).font(.caption.monospaced())
                        Text(request["reason"].text).font(.caption).textSelection(.enabled)
                    }
                    if !available.isEmpty {
                        ChildToolSelectionView(selection: $grant, available: available)
                        Button("Grant Selected Requested Tools") {
                            let admitted = grant.intersection(Set(available)).sorted()
                            model.perform("agent_grant_tools", ["job_id": job["id"].text, "tools": admitted])
                            grant = []
                        }.disabled(model.busy || grant.intersection(Set(available)).isEmpty)
                        Text("Grants apply to the next resume and remain subject to application policy.").font(.caption).foregroundStyle(.secondary)
                    }
                }
            }
            if resumable {
                TextField("Follow-up for this child", text: $message, axis: .vertical)
                Button("Resume Child") {
                    model.perform("agent_resume", ["job_id": job["id"].text, "message": message])
                }.disabled(model.busy || message.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                Text("Continues this child's private context and granted tools on the exact model instance. It starts no replacement model.").font(.caption).foregroundStyle(.secondary)
            }
        }
    }
}
