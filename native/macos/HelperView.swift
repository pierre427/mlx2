import AppKit
import SwiftUI

struct HelperView: View {
    @EnvironmentObject var model: DesktopModel
    @State private var provider = "local"
    @State private var prompt = ""
    @State private var selectedModel = ""
    @State private var proposal = ""
    private var preferences: JSON { model.snapshot["preferences"] }
    private var jobs: [JSON] { model.snapshot["helpers"]["jobs"].array }
    private func preference(_ key: String) -> Binding<Bool> {
        Binding(get: { preferences[key].bool }, set: { model.savePreferences([key: $0]) })
    }
    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                Text("Advisory Helpers & Adapter Drafts").font(.title2)
                Text("Generate advice and review-only adapter proposals in isolated workspaces. Generated adapters are untested and must be reviewed, integrated and validated before use.").foregroundStyle(.secondary)
                Toggle("Enable helpers and expose them to chat tools", isOn: preference("helper_enabled"))
                Toggle("Allow installed Codex CLI", isOn: preference("helper_codex_enabled"))
                Toggle("Allow installed Claude Code CLI", isOn: preference("helper_claude_enabled"))
                Text("Installed: Codex \(model.snapshot["helpers"]["available"]["codex"].bool ? "available" : "missing") · Claude Code \(model.snapshot["helpers"]["available"]["claude"].bool ? "available" : "missing")").font(.caption).foregroundStyle(.secondary)
                Picker("Local advisory model", selection: Binding(get: { preferences["adapter_model_path"].string ?? "" }, set: { model.savePreferences(["adapter_model_path": $0]) })) {
                    Text("Choose a model (start it before asking for help)").tag("")
                    ForEach(model.library["rows"].array.filter { $0["can_start"].bool }, id: \.rawIdentity) { row in
                        Text(row["name"].text).tag(row["path"].text)
                    }
                }
                Picker("Default adapter helper", selection: Binding(get: { preferences["adapter_helper"].string ?? "local" }, set: { model.savePreferences(["adapter_helper": $0]) })) {
                    Text("Local model").tag("local"); Text("Codex").tag("codex"); Text("Claude Code").tag("claude")
                }
                Divider()
                Picker("Helper for this request", selection: $provider) {
                    Text("Local model").tag("local"); Text("Codex").tag("codex"); Text("Claude Code").tag("claude")
                }
                TextEditor(text: $prompt).frame(height: 100).border(.separator)
                Button("Ask Helper") { model.perform("helper_start", ["provider": provider, "prompt": prompt]) }
                    .disabled(model.busy || prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty || !preferences["helper_enabled"].bool)
                Picker("Draft an adapter for", selection: $selectedModel) {
                    Text("Choose model").tag("")
                    ForEach(model.library["rows"].array, id: \.rawIdentity) { row in Text(row["name"].text).tag(row["path"].text) }
                }
                Button("Generate Review-Only Adapter Proposal") {
                    if let row = model.library["rows"].array.first(where: { $0["path"].string == selectedModel }) {
                        model.perform("adapter_draft", ["provider": provider, "path": row["path"].text, "identity": row["identity"].text])
                    }
                }.disabled(model.busy || selectedModel.isEmpty || !preferences["helper_enabled"].bool)
                ForEach(Array(jobs.enumerated()), id: \.offset) { _, row in
                    HStack {
                        VStack(alignment: .leading) { Text("\(row["provider"].text) · \(row["kind"].text) · \(row["status"].text)"); if let error = row["error"].string { Text(error).font(.caption).foregroundStyle(.orange) } }
                        Spacer()
                        if ["queued", "running"].contains(row["status"].text) {
                            Button("Cancel") { model.perform("helper_cancel", ["job_id": row["id"].text]) }
                        }
                        if row["status"].text == "completed" {
                            Button("Read Proposal") { Task { do { proposal = try await model.request("helper_result", ["job_id": row["id"].text])["proposal"].text } catch { model.error = error.localizedDescription } } }
                        }
                        Button("Show Workspace") { NSWorkspace.shared.selectFile(nil, inFileViewerRootedAtPath: row["workspace"].text) }
                    }.padding(10).background(.quaternary, in: RoundedRectangle(cornerRadius: 8))
                }
                if !proposal.isEmpty { Text(proposal).font(.system(.body, design: .monospaced)).textSelection(.enabled) }
            }.padding(20)
        }
    }
}

private extension JSON {
    var rawIdentity: String { self["identity"].string ?? self["path"].text }
}
