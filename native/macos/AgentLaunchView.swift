import SwiftUI

struct AgentLaunchView: View {
    @EnvironmentObject var model: DesktopModel
    @State private var path = ""
    @State private var instance = "new"
    @State private var goal = ""
    @State private var lanes = 1
    @State private var context = "0"
    @State private var inherit = false
    @State private var loraPath = ""
    @State private var loraName = ""
    @State private var launching = false
    @State private var childTools = Set<String>()
    private var jobs: [JSON] { model.snapshot["agent_launches"]["jobs"].array }
    var body: some View {
        GroupBox("Launch on a selected model or instance") {
            VStack(alignment: .leading, spacing: 10) {
                Picker("Model / quantized artifact", selection: $path) {
                    Text("Choose a model").tag("")
                    ForEach(Array(model.rows.filter { $0["can_start"].bool }.enumerated()), id: \.offset) { _, row in
                        Text(row["name"].text).tag(row["path"].text)
                    }
                }
                Picker("Instance", selection: $instance) {
                    Text("New independent instance").tag("new")
                    Text("Reuse one matching running instance").tag("reuse")
                }
                HStack {
                    Stepper("Lanes: \(lanes)", value: $lanes, in: 1...4)
                    TextField("Context tokens (0 = automatic)", text: $context)
                }
                TextField("Single goal", text: $goal, axis: .vertical)
                Toggle("Fork the selected conversation into the new agent", isOn: $inherit)
                DisclosureGroup("Child tools (none by default)") { ChildToolSelectionView(selection: $childTools) }
                DisclosureGroup("Optional LoRA") {
                    TextField("Local LoRA directory", text: $loraPath)
                    TextField("Adapter serving name", text: $loraName)
                    Text("LoRA requires an admitted ordinary route and a verified adapter. Different quants are separate model artifacts in the picker.").font(.caption).foregroundStyle(.secondary)
                }
                Button("Launch Agent") { Task { await launch() } }
                    .disabled(launching || model.busy || path.isEmpty || goal.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                ForEach(Array(jobs.enumerated()), id: \.offset) { _, job in
                    VStack(alignment: .leading, spacing: 5) {
                        HStack {
                            Text(job["state"].text).font(.headline)
                            Spacer()
                            if ["queued", "loading", "dispatching", "running"].contains(job["state"].text) {
                                Button("Cancel") { model.perform("agent_launch_cancel", ["job_id": job["id"].text]) }
                            }
                        }
                        Text(job["checkpoint"].text).font(.caption.monospaced()).textSelection(.enabled)
                        Text(job["goal"].text).font(.caption)
                        if let error = job["error"].string { Text(error).foregroundStyle(.orange).font(.caption) }
                        if let profile = job["profile_id"].string { Text("Instance: " + profile).font(.caption.monospaced()) }
                    }.padding(8).background(.quaternary, in: RoundedRectangle(cornerRadius: 6))
                }
            }.padding(8)
        }.onChange(of: model.snapshot["user_profiles"]["active_id"].text) { _, _ in childTools = [] }
    }
    @MainActor private func launch() async {
        guard let row = model.rows.first(where: { $0["path"].string == path }), let contextValue = Int(context), contextValue >= 0 else {
            model.error = "Select a current model and a nonnegative context limit."; return
        }
        launching = true
        defer { launching = false }
        var arguments: [String: Any] = ["path": path, "identity": row["identity"].text, "goal": goal,
            "tools": childTools.sorted(), "instance": instance, "lanes": lanes, "context": contextValue, "max_tokens": 2048,
            "parent_id": model.chat.selected.isEmpty ? "ui" : model.chat.selected]
        if !loraPath.isEmpty { arguments["lora_path"] = loraPath; arguments["lora_name"] = loraName }
        if inherit { arguments["_history"] = model.chat.completedForkHistory(for: model.chat.selected) }
        do { _ = try await model.request("agent_launch", arguments); await model.refresh() }
        catch { model.error = error.localizedDescription }
    }
}
