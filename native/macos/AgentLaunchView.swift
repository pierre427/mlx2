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
    @State private var cancelJob: JSON?
    private var jobs: [JSON] { model.snapshot["agent_launches"]["jobs"].array }
    private var launchBlocker: String? {
        if path.isEmpty { return model.rows.contains(where: { $0["can_start"].bool }) ? "Choose a model." : "No startable models in the library." }
        guard let value = Int(context.trimmingCharacters(in: .whitespaces)), value >= 0 else { return "Context tokens must be a whole number, 0 or more." }
        if goal.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty { return "Enter a goal for the agent." }
        if !loraPath.isEmpty && (loraName.isEmpty || loraName.count > 128 || !loraName.allSatisfy({ $0.isLetter || $0.isNumber || $0 == "-" || $0 == "_" })) {
            return "LoRA serving name: 1–128 letters, digits, hyphens or underscores."
        }
        return nil
    }
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
                    LabeledContent("Context tokens") { TextField("0 = automatic", text: $context).frame(width: 110).accessibilityLabel("Context tokens, 0 for automatic") }
                }
                TextField("Single goal", text: $goal, axis: .vertical)
                Toggle("Fork the selected conversation into the new agent", isOn: $inherit)
                DisclosureGroup("Child tools (none by default)") { ChildToolSelectionView(selection: $childTools) }
                DisclosureGroup("Optional LoRA") {
                    LabeledContent("LoRA directory") { TextField("Local path", text: $loraPath).accessibilityLabel("Local LoRA directory") }
                    LabeledContent("Serving name") { TextField("Adapter name", text: $loraName).accessibilityLabel("Adapter serving name") }
                    Text("LoRA requires an admitted ordinary route and a verified adapter. Different quants are separate model artifacts in the picker.").font(.caption).foregroundStyle(.secondary)
                }
                HStack {
                    Button(launching ? "Launching…" : "Launch Agent") { Task { await launch() } }
                        .disabled(launching || model.busy || launchBlocker != nil)
                        .help(launchBlocker ?? "Start the agent on the selected model.")
                    if let launchBlocker { Text(launchBlocker).font(.caption).foregroundStyle(.secondary) }
                }
                ForEach(Array(jobs.enumerated()), id: \.offset) { _, job in
                    VStack(alignment: .leading, spacing: 5) {
                        HStack {
                            Text(job["state"].text.capitalized).font(.headline)
                            Spacer()
                            if ["queued", "loading", "dispatching", "running"].contains(job["state"].text) {
                                Button("Cancel Launch…") { cancelJob = job }.disabled(model.busy)
                            }
                        }
                        Text(job["checkpoint"].text).font(.caption.monospaced()).textSelection(.enabled)
                        Text(job["goal"].text).font(.caption).textSelection(.enabled)
                        if let error = job["error"].string { Text(error).foregroundStyle(.orange).font(.caption).textSelection(.enabled) }
                        if let profile = job["profile_id"].string { Text("Instance: " + profile).font(.caption.monospaced()) }
                    }.padding(8).background(.quaternary, in: RoundedRectangle(cornerRadius: 6))
                }
            }.padding(8)
        }.onChange(of: model.snapshot["user_profiles"]["active_id"].text) { _, _ in childTools = []; cancelJob = nil }
        .confirmationDialog("Cancel this agent launch?", isPresented: Binding(get: { cancelJob != nil }, set: { if !$0 { cancelJob = nil } }), presenting: cancelJob) { job in
            Button("Cancel Launch", role: .destructive) { model.perform("agent_launch_cancel", ["job_id": job["id"].text]); cancelJob = nil }
            Button("Keep Running", role: .cancel) { cancelJob = nil }
        } message: { _ in Text("The agent stops before completing its goal.") }
    }
    @MainActor private func launch() async {
        guard let row = model.rows.first(where: { $0["path"].string == path }), let contextValue = Int(context.trimmingCharacters(in: .whitespaces)), contextValue >= 0 else {
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
