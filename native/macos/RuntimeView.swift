import AppKit
import SwiftUI

// Operator actions stay bound to an exact owned process. Research trial IDs are
// receipts, never entries in the chat model selector.
struct RuntimeView: View {
    @EnvironmentObject var model: DesktopModel
    @State private var profile = "default"
    @State private var details = JSON()
    @State private var operation = JSON()
    @State private var loading = false
    @State private var working = false
    @State private var error: String?
    @State private var generation = UUID()
    @State private var adapterName = ""
    @State private var adapterPath = ""
    @State private var modules = ""
    @State private var settings = "[]"
    @State private var calls = "[]"
    @State private var tool = ""
    @State private var fields: [String: String] = [:]

    private var worker: JSON {
        model.workers.first { ($0["profile_id"].string ?? "default") == profile } ?? JSON()
    }
    private var targetIdentity: String {
        model.snapshot["user_profiles"]["active_id"].text + "|" + profile + "|" + worker["url"].text
            + "|" + String(worker["pid"].double ?? 0) + "|" + worker["checkpoint"].text
    }
    private var ready: Bool { ["Ready", "Busy"].contains(worker["state"].text) && worker["execution_kind"].text == "autoregressive" }
    private var target: [String: Any] { ["profile_id": profile, "url": worker["url"].text, "pid": worker["pid"].raw] }
    private var tools: [JSON] { details["weight_space"]["tools"].array }
    private var schema: JSON { tools.first { $0["name"].text == tool }?["inputSchema"] ?? JSON() }
    private var requiredFieldsFilled: Bool {
        schema["required"].array.compactMap(\.string).allSatisfy {
            !(fields[$0] ?? "").trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        }
    }
    private var fieldNames: [String] {
        let required = Set(schema["required"].array.compactMap(\.string))
        return (schema["properties"].raw as? [String: Any] ?? [:]).keys.sorted {
            required.contains($0) == required.contains($1) ? $0 < $1 : required.contains($0)
        }
    }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 20) {
                HStack {
                    VStack(alignment: .leading, spacing: 4) {
                        Text("Runtime").font(.largeTitle.bold())
                        Text("Manage adapters and inspect live runtime changes.").foregroundStyle(.secondary)
                    }
                    Spacer()
                    Button("Refresh", systemImage: "arrow.clockwise") { Task { await refresh() } }
                        .disabled(!ready || loading || working)
                }
                Picker("Managed engine", selection: $profile) {
                    Text("Default engine").tag("default")
                    ForEach(model.preferences["profiles"].array, id: \.self["id"].text) { entry in
                        Text(entry["name"].string ?? entry["path"].string.map { URL(fileURLWithPath: $0).lastPathComponent } ?? entry["id"].text).tag(entry["id"].text)
                    }
                }.frame(maxWidth: 560)
                if loading { ProgressView("Reading engine capabilities…") }
                if working { ProgressView("Runtime operation in progress…") }
                if let error { Label(error, systemImage: "exclamationmark.triangle").foregroundStyle(.orange).textSelection(.enabled) }
                if !ready {
                    ContentUnavailableView("Select a ready managed engine", systemImage: "server.rack",
                        description: Text("Start an autoregressive model in Engines. Runtime controls follow that exact process."))
                    Button("Open Engines") { model.section = "Engines" }
                } else if !(details.raw is NSNull) {
                    HStack {
                        Label(details["status"]["model"].text, systemImage: "cube")
                        Spacer()
                        Text(details["status"]["qualification"].string ?? "Qualification: see route receipt")
                            .font(.caption).foregroundStyle(.secondary)
                    }
                    adapters
                    kernels
                    weightSpace
                    DisclosureGroup("Current route and capability receipt") { receipt(details["status"]) }
                }
                if !(operation.raw is NSNull) {
                    let presentation = RuntimeOperationPresentation(operation.raw as? [String: Any] ?? [:])
                    GroupBox("Last runtime operation") {
                        VStack(alignment: .leading, spacing: 10) {
                            if operation["pending"].bool && working { ProgressView("Request in progress…") }
                            else if operation["pending"].bool {
                                Text("Last observed request: pending. Current outcome is unconfirmed.")
                                Button("Check request status") {
                                    if let identity = operation["id"].string { Task { await poll(identity) } }
                                }
                            } else { Text(presentation.requestLabel).font(.headline) }
                            if let state = presentation.runtimeState {
                                LabeledContent("Runtime outcome", value: state.replacingOccurrences(of: "_", with: " "))
                                    .textSelection(.enabled).help("Exact reported state: " + state)
                            }
                            ForEach(presentation.errors, id: \.self) { message in Text(message).foregroundStyle(.orange).textSelection(.enabled) }
                            if let guidance = operation["guidance"].string { Text(guidance).font(.callout) }
                            if operation["state"].text == "submitted" {
                                Text("The repair trial is still running. Use Get repair attempt with its attempt ID to read the outcome.")
                                Button("Check repair attempt") {
                                    tool = "get_repair_attempt"
                                    fields = ["attempt_id": operation["receipt"]["id"].string ?? operation["receipt"]["attempt_id"].string ?? ""]
                                }
                            }
                            DisclosureGroup("Operation receipt") { receipt(operation) }
                        }.frame(maxWidth: .infinity, alignment: .leading).padding(8)
                    }
                }
            }.padding(24).frame(maxWidth: 1000, alignment: .leading).frame(maxWidth: .infinity)
        }
        .task(id: targetIdentity + "|" + String(ready)) {
            generation = UUID(); details = JSON(); operation = JSON(); error = nil
            working = false; loading = false; fields = [:]; tool = ""
            if ready { await refresh() }
            if let identity = model.runtimeOperations[targetIdentity] { await poll(identity) }
        }
        .onDisappear { generation = UUID() }
    }

    private var adapters: some View {
        GroupBox("LoRA adapters") {
            VStack(alignment: .leading, spacing: 12) {
                if details["status"]["multi_lora"]["enabled"].bool {
                    Text("Load an adapter from the configured folder, then select it in Chat. Base weights remain resident.")
                        .font(.callout).foregroundStyle(.secondary)
                    ForEach(details["status"]["multi_lora"]["registered"].array.compactMap(\.string), id: \.self) { name in
                        HStack {
                            Text(name).textSelection(.enabled)
                            Spacer()
                            Button("Unload") { submit("unload_lora", ["lora_name": name]) }.disabled(working || loading)
                        }
                    }
                    LabeledContent("Adapter name") {
                        TextField("Name shown in Chat", text: $adapterName).textFieldStyle(.roundedBorder).accessibilityLabel("Adapter name")
                    }
                    LabeledContent("Adapter folder") {
                        HStack {
                            TextField("Path inside the configured adapter directory", text: $adapterPath).textFieldStyle(.roundedBorder).accessibilityLabel("Adapter folder")
                            Button("Choose…") { chooseAdapter() }
                        }
                    }
                    Button("Load adapter") { submit("load_lora", ["lora_name": adapterName.trimmingCharacters(in: .whitespacesAndNewlines), "lora_path": adapterPath]) }
                        .disabled(working || loading || adapterName.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty || adapterPath.isEmpty)
                } else {
                    Text("Enable LoRA in this engine’s launch options, choose an adapter folder, and use ordinary decode. Restart the engine to apply launch changes.")
                        .foregroundStyle(.secondary)
                    Button("Open launch options") { model.section = profile == "default" ? "Settings" : "Engines" }
                }
            }.padding(8).frame(maxWidth: .infinity, alignment: .leading)
        }
    }

    private var kernels: some View {
        GroupBox("Kernel development") {
            VStack(alignment: .leading, spacing: 12) {
                if details["kernels"]["enabled"].bool {
                    Text("Reload computational code or change a live variant without reloading weights. Applying a change drains requests and clears APCv2 state.")
                        .font(.callout).foregroundStyle(.secondary)
                    LabeledContent("Modules") {
                        TextField("Module names, separated by commas", text: $modules).textFieldStyle(.roundedBorder).accessibilityLabel("Kernel modules")
                    }
                    DisclosureGroup("Advanced variant settings") {
                        jsonEditor("Settings (JSON array)", text: $settings)
                        jsonEditor("Setter calls (JSON array)", text: $calls)
                    }
                    Button("Apply runtime change") {
                        do {
                            let names = modules.split(whereSeparator: { $0 == "," || $0.isWhitespace }).map(String.init)
                            let settingsValue = try parse(settings), callsValue = try parse(calls)
                            guard let settingsList = settingsValue as? [Any], let callsList = callsValue as? [Any] else {
                                throw DesktopError.message("Settings and setter calls must be JSON arrays")
                            }
                            guard !names.isEmpty || !settingsList.isEmpty || !callsList.isEmpty else {
                                throw DesktopError.message("Enter at least one module, setting, or setter call")
                            }
                            submit("reload_kernels", ["modules": names, "settings": settingsList, "calls": callsList])
                        } catch { self.error = error.localizedDescription }
                    }.disabled(working || loading)
                    DisclosureGroup("Kernel status and last reload") { receipt(details["kernels"]) }
                } else {
                    Text("Disabled. Set a kernel development root in launch options to enable trusted local code changes. Changes remain unqualified until tested.")
                        .foregroundStyle(.secondary)
                }
            }.padding(8).frame(maxWidth: .infinity, alignment: .leading)
        }
    }

    private var weightSpace: some View {
        GroupBox("Weight-space experiments") {
            VStack(alignment: .leading, spacing: 12) {
                if tools.isEmpty {
                    Text("Disabled. Select an operator configuration file in launch options to enable repair, overlay, or model-surgery tools for this checkpoint.")
                        .foregroundStyle(.secondary)
                } else {
                    Text("Experimental · active changes apply to this running engine. Inspect receipts before promotion or reuse; available operations and limits come from its operator configuration.")
                        .font(.callout).foregroundStyle(.secondary)
                    Picker("Operation", selection: Binding(get: { tool }, set: { tool = $0; fields = [:] })) {
                        Text("Choose an operation").tag("")
                        ForEach(tools, id: \.self["name"].text) { item in
                            Text(label(item["name"].text)).tag(item["name"].text)
                        }
                    }
                    ForEach(fieldNames, id: \.self) { name in
                        let property = schema["properties"][name]
                        let propertySchema = property.raw as? [String: Any] ?? [:]
                        let propertyTypes = RuntimeArgumentValue.types(in: propertySchema)
                        let required = schema["required"].array.compactMap(\.string).contains(name)
                        let binding = Binding(get: { fields[name] ?? "" }, set: { fields[name] = $0 })
                        let title = label(name) + (required ? " (required)" : " (optional)")
                        if !property["enum"].array.isEmpty {
                            Picker(title, selection: binding) {
                                Text("Choose…").tag("")
                                ForEach(property["enum"].array.compactMap(\.string), id: \.self) { Text($0).tag($0) }
                            }
                        } else if propertyTypes == ["boolean"] {
                            Picker(title, selection: binding) {
                                Text("Use configured default").tag("")
                                Text("Yes").tag("true")
                                Text("No").tag("false")
                            }
                        } else if propertyTypes.contains(where: { ["object", "array"].contains($0) }) {
                            jsonEditor(title + " · " + RuntimeArgumentValue.typeHint(propertySchema), text: binding)
                        } else {
                            LabeledContent(title) {
                                TextField(RuntimeArgumentValue.typeHint(propertySchema), text: binding)
                                    .textFieldStyle(.roundedBorder).accessibilityLabel(title)
                                    .help(RuntimeArgumentValue.typeHint(propertySchema) + ". Leave optional fields blank to use the configured default.")
                            }
                        }
                    }
                    HStack {
                        Button("Run operation") {
                            do { submit("weight_space", ["tool": tool, "arguments": try arguments()]) }
                            catch { self.error = error.localizedDescription }
                        }.disabled(working || loading || tool.isEmpty || !requiredFieldsFilled)
                        Spacer()
                        if details["weight_space"]["runtime_trials_enabled"].bool { Text("Runtime trials enabled").font(.caption) }
                    }
                    DisclosureGroup("Configured capabilities, captures and programs") { receipt(details["weight_space"]) }
                }
            }.padding(8).frame(maxWidth: .infinity, alignment: .leading)
        }
    }

    private func label(_ name: String) -> String { name.replacingOccurrences(of: "_", with: " ").capitalized }
    private func parse(_ text: String) throws -> Any { try JSONSerialization.jsonObject(with: Data(text.utf8), options: [.fragmentsAllowed]) }
    private func arguments() throws -> [String: Any] {
        var result: [String: Any] = [:]
        for name in fieldNames {
            let text = (fields[name] ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
            if text.isEmpty {
                if schema["required"].array.compactMap(\.string).contains(name) { throw DesktopError.message(label(name) + " is required") }
                continue
            }
            result[name] = try RuntimeArgumentValue.decode(text,
                schema: schema["properties"][name].raw as? [String: Any] ?? [:], name: label(name))
        }
        return result
    }
    private func jsonEditor(_ title: String, text: Binding<String>) -> some View {
        VStack(alignment: .leading) {
            Text(title).font(.caption).foregroundStyle(.secondary)
            TextEditor(text: text).font(.system(.body, design: .monospaced)).frame(height: 90)
                .overlay(RoundedRectangle(cornerRadius: 5).stroke(.secondary.opacity(0.25))).accessibilityLabel(title)
        }
    }
    private func receipt(_ value: JSON) -> some View {
        let text = (try? JSONSerialization.data(withJSONObject: value.raw, options: [.prettyPrinted, .sortedKeys, .fragmentsAllowed]))
            .flatMap { String(data: $0, encoding: .utf8) } ?? "No receipt"
        return ScrollView {
            Text(text).font(.system(.caption, design: .monospaced)).textSelection(.enabled)
                .frame(maxWidth: .infinity, alignment: .leading).padding(8)
        }.frame(maxHeight: 340)
    }
    private func chooseAdapter() {
        let panel = NSOpenPanel(); panel.canChooseDirectories = true; panel.canChooseFiles = false
        if panel.runModal() == .OK, let url = panel.url {
            adapterPath = url.path
            if adapterName.isEmpty { adapterName = url.lastPathComponent }
        }
    }
    private func refresh() async {
        guard ready, !loading else { return }
        let captured = generation, selectedTarget = targetIdentity
        loading = true
        do {
            let result = try await model.request("runtime_status", target)
            guard captured == generation, selectedTarget == targetIdentity, !Task.isCancelled else { return }
            details = result; error = nil
        } catch { if captured == generation { self.error = error.localizedDescription } }
        if captured == generation { loading = false }
    }
    private func submit(_ action: String, _ body: [String: Any]) {
        guard ready, !working, !loading else { return }
        let captured = generation, selectedTarget = targetIdentity
        let request = target.merging(["action": action, "body": body]) { _, new in new }
        working = true; error = nil
        Task {
            do {
                let result = try await model.request("runtime_action", request)
                if let id = result["id"].string { model.runtimeOperations[selectedTarget] = id }
                guard captured == generation, selectedTarget == targetIdentity else { return }
                operation = result
                if let id = result["id"].string { await poll(id) }
            } catch { if captured == generation { self.error = error.localizedDescription; working = false } }
        }
    }
    private func poll(_ identity: String) async {
        let captured = generation, selectedTarget = targetIdentity
        let expectedURL = worker["url"].string, expectedPID = worker["pid"].double, expectedCheckpoint = worker["checkpoint"].string
        working = true
        var completedRead = false
        do {
            while !Task.isCancelled, captured == generation {
                let result = try await model.request("runtime_operation", ["operation_id": identity])
                guard !Task.isCancelled, captured == generation, selectedTarget == targetIdentity else { return }
                guard result["url"].string == expectedURL, result["pid"].double == expectedPID,
                      result["checkpoint"].string == expectedCheckpoint else {
                    throw DesktopError.message("This receipt belongs to another engine process. Refresh Runtime for the current engine.")
                }
                operation = result
                if !result["pending"].bool { completedRead = true; break }
                try await Task.sleep(for: .seconds(1))
            }
        } catch { if !Task.isCancelled, captured == generation { self.error = error.localizedDescription } }
        if captured == generation {
            working = false
            // A successful status refresh must not erase a failed operation read.
            // Checking status again never resubmits the original mutation.
            if completedRead { await refresh() }
        }
    }
}
