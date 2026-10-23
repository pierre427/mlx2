import SwiftUI

struct PromptSettingsView: View {
    @EnvironmentObject var model: DesktopModel
    @State private var selected = ""
    @State private var name = ""
    @State private var text = ""
    @State private var preset = "full"
    @State private var importingPreset = false
    @State private var presetCatalog: [JSON] = []
    private var prompts: [JSON] { model.preferences["system_prompts"].array }

    private func load() {
        let prompt = prompts.first { $0["id"].text == selected }
        name = prompt?["name"].string ?? ""
        text = prompt?["text"].string ?? ""
    }

    var body: some View {
        Section("System prompts") {
            HStack {
                Picker("MLX2 agent preset", selection: $preset) {
                    Text("Full · about 10k tokens").tag("full")
                    Text("Compact · smaller context windows").tag("compact")
                }
                Button(importingPreset ? "Importing…" : "Use Preset as Base") {
                    importingPreset = true
                    Task { @MainActor in
                        defer { importingPreset = false }
                        do {
                            let result = try await model.request("prompt_install", ["preset": preset, "activate": true])
                            await model.refresh()
                            selected = result["prompt_id"].text
                            load()
                        } catch { model.error = error.localizedDescription }
                    }
                }.disabled(importingPreset)
            }.task {
                if let result = try? await model.request("prompt_presets") { presetCatalog = result["presets"].array }
            }
            if let row = presetCatalog.first(where: { $0["preset"].text == preset }) {
                Text("Bundled original instructions · \(Int(row["estimated_tokens"].double ?? 0).formatted()) estimated tokens. Exact usage depends on the model. Imports an editable copy and preserves your other prompts.")
                    .font(.caption).foregroundStyle(.secondary)
            }
            Picker("Base system prompt", selection: Binding(get: { model.preferences["base_system_prompt_id"].string ?? "" }, set: { model.savePreferences(["base_system_prompt_id": $0]) })) {
                Text("Model default").tag("")
                ForEach(Array(prompts.enumerated()), id: \.offset) { _, prompt in Text(prompt["name"].text).tag(prompt["id"].text) }
            }
            Text("Choose extra instructions from the Chat panel's dropdown. They are appended to the base system prompt. Each user turn also carries its local date, time and time zone, with guidance to use that context only when relevant.")
                .font(.caption).foregroundStyle(.secondary)
            Picker("Edit saved prompt", selection: $selected) {
                Text("New prompt").tag("")
                ForEach(Array(prompts.enumerated()), id: \.offset) { _, prompt in Text(prompt["name"].text).tag(prompt["id"].text) }
            }.onChange(of: selected) { _, _ in load() }
            TextField("Prompt name", text: $name)
            TextEditor(text: $text).frame(minHeight: 120).font(.body)
                .overlay(RoundedRectangle(cornerRadius: 5).stroke(.separator))
            HStack {
                Button("Save Prompt") {
                    let identity = selected.isEmpty ? UUID().uuidString.lowercased() : selected
                    var rows = prompts.filter { $0["id"].text != identity }.map { $0.raw as? [String: Any] ?? [:] }
                    rows.append(["id": identity, "name": name, "text": text])
                    model.savePreferences(["system_prompts": rows])
                    selected = identity
                }.disabled(name.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                if !selected.isEmpty {
                    Button("Delete Prompt", role: .destructive) {
                        var panels = model.preferences["panel_system_prompt_ids"].raw as? [String: String] ?? [:]
                        panels = panels.filter { $0.value != selected }
                        model.savePreferences(["system_prompts": prompts.filter { $0["id"].text != selected }.map(\.raw),
                                               "base_system_prompt_id": model.preferences["base_system_prompt_id"].text == selected ? "" : model.preferences["base_system_prompt_id"].text,
                                               "panel_system_prompt_ids": panels])
                        selected = ""; name = ""; text = ""
                    }
                }
            }
        }
        Section("Built-in agent tools") {
            Toggle("Enable built-in planning and task records", isOn: Binding(get: { model.preferences["built_in_planning_enabled"].bool }, set: { model.savePreferences(["built_in_planning_enabled": $0]) }))
            Toggle("Enable local worker delegation", isOn: Binding(get: { model.preferences["agent_delegation_enabled"].bool }, set: { model.savePreferences(["agent_delegation_enabled": $0]) }))
            Text("Planning is part of MLX2 and does not require a CPG MCP connection. Each task has an explicit session, immutable goal, fenced ownership and a retained result. Worker claims remain separate from human verification.")
                .font(.caption).foregroundStyle(.secondary)
        }
    }
}
