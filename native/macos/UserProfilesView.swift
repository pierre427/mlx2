import AppKit
import SwiftUI
import UniformTypeIdentifiers

struct UserProfilesView: View {
    @EnvironmentObject var model: DesktopModel
    @State private var name = ""
    @State private var note = ""
    private var state: JSON { model.snapshot["user_profiles"] }
    private var active: String { state["active_id"].text }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                Text("User Profiles").font(.title2)
                Text("Named settings, prompts and worlds save automatically. User profiles are separate from model serving profiles.").foregroundStyle(.secondary)
                HStack {
                    TextField("New profile name", text: $name)
                    Button("Create from Current Settings") {
                        model.perform("user_profile_create", ["name": name]); name = ""
                    }.disabled(name.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty || model.busy)
                }
                ForEach(Array(state["profiles"].array.enumerated()), id: \.offset) { _, row in
                    HStack {
                        VStack(alignment: .leading) {
                            Text(row["name"].text + (row["id"].text == active ? " · Active" : ""))
                            if row["imported"].bool { Text("Imported · execution permissions require re-enabling").font(.caption).foregroundStyle(.secondary)
                                DisclosureGroup("Requested settings") { Text(String(describing: row["requested_import"].raw)).font(.caption.monospaced()).textSelection(.enabled) } }
                        }
                        Spacer()
                        Button("Rename") { rename(row) }
                        Button("Export…") { export(row) }
                        Button("Switch") { model.perform("user_profile_switch", ["profile_id": row["id"].text]) }.disabled(row["id"].text == active)
                        Button("Archive") { model.perform("user_profile_delete", ["profile_id": row["id"].text]) }.disabled(row["id"].text == active)
                    }.padding(10).background(.quaternary, in: RoundedRectangle(cornerRadius: 8))
                }
                Button("Import User Profile…") { importProfile() }
                Text("Portable exports include authored prompts and world content. Review that text for personal information before sharing. Credentials, local paths, connection definitions and private chat history are excluded. Imported profiles cannot automatically start models or enable sharing, helpers, MCP, Dream or autonomous scene changes.").font(.caption).foregroundStyle(.secondary)
                if !state["archived"].array.isEmpty {
                    Text("Archived profiles").font(.headline)
                    ForEach(Array(state["archived"].array.enumerated()), id: \.offset) { _, row in
                        HStack { Text(row.text).font(.caption.monospaced()); Button("Restore") { model.perform("user_profile_restore", ["profile_id": row.text]) } }
                    }
                }
                if !(state["error"].string ?? "").isEmpty { Text(state["error"].text).foregroundStyle(.orange) }
                if !note.isEmpty { Text(note).font(.caption) }
            }.padding(20)
        }
    }

    private func rename(_ row: JSON) {
        let alert = NSAlert(); alert.messageText = "Rename User Profile"
        let field = NSTextField(string: row["name"].text); field.frame = NSRect(x: 0, y: 0, width: 300, height: 24)
        alert.accessoryView = field; alert.addButton(withTitle: "Save"); alert.addButton(withTitle: "Cancel")
        if alert.runModal() == .alertFirstButtonReturn { model.perform("user_profile_rename", ["profile_id": row["id"].text, "name": field.stringValue]) }
    }

    private func export(_ row: JSON) {
        let panel = NSSavePanel(); panel.allowedContentTypes = [.json]; panel.nameFieldStringValue = "MLX2 User Profile.json"
        guard panel.runModal() == .OK, let url = panel.url else { return }
        Task {
            do {
                let result = try await model.request("user_profile_export", ["profile_id": row["id"].text])
                let data = try JSONSerialization.data(withJSONObject: result["document"].raw, options: [.prettyPrinted, .sortedKeys])
                try data.write(to: url, options: [.atomic]); note = "Exported \(row["name"].text)"
            } catch { model.error = error.localizedDescription }
        }
    }

    private func importProfile() {
        let panel = NSOpenPanel(); panel.allowedContentTypes = [.json]; panel.allowsMultipleSelection = false
        guard panel.runModal() == .OK, let url = panel.url else { return }
        Task {
            do {
                let size = try url.resourceValues(forKeys: [.fileSizeKey]).fileSize ?? 0
                guard size <= 16 * 1024 * 1024 else { throw NSError(domain: "MLX2", code: 1, userInfo: [NSLocalizedDescriptionKey: "Profile exceeds 16 MiB"]) }
                let document = try JSONSerialization.jsonObject(with: Data(contentsOf: url))
                _ = try await model.request("user_profile_import", ["document": document]); await model.refresh()
                note = "Imported profile saved. Select Switch to apply it."
            } catch { model.error = error.localizedDescription }
        }
    }
}
