import AppKit
import SwiftUI
import UniformTypeIdentifiers

enum SkillGuidance {
    static let prompt = "For reusable work, discover relevant procedural guidance with mlx2__skills_list and read only relevant mlx2__skill_view pages. Save tested or corrected reusable procedures with mlx2__skill_manage as policy permits; keep facts in memory instead. Record provenance, checks actually performed and uncertainty. Generated skills remain unverified guidance and never grant tool permissions."
}

struct SkillEditorDocument {
    let files: [String: String]
    init(body: String, referenceJSON: String) throws {
        guard body.utf8.count <= 64 * 1024 else { throw DesktopError.message("Skill body exceeds 64 KiB.") }
        guard referenceJSON.utf8.count <= 4 * 1024 * 1024,
              let data = referenceJSON.data(using: .utf8),
              let files = try JSONSerialization.jsonObject(with: data) as? [String: String],
              files.count <= 64, files.values.reduce(0, { $0 + $1.utf8.count }) <= 512 * 1024 else {
            throw DesktopError.message("Reference files must be a JSON object of at most 64 text files and 512 KiB of content.")
        }
        self.files = files
    }
}

struct SkillsPolicySettings: View {
    @EnvironmentObject var model: DesktopModel
    private func flag(_ key: String) -> Binding<Bool> {
        Binding(get: { model.snapshot["skills"]["policy"][key].bool }, set: { model.perform("skill_policy", ["profile_id": model.snapshot["user_profiles"]["active_id"].text, "changes": [key: $0]]) })
    }
    var body: some View {
        GroupBox("Skill library") {
            VStack(alignment: .leading, spacing: 8) {
                Toggle("Expose enabled skills to models", isOn: flag("enabled"))
                Toggle("Allow model skill changes without asking each time", isOn: flag("autonomous_writes"))
                Text("Skills contain reusable instructions and reference text. They do not grant tool permissions or execute files. Imported skills remain disabled until reviewed.").font(.caption).foregroundStyle(.secondary)
            }.frame(maxWidth: .infinity, alignment: .leading).padding(6)
        }.disabled(model.busy)
    }
}

struct SkillsView: View {
    @EnvironmentObject var model: DesktopModel
    @State private var selected = ""
    @State private var name = ""
    @State private var description = ""
    @State private var bodyText = ""
    @State private var references = "{}"
    @State private var enabled = false
    @State private var revision = 0
    @State private var working = false
    @State private var loadEpoch = UUID()
    @State private var page: JSON?
    @State private var offset = 0
    private var profileID: String { model.snapshot["user_profiles"]["active_id"].string ?? "" }
    private var catalog: JSON { page ?? model.snapshot["skills"] }
    private var rows: [JSON] { catalog["skills"].array }
    private var archived: Bool { rows.first { $0["name"].string == selected }?["deleted"].bool ?? false }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 14) {
                HStack { Text("Skills").font(.title2); Spacer(); Button("New Skill") { newSkill() }; Button("Import…") { importSkill() } }
                SkillsPolicySettings()
                Text("Save reusable model instructions in a private, revisioned library. Reference files stay text; this library has no execution tool.").foregroundStyle(.secondary)
                ForEach(Array(rows.enumerated()), id: \.offset) { _, row in
                    HStack {
                        Button { Task { await load(row) } } label: {
                            VStack(alignment: .leading) { Text(row["name"].text).fontWeight(.semibold); Text(row["description"].text).font(.caption).lineLimit(2) }
                                .frame(maxWidth: .infinity, alignment: .leading)
                        }.buttonStyle(.plain)
                        Text("r\(Int(row["revision"].double ?? 0)) · \(row["deleted"].bool ? "Archived" : row["enabled"].bool ? "Enabled" : "Disabled")").font(.caption).foregroundStyle(.secondary)
                    }.padding(10).background(selected == row["name"].text ? Color.accentColor.opacity(0.12) : Color.primary.opacity(0.035), in: RoundedRectangle(cornerRadius: 7))
                }
                HStack {
                    Button("Previous") { Task { await loadPage(max(0, offset - 100)) } }.disabled(offset == 0)
                    Button("Next") { Task { await loadPage(offset + 100) } }.disabled(catalog["next_offset"].double == nil)
                    Button("Refresh List") { Task { await loadPage(offset) } }
                    Text("\(offset + (rows.isEmpty ? 0 : 1))–\(offset + rows.count) of \(Int(catalog["total"].double ?? 0))").font(.caption).foregroundStyle(.secondary)
                }
                Divider()
                Text(selected.isEmpty ? "Create a skill" : "\(selected) · revision \(revision)").font(.headline)
                TextField("Skill name", text: $name).disabled(!selected.isEmpty)
                TextField("Description", text: $description)
                Toggle("Enabled", isOn: $enabled)
                Text("Instructions").font(.caption)
                TextEditor(text: $bodyText).font(.system(.body, design: .monospaced)).frame(minHeight: 220).border(.separator)
                DisclosureGroup("Reference files (JSON text map)") {
                    Text("Example: {\"references/notes.md\": \"Reference text\"}").font(.caption).foregroundStyle(.secondary)
                    TextEditor(text: $references).font(.system(.caption, design: .monospaced)).frame(minHeight: 120).border(.separator)
                }
                HStack {
                    Button(selected.isEmpty ? "Create Skill" : "Save Revision") { Task { await save() } }.disabled(archived || name.isEmpty)
                    if !selected.isEmpty {
                        Button("Reload") { if let row = rows.first(where: { $0["name"].string == selected }) { Task { await load(row) } } }
                        Button("Export…") { exportSkill() }
                        Button(archived ? "Restore" : "Archive") { Task { await mutate(archived ? "restore" : "archive") } }
                    }
                    if working { ProgressView().controlSize(.small) }
                }
                Text("Exports include authored instructions and reference text. Review that content for private information before sharing.").font(.caption).foregroundStyle(.secondary)
            }.padding(20).disabled(working || model.busy)
        }.onChange(of: profileID) { _, _ in newSkill(); page = nil; offset = 0; working = false }
    }

    @MainActor private func loadPage(_ target: Int) async {
        let profile = profileID, epoch = UUID()
        loadEpoch = epoch; working = true
        defer { if loadEpoch == epoch { working = false } }
        do {
            let result = try await model.request("skills_list", ["profile_id": profile, "includeall": true, "offset": target, "limit": 100])
            guard profileID == profile, loadEpoch == epoch else { return }
            offset = target; page = result
        } catch { if profileID == profile, loadEpoch == epoch { model.error = error.localizedDescription } }
    }

    private func updatePage(_ saved: JSON) {
        guard var value = page?.raw as? [String: Any], var records = value["skills"] as? [[String: Any]],
              let index = records.firstIndex(where: { $0["name"] as? String == saved["name"].string }),
              let record = saved.raw as? [String: Any] else { return }
        records[index] = record; value["skills"] = records; page = JSON(value)
    }

    private func newSkill() {
        loadEpoch = UUID(); selected = ""; name = ""; description = ""; bodyText = ""; references = "{}"; enabled = false; revision = 0
    }

    @MainActor private func load(_ row: JSON) async {
        let epoch = UUID(); loadEpoch = epoch; working = true
        let profile = profileID
        defer { if loadEpoch == epoch { working = false } }
        do {
            let result = try await model.request("skill_export", ["profile_id": profile, "name": row["name"].text])
            guard loadEpoch == epoch, profileID == profile else { return }
            let doc = result["document"]
            selected = row["name"].text; name = selected; description = doc["description"].string ?? ""; bodyText = doc["body"].string ?? ""
            enabled = doc["enabled"].bool; revision = Int(row["revision"].double ?? 0)
            let data = try JSONSerialization.data(withJSONObject: doc["files"].raw is NSNull ? [:] : doc["files"].raw, options: [.prettyPrinted, .sortedKeys])
            references = String(decoding: data, as: UTF8.self)
        } catch { if loadEpoch == epoch { model.error = error.localizedDescription } }
    }

    @MainActor private func save() async {
        let profile = profileID, epoch = UUID()
        loadEpoch = epoch
        working = true; defer { if loadEpoch == epoch { working = false } }
        do {
            let document = try SkillEditorDocument(body: bodyText, referenceJSON: references)
            let saved = try await model.request("skill_manage", ["profile_id": profile, "action": selected.isEmpty ? "create" : "update", "name": name, "expected_revision": revision,
                "description": description, "body": bodyText, "files": document.files, "enabled": enabled])
            guard profile == profileID, loadEpoch == epoch else { return }
            selected = name; revision = Int(saved["revision"].double ?? Double(revision)); updatePage(saved)
            await model.refresh()
            guard profile == profileID, loadEpoch == epoch else { return }
        } catch { model.error = error.localizedDescription }
    }

    @MainActor private func mutate(_ action: String) async {
        let profile = profileID, epoch = UUID()
        loadEpoch = epoch
        working = true; defer { if loadEpoch == epoch { working = false } }
        do {
            let saved = try await model.request("skill_manage", ["profile_id": profile, "action": action, "name": selected, "expected_revision": revision])
            guard profile == profileID, loadEpoch == epoch else { return }
            revision = Int(saved["revision"].double ?? Double(revision)); enabled = saved["enabled"].bool; updatePage(saved)
            await model.refresh()
        } catch { model.error = error.localizedDescription }
    }

    private func exportSkill() {
        let profile = profileID, skillName = selected
        let panel = NSSavePanel(); panel.allowedContentTypes = [.json]; panel.nameFieldStringValue = name + ".json"
        guard panel.runModal() == .OK, let url = panel.url else { return }
        Task { do {
            let result = try await model.request("skill_export", ["profile_id": profile, "name": skillName])
            guard profileID == profile else { throw DesktopError.message("User profile changed; export cancelled.") }
            try JSONSerialization.data(withJSONObject: result["document"].raw, options: [.prettyPrinted, .sortedKeys]).write(to: url, options: .atomic)
        } catch { model.error = error.localizedDescription } }
    }

    private func importSkill() {
        let profile = profileID
        let panel = NSOpenPanel(); panel.allowedContentTypes = [.json]; panel.allowsMultipleSelection = false
        guard panel.runModal() == .OK, let url = panel.url else { return }
        Task { do {
            guard (try url.resourceValues(forKeys: [.fileSizeKey]).fileSize ?? Int.max) <= 4 * 1024 * 1024 else { throw DesktopError.message("Skill import exceeds 4 MiB.") }
            _ = try await model.request("skill_import", ["profile_id": profile, "document": try JSONSerialization.jsonObject(with: Data(contentsOf: url))])
            await model.refresh()
        } catch { model.error = error.localizedDescription } }
    }
}
