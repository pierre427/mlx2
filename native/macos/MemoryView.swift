import SwiftUI

struct MemoryView: View {
    @EnvironmentObject var model: DesktopModel
    @State private var layout = "Notes"
    @State private var query = ""
    @State private var scope = "project:mlx2"
    @State private var kind = "note"
    @State private var title = ""
    @State private var bodyText = ""
    @State private var tags = ""
    @State private var sources = ""
    @State private var records: [JSON] = []
    @State private var detail = JSON()
    @State private var reason = ""
    @State private var targetID = ""
    @State private var relation = "related"
    @State private var localError: String?
    @State private var session = ""
    @State private var recipient = ""
    @State private var message = ""
    @State private var messages: [JSON] = []
    private var memory: JSON { model.snapshot["memory"] }
    private var kinds: [String] { ["note", "wiki", "forum", "tip", "procedure", "lesson"] }
    private var displayed: [JSON] {
        let values = records.isEmpty && query.isEmpty ? memory["recent"].array : records
        return values.filter { (scope.isEmpty || $0["scope"].text == scope) && (layout == "Notes" || $0["kind"].text == kind) }
    }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                HStack {
                    Text("Agent Memory").font(.largeTitle.weight(.semibold))
                    Spacer()
                    Text(memory["enabled"].bool ? "Enabled" : "Disabled").foregroundStyle(.secondary)
                }
                Text("Atomic notes, procedures, tips and discussion use one revisioned store. Captures start as candidates; review and promotion remain explicit. Retrieved text is reference material with provenance.").foregroundStyle(.secondary)
                HStack {
                    ForEach(["candidate", "reviewed", "promoted"], id: \.self) { status in
                        Text("\(status.capitalized): \(Int(memory["counts"][status].double ?? 0))").font(.caption)
                    }
                }
                Picker("View", selection: $layout) {
                    ForEach(["Notes", "Wiki", "Forum", "Messages"], id: \.self) { Text($0).tag($0) }
                }.pickerStyle(.segmented).onChange(of: layout) { _, value in
                    kind = value == "Wiki" ? "wiki" : value == "Forum" ? "forum" : "note"
                    records = []
                    Task { await search() }
                }
                if layout == "Messages" { messagePanel } else { notePanels }
                if let localError { Label(localError, systemImage: "exclamationmark.triangle").foregroundStyle(.orange) }
                Text(memory["retrieval"].text).font(.caption).foregroundStyle(.secondary)
                CapsulePolicyView()
                Text("Retrieval ceiling: \(Int(memory["context_tokens"].double ?? 1024)) tokens, at most \(Int(memory["context_percent"].double ?? 10))% of the trusted context window. UTF-8 byte bounds conservatively limit snippets; no corpus is inserted into chat.").font(.caption).foregroundStyle(.secondary)
            }.padding(24)
        }
    }
    private var notePanels: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack {
                TextField("Search relevant memories", text: $query).onSubmit { Task { await search() } }
                Button("Search") { Task { await search() } }
            }
            TextField("Scope · project:mlx2, machine:hostname, session:id, community:id", text: $scope)
            GroupBox("Records") {
                VStack(alignment: .leading, spacing: 10) {
                    if displayed.isEmpty { Text("No records in this view.").foregroundStyle(.secondary) }
                    ForEach(Array(displayed.enumerated()), id: \.offset) { _, note in
                        Button {
                            Task { await load(note["id"].text) }
                        } label: {
                            HStack {
                                VStack(alignment: .leading, spacing: 3) {
                                    Text(note["title"].text).font(.headline)
                                    Text("\(note["kind"].text) · \(note["scope"].text) · revision \(Int(note["revision"].double ?? 1))").font(.caption).foregroundStyle(.secondary)
                                }
                                Spacer()
                                Text(note["status"].text.capitalized).font(.caption)
                            }.contentShape(Rectangle())
                        }.buttonStyle(.plain)
                        Divider()
                    }
                }.padding(8)
            }
            GroupBox(detail["id"].string == nil ? "Capture a candidate" : "Record · \(detail["id"].text)") {
                VStack(alignment: .leading, spacing: 12) {
                    Picker("Kind", selection: $kind) { ForEach(kinds, id: \.self) { Text($0.capitalized).tag($0) } }
                    TextField("Title", text: $title)
                    TextEditor(text: $bodyText).frame(minHeight: 160)
                        .overlay(RoundedRectangle(cornerRadius: 5).stroke(.quaternary))
                    TextField("Tags (comma separated)", text: $tags)
                    TextField("Source pointers (one per line)", text: $sources, axis: .vertical).lineLimit(2...5)
                    if let captured = detail["created"].double {
                        Text("Captured \(Date(timeIntervalSince1970: captured).formatted()) · \(detail["created_by"].text)").font(.caption).foregroundStyle(.secondary)
                    }
                    if let revised = detail["revised_at"].double {
                        Text("Revision \(Int(detail["revision"].double ?? 1)) · \(Date(timeIntervalSince1970: revised).formatted()) · \(detail["status"].text)").font(.caption).foregroundStyle(.secondary)
                    }
                    TextField("Why this revision / review / feedback matters", text: $reason)
                    HStack {
                        Button("New note") { clear() }
                        if detail["id"].string != nil {
                            Button("Append candidate revision") { Task { await save() } }
                            Menu("Review") {
                                Button("Mark reviewed") { Task { await review("reviewed") } }
                                Button("Promote") { Task { await review("promoted") } }
                                Button("Return to candidate") { Task { await review("candidate") } }
                            }
                            Menu("Feedback") {
                                Button("Useful") { feedback(true) }
                                Button("Needs improvement") { feedback(false) }
                            }
                        } else {
                            Button("Capture") { Task { await save() } }.buttonStyle(.borderedProminent)
                        }
                        Spacer()
                    }
                    if detail["id"].string != nil { historyAndLinks }
                }.padding(8)
            }
        }
    }
    private var historyAndLinks: some View {
        VStack(alignment: .leading, spacing: 12) {
            DisclosureGroup("Immutable revision history") {
                ForEach(Array(detail["history"].array.enumerated()), id: \.offset) { _, version in
                    HStack {
                        Text("Revision \(Int(version["revision"].double ?? 0)) · \(version["status"].text)")
                        Text(version["reason"].text).foregroundStyle(.secondary)
                        Spacer()
                        Button("Read") { Task { await load(detail["id"].text, revision: Int(version["revision"].double ?? 1)) } }
                    }.font(.caption)
                }
                Text("Editing an old revision cannot overwrite a newer one. Load the current version before appending.").font(.caption).foregroundStyle(.secondary)
                Button("Load current revision") { Task { await load(detail["id"].text) } }
            }
            DisclosureGroup("Typed connections") {
                ForEach(Array(detail["links"].array.enumerated()), id: \.offset) { _, link in
                    Text("\(link["source_id"].text)@\(Int(link["source_revision"].double ?? 0)) → \(link["relation"].text) → \(link["target_id"].text)@\(Int(link["target_revision"].double ?? 0)) · \(link["status"].text)")
                        .font(.caption.monospaced()).textSelection(.enabled)
                }
                HStack {
                    TextField("Target note ID", text: $targetID)
                    TextField("Relation (supports, contradicts, supersedes, related…)", text: $relation)
                    Button("Link") { Task { await addLink() } }
                }
                Text("Connections are revision-bound. Similarity does not establish a factual relationship.").font(.caption).foregroundStyle(.secondary)
            }
        }
    }
    private var messagePanel: some View {
        GroupBox("Explicit worker messages") {
            VStack(alignment: .leading, spacing: 12) {
                Text("Agent access is bound to a specific session and explicit sibling recipients by the harness. A message is data; it grants no tool or execution permission.").font(.caption).foregroundStyle(.secondary)
                TextField("Session ID", text: $session)
                TextField("Recipient worker ID", text: $recipient)
                TextEditor(text: $message).frame(minHeight: 80)
                HStack {
                    Button("Read mailbox") { Task { await readMessages() } }
                    Spacer()
                    Button("Post") { Task { await postMessage() } }.disabled(session.isEmpty || recipient.isEmpty || message.isEmpty)
                }
                ForEach(Array(messages.enumerated()), id: \.offset) { _, entry in
                    VStack(alignment: .leading, spacing: 5) {
                        Text("\(entry["sender"].text) → \(entry["recipient"].text)").font(.caption.monospaced())
                        Text(entry["body"].text).textSelection(.enabled)
                        HStack {
                            Text(entry["acknowledged_at"].double == nil ? "Awaiting acknowledgement" : "Acknowledged").font(.caption).foregroundStyle(.secondary)
                            Button("Acknowledge") { model.perform("message_ack", ["session_id": session, "id": entry["id"].text]) }
                        }
                    }
                    Divider()
                }
            }.padding(8)
        }
    }
    private func invoke(_ command: String, _ args: [String: Any]) async -> JSON? {
        do { localError = nil; return try await model.request(command, args) }
        catch { localError = error.localizedDescription; return nil }
    }
    private func search() async {
        guard layout != "Messages" else { return }
        var args: [String: Any] = ["query": query, "limit": 50, "states": ["candidate", "reviewed", "promoted"]]
        if !scope.isEmpty { args["scope"] = scope }
        if layout == "Wiki" || layout == "Forum" { args["kind"] = kind }
        if let result = await invoke("memory_search", args) { records = result["records"].array }
    }
    private func load(_ id: String, revision: Int? = nil) async {
        var args: [String: Any] = ["id": id]
        if let revision { args["revision"] = revision }
        if let result = await invoke("memory_get", args) {
            detail = result; title = result["title"].text; bodyText = result["body"].text
            kind = result["kind"].text; scope = result["scope"].text
            tags = result["tags"].array.map(\.text).joined(separator: ", ")
            sources = result["sources"].array.map(\.text).joined(separator: "\n")
            reason = ""
        }
    }
    private func clear() {
        detail = JSON(); title = ""; bodyText = ""; tags = ""; sources = ""; reason = ""
    }
    private func save() async {
        var args: [String: Any] = ["title": title, "body": bodyText,
            "tags": tags.split(separator: ",").map { $0.trimmingCharacters(in: .whitespaces) }.filter { !$0.isEmpty },
            "sources": sources.split(separator: "\n").map(String.init)]
        let updating = detail["id"].string != nil
        if updating {
            guard !reason.isEmpty else { localError = "Describe why this revision helps."; return }
            args["id"] = detail["id"].text; args["revision"] = Int(detail["revision"].double ?? 1); args["reason"] = reason
        } else { args["scope"] = scope; args["kind"] = kind }
        if let result = await invoke(updating ? "memory_refine" : "memory_add", args) { await load(result["id"].text); await model.refresh(); await search() }
    }
    private func review(_ status: String) async {
        guard !reason.isEmpty else { localError = "Record the review reason before changing trust state."; return }
        if let result = await invoke("memory_review", ["id": detail["id"].text, "revision": Int(detail["revision"].double ?? 1), "status": status, "reason": reason]) { await load(result["id"].text); await model.refresh(); await search() }
    }
    private func feedback(_ useful: Bool) {
        guard !reason.isEmpty else { localError = "Describe what helped or needs improvement."; return }
        model.perform("memory_useful", ["id": detail["id"].text, "revision": Int(detail["revision"].double ?? 1), "useful": useful, "reason": reason])
    }
    private func addLink() async {
        guard !reason.isEmpty else { localError = "Explain the connection in the reason field."; return }
        _ = await invoke("memory_link", ["source_id": detail["id"].text, "source_revision": Int(detail["revision"].double ?? 1), "target_id": targetID, "relation": relation, "rationale": reason, "accepted": true])
        await load(detail["id"].text)
    }
    private func readMessages() async {
        if let result = await invoke("message_read", ["session_id": session, "limit": 50]) { messages = result["messages"].array }
    }
    private func postMessage() async {
        if await invoke("message_post", ["session_id": session, "recipient": recipient, "body": message]) != nil { message = ""; await readMessages() }
    }
}
