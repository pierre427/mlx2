import SwiftUI

struct AgentSessionSelection: Identifiable {
    let jobID: String
    let workerID: String
    var id: String { jobID + ":" + workerID }
}

enum AgentSessionSupport {
    static func active(_ status: String) -> Bool { ["queued", "running", "synthesizing", "preparing", "loading"].contains(status) }
    static func text(_ value: JSON) -> String {
        if let text = value.string { return text }
        guard !(value.raw is NSNull), let data = try? JSONSerialization.data(withJSONObject: value.raw, options: [.sortedKeys, .fragmentsAllowed]) else { return "" }
        return String(decoding: data, as: UTF8.self)
    }
    static func model(_ job: JSON) -> String {
        job["target_identity"]["model"].string ?? job["target_identity"]["model_path"].string.map { URL(fileURLWithPath: $0).lastPathComponent } ?? "Model identity unavailable"
    }
}

struct ChatAgentsButton: View {
    @EnvironmentObject var model: DesktopModel
    let parentID: String
    @State private var presented = false
    @State private var showAll = false
    @State private var session: AgentSessionSelection?
    private var workers: [(job: JSON, worker: JSON)] {
        model.snapshot["agents"]["jobs"].array
            .filter { showAll || $0["parent_id"].string == parentID }
            .flatMap { job in job["workers"].array.map { (job: job, worker: $0) } }
            .sorted { lhs, rhs in
                let a = AgentSessionSupport.active(lhs.worker["status"].text), b = AgentSessionSupport.active(rhs.worker["status"].text)
                if a != b { return a }
                let aTime = lhs.job["created"].double ?? 0, bTime = rhs.job["created"].double ?? 0
                return aTime == bTime ? lhs.worker["id"].text < rhs.worker["id"].text : aTime > bTime
            }
    }
    private var ownWorkers: [JSON] { model.snapshot["agents"]["jobs"].array.filter { $0["parent_id"].string == parentID }.flatMap { $0["workers"].array } }
    private var summary: String {
        guard !ownWorkers.isEmpty else { return "Subagents" }
        let active = ownWorkers.filter { AgentSessionSupport.active($0["status"].text) }.count
        return "Subagents · \(active) active · \(ownWorkers.count - active) finished"
    }
    var body: some View {
        Button { presented.toggle() } label: { Label(summary, systemImage: "person.3.sequence") }
            .help("\(ownWorkers.filter { AgentSessionSupport.active($0["status"].text) }.count) active; \(ownWorkers.filter { !AgentSessionSupport.active($0["status"].text) }.count) finished")
            .popover(isPresented: $presented) {
                VStack(alignment: .leading, spacing: 10) {
                    Text("Agent sessions").font(.headline)
                    Toggle("All agents in this user profile", isOn: $showAll)
                    if workers.isEmpty { Text(showAll ? "No agent sessions in this profile." : "No subagents for this conversation.").foregroundStyle(.secondary).padding(.vertical) }
                    ScrollView {
                        LazyVStack(alignment: .leading, spacing: 8) {
                            ForEach(Array(workers.enumerated()), id: \.offset) { _, row in
                                Button {
                                    presented = false
                                    session = AgentSessionSelection(jobID: row.job["id"].text, workerID: row.worker["id"].text)
                                } label: {
                                    VStack(alignment: .leading, spacing: 4) {
                                        HStack { Text(row.worker["goal"].text).lineLimit(2); Spacer(); Text(row.worker["status"].text).font(.caption).foregroundStyle(AgentSessionSupport.active(row.worker["status"].text) ? .blue : .secondary) }
                                        Text(AgentSessionSupport.model(row.job)).font(.caption).foregroundStyle(.secondary).lineLimit(1)
                                    }.frame(maxWidth: .infinity, alignment: .leading).padding(8).contentShape(Rectangle())
                                }.buttonStyle(.plain)
                                Divider()
                            }
                        }
                    }.frame(maxHeight: 350)
                }.padding(14).frame(width: 430)
            }
            .sheet(item: $session) { item in AgentSessionView(jobID: item.jobID, workerID: item.workerID) }
            .onChange(of: model.snapshot["user_profiles"]["active_id"].text) { _, _ in presented = false; session = nil; showAll = false }
    }
}

struct AgentSessionView: View {
    @EnvironmentObject var model: DesktopModel
    @Environment(\.dismiss) private var dismiss
    let jobID: String
    let workerID: String
    @State private var document = JSON()
    @State private var error: String?
    @State private var loading = false
    @State private var offset = 0
    @State private var followLatest = true
    @State private var requestID = UUID()
    private let pageSize = 50
    private var profile: String { model.snapshot["user_profiles"]["active_id"].text }
    private var active: Bool { AgentSessionSupport.active(document["worker_status"].text) }

    @MainActor private func load(_ requestedOffset: Int) async {
        let expectedProfile = profile, token = UUID()
        requestID = token; loading = true
        do {
            let result = try await model.request("agent_session", ["job_id": jobID, "worker_id": workerID, "offset": requestedOffset, "limit": pageSize])
            guard !Task.isCancelled, profile == expectedProfile, requestID == token else { return }
            guard result["job_id"].string == jobID, result["worker_id"].string == workerID else { throw DesktopError.message("Agent session identity changed.") }
            document = result; offset = requestedOffset; error = nil; loading = false
            let total = Int(result["total"].double ?? 0)
            if followLatest && AgentSessionSupport.active(result["worker_status"].text) && total > requestedOffset + pageSize {
                await load(max(0, total - pageSize))
            }
        } catch {
            guard !Task.isCancelled, profile == expectedProfile, requestID == token else { return }
            self.error = error.localizedDescription; loading = false
        }
    }

    @ViewBuilder private func message(_ row: JSON) -> some View {
        VStack(alignment: .leading, spacing: 7) {
            Text(row["role"].text.capitalized).font(.caption.weight(.semibold)).foregroundStyle(.secondary)
            if row["role"].string == "tool" {
                DisclosureGroup("Tool result · " + (row["tool_call_id"].string ?? "recorded")) {
                    Text(AgentSessionSupport.text(row["content"])).font(.system(.caption, design: .monospaced)).textSelection(.enabled)
                }
            } else {
                let content = AgentSessionSupport.text(row["content"])
                if !content.isEmpty { ChatMarkdownView(text: content) }
                if !row["tool_calls"].array.isEmpty {
                    DisclosureGroup("Tool calls (\(row["tool_calls"].array.count))") {
                        Text(AgentSessionSupport.text(row["tool_calls"])).font(.system(.caption, design: .monospaced)).textSelection(.enabled)
                    }
                }
            }
            if row["truncated"].bool { Text("This message is truncated in the bounded reader.").font(.caption).foregroundStyle(.orange) }
        }.frame(maxWidth: .infinity, alignment: .leading).padding(10).background(.quaternary, in: RoundedRectangle(cornerRadius: 10))
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text("Agent Session").font(.title2)
                Spacer()
                if loading { ProgressView().controlSize(.small) }
                Button("Close") { dismiss() }.keyboardShortcut(.cancelAction)
            }
            Text(document["goal"].string ?? "Loading worker goal…").font(.headline).textSelection(.enabled)
            HStack { Text(document["worker_status"].string ?? "Loading"); Text(document["model"].string ?? "").foregroundStyle(.secondary); Spacer(); Toggle("Follow latest", isOn: $followLatest).toggleStyle(.checkbox) }
            DisclosureGroup("Session identity") {
                VStack(alignment: .leading, spacing: 4) {
                    Text("Job: " + jobID); Text("Worker: " + workerID)
                    Text("Parent: " + document["parent_id"].text)
                    Text("Checkpoint: " + AgentSessionSupport.text(document["checkpoint"]))
                }.font(.caption.monospaced()).textSelection(.enabled)
            }
            Text("Read-only worker transcript. Parent context is not displayed.").font(.caption).foregroundStyle(.secondary)
            if let error { Text(error).foregroundStyle(.orange).textSelection(.enabled) }
            if let workerError = document["error"].string { Text(workerError).foregroundStyle(.orange).textSelection(.enabled) }
            if document["truncated"].bool { Text("The retained transcript is bounded; some content was truncated.").font(.caption).foregroundStyle(.orange) }
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 10) {
                    ForEach(Array(document["messages"].array.enumerated()), id: \.offset) { _, row in message(row) }
                    if active && !document["messages"].array.contains(where: { $0["role"].string == "assistant" && !AgentSessionSupport.text($0["content"]).isEmpty }) {
                        HStack { ProgressView().controlSize(.small); Text("Working — waiting for the next recorded response.").foregroundStyle(.secondary) }.padding(8)
                    }
                    if document["messages"].array.isEmpty { Text(active ? "Working — no response has been recorded yet." : "No recorded messages.").foregroundStyle(.secondary).padding() }
                }
            }
            HStack {
                Button("Previous") { followLatest = false; Task { await load(max(0, offset - pageSize)) } }.disabled(loading || offset == 0)
                Text("Messages \(document["messages"].array.isEmpty ? 0 : offset + 1)–\(offset + document["messages"].array.count) of \(Int(document["total"].double ?? 0))").font(.caption).foregroundStyle(.secondary)
                Button("Next") { if let next = document["next_offset"].double { followLatest = false; Task { await load(Int(next)) } } }.disabled(loading || document["next_offset"].double == nil)
                Spacer()
                Button("Refresh") { Task { await load(offset) } }.disabled(loading)
            }
        }.padding(20).frame(minWidth: 620, idealWidth: 820, minHeight: 500, idealHeight: 680)
            .task {
                let initialProfile = profile
                await load(0)
                while !Task.isCancelled && profile == initialProfile && active {
                    do { try await Task.sleep(nanoseconds: 2_000_000_000) } catch { return }
                    guard !Task.isCancelled, profile == initialProfile else { return }
                    if !loading { await load(offset) }
                }
            }
            .onChange(of: profile) { _, _ in requestID = UUID(); document = JSON(); error = nil; dismiss() }
            .onChange(of: followLatest) { _, enabled in if enabled { Task { await load(offset) } } }
    }
}
