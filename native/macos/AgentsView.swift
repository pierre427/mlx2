import SwiftUI

struct AgentsView: View {
    @EnvironmentObject var model: DesktopModel
    @State private var selected = ""
    @State private var goals = ""
    @State private var synthesisGoal = "Combine the worker reports. Preserve disagreements, evidence limits and uncertainties."
    @State private var concurrency = 1
    @State private var tokens = 2048
    @State private var childTools = Set<String>()
    @State private var agentSession: AgentSessionSelection?
    @State private var cancelJob: JSON?
    private var goalLines: [String] { goals.components(separatedBy: "\n").map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }.filter { !$0.isEmpty } }
    private var runBlocker: String? {
        if servers.isEmpty { return "Start a model from Model Library first; agents run on a running engine." }
        if selected.isEmpty { return "Choose a running engine." }
        if goalLines.isEmpty { return "Enter at least one worker goal." }
        if goalLines.count > 16 { return "Enter at most 16 worker goals (\(goalLines.count) entered)." }
        return nil
    }
    private var servers: [JSON] { model.servers }
    private var jobs: [JSON] { model.snapshot["agents"]["jobs"].array }
    private func start() {
        guard let server = servers.first(where: { ChatModel.identity($0) == selected }) else { return }
        let endpoint = server["endpoint"]
        let target: [String: Any]
        if endpoint["remote"].bool {
            target = ["remote": true, "nid": endpoint["nid"].text, "model": endpoint["remote_model"].text]
        } else {
            target = ["url": endpoint["url"].text, "pid": endpoint["pid"].raw]
        }
        model.perform("agent_delegate", ["target": target,
            "goals": goalLines,
            "tools": childTools.sorted(), "concurrency": concurrency, "max_tokens": tokens, "synthesis_goal": synthesisGoal])
    }
    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 14) {
                Text("Independent Agents").font(.title2)
                AgentLaunchView()
                Text("Assign one goal per worker. Each gets its own conversation on the selected engine; the orchestrator receives their reports and produces a separate synthesis. Workers receive only the checked tools and may request additional tools. Scoped memory, messaging and skill access remain subject to application policy. UI mutations and external MCPs are unavailable to workers.").foregroundStyle(.secondary)
                Picker("Running engine", selection: $selected) {
                    Text("Choose engine").tag("")
                    ForEach(Array(servers.enumerated()), id: \.offset) { _, row in Text(ChatModel.label(row) + " · " + row["endpoint"]["url"].text).tag(ChatModel.identity(row)) }
                }
                DisclosureGroup("Child tools (none by default)") { ChildToolSelectionView(selection: $childTools) }
                Stepper("Concurrent workers: \(concurrency)", value: $concurrency, in: 1...4)
                Text("Requests run concurrently within the engine's configured serving capacity. Physical batching depends on its lane settings.").font(.caption).foregroundStyle(.secondary)
                Stepper("Token limit per worker and synthesis: \(tokens)", value: $tokens, in: 128...8192, step: 128)
                Text("Worker goals — one per line (up to 16)").font(.headline)
                TextEditor(text: $goals).frame(height: 130).border(.separator).accessibilityLabel("Worker goals, one per line")
                TextField("Orchestrator synthesis goal", text: $synthesisGoal, axis: .vertical)
                Button("Run Agents", action: start).disabled(model.busy || runBlocker != nil).help(runBlocker ?? "Start one worker per goal on the selected engine.")
                if let runBlocker { Text(runBlocker).font(.caption).foregroundStyle(.secondary) }
                ForEach(Array(jobs.enumerated()), id: \.offset) { _, job in
                    VStack(alignment: .leading, spacing: 8) {
                        HStack {
                            Text("Task \(job["id"].text.prefix(8)) · \(job["status"].text.capitalized)").font(.headline).help("Task " + job["id"].text)
                            Spacer()
                            if ["queued", "running", "synthesizing"].contains(job["status"].text) { Button("Cancel Task…") { cancelJob = job }.disabled(model.busy) }
                        }
                        ForEach(Array(job["workers"].array.enumerated()), id: \.offset) { _, worker in
                            VStack(alignment: .leading, spacing: 6) {
                                HStack {
                                    Text("\(worker["status"].text.capitalized) · \(worker["goal"].text)").lineLimit(2).help(worker["goal"].text)
                                    Spacer()
                                    Button("Open Session") { agentSession = AgentSessionSelection(jobID: job["id"].text, workerID: worker["id"].text) }
                                }
                                DisclosureGroup("Report") {
                                    Text(worker["report"]["content"].string ?? worker["error"].string ?? "Waiting for report").textSelection(.enabled)
                                }
                            }
                        }
                        AgentJobControls(job: job).id(job["id"].text)
                        if let synthesis = job["synthesis"].string { Text("Orchestrator synthesis").font(.headline); Text(synthesis).textSelection(.enabled) }
                        if let error = job["error"].string { Text(error).foregroundStyle(.orange).textSelection(.enabled) }
                    }.padding(12).background(.quaternary, in: RoundedRectangle(cornerRadius: 8))
                }
            }.padding(20)
        }.confirmationDialog("Cancel this agent task?", isPresented: Binding(get: { cancelJob != nil }, set: { if !$0 { cancelJob = nil } }), presenting: cancelJob) { job in
            Button("Cancel Task", role: .destructive) { model.perform("agent_cancel", ["job_id": job["id"].text]); cancelJob = nil }
            Button("Keep Running", role: .cancel) { cancelJob = nil }
        } message: { _ in Text("Queued and running workers in this task stop before finishing their reports.") }
        .sheet(item: $agentSession) { item in AgentSessionView(jobID: item.jobID, workerID: item.workerID) }
            .onChange(of: model.snapshot["user_profiles"]["active_id"].text) { _, _ in childTools = []; agentSession = nil }
    }
}
