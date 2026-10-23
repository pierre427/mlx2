import AppKit
import SwiftUI

struct DreamView: View {
    @EnvironmentObject var model: DesktopModel
    @State private var enabled = false
    @State private var selectedTarget = ""
    @State private var idle = "300"
    @State private var backoff = "900"
    @State private var seconds = "180"
    @State private var context = "8192"
    @State private var output = "2048"
    @State private var dailyJobs = "12"
    @State private var dailyTokens = "16384"
    @State private var dailySeconds = "1800"
    @State private var cpu = "40"
    @State private var research = false
    @State private var style = "warm"
    @State private var preferences = ""
    @State private var goal = ""
    @State private var detail: String?
    private var state: JSON { model.snapshot["dream"] }
    private var policy: JSON { state["policy"] }
    private var targets: [JSON] {
        let local = Set(model.workers.compactMap { $0["url"].string })
        return model.servers.filter { !$0["endpoint"]["remote"].bool && local.contains($0["endpoint"]["url"].text) }
    }
    private var policyData: Data? { try? JSONSerialization.data(withJSONObject: policy.raw, options: [.sortedKeys]) }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                HStack {
                    Text("Idle Dream").font(.largeTitle.weight(.semibold))
                    Spacer()
                    Label(policy["enabled"].bool ? "Enabled" : "Disabled", systemImage: policy["enabled"].bool ? "moon.stars.fill" : "moon")
                }
                Text("Let an already-running model use idle time to research, write candidate notes and improve its own private workspace. Foreground activity takes priority. Dream never loads a model or edits files outside its enclave.").foregroundStyle(.secondary)
                GroupBox("Current activity") {
                    VStack(alignment: .leading, spacing: 8) {
                        HStack {
                            if state["running"].bool { ProgressView().controlSize(.small) }
                            Text(state["reason"].text)
                            Spacer()
                            Button("Yield now") { model.perform("dream_yield") }
                        }
                        Text("Today: \(Int(state["daily"]["jobs"].double ?? 0)) runs · \(Int(state["daily"]["tokens"].double ?? 0)) output tokens · \(Int(state["daily"]["seconds"].double ?? 0)) seconds").font(.caption).foregroundStyle(.secondary)
                        Text(state["activity_fresh"].bool ? "Native activity reporting is current." : "Waiting for a fresh native activity report before any work can run.").font(.caption)
                    }.padding(6)
                }
                GroupBox("Saved idle policy") {
                    VStack(alignment: .leading, spacing: 12) {
                        Toggle("Enable idle Dream", isOn: $enabled)
                        Picker("Already-running app-managed model", selection: $selectedTarget) {
                            Text("Choose a running model").tag("")
                            ForEach(targets, id: \.dreamServerIdentity) { server in
                                Text("\(ChatModel.label(server)) · \(server["endpoint"]["url"].text)").tag(ChatModel.identity(server))
                            }
                        }
                        if targets.isEmpty { Text("Start a model in Model Library first. Dream will wait if that model stops; it never starts one automatically.").font(.caption).foregroundStyle(.secondary) }
                        HStack {
                            number("Idle seconds", value: $idle)
                            number("Run seconds", value: $seconds)
                            number("Backoff seconds", value: $backoff)
                        }
                        HStack {
                            number("Context ceiling", value: $context)
                            number("Output tokens per run", value: $output)
                        }
                        Text("Context is also capped by the actual server limit. CPU, thermal, swap, queued GPU work and other model activity can postpone or end a run.").font(.caption).foregroundStyle(.secondary)
                        DisclosureGroup("Daily budgets and resource limits") {
                            VStack(alignment: .leading, spacing: 10) {
                                HStack { number("Runs per day", value: $dailyJobs); number("Output tokens per day", value: $dailyTokens) }
                                HStack { number("Seconds per day", value: $dailySeconds); number("Maximum CPU percent", value: $cpu) }
                            }.padding(.vertical, 8)
                        }
                        Toggle("Allow bounded Wikipedia research", isOn: $research)
                        Text(state["research"].text).font(.caption).foregroundStyle(.secondary)
                        Text("Research goal").font(.headline)
                        TextEditor(text: $goal).frame(height: 85).font(.body)
                        Picker("Social style", selection: $style) {
                            Text("Professional").tag("professional")
                            Text("Warm").tag("warm")
                            Text("Companion").tag("companion")
                        }.pickerStyle(.segmented)
                        Text("Your preferences, interests and boundaries").font(.headline)
                        TextEditor(text: $preferences).frame(height: 100).font(.body)
                        Text("The style can be friendly and personal while remaining honest about being an AI. Your preferences guide its work; private conversation history is not inserted into Dream.").font(.caption).foregroundStyle(.secondary)
                        Button("Save idle policy") { save() }.buttonStyle(.borderedProminent)
                    }.padding(6)
                }
                GroupBox("Private enclave and evidence") {
                    VStack(alignment: .leading, spacing: 10) {
                        Text(state["enclave"].text).font(.system(.caption, design: .monospaced)).textSelection(.enabled)
                        Button("Reveal enclave in Finder") { if let path = state["enclave"].string { NSWorkspace.shared.activateFileViewerSelecting([URL(fileURLWithPath: path)]) } }
                        Text("Changes stay inside this workspace. Previous file contents are archived for rollback; notes remain candidates. A model's confidence is recorded as its own assessment, not independent verification.").font(.callout).foregroundStyle(.secondary)
                        DisclosureGroup("Operating boundaries") { Text(state["loyalty"].text).font(.caption).textSelection(.enabled).padding(.vertical, 6) }
                    }.padding(6)
                }
                GroupBox("Recent runs") {
                    VStack(alignment: .leading, spacing: 8) {
                        if state["jobs"].array.isEmpty { Text("No Dream runs yet.").foregroundStyle(.secondary) }
                        ForEach(state["jobs"].array, id: \.dreamJobIdentity) { job in
                            HStack {
                                VStack(alignment: .leading, spacing: 3) {
                                    Text(job["state"].text.capitalized).bold()
                                    Text(job["checkpoint"].text).font(.caption).lineLimit(1)
                                }
                                Spacer()
                                Text("\(Int(job["charged_tokens"].double ?? 0)) tokens · \(Int(job["charged_seconds"].double ?? 0))s").font(.caption).monospacedDigit()
                                Button("Receipt") { detail = pretty(job) }
                            }
                            Divider()
                        }
                    }.padding(6)
                }
            }.padding(24)
        }
        .onAppear { load() }
        .onChange(of: policyData) { _, _ in load() }
        .sheet(isPresented: Binding(get: { detail != nil }, set: { if !$0 { detail = nil } })) {
            VStack(alignment: .leading) {
                HStack { Text("Dream activity receipt").font(.headline); Spacer(); Button("Done") { detail = nil } }
                ScrollView { Text(detail ?? "").font(.system(.caption, design: .monospaced)).textSelection(.enabled).frame(maxWidth: .infinity, alignment: .leading) }
            }.padding(20).frame(minWidth: 700, minHeight: 500)
        }
    }
    private func number(_ label: String, value: Binding<String>) -> some View {
        VStack(alignment: .leading, spacing: 3) { Text(label).font(.caption).foregroundStyle(.secondary); TextField(label, text: value) }
    }
    private func load() {
        enabled = policy["enabled"].bool
        idle = String(Int(policy["idle_seconds"].double ?? 300))
        backoff = String(Int(policy["backoff_seconds"].double ?? 900))
        seconds = String(Int(policy["max_job_seconds"].double ?? 180))
        context = String(Int(policy["max_context_tokens"].double ?? 8192))
        output = String(Int(policy["max_output_tokens"].double ?? 2048))
        dailyJobs = String(Int(policy["max_jobs_per_day"].double ?? 12))
        dailyTokens = String(Int(policy["max_tokens_per_day"].double ?? 16384))
        dailySeconds = String(Int(policy["max_seconds_per_day"].double ?? 1800))
        cpu = String(Int(policy["max_cpu_percent"].double ?? 40))
        research = policy["allow_research"].bool
        style = policy["social_style"].string ?? "warm"
        preferences = policy["operator_preferences"].string ?? ""
        goal = policy["goal"].string ?? ""
        selectedTarget = targets.first(where: { $0["endpoint"]["url"].string == policy["target"]["url"].string && $0["endpoint"]["pid"].double == policy["target"]["pid"].double }).map(ChatModel.identity) ?? ""
    }
    private func save() {
        guard let idle = Int(idle), let backoff = Int(backoff), let seconds = Int(seconds), let context = Int(context), let output = Int(output),
              let dailyJobs = Int(dailyJobs), let dailyTokens = Int(dailyTokens), let dailySeconds = Int(dailySeconds), let cpu = Int(cpu) else {
            model.error = "Dream budgets must be whole numbers."; return
        }
        var changes: [String: Any] = ["enabled": enabled, "idle_seconds": idle, "backoff_seconds": backoff,
            "max_job_seconds": seconds, "max_context_tokens": context, "max_output_tokens": output,
            "max_jobs_per_day": dailyJobs, "max_tokens_per_day": dailyTokens, "max_seconds_per_day": dailySeconds,
            "max_cpu_percent": cpu, "allow_research": research, "social_style": style, "operator_preferences": preferences, "goal": goal]
        if let target = targets.first(where: { ChatModel.identity($0) == selectedTarget }), let pid = target["endpoint"]["pid"].double {
            changes["target"] = ["url": target["endpoint"]["url"].text, "pid": Int(pid)]
        } else if enabled && policy["target"]["url"].string == nil {
            model.error = "Choose an already-running app-managed model."; return
        }
        model.perform("dream_configure", ["changes": changes])
    }
    private func pretty(_ value: JSON) -> String {
        guard let data = try? JSONSerialization.data(withJSONObject: value.raw, options: [.prettyPrinted, .sortedKeys]), let text = String(data: data, encoding: .utf8) else { return value.text }
        return text
    }
}

private extension JSON {
    var dreamServerIdentity: String { "\(self["endpoint"]["url"].text)|\(String(self["endpoint"]["pid"].double ?? 0))|\(self["endpoint"]["model_path"].text)" }
    var dreamJobIdentity: String { self["id"].text }
}
