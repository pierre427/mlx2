import AppKit
import SwiftUI

struct QualificationView: View {
    @ObservedObject var model: DesktopModel
    @State private var selected = Set<String>()
    @State private var level1 = true
    @State private var level2 = true
    @State private var level3 = true
    @State private var route = "auto"
    @State private var context = "32768"
    @State private var lanes = "4"
    @State private var cache = "4"
    @State private var features = ""
    @State private var showPreflight = false
    @State private var performanceSort = "decode_tps"
    @State private var comparisonGroup = ""
    @State private var details: String?

    private var campaign: JSON { model.snapshot["campaign"] }
    private var jobs: [JSON] { model.snapshot["jobs"].array.filter { $0["kind"].text == "qualification" } }
    private var levels: [Int] { [(1, level1), (2, level2), (3, level3)].filter { $0.1 }.map { $0.0 } }
    private var comparisons: [JSON] { campaign["comparisons"].array }
    private var groups: [String] { Array(Set(comparisons.compactMap { $0["group"].string })).sorted() }
    private var currentGroup: String { groups.contains(comparisonGroup) ? comparisonGroup : groups.first ?? "" }
    private var sortedComparisons: [JSON] {
        comparisons.filter { $0["group"].text == currentGroup }.sorted { ($0[performanceSort].double ?? 0) > ($1[performanceSort].double ?? 0) }
    }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                Text("Qualification").font(.title2.bold())
                Text("Records bind the exact model fingerprint, code checkpoint, runtime, host and serving profile. Qualified means all three levels passed together. The serving runtime retains its separate qualification receipt rules.")
                    .font(.callout).foregroundStyle(.secondary)
                GroupBox("Select models • runs serially, smallest first") {
                    VStack(alignment: .leading, spacing: 6) {
                        HStack {
                            Button("Select serving candidates") { selected = Set(model.rows.filter { $0["can_start"].bool }.map { $0["path"].text }) }
                            Button("Clear") { selected = [] }
                            Spacer()
                            Text("\(selected.count) selected").foregroundStyle(.secondary)
                        }
                        if model.rows.isEmpty {
                            Text(model.library["scanning"].bool ? "Scanning the model library…" : "No models in this library. Choose a folder in Model Library or download a model first.")
                                .foregroundStyle(.secondary).padding(.vertical, 8)
                        }
                        ForEach(model.rows.filter { $0["kind"].text != "Draft" }, id: \.rawPath) { row in
                            HStack {
                                Toggle(isOn: Binding(get: { selected.contains(row["path"].text) }, set: { if $0 { selected.insert(row["path"].text) } else { selected.remove(row["path"].text) } })) {
                                    Text(row["name"].text).lineLimit(1)
                                }.toggleStyle(.checkbox)
                                Spacer()
                                Text(row["qualification"]["label"].text).font(.caption).foregroundStyle(.secondary)
                            }
                        }
                    }.padding(5)
                }
                GroupBox("Test profile") {
                    VStack(alignment: .leading, spacing: 10) {
                        Toggle("Level 1 — loads and generates coherent output", isOn: $level1)
                        Toggle("Level 2 — functionality, advanced features and replicated cache/context checks", isOn: $level2)
                        Toggle("Level 3 — controlled prefill and decode measurements", isOn: $level3)
                        HStack {
                            Picker("Route", selection: $route) {
                                Text("Adapter default").tag("auto")
                                Text("Ordinary reference").tag("ordinary")
                                Text("Native MTP").tag("native_mtp")
                                Text("Prompt lookup").tag("prompt_lookup")
                            }.frame(maxWidth: 260)
                            TextField("Context", text: $context).frame(width: 90)
                            TextField("Lanes", text: $lanes).frame(width: 60)
                            TextField("Cache GiB", text: $cache).frame(width: 85)
                        }
                        TextField("Additional required feature IDs, separated by commas", text: $features)
                        Text("Level 1 asks you to review the retained ordinary output. Level 2 uses the approved qualifier; missing companions or unsupported profiles remain unqualified. Level 3 requires three clean nominal-thermal measurements per cell. Existing model servers must be stopped before queuing.")
                            .font(.caption).foregroundStyle(.secondary)
                        HStack {
                            Button(campaign["running"].bool ? "Campaign running" : "Run selected models") {
                                if level2 { showPreflight = true } else { enqueue(approved: false) }
                            }.disabled(selected.isEmpty || levels.isEmpty || campaign["running"].bool || model.busy)
                            if campaign["running"].bool {
                                ProgressView().controlSize(.small)
                                Button("Cancel queue") { model.perform("campaign_cancel") }
                            }
                            Spacer()
                            Button("Save test profile") { model.perform("campaign_configure", ["settings": ["profile": profile]]) }
                        }
                    }.padding(5)
                }
                if !jobs.isEmpty {
                    GroupBox("Campaign history and output review") {
                        VStack(alignment: .leading, spacing: 10) {
                            ForEach(jobs, id: \.jobID) { job in
                                VStack(alignment: .leading, spacing: 4) {
                                    HStack {
                                        Text(URL(fileURLWithPath: job["spec"]["path"].text).lastPathComponent).bold()
                                        Spacer()
                                        Text(job["state"].text).foregroundStyle(.secondary)
                                        Button("Details") { details = pretty(job) }
                                    }
                                    Text(job["progress"].text).font(.caption).foregroundStyle(.secondary)
                                    if job["result"]["review_pending"].bool {
                                        Text(job["result"]["ordinary_output"]["text"].text).textSelection(.enabled)
                                        HStack {
                                            Button("Output is coherent") { model.perform("campaign_review", ["job_id": job["id"].text, "coherent": true]) }
                                            Button("Reject output") { model.perform("campaign_review", ["job_id": job["id"].text, "coherent": false]) }
                                        }
                                    }
                                }
                                Divider()
                            }
                        }.padding(5)
                    }
                }
                GroupBox("Compare performance • identical host, code, runtime and shape") {
                    VStack(alignment: .leading, spacing: 10) {
                        if comparisons.isEmpty {
                            Text("No controlled performance records yet.").foregroundStyle(.secondary)
                        } else {
                            HStack {
                                Picker("Comparison shape", selection: $comparisonGroup) {
                                    ForEach(groups, id: \.self) { group in Text(groupLabel(group)).tag(group) }
                                }
                                Picker("Sort", selection: $performanceSort) {
                                    Text("Decode").tag("decode_tps")
                                    Text("Prefill").tag("prefill_tps")
                                }.frame(width: 150)
                            }
                            HStack { Text("Model").bold(); Spacer(); Text("Prefill tok/s").frame(width: 120); Text("Decode tok/s").frame(width: 120) }
                            ForEach(Array(sortedComparisons.enumerated()), id: \.offset) { _, row in
                                HStack {
                                    Text(row["checkpoint"].text).lineLimit(1).help(row["checkpoint"].text)
                                    Spacer()
                                    Text(String(format: "%.1f", row["prefill_tps"].double ?? 0)).monospacedDigit().frame(width: 120)
                                    Text(String(format: "%.1f", row["decode_tps"].double ?? 0)).monospacedDigit().frame(width: 120)
                                }
                            }
                        }
                    }.padding(5)
                }
            }.padding(20)
        }
        .onAppear {
            selected = model.checkedModels
            comparisonGroup = currentGroup
            let saved = campaign["settings"]["profile"]
            if let value = saved["route"].string { route = value }
            if let value = saved["context"].double { context = String(Int(value)) }
            if let value = saved["lanes"].double { lanes = String(Int(value)) }
            if let value = saved["cache_gib"].double { cache = String(Int(value)) }
            features = saved["features"].array.compactMap(\.string).joined(separator: ", ")
        }
        .onChange(of: groups) { _, value in if !value.contains(comparisonGroup) { comparisonGroup = value.first ?? "" } }
        .onChange(of: selected) { _, value in model.checkedModels = value }
        .onChange(of: model.checkedModels) { _, value in if selected != value { selected = value } }
        .alert("Approve full preflight for this frozen code snapshot?", isPresented: $showPreflight) {
            Button("Cancel", role: .cancel) {}
            Button("Approve preflight and run") { enqueue(approved: true) }
        } message: {
            Text("AGENTS.md requires your explicit approval before a full preflight. This runs the full CPU suite before GPU ownership, then the selected models serially. Existing services are never stopped automatically.")
        }
        .sheet(isPresented: Binding(get: { details != nil }, set: { if !$0 { details = nil } })) {
            VStack(alignment: .leading) {
                HStack { Text("Source-bound evidence").font(.headline); Spacer(); Button("Done") { details = nil } }
                ScrollView { Text(details ?? "").font(.system(.caption, design: .monospaced)).textSelection(.enabled).frame(maxWidth: .infinity, alignment: .leading) }
            }.padding(20).frame(minWidth: 700, minHeight: 500)
        }
    }

    private var profile: [String: Any] {
        ["route": route, "context": Int(context) ?? 0, "lanes": Int(lanes) ?? 0,
         "inflight": Int(lanes) ?? 0, "cache_gib": Int(cache) ?? 0,
         "features": features.split(separator: ",").map { $0.trimmingCharacters(in: .whitespaces) }.filter { !$0.isEmpty }]
    }
    private func enqueue(approved: Bool) {
        model.perform("campaign_enqueue", ["paths": Array(selected), "levels": levels, "profile": profile, "approve_preflight": approved])
    }
    private func groupLabel(_ group: String) -> String {
        guard let row = comparisons.first(where: { $0["group"].text == group }) else { return group }
        let shape = row["shape"]
        return "\(Int(shape["context_tokens"].double ?? 0)) tokens · width \(Int(shape["width"].double ?? 0)) · \(shape["profile"]["route"].text) · \(String(group.prefix(8)))"
    }
    private func pretty(_ value: JSON) -> String {
        guard let data = try? JSONSerialization.data(withJSONObject: value.raw, options: [.prettyPrinted, .sortedKeys]), let text = String(data: data, encoding: .utf8) else { return value.text }
        return text
    }
}

private extension JSON {
    var rawPath: String { self["path"].text }
    var jobID: String { self["id"].text }
}
