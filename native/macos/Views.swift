import AppKit
import Charts
import SwiftUI

private struct SidebarGroup: Identifiable {
    let id: String
    let title: String
    let sections: [String]
}
private let sidebarGroups = [
    SidebarGroup(id: "workspace", title: "Workspace", sections: ["Chat", "Code", "World", "Memory", "Agents", "Skills", "Wakeups", "Dream"]),
    SidebarGroup(id: "models", title: "Models", sections: ["Model Library", "Engines", "Runtime", "Qualification", "Media", "Downloads", "Conversions", "Ablation"]),
    SidebarGroup(id: "monitor", title: "Monitor", sections: ["Overview", "Performance", "Requests", "Cache", "Routes", "Logs"]),
    SidebarGroup(id: "connections", title: "Connections & Settings", sections: ["Community", "Helpers", "User Profiles", "Settings"])
]
private let symbols = ["Code": "chevron.left.forwardslash.chevron.right", "Chat": "bubble.left.and.bubble.right", "Overview": "chart.bar.xaxis", "Performance": "gauge.with.dots.needle.67percent", "Requests": "list.bullet", "Cache": "externaldrive",
                       "Routes": "point.3.connected.trianglepath.dotted", "Logs": "doc.text",
                       "Model Library": "shippingbox", "Settings": "gearshape", "Qualification": "checkmark.seal",
                       "Skills": "books.vertical", "Wakeups": "alarm", "User Profiles": "person.crop.circle", "World": "cube.transparent", "Dream": "moon.stars", "Engines": "server.rack", "Runtime": "slider.horizontal.3", "Community": "network", "Media": "photo.on.rectangle.angled",
                       "Downloads": "arrow.down.circle", "Conversions": "arrow.triangle.2.circlepath", "Ablation": "slider.vertical.3", "Helpers": "person.2.badge.gearshape", "Agents": "person.3.sequence", "Memory": "brain.head.profile"]

struct StatusDot: View {
    var state: String
    var body: some View {
        HStack(spacing: 6) {
            // The label carries the state; color is a secondary cue only.
            Circle().fill(["Ready", "Busy"].contains(state) ? Color.green
                          : ["Preparing", "Loading", "Stopping", "Suspending", "Resuming"].contains(state) ? .orange
                          : ["Error", "Unreachable", "SuspendError", "ResumeError", "SuspensionRecoveryRequired"].contains(state) ? .red
                          : .secondary)
                .frame(width: 7, height: 7)
                .accessibilityHidden(true)
            Text(state)
        }
        .font(.caption)
        .accessibilityElement(children: .combine)
        .accessibilityLabel("Model state: " + state)
    }
}

/// Stopping drains accepted requests and discards suspended state, so every
/// Stop control asks first, matching the Quit confirmation.
@MainActor func confirmStopModel(_ name: String) -> Bool {
    let alert = NSAlert()
    let label = name.isEmpty || name == "—" ? "the model" : name
    alert.messageText = "Stop \(label)?"
    alert.informativeText = "Accepted requests drain for up to 15 seconds, then the engine stops. Suspended state is discarded."
    alert.alertStyle = .warning
    alert.addButton(withTitle: "Stop Model")
    alert.addButton(withTitle: "Cancel")
    return alert.runModal() == .alertFirstButtonReturn
}

struct PickerView: View {
    @EnvironmentObject var model: DesktopModel
    @Environment(\.openWindow) private var openWindow

    private func open(_ section: String) {
        model.section = section
        NSApp.setActivationPolicy(.regular)
        openWindow(id: "details")
        NSApp.activate(ignoringOtherApps: true)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack {
                Text("MLX2").font(.headline)
                Spacer()
                if model.busy { ProgressView().controlSize(.small) }
                StatusDot(state: model.state)
            }
            if model.managedRunning {
                HStack(alignment: .top) {
                    Image(systemName: "cube").font(.title2).foregroundStyle(.secondary)
                    VStack(alignment: .leading, spacing: 4) {
                        Text(model.activeWorker["name"].text).fontWeight(.semibold).lineLimit(2)
                        Text(model.activeWorker["url"].text).font(.caption).foregroundStyle(.secondary)
                    }
                    Spacer()
                    if ["Ready", "Busy"].contains(model.activeWorker["state"].text) { Button("Suspend") { model.suspendWorker(model.activeWorker) }.disabled(model.busy) }
                    Button("Stop") { let worker = model.activeWorker; if confirmStopModel(worker["name"].text) { model.stopWorker(worker) } }.disabled(model.busy)
                }
                HStack {
                    metric("Generation", number(model.server["generation_tps"].double, suffix: " tok/s"))
                    Spacer()
                    metric("Model memory", bytes(model.server["footprint_bytes"].double))
                    Spacer()
                    metric("Requests", "\(number(model.server["running"].double, digits: 0)) / \(number(model.server["waiting"].double, digits: 0)) queued")
                }
            } else {
                Text(model.servers.isEmpty ? "Choose a local model to start serving." : "External servers are available in Details.")
                    .font(.callout).foregroundStyle(.secondary)
            }
            ForEach(Array(model.suspendedWorkers.enumerated()), id: \.offset) { _, worker in
                HStack { Text(worker["name"].text).lineLimit(1); Text("Suspended").font(.caption).foregroundStyle(.secondary); Spacer(); Button("Resume") { model.resumeWorker(worker) }.disabled(model.busy); Button("Stop") { if confirmStopModel(worker["name"].text) { model.stopWorker(worker) } }.disabled(model.busy) }
            }
            Divider()
            TextField("Search local models", text: $model.search)
                .textFieldStyle(.roundedBorder)
            if model.library["scanning"].bool {
                HStack { ProgressView().controlSize(.mini); Text("Scanning model library…").font(.caption).foregroundStyle(.secondary) }
            }
            ScrollView {
                LazyVStack(spacing: 4) {
                    ForEach(Array(model.filteredRows.enumerated()), id: \.offset) { _, row in
                        HStack(spacing: 10) {
                            Image(systemName: row["kind"].text == "Draft" ? "puzzlepiece.extension" : "cube")
                                .foregroundStyle(.secondary).frame(width: 22)
                            VStack(alignment: .leading, spacing: 3) {
                                Text(row["name"].text).lineLimit(1).truncationMode(.middle)
                                Text("\(row["kind"].text == "Draft" ? "Spec Drafter" : row["kind"].text) · \(bytes(row["allocated_bytes"].double))")
                                    .font(.caption).foregroundStyle(.secondary)
                            }
                            // Full name for truncated rows; the reason explains rows without Start.
                            .help(row["name"].text + (row["reason"].string.map { "\n" + $0 } ?? ""))
                            Spacer(minLength: 4)
                            if model.managedRunning && row["path"].string == model.worker["path"].string {
                                Image(systemName: "checkmark").foregroundStyle(.blue)
                            } else if row["can_start"].bool {
                                Button("Start") { model.start(row) }
                                    .disabled(model.busy || model.managedRunning)
                                    .help(row["reason"].text)
                            }
                        }
                        .padding(.vertical, 7)
                    }
                    if model.filteredRows.isEmpty && !model.library["scanning"].bool {
                        Text("No models found").foregroundStyle(.secondary).padding()
                    }
                }
            }
            .frame(maxHeight: 260)
            if let error = model.error ?? model.library["error"].string {
                Text(error).font(.caption).foregroundStyle(.orange).lineLimit(3).help(error).textSelection(.enabled)
                Button("View Details") { open("Logs") }.buttonStyle(.link)
            }
            Divider()
            nav("Details…", "chart.bar.xaxis", "Overview")
            nav("Chat…", "bubble.left.and.bubble.right", "Chat")
            nav("Performance…", "gauge.with.dots.needle.67percent", "Performance")
            nav("Model Library…", "shippingbox", "Model Library")
            Divider()
            HStack {
                Button("Settings…") { open("Settings") }.buttonStyle(.plain)
                Spacer()
                Button("Quit MLX2") { NSApp.terminate(nil) }.buttonStyle(.plain)
            }.font(.callout)
        }
        .padding(18)
        .frame(width: 390)
        .task { model.begin() }
    }

    private func metric(_ label: String, _ value: String) -> some View {
        VStack(alignment: .leading, spacing: 3) {
            Text(value).font(.callout).monospacedDigit()
            Text(label).font(.caption2).foregroundStyle(.secondary)
        }
    }

    private func nav(_ label: String, _ symbol: String, _ section: String) -> some View {
        Button { open(section) } label: {
            HStack { Label(label, systemImage: symbol); Spacer(); Image(systemName: "chevron.right").foregroundStyle(.tertiary) }
                .contentShape(Rectangle())
        }.buttonStyle(.plain)
    }
}

struct DetailsView: View {
    @EnvironmentObject var model: DesktopModel
    @State private var columnVisibility = NavigationSplitViewVisibility.all
    @AppStorage("mlx2.sidebar.workspace.expanded") private var workspaceExpanded = true
    @AppStorage("mlx2.sidebar.models.expanded") private var modelsExpanded = false
    @AppStorage("mlx2.sidebar.monitor.expanded") private var monitorExpanded = true
    @AppStorage("mlx2.sidebar.connections.expanded") private var connectionsExpanded = false

    private func expansion(for group: String) -> Binding<Bool> {
        switch group {
        case "workspace": return $workspaceExpanded
        case "models": return $modelsExpanded
        case "monitor": return $monitorExpanded
        default: return $connectionsExpanded
        }
    }

    var body: some View {
        NavigationSplitView(columnVisibility: $columnVisibility) {
            List(selection: $model.section) {
                ForEach(sidebarGroups) { group in
                    DisclosureGroup(isExpanded: expansion(for: group.id)) {
                        ForEach(group.sections, id: \.self) { section in
                            Label(section == "Wakeups" ? "Loops & Wakeups" : section, systemImage: symbols[section]!).tag(section)
                        }
                    } label: {
                        Text(group.title).font(.caption.weight(.semibold)).foregroundStyle(.secondary)
                    }
                }
            }
            .listStyle(.sidebar)
            .onChange(of: model.section) { _, selected in
                if let group = sidebarGroups.first(where: { $0.sections.contains(selected) }) {
                    expansion(for: group.id).wrappedValue = true
                }
            }
            .navigationSplitViewColumnWidth(min: 170, ideal: 190, max: 240)
        } detail: {
            VStack(spacing: 0) {
                switch model.section {
                case "Chat": BoundedWorkspace { ChatView(chat: model.chat) }
                case "Code": BoundedWorkspace { CodeView() }
                case "World": WorldView()
                case "Dream": DreamView()
                case "User Profiles": UserProfilesView()
                case "Wakeups": WakeupsView()
                case "Model Library": LibraryView()
                case "Settings": SettingsView()
                case "Logs": LogsView()
                case "Qualification": QualificationView(model: model)
                case "Engines": EngineProfilesView()
                case "Runtime": RuntimeView()
                case "Community": CommunityView()
                case "Media": MediaView()
                case "Downloads": DownloadsView()
                case "Conversions": ConversionView()
                case "Ablation": ResidualAblationView()
                case "Helpers": HelperView()
                case "Skills": SkillsView()
                case "Agents": AgentsView()
                case "Memory": MemoryView()
                case "Requests": RequestsView()
                case "Performance": PerformanceView()
                case "Cache": CacheView()
                case "Routes": ReceiptView()
                default: OverviewView()
                }
            }
            .navigationTitle(model.section == "Overview" ? "MLX2" : model.section)
            .toolbar {
                ToolbarItemGroup {
                    Button("3D Fullscreen", systemImage: "arrow.up.left.and.arrow.down.right") {
                        if model.section != "World" { model.worldReturnSection = model.section }
                        model.section = "World"
                        LifecycleActions.toggleFullscreen(enterOnly: true)
                    }.help("Open the native 3D workspace in fullscreen")
                    if !model.servers.isEmpty && !["Settings", "Model Library", "Chat"].contains(model.section) {
                        Picker("Server", selection: Binding(get: { model.server["endpoint"]["url"].text }, set: { model.selectedServerURL = $0; model.points = [] })) {
                            ForEach(Array(model.servers.enumerated()), id: \.offset) { _, server in
                                Text(server["model"].string ?? server["endpoint"]["url"].text).tag(server["endpoint"]["url"].text)
                            }
                        }.frame(maxWidth: 260)
                    }
                    StatusDot(state: model.state)
                    if model.isManagedSelection { Button("Suspend Model") { model.suspendSelectedServer() }.disabled(model.busy); Button("Stop Model") { if confirmStopModel(model.server["model"].string ?? "") { model.stopSelectedServer() } }.disabled(model.busy) }
                    ProgressView().controlSize(.small).frame(width: 16).opacity(model.busy ? 1 : 0)
                }
            }
            .overlay(alignment: .top) {
                if let error = model.error {
                    HStack(alignment: .top) {
                        Image(systemName: "exclamationmark.triangle").foregroundStyle(.orange)
                        Text(error).textSelection(.enabled)
                        Spacer()
                        Button { model.error = nil } label: { Image(systemName: "xmark") }.buttonStyle(.plain)
                            .help("Dismiss").accessibilityLabel("Dismiss error")
                    }.font(.callout).padding(12).background(.regularMaterial, in: RoundedRectangle(cornerRadius: 8))
                        .shadow(radius: 6).padding(12)
                }
            }
        }
        .overlay {
            if let image = model.expandedChatImage {
                ChatImageLightbox(item: image) { model.expandedChatImage = nil }
                    .id(image.id)
                    .zIndex(100)
            }
        }
        .onChange(of: model.chat.selected) { _, _ in model.expandedChatImage = nil }
        .modifier(AgentChoicePresenter(choices: model.choices))
        .onReceive(NotificationCenter.default.publisher(for: NSWindow.didEnterFullScreenNotification)) { event in
            if event.object as? NSWindow === LifecycleActions.detailsWindow, model.section == "World" { columnVisibility = .detailOnly }
        }
        .onReceive(NotificationCenter.default.publisher(for: NSWindow.didExitFullScreenNotification)) { event in
            if event.object as? NSWindow === LifecycleActions.detailsWindow { columnVisibility = .all }
        }
        .onChange(of: model.section) { _, section in
            columnVisibility = section == "World" && LifecycleActions.detailsWindow?.styleMask.contains(.fullScreen) == true ? .detailOnly : .all
        }
        .frame(minWidth: 960, minHeight: 650)
        .task { model.begin() }
    }
}

// Keep window fitting size independent of long transcripts and media. Without
// a finite pane proposal, AppKit may wrap the entire split view in a scroll host.
struct BoundedWorkspace<Content: View>: View {
    let content: Content
    init(@ViewBuilder content: () -> Content) { self.content = content() }
    var body: some View {
        GeometryReader { viewport in
            content.frame(width: viewport.size.width, height: viewport.size.height)
        }.clipped()
    }
}

struct MetricCard: View {
    let title: String
    let value: String
    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text(title).font(.callout).foregroundStyle(.secondary)
            Text(value).font(.title2.weight(.semibold)).monospacedDigit().lineLimit(1).minimumScaleFactor(0.6)
        }.frame(maxWidth: .infinity, alignment: .leading).padding(16)
            .background(.quaternary.opacity(0.35), in: RoundedRectangle(cornerRadius: 12))
    }
}

struct OverviewView: View {
    @EnvironmentObject var model: DesktopModel
    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                Text(model.server["model"].string ?? (model.managedRunning ? model.worker["name"].text : "No model running"))
                    .font(.title3.weight(.medium)).textSelection(.enabled)
                if let error = model.server["error"].string ?? model.server["status_error"].string {
                    Label(error, systemImage: "exclamationmark.triangle").foregroundStyle(.orange).font(.callout)
                }
                HStack(spacing: 12) {
                    MetricCard(title: "Generation", value: number(model.server["generation_tps"].double, suffix: " tok/s"))
                    MetricCard(title: "Prefill", value: number(model.server["prompt_tps"].double, suffix: " tok/s", digits: 0))
                    let ttft = model.server["latency"].array.first { $0[0].text == "TTFT" }?[1]["p50"].double
                    MetricCard(title: "TTFT · p50", value: number(ttft.map { $0 * 1000 }, suffix: " ms", digits: 0))
                    MetricCard(title: "Model memory", value: bytes(model.server["footprint_bytes"].double))
                }
                HStack(alignment: .top, spacing: 16) {
                    HistoryChart(title: "Generation throughput", key: "generation_tps", unit: "tokens / second")
                    HistoryChart(title: "Host memory", key: "host_memory", unit: "GiB", color: .purple)
                }
                HStack(alignment: .top, spacing: 16) {
                    GroupBox("Host") {
                        VStack(spacing: 12) {
                            let cpus = model.host["cpu_busy_percent"].array.compactMap(\.double)
                            field("CPU", number(cpus.isEmpty ? nil : cpus.reduce(0, +) / Double(cpus.count), suffix: "%"))
                            field("GPU", number(model.host["gpus"].array.first?["busy_percent"].double, suffix: "%"))
                            field("Memory available", bytes(model.host["memory"]["available_bytes"].double))
                            field("Swap used", bytes(model.host["swap"]["used_bytes"].double))
                            field("Thermal", model.host["thermal_state"].text.capitalized)
                            field("Memory pressure", model.host["memory"]["pressure"].text)
                        }.padding(12)
                    }.frame(maxWidth: .infinity)
                    GroupBox("APCv2 cache") { CacheFields().padding(12) }.frame(maxWidth: .infinity)
                }
                HStack {
                    Text("Route: \(model.server["route"].text)")
                    Divider().frame(height: 16)
                    Text("Qualification: \(model.server["qualification"].string ?? "Unknown")")
                    Spacer()
                    Button("View Route Details") { model.section = "Routes" }.buttonStyle(.link)
                }.font(.callout).padding(14).background(.quaternary.opacity(0.25), in: RoundedRectangle(cornerRadius: 10))
                field("Requests · active / queued", "\(number(model.server["running"].double, digits: 0)) / \(number(model.server["waiting"].double, digits: 0))")
                Text("Live read-only telemetry · unavailable values appear as — · refreshes every 2 seconds")
                    .font(.caption).foregroundStyle(.secondary)
            }.padding(22)
        }
    }
}

func field(_ label: String, _ value: String) -> some View {
    HStack(alignment: .firstTextBaseline) {
        Text(label).foregroundStyle(.secondary)
        Spacer(minLength: 14)
        Text(value).monospacedDigit().textSelection(.enabled).multilineTextAlignment(.trailing)
    }.font(.callout)
}

struct CacheFields: View {
    @EnvironmentObject var model: DesktopModel
    var body: some View {
        VStack(spacing: 12) {
            let hits = model.server["cache_window_hits"].double
            let lookups = model.server["cache_window_lookups"].double
            field("Window hit rate", number(lookups.flatMap { count in count > 0 ? hits.map { $0 / count * 100 } : nil }, suffix: "%"))
            field("Resident", bytes(model.server["cache_resident_bytes"].double))
            field("Capacity", bytes(model.server["cache_capacity_bytes"].double))
            field("Entries", number(model.server["cache_entries"].double, digits: 0))
            field("Disk", bytes(model.server["cache_disk_bytes"].double))
            field("Cached prompt tokens", number(model.server["cached_prompt_tokens_total"].double, digits: 0))
        }
    }
}

struct CacheView: View {
    @EnvironmentObject var model: DesktopModel
    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 20) {
                HStack(spacing: 12) {
                    MetricCard(title: "Lifetime hit rate", value: percentRatio(model.server["cache_hits"].double, model.server["cache_lookups"].double))
                    MetricCard(title: "Window hit rate", value: percentRatio(model.server["cache_window_hits"].double, model.server["cache_window_lookups"].double))
                    MetricCard(title: "Token reuse", value: percentRatio(model.server["cached_prompt_tokens_total"].double, model.server["cache_query_tokens"].double))
                }
                TelemetryCard(title: "APCv2 residency", symbol: "externaldrive") {
                    UsageBar(title: "Resident / capacity", value: model.server["cache_resident_bytes"].double,
                             maximum: model.server["cache_capacity_bytes"].double ?? 0,
                             formatted: "\(bytes(model.server["cache_resident_bytes"].double)) / \(bytes(model.server["cache_capacity_bytes"].double))", color: .purple)
                    CacheFields()
                }
                HistoryChart(title: "APCv2 resident history", key: "cache_resident_bytes", unit: "bytes", color: .purple)
                APCBranchesView()
                HStack(alignment: .top, spacing: 16) {
                    TelemetryCard(title: "Cache paths", symbol: "arrow.triangle.branch") {
                        ValueBars(values: [("Interior", "cache_interior_hits"), ("Rolling", "cache_rolling_hits"),
                                          ("Junction", "cache_junction_hits")].compactMap { title, key in
                            model.server[key].double.map { NamedValue(name: title, value: $0) }
                        }, unit: "hits", color: .purple)
                        field("Evictions · 30 s", number(model.server["cache_evictions_window"].double, digits: 0))
                        field("Disk entries", number(model.server["cache_disk_entries"].double, digits: 0))
                    }
                    TelemetryCard(title: "Host prompt cache", symbol: "text.page") {
                        let cache = model.server["host_prompt_cache"]
                        ValueBars(values: [("Hits", "hits"), ("Misses", "misses"), ("Entries", "entries")].compactMap { title, key in
                            cache[key].double.map { NamedValue(name: title, value: $0) }
                        }, color: .teal)
                        field("Retained tokens", number(cache["tokens"].double, digits: 0))
                    }
                }
                Text("Cache values come from the selected server. Non-cache execution kinds report unavailable values.")
                    .font(.callout).foregroundStyle(.secondary)
            }.padding(24)
        }
    }
}

struct RequestsView: View {
    @EnvironmentObject var model: DesktopModel
    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                HStack(spacing: 12) {
                    MetricCard(title: "Active", value: number(model.server["running"].double, digits: 0))
                    MetricCard(title: "Queued", value: number(model.server["waiting"].double, digits: 0))
                    MetricCard(title: "Memory waiting", value: number(model.server["memory_waiting"].double, digits: 0))
                }
                HStack(alignment: .top, spacing: 16) {
                    HistoryChart(title: "Active request history", key: "running", unit: "requests", color: .blue)
                    TelemetryCard(title: "Request outcomes", symbol: "checkmark.circle") {
                        ValueBars(values: pairs(model.server["outcomes"]), color: .green)
                    }
                }
                RequestTables()
                LatencyTable()
            }.frame(maxWidth: .infinity, alignment: .leading).padding(24)
        }
    }
}

struct ReceiptView: View {
    @EnvironmentObject var model: DesktopModel
    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                HStack(spacing: 12) {
                    MetricCard(title: "Selected route", value: model.server["route"].text)
                    MetricCard(title: "Qualification", value: model.server["qualification"].string ?? "Unknown")
                    MetricCard(title: "Mechanisms observed", value: "\(model.server["mechanisms"].array.count)")
                }
                TelemetryCard(title: "Runtime identity", symbol: "checkmark.shield") {
                    field("Runtime revision", model.server["revision"].text)
                    field("MLX version", model.server["mlx_version"].text)
                    field("Endpoint", model.server["endpoint"]["url"].text)
                    field("Profile", model.server["profile"].text)
                    field("State", model.server["state"].text)
                    if model.isManagedSelection {
                        field("Managed checkpoint", model.worker["checkpoint"].text)
                        field("Execution kind", model.worker["execution_kind"].text)
                        if let path = model.worker["api_key_file"].string { field("Local API key file", path) }
                    }
                }
                HStack(alignment: .top, spacing: 16) {
                    TelemetryCard(title: "Selected capabilities", symbol: "checkmark.circle") {
                        CapabilityChips(values: model.server["selected_capabilities"].array, color: .blue)
                    }
                    TelemetryCard(title: "Qualified capabilities", symbol: "checkmark.seal") {
                        CapabilityChips(values: model.server["qualified_capabilities"].array, color: .green)
                    }
                }
                MechanismsVisual()
                Text("Selection, qualification and observed mechanism use are separate evidence states.")
                    .font(.caption).foregroundStyle(.secondary)
            }.padding(24)
        }
    }
}

struct LogsView: View {
    @EnvironmentObject var model: DesktopModel
    @State private var query = ""
    @State private var severity = "All"
    // Number lines before filtering so a filtered view still shows each line's
    // position in the log, not its position in the search results.
    var lines: [(number: Int, text: String)] {
        var all = model.logs.components(separatedBy: .newlines)
        if all.last == "" { all.removeLast() }
        return all.enumerated().map { (number: $0.offset + 1, text: $0.element) }.filter { line in
            (query.isEmpty || line.text.localizedCaseInsensitiveContains(query))
                && (severity == "All" || line.text.localizedCaseInsensitiveContains(severity))
        }
    }
    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Picker("Log source", selection: $model.logSource) {
                    Text("Application").tag("application")
                    Text("Model worker").tag("worker")
                    Text("Qualification").tag("campaign")
                }.frame(width: 250)
                if model.logSource == "worker" {
                    Picker("Engine", selection: $model.logProfileID) {
                        Text("Default engine").tag("default")
                        ForEach(Array(model.workers.filter { $0["profile_id"].text != "default" }.enumerated()), id: \.offset) { _, worker in
                            Text(worker["name"].text).tag(worker["profile_id"].text)
                        }
                    }.frame(width: 220)
                }
                Spacer()
                Button("Refresh") { Task { await model.refreshLogs() } }
                if let path = model.worker["log_path"].string {
                    Button("Reveal Log") { NSWorkspace.shared.activateFileViewerSelecting([URL(fileURLWithPath: path)]) }
                }
            }
            HStack {
                TextField("Search log messages", text: $query).textFieldStyle(.roundedBorder)
                Picker("Severity", selection: $severity) {
                    ForEach(["All", "Error", "Warning", "Info"], id: \.self) { Text($0) }
                }.frame(width: 150)
                Text(lines.count == 1 ? "1 line" : "\(lines.count) lines").font(.caption).foregroundStyle(.secondary)
            }
            ScrollView([.horizontal, .vertical]) {
                LazyVStack(alignment: .leading, spacing: 4) {
                    if lines.isEmpty {
                        Text(model.logs.isEmpty ? "This log is empty." : "No lines match the current search and severity.")
                            .font(.callout).foregroundStyle(.secondary)
                    }
                    ForEach(lines, id: \.number) { line in
                        HStack(alignment: .top, spacing: 12) {
                            Text("\(line.number)").foregroundStyle(.tertiary).frame(width: 48, alignment: .trailing)
                            Text(line.text).foregroundStyle(line.text.localizedCaseInsensitiveContains("error") ? Color.red : line.text.localizedCaseInsensitiveContains("warning") ? Color.orange : Color.primary)
                                .textSelection(.enabled)
                        }.font(.system(.caption, design: .monospaced))
                    }
                }.frame(maxWidth: .infinity, alignment: .topLeading).padding(12)
            }.background(.black.opacity(0.12), in: RoundedRectangle(cornerRadius: 8))
        }.padding(20)
            .task { await model.refreshLogs() }
            .onChange(of: model.logSource) { _, _ in Task { await model.refreshLogs() } }
            .onChange(of: model.logProfileID) { _, _ in Task { await model.refreshLogs() } }
    }
}

struct Artifact: SortableDesktopRow {
    let data: JSON
    var id: String { data["id"].text }
}

struct LibraryView: View {
    @EnvironmentObject var model: DesktopModel
    @State private var filter = "All"
    @State private var sortOrder = [RowSort<Artifact>("name")]
    @State private var performanceGroup = ""
    private var comparisons: [JSON] { model.snapshot["performance_comparisons"].array }
    private var comparisonGroups: [String] { Array(Set(comparisons.map { $0["group"].text })).sorted() }
    var artifacts: [Artifact] {
        let filtered = model.filteredRows.filter { row in
            filter == "All" || (filter == "Serving" && row["can_start"].bool)
                || (filter == "Spec Drafter" && row["kind"].text == "Draft")
                || (filter == "Other" && !row["can_start"].bool && row["kind"].text != "Draft")
        }
        let rows = filtered.map { row in
            var values = row.raw as? [String: Any] ?? [:]
            values["_status"] = model.workerFor(row)?["state"].text ?? "Local"
            values["_last_loaded"] = model.snapshot["history"][row["id"].text].raw
            let measured = comparisons.first { $0["group"].text == performanceGroup && $0["checkpoint"].text == row["checkpoint"].text }
            values["_prefill_tps"] = measured?["prefill_tps"].double
            values["_decode_tps"] = measured?["decode_tps"].double
            return Artifact(data: JSON(values))
        }
        return sortedRows(rows, using: sortOrder)
    }
    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            VStack(alignment: .leading, spacing: 12) {
            HStack {
                Label(model.library["root"].text, systemImage: "folder").font(.callout).textSelection(.enabled).lineLimit(1).truncationMode(.middle)
                Spacer()
                Button("Choose Folder…") { model.chooseFolder() }.disabled(model.busy || model.servingSettingsLocked)
                Button { model.perform("refresh") } label: { Image(systemName: "arrow.clockwise") }.disabled(model.busy || model.library["scanning"].bool)
                    .help("Rescan model library").accessibilityLabel("Rescan model library")
            }
            HStack {
                TextField("Search models", text: $model.search).textFieldStyle(.roundedBorder).frame(maxWidth: 260)
                Picker("Kind", selection: $filter) { ForEach(["All", "Serving", "Spec Drafter", "Other"], id: \.self) { Text($0) } }.pickerStyle(.segmented).frame(maxWidth: 360)
                Spacer()
                Text("Click a column to sort").font(.caption).foregroundStyle(.secondary)
            }
            HStack {
                Text(model.library["error"].string ?? "\(model.rows.count) \(model.rows.count == 1 ? "folder" : "folders") · \(bytes(model.rows.reduce(0) { $0 + ($1["allocated_bytes"].double ?? 0) })) allocated")
                    .lineLimit(1).foregroundStyle(model.library["error"].string == nil ? Color.secondary : Color.orange)
                Spacer()
                ProgressView().controlSize(.small).opacity(model.library["scanning"].bool ? 1 : 0)
                Text("Scanning…").opacity(model.library["scanning"].bool ? 1 : 0)
            }.font(.caption).foregroundStyle(.secondary)
            LibraryStorageChart(rows: model.rows)
            }.frame(height: 158, alignment: .top)
            if !comparisonGroups.isEmpty {
                HStack {
                    Picker("Performance profile", selection: $performanceGroup) {
                        Text("Choose a matching measured profile").tag("")
                        ForEach(comparisonGroups, id: \.self) { group in
                            let shape = comparisons.first { $0["group"].text == group }?["shape"] ?? JSON()
                            Text("\(shape["host"]["hostname"].text) · context \(number(shape["context_tokens"].double, digits: 0)) · width \(number(shape["width"].double, digits: 0)) · \(String(shape["source_revision"].text.prefix(8)))").tag(group)
                        }
                    }
                    Button("Test Selected (\(model.checkedModels.count))") { model.section = "Qualification" }
                        .disabled(model.checkedModels.isEmpty).help("Check models in the Test column first")
                }
            } else if !model.checkedModels.isEmpty {
                Button("Test Selected (\(model.checkedModels.count))") { model.section = "Qualification" }
            }
            Table(artifacts, selection: $model.selection, sortOrder: $sortOrder) {
                TableColumn("Test") { artifact in
                    Toggle("Select for tests", isOn: Binding(get: { model.checkedModels.contains(artifact.id) }, set: { selected in
                        if selected { model.checkedModels.insert(artifact.id) } else { model.checkedModels.remove(artifact.id) }
                    })).labelsHidden().disabled(!artifact.data["can_start"].bool)
                        .help(artifact.data["can_start"].bool ? "Select for tests" : "Only startable models can be tested")
                }.width(35)
                TableColumn("Model", sortUsing: RowSort<Artifact>("name")) { Text($0.data["name"].text).help($0.data["reason"].text) }.width(min: 160, ideal: 260)
                TableColumn("Kind", sortUsing: RowSort<Artifact>("kind")) { Text($0.data["kind"].text == "Draft" ? "Spec Drafter" : $0.data["kind"].text) }.width(min: 90, ideal: 130)
                TableColumn("Qualification") { Text($0.data["qualification"]["label"].string ?? "Untested").help($0.data["qualification"]["label"].string ?? "Untested") }.width(min: 100, ideal: 120)
                TableColumn("Prefill tok/s", sortUsing: RowSort<Artifact>("_prefill_tps", numeric: true)) { Text(number($0.data["_prefill_tps"].double)) }.width(90)
                TableColumn("Decode tok/s", sortUsing: RowSort<Artifact>("_decode_tps", numeric: true)) { Text(number($0.data["_decode_tps"].double)) }.width(90)
                TableColumn("On disk", sortUsing: RowSort<Artifact>("allocated_bytes", numeric: true)) { Text(bytes($0.data["allocated_bytes"].double)).monospacedDigit() }.width(85)
                TableColumn("Status", sortUsing: RowSort<Artifact>("_status")) { Text($0.data["_status"].text) }.width(70)
                TableColumn("Client URL") { artifact in
                    if let endpoint = model.servingEndpoint(artifact.data) {
                        HStack {
                            Text(endpoint).font(.caption.monospaced()).lineLimit(1)
                            Button { model.copyEndpoint(artifact.data) } label: { Image(systemName: "doc.on.doc") }
                                .buttonStyle(.borderless).help("Copy client base URL")
                        }
                    } else { Text("—").foregroundStyle(.tertiary) }
                }.width(min: 140, ideal: 230)
                TableColumn("Last loaded", sortUsing: RowSort<Artifact>("_last_loaded", numeric: true)) { Text(timestamp($0.data["_last_loaded"].double)) }.width(min: 100, ideal: 150)
            }
            .contextMenu(forSelectionType: String.self) { targets in
                // SwiftUI supplies the context-clicked row, even when another
                // row was selected. Never use the inspector's fallback row.
                if targets.count == 1, let id = targets.first,
                   let artifact = artifacts.first(where: { $0.id == id }) {
                    let row = artifact.data
                    if model.servingEndpoint(row) != nil {
                        Button("Copy Client URL") { model.copyEndpoint(row) }
                        Divider()
                    }
                    if let worker = model.workerFor(row) {
                        if worker["state"].text == "Suspended" { Button("Resume Model") { model.resumeWorker(worker) }.disabled(model.busy) }
                        else if ["Ready", "Busy"].contains(worker["state"].text) { Button("Suspend Model") { model.suspendWorker(worker) }.disabled(model.busy) }
                        Button { if confirmStopModel(row["name"].text) { model.stop(row) } } label: { Label("Stop Model", systemImage: "stop.circle") }
                            .disabled(model.busy)
                    } else {
                        Button { model.start(row) } label: { Label("Start Model", systemImage: "play.circle") }
                            .disabled(model.busy || model.managedRunning || !row["can_start"].bool)
                    }
                    Divider()
                    Button(role: .destructive) { model.trash(row) } label: {
                        Label("Move to Trash…", systemImage: "trash")
                    }
                    .disabled(model.busy || row["kind"].text == "Link"
                              || model.owns(row))
                }
            }
            .frame(minHeight: 140)
            Divider()
            ScrollView {
            if model.selected["path"].string != nil {
                let row = model.selected
                VStack(alignment: .leading, spacing: 12) {
                    Text(row["name"].text).font(.title3.weight(.semibold)).lineLimit(1).truncationMode(.middle).textSelection(.enabled)
                    Text(row["path"].text).font(.caption).foregroundStyle(.secondary).lineLimit(1).truncationMode(.middle).textSelection(.enabled)
                    if let endpoint = model.servingEndpoint(row) {
                        HStack {
                            Text(endpoint).font(.callout.monospaced()).textSelection(.enabled)
                            Button("Copy Client URL", systemImage: "doc.on.doc") { model.copyEndpoint(row) }
                        }
                    }
                    HStack(alignment: .top, spacing: 35) {
                        VStack(spacing: 9) {
                            field("Logical size", bytes(row["logical_bytes"].double))
                            field("Allocated size", bytes(row["allocated_bytes"].double))
                            field("Accessed (atime)", timestamp(row["atime"].double))
                        }
                        VStack(spacing: 9) {
                            field("Modified (mtime)", timestamp(row["mtime"].double))
                            field("Metadata changed (ctime)", timestamp(row["ctime"].double))
                            field("Created", timestamp(row["created"].double))
                        }
                    }
                    Text(row["reason"].text).font(.caption).foregroundStyle(.secondary).lineLimit(2)
                    if row["can_start"].bool { ModelLaunchOptionsView(row: row).id(row["path"].text) }
                    if model.owns(row), let phase = model.worker["preparation_message"].string {
                        Label(phase, systemImage: "gearshape.2").font(.caption).foregroundStyle(.blue)
                    }
                    if !row["errors"].array.isEmpty {
                        ForEach(Array(row["errors"].array.enumerated()), id: \.offset) { _, error in
                            Label(error.text, systemImage: "exclamationmark.triangle").font(.caption).foregroundStyle(.orange)
                        }
                    }
                    HStack {
                        Text("Folder timestamps; Last loaded is tracked by MLX2. Sizes exclude symlinks.")
                            .font(.caption2).foregroundStyle(.secondary)
                        Spacer()
                        Button("Reveal in Finder") { model.reveal(row) }
                        Button("Move to Trash…") { model.trash(row) }.disabled(model.busy || row["kind"].text == "Link" || model.owns(row))
                            .help(model.owns(row) ? "Stop this model before moving it to Trash." : row["kind"].text == "Link" ? "Linked folders cannot be moved to Trash from MLX2." : "")
                        if let worker = model.workerFor(row) {
                            if worker["state"].text == "Suspended" { Button("Resume Model") { model.resumeWorker(worker) }.disabled(model.busy) }
                            else if ["Ready", "Busy"].contains(worker["state"].text) { Button("Suspend Model") { model.suspendWorker(worker) }.disabled(model.busy) }
                            Button("Stop Model") { if confirmStopModel(row["name"].text) { model.stop(row) } }.disabled(model.busy)
                        } else {
                            Button("Start Model") { model.start(row) }.buttonStyle(.borderedProminent)
                                .disabled(model.busy || model.managedRunning || !row["can_start"].bool)
                                .help(model.managedRunning ? "Stop the running model first." : row["can_start"].bool ? "" : row["reason"].text)
                        }
                    }
                }.padding(.vertical, 6)
            } else {
                Label("Select a model to view its details", systemImage: "cube").foregroundStyle(.secondary)
            }
            }.frame(height: 210, alignment: .top)
        }.padding(20)
    }
}

struct SettingsView: View {
    @EnvironmentObject var model: DesktopModel
    @State private var port = "8285"
    @State private var context = "0"
    @State private var lanes = "0"
    @State private var stickyPort = true
    @State private var route = "auto"
    @State private var runtimeOptions = RuntimeLaunchOptions()
    @State private var launchAtLogin = LoginItems.enabled
    @State private var loadedSettings = false
    @State private var confirmPermissionReset = false
    // Saved settings may hold any lane count the supervisor accepts (up to 32);
    // keep that value selectable instead of showing an empty picker.
    private var laneChoices: [Int] { Array(Set([1, 2, 3, 4] + [Int(lanes) ?? 1].filter { $0 > 0 })).sorted() }
    var body: some View {
        Form {
            Section {
                HStack(spacing: 16) {
                    Image(systemName: "cube.transparent").font(.system(size: 36)).foregroundStyle(.blue)
                    VStack(alignment: .leading, spacing: 5) {
                        Text("MLX2 Desktop").font(.title2.weight(.semibold))
                        Text("Local models · native controls · live telemetry").foregroundStyle(.secondary)
                    }
                    Spacer()
                    StatusDot(state: model.state)
                }.padding(.vertical, 10)
            }
            Section("Tool permissions") {
                Text("Allow Always remembers permission for a tool in the active user profile.").font(.caption).foregroundStyle(.secondary)
                Button("Reset Remembered Tool Permissions…") { confirmPermissionReset = true }
                    .confirmationDialog("Reset remembered tool permissions?", isPresented: $confirmPermissionReset) {
                        Button("Reset Permissions", role: .destructive) { model.chat.forgetToolPermissions() }
                        Button("Cancel", role: .cancel) {}
                    } message: {
                        Text("Every tool marked Allow Always in the active user profile will ask again before it runs.")
                    }
            }
            Section("Model library") {
                LabeledContent("Directory", value: model.library["root"].text)
                Button("Choose Model Folder…") { model.chooseFolder() }
            }
            Section("Managed serving profile") {
                Toggle("Keep this port between launches", isOn: $stickyPort)
                TextField("Loopback port", text: $port)
                    .disabled(!stickyPort)
                TextField("Context tokens (0 = automatic)", text: $context)
                Picker("Serving lanes", selection: $lanes) {
                    ForEach(laneChoices, id: \.self) { Text("\($0)").tag(String($0)) }
                }
                Picker("Decode route", selection: $route) {
                    Text("Automatic runtime selection").tag("auto")
                    Text("Ordinary reference").tag("ordinary")
                }
                Text("Automatic uses the runtime's artifact-specific defaults. Starts remain unqualified unless the server supplies matching qualification evidence.")
                    .font(.caption).foregroundStyle(.secondary)
                RuntimeLaunchOptionsView(options: $runtimeOptions)
                Button("Save Settings") {
                    guard let port = Int(port), let context = Int(context), let lanes = Int(lanes) else {
                        model.error = "Port, context and lanes must be whole numbers."; return
                    }
                    do {
                        var settings = model.snapshot["settings"].raw as? [String: Any] ?? [:]
                        settings.merge(["root": model.library["root"].text, "port": port, "context": context,
                                        "lanes": lanes, "route": route, "sticky_port": stickyPort]) { _, updated in updated }
                        model.perform("configure", ["settings": try runtimeOptions.applying(to: settings, route: route)])
                    } catch { model.error = error.localizedDescription }
                }
            }.disabled(model.servingSettingsLocked)
            Section("Application") {
                Toggle("Launch MLX2 at login", isOn: $launchAtLogin)
                    .onChange(of: launchAtLogin) { _, enabled in
                        do { try LoginItems.setEnabled(enabled) }
                        catch { model.error = error.localizedDescription; launchAtLogin = LoginItems.enabled }
                    }
                Text("Starts the menu bar app after login. Choose models to load automatically in Engines.")
                    .font(.caption).foregroundStyle(.secondary)
                Text("Suspend retains resumable state for app-managed models. Stop drains accepted requests for up to 15 seconds and discards suspended state. Quitting stops all app-managed engines; external services retain their existing ownership.")
                    .font(.caption).foregroundStyle(.secondary)
                Button("Open Application Support") {
                    NSWorkspace.shared.open(FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent("Library/Application Support/MLX2"))
                }
            }
            ConnectedToolsSettings()
            PromptSettingsView()
            SkillsPolicySettings()
            ChoiceSettingsView()
            ContextSettingsView()
        }.formStyle(.grouped).disabled(model.busy || !loadedSettings)
            // Recreate nested settings editors once their first snapshot arrives,
            // and when profiles change; polling must not overwrite unsaved edits.
            .id((model.snapshot["user_profiles"]["active_id"].string ?? "") + (loadedSettings ? ":ready" : ":loading"))
            .onAppear { loadSettings() }
            .onChange(of: model.snapshot["settings"]["root"].string) { _, _ in if !loadedSettings { loadSettings() } }
            .onChange(of: model.snapshot["user_profiles"]["active_id"].string) { _, _ in loadedSettings = false; loadSettings() }
    }
    private func loadSettings() {
        let settings = model.snapshot["settings"]
        guard settings["root"].string != nil else { return }
        port = String(Int(settings["port"].double ?? 8285))
        context = String(Int(settings["context"].double ?? 0))
        lanes = String(Int(settings["lanes"].double ?? 1))
        stickyPort = settings["sticky_port"].raw is NSNull ? true : settings["sticky_port"].bool
        route = settings["route"].string ?? "auto"
        runtimeOptions = RuntimeLaunchOptions(settings)
        loadedSettings = true
    }
}
