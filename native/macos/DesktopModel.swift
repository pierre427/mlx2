import AppKit
import CoreGraphics
import SwiftUI

struct HistoryPoint: Identifiable {
    let id = UUID()
    let time: Date
    let generation: Double?
    let memory: Double?
    let values: [String: Double]
}

@MainActor final class DesktopModel: ObservableObject {
    static let shared = DesktopModel()
    @Published var snapshot = JSON()
    @Published var error: String?
    @Published var busy = false
    @Published var polling = false
    @Published var search = ""
    @Published var selection: String?
    @Published var selectedServerURL: String?
    @Published var section = "Chat"
    @Published var expandedChatImage: ChatExpandedImage?
    @Published var worldReturnSection = "Chat"
    @Published var logs = ""
    @Published var logSource = "application"
    @Published var logProfileID = "default"
    @Published var checkedModels: Set<String> = []
    @Published var points: [HistoryPoint] = []
    let chat = ChatModel()
    let codeCoordinator = CodeCoordinator()
    let choices = AgentChoices()
    private let profileChats = ProfileChatPersistence()
    private let wakeupCoordinator = WakeupCoordinator()
    private let bridge = Bridge()
    private var timer: Task<Void, Never>?
    private var historyIdentity: String?
    private var previousUptime: Double?
    private var lastWorkerError: String?

    // A snapshot repeats persistent worker failures. Present each transition
    // once so dismissing an old failure survives the next status poll.
    func updateWorkerError(_ workerError: String?) {
        guard workerError != lastWorkerError else { return }
        if error == lastWorkerError { error = nil }
        lastWorkerError = workerError
        if let workerError { error = workerError }
    }

    var library: JSON { snapshot["library"] }
    var rows: [JSON] { library["rows"].array }
    var filteredRows: [JSON] { rows.filter { search.isEmpty || $0["name"].text.localizedCaseInsensitiveContains(search) } }
    var worker: JSON { snapshot["worker"] }
    var preferences: JSON { snapshot["preferences"] }
    var jobs: [JSON] { snapshot["jobs"].array }
    var managedRunning: Bool { workers.contains { ["Preparing", "Loading", "Ready", "Busy", "Stopping", "Unreachable", "Suspending", "Resuming"].contains($0["state"].text) || $0["process_running"].bool } }
    var suspendedWorkers: [JSON] { workers.filter { $0["state"].text == "Suspended" } }
    var servingSettingsLocked: Bool { managedRunning || workers.contains { ["Suspended", "ResumeError", "SuspensionRecoveryRequired"].contains($0["state"].text) } }
    var servers: [JSON] { snapshot["servers"].array }
    var server: JSON {
        servers.first { $0["endpoint"]["url"].string == selectedServerURL }
            ?? servers.first { $0["endpoint"]["url"].string == worker["url"].string }
            ?? servers.first ?? JSON()
    }
    var host: JSON { snapshot["host"] }
    var activeWorker: JSON { workers.first { ["Preparing", "Loading", "Ready", "Busy", "Stopping", "Unreachable", "Suspending", "Resuming"].contains($0["state"].text) || $0["process_running"].bool } ?? worker }
    var workers: [JSON] { snapshot["workers"].array.isEmpty ? [worker] : snapshot["workers"].array }

    func servingEndpoint(_ row: JSON) -> String? {
        let path = row["path"].string
        let managed = workers.first { $0["path"].string == path && ["Loading", "Ready", "Busy", "Unreachable"].contains($0["state"].text) }
        let observed = servers.first { $0["endpoint"]["model_path"].string == path && $0["reachable"].bool }
        guard let base = managed?["url"].string ?? observed?["endpoint"]["url"].string else { return nil }
        return base.trimmingCharacters(in: CharacterSet(charactersIn: "/")) + "/v1"
    }

    func copyEndpoint(_ row: JSON) {
        guard let endpoint = servingEndpoint(row) else { return }
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(endpoint, forType: .string)
    }
    var isManagedSelection: Bool { workers.contains { $0["url"].string == server["endpoint"]["url"].string && ["Loading", "Ready", "Busy", "Unreachable"].contains($0["state"].text) } }
    var selected: JSON { rows.first { $0["id"].string == selection } ?? filteredRows.first ?? JSON() }
    var state: String { managedRunning ? activeWorker["state"].text : !suspendedWorkers.isEmpty ? "Suspended" : servers.isEmpty ? "Stopped" : "Observing" }

    func stopWorker(_ selected: JSON) {
        if let profile = selected["profile_id"].string, profile != "default" { perform("profile_stop", ["profile_id": profile]) }
        else { perform("stop") }
    }

    func suspendWorker(_ worker: JSON) {
        perform("engine_suspend", ["profile_id": worker["profile_id"].string ?? "default"])
    }

    func resumeWorker(_ worker: JSON) {
        perform("engine_resume", ["profile_id": worker["profile_id"].string ?? "default"])
    }

    func suspendSelectedServer() {
        if let worker = workers.first(where: { $0["url"].string == server["endpoint"]["url"].string }) { suspendWorker(worker) }
    }

    func resumeSelectedServer() {
        if let worker = suspendedWorkers.first(where: { $0["url"].string == server["endpoint"]["url"].string }) ?? suspendedWorkers.first { resumeWorker(worker) }
    }

    func stopSelectedServer() {
        guard let selected = workers.first(where: { $0["url"].string == server["endpoint"]["url"].string }) else { return }
        stopWorker(selected)
    }

    func owns(_ row: JSON) -> Bool {
        workerFor(row) != nil
    }

    func workerFor(_ row: JSON) -> JSON? {
        workers.first { $0["path"].string == row["path"].string && ["Preparing", "Loading", "Ready", "Busy", "Stopping", "Unreachable", "Suspending", "Suspended", "Resuming", "SuspendError", "ResumeError", "SuspensionRecoveryRequired"].contains($0["state"].text) }
    }

    func begin() {
        guard timer == nil else { return }
        profileChats.start(self)
        wakeupCoordinator.start(self)
        codeCoordinator.start(self)
        timer = Task {
            while !Task.isCancelled {
                let idle = CGEventSource.secondsSinceLastEventType(.combinedSessionState, eventType: CGEventType(rawValue: UInt32.max)!)
                _ = try? await bridge.request("dream_activity", ["idle_seconds": idle, "busy": chat.generating || busy])
                await refresh()
                try? await Task.sleep(for: .seconds(2))
            }
        }
    }

    @discardableResult func refresh() async -> Bool {
        guard !polling && !busy else { return false }
        polling = true
        defer { polling = false }
        do {
            snapshot = try await bridge.request("snapshot")
            guard await profileChats.select(snapshot["user_profiles"]["active_id"].text) else { return false }
            chat.reconcileManagedResumes(servers)
            updateWorkerError(snapshot["error"].string)
            if let selection, !rows.contains(where: { $0["id"].string == selection }) { self.selection = nil }
            let identity = server["endpoint"]["url"].text + "|" + server["model"].text + "|" + server["revision"].text
            let uptime = server["uptime_seconds"].double
            if identity != historyIdentity || (uptime != nil && previousUptime != nil && uptime! < previousUptime!) {
                points = []
            }
            historyIdentity = identity
            previousUptime = uptime
            let memory = host["memory"]["total_bytes"].double.flatMap { total in
                host["memory"]["available_bytes"].double.map { total - $0 }
            }
            var values: [String: Double] = [:]
            for key in ["generation_tps", "generation_tps_window", "prompt_tps", "prompt_tps_window",
                        "cached_prompt_tps_window", "requests_per_second_window", "running", "waiting",
                        "active_lanes", "batch_width_window", "metal_active_bytes", "footprint_bytes",
                        "cache_resident_bytes", "scheduler_cycles_per_second"] {
                values[key] = server[key].double
            }
            let cpus = host["cpu_busy_percent"].array.compactMap(\.double)
            values["host_cpu"] = cpus.isEmpty ? nil : cpus.reduce(0, +) / Double(cpus.count)
            values["host_gpu"] = host["gpus"].array.first?["busy_percent"].double
            values["host_memory"] = memory.map { $0 / 1073741824 }
            values["swap_used"] = host["swap"]["used_bytes"].double.map { $0 / 1073741824 }
            values["gpu_power"] = host["soc_power"]["gpu_power_mw"].double.map { $0 / 1000 }
            values["cpu_power"] = host["soc_power"]["cpu_power_mw"].double.map { $0 / 1000 }
            values["ane_power"] = host["soc_power"]["ane_power_mw"].double.map { $0 / 1000 }
            values["temperature"] = host["soc_power"]["die_max_c"].double
            for pair in server["latency"].array {
                values[pair[0].text + "_p50_ms"] = pair[1]["p50"].double.map { $0 * 1000 }
            }
            points.append(HistoryPoint(time: Date(), generation: server["generation_tps"].double,
                                       memory: memory, values: values))
            points = Array(points.suffix(60))
            if section == "Logs" { await refreshLogs() }
            if profileTransitioning { finishProfileTransition() }
            Task { await wakeupCoordinator.poll() }
            return true
        } catch {
            self.error = error.localizedDescription
            points.append(HistoryPoint(time: Date(), generation: nil, memory: nil, values: [:]))
            points = Array(points.suffix(60))
            return false
        }
    }

    func perform(_ command: String, _ arguments: [String: Any] = [:]) {
        guard !busy else { return }
        if ["user_profile_create", "user_profile_switch"].contains(command) && chat.generating {
            error = "Stop the current reply before switching user profiles."
            return
        }
        busy = true
        Task {
            do {
                if ["user_profile_create", "user_profile_switch"].contains(command) {
                    while polling { try? await Task.sleep(for: .milliseconds(50)) }
                    profileTransitioning = true; chat.mutationsPaused = true
                    profileChats.pause()
                    await wakeupCoordinator.suspendForProfileChange()
                    await codeCoordinator.suspendForProfileChange()
                    await chat.stopAndWait()
                    guard await profileChats.flush() else { throw DesktopError.message(chat.autosaveError ?? "Chat autosave failed") }
                }
                _ = try await bridge.request(command, arguments)
                error = nil
            } catch { self.error = error.localizedDescription }
            busy = false
            await refresh()
        }
    }

    func request(_ command: String, _ arguments: [String: Any] = [:]) async throws -> JSON {
        try await bridge.request(command, arguments)
    }

    func savePreferences(_ changes: [String: Any]) {
        perform("preferences", ["changes": changes])
    }

    func refreshLogs() async {
        do { logs = try await bridge.request("logs", ["source": logSource, "profile_id": logProfileID])["text"].text }
        catch { logs = "Unable to read logs: \(error.localizedDescription)" }
    }

    func start(_ row: JSON) {
        if let profile = preferences["profiles"].array.first(where: { $0["path"].text == row["path"].text }) {
            perform("profile_start", ["profile_id": profile["id"].text])
        } else { perform("start", ["path": row["path"].text, "identity": row["identity"].text]) }
    }

    func stop(_ row: JSON) {
        if let active = workerFor(row), let profile = active["profile_id"].string, profile != "default" {
            perform("profile_stop", ["profile_id": profile])
        } else { perform("stop", ["path": row["path"].text]) }
    }

    func trash(_ row: JSON) {
        let alert = NSAlert()
        alert.messageText = "Move \(row["name"].text) to Trash?"
        alert.informativeText = "\(bytes(row["allocated_bytes"].double)) on disk. You can restore the folder from Trash. Active models must be stopped first."
        alert.alertStyle = .warning
        alert.addButton(withTitle: "Move to Trash")
        alert.addButton(withTitle: "Cancel")
        if alert.runModal() == .alertFirstButtonReturn {
            perform("trash", ["path": row["path"].text, "identity": row["identity"].text])
        }
    }

    func chooseFolder() {
        let panel = NSOpenPanel()
        panel.canChooseFiles = false
        panel.canChooseDirectories = true
        panel.allowsMultipleSelection = false
        panel.prompt = "Use Model Folder"
        if panel.runModal() == .OK, let url = panel.url {
            var settings = snapshot["settings"].raw as? [String: Any] ?? [:]
            settings["root"] = url.path
            perform("configure", ["settings": settings])
        }
    }

    func reveal(_ row: JSON) {
        guard let path = row["path"].string else { return }
        NSWorkspace.shared.activateFileViewerSelecting([URL(fileURLWithPath: path)])
    }

    func chatTarget(_ server: JSON) async throws -> JSON {
        if server["endpoint"]["remote"].bool {
            return try await bridge.request("chat_target", ["remote": true, "nid": server["endpoint"]["nid"].text,
                                                           "model": server["endpoint"]["remote_model"].text])
        }
        return try await bridge.request("chat_target", ["url": server["endpoint"]["url"].text, "pid": server["endpoint"]["pid"].raw])
    }

    func chatTools(_ server: JSON) async throws -> JSON {
        var target: [String: Any] = ["url": server["endpoint"]["url"].text, "pid": server["endpoint"]["pid"].raw]
        if server["endpoint"]["remote"].bool {
            target = ["remote": true, "nid": server["endpoint"]["nid"].text, "model": server["endpoint"]["remote_model"].text]
        }
        let goal = chat.conversations[chat.selected]?.last(where: { $0.role == "user" && $0.origin.isEmpty })?.text ?? ""
        return try await bridge.request("chat_tools", ["target": target, "chat_session_id": chat.selected, "task_goal": String(goal.prefix(2000))])
    }

    func finishResearchPlan(sessionID: String, outcome: String, parentID: String) async {
        _ = try? await bridge.request("research_plan_finish", ["session_id": sessionID, "outcome": outcome, "parent_id": parentID])
        await refresh()
    }

    func chatToolCall(sessionID: String, name: String, arguments: [String: Any], approved: Bool) async throws -> JSON {
        let operation = ChatModel.effectiveToolName(name, arguments: arguments)
        var request: [String: Any] = ["session_id": sessionID, "name": name, "arguments": arguments, "approved": approved]
        if ["mlx2__agent_fork", "mlx2__agent_launch"].contains(operation) {
            let history = chat.completedForkHistory(for: chat.selected)
            guard try JSONSerialization.data(withJSONObject: history).count <= 2_000_000 else {
                throw DesktopError.message("This conversation is too large to fork directly. Compact it first; the full session remains in the archive.")
            }
            request["history"] = history
        }
        let response = try await bridge.request("tool_call", request)
        if operation == "mlx2__choice_panel", response["native_action"].text == "choice_panel" {
            let selection = try await choices.present(response["panel"], world: section == "World")
            let receipt = try await bridge.request("choice_resolve", selection)
            return JSON(["result": receipt.raw])
        }
        if operation == "mlx2__login_startup", response["native_action"].text == "login_startup", approved {
            try LoginItems.setEnabled(response["enabled"].bool)
            return JSON(["result": ["enabled": LoginItems.enabled, "status": LoginItems.status]])
        }
        let action = response["result"]["native_action"].string != nil ? response["result"] : response
        if approved && operation == "mlx2__ui_restart" && action["native_action"].text == "ui_restart" {
            let prepared = try await bridge.request("lifecycle_prepare_restart", ["app_path": Bundle.main.bundleURL.path,
                "native_pid": Int(ProcessInfo.processInfo.processIdentifier)])
            LifecycleActions.schedule(prepared)
            return JSON(["result": prepared.raw])
        }
        if approved && operation == "mlx2__ui_close" && action["native_action"].text == "ui_close" { LifecycleActions.schedule(action) }
        return response
    }

    @Published private(set) var profileTransitioning = false
    private func finishProfileTransition() {
        profileTransitioning = false; chat.mutationsPaused = false
        profileChats.resume()
        codeCoordinator.resumeAfterProfileChange()
        wakeupCoordinator.start(self)
    }

    func shutdown() async -> Bool {
        busy = true
        while polling { try? await Task.sleep(for: .milliseconds(50)) }
        profileTransitioning = true; chat.mutationsPaused = true
        profileChats.pause()
        await wakeupCoordinator.suspendForProfileChange()
        await codeCoordinator.suspendForProfileChange()
        choices.cancel()
        await chat.stopAndWait()
        guard await profileChats.flush() else {
            busy = false
            finishProfileTransition()
            return false
        }
        await codeCoordinator.close()
        await profileChats.close()
        timer?.cancel()
        await bridge.finish()
        return true
    }
}
