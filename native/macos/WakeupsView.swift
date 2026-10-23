import SwiftUI

struct WakeupsView: View {
    @EnvironmentObject var model: DesktopModel
    @State private var message = ""
    @State private var mode = "interval"
    @State private var seconds = "3600"
    @State private var date = Date().addingTimeInterval(3600)
    @State private var suspendAfter = true
    @State private var creating = false
    @State private var offset = 0
    @State private var page = JSON()
    @State private var error: String?
    private var schedules: [JSON] { page["schedules"].array }
    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 14) {
                Text("Loops & Wakeups").font(.title2)
                GroupBox("Schedule work in the current chat") {
                    VStack(alignment: .leading, spacing: 9) {
                        Text(model.chat.historyTitle(model.chat.selected)).font(.caption).foregroundStyle(.secondary)
                        TextField("Message to send when the session wakes", text: $message, axis: .vertical)
                        Picker("Schedule", selection: $mode) { Text("Repeat at an interval").tag("interval"); Text("Once at a date and time").tag("date") }
                        if mode == "interval" { TextField("Interval in seconds", text: $seconds).frame(maxWidth: 220) }
                        else { DatePicker("Wake at", selection: $date, displayedComponents: [.date, .hourAndMinute]) }
                        Toggle("Suspend the model again when the turn finishes", isOn: $suspendAfter)
                        Button(creating ? "Scheduling…" : "Create Schedule") { Task { await create() } }
                            .disabled(creating || message.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty || model.chat.selected.isEmpty)
                    }.padding(8)
                }
                Text("A model can use wakeup_schedule to schedule an interval or an exact date and time in its current branch. Wakeups resume the same suspended model and can suspend it again after the turn. The menu-bar app must remain running; quitting the app stops delivery until it is reopened.").foregroundStyle(.secondary)
                HStack {
                    Text("\(number(page["total"].double, digits: 0)) saved schedules")
                    Spacer()
                    Button("Previous") { offset = max(0, offset - 100); Task { await refresh() } }.disabled(offset == 0)
                    Button("Next") { offset += 100; Task { await refresh() } }.disabled(offset + 100 >= Int(page["total"].double ?? 0))
                    Button("Refresh") { Task { await refresh() } }
                }
                if let error { Text(error).foregroundStyle(.orange) }
                if schedules.isEmpty { Text("No wakeups in this user profile.").foregroundStyle(.secondary) }
                ForEach(Array(schedules.enumerated()), id: \.offset) { _, item in
                    GroupBox {
                        VStack(alignment: .leading, spacing: 8) {
                            HStack {
                                Text(item["state"].text).font(.headline)
                                Spacer()
                                Text(timestamp(item["next_due"].double))
                                if !["cancelled", "completed"].contains(item["state"].text) {
                                    Button("Cancel") { model.perform("wakeup_cancel", ["wakeup_id": item["id"].text]); Task { await refresh() } }
                                }
                                if ["blocked", "interrupted", "failed"].contains(item["state"].text) {
                                    Button("Retry") { model.perform("wakeup_retry", ["wakeup_id": item["id"].text]); Task { await refresh() } }
                                }
                            }
                            Text(item["message"].text).textSelection(.enabled)
                            Text("Session: " + item["session_id"].text).font(.caption.monospaced()).textSelection(.enabled)
                            if let interval = item["every_seconds"].double { Text("Every \(number(interval, suffix: " seconds"))").font(.caption) }
                            if let warning = item["error"].string { Text(warning).foregroundStyle(.orange).font(.caption) }
                        }.padding(8)
                    }
                }
            }.padding(20)
        }.task { await refresh() }
        .onChange(of: model.snapshot["wakeups"]["total"].double) { _, _ in Task { await refresh() } }
    }
    @MainActor private func create() async {
        guard let server = model.servers.first(where: { ChatModel.identity($0) == model.chat.serverIdentity(for: model.chat.selected) }) else {
            error = "Select a running app-managed model in Chat first."; return
        }
        creating = true; defer { creating = false }
        do {
            let target = try await model.chatTarget(server)
            var arguments: [String: Any] = ["session_id": model.chat.selected, "_target": target.raw,
                "message": message, "suspend_after": suspendAfter, "every_seconds": NSNull(), "at": NSNull()]
            if mode == "interval" {
                guard let interval = Double(seconds), interval.isFinite, interval >= 1 else { throw DesktopError.message("Choose an interval of at least one second.") }
                arguments["every_seconds"] = interval
            } else { arguments["at"] = ISO8601DateFormatter().string(from: date) }
            _ = try await model.request("wakeup_schedule", arguments)
            message = ""; error = nil; await refresh()
        } catch { self.error = error.localizedDescription }
    }
    @MainActor private func refresh() async {
        do { page = try await model.request("wakeup_list", ["offset": offset, "limit": 100]); error = nil }
        catch { self.error = error.localizedDescription }
    }
}
