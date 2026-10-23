import AppKit
import SwiftUI

struct EngineProfilesView: View {
    @EnvironmentObject var model: DesktopModel
    @State private var path = ""
    @State private var port = "8286"
    @State private var lanes = 1
    @State private var context = "0"
    @State private var sticky = true
    @State private var startup = false
    @State private var share = false
    @State private var route = "auto"

    private var profiles: [JSON] { model.preferences["profiles"].array }
    private func update(_ profile: JSON, _ key: String, _ value: Any) {
        let rows = profiles.map { row -> [String: Any] in
            var record = row.raw as? [String: Any] ?? [:]
            if row["id"].text == profile["id"].text { record[key] = value }
            return record
        }
        model.savePreferences(["profiles": rows])
    }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                Text("Serving engines").font(.title2.weight(.semibold))
                Text("Saved profiles retain each model's port, lane count, route, startup and community sharing choices. Multiple engines share the app's GPU reservation; memory capacity still applies.")
                    .foregroundStyle(.secondary)
                ForEach(Array(profiles.enumerated()), id: \.offset) { _, profile in
                    let engine = model.workers.first { $0["profile_id"].text == profile["id"].text }
                    let running = model.workers.first { $0["profile_id"].text == profile["id"].text && ["Preparing", "Loading", "Ready", "Busy", "Unreachable", "Stopping", "Suspended", "Suspending", "Resuming", "SuspendError", "ResumeError", "SuspensionRecoveryRequired"].contains($0["state"].text) }
                    GroupBox {
                        VStack(alignment: .leading, spacing: 10) {
                            HStack {
                                Text(URL(fileURLWithPath: profile["path"].text).lastPathComponent).font(.headline)
                                Spacer()
                                Text("\(Int(profile["lanes"].double ?? 1)) lanes · \(profile["sticky_port"].raw is NSNull || profile["sticky_port"].bool ? "port " + number(profile["port"].double, digits: 0) : "automatic port")")
                                    .foregroundStyle(.secondary)
                                if let running {
                                    if running["state"].text == "Suspended" {
                                        Button("Resume") { model.resumeWorker(running) }
                                    } else if ["Ready", "Busy"].contains(running["state"].text) {
                                        Button("Suspend") { model.suspendWorker(running) }
                                    }
                                    Button("Stop") { model.perform("profile_stop", ["profile_id": profile["id"].text]) }
                                } else {
                                    Button("Start") { model.perform("profile_start", ["profile_id": profile["id"].text]) }
                                    Button("Remove", role: .destructive) {
                                        model.savePreferences(["profiles": profiles.filter { $0["id"].text != profile["id"].text }.map(\.raw)])
                                    }
                                }
                            }
                            HStack {
                                Toggle("Start when MLX2 launches", isOn: Binding(get: { profile["autostart"].bool }, set: { update(profile, "autostart", $0) }))
                                Toggle("Serve community", isOn: Binding(get: { profile["share"].bool }, set: { update(profile, "share", $0) }))
                                Spacer()
                                if let engine { Text(engine["state"].text).foregroundStyle(.secondary) }
                            }
                            if let failure = engine?["error"].string {
                                Label(failure, systemImage: "exclamationmark.triangle").font(.callout).foregroundStyle(.orange).textSelection(.enabled)
                            }
                            if running?["state"].text != "Suspended", let endpoint = running?["url"].string {
                                HStack {
                                    Text(endpoint + "/v1").font(.callout.monospaced()).textSelection(.enabled)
                                    Button("Copy URL", systemImage: "doc.on.doc") {
                                        NSPasteboard.general.clearContents()
                                        NSPasteboard.general.setString(endpoint + "/v1", forType: .string)
                                    }
                                }
                            }
                        }.padding(6)
                    }
                }
                GroupBox("Add a serving profile") {
                    VStack(alignment: .leading, spacing: 12) {
                        Picker("Model", selection: $path) {
                            Text("Choose a supported model").tag("")
                            ForEach(Array(model.rows.filter { $0["can_start"].bool }.enumerated()), id: \.offset) { _, row in
                                Text(row["name"].text).tag(row["path"].text)
                            }
                        }
                        HStack {
                            Toggle("Keep this port", isOn: $sticky)
                            TextField("Port", text: $port).frame(width: 90).disabled(!sticky)
                            Picker("Lanes", selection: $lanes) { ForEach(1...4, id: \.self) { Text("\($0)").tag($0) } }.frame(width: 130)
                            TextField("Context (0 = automatic)", text: $context).frame(width: 190)
                        }
                        Picker("Route", selection: $route) { Text("Automatic").tag("auto"); Text("Ordinary reference").tag("ordinary") }
                        HStack {
                            Toggle("Start when MLX2 launches", isOn: $startup)
                            Toggle("Serve admitted community systems", isOn: $share)
                        }
                        Button("Save Profile") {
                            guard let portValue = Int(port), let contextValue = Int(context) else { model.error = "Port and context must be whole numbers."; return }
                            var rows = profiles.map { $0.raw as? [String: Any] ?? [:] }
                            rows.append(["id": UUID().uuidString.lowercased(), "path": path, "port": portValue,
                                         "context": contextValue, "lanes": lanes, "route": route,
                                         "sticky_port": sticky, "autostart": startup, "share": share])
                            model.savePreferences(["profiles": rows])
                        }.disabled(path.isEmpty || model.busy)
                    }.padding(8)
                }
                Text("Auto-start begins after login, when the menu bar app starts. Enable Launch MLX2 at login in Settings. Unattended pre-login model serving requires a separate system service.")
                    .font(.caption).foregroundStyle(.secondary)
            }.padding(20)
        }
    }
}

struct CommunityView: View {
    @EnvironmentObject var model: DesktopModel
    @State private var home = ""
    @State private var front = "8290"
    @State private var fabric = "8291"
    @State private var invite = ""
    @State private var membershipResult = ""
    private var community: JSON { model.snapshot["community"] }

    private func membership(_ command: String, _ arguments: [String: Any] = [:]) {
        Task {
            do {
                let result = try await model.request(command, arguments)
                membershipResult = result["invite"].string ?? (result["joined"].bool ? "Joined community" : "Complete")
                await model.refresh()
            } catch { model.error = error.localizedDescription }
        }
    }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                Text("MLX2 community").font(.title2.weight(.semibold))
                Text("Admitted machines use the existing certificate-authenticated mlx2 fabric. Select a remote model in Chat to keep that conversation on its selected machine.")
                    .foregroundStyle(.secondary)
                GroupBox("Connection") {
                    VStack(alignment: .leading, spacing: 10) {
                        TextField("Community directory", text: $home)
                        HStack { TextField("Local client port", text: $front); TextField("Community TLS port", text: $fabric) }
                        Toggle("Connect when MLX2 launches", isOn: Binding(get: { model.preferences["community_enabled"].bool }, set: { model.savePreferences(["community_enabled": $0]) }))
                        HStack {
                            Button("Save Connection") {
                                guard let front = Int(front), let fabric = Int(fabric) else { model.error = "Ports must be whole numbers."; return }
                                model.savePreferences(["community_home": home, "community_front_port": front, "community_fabric_port": fabric])
                            }
                            if community["connected"].bool {
                                Button("Disconnect") { model.perform("community_disconnect") }
                            } else {
                                Button("Connect") { model.perform("community_connect") }
                            }
                        }
                    }.padding(8)
                }
                GroupBox("Membership") {
                    VStack(alignment: .leading, spacing: 10) {
                        Text("Use the same community invitation on your other Macs. Invitations contain a joining secret; share them privately.").font(.caption).foregroundStyle(.secondary)
                        HStack {
                            Button("Create Community") { membership("community_create") }
                            Button("Get Invitation") { membership("community_invite") }
                        }
                        SecureField("Invitation from an existing member", text: $invite)
                        Button("Join Community") { membership("community_join", ["invite": invite]) }.disabled(invite.isEmpty)
                        if !membershipResult.isEmpty {
                            Text(membershipResult).font(.caption.monospaced()).textSelection(.enabled)
                            Button("Copy Result") { NSPasteboard.general.clearContents(); NSPasteboard.general.setString(membershipResult, forType: .string) }
                        }
                    }.padding(8)
                }
                Toggle("Share the default managed engine", isOn: Binding(get: { model.preferences["default_share"].bool }, set: { model.savePreferences(["default_share": $0]) }))
                Text("Choose sharing per saved engine in Engines. External servers remain under their owner's lifecycle control.").font(.caption).foregroundStyle(.secondary)
                ForEach(Array(community["nodes"].array.enumerated()), id: \.offset) { _, node in
                    GroupBox {
                        VStack(alignment: .leading, spacing: 8) {
                            HStack {
                                Text(node["hostname"].string ?? node["nid"].text).font(.headline)
                                Spacer(); Text(node["state"].text).foregroundStyle(.secondary)
                            }
                            Text("\(node["device_class"].text) · \(bytes(node["free_bytes"].double)) free / \(bytes(node["memory_bytes"].double)) total")
                            Text(node["models"].array.map(\.text).joined(separator: ", ")).font(.caption).textSelection(.enabled)
                        }.padding(8)
                    }
                }
                if community["nodes"].array.isEmpty { Text("No connected community machines yet.").foregroundStyle(.secondary) }
                GroupBox("Share other running local engines") {
                    VStack(alignment: .leading) {
                        ForEach(Array(model.servers.filter { !$0["endpoint"]["remote"].bool && $0["endpoint"]["pid"].double != nil }.enumerated()), id: \.offset) { _, server in
                            let endpoint = server["endpoint"]
                            let saved = model.preferences["shared_servers"].array
                            let selected = saved.contains { $0["url"].text == endpoint["url"].text && $0["pid"].double == endpoint["pid"].double }
                            Toggle(server["model"].string ?? endpoint["url"].text, isOn: Binding(get: { selected }, set: { enabled in
                                var rows = saved.filter { $0["url"].text != endpoint["url"].text }.map { $0.raw as? [String: Any] ?? [:] }
                                if enabled { rows.append(["url": endpoint["url"].text, "pid": Int(endpoint["pid"].double ?? 0)]) }
                                model.savePreferences(["shared_servers": rows])
                            }))
                        }
                    }.padding(8)
                }
            }.padding(20)
        }.onAppear {
            home = model.preferences["community_home"].string ?? ""
            front = number(model.preferences["community_front_port"].double, digits: 0)
            fabric = number(model.preferences["community_fabric_port"].double, digits: 0)
        }
    }
}

struct ContextSettingsView: View {
    @EnvironmentObject var model: DesktopModel
    @State private var global = "32768"
    @State private var reserve = "4096"
    @State private var memoryBudget = "1024"
    @State private var path = ""
    @State private var override = "0"
    @State private var agentOutput = "65536"
    @State private var agentReasoning = "32768"
    @State private var agentThinking = true
    @State private var agentVerification = false
    var body: some View {
        Section("Context budgets") {
            TextField("Global context default (tokens)", text: $global)
            TextField("Reserved output tokens", text: $reserve)
            TextField("Maximum memory retrieval (tokens)", text: $memoryBudget)
            Toggle("Enable agent memory and session messaging", isOn: Binding(get: { model.preferences["memory_enabled"].bool }, set: { model.savePreferences(["memory_enabled": $0]) }))
            Text("Context defaults apply on the next model load. Actual model and server limits still apply. Chat reserves output room and removes complete older turns when needed; current attachments and tool-call groups remain intact. Memory retrieval is capped at 10% of the active context.")
                .font(.caption).foregroundStyle(.secondary)
            Picker("Per-model context", selection: $path) {
                Text("Choose a model").tag("")
                ForEach(Array(model.rows.filter { $0["can_start"].bool }.enumerated()), id: \.offset) { _, row in Text(row["name"].text).tag(row["path"].text) }
            }.onChange(of: path) { _, _ in
                override = String(Int(model.preferences["model_context_tokens"][path].double ?? 0))
                let options = model.preferences["model_chat_options"][path]
                agentOutput = String(Int(options["output_tokens"].double ?? 65536))
                agentReasoning = String(Int(options["thinking_budget"].double ?? 32768))
                agentThinking = options["enable_thinking"].raw is NSNull ? true : options["enable_thinking"].bool
                agentVerification = options["verification"].bool
            }
            TextField("Model context override (0 = global default)", text: $override).disabled(path.isEmpty)
            Group {
                Text("Agentic chat profile for selected model").font(.headline)
                TextField("Output token budget", text: $agentOutput)
                TextField("Reasoning token budget", text: $agentReasoning)
                Toggle("Enable thinking", isOn: $agentThinking)
                Toggle("Verification self-review (supported templates only)", isOn: $agentVerification)
                Button("Use 256K research profile") {
                    override = "262144"; agentOutput = "65536"; agentReasoning = "32768"; agentThinking = true
                }
                Button("Save selected model's agentic profile") {
                    guard let output = Int(agentOutput), let reasoning = Int(agentReasoning), let context = Int(override), context > 0 else {
                        model.error = "Select an explicit context and whole-number agent budgets."; return
                    }
                    var options = model.preferences["model_chat_options"].raw as? [String: Any] ?? [:]
                    options[path] = ["context_limit": context, "output_tokens": output,
                                     "thinking_budget": reasoning, "enable_thinking": agentThinking, "verification": agentVerification]
                    var contexts = model.preferences["model_context_tokens"].raw as? [String: Int] ?? [:]
                    contexts[path] = context
                    model.savePreferences(["model_chat_options": options, "model_context_tokens": contexts])
                }
                Text("The 256K context includes input and output. Verification is learned self-review, separate from speculative decoding. Subagents use spawn_subagent, isolated contexts and scoped tools; approval and cancellation remain available.")
                    .font(.caption).foregroundStyle(.secondary)
            }.disabled(path.isEmpty)
            Button("Save Context Settings") {
                guard let global = Int(global), let reserve = Int(reserve), let budget = Int(memoryBudget), let specific = Int(override) else { model.error = "Context budgets must be whole numbers."; return }
                var overrides = model.preferences["model_context_tokens"].raw as? [String: Int] ?? [:]
                if !path.isEmpty { overrides[path] = specific }
                model.savePreferences(["global_context_tokens": global, "context_reserve_tokens": reserve,
                                       "memory_retrieval_tokens": budget, "model_context_tokens": overrides])
            }
        }.onAppear {
            global = String(Int(model.preferences["global_context_tokens"].double ?? 32768))
            reserve = String(Int(model.preferences["context_reserve_tokens"].double ?? 4096))
            memoryBudget = String(Int(model.preferences["memory_retrieval_tokens"].double ?? 1024))
        }
    }
}

struct ModelLaunchOptionsView: View {
    @EnvironmentObject var model: DesktopModel
    let row: JSON
    @State private var lanes = 1
    @State private var port = "8285"
    @State private var sticky = true
    @State private var context = "0"
    private var profile: JSON? { model.preferences["profiles"].array.first { $0["path"].text == row["path"].text } }
    var body: some View {
        HStack(spacing: 12) {
            Picker("Lanes", selection: $lanes) { ForEach(1...4, id: \.self) { Text("\($0)").tag($0) } }.frame(width: 120)
            Toggle("Sticky port", isOn: $sticky)
            TextField("Port", text: $port).frame(width: 70).disabled(!sticky)
            TextField("Context: 0 = global", text: $context).frame(width: 135)
            Button("Save Launch Options") {
                guard let port = Int(port), let context = Int(context) else { model.error = "Port and context must be whole numbers."; return }
                var profiles = model.preferences["profiles"].array.map { $0.raw as? [String: Any] ?? [:] }
                let identity = profile?["id"].string ?? UUID().uuidString.lowercased()
                let saved: [String: Any] = ["id": identity, "path": row["path"].text, "port": port, "lanes": lanes,
                                            "context": context, "route": profile?["route"].string ?? "auto",
                                            "sticky_port": sticky, "autostart": profile?["autostart"].bool ?? false, "share": profile?["share"].bool ?? false]
                profiles.removeAll { ($0["id"] as? String) == identity }
                profiles.append(saved)
                model.savePreferences(["profiles": profiles])
            }.disabled(model.owns(row) || model.busy)
        }.font(.caption).onAppear {
            let values = profile ?? model.snapshot["settings"]
            lanes = max(1, min(4, Int(values["lanes"].double ?? 1)))
            port = String(Int(values["port"].double ?? 8285))
            context = String(Int(values["context"].double ?? 0))
            sticky = values["sticky_port"].raw is NSNull ? true : values["sticky_port"].bool
        }
    }
}
