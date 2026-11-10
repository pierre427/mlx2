import AppKit
import SwiftUI

struct DownloadsView: View {
    @EnvironmentObject var model: DesktopModel
    @State private var target = ""
    @State private var adapterAfter = false
    @State private var parallelism = 5
    @State private var throttle = "0"
    @State private var token = ""
    @State private var tokenFile = ""
    @State private var savingToken = false
    private var jobs: [JSON] { model.jobs.filter { $0["kind"].text == "download" } }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                Text("Model downloads").font(.title2.weight(.semibold))
                GroupBox("Hugging Face") {
                    VStack(alignment: .leading, spacing: 12) {
                        TextField("https://huggingface.co/owner/model or owner/model@revision", text: $target)
                            .accessibilityLabel("Hugging Face model link or repository")
                        Toggle("Request an adapter draft after download", isOn: $adapterAfter)
                        Text("The repository revision is resolved to an exact commit before downloading. Files are verified and the model appears in the library only after completion. Downloading never loads or qualifies a model.")
                            .font(.caption).foregroundStyle(.secondary)
                        Button("Download Model", systemImage: "arrow.down.circle") {
                            model.perform("download_start", ["target": target, "adapter_after": adapterAfter])
                        }.disabled(target.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty || model.busy)
                    }.padding(8)
                }
                GroupBox("Global download settings") {
                    VStack(alignment: .leading, spacing: 12) {
                        Stepper("\(parallelism) parallel file downloads", value: $parallelism, in: 1...32)
                        TextField("Global bandwidth cap (MiB/s, 0 = unlimited)", text: $throttle)
                        Text("The bandwidth cap is shared across all downloads. Parallelism defaults to five. Change parallelism when downloads are idle.")
                            .font(.caption).foregroundStyle(.secondary)
                        TextField("Protected HF token file (optional)", text: $tokenFile)
                        HStack {
                            SecureField("HF personal access token", text: $token)
                            Button(savingToken ? "Saving…" : "Save Token") { Task { await saveToken() } }
                                .disabled(token.isEmpty || savingToken || model.busy)
                        }
                        Text("Saved tokens stay in an owner-only file in Application Support; they are excluded from model requests, telemetry and job logs.")
                            .font(.caption).foregroundStyle(.secondary)
                        Button("Save Download Settings") {
                            guard let rate = Double(throttle.trimmingCharacters(in: .whitespaces)), rate.isFinite, rate >= 0, rate <= 1048576 else { model.error = "Enter a nonnegative MiB/s limit."; return }
                            model.savePreferences(["download_parallelism": parallelism,
                                                   "download_bytes_per_second": Int(rate * 1048576), "hf_token_file": tokenFile])
                        }
                    }.padding(8)
                }
                if jobs.isEmpty { ContentUnavailableView("No downloads yet", systemImage: "arrow.down.circle", description: Text("Paste a Hugging Face model link above to download without loading a model.")) }
                ForEach(Array(jobs.enumerated()), id: \.offset) { _, job in
                    GroupBox {
                        VStack(alignment: .leading, spacing: 8) {
                            HStack {
                                Text(job["spec"]["repository"].text).font(.headline)
                                Spacer(); Text(job["state"].text.capitalized).foregroundStyle(.secondary)
                                if ["queued", "running"].contains(job["state"].text) {
                                    Button("Cancel") { model.perform("download_cancel", ["job_id": job["id"].text]) }
                                }
                                if ["failed", "cancelled", "interrupted"].contains(job["state"].text) {
                                    Button("Resume") { model.perform("download_resume", ["job_id": job["id"].text]) }
                                }
                            }
                            let progress = model.snapshot["downloads"][job["id"].text]
                            if let total = progress["total_bytes"].double, total > 0 {
                                ProgressView(value: min(total, progress["downloaded_bytes"].double ?? 0), total: total)
                                Text("\(bytes(progress["downloaded_bytes"].double)) / \(bytes(total)) · \(number(progress["files_done"].double, digits: 0)) \(progress["files_done"].double == 1 ? "file" : "files")")
                                    .font(.caption).foregroundStyle(.secondary)
                            }
                            Text(job["progress"].text).font(.callout).textSelection(.enabled)
                            if let commit = progress["commit"].string ?? job["result"]["resolved_revision"].string {
                                Text(commit).font(.caption.monospaced()).foregroundStyle(.secondary).textSelection(.enabled)
                            }
                            if let path = job["result"]["model_path"].string {
                                Button("Reveal Model") { NSWorkspace.shared.activateFileViewerSelecting([URL(fileURLWithPath: path)]) }
                            }
                        }.padding(8)
                    }
                }
            }.padding(20)
        }.onAppear {
            parallelism = Int(model.preferences["download_parallelism"].double ?? 5)
            // Parsed back with Double(_:), so format locale-independently and keep
            // enough digits that a small nonzero cap never reads as 0 (unlimited).
            throttle = ((model.preferences["download_bytes_per_second"].double ?? 0) / 1048576).formatted(.number.locale(Locale(identifier: "en_US_POSIX")).grouping(.never).precision(.fractionLength(0...6)))
            tokenFile = model.preferences["hf_token_file"].string ?? ""
        }
    }
    @MainActor private func saveToken() async {
        guard !savingToken else { return }
        let submitted = token
        savingToken = true; defer { savingToken = false }
        do {
            let result = try await model.request("hf_save_token", ["token": submitted])
            if token == submitted { token = "" }
            tokenFile = result["token_file"].string ?? tokenFile
            await model.refresh()
        } catch { model.error = error.localizedDescription }
    }
}

struct ConnectedToolsSettings: View {
    @EnvironmentObject var model: DesktopModel
    @State private var name = ""
    @State private var url = "http://127.0.0.1:"
    @State private var token = ""
    @State private var allowed = ""
    @State private var autoApprove = false
    @State private var toolList = ""
    @State private var studioURL = ""
    @State private var adding = false
    @State private var testing = false
    @State private var tested = false
    @State private var removeID: String?
    var body: some View {
        Section("Chat tools and sister applications") {
            Toggle("Expose MLX2 controls to chat", isOn: Binding(get: { model.preferences["agent_controls"].bool }, set: { model.savePreferences(["agent_controls": $0]) }))
            Text("Chat controls use the same validated operations as the UI. Changes request approval in chat; reading status does not.").font(.caption).foregroundStyle(.secondary)
            Label("Web search and page fetching are built in", systemImage: "globe").font(.caption)
            Text("Use web_search and fetch_page in chat. They run locally without an MCP connection.").font(.caption).foregroundStyle(.secondary)
            ForEach(Array(model.preferences["mcp_servers"].array.enumerated()), id: \.offset) { _, server in
                HStack {
                    VStack(alignment: .leading) { Text(server["id"].text); Text(server["url"].text).font(.caption).foregroundStyle(.secondary) }
                    Spacer()
                    Text(server["auto_approve"].bool ? "Automatic calls" : "Ask for approval").font(.caption)
                    Button("Remove") { removeID = server["id"].text }
                }
            }
            TextField("MCP connection name", text: $name)
            TextField("MCP Streamable HTTP URL", text: $url)
            SecureField("Bearer token (optional)", text: $token)
            TextField("Allowed tool names, comma separated (empty = all)", text: $allowed)
            Toggle("Allow this connection's tools without approval", isOn: $autoApprove)
            Button(adding ? "Adding…" : "Add MCP Connection") {
                guard !adding else { return }
                adding = true
                Task {
                    defer { adding = false }
                    do {
                        var reference = ""
                        if !token.isEmpty {
                            reference = try await model.request("mcp_save_secret", ["headers": ["Authorization": "Bearer " + token]])["header_ref"].text
                        }
                        var rows = model.preferences["mcp_servers"].array.map { $0.raw as? [String: Any] ?? [:] }
                        rows.append(["id": name, "url": url, "enabled": true,
                                     "allowed_tools": allowed.split(separator: ",").map { $0.trimmingCharacters(in: .whitespaces) }.filter { !$0.isEmpty },
                                     "auto_approve": autoApprove, "header_ref": reference])
                        _ = try await model.request("preferences", ["changes": ["mcp_servers": rows]])
                        token = ""; name = ""; await model.refresh()
                    } catch { model.error = error.localizedDescription }
                }
            }.disabled(name.trimmingCharacters(in: .whitespaces).isEmpty || adding)
            HStack {
                Button(testing ? "Testing…" : "Test Connections and List Tools") {
                    guard !testing else { return }
                    testing = true
                    Task {
                        defer { testing = false }
                        do {
                            let result = try await model.request("tool_discover")
                            toolList = result["tools"].array.map { $0["function"]["name"].text }.joined(separator: "\n")
                            tested = true
                        } catch { model.error = error.localizedDescription }
                    }
                }.disabled(testing)
                if testing { ProgressView().controlSize(.small) }
            }
            if !toolList.isEmpty { Text(toolList).font(.caption.monospaced()).textSelection(.enabled) }
            else if tested { Text("The connections responded but listed no tools.").font(.caption).foregroundStyle(.secondary) }
            TextField("MLX Studio URL (optional)", text: $studioURL)
            Button("Save Studio URL") { model.savePreferences(["studio_url": studioURL]) }
            if let url = URL(string: model.preferences["studio_url"].text), ["http", "https"].contains(url.scheme ?? "") {
                Link("Open MLX Studio", destination: url)
            }
        }.onAppear { studioURL = model.preferences["studio_url"].string ?? "" }
        .confirmationDialog("Remove the MCP connection “\(removeID ?? "")”?", isPresented: Binding(get: { removeID != nil }, set: { if !$0 { removeID = nil } })) {
            Button("Remove Connection", role: .destructive) {
                if let removeID { model.savePreferences(["mcp_servers": model.preferences["mcp_servers"].array.filter { $0["id"].text != removeID }.map(\.raw)]) }
                removeID = nil
            }
            Button("Cancel", role: .cancel) { removeID = nil }
        } message: {
            Text("Chat loses access to this connection's tools. Adding it again requires its URL and token.")
        }
    }
}
