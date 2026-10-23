import AppKit
import AVKit
import SwiftUI
import UniformTypeIdentifiers

/// A small client of the shared Studio media job services, never another owner
/// of their models. Tool-specific options remain available as JSON.
struct MediaView: View {
    @EnvironmentObject var model: DesktopModel
    @State private var capability = "image"
    @State private var providerID = "mlx2-image"
    @State private var backendID = "selected"
    @State private var prompt = ""
    @State private var imageOperation = "generate"
    @State private var references = ""
    @State private var chooseReferences = false
    @State private var lyrics = ""
    @State private var width = "1024"
    @State private var height = "1024"
    @State private var duration = "30"
    @State private var options = "{}"
    @State private var serviceJSON = ""
    @State private var validation: String?
    @State private var directPython = ""
    @State private var canonicalSource = ""
    @State private var videoSource = ""
    @State private var videoModel = ""
    @State private var videoRuntime = ""
    @State private var confirmSharedStop = false
    @State private var voiceReference = ""
    @State private var chooseVoiceReference = false

    private var media: JSON { model.snapshot["media"] }
    private var services: [JSON] { media["services"].array.filter { $0["capability"].text == capability } }
    private var selected: JSON { services.first { $0["id"].text == providerID } ?? JSON() }
    private var images: [JSON] {
        media["providers"].array.first { $0["id"].text == "mlx2-image" }?["models"].array ?? []
    }
    private var remoteModels: [JSON] {
        let value = media["providers"].array.first { $0["id"].text == providerID }?["models"] ?? JSON()
        if !value.array.isEmpty { return value.array }
        if !value["models"].array.isEmpty { return value["models"].array }
        if !value["backends"].array.isEmpty { return value["backends"].array }
        return providerID == "mlx2-video" ? media["direct"]["video_models"].array : []
    }
    private var hasUsableModel: Bool {
        if capability == "image" && providerID == "mlx2-image" { return images.contains { $0["available"].bool } }
        if (selected["models_tool"].string ?? "").isEmpty { return true }
        return remoteModels.contains { $0["available"].bool || ($0["available"].raw is NSNull && ($0["ready"].bool || $0["ready"].raw is NSNull)) }
    }
    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                HStack {
                    Text("Media & Services").font(.largeTitle.weight(.semibold))
                    Spacer()
                    if let url = URL(string: media["studio_url"].text), ["http", "https"].contains(url.scheme ?? "") {
                        Link("Open MLX Studio", destination: url)
                    }
                    Button("Refresh") { model.perform("media_refresh") }
                }
                Text("Generate with the same model services and durable jobs used by MLX Studio. Results retain their service receipts and artifact paths.")
                    .foregroundStyle(.secondary)
                Picker("Capability", selection: $capability) {
                    Text("Images").tag("image")
                    Text("Video").tag("video")
                    Text("Music").tag("music")
                    Text("Voice").tag("voice")
                    Text("Voice training").tag("voice_training")
                }.pickerStyle(.segmented)
                    .onChange(of: capability) { _, value in
                        backendID = "selected"
                        providerID = value == "image" ? "mlx2-image" : value == "music" ? "yue2" : services.first?["id"].text ?? ""
                    }
                if services.isEmpty {
                    ContentUnavailableView("Connect a \(capability) service", systemImage: "cable.connector", description: Text("Add its MCP endpoint and generation/status tools below. The service owns model loading, accelerator admission and output receipts."))
                } else {
                    Picker("Model service", selection: $providerID) {
                        ForEach(services, id: \.rawID) { service in Text(service["label"].text).tag(service["id"].text) }
                    }
                    servicePanel
                    GroupBox("Generate") {
                        VStack(alignment: .leading, spacing: 12) {
                            if capability == "image" {
                                Picker("Operation", selection: $imageOperation) { Text("Text to image").tag("generate"); Text("Edit image").tag("edit") }.pickerStyle(.segmented)
                                if imageOperation == "edit" {
                                    TextField("Reference image paths (one per line)", text: $references, axis: .vertical).lineLimit(2...5)
                                    Button("Attach reference images") { chooseReferences = true }
                                }
                            }
                            if capability == "image" && providerID == "mlx2-image" {
                                Picker("Model", selection: $backendID) {
                                    Text("Service default").tag("selected")
                                    ForEach(images.filter { $0["available"].bool && (imageOperation == "generate" || $0["operations"].array.contains { $0.text == "edit" }) }, id: \.rawID) { item in
                                        Text(item["label"].text).tag(item["id"].text)
                                    }
                                }
                            }
                            if providerID != "mlx2-image" && !remoteModels.isEmpty {
                                Picker("Model", selection: $backendID) {
                                    Text("Service default").tag("selected")
                                    ForEach(Array(remoteModels.enumerated()), id: \.offset) { _, item in
                                        Text((item["label"].string ?? item["name"].string ?? item["id"].text) + (item["available"].bool ? "" : " · unavailable")).tag(item["id"].string ?? item["name"].text)
                                    }
                                }
                            }
                            if capability != "music" && remoteModels.isEmpty && !(capability == "image" && providerID == "mlx2-image" && !images.isEmpty) {
                                Text("No model routes discovered for this service. Connect or start the service, then refresh. Generation stays disabled until a usable route is advertised.").font(.caption).foregroundStyle(.secondary)
                            }
                            Text(capability == "music" ? "Caption" : capability == "voice" ? "Text to speak" : "Prompt").font(.headline)
                            if capability == "voice" {
                                TextField("Speaker reference WAV (optional; required by reference-conditioned routes)", text: $voiceReference)
                                Button("Use an audio or video voice sample") { chooseVoiceReference = true }
                                    .fileImporter(isPresented: $chooseVoiceReference, allowedContentTypes: [.audio, .movie], allowsMultipleSelection: false) { result in
                                        Task {
                                            do {
                                                guard let url = try result.get().first else { return }
                                                let access = url.startAccessingSecurityScopedResource()
                                                defer { if access { url.stopAccessingSecurityScopedResource() } }
                                                let reference = try await model.request("media_prepare_voice_reference", ["path": url.path])
                                                voiceReference = reference["path"].text
                                            } catch { validation = error.localizedDescription }
                                        }
                                    }
                                Text("Reference audio conditions a generation. It does not train the voice model. The shared Studio service verifies artifact/runtime prerequisites and owns its durable jobs.").font(.caption).foregroundStyle(.secondary)
                            }
                            if capability == "voice_training" {
                                Text("A training provider must advertise an actual train-voice route and its required data. Reference conditioning is not presented as training; an absent trainer remains unavailable.").font(.caption).foregroundStyle(.secondary)
                            }
                            TextEditor(text: $prompt).frame(minHeight: 90).font(.body)
                                .overlay(RoundedRectangle(cornerRadius: 5).stroke(.quaternary))
                            if capability == "music" {
                                Text("Lyrics (optional)").font(.caption)
                                TextEditor(text: $lyrics).frame(minHeight: 65)
                                TextField("Duration in seconds", text: $duration)
                            } else if capability == "image" {
                                HStack { TextField("Width", text: $width); Text("×"); TextField("Height", text: $height) }
                            }
                            DisclosureGroup("Additional service parameters (JSON)") {
                                TextEditor(text: $options).font(.system(.body, design: .monospaced)).frame(height: 100)
                                Text("These fields are merged into the service request. Choose a video model using its service's model/backend parameter.").font(.caption).foregroundStyle(.secondary)
                            }
                            HStack {
                                Text("Route remains unqualified until separately tested.").font(.caption).foregroundStyle(.secondary)
                                Spacer()
                                Button("Generate") { generate() }.buttonStyle(.borderedProminent)
                                    .disabled(prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty || !selected["configured"].bool || !selected["ready"].bool || !hasUsableModel)
                            }
                        }.padding(8)
                    }
                }
                if let validation { Label(validation, systemImage: "exclamationmark.triangle").foregroundStyle(.orange) }
                jobPanel
                directSettings
                DisclosureGroup("Connect or configure a media service") {
                    VStack(alignment: .leading, spacing: 10) {
                        Text("Use a capability-specific MCP service shared with Studio. Start/Stop manages only the explicitly listed local LaunchAgents. Stop refuses known active or unresolved jobs.").font(.caption).foregroundStyle(.secondary)
                        TextEditor(text: $serviceJSON).font(.system(.caption, design: .monospaced)).frame(height: 180)
                        HStack {
                            Button("Load service profiles") { loadProfiles() }
                            Button("Add video example") { addVideoExample() }
                            Spacer()
                            Button("Save profiles") { saveProfiles() }
                        }
                    }.padding(.top, 8)
                }
            }.padding(24)
        }.confirmationDialog("Stop this shared service?", isPresented: $confirmSharedStop) {
            Button("Stop shared service", role: .destructive) { model.perform("service_stop", ["service_id": providerID, "force_shared": true]) }
            Button("Cancel", role: .cancel) { }
        } message: {
            Text("This service lacks atomic admission drain. MLX2 checks known active jobs, but another Studio client could submit work after that check. Confirm only after coordinating other clients.")
        }.onAppear { loadDirectSettings() }
        .fileImporter(isPresented: $chooseReferences, allowedContentTypes: [.png, .jpeg, UTType(filenameExtension: "webp") ?? .image], allowsMultipleSelection: true) { result in
            do {
                let urls = try result.get()
                let folder = URL(fileURLWithPath: media["direct"]["settings"]["input_root"].text)
                try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true, attributes: [.posixPermissions: 0o700])
                var paths = references.split(separator: "\n").map(String.init)
                guard paths.count + urls.count <= 10 else { validation = "Attach at most ten reference images."; return }
                for url in urls {
                    let access = url.startAccessingSecurityScopedResource()
                    defer { if access { url.stopAccessingSecurityScopedResource() } }
                    let attributes = try FileManager.default.attributesOfItem(atPath: url.path)
                    guard let size = attributes[.size] as? NSNumber, size.int64Value <= 33_554_432 else { validation = "Reference images must be at most 32 MiB."; return }
                    let target = folder.appendingPathComponent(UUID().uuidString + "." + url.pathExtension.lowercased())
                    try FileManager.default.copyItem(at: url, to: target)
                    paths.append(target.path)
                }
                references = paths.joined(separator: "\n")
                validation = nil
            } catch { validation = error.localizedDescription }
        }
    }
    private var servicePanel: some View {
        GroupBox("Shared service") {
            VStack(alignment: .leading, spacing: 9) {
                HStack {
                    Label(selected["ready"].bool ? "Ready" : selected["reachable"].bool ? "Connected" : "Not connected", systemImage: selected["ready"].bool ? "checkmark.circle" : "circle.dotted")
                    Spacer()
                    if !selected["configured"].bool {
                        Button("Use this service") { model.perform("media_use_known", ["service_id": providerID]) }
                    } else {
                        if selected["transport"].text == "mlx2-media" {
                            Button("Prepare / Start") { model.perform("media_canonical_start") }
                            Button("Stop") { confirmSharedStop = true }
                            Button("Refresh") { model.perform("media_refresh") }
                        } else {
                        Button("Start") { model.perform("service_start", ["service_id": providerID]) }
                        Button("Stop") {
                            if selected["drain_tool"].string == nil || selected["resume_tool"].string == nil { confirmSharedStop = true }
                            else { model.perform("service_stop", ["service_id": providerID, "force_shared": false]) }
                        }
                        }
                    }
                }
                if ["mlx2-image", "mlx2-video"].contains(providerID) {
                    Button("Start shared \(capability) service") {
                        Task {
                            do {
                                let result = try await model.request("media_direct_start", ["capability": capability])
                                providerID = result["service_id"].text
                                await model.refresh()
                            } catch { validation = error.localizedDescription }
                        }
                    }
                }
                Text(selected["endpoint"].text).font(.system(.caption, design: .monospaced)).textSelection(.enabled)
                if selected["configured"].bool && selected["drain_tool"].string == nil {
                    Text("This shared service needs manual Stop confirmation because its current API cannot atomically drain admission.").font(.caption).foregroundStyle(.secondary)
                }
                Text("Both applications submit to this endpoint and keep the returned job ID; opening this page does not start another model.").font(.caption).foregroundStyle(.secondary)
                if let error = selected["error"].string { Text(error).font(.caption).foregroundStyle(.orange) }
            }.padding(7)
        }
    }
    private var jobPanel: some View {
        GroupBox("Jobs & artifacts") {
            VStack(alignment: .leading, spacing: 12) {
                if media["jobs"].array.isEmpty { Text("No jobs submitted by this app.").foregroundStyle(.secondary) }
                ForEach(media["jobs"].array, id: \.rawID) { job in
                    VStack(alignment: .leading, spacing: 6) {
                        HStack {
                            Text(job["spec"]["capability"].text.capitalized).font(.headline)
                            Text(job["state"].text).foregroundStyle(.secondary)
                            Spacer()
                            if !["completed", "failed", "cancelled", "partial", "interrupted", "refused"].contains(job["state"].text) {
                                Button("Cancel") { model.perform("media_cancel", ["job_id": job["id"].text]) }
                            }
                        }
                        Text(job["progress"].text).font(.caption)
                        Text(job["result"]["remote_job_id"].text).font(.system(.caption, design: .monospaced)).textSelection(.enabled)
                        ForEach(artifactPaths(job), id: \.self) { path in
                            MediaArtifactPreview(path: path)
                        }
                        DisclosureGroup("Service receipt / output paths") {
                            Text(pretty(job["result"].raw)).font(.system(.caption, design: .monospaced)).textSelection(.enabled)
                        }
                    }
                    Divider()
                }
            }.padding(8)
        }
    }
    private var directSettings: some View {
        DisclosureGroup("Local shared media runtime") {
            VStack(alignment: .leading, spacing: 10) {
                Text("A shared loopback MCP service lets MLX2 and Studio use the same durable jobs. Starting this lightweight service does not load a model. LTX generation verifies the original source, converted artifact and pinned runtime before execution.").font(.caption).foregroundStyle(.secondary)
                Text("Canonical Studio service · images, video and voice").font(.headline)
                TextField("Canonical mlx2 media source checkout", text: $canonicalSource)
                HStack {
                    Button("Save canonical source") { model.perform("media_canonical_configure", ["settings": ["source": canonicalSource]]) }
                    Button("Prepare / Start canonical service") { model.perform("media_canonical_start") }
                    if media["canonical"]["preparing"].bool { Button("Cancel preparation") { model.perform("media_canonical_cancel") } }
                }
                Text("Freezes the selected source and prepares only the media control plane. Voice routes keep their existing isolated runtime prerequisites. Reference audio conditions a voice; it does not train one.").font(.caption).foregroundStyle(.secondary)
                ForEach(media["canonical"]["jobs"].array, id: \.rawID) { job in Text("\(job["state"].text) · \(job["progress"].text)").font(.caption) }
                Divider()
                TextField("Python executable", text: $directPython)
                TextField("Original LTX-2.5 source snapshot", text: $videoSource)
                TextField("Receipt-bound LTX-2.5 MLX model", text: $videoModel)
                TextField("Pinned ltx-2-mlx runtime directory", text: $videoRuntime)
                ForEach(Array(media["direct"]["video_models"].array.enumerated()), id: \.offset) { _, item in
                    Text("\(item["label"].text) · \(item["available"].bool ? "Prerequisites present; unqualified" : "Prerequisites missing")").font(.caption)
                    ForEach(Array(item["unavailable_reasons"].array.enumerated()), id: \.offset) { _, reason in Text(reason.text).font(.caption).foregroundStyle(.orange) }
                }
                HStack {
                    Button("Reload settings") { loadDirectSettings() }
                    Spacer()
                    Button("Save media runtime") { model.perform("media_direct_configure", ["settings": ["python": directPython, "video_source": videoSource, "video_model": videoModel, "video_runtime": videoRuntime]]) }
                }
            }.padding(.top, 8)
        }
    }
    private func loadDirectSettings() {
        canonicalSource = media["canonical"]["settings"]["source"].text
        let settings = media["direct"]["settings"]
        directPython = settings["python"].string ?? ""
        videoSource = settings["video_source"].string ?? ""
        videoModel = settings["video_model"].string ?? ""
        videoRuntime = settings["video_runtime"].string ?? ""
    }
    private func artifactPaths(_ job: JSON) -> [String] {
        let receipt = job["result"]["receipt"]
        var values = receipt["results"].array.compactMap { $0["artifact_path"].string }
        let result = receipt["result"]
        for key in ["path", "artifact_path", "output_path", "audio_path", "video_path"] {
            if let path = result[key].string { values.append(path) }
        }
        values += result["artifacts"].array.compactMap { $0["path"].string }
        values += receipt["artifacts"].array.compactMap { $0["path"].string }
        return Array(Set(values.filter { $0.hasPrefix("/") })).sorted()
    }
    private func generate() {
        do {
            var params = try JSONSerialization.jsonObject(with: Data(options.utf8)) as? [String: Any]
            guard params != nil else { throw CocoaError(.propertyListReadCorrupt) }
            if capability == "music" {
                params!["caption"] = prompt
                if !lyrics.isEmpty { params!["lyrics"] = lyrics }
                guard let seconds = Double(duration), seconds.isFinite, seconds > 0 else { validation = "Enter a positive duration."; return }
                params!["duration_seconds"] = seconds
            } else {
                params!["prompt"] = prompt
                if capability == "image" {
                    guard let w = Int(width), let h = Int(height) else { validation = "Enter numeric image dimensions."; return }
                    params!["width"] = w; params!["height"] = h
                    params!["operation"] = imageOperation
                    if imageOperation == "edit" { params!["reference_images"] = references.split(separator: "\n").map(String.init) }
                    if providerID == "mlx2-image" { params!["backend"] = backendID }
                }
            }
            if capability == "voice" && !voiceReference.isEmpty { params!["speaker_reference"] = voiceReference }
            if providerID != "mlx2-image" && backendID != "selected" && !remoteModels.isEmpty {
                params![selected["model_parameter"].string ?? "model"] = backendID
            }
            validation = nil
            model.perform("media_generate", ["provider_id": providerID, "parameters": params!])
        } catch { validation = "Additional parameters must be a JSON object." }
    }
    private func pretty(_ value: Any?) -> String {
        guard let value, JSONSerialization.isValidJSONObject(value), let data = try? JSONSerialization.data(withJSONObject: value, options: [.prettyPrinted, .sortedKeys]), let text = String(data: data, encoding: .utf8) else { return "{}" }
        return text
    }
    private func loadProfiles() {
        serviceJSON = pretty(model.snapshot["preferences"]["media_services"].array.map { $0.raw })
    }
    private func saveProfiles() {
        guard let data = serviceJSON.data(using: .utf8), let rows = try? JSONSerialization.jsonObject(with: data) as? [[String: Any]] else { validation = "Service profiles must be a JSON array."; return }
        validation = nil
        model.perform("preferences", ["changes": ["media_services": rows]])
    }
    private func addVideoExample() {
        var rows = (try? JSONSerialization.jsonObject(with: Data(serviceJSON.utf8))) as? [[String: Any]] ?? []
        rows.append(["id": "studio-video", "label": "Studio video", "capability": "video", "endpoint": "http://127.0.0.1:8778/mcp", "generate_tool": "generate_video", "status_tool": "get_video_job", "cancel_tool": "cancel_video_job", "models_tool": "video_models", "model_parameter": "model", "list_tool": "list_video_jobs", "launch_agents": [], "state_file": "", "header_ref": ""])
        serviceJSON = pretty(rows)
    }
}

// JSON deliberately has no Identifiable requirement; stable inventory/job ids
// remain the identity of list rows across periodic snapshots.
private extension JSON {
    var rawID: String { self["id"].text }
}

private struct MediaArtifactPreview: View {
    let path: String
    @State private var player: AVPlayer?
    private var ext: String { URL(fileURLWithPath: path).pathExtension.lowercased() }
    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            if ["png", "jpg", "jpeg", "webp"].contains(ext),
               let attributes = try? FileManager.default.attributesOfItem(atPath: path),
               let size = attributes[.size] as? NSNumber, size.int64Value <= 67_108_864,
               let image = NSImage(contentsOfFile: path) {
                Image(nsImage: image).resizable().scaledToFit().frame(maxHeight: 280)
            } else if ["mp4", "mov", "webm", "wav", "mp3", "flac", "m4a"].contains(ext) {
                VideoPlayer(player: player).frame(height: ["wav", "mp3", "flac", "m4a"].contains(ext) ? 64 : 240)
                    .task(id: path) { player = AVPlayer(url: URL(fileURLWithPath: path)) }
                    .onDisappear { player?.pause() }
            }
            HStack {
                Text(path).font(.caption.monospaced()).textSelection(.enabled)
                Spacer()
                Button("Reveal") { NSWorkspace.shared.activateFileViewerSelecting([URL(fileURLWithPath: path)]) }
            }
        }
    }
}
