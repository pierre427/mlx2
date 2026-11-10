import SwiftUI

/// Original branch (left) and its overlay replay branch (right).
struct ChatComparePair: Equatable {
    let left: String
    let right: String
    let replayOf: String
    let checkpoint: String
}

/// "Flag for correction": saves the history up to an assistant reply as a must-fix case.
struct CorrectionFlagSheet: View {
    @EnvironmentObject var model: DesktopModel
    @Environment(\.dismiss) private var dismiss
    @ObservedObject var chat: ChatModel
    let message: ChatMessage
    let identity: String
    let server: JSON?
    @State private var reason = "refused"
    @State private var note = ""
    @State private var saving = false

    private var modelPath: String { server?["endpoint"]["model_path"].string ?? "" }
    private var checkpoint: String {
        model.rows.first { $0["path"].string == modelPath && !modelPath.isEmpty }?["checkpoint"].string
            ?? server?["endpoint"]["checkpoint"].string ?? ""
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("Flag for correction").font(.title2.weight(.semibold))
            Text("Saves this conversation up to the reply as a must-fix test case in Ablation. One example cannot estimate a refusal direction, so it is used to check an edit, not to train one.")
                .font(.callout).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
            Picker("What went wrong", selection: $reason) {
                Text("Refused").tag("refused")
                Text("Redirected to a safer task").tag("redirected")
                Text("Lectured / moralised").tag("lectured")
                Text("Other").tag("other")
            }.pickerStyle(.radioGroup)
            VStack(alignment: .leading, spacing: 4) {
                Text("What I wanted (optional)").font(.caption).foregroundStyle(.secondary)
                TextEditor(text: $note).font(.body).frame(minHeight: 70)
                    .overlay(RoundedRectangle(cornerRadius: 6).stroke(.quaternary))
            }
            Text(checkpoint.isEmpty ? "Model checkpoint unknown: the case can be reviewed, but overlays bind to a fingerprint." : "Model: " + checkpoint)
                .font(.caption.monospaced()).foregroundStyle(.secondary).lineLimit(1).truncationMode(.middle)
            HStack {
                Spacer()
                Button("Cancel") { dismiss() }
                Button(saving ? "Saving…" : "Flag & open Ablation") { Task { await save() } }
                    .buttonStyle(.borderedProminent).disabled(saving)
            }
        }.padding(22).frame(width: 480)
    }

    @MainActor private func save() async {
        guard let history = chat.correctionHistory(for: message, in: identity) else {
            model.error = "This reply cannot be flagged (it must answer a user message)."; return
        }
        saving = true
        defer { saving = false }
        do {
            let result = try await model.request("ablation_case_create", [
                "session_id": chat.rootSession(of: identity), "branch_id": identity,
                "flagged_message_id": message.id.uuidString, "history": history, "flagged_text": message.text,
                "model_path": modelPath, "checkpoint": checkpoint, "reason": reason, "note": note])
            model.ablationFocusCase = result["id"].string
            dismiss()
            model.section = "Ablation"
        } catch { model.error = error.localizedDescription }
    }
}

/// Split view: the original branch and the overlay replay, from the same flagged turn.
struct CorrectionCompareView: View {
    @EnvironmentObject var model: DesktopModel
    @ObservedObject var chat: ChatModel
    let pair: ChatComparePair
    let send: (String, String) -> Bool
    @State private var text = ""
    @State private var sendToBoth = false
    @State private var sending = false
    @State private var sharedExpanded = true
    @State private var capability = JSON()

    private var left: [ChatMessage] { chat.conversations[pair.left] ?? [] }
    private var right: [ChatMessage] { chat.conversations[pair.right] ?? [] }
    private var shared: Int {
        var count = 0
        while count < left.count && count < right.count && left[count].id == right[count].id { count += 1 }
        return count
    }
    private var rightServer: JSON? { model.servers.first { ChatModel.identity($0) == chat.serverIdentity(for: pair.right) } }
    private var overlayBlocked: String? {
        guard chat.overlayActive(for: pair.right) else { return nil }
        if capability["supported"].bool { return nil }
        return capability["reason"].string ?? "Checking whether the server supports overlays…"
    }
    private var fingerprint: String {
        let value = pair.checkpoint.split(separator: "@").last.map(String.init) ?? ""
        return value.isEmpty ? "fingerprint unknown" : String(value.prefix(12)) + "…"
    }

    var body: some View {
        VStack(spacing: 0) {
            HStack {
                Label("Original vs overlay", systemImage: "rectangle.split.2x1").font(.headline)
                Spacer()
                Button("Close comparison") { chat.comparePair = nil; chat.selected = pair.right }
            }.padding(12)
            Divider()
            ScrollView {
                VStack(alignment: .leading, spacing: 10) {
                    if shared > 0 {
                        DisclosureGroup(isExpanded: $sharedExpanded) {
                            VStack(alignment: .leading, spacing: 8) {
                                ForEach(Array(left.prefix(shared))) { CompareBubble(message: $0) }
                            }.padding(.top, 6)
                        } label: { Text("Shared history · \(shared) message\(shared == 1 ? "" : "s")").font(.caption.weight(.semibold)) }
                    }
                    HStack(spacing: 8) {
                        Rectangle().frame(height: 1)
                        Label("Diverges here: flagged turn", systemImage: "arrow.triangle.branch").font(.caption.weight(.semibold))
                        Rectangle().frame(height: 1)
                    }.foregroundStyle(.orange)
                    HStack(alignment: .top, spacing: 12) {
                        panel(pair.left, Array(left.dropFirst(shared)), overlay: false)
                        Divider()
                        panel(pair.right, Array(right.dropFirst(shared)), overlay: true)
                    }
                }.padding(16)
            }
            Divider()
            composer.padding(12)
        }
        .task(id: rightServer?["endpoint"]["url"].string ?? "") { await probe() }
    }

    private func panel(_ identity: String, _ messages: [ChatMessage], overlay: Bool) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            VStack(alignment: .leading, spacing: 2) {
                Text(chat.historyTitle(identity)).font(.subheadline.weight(.semibold)).lineLimit(1)
                Text(chat.overlayActive(for: identity) ? "Overlay: " + chat.overlayLabel(for: identity) + " · unqualified"
                     : chat.overlayID(for: identity) != nil ? "Overlay off" : "No overlay")
                    .font(.caption).foregroundStyle(chat.overlayActive(for: identity) ? Color.accentColor : .secondary)
                Text(fingerprint).font(.caption2.monospaced()).foregroundStyle(.tertiary)
            }
            if overlay, let overlayBlocked {
                Label(overlayBlocked, systemImage: "exclamationmark.triangle").font(.caption).foregroundStyle(.orange)
            }
            if messages.isEmpty { Text("No messages after the flagged turn yet.").font(.caption).foregroundStyle(.secondary) }
            ForEach(messages) { CompareBubble(message: $0) }
        }.frame(maxWidth: .infinity, alignment: .topLeading)
    }

    private var composer: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(alignment: .bottom) {
                TextField("Continue on the overlay branch…", text: $text, axis: .vertical).lineLimit(1...6)
                    .textFieldStyle(.roundedBorder)
                Toggle("Send to both", isOn: $sendToBoth).toggleStyle(.checkbox)
                    .help("Also send this message on the original branch (each branch generates on its own, one after the other)")
                Button("Send") { submit() }.buttonStyle(.borderedProminent)
                    .disabled(text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty || sending || chat.generating
                              || (overlayBlocked != nil && !sendToBoth))
            }
            Text(sendToBoth ? "Sends to the overlay branch, then the original branch." : "Sends only to the overlay branch (right).")
                .font(.caption).foregroundStyle(.secondary)
        }
    }

    private func submit() {
        let message = text
        text = ""
        sending = true
        var targets: [String] = []
        if overlayBlocked == nil { targets.append(pair.right) }
        if sendToBoth { targets.append(pair.left) }
        Task { @MainActor in
            defer { sending = false }
            for (index, identity) in targets.enumerated() {
                if index > 0 { while chat.generating { try? await Task.sleep(for: .milliseconds(200)) } }
                guard send(message, identity) else {
                    model.error = "Could not send on " + chat.historyTitle(identity) + "."
                    return
                }
            }
        }
    }

    @MainActor private func probe() async {
        guard let url = rightServer?["endpoint"]["url"].string else {
            capability = JSON(["supported": false, "reason": "The overlay branch's model is not running"]); return
        }
        do { capability = try await model.request("ablation_overlay_capability", ["url": url]) }
        catch { capability = JSON(["supported": false, "reason": error.localizedDescription]) }
    }
}

private struct CompareBubble: View {
    let message: ChatMessage
    var body: some View {
        let content = message.presentation
        VStack(alignment: .leading, spacing: 4) {
            Text(message.role == "user" ? "You" : "Assistant").font(.caption2.weight(.semibold)).foregroundStyle(.secondary)
            if message.role == "assistant", !content.answer.isEmpty { ChatMarkdownView(text: content.answer) }
            else { Text(content.answer.isEmpty ? "…" : content.answer).textSelection(.enabled).fixedSize(horizontal: false, vertical: true) }
        }
        .padding(10)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(message.isUserBubble ? Color.accentColor.opacity(0.12) : Color.primary.opacity(0.05), in: RoundedRectangle(cornerRadius: 12))
    }
}
