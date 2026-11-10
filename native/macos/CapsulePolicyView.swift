import AppKit
import SwiftUI

struct CapsulePolicyView: View {
    @EnvironmentObject var model: DesktopModel
    @State private var autonomy = "off"
    @State private var situations = Set(["user_requested", "correction", "tool_failure", "task_completed"])
    @State private var allowPromote = false
    @State private var operations = 8
    @State private var capacity = 256
    @State private var noteID = ""
    @State private var revision = "1"
    @State private var digest = ""
    @State private var resultText = ""
    @State private var capturing = false
    @State private var removeDigest: String?
    private var state: JSON { model.snapshot["capsules"] }
    private let options = ["user_requested", "correction", "tool_failure", "task_completed", "useful_feedback", "contradiction"]
    var body: some View {
        GroupBox("Semantic capsules & memory autonomy") {
            VStack(alignment: .leading, spacing: 12) {
                Text("Exact, scoped snapshots use the existing content-addressed capsule store. Notes are captured as evaluation records; this does not apply neural or activation state to a running model.").font(.caption).foregroundStyle(.secondary)
                Picker("Agent autonomy", selection: $autonomy) {
                    Text("Off · approval for each operation").tag("off")
                    Text("Situational · selected conditions").tag("situational")
                    Text("Always · within scope and budget").tag("always")
                }
                if autonomy == "situational" {
                    LazyVGrid(columns: [GridItem(.flexible()), GridItem(.flexible())], alignment: .leading) {
                        ForEach(options, id: \.self) { situation in
                            Toggle(situation.replacingOccurrences(of: "_", with: " ").capitalized, isOn: Binding(get: { situations.contains(situation) }, set: { enabled in if enabled { situations.insert(situation) } else { situations.remove(situation) } }))
                        }
                    }
                }
                Toggle("Allow model-driven note promotion within this policy", isOn: $allowPromote)
                Text("A model's autonomous review remains labelled model-policy-unverified. It is recorded separately from operator confirmation. A declared situation is the model's explanation, not proof that it occurred.").font(.caption).foregroundStyle(.secondary)
                Stepper("At most \(operations) autonomous operations per admitted session", value: $operations, in: 1...64)
                Stepper("At most \(capacity) capsules per scope", value: $capacity, in: 1...4096, step: 16)
                HStack {
                    Button("Reload Policy") { load() }.help("Discard unsaved changes and show the saved policy.")
                    Spacer()
                    Button("Save Autonomy Policy") {
                        model.perform("capsule_policy", ["changes": ["autonomy": autonomy, "allowed_situations": situations.sorted(), "allow_promote": allowPromote, "max_operations_per_session": operations, "max_capsules_per_scope": capacity]])
                    }.disabled(model.busy)
                }
                if autonomy == "situational" && situations.isEmpty { Text("No situations selected: situational autonomy will not run any operation.").font(.caption).foregroundStyle(.orange) }
                DisclosureGroup("Capture an exact note revision") {
                    HStack {
                        TextField("Note ID", text: $noteID).accessibilityLabel("Note ID")
                        TextField("Revision", text: $revision).frame(width: 80).accessibilityLabel("Note revision")
                        Button(capturing ? "Capturing…" : "Capture Snapshot") {
                            guard let value = Int(revision), value > 0 else { model.error = "Enter an exact positive note revision."; return }
                            capturing = true
                            Task {
                                defer { capturing = false }
                                do {
                                    let response = try await model.request("capsule_capture", ["id": noteID, "revision": value, "situation": "user_requested"])
                                    digest = response["digest"].text
                                    resultText = "Captured evaluation capsule · activation not applied"
                                    await model.refresh()
                                } catch { model.error = error.localizedDescription }
                            }
                        }.disabled(capturing || noteID.trimmingCharacters(in: .whitespaces).isEmpty || !(Int(revision).map { $0 > 0 } ?? false))
                    }
                    if !digest.isEmpty {
                        Text(digest).font(.caption.monospaced()).textSelection(.enabled)
                        Text(resultText).font(.caption).foregroundStyle(.secondary)
                    }
                }
                DisclosureGroup("Stored capsule receipts") {
                    if state["capsules"].array.isEmpty { Text("No stored capsules.").font(.caption).foregroundStyle(.secondary) }
                    ForEach(Array(state["capsules"].array.enumerated()), id: \.offset) { _, item in
                        VStack(alignment: .leading, spacing: 5) {
                            Text(item["digest"].text).font(.caption.monospaced()).textSelection(.enabled)
                            Text("\(item["scope"].text) · note \(item["note_id"].text)@\(Int(item["note_revision"].double ?? 0)) · \(item["authority"].text)").font(.caption).foregroundStyle(.secondary)
                            HStack {
                                Button("Copy Digest") { NSPasteboard.general.clearContents(); NSPasteboard.general.setString(item["digest"].text, forType: .string) }
                                Button("Remove Snapshot…", role: .destructive) { removeDigest = item["digest"].text }.disabled(model.busy)
                                    .help("Removes this snapshot if nothing references it.")
                            }
                        }
                        Divider()
                    }
                }
                Text("Activation application and approximate operations remain unavailable in this desktop bridge. Existing engine routes need their own exact artifact, runtime and qualification receipts.").font(.caption).foregroundStyle(.secondary)
            }.padding(8)
        }.onAppear { load() }
        .confirmationDialog("Remove this capsule snapshot?", isPresented: Binding(get: { removeDigest != nil }, set: { if !$0 { removeDigest = nil } }), presenting: removeDigest) { digest in
            Button("Remove Snapshot", role: .destructive) { model.perform("capsule_delete", ["digest": digest, "situation": "user_requested"]); removeDigest = nil }
            Button("Cancel", role: .cancel) { removeDigest = nil }
        } message: { digest in Text("Snapshot \(digest.prefix(16))… is permanently removed. Removal is refused while another snapshot references it.") }
    }
    private func load() {
        let policy = state["policy"]
        autonomy = policy["autonomy"].string ?? "off"
        situations = Set(policy["allowed_situations"].array.map(\.text))
        allowPromote = policy["allow_promote"].bool
        operations = Int(policy["max_operations_per_session"].double ?? 8)
        capacity = Int(policy["max_capsules_per_scope"].double ?? 256)
    }
}
