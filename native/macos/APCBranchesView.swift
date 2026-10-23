import SwiftUI

struct APCBranchesView: View {
    @EnvironmentObject var model: DesktopModel
    @State private var profile = ""
    @State private var state = JSON()
    @State private var error: String?
    @State private var loading = false
    private var workers: [JSON] { model.workers.filter { ["Ready", "Busy", "Suspended"].contains($0["state"].text) } }
    var body: some View {
        GroupBox("APCv2 branches") {
            VStack(alignment: .leading, spacing: 10) {
                HStack {
                    Picker("Managed model", selection: $profile) {
                        Text("Choose a model").tag("")
                        ForEach(Array(workers.enumerated()), id: \.offset) { _, worker in
                            Text(worker["name"].text + " · " + worker["state"].text).tag(worker["profile_id"].string ?? "default")
                        }
                    }
                    Button("Refresh") { Task { await refresh() } }.disabled(profile.isEmpty || loading)
                    if loading { ProgressView().controlSize(.small) }
                }
                Text("Conversation branches can reuse shared cached prefixes. Parent links below are inferred from retained token prefixes; they are not proof that a particular chat used a cache entry.")
                    .font(.caption).foregroundStyle(.secondary)
                if state["clock_frozen"].bool { Label("Suspended · expiry clocks are frozen", systemImage: "pause.circle").foregroundStyle(.orange) }
                if let error { Text(error).foregroundStyle(.orange).font(.caption) }
                if state["entries"].array.isEmpty { Text("No retained branches reported.").foregroundStyle(.secondary) }
                ForEach(Array(state["entries"].array.enumerated()), id: \.offset) { _, entry in
                    VStack(alignment: .leading, spacing: 4) {
                        HStack {
                            Text(entry["id"].text).font(.caption.monospaced()).textSelection(.enabled)
                            Spacer()
                            Text(entry["tier"].text)
                            Text(number(entry["token_count"].double, suffix: " tokens", digits: 0))
                        }
                        if let parent = entry["parent_id"].string {
                            Text("↳ \(parent) · shared prefix \(number(entry["shared_prefix_tokens"].double, digits: 0)) tokens").font(.caption)
                        } else if let peer = entry["shared_prefix_entry_id"].string {
                            Text("↔ \(peer) · diverges after \(number(entry["shared_prefix_tokens"].double, digits: 0)) shared tokens").font(.caption)
                        }
                        HStack {
                            Text("Pins: \(number(entry["pin_count"].double, digits: 0))")
                            if let ttl = entry["remaining_ttl_seconds"].double { Text("Remaining TTL: \(number(ttl, suffix: " s"))") }
                        }.font(.caption).foregroundStyle(.secondary)
                    }.padding(8).background(.quaternary, in: RoundedRectangle(cornerRadius: 6))
                }
                if state["truncated"].bool { Text("Showing the first 200 retained entries.").font(.caption).foregroundStyle(.secondary) }
            }.padding(8)
        }.onChange(of: profile) { _, _ in state = JSON(); Task { await refresh() } }
    }
    @MainActor private func refresh() async {
        guard !profile.isEmpty, !loading else { return }
        let selected = profile
        loading = true
        defer { loading = false }
        do {
            let result = try await model.request("engine_cache_branches", ["profile_id": selected])
            if profile == selected { state = result; error = nil }
        } catch { if profile == selected { state = JSON(); self.error = error.localizedDescription } }
    }
}
