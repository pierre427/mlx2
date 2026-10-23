import SwiftUI

/// Shows observed research work for this conversation, with retained history.
struct ChatResearchPlanView: View {
    @EnvironmentObject private var model: DesktopModel
    let parentID: String
    @State private var expanded = true
    @State private var showArchive = false

    private var plans: [JSON] {
        model.snapshot["research_plans"]["plans"].array.filter { $0["parent_id"].text == parentID }
    }
    private var active: [JSON] { plans.filter { $0["status"].text == "active" } }

    var body: some View {
        VStack(alignment: .leading, spacing: 5) {
            ForEach(Array(active.enumerated()), id: \.offset) { _, plan in
                DisclosureGroup(isExpanded: $expanded) {
                    ResearchPlanDetails(plan: plan, includeEvents: false).padding(.top, 5)
                } label: {
                    HStack(spacing: 6) {
                        ProgressView().controlSize(.mini)
                        Text("Research plan").font(.caption.weight(.semibold))
                        Text(plan["goal"].text).font(.caption).foregroundStyle(.secondary).lineLimit(1)
                        Spacer(minLength: 0)
                    }.contentShape(Rectangle())
                }
                .padding(.horizontal, 9).padding(.vertical, 6)
                .background(.quaternary.opacity(0.5), in: RoundedRectangle(cornerRadius: 7))
            }
            if !plans.isEmpty {
                Button { showArchive = true } label: {
                    Label("Plans (\(plans.count))", systemImage: "list.bullet.clipboard")
                }.buttonStyle(.plain).font(.caption).foregroundStyle(.secondary)
            }
        }
        .sheet(isPresented: $showArchive) {
            VStack(alignment: .leading, spacing: 12) {
                HStack {
                    Text("Research Plans").font(.headline)
                    Spacer()
                    Button("Done") { showArchive = false }.keyboardShortcut(.cancelAction)
                }
                Text("Saved progress for this chat. Tool success records activity; it does not verify source claims or conclusions.")
                    .font(.caption).foregroundStyle(.secondary)
                ScrollView {
                    LazyVStack(alignment: .leading, spacing: 14) {
                        ForEach(Array(plans.enumerated()), id: \.offset) { _, plan in
                            VStack(alignment: .leading, spacing: 6) {
                                HStack {
                                    Text(plan["goal"].text.isEmpty ? "Research" : plan["goal"].text).font(.subheadline.weight(.semibold))
                                    Spacer()
                                    Text(plan["status"].text.capitalized).font(.caption).foregroundStyle(.secondary)
                                }
                                if let created = plan["created"].double {
                                    Text(Date(timeIntervalSince1970: created), format: .dateTime.month().day().hour().minute())
                                        .font(.caption2).foregroundStyle(.secondary)
                                }
                                ResearchPlanDetails(plan: plan, includeEvents: true)
                            }
                            Divider()
                        }
                    }
                }
            }.padding(18).frame(minWidth: 520, idealWidth: 640, minHeight: 360, idealHeight: 520)
        }
        .onChange(of: parentID) { _, _ in showArchive = false; expanded = true }
    }
}

private struct ResearchPlanDetails: View {
    let plan: JSON
    let includeEvents: Bool
    private func symbol(_ status: String) -> String {
        switch status {
        case "completed": return "checkmark.circle.fill"
        case "running": return "arrow.triangle.2.circlepath"
        case "failed", "interrupted": return "exclamationmark.circle"
        case "not_observed": return "minus.circle"
        default: return "circle"
        }
    }
    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            ForEach(Array(plan["steps"].array.enumerated()), id: \.offset) { _, step in
                HStack(spacing: 6) {
                    Image(systemName: symbol(step["status"].text)).foregroundStyle(step["status"].text == "completed" ? Color.green : Color.secondary)
                    Text(step["label"].text)
                    Spacer(minLength: 8)
                    Text(step["status"].text.replacingOccurrences(of: "_", with: " ")).foregroundStyle(.secondary)
                }.font(.caption)
            }
            if let waiting = plan["waiting"].string, !waiting.isEmpty {
                Text(waiting).font(.caption2).foregroundStyle(.secondary)
            }
            if includeEvents {
                ForEach(Array(plan["events"].array.enumerated()), id: \.offset) { _, event in
                    DisclosureGroup {
                        VStack(alignment: .leading, spacing: 3) {
                            if !event["evidence"]["request"].text.isEmpty { Text(event["evidence"]["request"].text) }
                            Text(event["evidence"]["observation"].text)
                            ForEach(Array(event["evidence"]["references"].array.enumerated()), id: \.offset) { _, source in
                                if let url = URL(string: source["url"].text), ["https", "http"].contains(url.scheme?.lowercased() ?? "") {
                                    Link(source["title"].text.isEmpty ? url.absoluteString : source["title"].text, destination: url)
                                } else if !source["title"].text.isEmpty { Text(source["title"].text) }
                            }
                        }.font(.caption).textSelection(.enabled)
                    } label: {
                        Text(event["tool"].text.replacingOccurrences(of: "_", with: " ") + " · " + event["status"].text).font(.caption)
                    }
                }
                Text(plan["notice"].text).font(.caption2).foregroundStyle(.secondary)
            }
        }
    }
}
