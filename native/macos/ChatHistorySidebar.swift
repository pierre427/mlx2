import SwiftUI

struct ChatHistorySidebar: View {
    @ObservedObject var chat: ChatModel
    let newChat: () -> Void
    let collapse: () -> Void
    @State private var search = ""
    @State private var category = "active"
    @State private var renameID: String?
    @State private var title = ""
    private var identities: [String] {
        chat.conversations.keys.filter { id in
            chat.historyState(id) == category && (search.isEmpty || chat.historyTitle(id).localizedCaseInsensitiveContains(search))
        }.sorted { left, right in
            let a = chat.conversations[left]?.last?.createdAt ?? .distantPast
            let b = chat.conversations[right]?.last?.createdAt ?? .distantPast
            return a == b ? left < right : a > b
        }
    }
    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Text("Chats").font(.headline)
                Spacer()
                Button(action: newChat) { Image(systemName: "square.and.pencil") }.disabled(chat.generating)
                    .help("New chat").accessibilityLabel("New chat in history")
                Button(action: collapse) { Image(systemName: "sidebar.left") }.help("Collapse chat history").accessibilityLabel("Collapse chat history")
            }
            TextField("Search chats", text: $search).textFieldStyle(.roundedBorder)
            Picker("Chat history", selection: $category) {
                Text("Recent").tag("active"); Text("Archived").tag("archived"); Text("Trash").tag("deleted")
            }.labelsHidden()
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 4) {
                    if identities.isEmpty { Text("No chats here").font(.caption).foregroundStyle(.secondary).padding(.vertical, 12) }
                    ForEach(identities, id: \.self) { identity in
                        Button { chat.selected = identity } label: {
                            VStack(alignment: .leading, spacing: 4) {
                                Text(chat.historyTitle(identity)).font(.callout).lineLimit(2)
                                if let date = chat.conversations[identity]?.last?.createdAt {
                                    Text(date, format: .dateTime.month().day().hour().minute()).font(.caption2).foregroundStyle(.secondary)
                                }
                            }.frame(maxWidth: .infinity, alignment: .leading).padding(9)
                                .background(chat.selected == identity ? Color.accentColor.opacity(0.16) : .clear, in: RoundedRectangle(cornerRadius: 8))
                                .contentShape(Rectangle())
                        }.buttonStyle(.plain).disabled(chat.generating)
                        .contextMenu {
                            Button("Rename…") { renameID = identity; title = chat.historyTitle(identity) }
                            if category != "active" { Button("Restore to Recent") { chat.setHistoryState("active", for: identity) } }
                            if category == "active" { Button("Archive") { chat.setHistoryState("archived", for: identity) } }
                            if category != "deleted" { Button("Delete Chat", role: .destructive) { chat.setHistoryState("deleted", for: identity) } }
                        }
                    }
                }
            }
            if category == "deleted" { Text("Deleted chats remain in Trash and can be restored.").font(.caption2).foregroundStyle(.secondary) }
        }.disabled(chat.mutationsPaused).padding(12).frame(width: 210)
            .background(.quaternary.opacity(0.2))
            .alert("Rename Chat", isPresented: Binding(get: { renameID != nil }, set: { if !$0 { renameID = nil } })) {
                TextField("Title", text: $title)
                Button("Save") { if let id = renameID { chat.renameHistory(id, title: title) }; renameID = nil }
                Button("Cancel", role: .cancel) { renameID = nil }
            }
    }
}
