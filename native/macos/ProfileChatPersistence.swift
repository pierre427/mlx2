import Combine
import Foundation

@MainActor final class ProfileChatPersistence {
    private weak var model: DesktopModel?
    private var subscriptions: Set<AnyCancellable> = []
    private var debounce: Task<Void, Never>?
    private var profileID = ""
    private var restoring = false
    private var paused = false
    private var selectionEpoch = UUID()
    private var requestedProfileID = ""
    private var openedInitialChat = false

    func start(_ model: DesktopModel) {
        guard self.model == nil else { return }
        self.model = model
        model.chat.$conversations.dropFirst().sink { [weak self] _ in self?.schedule() }.store(in: &subscriptions)
        model.chat.$branches.dropFirst().sink { [weak self] _ in self?.schedule() }.store(in: &subscriptions)
        model.chat.$selected.dropFirst().sink { [weak self] _ in self?.schedule() }.store(in: &subscriptions)
        model.chat.$draft.dropFirst().sink { [weak self] _ in self?.schedule() }.store(in: &subscriptions)
        model.chat.$generating.dropFirst().sink { [weak self] generating in if !generating { self?.schedule() } }.store(in: &subscriptions)
    }

    private func schedule() {
        guard !restoring, !paused else { return }
        debounce?.cancel()
        debounce = Task { @MainActor [weak self] in
            try? await Task.sleep(for: .seconds(1))
            guard !Task.isCancelled else { return }
            self?.debounce = nil
            _ = await self?.flush()
        }
    }

    func pause() { paused = true; debounce?.cancel(); debounce = nil }
    func resume() { paused = false; schedule() }

    @discardableResult func flush() async -> Bool {
        debounce?.cancel(); debounce = nil
        guard let model, !profileID.isEmpty, !restoring else { return true }
        let identity = profileID
        let document = model.chat.profileArchive(compact: true)
        do {
            let size = try JSONSerialization.data(withJSONObject: document).count
            guard size <= 8 * 1024 * 1024 else {
                model.chat.autosaveError = "Chat autosave stopped: archive exceeds 8 MiB. Export or shorten chats before switching profiles or quitting."
                return false
            }
            _ = try await model.request("user_profile_chat_set", ["profile_id": identity, "document": document])
            model.chat.autosaveError = nil
            return true
        } catch {
            model.chat.autosaveError = "Chat autosave: " + error.localizedDescription
            return false
        }
    }

    @discardableResult func select(_ identity: String) async -> Bool {
        guard let model, !identity.isEmpty else { return false }
        guard (restoring ? identity != requestedProfileID : identity != profileID) else { return !restoring }
        requestedProfileID = identity
        model.chat.permissionProfile = identity
        debounce?.cancel()
        let epoch = UUID()
        selectionEpoch = epoch
        restoring = true
        defer { if selectionEpoch == epoch { restoring = false } }
        model.chat.restoreProfileArchive(JSON([:])); model.choices.cancel()
        do {
            let value = try await model.request("user_profile_chat_get", ["profile_id": identity])
            guard selectionEpoch == epoch else { return false }
            profileID = identity
            model.chat.restoreProfileArchive(value["document"])
            if !openedInitialChat {
                openedInitialChat = true
                model.chat.startFreshChat(servers: model.servers)
            }
            if model.chat.selected.isEmpty, let first = model.servers.first { model.chat.selected = ChatModel.identity(first) }
            return true
        } catch {
            if selectionEpoch == epoch { model.error = "Chat restore: " + error.localizedDescription }
            return false
        }
    }

    func close() async { debounce?.cancel(); await flush(); subscriptions.removeAll() }
}
