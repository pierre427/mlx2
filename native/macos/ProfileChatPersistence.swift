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
    private var pauseBeforeRestore: Bool?
    private let requestOverride: ((String, [String: Any]) async throws -> JSON)?

    init(request: ((String, [String: Any]) async throws -> JSON)? = nil) { requestOverride = request }

    private func request(_ command: String, _ arguments: [String: Any]) async throws -> JSON {
        if let requestOverride { return try await requestOverride(command, arguments) }
        guard let model else { throw DesktopError.message("Chat storage is unavailable.") }
        return try await model.request(command, arguments)
    }

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
        guard requestedProfileID == profileID else { return false }
        let identity = profileID
        let epoch = selectionEpoch
        let document = model.chat.profileArchive(compact: true)
        do {
            let size = try JSONSerialization.data(withJSONObject: document).count
            guard size <= 8 * 1024 * 1024 else {
                model.chat.autosaveError = "Chat autosave stopped: archive exceeds 8 MiB. Export or shorten chats before switching profiles or quitting."
                return false
            }
            _ = try await request("user_profile_chat_set", ["profile_id": identity, "document": document])
            if selectionEpoch == epoch { model.chat.autosaveError = nil }
            return true
        } catch {
            if selectionEpoch == epoch { model.chat.autosaveError = "Chat autosave: " + error.localizedDescription }
            return false
        }
    }

    @discardableResult func select(_ identity: String) async -> Bool {
        guard let model, !identity.isEmpty else { return false }
        if !restoring && identity == profileID {
            requestedProfileID = identity
            if let pauseBeforeRestore { model.chat.mutationsPaused = pauseBeforeRestore; self.pauseBeforeRestore = nil }
            return true
        }
        guard !restoring || identity != requestedProfileID else { return false }
        requestedProfileID = identity
        debounce?.cancel()
        let epoch = UUID()
        selectionEpoch = epoch
        restoring = true
        if pauseBeforeRestore == nil { pauseBeforeRestore = model.chat.mutationsPaused }
        model.chat.mutationsPaused = true
        defer { if selectionEpoch == epoch { restoring = false } }
        do {
            let value = try await request("user_profile_chat_get", ["profile_id": identity])
            guard selectionEpoch == epoch else { return false }
            guard value["profile_id"].string == identity,
                  let document = value["document"].raw as? [String: Any],
                  document.isEmpty || value["document"]["schema_version"].double == 1 else {
                throw DesktopError.message("The saved chat document is invalid or belongs to another profile. Existing chats were preserved.")
            }
            profileID = identity
            model.chat.permissionProfile = identity
            model.choices.cancel()
            model.chat.restoreProfileArchive(value["document"])
            if !openedInitialChat {
                openedInitialChat = true
                model.chat.startFreshChat(servers: model.servers)
            }
            if model.chat.selected.isEmpty, let first = model.servers.first { model.chat.selected = ChatModel.identity(first) }
            model.chat.mutationsPaused = pauseBeforeRestore ?? false
            pauseBeforeRestore = nil
            model.chat.autosaveError = nil
            return true
        } catch {
            if selectionEpoch == epoch { model.error = "Chat restore: " + error.localizedDescription }
            return false
        }
    }

    func close() async { debounce?.cancel(); await flush(); subscriptions.removeAll() }
}
