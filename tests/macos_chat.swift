// Synthetic HTTP only. No checkpoint loaded, no inference or qualification.
import AppKit
import Foundation
import SwiftUI

@main enum ChatChecks {
    @MainActor static func wait(_ chat: ChatModel) async throws {
        for _ in 0..<500 {
            if !chat.generating { return }
            try await Task.sleep(for: .milliseconds(20))
        }
        fatalError("Chat did not finish")
    }

    @MainActor static func main() async throws {
        _ = NSApplication.shared
        // The harness is linked as LSUIElement; never promote it to a regular app.
        precondition(NSApp.activationPolicy() != .regular)
        let envelope = ChatMessage(role: "assistant", text: "<thinking>Inspect tools.</thinking>\n<answer>**Yes.**</answer>")
        precondition(envelope.presentation.answer == "**Yes.**" && envelope.presentation.reasoning == "Inspect tools.")
        precondition(envelope.clipboardText == "**Yes.**")
        precondition(ChatMessage(role: "assistant", text: "<think>Still working").presentation.answer.isEmpty)
        precondition(ChatMessage(role: "assistant", text: "<answer>Done</ans").presentation.answer == "Done")
        precondition(ChatMessage(role: "assistant", text: "```xml\n<thinking>literal</thinking>\n```").presentation.reasoning.isEmpty)
        precondition(ChatMessage(role: "user", text: "<thinking>literal</thinking>").presentation.answer == "<thinking>literal</thinking>")
        precondition(ChatModel.isReadyForChat(JSON(["reachable": true, "ready": true, "status_error": "timed out"])))
        precondition(!ChatModel.isReadyForChat(JSON(["reachable": false, "ready": true])))
        precondition(!ChatModel.isReadyForChat(JSON(["ready": false])))
        precondition(!ChatModel.isReadyForChat(JSON(["status_error": "timed out"])))
        var attachment = ChatScrollAttachment()
        attachment.observe(height: 1000, viewport: 400, distance: 0)
        precondition(attachment.followsBottom)
        attachment.observe(height: 1200, viewport: 400, distance: 200)
        precondition(attachment.followsBottom) // Growth preserves prior attachment.
        attachment.observe(height: 1800, viewport: 600, distance: 150, userScroll: false)
        attachment.observe(height: 1800, viewport: 600, distance: 150, userScroll: false)
        precondition(attachment.followsBottom) // Repeated layout passes are not a user scroll.
        attachment.observe(height: 1200, viewport: 400, distance: 250, userScroll: true)
        precondition(!attachment.followsBottom)
        attachment.observe(height: 1400, viewport: 400, distance: 450)
        precondition(!attachment.followsBottom)
        attachment.observe(height: 1400, viewport: 400, distance: 0, userScroll: true)
        precondition(attachment.followsBottom)
        let callback = ChatMessage(role: "user", text: "Internal callback instructions\n{\"job_id\":\"child\",\"status\":\"failed\",\"workers\":[{\"worker_id\":\"worker\",\"failure\":{\"cause\":\"Worker tool round limit reached\"}}]}", origin: "Subagent result")
        precondition(!callback.isUserBubble && callback.agentCallback?["job_id"].string == "child")
        precondition(callback.presentation.answer.contains("Worker tool round limit reached"))
        precondition(!callback.presentation.answer.contains("Internal callback instructions"))
        precondition(!callback.clipboardText.contains("job_id"))
        precondition(ChatMessage(role: "user", text: "My request").isUserBubble)
        let history = ChatModel()
        history.startFreshChat(servers: [])
        let historyID = history.selected
        history.renameHistory(historyID, title: "Model Serving Research")
        history.setHistoryState("archived", for: historyID)
        let historyRestored = ChatModel()
        historyRestored.restoreProfileArchive(JSON(history.profileArchive()))
        precondition(historyRestored.historyState(historyID) == "archived")
        precondition(historyRestored.historyTitle(historyID) == "Model Serving Research")
        historyRestored.setHistoryState("deleted", for: historyID)
        precondition(historyRestored.historyState(historyID) == "deleted")
        precondition(historyRestored.conversations[historyID] != nil) // Recoverable deletion.
        historyRestored.setHistoryState("active", for: historyID)
        precondition(historyRestored.historyState(historyID) == "active")
        let firstModel = JSON(["endpoint": ["url": "http://127.0.0.1:8991", "pid": 11, "model_path": "/models/first"], "model": "First"])
        let secondModel = JSON(["endpoint": ["url": "http://127.0.0.1:8992", "pid": 12, "model_path": "/models/second"], "model": "Second"])
        let lockedChat = ChatModel()
        lockedChat.startFreshChat(servers: [firstModel])
        precondition(lockedChat.bindEmptyConversation(to: secondModel))
        precondition(lockedChat.conversationModel(lockedChat.selected) == "second")
        lockedChat.conversations[lockedChat.selected] = [ChatMessage(role: "user", text: "Started")]
        precondition(!lockedChat.bindEmptyConversation(to: firstModel))
        precondition(lockedChat.conversationModel(lockedChat.selected) == "second")
        let fittingChat = ChatModel()
        fittingChat.startFreshChat(servers: [])
        fittingChat.conversations[fittingChat.selected] = (0..<30).map { index in
            ChatMessage(role: index % 2 == 0 ? "user" : "assistant", text: String(repeating: "Long transcript content ", count: 80))
        }
        let fittingHost = NSHostingView(rootView: BoundedWorkspace { ChatView(chat: fittingChat) }.environmentObject(DesktopModel()))
        fittingHost.frame = NSRect(x: 0, y: 0, width: 1100, height: 700)
        fittingHost.layoutSubtreeIfNeeded()
        try await Task.sleep(for: .milliseconds(50))
        precondition(fittingHost.fittingSize.height <= 700, "Chat content grew the window's minimum height")
        let fiveImages = (1...5).map { "![Image \($0)](https://example.com/penguin-\($0).jpg)" }.joined(separator: "\n\n")
        let fiveBlocks = ChatMarkdown.blocks(fiveImages)
        precondition(fiveBlocks.count == 5 && fiveBlocks.allSatisfy { $0.kind == "image" })
        precondition(fiveBlocks.map(\.marker) == (1...5).map { "https://example.com/penguin-\($0).jpg" })
        let imageMarkdown = ChatMarkdown.blocks("Here's a photo:\n\n![Photo](https://upload.wikimedia.org/example_%28one%29.jpg)\n\n*Credit: photographer*")
        precondition(imageMarkdown.map(\.kind) == ["text", "image", "text"])
        precondition(imageMarkdown[1].marker == "https://upload.wikimedia.org/example_%28one%29.jpg")
        let mixedMedia = ChatMarkdown.blocks("[Source](https://example.com/page) ![Caption](https://example.com/a(b).png) after")
        precondition(mixedMedia.map(\.kind) == ["text", "image", "text"])
        precondition(mixedMedia[0].text.contains("[Source]") && mixedMedia[2].text == " after")
        precondition(ChatMarkdown.blocks("[Song](https://example.com/track.mp3?download=1)").first?.kind == "audio")
        precondition(ChatMarkdown.blocks("https://example.com/clip.mp4").first?.kind == "video")
        precondition(ChatMarkdown.blocks("```\n![literal](https://example.com/a.png)\n```").first?.kind == "code")
        precondition(ChatMediaSource.url("https://user:secret@example.com/a.png") == nil)
        precondition(ChatMediaSource.url("javascript:alert(1)") == nil)
        precondition(ChatMediaSource.url("http://example.com/a.png") == nil)
        precondition(ChatMediaSource.url("http://127.0.0.1:8776/artifacts/picture.png") != nil)
        precondition(ChatMediaSource.url("/tmp/generated image.png")?.isFileURL == true)
        let markdown = ChatMarkdown.blocks("## Title\n- **bold**\n1. item\n> quote\n```swift\nlet x = 1\n```\n| A | B |\n| --- | --- |\n| 1 | 2 |")
        precondition(markdown.map(\.kind) == ["heading", "list", "list", "quote", "code", "table"])
        precondition(markdown[0].level == 2 && markdown[4].text == "let x = 1")
        precondition(markdown[5].rows.count == 2)
        precondition(String(ChatMarkdown.inline("**bold** and `code`").characters) == "bold and code")
        precondition(ChatMarkdown.blocks("```\nunfinished").first?.text == "unfinished")
        let permissionSuite = "mlx2-test-permissions-" + UUID().uuidString
        let permissionDefaults = UserDefaults(suiteName: permissionSuite)!
        defer { permissionDefaults.removePersistentDomain(forName: permissionSuite) }
        let permissionChat = ChatModel()
        permissionChat.permissionProfile = "profile-a"
        precondition(!permissionChat.alwaysAllowsTool("mlx2__research_search", defaults: permissionDefaults))
        permissionChat.rememberTool("mlx2__research_search", defaults: permissionDefaults)
        let relaunchedChat = ChatModel()
        relaunchedChat.permissionProfile = "profile-a"
        precondition(relaunchedChat.alwaysAllowsTool("mlx2__research_search", defaults: permissionDefaults))
        precondition(!relaunchedChat.alwaysAllowsTool("mlx2__preferences", defaults: permissionDefaults))
        relaunchedChat.permissionProfile = "profile-b"
        precondition(!relaunchedChat.alwaysAllowsTool("mlx2__research_search", defaults: permissionDefaults))
        relaunchedChat.permissionProfile = "profile-a"
        relaunchedChat.forgetToolPermissions(defaults: permissionDefaults)
        precondition(!permissionChat.alwaysAllowsTool("mlx2__research_search", defaults: permissionDefaults))
        let fresh = ChatModel()
        let live = JSON(["endpoint": ["url": "http://127.0.0.1:8285", "pid": 99, "model_path": "/synthetic/model"]])
        let oldID = ChatModel.identity(live)
        fresh.selected = oldID
        fresh.conversations[oldID] = [ChatMessage(role: "user", text: "Retain this old chat")]
        fresh.draft = "old unsent draft"
        fresh.startFreshChat(servers: [])
        let freshID = fresh.selected
        precondition(freshID != oldID && fresh.draft.isEmpty)
        precondition(fresh.conversations[freshID]?.isEmpty == true)
        let preparing = JSON(["state": "Preparing", "name": "Synthetic", "path": "/synthetic/model", "preparation_message": "Checking runner"])
        precondition(fresh.pendingWorker([preparing])?["state"].text == "Preparing")
        fresh.selectFirstIfNeeded([live])
        precondition(fresh.selected == freshID && fresh.serverIdentity(for: freshID) == oldID)
        precondition(fresh.conversations[oldID]?.first?.text == "Retain this old chat")
        precondition(fresh.pendingWorker([JSON(["state": "Loading", "url": "http://127.0.0.1:8285", "path": "/synthetic/model"])]) != nil)
        precondition(fresh.pendingWorker([JSON(["state": "Ready", "url": "http://127.0.0.1:8285", "path": "/synthetic/model"])]) == nil)
        precondition(fresh.pendingWorker([JSON(["state": "Loading", "url": "http://127.0.0.1:9000", "path": "/different/model"])]) == nil)
        precondition(fresh.pendingWorker([JSON(["state": "Error", "url": "http://127.0.0.1:8285", "error": "Port conflict"])])?["error"].text == "Port conflict")
        let desktop = DesktopModel()
        desktop.updateWorkerError("port occupied")
        precondition(desktop.error == "port occupied")
        desktop.error = nil // Dismiss the old launch failure.
        desktop.updateWorkerError("port occupied")
        precondition(desktop.error == nil)
        desktop.updateWorkerError(nil)
        desktop.updateWorkerError("port occupied")
        precondition(desktop.error == "port occupied")
        desktop.updateWorkerError(nil)
        precondition(desktop.error == nil)
        desktop.error = "independent command failure"
        desktop.updateWorkerError(nil)
        precondition(desktop.error == "independent command failure")
        precondition(!DesktopLaunch.background(arguments: ["MLX2"], loginItem: false))
        precondition(DesktopLaunch.background(arguments: ["MLX2"], loginItem: true))
        precondition(DesktopLaunch.background(arguments: ["MLX2", "--background"], loginItem: false))
        precondition(!DesktopLaunch.background(arguments: ["MLX2", "--details"], loginItem: true))
        let window = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 320, height: 160),
            styleMask: [.titled, .closable, .miniaturizable], backing: .buffered, defer: false)
        window.isReleasedWhenClosed = false
        var policies: [NSApplication.ActivationPolicy] = []
        var focusRequests = 0
        let presence = DockWindowObserver(setPolicy: { policies.append($0) }, focusWindow: { _ in focusRequests += 1 })
        window.contentView!.addSubview(presence)
        precondition(policies == [.regular])
        // Exercise lifecycle transitions without showing, minimizing or focusing a window.
        for name in [NSWindow.didBecomeKeyNotification, NSWindow.didDeminiaturizeNotification, NSWindow.willCloseNotification] {
            NotificationCenter.default.post(name: name, object: window)
        }
        precondition(policies == [.regular, .regular, .regular, .accessory])
        presence.removeFromSuperview()
        NotificationCenter.default.post(name: NSWindow.didBecomeKeyNotification, object: window)
        precondition(policies.count == 4) // Detached windows no longer affect the app.
        window.contentView!.addSubview(presence)
        precondition(policies.last == .regular && policies.count == 5)
        precondition(!window.isVisible && NSApp.activationPolicy() != .regular && focusRequests == 0)

        let composer = ChatTextView(frame: NSRect(x: 0, y: 0, width: 400, height: 80))
        var submissions = 0
        composer.submit = { submissions += 1 }
        composer.string = "hello"
        composer.setSelectedRange(NSRange(location: 5, length: 0))
        func press(_ flags: NSEvent.ModifierFlags, code: UInt16 = 36) {
            let event = NSEvent.keyEvent(with: .keyDown, location: .zero,
                modifierFlags: flags, timestamp: 0, windowNumber: 0, context: nil,
                characters: "\r", charactersIgnoringModifiers: "\r", isARepeat: false, keyCode: code)!
            composer.keyDown(with: event)
        }
        press([])
        precondition(submissions == 1 && composer.string == "hello")
        press(.shift)
        precondition(submissions == 1 && composer.string == "hello\n")
        press(.command)
        press([], code: 76)
        precondition(submissions == 3)

        let base = CommandLine.arguments[1]
        let fixtureData = Data(base64Encoded: "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+ip1sAAAAASUVORK5CYII=")!
        let fetched = try await ChatMediaLoader.data(from: URL(string: base + "/fixture.png")!)
        precondition(fetched == fixtureData)
        _ = try ChatMediaLoader.image(from: fetched)
        let redirected = try await ChatMediaLoader.data(from: URL(string: base + "/image-redirect")!)
        precondition(redirected == fixtureData)
        do {
            _ = try await ChatMediaLoader.data(from: URL(string: base + "/oversized-image")!)
            fatalError("Oversized media must be refused before reading the body")
        } catch { precondition(error.localizedDescription.contains("limit"), error.localizedDescription) }
        let localMedia = FileManager.default.temporaryDirectory.appendingPathComponent("mlx2-chat-media-test-" + UUID().uuidString + ".png")
        try fixtureData.write(to: localMedia)
        defer { try? FileManager.default.removeItem(at: localMedia) }
        let localResource = try await ChatMediaLoader.resource(from: localMedia, maximumBytes: 1024)
        precondition(localResource.data == fixtureData && localResource.filename.hasSuffix(".png"))
        let chat = ChatModel()
        func server(_ name: String, _ pid: Int = 123) -> JSON {
            JSON(["endpoint": ["url": base, "pid": pid, "model_path": name], "model": name])
        }
        let first = server("synthetic-a")
        let target = JSON(["url": base, "api_key": "synthetic-secret"])
        chat.selected = ChatModel.identity(first)
        chat.draft = "hello"
        chat.send(server: first) { target }
        try await wait(chat)
        precondition(chat.error == nil, chat.error ?? "")
        precondition(chat.conversations[chat.selected]?.last?.text == "Hello 🌍")
        precondition(chat.conversations[chat.selected]?.last?.reasoning == "Considering")
        chat.draft = "follow up"
        chat.send(server: first) { target }
        try await wait(chat)
        precondition(chat.error == nil)
        precondition(chat.conversations[chat.selected]?.count == 4)
        let other = server("synthetic-b")
        chat.selected = ChatModel.identity(other)
        precondition(chat.conversations[chat.selected] == nil)
        precondition(ChatModel.identity(first) != ChatModel.identity(server("synthetic-a", 124)))
        chat.draft = "slow"
        chat.send(server: other) { target }
        try await Task.sleep(for: .milliseconds(300))
        chat.stop()
        try await wait(chat)
        precondition(chat.error == "Reply stopped.")
        precondition(chat.conversations[chat.selected]?.allSatisfy { !$0.committed } == true)
        chat.draft = "after cancel"
        chat.send(server: other) { target }
        try await wait(chat)
        precondition(chat.error == nil)
        chat.clear()
        precondition(chat.conversations[chat.selected]?.isEmpty == true)
        chat.draft = "unauthorized"
        chat.send(server: other) { JSON(["url": base]) }
        try await wait(chat)
        precondition(chat.error?.contains("Access denied") == true)
        chat.draft = "redirect"
        chat.send(server: other) { target }
        try await wait(chat)
        precondition(chat.error?.contains("HTTP 307") == true)
        chat.draft = "broken"
        chat.send(server: other) { target }
        try await wait(chat)
        precondition(chat.error?.contains("before the reply finished") == true)
        chat.draft = "bad request"
        chat.send(server: other) { target }
        try await wait(chat)
        precondition(chat.error?.contains("tools must contain 1 to 64") == true)
        precondition(ChatModel.effectiveToolName("mlx2_tool_call", arguments: ["name": "mlx2__session_compact"]) == "mlx2__session_compact")
        precondition(ChatModel.effectiveToolName("mlx2__snapshot", arguments: ["name": "mlx2__stop"]) == "mlx2__snapshot")
        print("Native chat, server error detail, discovery routing, composer key and Dock lifecycle checks passed")
    }
}
