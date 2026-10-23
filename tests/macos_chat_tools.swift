// Synthetic HTTP only; no model or GPU work.
import AppKit
import Foundation
import SwiftUI

@main enum ChatToolChecks {
    @MainActor static func wait(_ chat: ChatModel) async throws {
        for _ in 0..<500 {
            if !chat.generating { return }
            try await Task.sleep(for: .milliseconds(20))
        }
        fatalError("Tool chat did not finish")
    }
    @MainActor static func choiceChecks() async throws {
        func panel(_ id: String, presentation: String = "auto", autoPick: Bool = false, seconds: Int = 1) -> JSON {
            JSON(["panel_id": id, "presentation": presentation, "title": "Synthetic preference",
                  "question": "Choose a display style", "recommended_id": "b",
                  "choices": [["id": "a", "label": "A", "description": "First"], ["id": "b", "label": "B", "description": "Recommended"]],
                  "policy": ["auto_pick": autoPick, "timeout_seconds": seconds]])
        }
        func pending(_ choices: AgentChoices, id: String) async throws {
            for _ in 0..<100 {
                if choices.pending?.id == id { return }
                try await Task.sleep(for: .milliseconds(10))
            }
            fatalError("Synthetic choice did not become pending")
        }
        let choices = AgentChoices()
        do {
            _ = try await choices.present(JSON(["panel_id": "bad", "policy": ["timeout_seconds": Double.infinity]]), world: false)
            fatalError("Infinite choice timeout accepted")
        } catch {}
        precondition(choices.pending == nil)

        let popup = Task { @MainActor in try await choices.present(panel("popup"), world: false) }
        try await pending(choices, id: "popup")
        precondition(choices.pending?.value["presentation"].text == "popup")
        choices.resolve("not-offered")
        choices.resolve("b", source: "timeout")
        precondition(choices.pending?.id == "popup")
        do {
            _ = try await choices.present(panel("overlap"), world: false)
            fatalError("Concurrent choice continuation replaced the first one")
        } catch {}
        choices.resolve("a", panelID: "popup")
        let first = try await popup.value
        precondition(first["choice_id"] as? String == "a")
        precondition(first["source"] as? String == "user")
        precondition(first["is_authorization"] as? Bool == false)
        choices.resolve("b")  // resolved continuations are never resumed twice
        let spatial = Task { @MainActor in try await choices.present(panel("spatial"), world: true) }
        try await pending(choices, id: "spatial")
        precondition(choices.pending?.value["presentation"].text == "object")
        choices.cancel(panelID: "popup")
        choices.resolve("b", panelID: "popup")
        precondition(choices.pending?.id == "spatial")  // stale sheet callbacks cannot cancel/choose a newer panel
        choices.resolve("b", panelID: "spatial")
        let second = try await spatial.value
        precondition(second["is_authorization"] as? Bool == false)
        let explicitPopup = Task { @MainActor in try await choices.present(panel("explicit", presentation: "popup"), world: true) }
        try await pending(choices, id: "explicit")
        precondition(choices.pending?.value["presentation"].text == "popup")
        choices.cancel(panelID: "explicit")
        let cancelled = try await explicitPopup.value
        precondition(cancelled["source"] as? String == "cancelled")
        let disabled = Task { @MainActor in try await choices.present(panel("disabled", autoPick: true), world: false) }
        try await pending(choices, id: "disabled")
        choices.autoPick = false
        try await Task.sleep(for: .milliseconds(1150))
        precondition(choices.pending?.id == "disabled" && choices.remaining == 1)
        disabled.cancel()
        let stopped = try await disabled.value
        precondition(stopped["source"] as? String == "cancelled")
        precondition(stopped["is_authorization"] as? Bool == false)
        precondition(choices.pending == nil)
        let timeout = Task { @MainActor in try await choices.present(panel("timeout", autoPick: true), world: true) }
        try await pending(choices, id: "timeout")
        let timed = try await timeout.value
        precondition(timed["choice_id"] as? String == "b")
        precondition(timed["source"] as? String == "timeout")
        precondition(timed["is_authorization"] as? Bool == false)
        precondition(choices.pending == nil)
        print("Native choice popup/object routing, immutable panel fencing, disabled auto-pick, recommendation timeout, cancellation and non-authorization checks passed")
    }
    @MainActor static func toolCardChecks() throws {
        let running = ChatToolEvent(id: "card-call", name: "mlx2__memory_get", arguments: "{\"id\":\"note\"}")
        precondition(running.title == "Read memory" && running.statusTitle == "Running")
        let routed = ChatToolEvent(name: "mlx2_tool_call", arguments: "{\"name\":\"mlx2__skill_view\"}")
        precondition(routed.title == "Read a skill")
        precondition(ChatToolEvent(name: "server__search_documents").title == "Search Documents")
        let model = ChatModel()
        model.conversations["cards"] = [ChatMessage(role: "user", text: "Original prompt"), ChatMessage(role: "assistant", text: "Answer")]
        model.upsertToolEvent(running, conversationID: "cards", messageIndex: 1)
        let completed = ChatToolEvent(id: running.id, name: running.name, arguments: running.arguments, result: "{\"found\":true}", status: "completed")
        model.upsertToolEvent(completed, conversationID: "cards", messageIndex: 1)
        model.upsertToolEvent(completed, conversationID: "cards", messageIndex: 99)
        model.upsertToolEvent(completed, conversationID: "cards", messageIndex: 0)
        precondition(model.conversations["cards"]?[0].toolEvents.isEmpty == true)
        precondition(model.conversations["cards"]?[1].toolEvents.count == 1)
        let restored = ChatModel()
        let data = try JSONSerialization.data(withJSONObject: model.profileArchive())
        restored.restoreProfileArchive(JSON(try JSONSerialization.jsonObject(with: data)))
        let event = restored.conversations["cards"]?[1].toolEvents.first
        precondition(event?.id == running.id && event?.result == completed.result && event?.status == "completed")
        precondition(event?.name == completed.name && event?.arguments == completed.arguments)
        precondition(ChatToolEvent.restore([running.archive, running.archive]).count == 1)
        precondition(ChatToolEvent.restore([running.archive]).first?.statusTitle == "Interrupted")
        restored.restoreProfileArchive(JSON(["schema_version": 1, "conversations": ["legacy": [["role": "assistant", "text": "Older answer", "tool_activity": "Older tool activity"]]]]))
        precondition(restored.conversations["legacy"]?.first?.toolEvents.isEmpty == true)
        precondition(restored.conversations["legacy"]?.first?.toolActivity == "Older tool activity")
        precondition(restored.conversations["legacy"]?.first?.createdAt == nil)
        let time = Date(timeIntervalSince1970: 1234567890.125)
        let message = ChatMessage(role: "assistant", text: " Exact answer\n", createdAt: time)
        model.conversations["dated"] = [ChatMessage(role: "user", text: "Prompt", createdAt: time), message]
        restored.restoreProfileArchive(JSON(try JSONSerialization.jsonObject(with: JSONSerialization.data(withJSONObject: model.profileArchive()))))
        precondition(restored.conversations["dated"]?.allSatisfy { $0.createdAt == time } == true)
        precondition(ChatMessage(role: "assistant", text: "New reply").createdAt != nil)
        let clipboard = NSPasteboard.withUniqueName()
        defer { clipboard.releaseGlobally() }
        precondition(message.copy(to: clipboard) && clipboard.string(forType: .string) == " Exact answer\n")
        let toolOnly = ChatMessage(role: "assistant", text: "", toolEvents: [completed])
        precondition(toolOnly.copy(to: clipboard) && clipboard.string(forType: .string) == "Read memory: Completed")
        print("Tool card friendly labels, status, safe updates and archive compatibility checks passed")
    }

    @MainActor static func payloadLayoutChecks() async throws {
        let short = NSHostingView(rootView: ChatToolPayload(title: "Arguments", value: "{\"job_id\":\"fixture\"}").frame(width: 560))
        let long = NSHostingView(rootView: ChatToolPayload(title: "Result", value: String(repeating: "A bounded result line\n", count: 100)).frame(width: 560))
        let window = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 600, height: 700), styleMask: [.borderless], backing: .buffered, defer: false)
        window.isReleasedWhenClosed = false
        let stack = NSStackView(views: [short, long]); stack.orientation = .vertical
        window.contentView = stack
        stack.layoutSubtreeIfNeeded()
        try await Task.sleep(for: .milliseconds(150))
        stack.layoutSubtreeIfNeeded()
        precondition(short.fittingSize.height < 80, "Short arguments must hug their content")
        precondition(long.fittingSize.height >= 180 && long.fittingSize.height < 230, "Long results must remain scrollable and capped")
        window.close()
        print("Tool payload short-content sizing and long-content scroll cap checks passed")
    }

    @MainActor static func bugsweepChecks() async throws {
        let unicode = "a\u{2028}b\u{2029}c\u{0085}d"
        let wire = ": comment\r\ndata: " + unicode + "\r\ndata: second\r\n\r\ndata: [DONE]\n\n"
        let stream = AsyncStream<UInt8> { continuation in
            for byte in wire.utf8 { continuation.yield(byte) }
            continuation.finish()
        }
        var payloads: [String] = []
        for try await payload in ServerSentEvents(stream) { payloads.append(payload) }
        precondition(payloads == [unicode + "\nsecond", "[DONE]"])

        let chat = ChatModel()
        let result = String(repeating: "R", count: 1024 * 1024)
        let event = ChatToolEvent(name: "fetch_page", result: result, status: "completed")
        chat.restoreProfileArchive(JSON(["schema_version": 1, "selected": "large",
            "wire_history": ["large": [["role": "tool", "tool_call_id": event.id, "content": result]]]]))
        chat.conversations["large"] = [ChatMessage(role: "assistant", text: "Answer", toolEvents: Array(repeating: event, count: 10))]
        chat.conversations["trash"] = [ChatMessage(role: "user", text: result)]
        chat.setHistoryState("deleted", for: "trash")
        let fullSize = try JSONSerialization.data(withJSONObject: chat.profileArchive()).count
        precondition(fullSize > 8 * 1024 * 1024)
        let compact = chat.profileArchive(compact: true)
        let compactSize = try JSONSerialization.data(withJSONObject: compact).count
        precondition(compactSize < 8 * 1024 * 1024)
        precondition((compact["conversations"] as? [String: Any])?["trash"] != nil)
        precondition((compact["wire_history"] as? [String: [[String: Any]]])?["large"]?.first?["content"] as? String == result)

        let endpoint = JSON(["endpoint": ["url": "http://127.0.0.1:1", "pid": 2, "model_path": "fixture", "checkpoint": "fixture@sha256:abc"]])
        let empty = ChatModel()
        empty.selected = "empty"
        empty.contextUsageByConversation["empty"] = ChatContextUsage(input: 100, capacity: 9999, reserved: 1, output: 1, exact: true)
        precondition(empty.bindEmptyConversation(to: endpoint))
        precondition(empty.contextUsageByConversation["empty"] == nil)
        empty.mutationsPaused = true
        empty.draft = "Keep this draft across the transition"
        var resolvedWhilePaused = false
        empty.send(server: endpoint, resolve: { resolvedWhilePaused = true; return JSON([:]) })
        precondition(empty.generationID == nil && !resolvedWhilePaused)
        precondition(empty.draft == "Keep this draft across the transition")

        let soleCopy = ChatModel()
        soleCopy.conversations["only"] = [ChatMessage(role: "assistant", text: "Answer", toolEvents: [event])]
        let savedOnly = JSON(soleCopy.profileArchive(compact: true))
        precondition(savedOnly["conversations"]["only"][0]["tool_events"][0]["result"].string == result)
        print("Bug-sweep SSE Unicode/multiline framing, bounded archive and context reset checks passed")
    }

    @MainActor static func main() async throws {
        _ = NSApplication.shared
        precondition(NSApp.activationPolicy() != .regular)
        try await bugsweepChecks()
        try toolCardChecks()
        try await payloadLayoutChecks()
        try await choiceChecks()
        // Profile replacement must fence callbacks from a cancelled old resolver.
        let offline = ChatModel()
        let oldServer = JSON(["endpoint": ["url": "http://127.0.0.1:1", "pid": 1, "model_path": "offline"]])
        offline.selected = ChatModel.identity(oldServer); offline.draft = "Old private draft"
        var oldResolution: CheckedContinuation<JSON, Never>?
        offline.send(server: oldServer, resolve: { await withCheckedContinuation { oldResolution = $0 } })
        let oldGeneration = offline.generationID
        precondition(oldGeneration != nil)
        while oldResolution == nil { await Task.yield() }
        offline.restoreProfileArchive(JSON(["schema_version": 1, "draft": "New private draft"]))
        precondition(!offline.generating && offline.generationID == nil && offline.selected.isEmpty && offline.error == nil)
        offline.selected = ChatModel.identity(oldServer)
        var newResolution: CheckedContinuation<JSON, Error>?
        offline.send(server: oldServer, resolve: { try await withCheckedThrowingContinuation { newResolution = $0 } })
        let newGeneration = offline.generationID
        precondition(newGeneration != nil && newGeneration != oldGeneration)
        while newResolution == nil { await Task.yield() }
        oldResolution?.resume(returning: JSON(["url": "http://127.0.0.1:1"]))
        for _ in 0..<10 { await Task.yield() }
        precondition(offline.generating && offline.generationID == newGeneration && offline.error == nil && offline.draft.isEmpty)
        newResolution?.resume(throwing: DesktopError.message("Synthetic model is offline"))
        try await wait(offline)
        precondition(offline.generationSucceeded(oldGeneration!) == false && offline.generationSucceeded(newGeneration!) == false)
        precondition(offline.generationID == nil && offline.draft.isEmpty && offline.failedPrompt == "New private draft" && offline.error == "Synthetic model is offline")
        offline.restoreProfileArchive(JSON([:]))
        precondition(offline.error == nil && offline.draft.isEmpty && offline.selected.isEmpty)
        print("Offline resolver and profile-switch cancellation isolation checks passed")
        let base = CommandLine.arguments[1]
        let server = JSON(["endpoint": ["url": base, "pid": 123, "model_path": "synthetic-tools"]])
        let target = JSON(["url": base, "input_modalities": ["text", "image"]])
        let context = JSON(["session_id": "synthetic-session", "tools": [["type": "function", "function": ["name": "mlx2__preferences", "parameters": ["type": "object"]]]]])
        let retried = ChatModel()
        retried.selected = ChatModel.identity(server); retried.draft = "retry failed tool"
        var attempts = 0
        retried.send(server: server, resolve: { target }, origin: "Subagent result", tools: { context }, execute: { _, _, _, _ in
            attempts += 1
            return JSON(["error": "synthetic missing capability"])
        })
        try await wait(retried)
        precondition(attempts == 1 && retried.error == nil)
        precondition(retried.conversations[retried.selected]?.last?.text.contains("change the approach") == true)
        precondition(retried.conversations[retried.selected]?.first?.origin == "Subagent result")
        let restoredCallback = ChatModel()
        restoredCallback.restoreProfileArchive(JSON(retried.profileArchive()))
        precondition(restoredCallback.conversations[retried.selected]?.first?.origin == "Subagent result")
        print("Repeated failed tool execution suppression and callback origin persistence checks passed")
        let lateSchema = JSON(["type": "function", "function": ["name": "mlx2__late_action", "description": "Synthetic allowed action", "parameters": ["type": "object", "properties": [:], "additionalProperties": false]]])
        let lateContext = JSON(["session_id": "synthetic-session", "tools": [["type": "function", "function": ["name": "mlx2_tool_search", "parameters": ["type": "object"]]], ["type": "function", "function": ["name": "mlx2_tool_call", "parameters": ["type": "object"]]]]])
        for prompt in ["late declared", "late alias", "discovered direct", "discovered routed", "late forbidden"] {
            let late = ChatModel()
            late.selected = ChatModel.identity(server); late.draft = prompt
            var metadataReads = 0, executedActions = 0
            late.send(server: server, resolve: { target }, tools: { lateContext }, execute: { _, name, arguments, _ in
                if name == "mlx2_tool_search" {
                    metadataReads += 1
                    return JSON(["result": ["tools": prompt == "late forbidden" ? [] : [lateSchema.raw]]])
                }
                precondition(name == "mlx2__late_action" || (name == "mlx2_tool_call" && arguments["name"] as? String == "mlx2__late_action"))
                executedActions += 1
                return JSON(["result": ["completed": true]])
            })
            try await wait(late)
            precondition(metadataReads == (prompt == "discovered direct" ? 2 : 1))
            if prompt == "late forbidden" {
                precondition(executedActions == 0 && late.error?.contains("undeclared") == true)
            } else {
                precondition(late.error == nil, late.error ?? "")
                precondition(executedActions == 1 && late.conversations[late.selected]?.last?.text == "Declared tool completed")
            }
        }
        let fullInventory = (0..<64).map { JSON(["type": "function", "function": ["name": "existing_\($0)"]]).raw }
        let rotated = ChatModel.declaringTools([lateSchema], in: fullInventory)
        precondition(rotated.count == 64 && JSON(rotated[0])["function"]["name"].string == "existing_0")
        precondition(JSON(rotated.last!)["function"]["name"].string == "mlx2__late_action")
        var toolBudget = ChatToolBudget()
        precondition(toolBudget.guidance(calls: 1, negotiating: false, previousContext: 100, contextLimit: 32768) == toolBudget.guidance(calls: 12, negotiating: false, previousContext: 900, contextLimit: 32768))
        let planned = toolBudget.request(["additional_calls": 20, "reason": "Read source chunks"], calls: 1)
        precondition(planned["result"]["total_call_limit"].double == 26)
        precondition(toolBudget.request(["additional_calls": true, "reason": "Invalid"], calls: 2)["error"].string != nil)
        for _ in 0..<5 { _ = toolBudget.request(["additional_calls": 96, "reason": "Bounded large task"], calls: 120) }
        precondition(toolBudget.limit <= 128 && !toolBudget.canExtend(calls: 120))
        for prompt in ["planned tools", "budget boundary", "budget boundary rejection"] {
            let plannedChat = ChatModel()
            plannedChat.selected = ChatModel.identity(server); plannedChat.draft = prompt
            var work = 0
            plannedChat.send(server: server, resolve: { target }, tools: { context }, execute: { _, name, _, _ in
                precondition(name != ChatToolBudget.name) // Local budget tools never execute backend operations.
                work += 1
                return JSON(["result": ["completed": true]])
            })
            try await wait(plannedChat)
            precondition(plannedChat.error == nil, plannedChat.error ?? "")
            precondition(work == 20)
            precondition(plannedChat.conversations[plannedChat.selected]?.last?.text == "Completed twenty planned calls.")
        }
        let recovered = ChatModel()
        recovered.selected = ChatModel.identity(server); recovered.draft = "recover answer"
        var recoveryExecutions = 0
        recovered.send(server: server, resolve: { target }, tools: { context }, execute: { _, _, _, _ in
            recoveryExecutions += 1
            return JSON(["result": ["complete": true]])
        })
        try await wait(recovered)
        precondition(recovered.error == nil, recovered.error ?? "")
        precondition(recoveryExecutions == 1)
        precondition(recovered.conversations[recovered.selected]?.last?.text == "Answer recovered from saved tool results.")
        precondition(recovered.profileArchive()["wire_history"] != nil)
        let looping = ChatModel()
        looping.selected = ChatModel.identity(server); looping.draft = "loop tools"
        var loopExecutions = 0
        looping.send(server: server, resolve: { JSON(["url": base, "context_limit": 32768]) }, tools: { context }, execute: { _, _, _, _ in
            loopExecutions += 1
            return JSON(["result": ["complete": true, "next_offset": NSNull(), "groups": ["memory": ["memory_search"]]]])
        })
        try await wait(looping)
        precondition(looping.error == nil, looping.error ?? "")
        precondition(loopExecutions == 2)
        precondition(looping.contextUsageByConversation[looping.selected]?.used == 350)
        precondition(looping.contextUsageByConversation[looping.selected]?.capacity == 32768)
        let restoredUsage = ChatModel()
        restoredUsage.restoreProfileArchive(JSON(looping.profileArchive()))
        precondition(restoredUsage.contextUsageByConversation[looping.selected]?.used == 350)
        restoredUsage.restoreProfileArchive(JSON([:]))
        precondition(restoredUsage.contextUsageByConversation.isEmpty)
        precondition(ChatContextUsage.restore(JSON(["capacity": Double.infinity])) == nil)
        precondition(ChatContextUsage(input: 200, capacity: 100, reserved: 20).percent == 100)
        precondition(looping.conversations[looping.selected]?.count == 2)
        precondition(looping.conversations[looping.selected]?.last?.text == "Here is what I found within the tool budget.")
        precondition(looping.conversations[looping.selected]?.last?.toolActivity.contains("mlx2_tool_search") == true)
        precondition(looping.conversations[looping.selected]?.last?.toolEvents.count == 2)
        let cappedResult = JSON(["result": ["text": String(repeating: "🌍\n\"", count: 1000)], "result_context_max_bytes": 1024,
            "native_action": "session_compact", "checkpoint": ["private_native_receipt": "Not model-facing"]])
        let cappedContent = ChatModel.toolResultContent(cappedResult, operation: "synthetic", encoded: Data())
        precondition(cappedContent.utf8.count <= 1024 && !cappedContent.contains("private_native_receipt"))
        let summaryResult = JSON(["result": ["summary": "Tool search completed. Memory tools are available."]])
        precondition(ChatModel.toolResultContent(summaryResult, operation: "mlx2_tool_search", encoded: Data("raw".utf8)) == "Tool search completed. Memory tools are available.")
        precondition(ChatModel.toolResultContent(summaryResult, operation: "memory_search", encoded: Data("raw".utf8)).contains("Tool memory_search finished."))
        precondition(ChatModel.toolResultContent(JSON(["result": ["workers": []]]), operation: "mlx2__snapshot", encoded: Data()).contains("app-managed worker count = 0"))
        precondition(!ChatModel.toolResultContent(JSON(["result": ["truncated": true]]), operation: "mlx2__snapshot", encoded: Data()).contains("worker count = 0"))
        precondition(ChatModel.isRawToolRequest("<tool_call>\n<function=search>"))
        precondition(!ChatModel.isRawToolRequest("Example documentation: <tool_call>"))
        let rawToolText = ChatModel()
        rawToolText.selected = ChatModel.identity(server); rawToolText.draft = "raw tool text"
        var rawExecutions = 0
        rawToolText.send(server: server, resolve: { target }, tools: { context }, execute: { _, _, _, _ in
            rawExecutions += 1; return JSON([:])
        })
        try await wait(rawToolText)
        precondition(rawExecutions == 0)
        precondition(rawToolText.error?.contains("not executed") == true)
        precondition(rawToolText.conversations[rawToolText.selected]?.last?.text.isEmpty == true)
        precondition(rawToolText.conversations[rawToolText.selected]?.last?.toolEvents.last?.status == "skipped")
        let restoredActivity = ChatModel()
        restoredActivity.restoreProfileArchive(JSON(looping.profileArchive()))
        precondition(restoredActivity.conversations[looping.selected]?.last?.toolActivity == looping.conversations[looping.selected]?.last?.toolActivity)
        let lost = ChatModel()
        lost.selected = ChatModel.identity(server); lost.draft = "tool request"
        var disconnectedCalls = 0
        lost.send(server: server, resolve: { target }, tools: { context }, execute: { _, _, _, _ in
            disconnectedCalls += 1
            throw DesktopConnectionFailure(message: "Synthetic wire loss; reopen the app")
        })
        try await wait(lost)
        precondition(disconnectedCalls == 1)
        precondition(lost.error?.contains("Synthetic wire loss") == true)
        precondition(lost.conversations[lost.selected]?.last?.toolEvents.last?.status == "failed")
        let chat = ChatModel()
        chat.selected = ChatModel.identity(server)
        let file = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString + ".png")
        try Data(base64Encoded: "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aA3cAAAAASUVORK5CYII=")!.write(to: file)
        defer { try? FileManager.default.removeItem(at: file) }
        let image = try ChatAttachment.read(file)
        precondition(image.modality == "image")
        do {
            _ = try ChatAttachment.content(prompt: "x", attachments: [image], target: JSON([:]))
            fatalError("Unadvertised image input accepted")
        } catch {}
        var executed = 0
        var approvals = 0
        func execute(_ session: String, _ name: String, _ arguments: [String: Any], _ approved: Bool) async throws -> JSON {
            precondition(session == "synthetic-session" && name == "mlx2__preferences")
            precondition((arguments["changes"] as? [String: Any])?["agent_controls"] as? Bool == false)
            executed += 1
            if !approved { return JSON(["requires_approval": true]) }
            approvals += 1
            return JSON(["result": ["updated": true]])
        }
        for invalidBudget in [JSON(["url": base, "output_tokens": Double.infinity]), JSON(["url": base, "context_limit": Double.nan])] {
            let invalid = ChatModel()
            invalid.selected = ChatModel.identity(server); invalid.draft = "invalid budget"
            invalid.send(server: server, resolve: { invalidBudget })
            try await wait(invalid)
            precondition(invalid.error?.contains("Invalid") == true)
        }
        let smallBudget = ChatModel()
        smallBudget.selected = ChatModel.identity(server); smallBudget.draft = "budget small context"
        smallBudget.send(server: server, resolve: {
            JSON(["url": base, "context_limit": 1024, "output_tokens": 128,
                  "chat_options": ["thinking_budget": 0]])
        })
        try await wait(smallBudget)
        precondition(smallBudget.error == nil)
        let unrelated = ChatModel()
        unrelated.selected = ChatModel.identity(server); unrelated.draft = "budget unrelated UI failure"
        unrelated.send(server: server, resolve: {
            unrelated.error = "Unrelated attachment failure"
            return JSON(["url": base, "context_limit": 8192, "output_tokens": 128])
        })
        let unrelatedTurn = unrelated.generationID!
        try await wait(unrelated)
        precondition(unrelated.generationSucceeded(unrelatedTurn) == true)
        let oversizedCount = ChatModel()
        oversizedCount.selected = ChatModel.identity(server); oversizedCount.draft = "count overflow"
        oversizedCount.send(server: server, resolve: { JSON(["url": base, "context_limit": 8192, "output_tokens": 4096]) })
        try await wait(oversizedCount)
        precondition(oversizedCount.error?.contains("context budget") == true)
        let managedA = JSON(["endpoint": ["managed": true, "resume_lineage_id": "synthetic-chain", "checkpoint": "synthetic@sha", "url": base, "pid": 11, "model_path": "synthetic-managed"]])
        let managedB = JSON(["endpoint": ["managed": true, "resume_lineage_id": "synthetic-chain", "checkpoint": "synthetic@sha", "url": base, "pid": 12, "model_path": "synthetic-managed", "resume_previous_pid": 11, "resume_previous_url": base, "resume_previous_checkpoint": "synthetic@sha"]])
        precondition(ChatModel.identity(managedA) == ChatModel.identity(managedB))
        let fresh = JSON(["endpoint": ["managed": true, "resume_lineage_id": "fresh-load", "checkpoint": "synthetic@sha", "url": base, "pid": 12, "model_path": "synthetic-managed"]])
        precondition(ChatModel.identity(fresh) != ChatModel.identity(managedA))
        let legacy = ChatModel.identity(JSON(["endpoint": ["url": base, "pid": 11, "model_path": "synthetic-managed"]]))
        let bound = ChatModel()
        let binding: [String: Any] = ["url": base, "pid": 11, "checkpoint": "synthetic@sha"]
        let archivedMessages = [["role": "user", "text": "Private before suspension", "committed": true]]
        bound.restoreProfileArchive(JSON(["schema_version": 1, "selected": legacy, "conversations": [legacy: archivedMessages], "endpoint_bindings": [legacy: binding]]))
        bound.reconcileManagedResumes([managedB])
        precondition(bound.selected == legacy && bound.serverIdentity(for: legacy) == ChatModel.identity(managedB))
        precondition(bound.conversations[bound.selected]?.first?.text == "Private before suspension")
        let unbound = ChatModel()
        unbound.restoreProfileArchive(JSON(["schema_version": 1, "selected": legacy, "conversations": [legacy: archivedMessages]]))
        unbound.reconcileManagedResumes([managedB])
        precondition(unbound.selected == legacy && unbound.conversations[ChatModel.identity(managedB)] == nil)
        let changed = ChatModel()
        changed.restoreProfileArchive(JSON(["schema_version": 1, "selected": legacy, "conversations": [legacy: archivedMessages], "endpoint_bindings": [legacy: binding]]))
        let changedServer = JSON(["endpoint": ["managed": true, "resume_lineage_id": "synthetic-chain", "checkpoint": "changed@sha", "url": base, "pid": 12, "model_path": "synthetic-managed", "resume_previous_pid": 11, "resume_previous_url": base, "resume_previous_checkpoint": "synthetic@sha"]])
        changed.reconcileManagedResumes([changedServer])
        precondition(changed.selected == legacy)
        let incomplete = ChatModel()
        incomplete.restoreProfileArchive(JSON(["schema_version": 1, "wire_history": [legacy: [["role": "assistant", "tool_calls": [["id": "pending"]]], ["role": "assistant", "content": "invalid missing tool result"]]]]))
        precondition(incomplete.completedForkHistory(for: legacy).isEmpty)
        bound.reconcileManagedResumes([fresh])
        precondition(bound.selected == legacy && bound.serverIdentity(for: legacy) == ChatModel.identity(fresh))
        let freshChat = ChatModel()
        let oldSession = ChatModel.identity(server)
        freshChat.restoreProfileArchive(JSON(["schema_version": 1, "selected": oldSession,
            "conversations": [oldSession: [["role": "user", "text": "Previous private conversation", "committed": true]]],
            "wire_history": [oldSession: [["role": "user", "content": "Previous private conversation"]]]]))
        let newSession = freshChat.newConversation()!
        precondition(newSession != oldSession && freshChat.serverIdentity(for: newSession) == oldSession)
        precondition(freshChat.branches[newSession]?["parent_id"] == nil)
        precondition(freshChat.completedForkHistory(for: newSession).isEmpty)
        precondition(freshChat.conversations[oldSession]?.first?.text == "Previous private conversation")
        let freshDocument = try JSONSerialization.data(withJSONObject: freshChat.sessionArchiveDocument(for: newSession))
        precondition(!String(decoding: freshDocument, as: UTF8.self).contains("Previous private conversation"))
        let reopenedFreshChat = ChatModel()
        reopenedFreshChat.restoreProfileArchive(JSON(freshChat.profileArchive()))
        precondition(reopenedFreshChat.selected == newSession && reopenedFreshChat.serverIdentity(for: newSession) == oldSession)
        precondition(reopenedFreshChat.branches[newSession]?["parent_id"] == nil)
        let secondServer = JSON(["endpoint": ["url": base + "/second", "pid": 998, "model_path": "synthetic-second"]])
        let selection = ChatModel()
        let blankRoot = selection.newConversation(servers: [server, secondServer])!
        precondition(selection.serverIdentity(for: blankRoot) == ChatModel.identity(server))
        selection.selected = ChatModel.identity(secondServer)
        let preserved = selection.newConversation(servers: [server, secondServer])!
        precondition(selection.serverIdentity(for: preserved) == ChatModel.identity(secondServer))
        selection.selected = "stale-offline"
        selection.conversations["stale-offline"] = [ChatMessage(role: "user", text: "Keep private history")]
        let replacement = selection.newConversation(servers: [server, secondServer])!
        precondition(selection.serverIdentity(for: replacement) == ChatModel.identity(server))
        precondition(selection.conversations["stale-offline"]?.first?.text == "Keep private history")
        selection.selected = "stale-offline"
        selection.selectFirstIfNeeded([server])
        precondition(selection.selected == "stale-offline")
        selection.selected = ""
        selection.selectFirstIfNeeded([server])
        precondition(selection.selected == ChatModel.identity(server))
        selection.selected = preserved
        selection.selectFirstIfNeeded([server])
        precondition(selection.selected == preserved && selection.serverIdentity(for: preserved) == ChatModel.identity(server))
        precondition(selection.newConversation(servers: []) == nil)
        let automatedFailure = ChatModel()
        automatedFailure.selected = ChatModel.identity(server); automatedFailure.draft = "Synthetic scheduled prompt"
        automatedFailure.send(server: server, resolve: { throw DesktopError.message("Synthetic offline") }, restoreFailedDraft: false, tools: nil, execute: nil)
        try await wait(automatedFailure)
        precondition(automatedFailure.draft.isEmpty && automatedFailure.error == "Synthetic offline")
        precondition(automatedFailure.canRetryFailedPrompt)
        automatedFailure.draft = "New user edit"
        precondition(!automatedFailure.retryFailedPrompt() && automatedFailure.draft == "New user edit")
        automatedFailure.draft = ""
        precondition(automatedFailure.retryFailedPrompt() && automatedFailure.draft == "Synthetic scheduled prompt")
        let attachmentEdit = ChatModel()
        attachmentEdit.selected = ChatModel.identity(server); attachmentEdit.draft = "Earlier failed prompt"
        var attachmentResolution: CheckedContinuation<JSON, Error>?
        attachmentEdit.send(server: server, resolve: { try await withCheckedThrowingContinuation { attachmentResolution = $0 } })
        while attachmentResolution == nil { await Task.yield() }
        attachmentEdit.draft = "New typed draft"
        attachmentEdit.attachments = [.init(name: "new-document.txt", modality: "text", part: ["type": "text", "text": "New user attachment"])]
        attachmentResolution?.resume(throwing: DesktopError.message("Synthetic offline"))
        try await wait(attachmentEdit)
        precondition(attachmentEdit.draft == "New typed draft" && attachmentEdit.attachments.first?.name == "new-document.txt")
        precondition(attachmentEdit.failedPrompt == "Earlier failed prompt")
        let skillDocument = try SkillEditorDocument(body: "Reusable checked instructions", referenceJSON: "{\"references/notes.md\":\"Reference text\"}")
        precondition(skillDocument.files["references/notes.md"] == "Reference text")
        for invalidReferences in ["[]", "{\"bad\":42}", "{not json}"] {
            do { _ = try SkillEditorDocument(body: "Instructions", referenceJSON: invalidReferences); fatalError("Invalid skill reference editor accepted") } catch {}
        }
        do { _ = try SkillEditorDocument(body: String(repeating: "x", count: 65537), referenceJSON: "{}"); fatalError("Oversized skill instructions accepted") } catch {}
        precondition(SkillGuidance.prompt.contains("unverified") && SkillGuidance.prompt.contains("never grant tool permissions"))
        let requestedTools = JSON(["tools": ["memory_get"], "tool_requests": [["tools": ["memory_get", "skills_list", "shell_execute"], "reason": "Need procedural guidance"]]])
        precondition(ChildTools.grantable(in: requestedTools) == ["skills_list"])
        precondition(!ChildTools.requested(in: requestedTools).contains("shell_execute"))
        let invalidIndex = ChatModel()
        invalidIndex.selected = ChatModel.identity(server); invalidIndex.draft = "invalid index"
        invalidIndex.send(server: server, resolve: { target }, tools: { context }, execute: execute)
        try await wait(invalidIndex)
        precondition(invalidIndex.error == "Invalid tool call index.")
        precondition(executed == 0)
        let cancelledTool = ChatModel()
        cancelledTool.selected = ChatModel.identity(server); cancelledTool.draft = "tool request"
        var pendingTool: CheckedContinuation<JSON, Never>?
        cancelledTool.send(server: server, resolve: { target }, tools: { context }, execute: { _, _, _, _ in
            await withCheckedContinuation { pendingTool = $0 }
        })
        for _ in 0..<300 { if pendingTool != nil { break }; try await Task.sleep(for: .milliseconds(10)) }
        precondition(pendingTool != nil)
        cancelledTool.restoreProfileArchive(JSON(["schema_version": 1, "selected": ChatModel.identity(server), "draft": "Replacement profile draft"]))
        pendingTool?.resume(returning: JSON(["requires_approval": true]))
        for _ in 0..<20 { await Task.yield() }
        precondition(cancelledTool.approval == nil && !cancelledTool.generating)
        precondition(cancelledTool.draft == "Replacement profile draft" && cancelledTool.conversations.isEmpty)
        precondition((cancelledTool.profileArchive()["wire_history"] as? [String: [[String: Any]]])?.isEmpty == true)
        chat.attachments = [image]
        chat.draft = "tool request"
        chat.send(server: server, resolve: { target }, systemPrompt: "base and extra prompt", tools: {
            precondition(chat.completedForkHistory(for: chat.selected).last?["role"] as? String == "user")
            return context
        }, execute: execute)
        for _ in 0..<300 {
            if chat.approval != nil { break }
            try await Task.sleep(for: .milliseconds(10))
        }
        precondition(chat.approval?.name == "mlx2__preferences")
        precondition(executed == 1 && approvals == 0)
        chat.approveTool(true)
        let successfulGeneration = chat.generationID!
        let scheduledInteraction = chat.interactionRevision
        try await wait(chat)
        precondition(chat.generationSucceeded(successfulGeneration) == true)
        precondition(chat.mayRestoreScheduledSelection(generation: successfulGeneration, interaction: scheduledInteraction, sessionID: chat.selected))
        chat.draft = "User began composing after background completion"
        chat.draft = ""
        precondition(!chat.mayRestoreScheduledSelection(generation: successfulGeneration, interaction: scheduledInteraction, sessionID: chat.selected))

        precondition(chat.error == nil, chat.error ?? "")
        precondition(executed == 2 && approvals == 1)
        precondition(chat.conversations[chat.selected]?.last?.text == "Tool completed")
        let interruptedHistory = ChatModel()
        interruptedHistory.restoreProfileArchive(JSON(chat.profileArchive()))
        interruptedHistory.draft = "tool request"
        var interruptedResult: CheckedContinuation<JSON, Never>?
        interruptedHistory.send(server: server, resolve: { target }, tools: { context }, execute: { _, _, _, _ in
            await withCheckedContinuation { interruptedResult = $0 }
        })
        for _ in 0..<300 { if interruptedResult != nil { break }; try await Task.sleep(for: .milliseconds(10)) }
        precondition(interruptedResult != nil)
        interruptedHistory.stop()
        interruptedResult?.resume(returning: JSON(["result": ["done": true]]))
        try await wait(interruptedHistory)
        let interruptedWire = (interruptedHistory.profileArchive()["wire_history"] as? [String: [[String: Any]]])?[interruptedHistory.selected] ?? []
        precondition(interruptedWire.last?["role"] as? String == "tool")
        precondition((interruptedWire.last?["content"] as? String)?.contains("interrupted") == true)
        chat.draft = "follow up"
        chat.send(server: server, resolve: { target }, systemPrompt: "new base prompt", tools: { context }, execute: execute)
        try await wait(chat)
        precondition(chat.error == nil)
        precondition(chat.conversations[chat.selected]?.last?.text == "History preserved")
        let parentConversation = chat.selected
        let parentCount = chat.conversations[parentConversation]?.count
        let forkPoint = chat.conversations[parentConversation]!.first(where: { $0.text == "Tool completed" })!
        precondition(chat.canFork(at: forkPoint))
        let branch = chat.forkConversation(at: forkPoint)!
        precondition(chat.branches[branch]?["parent_id"] == parentConversation)
        precondition(chat.serverIdentity(for: branch) == ChatModel.identity(server))
        precondition(chat.conversations[parentConversation]?.count == parentCount && !chat.generating)
        let forkArchive = chat.profileArchive()
        let forkWire = (forkArchive["wire_history"] as? [String: [[String: Any]]])?[branch] ?? []
        precondition(forkWire.last?["role"] as? String == "assistant" && forkWire.last?["tool_calls"] == nil)
        let restoredFork = ChatModel()
        restoredFork.restoreProfileArchive(JSON(try JSONSerialization.jsonObject(with: JSONSerialization.data(withJSONObject: forkArchive))))
        precondition(restoredFork.branches[branch]?["parent_id"] == parentConversation)
        precondition(restoredFork.canFork(at: restoredFork.conversations[branch]!.last!))
        var archivedForkPoint: [String: Any] = [:]
        let childBranch = try await restoredFork.forkConversation(at: restoredFork.conversations[branch]!.last!, archive: { id, document in
            archivedForkPoint = document
            return JSON(["session_id": id, "revision": 7, "digest": "fork-point-digest"])
        })!
        precondition(restoredFork.branches[childBranch]?["parent_revision"] == "7")
        precondition(restoredFork.branches[childBranch]?["parent_id"] == branch)
        precondition((archivedForkPoint["wire_history"] as? [String: [[String: Any]]])?[branch]?.last?["role"] as? String == "assistant")

        chat.draft = "follow up"
        chat.send(server: server, resolve: { target }, systemPrompt: "new base prompt", tools: { context }, execute: execute)
        try await wait(chat)
        precondition(chat.error == nil && chat.conversations[parentConversation]?.count == parentCount)
        chat.clear()
        chat.draft = "tool request"
        chat.send(server: server, resolve: { target }, tools: { context }, execute: execute)
        for _ in 0..<300 {
            if chat.approval != nil { break }
            try await Task.sleep(for: .milliseconds(10))
        }
        precondition(chat.approval != nil)
        chat.stop()
        try await wait(chat)
        precondition(chat.error == "Reply stopped.")
        precondition(chat.approval == nil && approvals == 1)
        precondition(chat.conversations[chat.selected]?.last?.toolEvents.last?.status == "cancelled")
        for exposedName in ["mlx2__engine_unload_self", "mlx2_tool_call"] {
        let terminal = ChatModel()
        terminal.selected = ChatModel.identity(server)
        terminal.draft = "tool request"
        terminal.send(server: server, resolve: { target }, tools: { JSON(["session_id": "synthetic-session", "tools": [["type": "function", "function": ["name": exposedName, "parameters": ["type": "object"]]]]]) }, execute: { _, _, _, _ in
            JSON(["result": ["terminal_action": true, "message": "Synthetic self-unload accepted"]])
        })
        try await wait(terminal)
        precondition(terminal.error == nil)
        precondition(terminal.conversations[terminal.selected]?.last?.text == "Synthetic self-unload accepted")
        precondition(terminal.conversations[terminal.selected]?.allSatisfy(\.committed) == true)
        }
        for (toolName, action, noun) in [("mlx2__agent_result", "wait_for_agent", "subagent"), ("mlx2__code_result", "wait_for_code", "coding job")] {
            let awaitingChild = ChatModel()
            awaitingChild.selected = ChatModel.identity(server)
            awaitingChild.draft = "tool request"
            var pendingReads = 0
            awaitingChild.send(server: server, resolve: { target }, tools: {
                JSON(["session_id": "synthetic-session", "tools": [["type": "function", "function": ["name": toolName, "parameters": ["type": "object"]]]]])
            }, execute: { _, _, _, _ in
                pendingReads += 1
                return JSON(["native_action": action, "result": ["status": "running"]])
            })
            try await wait(awaitingChild)
            precondition(pendingReads == 1 && awaitingChild.error == nil)
            precondition(awaitingChild.conversations[awaitingChild.selected]?.last?.text == "Waiting for \(noun). Its report will resume this conversation automatically.")
            precondition(awaitingChild.conversations[awaitingChild.selected]?.allSatisfy(\.committed) == true)
        }
        var compactArchives: [[String: Any]] = []
        var compactRevision = 0
        let compact = ChatModel()
        let compactID = ChatModel.identity(server)
        let older: [[String: Any]] = (1...3).flatMap { index in
            let detail = index < 3 ? String(repeating: " historical detail", count: 200) : ""
            return [["role": "user", "content": "old user \(index)" + detail], ["role": "assistant", "content": "old answer \(index)" + detail]]
        }
        compact.restoreProfileArchive(JSON(["schema_version": 1, "selected": compactID, "conversations": [compactID: [["role": "assistant", "text": "old visible answer", "committed": true]], "other-private-session": [["role": "user", "text": "OTHER PRIVATE SECRET"]]], "wire_history": [compactID: older]]))
        compact.draft = "compact request"
        compact.send(server: server, resolve: { target }, archive: { id, document in
            compactArchives.append(document); compactRevision += 1
            return JSON(["session_id": id, "revision": compactRevision, "digest": "digest-" + String(compactRevision)])
        }, tools: { JSON(["session_id": "synthetic-session", "tools": [["type": "function", "function": ["name": "mlx2__session_compact", "parameters": ["type": "object"]]]]]) }, execute: { _, _, _, _ in
            JSON(["native_action": "session_compact", "checkpoint": ["session_id": compactID, "input_revision": compactRevision, "input_digest": "digest-" + String(compactRevision), "authority": "model-authored-candidate", "original_retained": true, "summary": ["goal": "Test summary", "constraints": [], "decisions": [], "evidence": [], "open_tasks": [], "next_steps": [], "uncertainties": []]], "result": ["checkpoint_saved": true]])
        })
        try await wait(compact)
        precondition(compact.error == nil, compact.error ?? "")
        precondition(compactArchives.count == 2)
        let compactArchiveData = try JSONSerialization.data(withJSONObject: compactArchives)
        precondition(!String(decoding: compactArchiveData, as: UTF8.self).contains("OTHER PRIVATE SECRET"))
        let compactWire = (compact.profileArchive()["wire_history"] as? [String: [[String: Any]]])?[compactID] ?? []
        precondition(compactWire.first?["role"] as? String == "assistant")
        precondition((compactWire.first?["content"] as? String)?.contains("lossy session summary") == true)
        precondition(!compactWire.contains { $0["content"] as? String == "old user 1" })
        precondition(compactWire.contains { $0["content"] as? String == "old user 3" })
        let exactBeforeCompaction = try JSONSerialization.data(withJSONObject: compactArchives.last!)
        precondition(String(decoding: exactBeforeCompaction, as: UTF8.self).contains("old user 1"))
        let growingSummary = ChatModel()
        let shortOlder: [[String: Any]] = (1...3).flatMap { index in [["role": "user", "content": "short old user \(index)"], ["role": "assistant", "content": "short old answer \(index)"]] }
        growingSummary.restoreProfileArchive(JSON(["schema_version": 1, "selected": compactID, "wire_history": [compactID: shortOlder]]))
        growingSummary.draft = "compact request"
        var growingArchives: [[String: Any]] = []
        growingSummary.send(server: server, resolve: { target }, archive: { id, document in
            growingArchives.append(document)
            return JSON(["session_id": id, "revision": growingArchives.count, "digest": "short-history-" + String(growingArchives.count)])
        }, tools: { JSON(["session_id": "synthetic-session", "tools": [["type": "function", "function": ["name": "mlx2__session_compact", "parameters": ["type": "object"]]]]]) }, execute: { _, _, _, _ in
            JSON(["native_action": "session_compact", "checkpoint": ["session_id": compactID, "input_revision": growingArchives.count, "input_digest": "short-history-" + String(growingArchives.count), "authority": "model-authored-candidate", "original_retained": true, "summary": ["goal": String(repeating: "summary", count: 3000), "constraints": [], "decisions": [], "evidence": [], "open_tasks": [], "next_steps": [], "uncertainties": []]], "result": ["checkpoint_saved": true]])
        })
        try await wait(growingSummary)
        precondition(growingSummary.error?.contains("does not reduce serialized context size") == true)
        precondition(growingArchives.count == 2)
        let growingWire = (growingSummary.profileArchive()["wire_history"] as? [String: [[String: Any]]])?[compactID] ?? []
        let retainedOldBytes = try JSONSerialization.data(withJSONObject: Array(growingWire.prefix(shortOlder.count)), options: [.sortedKeys])
        let originalOldBytes = try JSONSerialization.data(withJSONObject: shortOlder, options: [.sortedKeys])
        precondition(retainedOldBytes == originalOldBytes)
        precondition(!growingWire.contains { ($0["content"] as? String)?.hasPrefix("Model-authored lossy session summary") == true })
        func fakeArchive(_ id: String, _ document: [String: Any]) async throws -> JSON { JSON(["session_id": id, "revision": 1, "digest": "synthetic-exact-archive"]) }
        let agentic = ChatModel()
        agentic.selected = ChatModel.identity(server)
        agentic.draft = "agentic profile"
        agentic.send(server: server, resolve: { JSON(["url": base, "context_limit": 262144, "output_tokens": 65536,
            "chat_options": ["enable_thinking": true, "verification": true, "thinking_budget": 32768]]) },
            tools: nil, execute: nil)
        try await wait(agentic)
        precondition(agentic.error == nil, agentic.error ?? "")
        precondition(agentic.conversations[agentic.selected]?.last?.reasoning == "Checked the evidence.")
        print("Native 256K context, 64K output, 32K reasoning and verification payload checks passed")
        var polls = 0
        let completedBatch = try await SubagentToolWaiter.wait(sessionID: "scope", jobID: "job") { command, arguments in
            precondition(command == "tool_call" && arguments["session_id"] as? String == "scope")
            precondition((arguments["arguments"] as? [String: String])?["job_id"] == "job")
            polls += 1
            return JSON(["result": ["status": "completed", "results": [["answer": "41", "status": "completed", "evidence": [], "unresolved_issues": []]]]])
        }
        precondition(polls == 1 && completedBatch["result"][0]["answer"].text == "41")
        var cancelled = false
        let cancelTask = Task { @MainActor in
            try await SubagentToolWaiter.wait(sessionID: "scope", jobID: "cancel-job") { _, arguments in
                if arguments["name"] as? String == "mlx2__agent_cancel" { cancelled = true }
                return JSON(["result": ["status": "running"]])
            }
        }
        await Task.yield(); cancelTask.cancel()
        do { _ = try await cancelTask.value; fatalError("Cancelled batch returned success") } catch {}
        precondition(cancelled)
        let discovery = ChatModel()
        discovery.selected = ChatModel.identity(server); discovery.draft = "discover tools"
        let searchTool: [String: Any] = ["type": "function", "function": ["name": "mlx2__tool_search", "parameters": ["type": "object"]]]
        discovery.send(server: server, resolve: { target }, tools: { JSON(["session_id": "discovery", "tools": [searchTool]]) }, execute: { _, name, _, _ in
            precondition(name == "mlx2__tool_search")
            let definitions = ["mlx2__tool_search", "spawn_subagent", "mlx2__world_export"].map { ["type": "function", "function": ["name": $0, "parameters": ["type": "object"]]] as [String: Any] }
            return JSON(["native_action": "tool_inventory", "tools": definitions, "result": ["matches": ["mlx2__world_export"]]])
        })
        try await wait(discovery)
        precondition(discovery.error == nil, discovery.error ?? "")
        precondition(discovery.conversations[discovery.selected]?.last?.text == "Discovery wired")
        print("Native canonical subagent result/cancellation and bounded dynamic tool discovery checks passed")
        let budget = ChatModel()
        budget.selected = ChatModel.identity(server)
        let budgetTarget = JSON(["url": base, "context_limit": 900, "output_tokens": 128])
        budget.draft = "budget one"
        budget.send(server: server, resolve: { budgetTarget }, archive: fakeArchive, tools: nil, execute: nil)
        try await wait(budget)
        precondition(budget.error == nil, budget.error ?? "")
        precondition(budget.contextNotice.contains("counted by the engine"))
        let noArchive = ChatModel()
        noArchive.restoreProfileArchive(JSON(budget.profileArchive()))
        let preservedHistory = try JSONSerialization.data(withJSONObject: noArchive.profileArchive()["wire_history"] ?? [:], options: [.sortedKeys])
        noArchive.draft = "budget two"
        noArchive.send(server: server) { budgetTarget }
        try await wait(noArchive)
        precondition(noArchive.error?.contains("archiving is unavailable") == true)
        let afterFailedArchive = try JSONSerialization.data(withJSONObject: noArchive.profileArchive()["wire_history"] ?? [:], options: [.sortedKeys])
        precondition(afterFailedArchive == preservedHistory)

        budget.draft = "budget two"
        budget.send(server: server, resolve: { budgetTarget }, archive: fakeArchive, tools: nil, execute: nil)
        try await wait(budget)
        precondition(budget.error == nil, budget.error ?? "")
        precondition(budget.contextNotice.contains("Removed 1 older completed"))
        precondition(budget.conversations[budget.selected]?.count == 4)
        budget.draft = "An autosaved draft"
        let archive = budget.profileArchive()
        let restored = ChatModel()
        restored.restoreProfileArchive(JSON(archive))
        precondition(restored.selected == budget.selected && restored.draft == budget.draft)
        precondition(restored.conversations[restored.selected]?.map(\.text) == budget.conversations[budget.selected]?.map(\.text))
        let restoredWire = restored.profileArchive()["wire_history"] as? [String: [[String: Any]]]
        let originalWire = archive["wire_history"] as? [String: [[String: Any]]]
        let restoredData = try JSONSerialization.data(withJSONObject: restoredWire ?? [:], options: [.sortedKeys])
        let originalData = try JSONSerialization.data(withJSONObject: originalWire ?? [:], options: [.sortedKeys])
        precondition(restoredData == originalData)
        restored.restoreProfileArchive(JSON([:]))
        precondition(restored.conversations.isEmpty && restored.draft.isEmpty && restored.attachments.isEmpty)
        print("Native streamed tool fragments, visible approval, tool history, prompt replacement, timestamps, image capability, cancellation and exact context pruning checks passed")
    }
}
