import Foundation

@main enum BridgeChecks {
    static func main() async throws {
        precondition(bytes(.nan) == "—" && bytes(.infinity) == "—" && bytes(Double(Int64.max)) == "—")
        precondition(number(.infinity) == "—" && timestamp(.nan) == "Never")
        let script = CommandLine.arguments[1]
        let executable = CommandLine.arguments[2]
        func bridge(_ mode: String, timeout: Double = 2) -> Bridge {
            Bridge(requestTimeout: timeout, processFactory: {
                let task = Process()
                task.executableURL = URL(fileURLWithPath: executable)
                task.arguments = [script, mode]
                return task
            })
        }
        let healthy = bridge("healthy")
        do { _ = try await healthy.request("invalid"); fatalError("Expected refusal") }
        catch { precondition(error.localizedDescription == "Synthetic refusal") }
        let value = try await healthy.request("valid")
        precondition(value["value"].string == "valid")
        await healthy.finish()
        do { _ = try await healthy.request("after close"); fatalError("Closed bridge restarted") }
        catch { precondition(error.localizedDescription.contains("closed")) }

        // A timeout fails only its own request; the late 1 MiB reply is drained and discarded.
        let delayed = bridge("healthy", timeout: 0.1)
        // Interpreter startup can outlast the short timeout; since timeouts no
        // longer fence the connection, retry until the child answers.
        var warmed = false
        for _ in 0..<100 where !warmed { warmed = (try? await delayed.request("warm")) != nil }
        precondition(warmed, "child never answered")
        do { _ = try await delayed.request("slow"); fatalError("Expected timeout") }
        catch { precondition(!(error is DesktopConnectionFailure) && error.localizedDescription.contains("timed out")) }
        let reply1 = try await delayed.request("after timeout")
        precondition(reply1["value"].string == "after timeout")
        try await Task.sleep(for: .milliseconds(700))
        let reply2 = try await delayed.request("after late reply")
        precondition(reply2["value"].string == "after late reply")
        await delayed.finish()

        // A slow request does not hold up the next one; replies pair by id.
        let parallel = bridge("healthy")
        _ = try await parallel.request("warm")  // exclude interpreter startup from the timing
        let started = Date()
        async let waiting = parallel.request("wait")
        try await Task.sleep(for: .milliseconds(20))
        let reply3 = try await parallel.request("fast")
        precondition(reply3["value"].string == "fast")
        precondition(Date().timeIntervalSince(started) < 0.35, "fast request waited behind the slow one")
        let reply4 = try await waiting
        precondition(reply4["value"].string == "wait")
        // Finishing still delivers replies to requests already in flight.
        async let drained = parallel.request("wait")
        try await Task.sleep(for: .milliseconds(50))
        await parallel.finish()
        let reply5 = try await drained
        precondition(reply5["value"].string == "wait")

        // A companion that stops reading stdin cannot wedge the bridge: large
        // writes block only the writer, and every request still times out.
        let stalled = bridge("stall", timeout: 0.5)
        let stallStart = Date()
        let big = String(repeating: "y", count: 300_000)
        await withTaskGroup(of: Void.self) { group in
            for index in 0..<40 {
                group.addTask { _ = try? await stalled.request("blocked", index == 3 ? ["payload": big] : [:]) }
            }
        }
        precondition(Date().timeIntervalSince(stallStart) < 2.5, "requests did not time out while the companion stalled")
        await stalled.finish()
        await stalled.finish()  // a second finish returns too

        // An unreadable reply fences the connection.
        let garbled = bridge("garbled")
        do { _ = try await garbled.request("any"); fatalError("Expected connection failure") }
        catch { precondition(error is DesktopConnectionFailure) }
        do { _ = try await garbled.request("must not run"); fatalError("Broken bridge accepted work") }
        catch { precondition(error is DesktopConnectionFailure && error.localizedDescription.contains("Quit and reopen")) }
        await garbled.finish()
        print("Bridge checks passed: refusals recover, timeouts fail alone, slow commands do not block, stalls cannot wedge, finish drains and repeats, corrupt wire fences, close is final")
    }
}
