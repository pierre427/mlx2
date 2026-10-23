import Foundation

@main enum BridgeChecks {
    static func main() async throws {
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

        let delayed = bridge("delayed", timeout: 0.02)
        do { _ = try await delayed.request("slow"); fatalError("Expected timeout") }
        catch { precondition(error is DesktopConnectionFailure && error.localizedDescription.contains("timed out")) }
        do { _ = try await delayed.request("must not run"); fatalError("Broken bridge accepted work") }
        catch { precondition(error is DesktopConnectionFailure && error.localizedDescription.contains("Quit and reopen")) }
        await delayed.finish()
        print("Bridge checks passed: refusals recover, timeout fences later commands, large late replies drain, close is final")
    }
}
