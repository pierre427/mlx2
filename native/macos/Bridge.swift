import AppKit
import Darwin
import Foundation

// JSON stays at the desktop boundary. Runtime receipts retain all original fields.
struct JSON: @unchecked Sendable {
    let raw: Any
    init(_ raw: Any = NSNull()) { self.raw = raw }
    subscript(_ key: String) -> JSON { JSON((raw as? [String: Any])?[key] ?? NSNull()) }
    subscript(_ index: Int) -> JSON { array.indices.contains(index) ? array[index] : JSON() }
    var string: String? { raw as? String }
    var double: Double? { (raw as? NSNumber)?.doubleValue }
    var bool: Bool { (raw as? Bool) ?? false }
    var array: [JSON] { (raw as? [Any] ?? []).map(JSON.init) }
    var text: String { string ?? "—" }
}

enum DesktopError: LocalizedError {
    case message(String)
    var errorDescription: String? {
        switch self { case .message(let message): return message }
    }
}

struct DesktopConnectionFailure: LocalizedError {
    let message: String
    var errorDescription: String? { message }
}

final class Bridge: @unchecked Sendable {
    private let queue = DispatchQueue(label: "mlx2.desktop.bridge", qos: .utility)
    private var process: Process?
    private var input: FileHandle?
    private var output: FileHandle?
    private var buffer = Data()
    private var stderr: FileHandle?
    private var transportFailure: String?
    private var finished = false
    private let requestTimeout: TimeInterval
    private let processFactory: (() throws -> Process)?

    init(requestTimeout: TimeInterval = 90, processFactory: (() throws -> Process)? = nil) {
        self.requestTimeout = requestTimeout
        self.processFactory = processFactory
    }

    private func configuredProcess() throws -> Process {
        if let processFactory { return try processFactory() }
        guard let resources = Bundle.main.resourceURL else {
            throw DesktopError.message("Application resources are missing")
        }
        let config = JSON(try JSONSerialization.jsonObject(with: Data(contentsOf: resources.appendingPathComponent("runtime.json"))))
        guard let python = config["python"].string, let source = config["source"].string else {
            throw DesktopError.message("Rebuild MLX2.app: runtime.json is incomplete")
        }
        let home = FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent("Library/Application Support/MLX2")
        try FileManager.default.createDirectory(at: home, withIntermediateDirectories: true)
        let log = home.appendingPathComponent("desktop.log")
        if !FileManager.default.fileExists(atPath: log.path) { FileManager.default.createFile(atPath: log.path, contents: nil) }
        stderr = try FileHandle(forWritingTo: log)
        try stderr?.seekToEnd()
        let task = Process()
        task.executableURL = URL(fileURLWithPath: python)
        task.arguments = ["-u", "-m", "mlx2.desktop", "--trash-helper", Bundle.main.executableURL!.path]
        task.currentDirectoryURL = URL(fileURLWithPath: source)
        var environment = ProcessInfo.processInfo.environment
        environment["PYTHONPATH"] = URL(fileURLWithPath: source).appendingPathComponent("src").path
        environment["PYTHONUNBUFFERED"] = "1"
        task.environment = environment
        return task
    }

    private func launch() throws {
        guard !finished else { throw DesktopError.message("The desktop connection has closed.") }
        if let transportFailure { throw DesktopConnectionFailure(message: transportFailure) }
        if let process, process.isRunning { return }
        if process != nil { throw DesktopConnectionFailure(message: "The desktop supervisor exited. Quit and reopen MLX2; see desktop.log for details.") }
        let task = try configuredProcess()
        let incoming = Pipe(), outgoing = Pipe()
        task.standardInput = incoming
        task.standardOutput = outgoing
        task.standardError = stderr
        try task.run()
        process = task
        input = incoming.fileHandleForWriting
        output = outgoing.fileHandleForReading
        buffer.removeAll()
    }

    func request(_ command: String, _ arguments: [String: Any] = [:]) async throws -> JSON {
        try await withCheckedThrowingContinuation { continuation in
            queue.async {
                var sent = false
                var received = false
                do {
                    try self.launch()
                    let id = UUID().uuidString
                    var request = arguments
                    request["command"] = command
                    request["id"] = id
                    var data = try JSONSerialization.data(withJSONObject: request)
                    data.append(10)
                    sent = true
                    try self.input!.write(contentsOf: data)
                    let deadline = Date().addingTimeInterval(self.requestTimeout)
                    while !self.buffer.contains(10) {
                        var descriptor = pollfd(fd: self.output!.fileDescriptor, events: Int16(POLLIN), revents: 0)
                        let ready = Darwin.poll(&descriptor, 1, 250)
                        if ready < 0 && errno != EINTR { throw DesktopError.message("Desktop pipe polling failed") }
                        if Date() > deadline { throw DesktopError.message("Desktop command timed out. See desktop.log before retrying.") }
                        if ready <= 0 { continue }
                        let chunk = self.output!.availableData
                        if chunk.isEmpty { throw DesktopError.message("Desktop supervisor stopped. See ~/Library/Application Support/MLX2/desktop.log") }
                        self.buffer.append(chunk)
                        let maximumResponse = ["world_export", "user_profile_export", "user_profile_chat_get"].contains(command) ? 64 * 1024 * 1024 : 16 * 1024 * 1024
                        if self.buffer.count > maximumResponse { throw DesktopError.message("Desktop response exceeded its size limit") }
                    }
                    let end = self.buffer.firstIndex(of: 10)!
                    let line = self.buffer.prefix(upTo: end)
                    self.buffer.removeSubrange(...end)
                    let reply = JSON(try JSONSerialization.jsonObject(with: line))
                    guard reply["id"].string == id else { throw DesktopError.message("Desktop response identity mismatch") }
                    received = true
                    guard reply["ok"].bool else { throw DesktopError.message(reply["error"].text) }
                    continuation.resume(returning: reply["result"])
                } catch {
                    if sent && !received {
                        // A late reply cannot safely be paired with the next command.
                        // Keep owned engines alive, but refuse further mutations until restart.
                        self.transportFailure = "Desktop connection lost: \(error.localizedDescription) Quit and reopen MLX2 before retrying."
                    }
                    if let failure = self.transportFailure { continuation.resume(throwing: DesktopConnectionFailure(message: failure)) }
                    else { continuation.resume(throwing: error) }
                }
            }
        }
    }

    func finish() async {
        // Serialized after any pending command. EOF lets Python drain and stop
        // its own worker before releasing the paired reservation.
        await withCheckedContinuation { continuation in
            queue.async {
                self.finished = true
                try? self.input?.close()
                // A timed-out command can still be writing a large reply. Drain it
                // before waiting, so the child cannot block on a full stdout pipe.
                if self.output != nil { _ = try? self.output?.readToEnd() }
                self.process?.waitUntilExit()
                try? self.output?.close()
                try? self.stderr?.close()
                continuation.resume()
            }
        }
    }
}

func bytes(_ value: Double?) -> String {
    guard let value else { return "—" }
    return ByteCountFormatter.string(fromByteCount: Int64(value), countStyle: .binary)
}

func number(_ value: Double?, suffix: String = "", digits: Int = 1) -> String {
    guard let value else { return "—" }
    return String(format: "%.*f", digits, value) + suffix
}

func timestamp(_ value: Double?) -> String {
    guard let value else { return "Never" }
    return Date(timeIntervalSince1970: value).formatted(date: .abbreviated, time: .shortened)
}
