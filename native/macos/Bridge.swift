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

/// Multiplexed JSON-lines connection to the Python companion.
///
/// Requests are written in call order and each reply is paired by its id, so a
/// slow command (for example a Wikimedia search) no longer holds up the ones
/// after it. The companion still runs ordinary commands one at a time in arrival
/// order; only work that steps out of its request lane runs in parallel.
final class Bridge: @unchecked Sendable {
    private struct Pending {
        let command: String
        let continuation: CheckedContinuation<JSON, Error>
    }
    // Guards every field below; the reader thread hands lines over through it.
    private let queue = DispatchQueue(label: "mlx2.desktop.bridge", qos: .utility)
    // Writes can block when the companion is not reading; they never hold up
    // timeouts or reply delivery on `queue`. Serial, so request order is kept.
    private let writer = DispatchQueue(label: "mlx2.desktop.bridge.writer", qos: .utility)
    private var finishing: [CheckedContinuation<Void, Never>] = []
    private var closed = false
    private var process: Process?
    private var input: FileHandle?
    private var output: FileHandle?
    private var stderr: FileHandle?
    private var transportFailure: String?
    private var finished = false
    private var pending: [String: Pending] = [:]
    private var readerFinished: DispatchSemaphore?
    private let requestTimeout: TimeInterval
    private let processFactory: (() throws -> Process)?
    private static let lineLimit = 64 * 1024 * 1024

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
        let home = config["desktop_home"].string.map { URL(fileURLWithPath: $0) }
            ?? FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent("Library/Application Support/MLX2")
        try FileManager.default.createDirectory(at: home, withIntermediateDirectories: true)
        let log = home.appendingPathComponent("desktop.log")
        if !FileManager.default.fileExists(atPath: log.path) { FileManager.default.createFile(atPath: log.path, contents: nil) }
        stderr = try FileHandle(forWritingTo: log)
        try stderr?.seekToEnd()
        let task = Process()
        task.executableURL = URL(fileURLWithPath: python)
        task.arguments = ["-u", "-m", "mlx2.desktop", "--trash-helper", Bundle.main.executableURL!.path]
        if config["desktop_home"].string != nil { task.arguments! += ["--home", home.path] }
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
        startReader(outgoing.fileHandleForReading)
    }

    private func startReader(_ handle: FileHandle) {
        let done = DispatchSemaphore(value: 0)
        readerFinished = done
        let thread = Thread { [self] in
            var buffer = Data()
            var discarding = false
            defer { done.signal() }
            while true {
                let chunk = handle.availableData
                if chunk.isEmpty {
                    queue.async { self.connectionEnded() }
                    return
                }
                buffer.append(chunk)
                while let end = buffer.firstIndex(of: 10) {
                    let line = Data(buffer[buffer.startIndex..<end])
                    buffer.removeSubrange(buffer.startIndex...end)
                    if discarding { discarding = false; continue }
                    queue.async { self.deliver(line) }
                }
                if buffer.count > Self.lineLimit {
                    // Keep draining so the companion never blocks on a full pipe.
                    buffer.removeAll(keepingCapacity: false)
                    if !discarding { queue.async { self.failAll("Desktop response exceeded its size limit. Quit and reopen MLX2 before retrying.") } }
                    discarding = true
                }
            }
        }
        thread.name = "mlx2.desktop.bridge.reader"
        thread.start()
    }

    private func deliver(_ line: Data) {
        guard let object = try? JSONSerialization.jsonObject(with: line) else {
            failAll("Desktop connection lost: the companion wrote an unreadable reply. Quit and reopen MLX2 before retrying.")
            return
        }
        let reply = JSON(object)
        // A reply whose request already timed out is discarded; ids never repeat.
        guard let id = reply["id"].string, let entry = pending.removeValue(forKey: id) else { return }
        let limit = ["world_export", "user_profile_export", "user_profile_chat_get"].contains(entry.command) ? 64 * 1024 * 1024 : 16 * 1024 * 1024
        if line.count > limit {
            entry.continuation.resume(throwing: DesktopError.message("Desktop response exceeded its size limit"))
        } else if reply["ok"].bool {
            entry.continuation.resume(returning: reply["result"])
        } else {
            entry.continuation.resume(throwing: DesktopError.message(reply["error"].text))
        }
    }

    private func failAll(_ message: String) {
        if transportFailure == nil { transportFailure = message }
        let waiting = pending.values
        pending.removeAll()
        for entry in waiting { entry.continuation.resume(throwing: DesktopConnectionFailure(message: transportFailure!)) }
    }

    private func connectionEnded() {
        if finished {
            let waiting = pending.values
            pending.removeAll()
            for entry in waiting { entry.continuation.resume(throwing: DesktopError.message("The desktop connection has closed.")) }
        } else {
            failAll("Desktop supervisor stopped. Quit and reopen MLX2; see ~/Library/Application Support/MLX2/desktop.log")
        }
    }

    func request(_ command: String, _ arguments: [String: Any] = [:]) async throws -> JSON {
        try await withCheckedThrowingContinuation { continuation in
            queue.async {
                do {
                    try self.launch()
                    let id = UUID().uuidString
                    var request = arguments
                    request["command"] = command
                    request["id"] = id
                    // The timer below starts now, while the companion may still be
                    // running earlier commands one at a time. It refuses, unrun, a
                    // request that reaches its lane after this instant, so a command
                    // reported as timed out never applies later beside a retry.
                    request["bridge_deadline"] = Date().timeIntervalSince1970 + self.requestTimeout
                    var data = try JSONSerialization.data(withJSONObject: request)
                    data.append(10)
                    self.pending[id] = Pending(command: command, continuation: continuation)
                    let input = self.input!
                    self.writer.async {
                        // A request that timed out before it could be written is
                        // dropped, so a stalled companion never runs a stale backlog.
                        // (`queue` never waits on `writer`, so this cannot deadlock.)
                        guard self.queue.sync(execute: { self.pending[id] != nil }) else { return }
                        do { try input.write(contentsOf: data) }
                        catch {
                            // A partial write leaves the wire unusable for everyone.
                            self.queue.async { self.failAll("Desktop connection lost: \(error.localizedDescription) Quit and reopen MLX2 before retrying.") }
                        }
                    }
                    self.queue.asyncAfter(deadline: .now() + self.requestTimeout) {
                        guard let entry = self.pending.removeValue(forKey: id) else { return }
                        entry.continuation.resume(throwing: DesktopError.message(
                            "Desktop command timed out; if it had already started it may still complete. See desktop.log before retrying."))
                    }
                } catch {
                    if let failure = self.transportFailure { continuation.resume(throwing: DesktopConnectionFailure(message: failure)) }
                    else { continuation.resume(throwing: error) }
                }
            }
        }
    }

    func finish() async {
        // EOF lets Python drain its in-flight commands, stop its worker and
        // release the paired reservation; their replies are still delivered.
        await withCheckedContinuation { continuation in
            queue.async {
                // Every caller returns once the companion has exited.
                if self.closed { continuation.resume(); return }
                self.finishing.append(continuation)
                guard !self.finished else { return }
                self.finished = true
                let input = self.input, reader = self.readerFinished, process = self.process, output = self.output, log = self.stderr
                // Close after every queued write, so in-flight requests still arrive.
                self.writer.async {
                    try? input?.close()
                    DispatchQueue.global(qos: .utility).async {
                        reader?.wait()
                        process?.waitUntilExit()
                        self.queue.async {
                            try? output?.close()
                            try? log?.close()
                            self.closed = true
                            let waiting = self.finishing
                            self.finishing.removeAll()
                            waiting.forEach { $0.resume() }
                        }
                    }
                }
            }
        }
    }
}

func bytes(_ value: Double?) -> String {
    guard let value, value.isFinite, value >= Double(Int64.min), value < Double(Int64.max) else { return "—" }
    return ByteCountFormatter.string(fromByteCount: Int64(value), countStyle: .binary)
}

func number(_ value: Double?, suffix: String = "", digits: Int = 1) -> String {
    guard let value, value.isFinite else { return "—" }
    return String(format: "%.*f", digits, value) + suffix
}

func timestamp(_ value: Double?) -> String {
    guard let value, value.isFinite, abs(value) < 253402300800 else { return "Never" }
    return Date(timeIntervalSince1970: value).formatted(date: .abbreviated, time: .shortened)
}
