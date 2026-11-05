import Foundation

// SSE framing uses wire line endings, never Unicode paragraph separators in JSON.
struct ServerSentEvents<Source: AsyncSequence>: AsyncSequence where Source.Element == UInt8 {
    typealias Element = String
    let source: Source
    init(_ source: Source) { self.source = source }
    func makeAsyncIterator() -> Iterator { Iterator(source: source.makeAsyncIterator()) }

    struct Iterator: AsyncIteratorProtocol {
        var source: Source.AsyncIterator
        private var line: [UInt8] = []
        private var fields: [String] = []
        private var ended = false
        private var skipLF = false
        private var eventBytes = 0
        init(source: Source.AsyncIterator) { self.source = source }

        private mutating func finishLine() throws -> String? {
            defer { line.removeAll(keepingCapacity: true) }
            guard let text = String(bytes: line, encoding: .utf8) else {
                throw DesktopError.message("Invalid UTF-8 in chat stream")
            }
            if text.isEmpty {
                eventBytes = 0
                guard !fields.isEmpty else { return nil }
                defer { fields.removeAll(keepingCapacity: true) }
                return fields.joined(separator: "\n")
            }
            if text == "data" { fields.append("") }
            else if text.hasPrefix("data:") {
                var value = String(text.dropFirst(5))
                if value.first == " " { value.removeFirst() }
                fields.append(value)
            }
            return nil
        }

        mutating func next() async throws -> String? {
            guard !ended else { return nil }
            while let byte = try await source.next() {
                try Task.checkCancellation()
                if skipLF {
                    skipLF = false
                    if byte == 10 { continue }
                }
                eventBytes += 1
                guard eventBytes <= 2 * 1024 * 1024 else { throw DesktopError.message("Chat stream event exceeds the 2 MiB limit.") }
                if byte == 10 || byte == 13 {
                    skipLF = byte == 13
                    if let payload = try finishLine() { return payload }
                } else { line.append(byte) }
            }
            ended = true
            // An event requires a blank-line terminator; an incomplete event is dropped.
            return nil
        }
    }
}
