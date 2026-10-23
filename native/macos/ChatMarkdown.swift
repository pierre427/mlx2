import AppKit
import SwiftUI

// Native rendering: Markdown never executes HTML; media uses bounded native loaders.
struct ChatMarkdownBlock {
    var kind: String
    var text: String
    var level = 0
    var marker = ""
    var rows: [[String]] = []
}

enum ChatMarkdown {
    static func inline(_ text: String) -> AttributedString {
        (try? AttributedString(markdown: text, options: .init(interpretedSyntax: .inlineOnlyPreservingWhitespace))) ?? AttributedString(text)
    }
    static func mediaBlocks(_ line: String) -> [ChatMarkdownBlock]? {
        var blocks: [ChatMarkdownBlock] = []
        var cursor = line.startIndex
        var scan = cursor
        var found = false
        while scan < line.endIndex, let opening = line[scan...].firstIndex(of: "[") {
            let image = opening > cursor && line[line.index(before: opening)] == "!"
            let start = image ? line.index(before: opening) : opening
            guard let close = line[opening...].range(of: "](") else { break }
            var index = close.upperBound
            let destinationStart = index
            var depth = 1
            var escaped = false
            while index < line.endIndex {
                let character = line[index]
                if escaped { escaped = false }
                else if character == "\\" { escaped = true }
                else if character == "(" { depth += 1 }
                else if character == ")" { depth -= 1; if depth == 0 { break } }
                index = line.index(after: index)
            }
            guard depth == 0 else { break }
            let raw = String(line[destinationStart..<index]).trimmingCharacters(in: .whitespaces)
            let source: String
            if raw.hasPrefix("<"), let end = raw.firstIndex(of: ">") { source = String(raw[raw.index(after: raw.startIndex)..<end]) }
            else { source = String(raw.prefix(while: { !$0.isWhitespace })) }
            let kind = image ? "image" : ChatMediaSource.kind(source)
            guard let kind else { scan = line.index(after: index); continue }
            // Prefix is measured from the last emitted segment, so ordinary
            // links surrounding a media link retain their original text.
            let prefix = String(line[cursor..<start])
            if !prefix.trimmingCharacters(in: .whitespaces).isEmpty { blocks.append(ChatMarkdownBlock(kind: "text", text: prefix)) }
            let title = String(line[line.index(after: opening)..<close.lowerBound])
            blocks.append(ChatMarkdownBlock(kind: kind, text: title, marker: source))
            cursor = line.index(after: index)
            scan = cursor
            found = true
        }
        guard found else {
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            if let kind = ChatMediaSource.kind(trimmed), ChatMediaSource.url(trimmed) != nil {
                return [ChatMarkdownBlock(kind: kind, text: "", marker: trimmed)]
            }
            return nil
        }
        let suffix = String(line[cursor...])
        if !suffix.trimmingCharacters(in: .whitespaces).isEmpty { blocks.append(ChatMarkdownBlock(kind: "text", text: suffix)) }
        return blocks
    }

    static func blocks(_ source: String) -> [ChatMarkdownBlock] {
        let lines = source.components(separatedBy: .newlines)
        var result: [ChatMarkdownBlock] = []
        var index = 0
        while index < lines.count {
            let line = lines[index]
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            index += 1
            if trimmed.isEmpty { continue }
            if trimmed.hasPrefix("```") || trimmed.hasPrefix("~~~") {
                let fence = String(trimmed.prefix(3))
                let language = String(trimmed.dropFirst(3)).trimmingCharacters(in: .whitespaces)
                var code: [String] = []
                while index < lines.count && !lines[index].trimmingCharacters(in: .whitespaces).hasPrefix(fence) {
                    code.append(lines[index]); index += 1
                }
                if index < lines.count { index += 1 }
                result.append(ChatMarkdownBlock(kind: "code", text: code.joined(separator: "\n"), marker: language))
                continue
            }
            if let media = mediaBlocks(line) { result.append(contentsOf: media); continue }
            let hashes = trimmed.prefix(while: { $0 == "#" }).count
            if (1...6).contains(hashes), trimmed.dropFirst(hashes).first == " " {
                result.append(ChatMarkdownBlock(kind: "heading", text: String(trimmed.dropFirst(hashes + 1)), level: hashes)); continue
            }
            if ["---", "***", "___"].contains(trimmed) {
                result.append(ChatMarkdownBlock(kind: "rule", text: "")); continue
            }
            if trimmed.hasPrefix(">") {
                result.append(ChatMarkdownBlock(kind: "quote", text: String(trimmed.dropFirst()).trimmingCharacters(in: .whitespaces))); continue
            }
            if trimmed.hasPrefix("|"), index < lines.count,
               lines[index].contains("|"), lines[index].contains("---") {
                func cells(_ row: String) -> [String] {
                    var parts = row.split(separator: "|", omittingEmptySubsequences: false).map { $0.trimmingCharacters(in: .whitespaces) }
                    if parts.first == "" { parts.removeFirst() }; if parts.last == "" { parts.removeLast() }
                    return parts
                }
                var rows = [cells(trimmed)]
                index += 1
                while index < lines.count && lines[index].trimmingCharacters(in: .whitespaces).hasPrefix("|") {
                    rows.append(cells(lines[index])); index += 1
                }
                result.append(ChatMarkdownBlock(kind: "table", text: "", rows: rows)); continue
            }
            let indent = min(6, (line.count - line.drop(while: { $0 == " " }).count) / 2)
            if ["- ", "* ", "+ "].contains(String(trimmed.prefix(2))) {
                result.append(ChatMarkdownBlock(kind: "list", text: String(trimmed.dropFirst(2)), level: indent, marker: "•")); continue
            }
            let digits = trimmed.prefix(while: { $0.isNumber })
            let suffix = trimmed.dropFirst(digits.count)
            if !digits.isEmpty && (suffix.hasPrefix(". ") || suffix.hasPrefix(") ")) {
                result.append(ChatMarkdownBlock(kind: "list", text: String(suffix.dropFirst(2)), level: indent, marker: String(digits) + ".")); continue
            }
            result.append(ChatMarkdownBlock(kind: "text", text: line))
        }
        return result
    }
}

struct ChatMarkdownView: View {
    let text: String
    var body: some View {
        VStack(alignment: .leading, spacing: 7) {
            ForEach(Array(ChatMarkdown.blocks(text).enumerated()), id: \.offset) { _, block in
                switch block.kind {
                case "image", "audio", "video":
                    ChatMediaView(source: block.marker, title: block.text, kind: block.kind)
                case "heading":
                    Text(ChatMarkdown.inline(block.text))
                        .font(.system(size: block.level == 1 ? 23 : block.level == 2 ? 19 : 15, weight: .semibold))
                        .padding(.top, 5)
                case "code":
                    VStack(alignment: .leading, spacing: 6) {
                        HStack {
                            Text(block.marker.isEmpty ? "Code" : block.marker).foregroundStyle(.secondary)
                            Spacer()
                            Button("Copy", systemImage: "doc.on.doc") {
                                NSPasteboard.general.clearContents()
                                NSPasteboard.general.setString(block.text, forType: .string)
                            }.buttonStyle(.plain)
                        }.font(.caption)
                        ScrollView(.horizontal) { Text(verbatim: block.text).font(.system(.callout, design: .monospaced)) }
                    }.padding(10).background(Color.black.opacity(0.16), in: RoundedRectangle(cornerRadius: 8))
                case "list":
                    HStack(alignment: .firstTextBaseline, spacing: 7) {
                        Text(block.marker).foregroundStyle(.secondary)
                        Text(ChatMarkdown.inline(block.text))
                    }.padding(.leading, CGFloat(block.level * 14))
                case "quote":
                    HStack(alignment: .top, spacing: 9) {
                        RoundedRectangle(cornerRadius: 2).fill(Color.secondary.opacity(0.4)).frame(width: 3)
                        Text(ChatMarkdown.inline(block.text)).foregroundStyle(.secondary)
                    }.fixedSize(horizontal: false, vertical: true)
                case "rule": Divider()
                case "table":
                    ScrollView(.horizontal) {
                        Grid(alignment: .topLeading, horizontalSpacing: 16, verticalSpacing: 7) {
                            ForEach(Array(block.rows.enumerated()), id: \.offset) { row, cells in
                                GridRow {
                                    ForEach(Array(cells.enumerated()), id: \.offset) { _, cell in
                                        Text(ChatMarkdown.inline(cell)).fontWeight(row == 0 ? .semibold : .regular)
                                    }
                                }
                                if row == 0 { Divider().gridCellColumns(cells.count) }
                            }
                        }.padding(8)
                    }.background(Color.secondary.opacity(0.05), in: RoundedRectangle(cornerRadius: 6))
                default: Text(ChatMarkdown.inline(block.text))
                }
            }
        }.textSelection(.enabled).fixedSize(horizontal: false, vertical: true)
    }
}
