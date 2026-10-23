import CryptoKit
import Foundation

extension DesktopModel {
    @MainActor func archiveChatSession(conversationID: String, document: [String: Any]) async throws -> JSON {
        let data = try JSONSerialization.data(withJSONObject: document, options: [.sortedKeys])
        guard data.count <= 64 * 1024 * 1024 else {
            throw DesktopError.message("This session exceeds the 64 MiB archive transfer limit. Context will not be discarded.")
        }
        var binding: [String: Any] = ["session_id": conversationID]
        if let branches = document["branches"] as? [String: [String: String]], let branch = branches[conversationID],
           let parent = branch["parent_id"], let text = branch["parent_revision"], let revision = Int(text), revision > 0 {
            binding["parent_id"] = parent
            binding["parent_revision"] = revision
        }
        if data.count <= 8 * 1024 * 1024 {
            return try await request("session_archive", binding.merging(["document": document]) { _, new in new })
        }
        let hash = SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
        let start = try await request("session_archive_begin", binding.merging(["total_bytes": data.count, "sha256": hash]) { _, new in new })
        let upload = start["upload_id"].text
        do {
            for (index, offset) in stride(from: 0, to: data.count, by: 1 << 20).enumerated() {
                try Task.checkCancellation()
                let chunk = data.subdata(in: offset..<min(data.count, offset + (1 << 20)))
                _ = try await request("session_archive_chunk", ["session_id": conversationID, "upload_id": upload,
                    "index": index, "data_base64": chunk.base64EncodedString()])
            }
            return try await request("session_archive_commit", ["session_id": conversationID, "upload_id": upload])
        } catch {
            _ = try? await request("session_archive_abort", ["session_id": conversationID, "upload_id": upload])
            throw error
        }
    }
}
