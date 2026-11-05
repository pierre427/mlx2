import AppKit
import Foundation
import ImageIO
import PDFKit
import UniformTypeIdentifiers

struct ChatToolApproval { let name: String; let arguments: String }
struct ChatToolDelta {
    var id = "", name = "", arguments = ""
    var wire: [String: Any] { ["id": id, "type": "function", "function": ["name": name, "arguments": arguments]] }
}

struct ChatAttachment: Identifiable {
    let id = UUID()
    let name: String
    let modality: String
    let part: [String: Any]

    private static func imageContent(_ data: Data) throws -> (Data, String) {
        guard let source = CGImageSourceCreateWithData(data as CFData, [kCGImageSourceShouldCache: false] as CFDictionary),
              let identifier = CGImageSourceGetType(source), let type = UTType(identifier as String),
              let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any],
              let width = properties[kCGImagePropertyPixelWidth] as? NSNumber,
              let height = properties[kCGImagePropertyPixelHeight] as? NSNumber,
              width.doubleValue > 0, height.doubleValue > 0,
              width.doubleValue * height.doubleValue <= 40_000_000 else {
            throw DesktopError.message("Use a valid image with at most 40 million pixels.")
        }
        // The serving image decoder accepts PNG, JPEG and WebP. Native photos
        // such as HEIC/TIFF need conversion before they reach that contract.
        if [UTType.png.identifier, UTType.jpeg.identifier, "org.webmproject.webp"].contains(type.identifier) {
            return (data, type.preferredMIMEType ?? "image/webp")
        }
        guard CGImageSourceGetCount(source) == 1 else {
            throw DesktopError.message("Attach one image frame or a supported video. Multi-frame images cannot be converted without discarding frames.")
        }
        guard let image = CGImageSourceCreateImageAtIndex(source, 0, [kCGImageSourceShouldCache: false] as CFDictionary) else {
            throw DesktopError.message("This image could not be converted to PNG.")
        }
        let converted = NSMutableData()
        guard let destination = CGImageDestinationCreateWithData(converted, UTType.png.identifier as CFString, 1, nil) else {
            throw DesktopError.message("This image could not be converted to PNG.")
        }
        CGImageDestinationAddImage(destination, image, properties as CFDictionary)
        guard CGImageDestinationFinalize(destination), converted.length <= 32 * 1024 * 1024 else {
            throw DesktopError.message("The converted PNG exceeds 32 MiB. Resize or export this image as JPEG before attaching it.")
        }
        return (converted as Data, "image/png")
    }

    static func read(_ url: URL) throws -> ChatAttachment {
        let size = try url.resourceValues(forKeys: [.fileSizeKey, .isRegularFileKey])
        guard size.isRegularFile == true, (size.fileSize ?? Int.max) <= 32 * 1024 * 1024 else {
            throw DesktopError.message("Attachments must be regular files of at most 32 MiB.")
        }
        let data = try Data(contentsOf: url, options: .mappedIfSafe)
        let ext = url.pathExtension.lowercased()
        let type = UTType(filenameExtension: ext)
        let name = url.lastPathComponent
        if ext == "pdf" {
            guard let pdf = PDFDocument(data: data), let text = pdf.string, !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
                throw DesktopError.message("This PDF has no extractable text. Attach page images to a vision model.")
            }
            guard text.utf8.count <= 1_000_000 else { throw DesktopError.message("Extracted document text exceeds 1 MiB.") }
            return .init(name: name, modality: "text", part: ["type": "text", "text": "Attached document: \(name)\n\(text)"])
        }
        if type?.conforms(to: .image) == true {
            let (payload, mime) = try imageContent(data)
            return .init(name: name, modality: "image", part: ["type": "image_url", "image_url": ["url": "data:\(mime);base64," + payload.base64EncodedString()]])
        }
        if type?.conforms(to: .audio) == true {
            guard ["wav", "mp3", "flac", "m4a", "ogg", "aiff", "aac"].contains(ext) else { throw DesktopError.message("Unsupported audio container.") }
            return .init(name: name, modality: "audio", part: ["type": "input_audio", "input_audio": ["data": data.base64EncodedString(), "format": ext]])
        }
        if type?.conforms(to: .movie) == true || type?.conforms(to: .video) == true {
            let mime = type?.preferredMIMEType ?? "video/mp4"
            return .init(name: name, modality: "video", part: ["type": "input_video", "video_url": "data:\(mime);base64," + data.base64EncodedString()])
        }
        if ext == "docx" {
            // NSAttributedString uses the native document importer, without shelling out or executing macros.
            let document = try NSAttributedString(url: url, options: [.documentType: NSAttributedString.DocumentType.officeOpenXML], documentAttributes: nil)
            guard document.string.utf8.count <= 1_000_000 else { throw DesktopError.message("Extracted document text exceeds 1 MiB.") }
            return .init(name: name, modality: "text", part: ["type": "text", "text": "Attached document: \(name)\n\(document.string)"])
        }
        guard data.count <= 1_000_000, let text = String(data: data, encoding: .utf8), !text.contains("\0") else {
            throw DesktopError.message("Use UTF-8 text, PDF, DOCX, or a supported image, audio or video file.")
        }
        return .init(name: name, modality: "text", part: ["type": "text", "text": "Attached document: \(name)\n\(text)"])
    }

    static func content(prompt: String, attachments: [ChatAttachment], target: JSON) throws -> Any {
        if attachments.isEmpty { return prompt }
        let modalities = Set(target["input_modalities"].array.compactMap(\.string) + ["text"])
        for item in attachments where !modalities.contains(item.modality) {
            throw DesktopError.message("This running route does not advertise \(item.modality) input. Choose a compatible model before attaching \(item.name).")
        }
        let parts: [[String: Any]] = [["type": "text", "text": prompt]] + attachments.map(\.part)
        let size = try JSONSerialization.data(withJSONObject: parts).count
        guard size <= 48 * 1024 * 1024 else { throw DesktopError.message("Combined attachments exceed the 48 MiB request limit.") }
        return parts
    }
}
