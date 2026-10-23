import AppKit
import AVKit
import ImageIO
import SwiftUI
import UniformTypeIdentifiers

/// Model-provided media is data, never executable HTML or an authenticated API request.
enum ChatMediaSource {
    static let maximumBytes = 16 * 1024 * 1024
    static let maximumPlaybackBytes = 128 * 1024 * 1024

    static func url(_ source: String) -> URL? { validatedURL(source) }
    static func kind(_ source: String) -> String? { kind(for: source) }

    static func validatedURL(_ source: String) -> URL? {
        let value = source.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !value.isEmpty, !value.contains("\n"), !value.contains("\r") else { return nil }
        if value.hasPrefix("/") { return URL(fileURLWithPath: value).standardizedFileURL }
        guard let url = URL(string: value), let components = URLComponents(url: url, resolvingAgainstBaseURL: false),
              components.user == nil, components.password == nil else { return nil }
        let scheme = components.scheme?.lowercased() ?? ""
        let host = components.host?.lowercased() ?? ""
        if scheme == "file" {
            guard host.isEmpty || host == "localhost", url.path.hasPrefix("/") else { return nil }
            return url.standardizedFileURL
        }
        guard !host.isEmpty else { return nil }
        if scheme == "https" { return url }
        if scheme == "http", ["localhost", "127.0.0.1", "::1", "[::1]"].contains(host) { return url }
        return nil
    }

    static func kind(for source: String) -> String? {
        guard let url = validatedURL(source) else { return nil }
        switch url.pathExtension.lowercased() {
        case "png", "jpg", "jpeg", "gif", "webp", "heic", "heif", "tif", "tiff", "avif": return "image"
        case "mp4", "mov", "m4v", "webm": return "video"
        case "mp3", "m4a", "wav", "flac", "aac", "aiff", "ogg", "opus": return "audio"
        default: return nil
        }
    }
}

private enum ChatMediaError: LocalizedError {
    case unsupported, tooLarge(Int), invalidImage, response(Int)
    var errorDescription: String? {
        switch self {
        case .unsupported: return "This media source is not supported."
        case .tooLarge(let limit): return "Media exceeds the \(limit / (1024 * 1024)) MB preview/download limit. Open the source to view it."
        case .invalidImage: return "This image cannot be previewed safely."
        case .response(let code): return "Media could not be loaded (HTTP \(code))."
        }
    }
}

private final class ChatMediaNetworkGuard: NSObject, URLSessionTaskDelegate, @unchecked Sendable {
    func urlSession(_ session: URLSession, task: URLSessionTask, willPerformHTTPRedirection response: HTTPURLResponse,
                    newRequest request: URLRequest, completionHandler: @escaping (URLRequest?) -> Void) {
        guard let target = request.url, ChatMediaSource.validatedURL(target.absoluteString) != nil,
              !target.isFileURL,
              !(response.url?.scheme == "https" && target.scheme != "https") else {
            completionHandler(nil); return
        }
        var clean = URLRequest(url: target)
        clean.httpShouldHandleCookies = false
        clean.setValue("MLX2 native media preview", forHTTPHeaderField: "User-Agent")
        completionHandler(clean)
    }
    func urlSession(_ session: URLSession, task: URLSessionTask, didReceive challenge: URLAuthenticationChallenge,
                    completionHandler: @escaping (URLSession.AuthChallengeDisposition, URLCredential?) -> Void) {
        if challenge.protectionSpace.authenticationMethod == NSURLAuthenticationMethodServerTrust {
            completionHandler(.performDefaultHandling, nil)
        } else {
            completionHandler(.cancelAuthenticationChallenge, nil)
        }
    }
}

struct ChatMediaResource {
    let data: Data
    let filename: String
}

enum ChatMediaLoader {
    static func data(from url: URL, maximumBytes: Int = ChatMediaSource.maximumBytes) async throws -> Data {
        try await resource(from: url, maximumBytes: maximumBytes).data
    }
    static func resource(from url: URL, maximumBytes: Int) async throws -> ChatMediaResource {
        guard ChatMediaSource.validatedURL(url.absoluteString) != nil else { throw ChatMediaError.unsupported }
        guard maximumBytes > 0 else { throw ChatMediaError.tooLarge(maximumBytes) }
        if url.isFileURL {
            let values = try url.resourceValues(forKeys: [.fileSizeKey, .isRegularFileKey])
            guard values.isRegularFile == true else { throw ChatMediaError.unsupported }
            guard let size = values.fileSize, size <= maximumBytes else { throw ChatMediaError.tooLarge(maximumBytes) }
            let data = try Data(contentsOf: url, options: .mappedIfSafe)
            guard data.count <= maximumBytes else { throw ChatMediaError.tooLarge(maximumBytes) }
            return ChatMediaResource(data: data, filename: filename(url.lastPathComponent, data: data, mimeType: nil))
        }
        let configuration = URLSessionConfiguration.ephemeral
        configuration.httpCookieStorage = nil
        configuration.httpShouldSetCookies = false
        configuration.urlCredentialStorage = nil
        configuration.urlCache = nil
        configuration.timeoutIntervalForRequest = 30
        configuration.timeoutIntervalForResource = 60
        let session = URLSession(configuration: configuration, delegate: ChatMediaNetworkGuard(), delegateQueue: nil)
        defer { session.invalidateAndCancel() }
        var request = URLRequest(url: url)
        request.httpShouldHandleCookies = false
        request.setValue("MLX2 native media preview", forHTTPHeaderField: "User-Agent")
        let (bytes, response) = try await session.bytes(for: request)
        guard let http = response as? HTTPURLResponse else { throw ChatMediaError.unsupported }
        guard (200...299).contains(http.statusCode) else { throw ChatMediaError.response(http.statusCode) }
        guard response.expectedContentLength <= Int64(maximumBytes) else { throw ChatMediaError.tooLarge(maximumBytes) }
        var data = Data()
        for try await byte in bytes {
            try Task.checkCancellation()
            guard data.count < maximumBytes else { throw ChatMediaError.tooLarge(maximumBytes) }
            data.append(byte)
        }
        return ChatMediaResource(data: data, filename: filename(response.suggestedFilename ?? url.lastPathComponent, data: data, mimeType: response.mimeType))
    }

    static func filename(_ suggestion: String, data: Data, mimeType: String?) -> String {
        let name = URL(fileURLWithPath: suggestion).lastPathComponent
        let stem = name.isEmpty || name == "." ? "media" : name
        let detectedType: UTType?
        if let source = CGImageSourceCreateWithData(data as CFData, [kCGImageSourceShouldCache: false] as CFDictionary),
           let identifier = CGImageSourceGetType(source) {
            detectedType = UTType(identifier as String)
        } else {
            detectedType = mimeType.flatMap { UTType(mimeType: $0) }
        }
        if let ext = detectedType?.preferredFilenameExtension {
            let currentType = UTType(filenameExtension: URL(fileURLWithPath: stem).pathExtension)
            if currentType != detectedType {
                return URL(fileURLWithPath: stem).deletingPathExtension().lastPathComponent + "." + ext
            }
        }
        return stem
    }

    static func image(from data: Data) throws -> NSImage {
        guard let source = CGImageSourceCreateWithData(data as CFData, [kCGImageSourceShouldCache: false] as CFDictionary),
              let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any],
              let width = properties[kCGImagePropertyPixelWidth] as? NSNumber,
              let height = properties[kCGImagePropertyPixelHeight] as? NSNumber,
              width.doubleValue > 0, height.doubleValue > 0,
              width.doubleValue * height.doubleValue <= 100_000_000,
              let thumbnail = CGImageSourceCreateThumbnailAtIndex(source, 0, [
                kCGImageSourceCreateThumbnailFromImageAlways: true,
                kCGImageSourceCreateThumbnailWithTransform: true,
                kCGImageSourceThumbnailMaxPixelSize: 1600,
                kCGImageSourceShouldCacheImmediately: true
              ] as CFDictionary) else { throw ChatMediaError.invalidImage }
        return NSImage(cgImage: thumbnail, size: .zero)
    }
}

struct ChatMediaView: View {
    @EnvironmentObject private var model: DesktopModel
    let source: String
    let title: String
    let kind: String
    @State private var image: NSImage?
    @State private var player: AVPlayer?
    @State private var temporaryURL: URL?
    @State private var loading = false
    @State private var error: String?
    @State private var playbackTask: Task<Void, Never>?
    @State private var original: ChatMediaResource?
    @State private var hovering = false
    @State private var saving = false
    @State private var saveTask: Task<Void, Never>?

    var body: some View {
        VStack(alignment: .leading, spacing: 5) {
            if kind == "image", let image {
                Image(nsImage: image).resizable().scaledToFit()
                    .frame(width: previewSize(image).width, height: previewSize(image).height)
                    .clipShape(RoundedRectangle(cornerRadius: 8))
                    .accessibilityLabel(title.isEmpty ? "Chat image" : title)
                    .contentShape(Rectangle())
                    .onTapGesture {
                        expand(image)
                    }
                    .onHover { inside in (inside ? NSCursor.pointingHand : NSCursor.arrow).set() }
                    .help("Click to expand image; click again to return to chat")
                    .accessibilityAddTraits(.isButton)
                    .accessibilityAction {
                        expand(image)
                    }
            } else if let player {
                VideoPlayer(player: player).frame(maxWidth: 560).frame(height: kind == "audio" ? 64 : 280)
            } else if loading {
                HStack(spacing: 6) { ProgressView().controlSize(.small); Text("Loading \(kind)…").font(.caption) }
            } else if error == nil, kind != "image" {
                Button {
                    playbackTask?.cancel()
                    playbackTask = Task { await load(play: true) }
                } label: {
                    Label(title.isEmpty ? "Play \(kind)" : title, systemImage: "play.circle.fill")
                }
            }
            if let error { Text(error).font(.caption).foregroundStyle(.secondary) }
            if let url = ChatMediaSource.validatedURL(source) {
                Link(title.isEmpty ? "Open \(kind) source" : title, destination: url)
                    .font(.caption).lineLimit(2)
            } else {
                Text("Unsupported media source").font(.caption).foregroundStyle(.secondary)
            }
        }
        .overlay(alignment: .topTrailing) {
            if (hovering || saving), ChatMediaSource.validatedURL(source) != nil {
                Button { saveTask = Task { await saveOriginal() } } label: {
                    Label(saving ? "Saving…" : "Download", systemImage: "arrow.down.to.line")
                        .font(.caption).padding(.horizontal, 8).padding(.vertical, 5)
                        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 6))
                }
                .buttonStyle(.plain).disabled(saving).padding(5)
                .help("Save original media")
            }
        }
        .onHover { hovering = $0 }
        .task(id: source) {
            cleanup()
            image = nil; error = nil; original = nil
            if kind == "image" { await load(play: false) }
        }
        .onDisappear { cleanup(); image = nil; original = nil }
    }

    private func previewSize(_ image: NSImage) -> CGSize {
        let width = max(1, image.size.width), height = max(1, image.size.height)
        let scale = min(1, min(560 / width, 340 / height))
        return CGSize(width: width * scale, height: height * scale)
    }

    @MainActor private func expand(_ image: NSImage) {
        var seen = Set<String>()
        var gallery = (model.chat.conversations[model.chat.selected] ?? []).flatMap { message in
            ChatMarkdown.blocks(message.presentation.answer).filter { $0.kind == "image" }.map {
                ChatImageReference(source: $0.marker, title: $0.text)
            }
        }.filter { ChatMediaSource.url($0.source) != nil && seen.insert($0.source).inserted }
        if !gallery.contains(where: { $0.source == source }) { gallery.append(ChatImageReference(source: source, title: title)) }
        model.expandedChatImage = ChatExpandedImage(image: image, resource: original, source: source, title: title, gallery: gallery)
    }

    @MainActor private func load(play: Bool) async {
        guard let url = ChatMediaSource.validatedURL(source), ["image", "video", "audio"].contains(kind) else {
            error = ChatMediaError.unsupported.localizedDescription; return
        }
        loading = true; error = nil
        defer { loading = false }
        do {
            let resource = try await ChatMediaLoader.resource(from: url, maximumBytes: byteLimit)
            try Task.checkCancellation()
            original = resource
            let data = resource.data
            if kind == "image" {
                image = try ChatMediaLoader.image(from: data)
            } else {
                // AVPlayer reads only this downloaded local copy: no cookies, credentials,
                // playlist subrequests or arbitrary network loads through AVFoundation.
                let resourceExtension = URL(fileURLWithPath: resource.filename).pathExtension
                let ext = resourceExtension.isEmpty ? (kind == "audio" ? "m4a" : "mp4") : resourceExtension
                let local = FileManager.default.temporaryDirectory.appendingPathComponent("mlx2-chat-\(UUID().uuidString).\(ext)")
                try data.write(to: local, options: .atomic)
                temporaryURL = local
                let asset = AVURLAsset(url: local, options: [AVURLAssetReferenceRestrictionsKey: AVAssetReferenceRestrictions.forbidAll.rawValue])
                guard try await asset.load(.isPlayable) else { throw ChatMediaError.unsupported }
                try Task.checkCancellation()
                let newPlayer = AVPlayer(playerItem: AVPlayerItem(asset: asset))
                player = newPlayer
                if play { newPlayer.play() }
            }
        } catch is CancellationError {
        } catch {
            if !Task.isCancelled { self.error = error.localizedDescription }
        }
    }

    private var byteLimit: Int { kind == "image" ? ChatMediaSource.maximumBytes : ChatMediaSource.maximumPlaybackBytes }

    @MainActor private func saveOriginal() async {
        guard !saving, let url = ChatMediaSource.validatedURL(source) else { return }
        saving = true
        defer { saving = false }
        do {
            let resource: ChatMediaResource
            if let original { resource = original }
            else { resource = try await ChatMediaLoader.resource(from: url, maximumBytes: byteLimit) }
            try Task.checkCancellation()
            original = resource
            try await ChatMediaSaver.save(resource)
        } catch is CancellationError {
        } catch {
            if !Task.isCancelled { self.error = error.localizedDescription }
        }
    }

    private func cleanup() {
        saveTask?.cancel(); saveTask = nil
        playbackTask?.cancel(); playbackTask = nil
        player?.pause(); player = nil
        if let temporaryURL { try? FileManager.default.removeItem(at: temporaryURL) }
        temporaryURL = nil
    }
}


private enum ChatMediaSaver {
    @MainActor static func save(_ resource: ChatMediaResource) async throws {
        let panel = NSSavePanel()
        panel.title = "Save original media"
        panel.nameFieldStringValue = resource.filename
        panel.canCreateDirectories = true
        let response: NSApplication.ModalResponse = await withCheckedContinuation { continuation in
            panel.begin { continuation.resume(returning: $0) }
        }
        try Task.checkCancellation()
        if response == .OK, let destination = panel.url {
            try resource.data.write(to: destination, options: .atomic)
        }
    }
}

struct ChatImageReference: Identifiable {
    let source: String
    let title: String
    var id: String { source }
}

struct ChatExpandedImage: Identifiable {
    let id = UUID()
    let image: NSImage
    let resource: ChatMediaResource?
    let source: String
    let title: String
    var gallery: [ChatImageReference] = []
}

/// In-window expansion; its gallery is a snapshot of this conversation only.
struct ChatImageLightbox: View {
    let item: ChatExpandedImage
    let dismiss: () -> Void
    @FocusState private var keyboardFocused: Bool
    @State private var index: Int
    @State private var displayedImage: NSImage?
    @State private var displayedResource: ChatMediaResource?
    @State private var loading = false
    @State private var saving = false
    @State private var error: String?
    @State private var saveTask: Task<Void, Never>?

    init(item: ChatExpandedImage, dismiss: @escaping () -> Void) {
        self.item = item
        self.dismiss = dismiss
        _index = State(initialValue: item.gallery.firstIndex { $0.source == item.source } ?? 0)
        _displayedImage = State(initialValue: item.image)
        _displayedResource = State(initialValue: item.resource)
    }

    private var gallery: [ChatImageReference] {
        item.gallery.isEmpty ? [ChatImageReference(source: item.source, title: item.title)] : item.gallery
    }
    private var current: ChatImageReference { gallery[min(index, gallery.count - 1)] }

    var body: some View {
        VStack(spacing: 0) {
            header
                .padding(.horizontal, 16).padding(.vertical, 12)
                .fixedSize(horizontal: false, vertical: true)
            imageArea
                .padding(.horizontal, 12).padding(.bottom, 12)
            if let error {
                VStack(spacing: 5) {
                    Text(error).font(.caption)
                    if let url = ChatMediaSource.url(current.source) { Link("Open image source", destination: url).font(.caption) }
                }.padding(8).background(.regularMaterial, in: RoundedRectangle(cornerRadius: 6))
                    .padding(.horizontal, 16).padding(.bottom, 12)
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background { Color.black.opacity(0.96).contentShape(Rectangle()).onTapGesture(perform: dismiss) }
        .foregroundStyle(.white)
        .focusable().focused($keyboardFocused).focusEffectDisabled()
        .onAppear { keyboardFocused = true }
        .onKeyPress(.leftArrow) {
            if !saving, index > 0 { index -= 1 }
            return .handled
        }
        .onKeyPress(.rightArrow) {
            if !saving, index + 1 < gallery.count { index += 1 }
            return .handled
        }
        .onExitCommand(perform: dismiss)
        .task(id: current.source) { await loadCurrent() }
        .onDisappear { saveTask?.cancel(); NSCursor.arrow.set() }
        .accessibilityIdentifier("chat-image-lightbox")
    }

    private var header: some View {
        HStack(spacing: 8) {
            if !current.title.isEmpty { Text(current.title).font(.callout).lineLimit(1) }
            if gallery.count > 1 { Text("\(index + 1) / \(gallery.count)").font(.caption).foregroundStyle(.secondary) }
            Spacer(minLength: 8)
            Button { saveTask = Task { await saveOriginal() } } label: {
                Label(saving ? "Saving…" : "Download", systemImage: "arrow.down.to.line")
                    .padding(.horizontal, 9).padding(.vertical, 6)
                    .background(.regularMaterial.opacity(0.85), in: RoundedRectangle(cornerRadius: 6))
            }.buttonStyle(.plain).disabled(saving || loading).fixedSize()
            Button(action: dismiss) { Image(systemName: "xmark.circle.fill").font(.title2) }
                .buttonStyle(.plain).keyboardShortcut(.cancelAction).help("Close image (Escape)")
        }
    }

    private var imageArea: some View {
        GeometryReader { geometry in
            ZStack {
                Color.clear.contentShape(Rectangle()).onTapGesture(perform: dismiss)
                if let image = displayedImage {
                    Image(nsImage: image).resizable().scaledToFit()
                        .frame(width: max(1, geometry.size.width), height: max(1, geometry.size.height))
                        .contentShape(Rectangle()).onTapGesture(perform: dismiss)
                        .onHover { inside in (inside ? NSCursor.pointingHand : NSCursor.arrow).set() }
                        .accessibilityLabel(current.title.isEmpty ? "Expanded chat image" : current.title)
                        .accessibilityAddTraits(.isButton).accessibilityAction { dismiss() }
                        .help("Click to return to chat")
                }
                if loading { ProgressView("Loading image…").padding(16).background(.regularMaterial, in: RoundedRectangle(cornerRadius: 8)) }
                HStack {
                    if index > 0 {
                        navigationButton("Previous image", symbol: "chevron.left") { index -= 1 }
                    }
                    Spacer()
                    if index + 1 < gallery.count {
                        navigationButton("Next image", symbol: "chevron.right") { index += 1 }
                    }
                }.padding(.horizontal, 2)
            }
            .frame(width: geometry.size.width, height: geometry.size.height)
        }
    }

    private func navigationButton(_ title: String, symbol: String, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            Image(systemName: symbol).font(.title2.weight(.medium)).frame(width: 42, height: 56)
                .background(.regularMaterial.opacity(0.75), in: RoundedRectangle(cornerRadius: 9))
        }.buttonStyle(.plain).disabled(saving).help(title)
            .accessibilityLabel(title)
    }

    @MainActor private func loadCurrent() async {
        saveTask?.cancel()
        let reference = current
        displayedImage = nil; displayedResource = nil; error = nil
        if reference.source == item.source {
            displayedImage = item.image; displayedResource = item.resource; loading = false; return
        }
        loading = true
        defer { if current.source == reference.source { loading = false } }
        do {
            guard let url = ChatMediaSource.url(reference.source) else { throw ChatMediaError.unsupported }
            let resource = try await ChatMediaLoader.resource(from: url, maximumBytes: ChatMediaSource.maximumBytes)
            try Task.checkCancellation()
            let image = try ChatMediaLoader.image(from: resource.data)
            guard current.source == reference.source else { return }
            displayedResource = resource; displayedImage = image
        } catch is CancellationError {
        } catch {
            if !Task.isCancelled, current.source == reference.source { self.error = error.localizedDescription }
        }
    }

    @MainActor private func saveOriginal() async {
        guard !saving, let url = ChatMediaSource.url(current.source) else { return }
        saving = true; error = nil
        defer { saving = false }
        do {
            let resource: ChatMediaResource
            if let original = displayedResource { resource = original }
            else { resource = try await ChatMediaLoader.resource(from: url, maximumBytes: ChatMediaSource.maximumBytes) }
            try Task.checkCancellation()
            try await ChatMediaSaver.save(resource)
        } catch is CancellationError {
        } catch {
            if !Task.isCancelled { self.error = error.localizedDescription }
        }
    }
}
