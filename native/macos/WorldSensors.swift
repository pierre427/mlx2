import AppKit
@preconcurrency import AVFoundation
import Foundation
import Speech
import SwiftUI

// Hardware and speech permissions are requested only by explicit start actions.
// Camera preview stays local; cameraAttachment captures exactly one still photo.
@MainActor final class WorldSensors: NSObject, ObservableObject, AVCapturePhotoCaptureDelegate, AVSpeechSynthesizerDelegate {
    @Published private(set) var cameraEnabled = false
    @Published private(set) var voiceInputEnabled = false
    @Published private(set) var speechOutputEnabled = false
    @Published private(set) var isSpeaking = false
    @Published var transcript = ""
    @Published var error: String?
    var onFinalTranscript: ((String) -> Void)?
    let captureSession = AVCaptureSession()
    private let cameraQueue = DispatchQueue(label: "mlx2.world.camera")
    private let photoOutput = AVCapturePhotoOutput()
    private var photos: [Int64: CheckedContinuation<ChatAttachment, Error>] = [:]
    private let audioEngine = AVAudioEngine()
    private var speechRequest: SFSpeechAudioBufferRecognitionRequest?
    private var speechTask: SFSpeechRecognitionTask?
    private var microphoneTapInstalled = false
    private let speaker = AVSpeechSynthesizer()
    private var cameraStarting = false
    private var voiceStarting = false
    private var cameraGeneration = 0
    private var voiceGeneration = 0

    override init() {
        super.init()
        speaker.delegate = self
    }

    nonisolated func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didStart utterance: AVSpeechUtterance) {
        Task { @MainActor [weak self] in self?.isSpeaking = self?.speaker.isSpeaking ?? false }
    }
    nonisolated func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didFinish utterance: AVSpeechUtterance) {
        Task { @MainActor [weak self] in self?.isSpeaking = self?.speaker.isSpeaking ?? false }
    }
    nonisolated func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didCancel utterance: AVSpeechUtterance) {
        Task { @MainActor [weak self] in self?.isSpeaking = self?.speaker.isSpeaking ?? false }
    }

    static let voiceTrainingPrerequisites = "System speech uses installed macOS voices. Voice cloning or training requires a separately configured compatible audio provider, a supported voice-training model, authorized reference recordings, and that provider's training and inference endpoints. No voice is cloned by this control."

    func startCamera() async {
        guard !cameraEnabled, !cameraStarting else { return }
        cameraStarting = true
        cameraGeneration += 1
        let generation = cameraGeneration
        defer { cameraStarting = false }
        let granted = await AVCaptureDevice.requestAccess(for: .video)
        guard generation == cameraGeneration else { return }
        guard granted else { error = "Camera access was not granted. Enable it in macOS Privacy & Security to use the preview."; return }
        let session = captureSession
        let output = photoOutput
        do {
            try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
                cameraQueue.async {
                    do {
                        guard let device = AVCaptureDevice.default(for: .video) else { throw DesktopError.message("No camera is available.") }
                        let input = try AVCaptureDeviceInput(device: device)
                        session.beginConfiguration()
                        session.sessionPreset = .photo
                        guard session.canAddInput(input), session.canAddOutput(output) else {
                            session.commitConfiguration()
                            throw DesktopError.message("This camera cannot provide a still-photo preview.")
                        }
                        session.addInput(input)
                        session.addOutput(output)
                        session.commitConfiguration()
                        session.startRunning()
                        continuation.resume()
                    } catch { continuation.resume(throwing: error) }
                }
            }
            guard generation == cameraGeneration else { stopCamera(); return }
            cameraEnabled = true
            error = nil
        } catch { self.error = error.localizedDescription }
    }

    func stopCamera() {
        cameraGeneration += 1
        cameraEnabled = false
        let session = captureSession
        cameraQueue.async {
            session.stopRunning()
            session.beginConfiguration()
            for input in session.inputs { session.removeInput(input) }
            for output in session.outputs { session.removeOutput(output) }
            session.commitConfiguration()
        }
        let pending = Array(photos.values)
        photos.removeAll()
        for continuation in pending { continuation.resume(throwing: CancellationError()) }
    }

    func cameraAttachment() async throws -> ChatAttachment {
        guard cameraEnabled, captureSession.isRunning else { throw DesktopError.message("Enable the camera preview before taking a photo.") }
        guard photos.isEmpty else { throw DesktopError.message("A camera photo is already being captured.") }
        let settings = AVCapturePhotoSettings()
        return try await withCheckedThrowingContinuation { continuation in
            photos[settings.uniqueID] = continuation
            photoOutput.capturePhoto(with: settings, delegate: self)
        }
    }

    nonisolated func photoOutput(_ output: AVCapturePhotoOutput, didFinishProcessingPhoto photo: AVCapturePhoto, error: Error?) {
        let identity = photo.resolvedSettings.uniqueID
        let data = photo.fileDataRepresentation()
        Task { @MainActor [weak self] in
            guard let self, let continuation = self.photos.removeValue(forKey: identity) else { return }
            if let error { continuation.resume(throwing: error); return }
            guard self.cameraEnabled, let data, let image = NSImage(data: data), let tiff = image.tiffRepresentation,
                  let bitmap = NSBitmapImageRep(data: tiff), let jpeg = bitmap.representation(using: .jpeg, properties: [.compressionFactor: 0.85]) else {
                continuation.resume(throwing: DesktopError.message("The camera did not return a usable image.")); return
            }
            guard jpeg.count <= 32 * 1024 * 1024 else {
                continuation.resume(throwing: DesktopError.message("Camera image exceeds the 32 MiB attachment limit.")); return
            }
            let formatter = ISO8601DateFormatter()
            continuation.resume(returning: ChatAttachment(name: "Camera " + formatter.string(from: Date()) + ".jpg", modality: "image",
                part: ["type": "image_url", "image_url": ["url": "data:image/jpeg;base64," + jpeg.base64EncodedString()]]))
        }
    }

    nonisolated func photoOutput(_ output: AVCapturePhotoOutput, didFinishCaptureFor resolvedSettings: AVCaptureResolvedPhotoSettings, error: Error?) {
        guard let error else { return }
        Task { @MainActor [weak self] in
            self?.photos.removeValue(forKey: resolvedSettings.uniqueID)?.resume(throwing: error)
        }
    }

    func startVoiceInput(allowNetwork: Bool = false) async {
        guard !voiceInputEnabled, !voiceStarting else { return }
        voiceStarting = true
        voiceGeneration += 1
        let generation = voiceGeneration
        defer { voiceStarting = false }
        let microphone = await AVCaptureDevice.requestAccess(for: .audio)
        guard generation == voiceGeneration else { return }
        guard microphone else { error = "Microphone access was not granted."; return }
        let authorization = await withCheckedContinuation { continuation in
            SFSpeechRecognizer.requestAuthorization { continuation.resume(returning: $0) }
        }
        guard generation == voiceGeneration else { return }
        guard authorization == .authorized else { error = "Speech recognition access was not granted."; return }
        guard let recognizer = SFSpeechRecognizer(locale: .current), recognizer.isAvailable else { error = "Speech recognition is unavailable for the current language."; return }
        guard allowNetwork || recognizer.supportsOnDeviceRecognition else {
            error = "On-device speech recognition is unavailable. Enable the explicit network speech option to use the system service."; return
        }
        do {
            let request = SFSpeechAudioBufferRecognitionRequest()
            request.shouldReportPartialResults = true
            request.requiresOnDeviceRecognition = !allowNetwork
            let input = audioEngine.inputNode
            let format = input.outputFormat(forBus: 0)
            guard format.sampleRate > 0, format.channelCount > 0 else { throw DesktopError.message("The selected microphone has no usable audio format.") }
            speechRequest = request
            let bufferSize = AVAudioFrameCount(max(1024, format.sampleRate * 0.2))
            if #available(macOS 27.0, *) {
                try input.__installTap(onBus: 0, bufferSize: bufferSize, format: format, error: (), block: { buffer, _ in request.append(buffer) })
            } else {
                input.installTap(onBus: 0, bufferSize: bufferSize, format: format) { buffer, _ in request.append(buffer) }
            }
            microphoneTapInstalled = true
            audioEngine.prepare()
            try audioEngine.start()
            transcript = ""
            voiceInputEnabled = true
            error = nil
            speechTask = recognizer.recognitionTask(with: request) { [weak self] result, failure in
                Task { @MainActor in
                    guard let self, self.voiceInputEnabled, generation == self.voiceGeneration else { return }
                    if let result {
                        self.transcript = result.bestTranscription.formattedString
                        if result.isFinal {
                            let text = self.transcript
                            self.stopVoiceInput(deliverTranscript: false)
                            self.onFinalTranscript?(text)
                        }
                    }
                    if let failure { self.error = failure.localizedDescription; self.stopVoiceInput(deliverTranscript: false) }
                }
            }
        } catch {
            stopVoiceInput(deliverTranscript: false)
            self.error = error.localizedDescription
        }
    }

    func stopVoiceInput(deliverTranscript: Bool = true) {
        voiceGeneration += 1
        let wasRunning = voiceInputEnabled || audioEngine.isRunning || speechRequest != nil
        voiceInputEnabled = false
        audioEngine.stop()
        if microphoneTapInstalled { audioEngine.inputNode.removeTap(onBus: 0); microphoneTapInstalled = false }
        speechRequest?.endAudio()
        speechTask?.cancel()
        speechTask = nil
        speechRequest = nil
        if deliverTranscript, wasRunning, !transcript.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty { onFinalTranscript?(transcript) }
    }

    func setSpeechOutputEnabled(_ enabled: Bool) {
        speechOutputEnabled = enabled
        if !enabled { stopSpeaking() }
    }

    func speak(_ text: String) {
        guard speechOutputEnabled, !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return }
        speaker.stopSpeaking(at: .immediate)
        let utterance = AVSpeechUtterance(string: String(text.prefix(20000)))
        utterance.voice = AVSpeechSynthesisVoice(language: Locale.current.identifier.replacingOccurrences(of: "_", with: "-"))
        speaker.speak(utterance)
    }

    func stopSpeaking() { speaker.stopSpeaking(at: .immediate) }
    func stopAll() { stopCamera(); stopVoiceInput(deliverTranscript: false); stopSpeaking(); setSpeechOutputEnabled(false) }
}

private final class WorldCameraPreviewView: NSView {
    let preview = AVCaptureVideoPreviewLayer()
    override init(frame: NSRect) {
        super.init(frame: frame)
        wantsLayer = true
        layer = CALayer()
        preview.videoGravity = .resizeAspect
        layer?.addSublayer(preview)
    }
    required init?(coder: NSCoder) { fatalError("init(coder:) is not supported") }
    override func layout() { super.layout(); preview.frame = bounds }
}

struct WorldCameraPreview: NSViewRepresentable {
    let session: AVCaptureSession
    func makeNSView(context: Context) -> NSView {
        let view = WorldCameraPreviewView(frame: .zero)
        view.preview.session = session
        return view
    }
    func updateNSView(_ nsView: NSView, context: Context) {
        (nsView as? WorldCameraPreviewView)?.preview.session = session
    }
}
