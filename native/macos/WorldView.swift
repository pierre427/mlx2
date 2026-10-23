import AppKit
import Combine
import SceneKit
import SwiftUI
import UniformTypeIdentifiers

private func worldColor(_ value: String, fallback: NSColor = .systemCyan) -> NSColor {
    let hex = value.hasPrefix("#") ? String(value.dropFirst()) : value
    guard hex.count == 6, let number = UInt32(hex, radix: 16) else { return fallback }
    return NSColor(srgbRed: CGFloat((number >> 16) & 255) / 255,
                   green: CGFloat((number >> 8) & 255) / 255,
                   blue: CGFloat(number & 255) / 255, alpha: 1)
}

private func worldVector(_ value: JSON, fallback: Float = 0) -> SCNVector3 {
    SCNVector3(Float(value[0].double ?? Double(fallback)), Float(value[1].double ?? Double(fallback)),
               Float(value[2].double ?? Double(fallback)))
}

private func worldDocumentData(_ value: JSON) -> Data? {
    try? JSONSerialization.data(withJSONObject: value.raw, options: [.sortedKeys])
}

private final class WorldSceneView: SCNView {
    var selectObject: ((String?) -> Void)?
    override func mouseDown(with event: NSEvent) {
        let point = convert(event.locationInWindow, from: nil)
        var selected = hitTest(point, options: [.searchMode: SCNHitTestSearchMode.closest.rawValue]).first?.node
        while let node = selected, !(node.name?.hasPrefix("object:") ?? false) { selected = node.parent }
        selectObject?(selected?.name.map { String($0.dropFirst(7)) })
        super.mouseDown(with: event)
    }
}

private struct NativeWorldScene: NSViewRepresentable {
    var world: JSON
    var selection: String?
    var assetRoot: String
    var activity: String
    var resetCamera: Int
    var select: (String?) -> Void

    final class Coordinator {
        var document: Data?
        var selected: String?
        var reset = 0
        var objects: [String: Data] = [:]
        var activity = ""
        let camera = SCNNode()
    }
    func makeCoordinator() -> Coordinator { Coordinator() }

    func makeNSView(context: Context) -> WorldSceneView {
        let view = WorldSceneView()
        view.scene = SCNScene()
        view.allowsCameraControl = true
        view.defaultCameraController.interactionMode = .orbitTurntable
        view.defaultCameraController.inertiaEnabled = true
        view.antialiasingMode = .multisampling4X
        view.preferredFramesPerSecond = 60
        view.autoenablesDefaultLighting = false
        view.backgroundColor = worldColor("#071321")
        let camera = context.coordinator.camera
        camera.name = "world-camera"
        camera.camera = SCNCamera()
        camera.camera?.zNear = 0.05
        camera.camera?.zFar = 1000
        camera.camera?.wantsHDR = true
        camera.position = SCNVector3(0, 2.8, 12)
        camera.look(at: SCNVector3(0, 0, 0))
        view.scene?.rootNode.addChildNode(camera)
        view.pointOfView = camera
        view.defaultCameraController.target = SCNVector3(0, 0, 0)
        let ambient = SCNNode()
        ambient.light = SCNLight()
        ambient.light?.type = .ambient
        ambient.light?.intensity = 450
        ambient.light?.color = NSColor(calibratedWhite: 0.8, alpha: 1)
        view.scene?.rootNode.addChildNode(ambient)
        let light = SCNNode()
        light.name = "world-keylight"
        light.light = SCNLight()
        light.light?.type = .omni
        light.light?.intensity = 1100
        light.position = SCNVector3(2, 7, 7)
        view.scene?.rootNode.addChildNode(light)
        let floor = SCNFloor()
        floor.reflectivity = 0.08
        floor.firstMaterial?.diffuse.contents = worldColor("#102333")
        floor.firstMaterial?.roughness.contents = 0.85
        let floorNode = SCNNode(geometry: floor)
        floorNode.name = "world-floor"
        floorNode.position.y = -1.8
        view.scene?.rootNode.addChildNode(floorNode)
        // The ground grid is actual geometry, giving camera motion depth cues.
        for index in -12...12 {
            for horizontal in [true, false] {
                let box = SCNBox(width: horizontal ? 24 : 0.012, height: 0.008,
                                 length: horizontal ? 0.012 : 24, chamferRadius: 0)
                box.firstMaterial?.diffuse.contents = worldColor("#204455")
                box.firstMaterial?.emission.contents = worldColor("#102633")
                let line = SCNNode(geometry: box)
                line.position = horizontal ? SCNVector3(0, -1.79, Float(index)) : SCNVector3(Float(index), -1.79, 0)
                view.scene?.rootNode.addChildNode(line)
            }
        }
        return view
    }

    func updateNSView(_ view: WorldSceneView, context: Context) {
        view.selectObject = select
        let coordinator = context.coordinator
        if coordinator.reset != resetCamera {
            coordinator.reset = resetCamera
            coordinator.camera.position = SCNVector3(0, 2.8, 12)
            coordinator.camera.eulerAngles = SCNVector3Zero
            coordinator.camera.look(at: SCNVector3(0, 0, 0))
            view.pointOfView = coordinator.camera
        }
        let data = worldDocumentData(world)
        if data != coordinator.document {
            coordinator.document = data
            view.scene?.background.contents = worldColor(world["theme"]["background"].text)
            coordinator.camera.camera?.exposureOffset = (world["theme"]["exposure"].double ?? 1) - 1
            let rows = world["objects"].array
            let live = Set(rows.map { $0["id"].text })
            for node in view.scene?.rootNode.childNodes ?? [] {
                if let name = node.name, name.hasPrefix("object:"), !live.contains(String(name.dropFirst(7))) {
                    node.removeFromParentNode()
                }
            }
            coordinator.objects = coordinator.objects.filter { live.contains($0.key) }
            SCNTransaction.begin()
            let transition = world["transitions"]["style"].text
            SCNTransaction.animationDuration = transition == "instant" ? 0 : (world["transitions"]["duration"].double ?? 0.6)
            SCNTransaction.animationTimingFunction = CAMediaTimingFunction(name: transition == "linear" ? .linear : .easeInEaseOut)
            view.scene?.rootNode.childNode(withName: "world-keylight", recursively: false)?.light?.color = worldColor(world["theme"]["accent"].text)
            for row in rows {
                let identity = row["id"].text
                guard let objectData = worldDocumentData(row), coordinator.objects[identity] != objectData else { continue }
                coordinator.objects[identity] = objectData
                let node = view.scene?.rootNode.childNode(withName: "object:" + identity, recursively: false) ?? SCNNode()
                node.name = "object:" + identity
                node.removeAllActions()
                node.childNodes.forEach { $0.removeFromParentNode() }
                node.geometry = geometry(row)
                node.geometry?.materials = [material(row)]
                node.position = worldVector(row["position"])
                let rotation = worldVector(row["rotation"])
                node.eulerAngles = SCNVector3(rotation.x * .pi / 180, rotation.y * .pi / 180, rotation.z * .pi / 180)
                node.scale = worldVector(row["scale"], fallback: 1)
                if node.parent == nil { view.scene?.rootNode.addChildNode(node) }
                if row["shape"].text != "panel" { addLabel(row, to: node) }
                animate(row, node: node)
            }
            SCNTransaction.commit()
        }
        for row in world["objects"].array where row["kind"].text == "avatar" {
            guard let node = view.scene?.rootNode.childNode(withName: "object:" + row["id"].text, recursively: false) else { continue }
            let indicator = node.childNode(withName: "agent-activity", recursively: false)
            if coordinator.activity != activity || indicator == nil {
                indicator?.removeFromParentNode()
                let ring = SCNTorus(ringRadius: 0.94, pipeRadius: 0.045)
                let statusColor: NSColor = activity == "thinking" ? .systemOrange : activity == "speaking" ? .systemGreen : .systemCyan
                ring.firstMaterial?.diffuse.contents = statusColor
                ring.firstMaterial?.emission.contents = statusColor
                let status = SCNNode(geometry: ring)
                status.name = "agent-activity"
                status.eulerAngles.x = .pi / 2
                if activity != "idle" { status.runAction(.repeatForever(.rotateBy(x: 0, y: .pi * 2, z: 0, duration: activity == "speaking" ? 1.1 : 3))) }
                node.addChildNode(status)
            }
        }
        coordinator.activity = activity
        if coordinator.selected != selection || data != nil {
            coordinator.selected = selection
            for node in view.scene?.rootNode.childNodes ?? [] where node.name?.hasPrefix("object:") == true {
                let chosen = node.name == "object:" + (selection ?? "")
                node.geometry?.firstMaterial?.emission.intensity = chosen ? 0.6 : 0.12
            }
        }
    }

    private func geometry(_ row: JSON) -> SCNGeometry {
        switch row["shape"].text {
        case "sphere": return SCNSphere(radius: 0.72)
        case "cube": return SCNBox(width: 1.35, height: 1.35, length: 1.35, chamferRadius: 0.12)
        case "ring": return SCNTorus(ringRadius: 0.74, pipeRadius: 0.16)
        default: let plane = SCNPlane(width: 3.2, height: 2); plane.cornerRadius = 0.12; return plane
        }
    }
    private func material(_ row: JSON) -> SCNMaterial {
        let material = SCNMaterial()
        let color = worldColor(row["color"].text)
        if let image = grantedTexture(row["resource"].string ?? "") {
            material.diffuse.contents = image
        } else {
            material.diffuse.contents = row["shape"].text == "panel" ? texture(row) : color
        }
        material.emission.contents = color
        material.emission.intensity = 0.12
        material.metalness.contents = 0.2
        material.roughness.contents = 0.35
        material.isDoubleSided = true
        return material
    }
    private func grantedTexture(_ path: String) -> NSImage? {
        guard !assetRoot.isEmpty, path.hasPrefix("/") else { return nil }
        let root = URL(fileURLWithPath: assetRoot).standardizedFileURL
        let file = URL(fileURLWithPath: path).standardizedFileURL
        guard root == root.resolvingSymlinksInPath(), file == file.resolvingSymlinksInPath(),
              file.deletingLastPathComponent() == root, ["png", "jpg", "jpeg", "webp", "image"].contains(file.pathExtension.lowercased()),
              let properties = try? file.resourceValues(forKeys: [.isRegularFileKey, .isSymbolicLinkKey, .fileSizeKey]),
              properties.isRegularFile == true, properties.isSymbolicLink != true,
              let size = properties.fileSize, size > 0, size <= 8 * 1024 * 1024,
              let data = try? Data(contentsOf: file), data.count <= 8 * 1024 * 1024 else { return nil }
        return NSImage(data: data)
    }
    private func texture(_ row: JSON) -> NSImage {
        let image = NSImage(size: NSSize(width: 960, height: 600))
        image.lockFocus()
        let color = worldColor(row["color"].text)
        worldColor("#102333").setFill()
        NSBezierPath(roundedRect: NSRect(x: 0, y: 0, width: 960, height: 600), xRadius: 28, yRadius: 28).fill()
        color.setStroke()
        let border = NSBezierPath(roundedRect: NSRect(x: 5, y: 5, width: 950, height: 590), xRadius: 26, yRadius: 26)
        border.lineWidth = 7
        border.stroke()
        (row["label"].text as NSString).draw(in: NSRect(x: 42, y: 470, width: 876, height: 90), withAttributes: [
            .font: NSFont.systemFont(ofSize: 58, weight: .semibold), .foregroundColor: color])
        (String(row["text"].text.prefix(1500)) as NSString).draw(in: NSRect(x: 42, y: 95, width: 876, height: 355), withAttributes: [
            .font: NSFont.systemFont(ofSize: 30), .foregroundColor: NSColor.white])
        (row["kind"].text.uppercased() as NSString).draw(at: NSPoint(x: 42, y: 35), withAttributes: [
            .font: NSFont.monospacedSystemFont(ofSize: 21, weight: .medium), .foregroundColor: color])
        image.unlockFocus()
        return image
    }
    private func addLabel(_ row: JSON, to node: SCNNode) {
        let text = SCNText(string: row["label"].text, extrusionDepth: 0.015)
        text.font = NSFont.systemFont(ofSize: 0.22, weight: .medium)
        text.flatness = 0.1
        text.firstMaterial?.diffuse.contents = NSColor.white
        let label = SCNNode(geometry: text)
        label.position = SCNVector3(-0.65, -1.1, 0.1)
        let constraint = SCNBillboardConstraint()
        constraint.freeAxes = .Y
        label.constraints = [constraint]
        node.addChildNode(label)
    }
    private func animate(_ row: JSON, node: SCNNode) {
        let speed = max(0.05, row["animation"]["speed"].double ?? 1)
        switch row["animation"]["kind"].text {
        case "spin": node.runAction(.repeatForever(.rotateBy(x: 0, y: .pi * 2, z: 0, duration: 12 / speed)))
        case "float": node.runAction(.repeatForever(.sequence([
            .moveBy(x: 0, y: 0.18, z: 0, duration: 1.5 / speed), .moveBy(x: 0, y: -0.18, z: 0, duration: 1.5 / speed)])))
        case "pulse":
            let original = node.scale
            node.runAction(.repeatForever(.sequence([
                .customAction(duration: 1.5 / speed) { node, elapsed in
                    let phase: CGFloat = CGFloat(elapsed) * CGFloat(speed) / 1.5
                    let factor: CGFloat = 1 + 0.05 * sin(phase * .pi * 2)
                    let x: CGFloat = original.x * factor
                    let y: CGFloat = original.y * factor
                    let z: CGFloat = original.z * factor
                    node.scale = SCNVector3Make(x, y, z)
                }])))
        default: break
        }
    }
}

struct WorldView: View {
    @EnvironmentObject var model: DesktopModel
    @State private var selection: String?
    @State private var selectedWorld = ""
    @State private var newWorldName = "My world"
    @State private var activeName = ""
    @State private var background = "#071321"
    @State private var accent = "#38CBFF"
    @State private var exposure = "1"
    @State private var transitionDuration = "0.6"
    @State private var transitionStyle = "ease"
    @State private var showInspector = true
    @State private var showChat = true
    @State private var showMemory = false
    @State private var resetCamera = 0
    @State private var label = ""
    @State private var kind = "chat"
    @State private var shape = "panel"
    @State private var position = "0, 0, 0"
    @State private var rotation = "0, 0, 0"
    @State private var scale = "1, 1, 1"
    @State private var color = "#38CBFF"
    @State private var text = ""
    @State private var resource = ""
    @State private var animation = "none"
    @State private var speed = "1"
    @State private var localError: String?
    @StateObject private var sensors = WorldSensors()
    @State private var imageQuery = ""
    @State private var imageResults: [JSON] = []
    @State private var searchingImages = false
    @State private var lastSpoken: UUID?
    @State private var pendingObjectID: String?
    @State private var thinking = false
    @State private var speaking = false
    private let kinds = ["chat", "memory", "knowledge", "web", "file", "folder", "avatar", "media"]
    private var state: JSON { model.snapshot["world"] }
    private var world: JSON { state["active"] }
    private var selected: JSON { world["objects"].array.first { $0["id"].string == selection } ?? JSON() }

    var body: some View {
        GeometryReader { geometry in
            ZStack(alignment: .topLeading) {
                NativeWorldScene(world: world, selection: selection, assetRoot: state["asset_root"].string ?? "", activity: speaking ? "speaking" : thinking ? "thinking" : "idle", resetCamera: resetCamera) { identity in
                    selection = identity
                    if world["objects"].array.first(where: { $0["id"].string == identity })?["kind"].text == "chat" { showChat = true }
                }.accessibilityLabel("Interactive three dimensional world. Drag to orbit; scroll to zoom; click an object to select it.")
                VStack(alignment: .leading, spacing: 10) {
                    toolbar
                    if !state["errors"].array.isEmpty {
                        Text(state["errors"].array.map(\.text).joined(separator: "\n"))
                            .font(.caption).foregroundStyle(.orange).textSelection(.enabled)
                    }
                    HStack(alignment: .top, spacing: 12) {
                        if showInspector { inspector.frame(width: 275, height: max(300, geometry.size.height - 140)) }
                        Spacer(minLength: 0)
                        if showChat {
                            VStack(spacing: 0) {
                                HStack { Label("Chat", systemImage: "bubble.left.and.bubble.right"); Spacer(); Button { showChat = false } label: { Image(systemName: "xmark") }.buttonStyle(.plain) }.padding(12)
                                Divider()
                                ChatView(chat: model.chat)
                            }.frame(width: min(500, max(350, geometry.size.width * 0.46)), height: max(350, geometry.size.height - 130))
                                .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 14))
                        }
                    }
                    Spacer(minLength: 0)
                    HStack {
                        Text("Drag to orbit · scroll to zoom · click objects to edit").font(.caption)
                        Spacer()
                        if let localError { Text(localError).font(.caption).foregroundStyle(.orange).textSelection(.enabled) }
                    }.padding(10).background(.regularMaterial, in: RoundedRectangle(cornerRadius: 10))
                }.padding(14)
            }
        }
        .onAppear {
            thinking = model.chat.generating
            speaking = sensors.isSpeaking
            lastSpoken = model.chat.conversations[model.chat.selected]?.last(where: { $0.role == "assistant" && $0.committed })?.id
            selectedWorld = world["id"].text
            selection = world["objects"].array.first(where: { $0["kind"].text == "chat" })?["id"].string
            loadInspector()
            loadWorldControls()
            sensors.onFinalTranscript = { value in
                if !model.chat.draft.isEmpty { model.chat.draft += "\n" }
                model.chat.draft += value
                showChat = true
            }
        }
        .onDisappear { sensors.stopAll() }
        .onReceive(model.chat.$generating.dropFirst()) { generating in
            thinking = generating
            guard !generating, sensors.speechOutputEnabled,
                  let response = model.chat.conversations[model.chat.selected]?.last(where: { $0.role == "assistant" && $0.committed }),
                  !response.text.isEmpty, response.id != lastSpoken else { return }
            lastSpoken = response.id
            sensors.speak(response.text)
        }
        .onReceive(sensors.$isSpeaking) { speaking = $0 }
        .onChange(of: worldDocumentData(world)) { _, _ in
            if pendingObjectID == selection && selected["id"].string != nil { loadInspector(); pendingObjectID = nil }
        }
        .onChange(of: selection) { _, _ in loadInspector() }
        .onChange(of: world["id"].string) { _, identity in
            selectedWorld = identity ?? ""
            selection = world["objects"].array.first(where: { $0["kind"].text == "chat" })?["id"].string
            loadInspector()
            loadWorldControls()
        }
        .onExitCommand { leaveFullScreen() }
        .sheet(isPresented: $showMemory) { VStack { HStack { Text("Memory and knowledge").font(.headline); Spacer(); Button("Done") { showMemory = false } }.padding(); MemoryView() }.frame(minWidth: 800, minHeight: 600) }
    }

    private var toolbar: some View {
        HStack(spacing: 10) {
            Button { leaveFullScreen(); model.section = model.worldReturnSection } label: { Label("Back", systemImage: "arrow.left") }
            Picker("World", selection: $selectedWorld) {
                ForEach(state["worlds"].array, id: \.worldIdentity) { item in Text(item["name"].text).tag(item["id"].text) }
            }.frame(maxWidth: 220).onChange(of: selectedWorld) { _, identity in
                if !identity.isEmpty && identity != world["id"].text { model.perform("world_load", ["world_id": identity]) }
            }
            Menu {
                ForEach(kinds, id: \.self) { item in Button("Add \(item.capitalized)") { addObject(item) } }
            } label: { Label("Object", systemImage: "plus") }
            Button { showInspector.toggle() } label: { Image(systemName: "slider.horizontal.3") }.help("Object inspector")
            Button { showChat.toggle() } label: { Image(systemName: "bubble.left.and.bubble.right") }.help("Conversation")
            Button { resetCamera += 1 } label: { Image(systemName: "viewfinder") }.help("Reset camera")
            Spacer(minLength: 0)
            Menu {
                Button("Create world") { model.perform("world_create", ["name": newWorldName]) }
                Button("Save world") { model.perform("world_save", ["world": world.raw]) }
                Divider()
                Button("Import world JSON…") { importWorld() }
                Button("Export world JSON…") { exportWorld() }
            } label: { Label("World", systemImage: "square.and.arrow.down") }
            Button { LifecycleActions.toggleFullscreen() } label: { Image(systemName: "arrow.up.left.and.arrow.down.right") }.help("Toggle full screen")
        }.padding(10).background(.regularMaterial, in: RoundedRectangle(cornerRadius: 12))
    }

    private var inspector: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 10) {
                DisclosureGroup("World appearance") {
                    VStack(alignment: .leading, spacing: 8) {
                        field("World name", value: $activeName)
                        field("Background #RRGGBB", value: $background)
                        field("Accent light #RRGGBB", value: $accent)
                        field("Exposure (0.1–4)", value: $exposure)
                        Picker("Transition", selection: $transitionStyle) { ForEach(["ease", "linear", "instant"], id: \.self) { Text($0.capitalized).tag($0) } }
                        field("Transition seconds (0–5)", value: $transitionDuration)
                        Button("Apply world appearance") { applyWorldControls() }
                        Divider()
                        field("New world name", value: $newWorldName)
                        Button("Create world") { model.perform("world_create", ["name": newWorldName]) }
                    }.padding(.vertical, 8)
                }
                Text("Scene objects").font(.headline)
                ForEach(world["objects"].array, id: \.worldIdentity) { item in
                    Button { selection = item["id"].text; if item["kind"].text == "chat" { showChat = true } } label: {
                        HStack { Image(systemName: icon(item["kind"].text)); Text(item["label"].text).lineLimit(1); Spacer(); if selection == item["id"].text { Image(systemName: "checkmark") } }
                    }.buttonStyle(.plain).padding(5)
                }
                Divider()
                Toggle("Agent may change the scene", isOn: Binding(get: { state["policy"]["autonomous_scene"].bool }, set: { model.perform("world_policy", ["changes": ["autonomous_scene": $0]]) }))
                    .help("Allow bounded declarative scene tools to apply changes without a separate confirmation for each action.")
                sensorPanel
                if selection != nil {
                    Divider()
                    Text("Selected object").font(.headline)
                    TextField("Label", text: $label)
                    Picker("Kind", selection: $kind) { ForEach(kinds, id: \.self) { Text($0.capitalized).tag($0) } }
                    Picker("Shape", selection: $shape) { ForEach(["panel", "sphere", "cube", "ring"], id: \.self) { Text($0.capitalized).tag($0) } }
                    field("Position x, y, z", value: $position)
                    field("Rotation x, y, z (degrees)", value: $rotation)
                    field("Scale x, y, z", value: $scale)
                    field("Material color #RRGGBB", value: $color)
                    Text("Plain text").font(.caption).foregroundStyle(.secondary)
                    TextEditor(text: $text).frame(height: 90).font(.callout)
                    field("Resource reference", value: $resource)
                    Text("Resources are labels until you explicitly open them. Scenes do not fetch pages or read local files.").font(.caption2).foregroundStyle(.secondary)
                    Picker("Animation", selection: $animation) { ForEach(["none", "pulse", "spin", "float"], id: \.self) { Text($0.capitalized).tag($0) } }
                    field("Animation speed", value: $speed)
                    HStack {
                        Button("Apply") { applyObject() }.buttonStyle(.borderedProminent)
                        Button("Remove", role: .destructive) { if let selection { model.perform("world_object_remove", ["object_id": selection]); self.selection = nil } }
                    }
                    if ["memory", "knowledge"].contains(kind) { Button("Open memory and knowledge") { showMemory = true } }
                    Divider()
                    Text("Image material").font(.headline)
                    Button("Choose image from disk…") { chooseTexture() }
                    TextField("Search Wikimedia Commons", text: $imageQuery)
                    HStack {
                        Button("Search images") { searchImages() }.disabled(imageQuery.trimmingCharacters(in: .whitespaces).isEmpty || searchingImages)
                        if searchingImages { ProgressView().controlSize(.small) }
                    }
                    ForEach(imageResults, id: \.worldIdentity) { result in
                        VStack(alignment: .leading, spacing: 3) {
                            Text(result["title"].text).font(.caption).lineLimit(2)
                            Text(result["license"].text + " · " + result["credit"].text).font(.caption2).foregroundStyle(.secondary).lineLimit(2)
                            Button("Use as material") { fetchImage(result) }
                        }
                    }
                    if !resource.isEmpty { Button(resource.hasPrefix("/") ? "Reveal resource in Finder" : "Open reference") { openResource() } }
                }
            }.padding(12)
        }.frame(maxHeight: 650).background(.regularMaterial, in: RoundedRectangle(cornerRadius: 14))
    }

    private var sensorPanel: some View {
        VStack(alignment: .leading, spacing: 8) {
            Divider()
            Text("Camera and voice").font(.headline)
            HStack {
                Button(sensors.cameraEnabled ? "Stop camera" : "Start camera") {
                    if sensors.cameraEnabled { sensors.stopCamera() } else { Task { await sensors.startCamera() } }
                }
                Button(sensors.voiceInputEnabled ? "Stop listening" : "Voice input") {
                    if sensors.voiceInputEnabled { sensors.stopVoiceInput() } else { Task { await sensors.startVoiceInput() } }
                }
            }
            if sensors.cameraEnabled {
                WorldCameraPreview(session: sensors.captureSession).frame(height: 140).clipShape(RoundedRectangle(cornerRadius: 8))
                Button("Attach a photo to chat") {
                    Task { @MainActor in
                        do { model.chat.attachments.append(try await sensors.cameraAttachment()); showChat = true }
                        catch { localError = error.localizedDescription }
                    }
                }
            }
            Toggle("Speak assistant replies", isOn: Binding(get: { sensors.speechOutputEnabled }, set: { sensors.setSpeechOutputEnabled($0) }))
            if !sensors.transcript.isEmpty { Text(sensors.transcript).font(.caption).lineLimit(3) }
            if let error = sensors.error { Text(error).font(.caption).foregroundStyle(.orange) }
            Text("Voice input fills the composer. Camera images are sent only when you attach a photo and submit chat.").font(.caption2).foregroundStyle(.secondary)
        }
    }
    private func chooseTexture() {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.png, .jpeg, UTType(filenameExtension: "webp") ?? .image]
        panel.allowsMultipleSelection = false
        panel.canChooseDirectories = false
        guard panel.runModal() == .OK, let url = panel.url else { return }
        Task { @MainActor in
            do {
                let result = try await model.request("world_asset_import", ["path": url.path])
                resource = result["resource"].text
                shape = "panel"
                applyObject()
                localError = nil
            } catch { localError = error.localizedDescription }
        }
    }
    private func searchImages() {
        searchingImages = true
        Task { @MainActor in
            defer { searchingImages = false }
            do {
                imageResults = try await model.request("world_image_search", ["query": imageQuery, "limit": 8])["results"].array
                localError = nil
            } catch { localError = error.localizedDescription }
        }
    }
    private func fetchImage(_ result: JSON) {
        Task { @MainActor in
            do {
                let response = try await model.request("world_asset_fetch", ["result_id": result["id"].text])
                resource = response["resource"].text
                shape = "panel"
                applyObject()
                localError = nil
            } catch { localError = error.localizedDescription }
        }
    }
    private func field(_ label: String, value: Binding<String>) -> some View {
        VStack(alignment: .leading, spacing: 3) { Text(label).font(.caption).foregroundStyle(.secondary); TextField(label, text: value) }
    }
    private func icon(_ kind: String) -> String {
        ["chat": "bubble.left", "memory": "brain", "knowledge": "books.vertical", "web": "globe", "file": "doc", "folder": "folder", "avatar": "person.crop.circle", "media": "photo.on.rectangle"][kind] ?? "cube"
    }
    private func loadWorldControls() {
        activeName = world["name"].string ?? "My world"
        background = world["theme"]["background"].string ?? "#071321"
        accent = world["theme"]["accent"].string ?? "#38CBFF"
        exposure = String(world["theme"]["exposure"].double ?? 1)
        transitionDuration = String(world["transitions"]["duration"].double ?? 0.6)
        transitionStyle = world["transitions"]["style"].string ?? "ease"
    }
    private func applyWorldControls() {
        guard var document = world.raw as? [String: Any], let exposure = Double(exposure), exposure.isFinite,
              let duration = Double(transitionDuration), duration.isFinite else { localError = "World appearance needs finite exposure and duration."; return }
        document["name"] = activeName
        document["theme"] = ["background": background, "accent": accent, "exposure": exposure]
        document["transitions"] = ["style": transitionStyle, "duration": duration]
        model.perform("world_save", ["world": document])
        localError = nil
    }
    private func loadInspector() {
        guard selection != nil else { return }
        label = selected["label"].string ?? ""
        kind = selected["kind"].string ?? "chat"
        shape = selected["shape"].string ?? "panel"
        position = vectorString(selected["position"], fallback: 0)
        rotation = vectorString(selected["rotation"], fallback: 0)
        scale = vectorString(selected["scale"], fallback: 1)
        color = selected["color"].string ?? "#38CBFF"
        text = selected["text"].string ?? ""
        resource = selected["resource"].string ?? ""
        animation = selected["animation"]["kind"].string ?? "none"
        speed = String(selected["animation"]["speed"].double ?? 1)
    }
    private func vectorString(_ value: JSON, fallback: Double) -> String { (0..<3).map { String(value[$0].double ?? fallback) }.joined(separator: ", ") }
    private func parseVector(_ value: String) -> [Double]? {
        let parts = value.split(separator: ",", omittingEmptySubsequences: false).map { Double($0.trimmingCharacters(in: .whitespaces)) }
        guard parts.count == 3, parts.allSatisfy({ $0?.isFinite == true }) else { return nil }
        return parts.compactMap { $0 }
    }
    private func applyObject() {
        guard let selection, let position = parseVector(position), let rotation = parseVector(rotation), let scale = parseVector(scale), let speed = Double(speed), speed.isFinite else {
            localError = "Transforms need three comma-separated numbers and a finite animation speed."; return
        }
        localError = nil
        model.perform("world_object_upsert", ["object": ["id": selection, "kind": kind, "label": label, "shape": shape,
            "position": position, "rotation": rotation, "scale": scale, "color": color, "text": text,
            "resource": resource, "animation": ["kind": animation, "speed": speed]]])
    }
    private func addObject(_ kind: String) {
        let identity = UUID().uuidString.lowercased()
        let count = world["objects"].array.count
        model.perform("world_object_upsert", ["object": ["id": identity, "kind": kind, "label": kind.capitalized,
            "shape": kind == "avatar" ? "sphere" : "panel", "position": [Double((count % 5) - 2) * 3.7, 0, Double(count / 5) * -3],
            "rotation": [0, 0, 0], "scale": [1, 1, 1], "color": "#38CBFF", "text": "", "resource": "",
            "animation": ["kind": "none", "speed": 1]]])
        pendingObjectID = identity
        selection = identity
    }
    private func leaveFullScreen() {
        LifecycleActions.toggleFullscreen(exitOnly: true)
    }
    private func openResource() {
        if resource.hasPrefix("/") { NSWorkspace.shared.activateFileViewerSelecting([URL(fileURLWithPath: resource)]); return }
        guard let url = URL(string: resource), url.scheme == "https", url.host != nil else { localError = "This reference is stored as text. Open memory references through the memory panel."; return }
        NSWorkspace.shared.open(url)
    }
    private func importWorld() {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.json]
        panel.allowsMultipleSelection = false
        panel.canChooseDirectories = false
        guard panel.runModal() == .OK, let url = panel.url else { return }
        do {
            let size = try url.resourceValues(forKeys: [.fileSizeKey]).fileSize ?? 0
            guard size <= 32_000_000 else { throw DesktopError.message("World JSON exceeds 32 MB") }
            let document = try JSONSerialization.jsonObject(with: Data(contentsOf: url))
            model.perform("world_import", ["document": document])
            localError = nil
        } catch { localError = error.localizedDescription }
    }
    private func exportWorld() {
        let panel = NSSavePanel()
        panel.allowedContentTypes = [.json]
        panel.nameFieldStringValue = "world.json"
        guard panel.runModal() == .OK, let url = panel.url, let identity = world["id"].string else { return }
        Task { @MainActor in
            do {
                let response = try await model.request("world_export", ["world_id": identity])
                let document = response["document"].raw
                let data = try JSONSerialization.data(withJSONObject: document, options: [.prettyPrinted, .sortedKeys])
                try data.write(to: url, options: [.atomic])
                localError = nil
            } catch { localError = error.localizedDescription }
        }
    }
}

private extension JSON {
    var worldIdentity: String { self["id"].text }
}
