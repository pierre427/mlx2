import AppKit
import SceneKit
import SwiftUI

struct AgentChoiceRequest: Identifiable {
    let id: String
    let value: JSON
}

@MainActor final class AgentChoices: ObservableObject {
    @Published var pending: AgentChoiceRequest?
    @Published var remaining = 0
    @Published var autoPick = false
    private var continuation: CheckedContinuation<[String: Any], Error>?
    private var timer: Task<Void, Never>?

    func present(_ value: JSON, world: Bool) async throws -> [String: Any] {
        guard pending == nil else { throw DesktopError.message("Another choice panel is already open.") }
        try Task.checkCancellation()
        let timeout = value["policy"]["timeout_seconds"].double ?? 60
        guard timeout.isFinite, timeout >= 1, timeout <= 3600, timeout.rounded(.towardZero) == timeout,
              let panelID = value["panel_id"].string, !panelID.isEmpty, panelID.count <= 64 else {
            throw DesktopError.message("Invalid choice panel identity or timeout.")
        }
        var data = value.raw as? [String: Any] ?? [:]
        if value["presentation"].text == "auto" { data["presentation"] = world ? "object" : "popup" }
        pending = AgentChoiceRequest(id: panelID, value: JSON(data))
        remaining = Int(timeout)
        autoPick = value["policy"]["auto_pick"].bool
        return try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { continuation in
                self.continuation = continuation
                timer = Task { @MainActor [weak self] in
                    while let self, self.pending != nil, !Task.isCancelled {
                        try? await Task.sleep(for: .seconds(1))
                        guard !Task.isCancelled else { return }
                        if self.autoPick {
                            self.remaining = max(0, self.remaining - 1)
                            if self.remaining == 0 {
                                self.resolve(self.pending?.value["recommended_id"].text ?? "", source: "timeout")
                                return
                            }
                        }
                    }
                }
            }
        } onCancel: {
            Task { @MainActor [weak self] in self?.cancel(panelID: panelID) }
        }
    }

    func resolve(_ choice: String, source: String = "user", panelID: String? = nil) {
        guard let request = pending, panelID == nil || panelID == request.id else { return }
        guard ["user", "timeout", "cancelled"].contains(source) else { return }
        if source != "cancelled" && !request.value["choices"].array.contains(where: { $0["id"].text == choice }) { return }
        if source == "timeout" && (!autoPick || remaining > 0 || choice != request.value["recommended_id"].text) { return }
        let reply = continuation
        continuation = nil
        timer?.cancel(); timer = nil
        pending = nil
        reply?.resume(returning: ["panel_id": request.id, "choice_id": choice, "source": source, "is_authorization": false])
    }
    func cancel(panelID: String? = nil) {
        if let panelID, pending?.id != panelID { return }
        resolve("", source: "cancelled")
    }
}

private final class ChoiceScene: SCNView {
    var choose: ((String) -> Void)?
    override func mouseDown(with event: NSEvent) {
        let point = convert(event.locationInWindow, from: nil)
        var node = hitTest(point).first?.node
        while let current = node {
            if let name = current.name, name.hasPrefix("choice:") { choose?(String(name.dropFirst(7))); return }
            node = current.parent
        }
        super.mouseDown(with: event)
    }
}

private struct SpatialChoices: NSViewRepresentable {
    let request: JSON
    let choose: (String) -> Void
    func makeNSView(context: Context) -> ChoiceScene {
        let view = ChoiceScene()
        view.scene = SCNScene()
        view.backgroundColor = NSColor(calibratedRed: 0.025, green: 0.06, blue: 0.1, alpha: 1)
        view.autoenablesDefaultLighting = true
        view.antialiasingMode = .multisampling4X
        let camera = SCNNode(); camera.camera = SCNCamera()
        camera.position = SCNVector3(0, 0, 13)
        view.scene?.rootNode.addChildNode(camera)
        view.pointOfView = camera
        let rows = request["choices"].array
        for (index, row) in rows.enumerated() {
            let shape = SCNBox(width: 7.2, height: 0.8, length: 0.22, chamferRadius: 0.1)
            let recommended = row["id"].text == request["recommended_id"].text
            shape.firstMaterial?.diffuse.contents = recommended ? NSColor.systemTeal : NSColor.darkGray
            shape.firstMaterial?.metalness.contents = 0.5
            shape.firstMaterial?.roughness.contents = 0.35
            let object = SCNNode(geometry: shape)
            object.name = "choice:" + row["id"].text
            object.position = SCNVector3(0, Float(rows.count - 1) * 0.48 - Float(index) * 0.96, 0)
            object.eulerAngles.y = 0.06
            let text = SCNText(string: row["label"].text + (recommended ? "  ★" : ""), extrusionDepth: 0.005)
            text.font = NSFont.systemFont(ofSize: 0.23, weight: .semibold)
            text.firstMaterial?.diffuse.contents = NSColor.white
            let label = SCNNode(geometry: text)
            label.position = SCNVector3(-3.3, -0.13, 0.15)
            object.addChildNode(label)
            view.scene?.rootNode.addChildNode(object)
        }
        return view
    }
    func updateNSView(_ view: ChoiceScene, context: Context) { view.choose = choose }
}

struct AgentChoicePanel: View {
    @EnvironmentObject var model: DesktopModel
    @ObservedObject var choices: AgentChoices
    let request: AgentChoiceRequest
    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text(request.value["title"].text).font(.title2.weight(.semibold))
            Text(request.value["question"].text).textSelection(.enabled)
            if request.value["presentation"].text == "object" {
                SpatialChoices(request: request.value) { choices.resolve($0, panelID: request.id) }.frame(height: CGFloat(max(240, request.value["choices"].array.count * 65)))
            }
            ForEach(Array(request.value["choices"].array.enumerated()), id: \.offset) { _, row in
                Button { choices.resolve(row["id"].text, panelID: request.id) } label: {
                    VStack(alignment: .leading, spacing: 4) {
                        HStack { Text(row["label"].text).fontWeight(.semibold); if row["id"].text == request.value["recommended_id"].text { Text("Recommended").foregroundStyle(.teal) }; Spacer() }
                        if !row["description"].text.isEmpty { Text(row["description"].text).font(.caption).foregroundStyle(.secondary) }
                    }.padding(6).frame(maxWidth: .infinity, alignment: .leading)
                }.buttonStyle(.bordered)
            }
            Toggle("Automatically choose the recommendation", isOn: $choices.autoPick)
                .onChange(of: choices.autoPick) { _, value in model.perform("choice_policy", ["changes": ["auto_pick": value]]) }
            if choices.autoPick { Text("Recommendation in \(choices.remaining) seconds").font(.caption).monospacedDigit() }
            HStack { Text("Choices do not approve tool actions.").font(.caption).foregroundStyle(.secondary); Spacer(); Button("Dismiss") { choices.cancel(panelID: request.id) } }
        }.padding(24).frame(width: 620)
    }
}

struct ChoiceSettingsView: View {
    @EnvironmentObject var model: DesktopModel
    @State private var enabled = true
    @State private var timeout = 60
    var body: some View {
        GroupBox("Agent choice panels") {
            VStack(alignment: .leading) {
                Toggle("Automatically choose the recommended option after a timeout", isOn: $enabled)
                Stepper("Timeout: \(timeout) seconds", value: $timeout, in: 5...3600, step: 5)
                Button("Save choice settings") { model.perform("choice_policy", ["changes": ["auto_pick": enabled, "timeout_seconds": timeout]]) }
            }.padding(8)
        }.onAppear {
            enabled = model.snapshot["choices"]["policy"]["auto_pick"].bool
            timeout = Int(model.snapshot["choices"]["policy"]["timeout_seconds"].double ?? 60)
        }
    }
}

struct AgentChoicePresenter: ViewModifier {
    @EnvironmentObject var model: DesktopModel
    @ObservedObject var choices: AgentChoices
    func body(content: Content) -> some View {
        content.sheet(item: $choices.pending) { request in
            AgentChoicePanel(choices: choices, request: request).environmentObject(model)
                .onDisappear { choices.cancel(panelID: request.id) }
        }
    }
}
