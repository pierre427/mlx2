import AppKit
import Foundation

@MainActor enum LifecycleActions {
    static weak var detailsWindow: NSWindow?
    static var authorizedRestart = false

    static func mainWindow() -> NSWindow? {
        let windows = NSApp.windows.filter { !($0 is NSPanel) && $0.contentView != nil }
        let selected = windows.first { $0.identifier?.rawValue.contains("details") == true }
            ?? windows.first { $0.isVisible && $0.styleMask.contains(.titled) }
            ?? detailsWindow ?? NSApp.keyWindow
        if let selected { detailsWindow = selected }
        return selected
    }

    static func toggleFullscreen(enterOnly: Bool = false, exitOnly: Bool = false) {
        let window = mainWindow()
        let receipt: [String: Any] = ["time": Date().timeIntervalSince1970, "active": NSApp.isActive,
            "selected": window?.identifier?.rawValue ?? "none", "windows": NSApp.windows.map {
                ["title": $0.title, "id": $0.identifier?.rawValue ?? "", "class": String(describing: type(of: $0)),
                 "style": $0.styleMask.rawValue, "collection": $0.collectionBehavior.rawValue,
                 "visible": $0.isVisible, "key": $0.isKeyWindow] as [String: Any]
            }]
        if let folder = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first?.appendingPathComponent("MLX2"),
           let data = try? JSONSerialization.data(withJSONObject: receipt, options: [.sortedKeys]) {
            let path = folder.appendingPathComponent("window-transition.json")
            try? data.write(to: path, options: .atomic)
            try? FileManager.default.setAttributes([.posixPermissions: 0o600], ofItemAtPath: path.path)
        }
        guard let window else { return }
        if enterOnly && window.styleMask.contains(.fullScreen) { return }
        if exitOnly && !window.styleMask.contains(.fullScreen) { return }
        NSApp.setActivationPolicy(.regular)
        window.styleMask.insert(.resizable)
        window.collectionBehavior.remove([.fullScreenAuxiliary, .fullScreenNone, .auxiliary, .canJoinAllApplications])
        window.collectionBehavior.insert([.fullScreenPrimary, .primary])
        window.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
        Task { @MainActor in
            // Key-window activation is delivered on the next AppKit turn.
            try? await Task.sleep(for: .milliseconds(150))
            window.collectionBehavior.remove([.fullScreenAuxiliary, .fullScreenNone, .auxiliary, .canJoinAllApplications])
            window.collectionBehavior.insert([.fullScreenPrimary, .primary])
            window.toggleFullScreen(nil)
        }
    }

    static func schedule(_ response: JSON) {
        let value = response["result"]["terminal_action"].bool ? response["result"] : response
        let action = value["native_action"].string
        guard action == "ui_close" || (action == "ui_restart" && value["restart_prepared"].bool) else { return }
        Task {
            // Let ChatModel publish its terminal tool receipt before the window closes.
            try? await Task.sleep(for: .milliseconds(300))
            if action == "ui_close" { mainWindow()?.performClose(nil) }
            else { authorizedRestart = true; NSApp.terminate(nil) }
        }
    }
}
