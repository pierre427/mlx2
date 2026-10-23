import AppKit
import SwiftUI

enum DesktopLaunch {
    static func background(arguments: [String], loginItem: Bool) -> Bool {
        if arguments.contains(where: { ["--chat", "--details", "--performance", "--library"].contains($0) }) { return false }
        return arguments.contains("--background") || loginItem
    }

    static var background: Bool {
        let event = NSAppleEventManager.shared().currentAppleEvent
        let login = event?.paramDescriptor(forKeyword: AEKeyword(keyAEPropData))?.enumCodeValue == OSType(keyAELaunchedAsLogInItem)
        return background(arguments: CommandLine.arguments, loginItem: login)
    }
}

// Track only the main UI window; the menu bar popover must not create a Dock icon.
struct DetailsWindowPresence: NSViewRepresentable {
    func makeNSView(context: Context) -> DockWindowObserver { DockWindowObserver() }
    func updateNSView(_ view: DockWindowObserver, context: Context) {}
}

final class DockWindowObserver: NSView {
    private var observations: [NSObjectProtocol] = []
    private var visibilityObservation: NSKeyValueObservation?
    private let setPolicy: (NSApplication.ActivationPolicy) -> Void
    private let focusWindow: (NSWindow) -> Void

    init(
        setPolicy: @escaping (NSApplication.ActivationPolicy) -> Void = { _ = NSApp.setActivationPolicy($0) },
        focusWindow: @escaping (NSWindow) -> Void = {
            $0.makeKeyAndOrderFront(nil)
            NSApp.activate(ignoringOtherApps: true)
        }
    ) {
        self.setPolicy = setPolicy
        self.focusWindow = focusWindow
        super.init(frame: .zero)
    }

    required init?(coder: NSCoder) { fatalError("init(coder:) has not been implemented") }

    override func viewDidMoveToWindow() {
        super.viewDidMoveToWindow()
        observations.forEach(NotificationCenter.default.removeObserver)
        observations.removeAll()
        visibilityObservation = nil
        guard let window else { return }
        LifecycleActions.detailsWindow = window
        window.styleMask.insert(.resizable)
        window.collectionBehavior.remove([.fullScreenAuxiliary, .fullScreenNone, .auxiliary, .canJoinAllApplications])
        window.collectionBehavior.insert([.fullScreenPrimary, .primary])
        setPolicy(.regular)
        DispatchQueue.main.async { [weak self, weak window] in
            guard let window, window.isVisible else { return }
            self?.focusWindow(window)
        }
        visibilityObservation = window.observe(\.isVisible, options: [.new]) { [weak self] window, _ in
            MainActor.assumeIsolated {
                if window.isVisible { self?.setPolicy(.regular) }
            }
        }
        for name in [NSWindow.didBecomeKeyNotification, NSWindow.didDeminiaturizeNotification] {
            observations.append(NotificationCenter.default.addObserver(
                forName: name, object: window, queue: .main
            ) { [weak self] _ in
                MainActor.assumeIsolated { self?.setPolicy(.regular) }
            })
        }
        observations.append(NotificationCenter.default.addObserver(
            forName: NSWindow.willCloseNotification, object: window, queue: .main
        ) { [weak self] _ in
            MainActor.assumeIsolated { self?.setPolicy(.accessory) }
        })
    }

    deinit { observations.forEach(NotificationCenter.default.removeObserver) }
}
