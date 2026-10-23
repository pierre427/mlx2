import AppKit
import Darwin
import SwiftUI

@MainActor final class AppDelegate: NSObject, NSApplicationDelegate {
    var model: DesktopModel? = DesktopModel.shared
    var openDetails: (() -> Void)?
    private var quitting = false

    func applicationDidFinishLaunching(_ notification: Notification) {
        model?.begin()
        if !DesktopLaunch.background {
            NSApp.setActivationPolicy(.regular)
            NSApp.activate(ignoringOtherApps: true)
        }
    }

    func applicationShouldHandleReopen(_ sender: NSApplication, hasVisibleWindows flag: Bool) -> Bool {
        guard !quitting else { return false }
        NSApp.setActivationPolicy(.regular)
        openDetails?()
        if let window = LifecycleActions.mainWindow() {
            if window.isMiniaturized { window.deminiaturize(nil) }
            window.makeKeyAndOrderFront(nil)
        }
        NSApp.activate(ignoringOtherApps: true)
        return false
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        false
    }

    func applicationShouldTerminate(_ sender: NSApplication) -> NSApplication.TerminateReply {
        guard !quitting else { return .terminateLater }
        if model?.managedRunning == true && !LifecycleActions.authorizedRestart {
            let alert = NSAlert()
            alert.messageText = "Stop the managed model and quit MLX2?"
            alert.informativeText = "Accepted requests will drain before the model stops."
            alert.addButton(withTitle: "Stop and Quit")
            alert.addButton(withTitle: "Cancel")
            if alert.runModal() != .alertFirstButtonReturn { return .terminateCancel }
        }
        quitting = true
        Task {
            let saved = await model?.shutdown() ?? true
            if !saved { quitting = false }
            sender.reply(toApplicationShouldTerminate: saved)
        }
        return .terminateLater
    }
}

private struct WindowReopenRegistration: View {
    @Environment(\.openWindow) private var openWindow
    let delegate: AppDelegate
    var body: some View {
        Color.clear.onAppear { delegate.openDetails = { openWindow(id: "details") } }
    }
}

struct MLX2App: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) private var delegate
    @StateObject private var model = DesktopModel.shared

    var body: some Scene {
        MenuBarExtra {
            PickerView().environmentObject(model).onAppear { delegate.model = model }
        } label: {
            Image(systemName: "cube.transparent")
                .accessibilityLabel("MLX2 model picker")
                .background(WindowReopenRegistration(delegate: delegate))
        }.menuBarExtraStyle(.window)
        Window("MLX2", id: "details") {
            DetailsView().environmentObject(model)
                .windowFullScreenBehavior(.enabled)
                .background(DetailsWindowPresence())
                .background(WindowReopenRegistration(delegate: delegate))
                .onAppear { delegate.model = model }
        }.windowManagerRole(.principal)
            .defaultSize(width: 1120, height: 760)
            .defaultLaunchBehavior(DesktopLaunch.background ? .suppressed : .presented)
            .commands { CommandGroup(replacing: .newItem) {} }
    }
}

@main enum Launcher {
    @MainActor static func main() {
        signal(SIGPIPE, SIG_IGN)
        if CommandLine.arguments.contains("--login-status") || CommandLine.arguments.contains("--enable-login") {
            do {
                if CommandLine.arguments.contains("--enable-login") { try LoginItems.setEnabled(true) }
                print(LoginItems.status)
                exit(0)
            } catch {
                FileHandle.standardError.write(Data((error.localizedDescription + "\n").utf8))
                exit(1)
            }
        }
        if CommandLine.arguments.contains("--chat") { DesktopModel.shared.section = "Chat" }
        if CommandLine.arguments.contains("--performance") { DesktopModel.shared.section = "Performance" }
        if CommandLine.arguments.contains("--library") { DesktopModel.shared.section = "Model Library" }
        if CommandLine.arguments.count > 1 && CommandLine.arguments[1] == "--trash" {
            do {
                guard CommandLine.arguments.count == 4 else { throw DesktopError.message("Expected folder and identity") }
                let path = CommandLine.arguments[2]
                var info = stat()
                guard lstat(path, &info) == 0, (info.st_mode & S_IFMT) == S_IFDIR else {
                    throw DesktopError.message("Folder no longer exists or is a link")
                }
                let ns = Int64(info.st_ctimespec.tv_sec) * 1_000_000_000 + Int64(info.st_ctimespec.tv_nsec)
                let identity = "\(info.st_dev):\(info.st_ino):\(ns)"
                guard identity == CommandLine.arguments[3] else { throw DesktopError.message("Folder changed before Trash operation") }
                var destination: NSURL?
                try FileManager.default.trashItem(at: URL(fileURLWithPath: path), resultingItemURL: &destination)
                let result = ["trashed": path, "destination": destination?.path ?? ""]
                let data = try JSONSerialization.data(withJSONObject: result)
                FileHandle.standardOutput.write(data + Data([10]))
            } catch {
                FileHandle.standardError.write(Data((error.localizedDescription + "\n").utf8))
                exit(1)
            }
            exit(0)
        }
        MLX2App.main()
    }
}
