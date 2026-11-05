import Foundation
import AppKit
import SwiftUI

/// Typed form state at the desktop boundary. Applying an edit preserves fields
/// owned by other launch controls and clears only options the user removed.
struct RuntimeLaunchOptions {
    var loraRoot = ""
    var loraSlots = "1"
    var loraRank = "16"
    var loraPrecision = ""
    var kernelRoot = ""
    var weightSpaceConfig = ""
    var microscalingPrefill = "off"

    init(_ value: JSON = JSON()) {
        loraRoot = value["lora_root"].string ?? ""
        loraSlots = Self.integerText(value["max_loras"], fallback: 1, range: 0...64)
        loraRank = Self.integerText(value["max_lora_rank"], fallback: 16, range: 1...1024)
        loraPrecision = value["multi_lora_delta_precision"].string ?? ""
        kernelRoot = value["kernel_dev_root"].string ?? ""
        weightSpaceConfig = value["weight_space_config"].string ?? ""
        microscalingPrefill = value["microscaling_prefill"].string ?? "off"
    }

    private static func integerText(_ value: JSON, fallback: Int, range: ClosedRange<Int>) -> String {
        if value.raw is NSNull { return String(fallback) }
        guard let number = value.double, number.isFinite, number.rounded() == number,
              number >= Double(range.lowerBound), number <= Double(range.upperBound) else {
            return "" // Keep invalid saved budgets editable instead of trapping or silently clamping.
        }
        return String(Int(number))
    }

    func applying(to original: [String: Any], route: String) throws -> [String: Any] {
        var result = original
        let root = loraRoot.trimmingCharacters(in: .whitespacesAndNewlines)
        for key in ["lora_root", "max_loras", "max_lora_rank", "multi_lora_delta_precision", "kernel_dev_root", "weight_space_config", "microscaling_prefill"] {
            result.removeValue(forKey: key)
        }
        if !root.isEmpty {
            guard route == "ordinary" else { throw DesktopError.message("Select Ordinary reference to use LoRA adapters.") }
            guard let slots = Int(loraSlots), (0...64).contains(slots),
                  let rank = Int(loraRank), (1...1024).contains(rank) else {
                throw DesktopError.message("LoRA slots must be 0–64 and maximum rank 1–1024.")
            }
            result["lora_root"] = root
            result["max_loras"] = slots
            result["max_lora_rank"] = rank
        }
        guard ["", "adapter", "fp32"].contains(loraPrecision) else { throw DesktopError.message("Choose a supported LoRA delta precision.") }
        if !loraPrecision.isEmpty { result["multi_lora_delta_precision"] = loraPrecision }
        let kernels = kernelRoot.trimmingCharacters(in: .whitespacesAndNewlines)
        let weightSpace = weightSpaceConfig.trimmingCharacters(in: .whitespacesAndNewlines)
        if !kernels.isEmpty { result["kernel_dev_root"] = kernels }
        if !weightSpace.isEmpty { result["weight_space_config"] = weightSpace }
        guard ["off", "experts", "mlp", "all"].contains(microscalingPrefill) else {
            throw DesktopError.message("Choose a supported SC-MXFP4 prefill scope.")
        }
        if microscalingPrefill != "off" { result["microscaling_prefill"] = microscalingPrefill }
        return result
    }
}

struct RuntimeLaunchOptionsView: View {
    @Binding var options: RuntimeLaunchOptions
    var body: some View {
        DisclosureGroup("Adapters and runtime tools") {
            VStack(alignment: .leading, spacing: 10) {
                pathField("LoRA adapter directory (optional)", text: $options.loraRoot, directory: true)
                HStack {
                    budgetField("Adapter slots", hint: "0–64", text: $options.loraSlots)
                    budgetField("Maximum LoRA rank", hint: "1–1024", text: $options.loraRank)
                }.disabled(options.loraRoot.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                Picker("LoRA delta precision", selection: $options.loraPrecision) {
                    Text("Runtime default").tag("")
                    Text("Float32").tag("fp32")
                    Text("Adapter precision").tag("adapter")
                }
                Text("LoRA requires the Ordinary reference route and compatible adapters. Start the engine, then use Runtime to register and manage adapters.")
                    .font(.caption).foregroundStyle(.secondary)
                Divider()
                Picker("SC-MXFP4 prefill", selection: $options.microscalingPrefill) {
                    Text("Off (default)").tag("off")
                    Text("Routed experts").tag("experts")
                    Text("Experts and MLP layers").tag("mlp")
                    Text("All eligible projections").tag("all")
                }
                if options.microscalingPrefill != "off" {
                    Text("Requires M5 hardware and a compatible, separately prepared SC-MXFP4 artifact. INT8 prefill activations are approximate; unqualified runs disable APCv2 reuse. Unsupported artifacts or projection combinations are refused at launch.")
                        .font(.caption).foregroundStyle(.secondary)
                    Text("Prepare the separate artifact with scripts/convert_sc_mxfp4.py. The desktop Conversion recipes currently provide affine Q and calibrated oQ formats.")
                        .font(.caption).foregroundStyle(.secondary)
                    if let guide = Bundle.main.url(forResource: "MICROSCALING", withExtension: "md") {
                        Button("Open SC-MXFP4 preparation guide") { NSWorkspace.shared.open(guide) }
                    }
                }
                Divider()
                pathField("Kernel development directory (optional)", text: $options.kernelRoot, directory: true)
                pathField("Weight-space configuration file (optional)", text: $options.weightSpaceConfig, directory: false)
                Text("These opt-in tools apply on the next engine start. Runtime shows supported actions and receipts. A weight-space configuration must be an existing private file; enabling it does not qualify a repair or change a default.")
                    .font(.caption).foregroundStyle(.secondary)
            }.padding(.top, 8)
        }
    }
    private func budgetField(_ title: String, hint: String, text: Binding<String>) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(title).font(.caption)
            TextField(hint, text: text).textFieldStyle(.roundedBorder).accessibilityLabel(title)
        }
    }
    private func pathField(_ title: String, text: Binding<String>, directory: Bool) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(title).font(.caption)
            HStack {
                TextField("", text: text).textFieldStyle(.roundedBorder).accessibilityLabel(title)
                Button("Choose…") {
                    let panel = NSOpenPanel()
                    panel.title = title
                    panel.canChooseDirectories = directory
                    panel.canChooseFiles = !directory
                    panel.allowsMultipleSelection = false
                    if panel.runModal() == .OK, let url = panel.url { text.wrappedValue = url.path }
                }.help(title).accessibilityLabel("Choose " + title)
            }
        }
    }
}
