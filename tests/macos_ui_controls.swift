import Foundation

@main private enum UIControlChecks {
    static var checks = 0
    static func require(_ condition: Bool, _ message: String) throws {
        if !condition { throw DesktopError.message(message) }
        checks += 1
    }
    static func main() throws {
        let existing: [String: Any] = ["id": "engine-one", "path": "/models/model", "route": "ordinary",
            "sticky_port": false, "lora_root": "/adapters", "max_loras": 4, "max_lora_rank": 32,
            "multi_lora_delta_precision": "adapter", "kernel_dev_root": "/kernels",
            "weight_space_config": "/config/repair.json", "microscaling_prefill": "mlp", "future_setting": "preserve"]
        var options = RuntimeLaunchOptions(JSON(existing))
        let roundTrip = try options.applying(to: existing, route: "ordinary")
        try require(NSDictionary(dictionary: roundTrip).isEqual(to: existing), "Editing launch options must preserve identity, defaults and unrelated fields")
        var zeroSlots = options
        zeroSlots.loraSlots = "0"
        let zeroResult = try zeroSlots.applying(to: existing, route: "ordinary")
        try require(RuntimeLaunchOptions(JSON(zeroResult)).loraSlots == "0", "Opening the form must not silently enable disabled adapter slots")
        for invalid in [Double.nan, .infinity, -.infinity, 1e100, -1e100, 1.5] {
            let invalidOptions = RuntimeLaunchOptions(JSON(["lora_root": "/adapters", "max_loras": invalid, "max_lora_rank": invalid]))
            try require(invalidOptions.loraSlots.isEmpty && invalidOptions.loraRank.isEmpty,
                        "Invalid saved numeric budgets must remain editable without integer-conversion traps")
            do { _ = try invalidOptions.applying(to: [:], route: "ordinary"); throw DesktopError.message("Accepted invalid budget") }
            catch { try require(error.localizedDescription.contains("LoRA slots"), "Invalid budgets must require correction before saving") }
        }
        do { _ = try options.applying(to: existing, route: "auto"); throw DesktopError.message("Accepted LoRA with automatic route") }
        catch { try require(error.localizedDescription.contains("Ordinary reference"), "LoRA needs an actionable ordinary-route error") }
        options.loraRoot = " "; options.kernelRoot = ""; options.weightSpaceConfig = ""; options.loraPrecision = ""; options.microscalingPrefill = "off"
        let cleared = try options.applying(to: existing, route: "auto")
        try require(cleared["lora_root"] == nil && cleared["max_loras"] == nil && cleared["kernel_dev_root"] == nil && cleared["weight_space_config"] == nil && cleared["microscaling_prefill"] == nil && cleared["future_setting"] as? String == "preserve", "Clearing options must disable them without dropping unrelated launch settings")
        try require(RuntimeLaunchOptions().microscalingPrefill == "off", "Opening new launch options must preserve the off default")
        for scope in ["experts", "mlp", "all"] {
            options.microscalingPrefill = scope
            let updated = try options.applying(to: ["future_setting": "preserve"], route: "auto")
            try require(RuntimeLaunchOptions(JSON(updated)).microscalingPrefill == scope && updated["future_setting"] as? String == "preserve", "Microscaling scope must round-trip without altering unrelated launch settings")
        }
        options.microscalingPrefill = "invented"
        do { _ = try options.applying(to: [:], route: "auto"); throw DesktopError.message("Accepted invalid microscaling scope") }
        catch { try require(error.localizedDescription.contains("SC-MXFP4"), "Invalid microscaling scope must be refused") }

        let models = [JSON(["id": "default", "selected": true, "available": false, "operations": ["generate"]]),
                      JSON(["id": "text", "available": true, "operations": ["generate"]]),
                      JSON(["id": "editor", "available": true, "operations": ["generate", "edit"]])]
        try require(!MediaRouteSelection.usable(models, selection: "default"), "An available sibling must not enable an unavailable selected model")
        try require(!MediaRouteSelection.usable(models, selection: "selected", exactDefault: true), "Service default is an exact backend, not any available backend")
        try require(!MediaRouteSelection.usable(models, selection: "text", operation: "edit"), "Text-only model must not enable image editing")
        try require(MediaRouteSelection.usable(models, selection: "editor", operation: "edit"), "Available editor must remain usable")
        try require(!MediaRouteSelection.usable(models, selection: "removed"), "Removed or other-service selection must stay unavailable")
        let canonical = [JSON(["id": "edit", "available": true, "operation": "edit"]), JSON(["id": "tts", "ready": false])]
        try require(MediaRouteSelection.usable(canonical, selection: "selected", operation: "edit"), "Canonical default may choose an advertised editor")
        try require(!MediaRouteSelection.usable(canonical, selection: "edit", operation: "generate"), "Canonical edit-only route must not enable text-to-image")
        try require(!MediaRouteSelection.usable(canonical, selection: "tts"), "Explicit ready false must disable generation")
        try require(MediaRouteSelection.available(JSON(["id": "legacy"])), "Legacy catalogs without readiness fields retain their advertised availability")
        try require(!MediaRouteSelection.available(JSON(["available": false, "ready": true])), "Explicit availability refusal takes precedence over readiness")
        print("\(checks) native UI control checks passed; synthetic fixtures, no model checkpoint or GPU")
    }
}
