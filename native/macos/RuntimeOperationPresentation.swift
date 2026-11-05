import Foundation

/// HTTP/admin completion and the runtime's reported outcome are separate facts.
struct RuntimeOperationPresentation {
    let requestLabel: String
    let runtimeState: String?
    let errors: [String]

    init(_ operation: [String: Any]) {
        let state = operation["state"] as? String ?? "unknown"
        requestLabel = "Request " + state.replacingOccurrences(of: "_", with: " ")
        let receipt = operation["receipt"] as? [String: Any] ?? [:]
        runtimeState = receipt["state"] as? String ?? receipt["phase"] as? String
            ?? receipt["status"] as? String ?? (state == "unknown" ? "unconfirmed" : nil)
        func message(_ raw: Any?) -> String? {
            if let text = raw as? String, !text.isEmpty { return text }
            return (raw as? [String: Any])?["message"] as? String
        }
        let worker = receipt["trial_worker_error"] as? [String: Any] ?? [:]
        var seen = Set<String>()
        errors = [message(operation["error"]), message(receipt["error"]),
                  message(worker["error"]), message(receipt["controller_receipt_error"])]
            .compactMap { $0 }.filter { seen.insert($0).inserted }
    }
}
