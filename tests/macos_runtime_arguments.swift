// Actual advertised live tool schemas; no model or accelerator is loaded.
import Foundation

@main private enum RuntimeArgumentChecks {
    static func main() throws {
        let data = try Data(contentsOf: URL(fileURLWithPath: CommandLine.arguments[1]))
        let tools = try JSONSerialization.jsonObject(with: data) as! [String: [String: Any]]
        func property(_ tool: String, _ field: String) -> [String: Any] {
            let schema = tools[tool]!["inputSchema"] as! [String: Any]
            return (schema["properties"] as! [String: [String: Any]])[field]!
        }
        var checks = 0
        for (tool, field) in [("compile_abliteration", "full_rank"), ("plan_expert_residency", "maximum_resident_bytes")] {
            let schema = property(tool, field)
            let integer = try RuntimeArgumentValue.decode("16", schema: schema, name: field)
            precondition(integer as? Int == 16 && !(integer is String))
            precondition(tryIsNull("null", schema: schema))
            precondition(RuntimeArgumentValue.typeHint(schema) == "integer or null (JSON)")
            for invalid in ["1.5", "true", "\"16\"", "{}", "[]"] {
                do { _ = try RuntimeArgumentValue.decode(invalid, schema: schema, name: field); fatalError("Accepted invalid union input: " + invalid) }
                catch { precondition(error.localizedDescription.contains("integer or null")) }
                checks += 1
            }
            let roundTrip = try JSONSerialization.jsonObject(with: JSONSerialization.data(withJSONObject: [field: integer])) as! [String: Any]
            precondition(roundTrip[field] as? Int == 16)
            checks += 4
        }
        let boolean = try RuntimeArgumentValue.decode("false", schema: property("compile_abliteration", "orthogonalize_reference"), name: "reference")
        precondition(boolean as? Bool == false)
        let identifier = try RuntimeArgumentValue.decode("capture-one", schema: property("compile_abliteration", "capture_id"), name: "capture")
        precondition(identifier as? String == "capture-one")
        let object = try RuntimeArgumentValue.decode("{\"layer\":2}", schema: property("compile_expert_mask", "keep_counts"), name: "counts")
        precondition((object as? [String: Int])?["layer"] == 2)
        let failed = RuntimeOperationPresentation(["state": "completed", "receipt": ["state": "trial_failed", "error": "Trial failed"]])
        precondition(failed.requestLabel == "Request completed" && failed.runtimeState == "trial_failed" && failed.errors == ["Trial failed"])
        let recovery = RuntimeOperationPresentation(["state": "completed", "receipt": ["state": "promotion_recovery_required", "trial_worker_error": ["error": "Rollback could not settle"]]])
        precondition(recovery.runtimeState == "promotion_recovery_required" && recovery.errors == ["Rollback could not settle"])
        let structuredError = RuntimeOperationPresentation(["state": "completed", "receipt": ["error": ["message": "Unconfirmed change"]]])
        precondition(structuredError.errors == ["Unconfirmed change"])
        let unknown = RuntimeOperationPresentation(["state": "unknown", "error": "Transport timeout"])
        precondition(unknown.requestLabel == "Request unknown" && unknown.runtimeState == "unconfirmed" && unknown.errors == ["Transport timeout"])
        print("\(checks + 7) native runtime argument/outcome checks passed against live schemas; no checkpoint or GPU")
    }

    static func tryIsNull(_ value: String, schema: [String: Any]) -> Bool {
        (try? RuntimeArgumentValue.decode(value, schema: schema, name: "value")) is NSNull
    }
}
