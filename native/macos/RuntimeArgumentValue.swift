import CoreFoundation
import Foundation

/// Convert form text using the advertised wire type, including nullable unions.
/// Detailed schema constraints remain authoritative in the runtime service.
enum RuntimeArgumentValue {
    static func types(in schema: [String: Any]) -> [String] {
        if let type = schema["type"] as? String { return [type] }
        return schema["type"] as? [String] ?? []
    }

    static func typeHint(_ schema: [String: Any]) -> String {
        let declared = types(in: schema)
        return declared.isEmpty ? "Value" : declared.joined(separator: " or ") + (usesJSON(schema) ? " (JSON)" : "")
    }

    private static func usesJSON(_ schema: [String: Any]) -> Bool {
        types(in: schema).contains { ["number", "integer", "boolean", "object", "array", "null"].contains($0) }
    }

    static func decode(_ text: String, schema: [String: Any], name: String) throws -> Any {
        guard usesJSON(schema) else { return text }
        let value: Any
        do { value = try JSONSerialization.jsonObject(with: Data(text.utf8), options: [.fragmentsAllowed]) }
        catch { throw ValueError(message: name + " must be valid " + typeHint(schema) + ".") }
        let allowed = types(in: schema)
        let matches = allowed.contains { type in
            switch type {
            case "null": return value is NSNull
            case "string": return value is String
            case "object": return value is [String: Any]
            case "array": return value is [Any]
            case "boolean":
                return (value as? NSNumber).map { CFGetTypeID($0) == CFBooleanGetTypeID() } ?? false
            case "number", "integer":
                guard let number = value as? NSNumber, CFGetTypeID(number) != CFBooleanGetTypeID(), number.doubleValue.isFinite else { return false }
                return type == "number" || number.doubleValue.rounded(.towardZero) == number.doubleValue
            default: return false
            }
        }
        guard matches else { throw ValueError(message: name + " must be " + typeHint(schema) + ".") }
        return value
    }

    private struct ValueError: LocalizedError {
        let message: String
        var errorDescription: String? { message }
    }
}
