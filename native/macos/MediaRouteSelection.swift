import Foundation

enum MediaRouteSelection {
    static func id(_ row: JSON) -> String { row["id"].string ?? row["name"].string ?? "" }

    static func available(_ row: JSON) -> Bool {
        if !(row["available"].raw is NSNull) { return row["available"].bool }
        if !(row["ready"].raw is NSNull) { return row["ready"].bool }
        return true // Older external catalogs advertise a route without a readiness field.
    }

    static func supports(_ row: JSON, operation: String?) -> Bool {
        guard let operation else { return true }
        if !row["operations"].array.isEmpty { return row["operations"].array.contains { $0.string == operation } }
        if let advertised = row["operation"].string { return advertised == operation }
        return true
    }

    static func usable(_ rows: [JSON], selection: String, operation: String? = nil, exactDefault: Bool = false) -> Bool {
        let candidates: [JSON]
        if selection != "selected" { candidates = rows.filter { id($0) == selection } }
        else if exactDefault { candidates = rows.filter { $0["selected"].bool } }
        else { candidates = rows }
        return candidates.contains { available($0) && supports($0, operation: operation) }
    }
}
