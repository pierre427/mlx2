import Foundation

private struct FixtureRow: SortableDesktopRow {
    let data: JSON
    var id: String { data["id"].text }
    init(_ id: String, _ values: [String: Any]) {
        data = JSON(values.merging(["id": id]) { _, new in new })
    }
}

@main private enum SortingChecks {
    static func require(_ condition: Bool, _ message: String) throws {
        if !condition { throw DesktopError.message(message) }
    }

    static func main() throws {
        let files = [FixtureRow("large", ["bytes": 2_000_000_000_000.0]),
                     FixtureRow("small", ["bytes": 40_000.0]),
                     FixtureRow("unknown", [:]),
                     FixtureRow("medium", ["bytes": 1_570_000_000.0])]
        try require(sortedRows(files, using: [RowSort("bytes", numeric: true)]).map(\.id)
                    == ["small", "medium", "large", "unknown"], "Sizes must sort as bytes")
        try require(sortedRows(files, using: [RowSort("bytes", numeric: true, order: .reverse)]).map(\.id)
                    == ["large", "medium", "small", "unknown"], "Descending sizes must keep unknown last")

        let dates = [FixtureRow("never", [:]), FixtureRow("new", ["date": 1000.0]),
                     FixtureRow("epoch", ["date": 0.0]), FixtureRow("old", ["date": 100.0])]
        try require(sortedRows(dates, using: [RowSort("date", numeric: true)]).map(\.id)
                    == ["epoch", "old", "new", "never"], "Date sorting must use timestamps, including epoch zero")
        try require(sortedRows(dates, using: [RowSort("date", numeric: true, order: .reverse)]).map(\.id)
                    == ["new", "old", "epoch", "never"], "Never must remain last for descending dates")

        let names = [FixtureRow("10", ["name": "Model 10"]), FixtureRow("missing", [:]),
                     FixtureRow("2", ["name": "Model 2"])]
        try require(sortedRows(names, using: [RowSort("name")]).map(\.id) == ["2", "10", "missing"],
                    "Text sorting must use natural model-name ordering")
        try require(sortedRows(names, using: [RowSort("name", order: .reverse)]).map(\.id)
                    == ["10", "2", "missing"], "Text direction must toggle without moving missing values first")

        let rates = [FixtureRow("a", ["rate": 0.2]), FixtureRow("b", ["rate": 0.01]),
                     FixtureRow("invalid", ["rate": Double.nan]), FixtureRow("zero", ["rate": 0])]
        try require(sortedRows(rates, using: [RowSort("rate", numeric: true)]).map(\.id)
                    == ["zero", "b", "a", "invalid"], "Fractional telemetry and unavailable values must sort correctly")

        let ties = [FixtureRow("c", ["status": "Local", "count": 2]),
                    FixtureRow("a", ["status": "Local", "count": 2]),
                    FixtureRow("b", ["status": "Local", "count": 1])]
        try require(sortedRows(ties, using: [RowSort("status"), RowSort("count", numeric: true)]).map(\.id)
                    == ["b", "a", "c"], "Secondary sort and stable identity tie-break must both apply")
        print("8 native table sorting checks passed")
    }
}
