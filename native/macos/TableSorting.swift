import Foundation

protocol SortableDesktopRow: Identifiable where ID == String {
    var data: JSON { get }
}

/// Compare raw values, never formatted sizes, rates, dates, or placeholder text.
/// Unavailable measurements stay last in either direction.
struct RowSort<Row: SortableDesktopRow>: SortComparator {
    let key: String
    let numeric: Bool
    var order: SortOrder

    init(_ key: String, numeric: Bool = false, order: SortOrder = .forward) {
        self.key = key
        self.numeric = numeric
        self.order = order
    }

    func compare(_ lhs: Row, _ rhs: Row) -> ComparisonResult {
        let result: ComparisonResult
        if numeric {
            let left = lhs.data[key].double.flatMap { $0.isFinite ? $0 : nil }
            let right = rhs.data[key].double.flatMap { $0.isFinite ? $0 : nil }
            switch (left, right) {
            case (nil, nil): return .orderedSame
            case (nil, _): return .orderedDescending
            case (_, nil): return .orderedAscending
            case let (left?, right?):
                result = left == right ? .orderedSame : left < right ? .orderedAscending : .orderedDescending
            }
        } else {
            switch (lhs.data[key].string, rhs.data[key].string) {
            case (nil, nil): return .orderedSame
            case (nil, _): return .orderedDescending
            case (_, nil): return .orderedAscending
            case let (left?, right?): result = left.localizedStandardCompare(right)
            }
        }
        if order == .forward || result == .orderedSame { return result }
        return result == .orderedAscending ? .orderedDescending : .orderedAscending
    }
}

func sortedRows<Row: SortableDesktopRow>(_ rows: [Row], using comparators: [RowSort<Row>]) -> [Row] {
    rows.sorted { left, right in
        for comparator in comparators {
            let result = comparator.compare(left, right)
            if result != .orderedSame { return result == .orderedAscending }
        }
        return left.id.localizedStandardCompare(right.id) == .orderedAscending
    }
}
