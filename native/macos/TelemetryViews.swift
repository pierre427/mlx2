import Charts
import SwiftUI

struct TelemetryCard<Content: View>: View {
    let title: String
    let symbol: String
    @ViewBuilder let content: Content
    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Label(title, systemImage: symbol).font(.headline)
            content
        }.padding(18).frame(maxWidth: .infinity, alignment: .topLeading)
            .background(.quaternary.opacity(0.25), in: RoundedRectangle(cornerRadius: 12))
            .overlay(RoundedRectangle(cornerRadius: 12).stroke(.quaternary, lineWidth: 0.5))
    }
}

struct UsageBar: View {
    let title: String
    let value: Double?
    var maximum: Double = 100
    var formatted: String? = nil
    var color: Color = .blue
    var body: some View {
        VStack(spacing: 6) {
            HStack {
                Text(title).foregroundStyle(.secondary)
                Spacer()
                Text(formatted ?? number(value, suffix: "%")).monospacedDigit()
            }.font(.callout)
            GeometryReader { geometry in
                Capsule().fill(.quaternary)
                if let value, maximum > 0 {
                    Capsule().fill(color).frame(width: geometry.size.width * min(1, max(0, value / maximum)))
                }
            }.frame(height: 6)
        }
        .accessibilityElement(children: .combine)
    }
}

struct ChartSample: Identifiable {
    let id = UUID()
    let time: Date
    let value: Double
    let segment: Int
}

struct HistoryChart: View {
    @EnvironmentObject var model: DesktopModel
    let title: String
    let key: String
    let unit: String
    var color: Color = .blue
    var samples: [ChartSample] {
        var segment = 0
        return model.points.compactMap { point in
            guard let value = point.values[key] else { segment += 1; return nil }
            return ChartSample(time: point.time, value: value, segment: segment)
        }
    }
    var body: some View {
        TelemetryCard(title: title, symbol: "waveform.path.ecg") {
            if samples.isEmpty {
                VStack(spacing: 8) {
                    Image(systemName: "chart.xyaxis.line").font(.title).foregroundStyle(.tertiary)
                    Text("Waiting for available telemetry").font(.caption).foregroundStyle(.secondary)
                }.frame(maxWidth: .infinity).frame(height: 140)
            } else {
                Chart(samples) { sample in
                    LineMark(x: .value("Time", sample.time), y: .value(unit, sample.value),
                             series: .value("Interval", sample.segment))
                        .foregroundStyle(color).lineStyle(StrokeStyle(lineWidth: 1.8))
                }
                .chartLegend(.hidden)
                .chartYScale(domain: .automatic(includesZero: true))
                .chartXAxis { AxisMarks(values: .automatic(desiredCount: 3)) }
                .frame(height: 140)
                Text("\(unit) · up to two minutes · gaps mean unavailable data")
                    .font(.caption2).foregroundStyle(.secondary)
            }
        }
    }
}

struct NamedValue: Identifiable {
    let name: String
    let value: Double
    var id: String { name }
}

func pairs(_ value: JSON) -> [NamedValue] {
    value.array.compactMap { pair in
        guard let name = pair[0].string ?? pair[0].double.map({ number($0, digits: 0) }),
              let value = pair[1].double else { return nil }
        return NamedValue(name: name, value: value)
    }
}

struct ValueBars: View {
    let values: [NamedValue]
    var unit = "events"
    var color: Color = .blue
    var body: some View {
        if values.isEmpty {
            Label("No reported data", systemImage: "minus.circle").font(.callout).foregroundStyle(.secondary)
        } else {
            Chart(values) { value in
                BarMark(x: .value(unit, value.value), y: .value("Category", String(value.name.prefix(44))))
                    .foregroundStyle(color.gradient).cornerRadius(3)
                    .annotation(position: .trailing) { Text(number(value.value, digits: value.value.rounded() == value.value ? 0 : 2)).font(.caption2).monospacedDigit() }
            }.chartXAxis { AxisMarks(values: .automatic(desiredCount: 3)) }
                .frame(height: CGFloat(max(1, values.count)) * 27 + 24)
        }
    }
}

struct LatencyItem: SortableDesktopRow {
    let data: JSON
    var id: String { data["_metric"].text }
}

struct LatencyTable: View {
    @EnvironmentObject var model: DesktopModel
    @State private var sortOrder = [RowSort<LatencyItem>("_metric")]
    var items: [LatencyItem] {
        sortedRows(model.server["latency"].array.map { pair in
            var values = pair[1].raw as? [String: Any] ?? [:]
            values["_metric"] = pair[0].text
            return LatencyItem(data: JSON(values))
        }, using: sortOrder)
    }
    var body: some View {
        TelemetryCard(title: "Latency distribution", symbol: "stopwatch") {
            Table(items, sortOrder: $sortOrder) {
                TableColumn("Metric", sortUsing: RowSort<LatencyItem>("_metric")) { Text($0.id).fontWeight(.medium) }.width(70)
                TableColumn("p50", sortUsing: RowSort<LatencyItem>("p50", numeric: true)) { Text(milliseconds($0.data["p50"].double)).monospacedDigit() }.width(min: 70, ideal: 85)
                TableColumn("p95", sortUsing: RowSort<LatencyItem>("p95", numeric: true)) { Text(milliseconds($0.data["p95"].double)).monospacedDigit() }.width(min: 70, ideal: 85)
                TableColumn("p99", sortUsing: RowSort<LatencyItem>("p99", numeric: true)) { Text(milliseconds($0.data["p99"].double)).monospacedDigit() }.width(min: 70, ideal: 85)
                TableColumn("Average", sortUsing: RowSort<LatencyItem>("average", numeric: true)) { Text(milliseconds($0.data["average"].double)).monospacedDigit() }.width(min: 70, ideal: 85)
                TableColumn("Samples", sortUsing: RowSort<LatencyItem>("count", numeric: true)) { Text(number($0.data["count"].double, digits: 0)).monospacedDigit() }.width(65)
                TableColumn("30s p50", sortUsing: RowSort<LatencyItem>("window_p50", numeric: true)) { Text(milliseconds($0.data["window_p50"].double)).monospacedDigit() }.width(min: 70, ideal: 85)
                TableColumn("30s p95", sortUsing: RowSort<LatencyItem>("window_p95", numeric: true)) { Text(milliseconds($0.data["window_p95"].double)).monospacedDigit() }.width(min: 70, ideal: 85)
            }.frame(height: CGFloat(max(items.count, 1)) * 27 + 40)
            LazyVGrid(columns: [GridItem(.adaptive(minimum: 190), alignment: .leading)], alignment: .leading, spacing: 8) {
                ForEach(Array(model.server["latency"].array.enumerated()), id: \.offset) { _, pair in
                    Label("\(pair[0].text): \(pair[1]["exact"].bool ? "exact samples" : "histogram estimate")",
                          systemImage: pair[1]["exact"].bool ? "checkmark.circle" : "chart.bar")
                }
            }.font(.caption2).foregroundStyle(.secondary)
        }
    }
}

struct LibraryStorageChart: View {
    let rows: [JSON]
    var values: [NamedValue] {
        var groups: [String: Double] = [:]
        for row in rows {
            if let bytes = row["allocated_bytes"].double { groups[row["kind"].text == "Draft" ? "Spec Drafter" : row["kind"].text, default: 0] += bytes }
        }
        return groups.sorted { $0.key < $1.key }.map { NamedValue(name: $0.key, value: $0.value) }
    }
    var body: some View {
        Chart(values) { value in
            BarMark(x: .value("Allocated bytes", value.value), y: .value("Library", "Storage"))
                .foregroundStyle(by: .value("Artifact kind", value.name))
        }.chartXAxis(.hidden).chartYAxis(.hidden).chartLegend(position: .bottom, alignment: .leading)
            .frame(height: 46)
            .accessibilityLabel("Library storage by artifact kind")
    }
}

func milliseconds(_ seconds: Double?) -> String {
    number(seconds.map { $0 * 1000 }, suffix: " ms", digits: 1)
}

struct CapabilityChips: View {
    let values: [JSON]
    let color: Color
    var body: some View {
        if values.isEmpty { Text("None reported").font(.callout).foregroundStyle(.secondary) }
        LazyVGrid(columns: [GridItem(.adaptive(minimum: 130), alignment: .leading)], alignment: .leading, spacing: 8) {
            ForEach(Array(values.enumerated()), id: \.offset) { _, value in
                Label(value.text.replacingOccurrences(of: "_", with: " "), systemImage: "checkmark.circle.fill")
                    .font(.caption).foregroundStyle(color).padding(.horizontal, 10).padding(.vertical, 7)
                    .background(color.opacity(0.09), in: RoundedRectangle(cornerRadius: 7))
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
        }
    }
}

struct MechanismItem: SortableDesktopRow {
    let data: JSON
    var id: String { data["name"].text }
}

struct MechanismsVisual: View {
    @EnvironmentObject var model: DesktopModel
    @State private var sortOrder = [RowSort<MechanismItem>("per_second", numeric: true, order: .reverse)]
    var items: [MechanismItem] { sortedRows(model.server["mechanisms"].array.map { MechanismItem(data: $0) }, using: sortOrder) }
    var body: some View {
        TelemetryCard(title: "Observed mechanisms", symbol: "point.3.connected.trianglepath.dotted") {
            ValueBars(values: Array(items.sorted { ($0.data["per_second"].double ?? 0) > ($1.data["per_second"].double ?? 0) }.prefix(10)).compactMap {
                guard let rate = $0.data["per_second"].double else { return nil }
                return NamedValue(name: $0.id, value: rate)
            }, unit: "events / second", color: .teal)
            if !items.isEmpty {
                Table(items, sortOrder: $sortOrder) {
                    TableColumn("Mechanism", sortUsing: RowSort<MechanismItem>("name")) { Text($0.id).font(.caption).textSelection(.enabled) }.width(min: 260, ideal: 440)
                    TableColumn("Events / s", sortUsing: RowSort<MechanismItem>("per_second", numeric: true)) { Text(number($0.data["per_second"].double, digits: 2)).monospacedDigit() }.width(95)
                    TableColumn("Lifetime events", sortUsing: RowSort<MechanismItem>("total", numeric: true)) { Text(number($0.data["total"].double, digits: 0)).monospacedDigit() }.width(120)
                }.frame(height: min(330, CGFloat(items.count) * 25 + 40))
            }
            Text("These are server-reported mechanism events. NAX-related events are evidence of runtime use, not a system-wide hardware utilization percentage.")
                .font(.caption).foregroundStyle(.secondary)
        }
    }
}

struct RequestItem: SortableDesktopRow {
    let data: JSON
    var id: String { data["request_id"].text }
}

struct RequestTables: View {
    @EnvironmentObject var model: DesktopModel
    @State private var activeSortOrder = [RowSort<RequestItem>("age_seconds", numeric: true, order: .reverse)]
    @State private var recentSortOrder = [RowSort<RequestItem>]()
    var active: [RequestItem] { sortedRows(model.server["active_requests"].array.map { RequestItem(data: $0) }, using: activeSortOrder) }
    var recent: [RequestItem] {
        let rows = model.server["recent_requests"].array.map { RequestItem(data: $0) }
        return recentSortOrder.isEmpty ? rows : sortedRows(rows, using: recentSortOrder)
    }
    var body: some View {
        VStack(spacing: 18) {
            TelemetryCard(title: "In flight", symbol: "arrow.trianglehead.2.clockwise.rotate.90") {
                if active.isEmpty { Label("No active request receipts", systemImage: "checkmark.circle").foregroundStyle(.secondary) }
                else {
                    Table(active, sortOrder: $activeSortOrder) {
                        TableColumn("Request", sortUsing: RowSort<RequestItem>("request_id")) { Text($0.id).lineLimit(1).truncationMode(.middle).help($0.id) }.width(min: 100, ideal: 150)
                        TableColumn("Phase", sortUsing: RowSort<RequestItem>("phase")) { Text($0.data["phase"].text).foregroundStyle(.blue) }.width(85)
                        TableColumn("Age", sortUsing: RowSort<RequestItem>("age_seconds", numeric: true)) { Text(number($0.data["age_seconds"].double, suffix: " s")) }.width(75)
                        TableColumn("Queue", sortUsing: RowSort<RequestItem>("queue_ms", numeric: true)) { Text(number($0.data["queue_ms"].double, suffix: " ms")) }.width(85)
                        TableColumn("TTFT", sortUsing: RowSort<RequestItem>("ttft_ms", numeric: true)) { Text(number($0.data["ttft_ms"].double, suffix: " ms")) }.width(85)
                        TableColumn("Mechanism", sortUsing: RowSort<RequestItem>("mechanism")) { Text($0.data["mechanism"].text) }.width(min: 120, ideal: 180)
                    }.frame(height: min(260, CGFloat(active.count) * 27 + 40))
                }
            }
            TelemetryCard(title: "Recent requests", symbol: "clock.arrow.circlepath") {
                if recent.isEmpty { Label("No recent request receipts", systemImage: "tray").foregroundStyle(.secondary) }
                else {
                    Table(recent, sortOrder: $recentSortOrder) {
                        TableColumn("Request", sortUsing: RowSort<RequestItem>("request_id")) { Text($0.id).lineLimit(1).truncationMode(.middle).help($0.id) }.width(min: 90, ideal: 130)
                        TableColumn("Status", sortUsing: RowSort<RequestItem>("status")) { Text($0.data["status"].text) }.width(80)
                        TableColumn("Prompt", sortUsing: RowSort<RequestItem>("prompt_tokens", numeric: true)) { Text(number($0.data["prompt_tokens"].double, digits: 0)) }.width(65)
                        TableColumn("Cached", sortUsing: RowSort<RequestItem>("cached_tokens", numeric: true)) { Text(number($0.data["cached_tokens"].double, digits: 0)) }.width(65)
                        TableColumn("Output", sortUsing: RowSort<RequestItem>("completion_tokens", numeric: true)) { Text(number($0.data["completion_tokens"].double, digits: 0)) }.width(65)
                        TableColumn("TTFT", sortUsing: RowSort<RequestItem>("ttft_seconds", numeric: true)) { Text(milliseconds($0.data["ttft_seconds"].double)) }.width(85)
                        TableColumn("Elapsed", sortUsing: RowSort<RequestItem>("elapsed_seconds", numeric: true)) { Text(number($0.data["elapsed_seconds"].double, suffix: " s")) }.width(80)
                        TableColumn("Route", sortUsing: RowSort<RequestItem>("route")) { Text($0.data["route"].text) }.width(min: 100, ideal: 140)
                    }.frame(height: min(310, CGFloat(recent.count) * 27 + 40))
                }
            }
        }
    }
}
