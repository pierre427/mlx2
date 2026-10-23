import SwiftUI

struct PerformanceView: View {
    @EnvironmentObject var model: DesktopModel
    @State private var panel = "Inference"
    private let columns = [GridItem(.flexible()), GridItem(.flexible())]
    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 20) {
                HStack {
                    VStack(alignment: .leading, spacing: 5) {
                        Text("Performance").font(.largeTitle.weight(.semibold))
                        Text(model.server["model"].string ?? "Host telemetry · no selected model")
                            .font(.callout).foregroundStyle(.secondary).textSelection(.enabled)
                    }
                    Spacer()
                    VStack(alignment: .trailing, spacing: 5) {
                        Label("Live · 2 s samples", systemImage: "waveform.path").foregroundStyle(.green)
                        Text("Read-only · no inference probes").font(.caption).foregroundStyle(.secondary)
                    }
                }
                HStack(spacing: 12) {
                    MetricCard(title: "Decode · now", value: number(model.server["generation_tps"].double, suffix: " tok/s"))
                    MetricCard(title: "Decode · 30 s", value: number(model.server["generation_tps_window"].double, suffix: " tok/s"))
                    MetricCard(title: "Prefill · 30 s", value: number(model.server["prompt_tps_window"].double, suffix: " tok/s", digits: 0))
                    MetricCard(title: "Requests · 30 s", value: number(model.server["requests_per_second_window"].double, suffix: " /s", digits: 2))
                }
                Picker("Performance category", selection: $panel) {
                    ForEach(["Inference", "Host", "Memory", "Mechanisms"], id: \.self) { Text($0).tag($0) }
                }.pickerStyle(.segmented).frame(maxWidth: 560)
                if let error = model.server["error"].string ?? model.server["status_error"].string {
                    Label(error, systemImage: "exclamationmark.triangle").font(.callout).foregroundStyle(.orange)
                }
                switch panel {
                case "Host": hostPanels
                case "Memory": memoryPanels
                case "Mechanisms": mechanismPanels
                default: inferencePanels
                }
                HStack {
                    Image(systemName: "info.circle")
                    Text("Rates and distributions use the same collectors as mlx2-top. Missing measurements stay unavailable; these observations are not controlled benchmark results.")
                }.font(.caption).foregroundStyle(.secondary)
            }.padding(24)
        }
    }

    private var inferencePanels: some View {
        VStack(spacing: 18) {
            LazyVGrid(columns: columns, spacing: 16) {
                HistoryChart(title: "Generation", key: "generation_tps", unit: "tokens / second")
                HistoryChart(title: "Prefill", key: "prompt_tps", unit: "tokens / second", color: .purple)
                HistoryChart(title: "Time to first token · p50", key: "TTFT_p50_ms", unit: "milliseconds", color: .orange)
                HistoryChart(title: "Inter-token latency · p50", key: "ITL_p50_ms", unit: "milliseconds", color: .teal)
            }
            LatencyTable()
            LazyVGrid(columns: columns, alignment: .leading, spacing: 16) {
                TelemetryCard(title: "Scheduler & batching", symbol: "rectangle.3.group") {
                    UsageBar(title: "Active lanes", value: model.server["active_lanes"].double,
                             maximum: model.server["max_lanes"].double ?? 0,
                             formatted: "\(number(model.server["active_lanes"].double, digits: 0)) / \(number(model.server["max_lanes"].double, digits: 0))")
                    field("Running / queued", "\(number(model.server["running"].double, digits: 0)) / \(number(model.server["waiting"].double, digits: 0))")
                    field("Memory waiting", number(model.server["memory_waiting"].double, digits: 0))
                    field("Batch width · 30 s", number(model.server["batch_width_window"].double, digits: 2))
                    field("Batch width · lifetime", number(model.server["batch_width_lifetime"].double, digits: 2))
                    field("Scheduler cycles", number(model.server["scheduler_cycles_per_second"].double, suffix: " /s"))
                    Text("Physical batch width · batches").font(.caption).foregroundStyle(.secondary)
                    ValueBars(values: pairs(model.server["batch_composition"]), unit: "batches", color: .purple)
                    field("Context / inflight limit", "\(number(model.server["max_context"].double, digits: 0)) / \(number(model.server["max_inflight"].double, digits: 0))")
                }
                TelemetryCard(title: "Token accounting", symbol: "number") {
                    ValueBars(values: [
                        ("Generated", "generation_tokens_total"), ("Prompt", "prompt_tokens_total"),
                        ("Cached prompt", "cached_prompt_tokens_total")
                    ].compactMap { name, key in model.server[key].double.map { NamedValue(name: name, value: $0) } }, unit: "tokens")
                    field("Cached prefill · 30 s", number(model.server["cached_prompt_tps_window"].double, suffix: " tok/s"))
                    field("HTTP requests · 30 s", number(model.server["http_requests_per_second"].double, suffix: " /s", digits: 2))
                    field("Uptime", number(model.server["uptime_seconds"].double, suffix: " s", digits: 0))
                }
            }
            SpeculationPanel()
        }
    }

    private var hostPanels: some View {
        VStack(spacing: 18) {
            LazyVGrid(columns: columns, spacing: 16) {
                HistoryChart(title: "CPU busy", key: "host_cpu", unit: "percent")
                HistoryChart(title: "GPU busy", key: "host_gpu", unit: "percent", color: .purple)
            }
            TelemetryCard(title: "Logical CPU cores", symbol: "cpu") {
                if model.host["cpu_busy_percent"].array.isEmpty {
                    Text("Mach CPU counters unavailable").foregroundStyle(.secondary)
                }
                LazyVGrid(columns: [GridItem(.adaptive(minimum: 150))], spacing: 16) {
                    ForEach(Array(model.host["cpu_busy_percent"].array.enumerated()), id: \.offset) { index, value in
                        UsageBar(title: String(format: "CPU %02d", index), value: value.double)
                    }
                }
                HStack(spacing: 25) {
                    ForEach(Array(model.host["load_average"].array.enumerated()), id: \.offset) { index, value in
                        Text("\([1, 5, 15][index]) min load: \(number(value.double, digits: 2))").font(.caption).foregroundStyle(.secondary)
                    }
                }
            }
            ForEach(Array(model.host["gpus"].array.enumerated()), id: \.offset) { _, gpu in
                TelemetryCard(title: gpu["name"].text, symbol: "gpu") {
                    HStack(alignment: .top, spacing: 30) {
                        VStack(spacing: 12) {
                            UsageBar(title: "Device utilization", value: gpu["busy_percent"].double, color: .purple)
                            UsageBar(title: "Renderer", value: gpu["renderer_percent"].double, color: .purple)
                            UsageBar(title: "Tiler", value: gpu["tiler_percent"].double, color: .purple)
                        }
                        VStack(spacing: 12) {
                            field("Clock", number(gpu["frequency_mhz"].double, suffix: " MHz", digits: 0))
                            field("Power", number(gpu["power_mw"].double.map { $0 / 1000 }, suffix: " W", digits: 2))
                            field("Driver in use", bytes(gpu["memory_bytes"].double))
                            field("Driver allocated", bytes(gpu["allocated_bytes"].double))
                            field("Source", gpu["source"].text)
                        }
                    }
                }
            }
            LazyVGrid(columns: columns, spacing: 16) {
                HistoryChart(title: "GPU power", key: "gpu_power", unit: "watts", color: .purple)
                HistoryChart(title: "CPU power", key: "cpu_power", unit: "watts", color: .blue)
                HistoryChart(title: "ANE power", key: "ane_power", unit: "watts", color: .teal)
                HistoryChart(title: "Maximum die temperature", key: "temperature", unit: "°C", color: .orange)
            }
            TelemetryCard(title: "Power & thermals", symbol: "thermometer.medium") {
                HStack(alignment: .top, spacing: 35) {
                    VStack(spacing: 12) {
                        field("CPU power", watts("cpu_power_mw"))
                        field("GPU power", watts("gpu_power_mw"))
                        field("DRAM power", watts("dram_power_mw"))
                        field("ANE power", watts("ane_power_mw"))
                        field("GPU active residency", number(model.host["soc_power"]["gpu_active_percent"].double, suffix: "%"))
                        field("GPU mean P-state", number(model.host["soc_power"]["gpu_mean_pstate"].double, digits: 2))
                    }
                    VStack(spacing: 12) {
                        field("Thermal state", model.host["thermal_state"].text.capitalized)
                        field("Thermal pressure", model.host["thermal_pressure"].text)
                        field("Die maximum", number(model.host["soc_power"]["die_max_c"].double, suffix: " °C"))
                        field("Die mean", number(model.host["soc_power"]["die_mean_c"].double, suffix: " °C"))
                        field("Collector interval", number(model.host["interval_seconds"].double, suffix: " s", digits: 2))
                    }
                }
                Text("Power and temperature use unprivileged IOReport/HID when available. ANE power is not an ANE utilization percentage.")
                    .font(.caption).foregroundStyle(.secondary)
                if let warning = model.host["soc_power_error"].string { Label(warning, systemImage: "info.circle").font(.caption).foregroundStyle(.secondary) }
            }
        }
    }

    private func watts(_ key: String) -> String {
        number(model.host["soc_power"][key].double.map { $0 / 1000 }, suffix: " W", digits: 2)
    }

    private var memoryPanels: some View {
        VStack(spacing: 18) {
            LazyVGrid(columns: columns, spacing: 16) {
                HistoryChart(title: "Physical memory in use", key: "host_memory", unit: "GiB", color: .purple)
                HistoryChart(title: "Swap allocated", key: "swap_used", unit: "GiB", color: .orange)
            }
            TelemetryCard(title: "Physical memory", symbol: "memorychip") {
                let memory = model.host["memory"]
                let used = memory["total_bytes"].double.flatMap { total in memory["available_bytes"].double.map { total - $0 } }
                UsageBar(title: "Used / total", value: used, maximum: memory["total_bytes"].double ?? 0,
                         formatted: "\(bytes(used)) / \(bytes(memory["total_bytes"].double))", color: .purple)
                LazyVGrid(columns: columns, alignment: .leading, spacing: 12) {
                    ForEach([("Available", "available_bytes"), ("Active", "active_bytes"),
                             ("Inactive", "inactive_bytes"), ("Free", "free_bytes"),
                             ("Wired", "wired_bytes"), ("Compressed", "compressed_bytes"),
                             ("Before compression", "uncompressed_in_compressor_bytes"), ("File backed", "file_backed_bytes"),
                             ("Anonymous", "anonymous_bytes"), ("Purgeable", "purgeable_bytes"),
                             ("Speculative", "speculative_bytes")], id: \.0) { label, key in
                        field(label, bytes(memory[key].double))
                    }
                }
                field("Memory pressure", memory["pressure"].text)
            }
            LazyVGrid(columns: columns, alignment: .leading, spacing: 16) {
                TelemetryCard(title: "Model & Metal memory", symbol: "cube") {
                    ForEach([("Process footprint", "footprint_bytes"), ("Metal active", "metal_active_bytes"),
                             ("Metal peak", "metal_peak_bytes"), ("Admission headroom", "headroom_bytes"),
                             ("Weights", "weights_bytes"), ("Wired limit", "wired_limit_bytes")], id: \.0) { label, key in
                        field(label, bytes(model.server[key].double))
                    }
                }
                TelemetryCard(title: "Swap & VM activity", symbol: "arrow.up.arrow.down") {
                    field("Swap used", bytes(model.host["swap"]["used_bytes"].double))
                    field("Swap total", bytes(model.host["swap"]["total_bytes"].double))
                    field("Swap in", rateBytes(model.host["swap"]["swapins_per_second"].double))
                    field("Swap out", rateBytes(model.host["swap"]["swapouts_per_second"].double))
                    ForEach([("Page in", "pageins_per_second"), ("Page out", "pageouts_per_second"),
                             ("Compress", "compressed_per_second"), ("Decompress", "decompressed_per_second")], id: \.0) { label, key in
                        field(label, number(model.host["memory"][key].double, suffix: " pages/s", digits: 0))
                    }
                }
            }
        }
    }

    private var mechanismPanels: some View {
        VStack(spacing: 18) {
            MechanismsVisual()
            LazyVGrid(columns: columns, alignment: .leading, spacing: 16) {
                TelemetryCard(title: "Request outcomes", symbol: "checkmark.seal") {
                    ValueBars(values: pairs(model.server["outcomes"]), color: .green)
                }
                TelemetryCard(title: "Failures & HTTP errors", symbol: "exclamationmark.shield") {
                    ValueBars(values: pairs(model.server["failures"]), color: .orange)
                    Divider()
                    ValueBars(values: pairs(model.server["http_errors"]), color: .red)
                }
            }
            TelemetryCard(title: "Tenant throughput", symbol: "person.2") {
                ValueBars(values: pairs(model.server["tenant_token_rates"]), unit: "tokens / second", color: .purple)
                field("Jain fairness index", number(model.server["jain_fairness"].double, digits: 3))
            }
        }
    }
}

func rateBytes(_ value: Double?) -> String { value == nil ? "—" : bytes(value) + "/s" }

struct SpeculationPanel: View {
    @EnvironmentObject var model: DesktopModel
    var spec: JSON { model.server["speculation"] }
    var body: some View {
        TelemetryCard(title: "Speculative decoding", symbol: "arrow.triangle.branch") {
            if spec["mode"].string == nil && (spec["receipt_requests"].double ?? 0) == 0 {
                Label("No speculative route activity reported", systemImage: "minus.circle").foregroundStyle(.secondary)
            } else {
                HStack(alignment: .top, spacing: 35) {
                    VStack(spacing: 12) {
                        field("Mode", spec["mode"].text)
                        field("Draft depth", number(spec["num_draft"].double, digits: 0))
                        field("Verify cycles", number(spec["cycles"].double, digits: 0))
                        ValueBars(values: [("All accepted", "accepted_all"), ("Partially accepted", "accepted_partial"),
                                          ("None accepted", "accepted_zero")].compactMap { label, key in
                            spec[key].double.map { NamedValue(name: label, value: $0) }
                        }, unit: "cycles", color: .teal)
                    }
                    VStack(spacing: 12) {
                        let proposed = spec["receipt_proposed"].double
                        let accepted = spec["receipt_accepted"].double
                        UsageBar(title: "Draft acceptance · recent receipts", value: proposed.flatMap { total in total > 0 ? accepted.map { $0 / total * 100 } : nil }, color: .teal)
                        field("Recent requests", number(spec["receipt_requests"].double, digits: 0))
                        field("Proposed / accepted", "\(number(proposed, digits: 0)) / \(number(accepted, digits: 0))")
                        field("Tokens / cycle", ratio(spec["receipt_emitted"].double, spec["receipt_cycles"].double))
                        field("Copy proposed / accepted", "\(number(spec["copy_proposed"].double, digits: 0)) / \(number(spec["copy_accepted"].double, digits: 0))")
                        field("30 s all-accept rate", percentRatio(spec["window_accepted_all"].double, spec["window_cycles"].double))
                    }
                }
                if (spec["progressive_rounds"].double ?? 0) > 0 {
                    Divider()
                    LazyVGrid(columns: [GridItem(.flexible()), GridItem(.flexible())], spacing: 12) {
                        ForEach([("Tile rows", "progressive_tile"), ("Rounds", "progressive_rounds"),
                                 ("Launches", "progressive_launches"), ("Target rows", "progressive_target_rows"),
                                 ("Padding rows", "progressive_padding_rows"), ("Ragged launches", "progressive_ragged_launches"),
                                 ("Rejected", "progressive_rejections"), ("Continued full tiles", "progressive_full_tiles")], id: \.0) { title, key in
                            field(title, number(spec[key].double, digits: 0))
                        }
                    }
                }
            }
        }
    }
}

func ratio(_ numerator: Double?, _ denominator: Double?) -> String {
    guard let numerator, let denominator, denominator > 0 else { return "—" }
    return number(numerator / denominator, digits: 2)
}

func percentRatio(_ numerator: Double?, _ denominator: Double?) -> String {
    guard let numerator, let denominator, denominator > 0 else { return "—" }
    return number(numerator / denominator * 100, suffix: "%")
}
