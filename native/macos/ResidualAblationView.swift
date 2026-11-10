import Charts
import SwiftUI

// MARK: - Strength profile (AblationParameters.as_dict shape)

struct AblationKernel: Equatable {
    var maxWeight = 1.0, maxPosition = 0.0, minWeight = 0.0, minDistance = 1.0
    init(maxWeight: Double = 1, maxPosition: Double = 0, minWeight: Double = 0, minDistance: Double = 1) {
        self.maxWeight = maxWeight; self.maxPosition = maxPosition; self.minWeight = minWeight; self.minDistance = minDistance
    }
    init(_ json: JSON) {
        maxWeight = json["max_weight"].double ?? 0; maxPosition = json["max_weight_position"].double ?? 0
        minWeight = json["min_weight"].double ?? 0; minDistance = json["min_weight_distance"].double ?? 1
    }
    var dict: [String: Any] { ["max_weight": maxWeight, "max_weight_position": maxPosition, "min_weight": minWeight, "min_weight_distance": minDistance] }
}

struct AblationProfile: Equatable {
    var attention = AblationKernel()
    var mlp = AblationKernel()
    var directionIndex: Double?
    init() {}
    init(_ json: JSON) {
        attention = AblationKernel(json["kernels"]["attention"]); mlp = AblationKernel(json["kernels"]["mlp"])
        directionIndex = json["direction_index"].double
    }
    var dict: [String: Any] {
        ["direction_index": directionIndex.map { $0 as Any } ?? NSNull(), "kernels": ["attention": attention.dict, "mlp": mlp.dict]]
    }
    var key: String { "\(attention)|\(mlp)|\(String(describing: directionIndex))" }
    /// Simple mode: scale both components' strengths and move their peaks together.
    func simple(scale: Double, peak: Double) -> AblationProfile {
        var out = self
        let shift = peak - attention.maxPosition
        for path in [\AblationProfile.attention, \AblationProfile.mlp] {
            out[keyPath: path].maxWeight = self[keyPath: path].maxWeight * scale
            out[keyPath: path].minWeight = self[keyPath: path].minWeight * scale
            out[keyPath: path].maxPosition = max(0, self[keyPath: path].maxPosition + shift)
        }
        return out
    }
}

private struct LayerValue: Identifiable { let id = UUID(); let layer: Int; let value: Double; let series: String }
private struct TrialPoint: Identifiable { let id: Int; let refusal: Double; let kl: Double; let pareto: Bool }

private let stepTitles = ["Model & data", "Measure", "Strength", "Search", "Compare", "Bake", "Flagged", "Jobs"]

// MARK: - View

struct ResidualAblationView: View {
    @EnvironmentObject var model: DesktopModel
    @State private var step = "Model & data"
    @State private var modelPath = ""
    @State private var modelInfo = JSON()
    @State private var workspacePath = ""
    @State private var workspace = JSON()
    @State private var loadError: String?
    // strength editor
    @State private var advanced = false
    @State private var basis = AblationProfile()
    @State private var basisLabel = "Default probe"
    @State private var basisTrial: Int?
    @State private var profile = AblationProfile()
    @State private var simpleScale = 1.0
    @State private var simplePeak = 0.0
    @State private var preview = JSON()
    @State private var previewError: String?
    // search
    @State private var selectedTrial: Int?
    @State private var trialDetail = JSON()
    @State private var samplesShown = false
    @State private var trials = 100
    @State private var windowTrials = 50
    @State private var startup = 30
    // judge & decode
    @State private var judge = "d3"
    @State private var judgeOptions = "three"
    @State private var batch = 50
    @State private var maxTokens = 100
    @State private var seed = 20261105
    // compare / bake
    @State private var compareTrials = ""
    @State private var destination = ""
    @State private var destinationCheck = JSON()
    @State private var destinationError: String?
    @State private var rowNormalization = "none"
    @State private var precision = "bf16"
    // service gate (per job)
    @State private var services: [JSON] = []
    @State private var stopService = false
    @State private var serviceLabel = ""
    @State private var serviceHealth = ""
    @State private var serviceConfirmed = false
    // settings
    @State private var evidenceRoot = ""
    @State private var d3Path = ""
    @State private var python = ""
    // jobs & cases
    @State private var logText: String?
    @State private var selectedCase: String?
    @State private var caseDetail = JSON()
    @State private var overlayLabel = ""
    @State private var capability = JSON()
    @State private var autoPreview = false

    private var state: JSON { model.snapshot["residual_ablation"] }
    private var running: Bool { state["running"].bool }
    private var layers: Int { Int(workspace["layers"].double ?? modelInfo["layers"].double ?? 0) }
    private var activeJob: JSON? { state["jobs"].array.first { ["queued", "waiting", "running", "cancelling"].contains($0["state"].text) } }
    private var modified: Bool { basisTrial != nil && profile != basis }

    var body: some View {
        VStack(spacing: 0) {
            header.padding([.horizontal, .top], 20).padding(.bottom, 10)
            Picker("Step", selection: $step) { ForEach(stepTitles, id: \.self) { Text($0).tag($0) } }
                .pickerStyle(.segmented).labelsHidden().padding(.horizontal, 20)
            if let job = activeJob { jobStrip(job).padding(.horizontal, 20).padding(.top, 8) }
            Divider().padding(.top, 10)
            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    if state["runner"].raw is NSNull { Text("Residual ablation is not part of this build.").foregroundStyle(.secondary) }
                    switch step {
                    case "Measure": measureStep
                    case "Strength": strengthStep
                    case "Search": searchStep
                    case "Compare": compareStep
                    case "Bake": bakeStep
                    case "Flagged": flaggedStep
                    case "Jobs": jobsStep
                    default: modelStep
                    }
                }.padding(20).frame(maxWidth: .infinity, alignment: .leading)
            }
        }
        .onAppear(perform: appear)
        .onChange(of: modelPath) { _, _ in Task { await loadModel() } }
        .onChange(of: model.ablationFocusCase) { _, _ in focusCase() }
        .task(id: running) { await pollWhileRunning() }
        .task(id: profile.key + "|\(layers)") { await refreshPreview() }
        .task(id: autoPreview ? profile.key + (selectedCase ?? "") : "") { if autoPreview { try? await Task.sleep(for: .milliseconds(600)); await previewCase() } }
        .onChange(of: simpleScale) { _, _ in if !advanced { profile = basis.simple(scale: simpleScale, peak: simplePeak) } }
        .onChange(of: simplePeak) { _, _ in if !advanced { profile = basis.simple(scale: simpleScale, peak: simplePeak) } }
        .onChange(of: destination) { _, _ in Task { await checkDestination() } }
        .sheet(isPresented: Binding(get: { logText != nil }, set: { if !$0 { logText = nil } })) {
            VStack(alignment: .leading) {
                HStack { Text("Job log").font(.headline); Spacer(); Button("Close") { logText = nil } }
                ScrollView { Text(logText ?? "").font(.caption.monospaced()).textSelection(.enabled).frame(maxWidth: .infinity, alignment: .leading) }
            }.padding(16).frame(width: 760, height: 520)
        }
    }

    // MARK: header & shared pieces

    private var header: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Residual Ablation").font(.largeTitle.weight(.semibold))
            Text("Find the direction a model uses to refuse, test edits that remove it, and write an edited copy. Experimental: every result here is unqualified.")
                .foregroundStyle(.secondary)
            HStack {
                Picker("Model", selection: $modelPath) {
                    Text("Choose a model").tag("")
                    ForEach(model.rows.filter { $0["kind"].string == "Serving candidate" }, id: \.ablationPath) { row in Text(row["name"].text).tag(row["path"].text) }
                }.frame(maxWidth: 420)
                if let checkpoint = modelInfo["checkpoint"].string {
                    Text(checkpoint).font(.caption.monospaced()).foregroundStyle(.secondary).lineLimit(1).truncationMode(.middle).textSelection(.enabled)
                        .help("Artifact name @ fingerprint. Every result is bound to this exact checkpoint.")
                }
            }
            HStack {
                TextField("Evidence folder", text: $workspacePath).textFieldStyle(.roundedBorder)
                Button("Load") { Task { await loadWorkspace() } }.disabled(workspacePath.isEmpty)
            }
            if let loadError { Text(loadError).font(.caption).foregroundStyle(.orange) }
        }
    }

    private func jobStrip(_ job: JSON) -> some View {
        HStack(spacing: 10) {
            ProgressView(value: job["result"]["percent"].double ?? 0, total: 100).frame(width: 120)
                .opacity(job["result"]["percent"].double == nil ? 0.3 : 1)
            Text(job["spec"]["phase"].text.capitalized).font(.caption.weight(.semibold))
            Text(job["progress"].text).font(.caption).lineLimit(1).truncationMode(.tail)
            Spacer()
            Button("Log") { Task { await showLog(job["id"].text) } }.controlSize(.small)
            Button("Cancel") { model.perform("ablation_cancel") }.controlSize(.small).disabled(job["state"].text == "cancelling")
                .help("Stops the window cleanly: the unit is stopped, GPU ownership released and any stopped service restarted")
        }.padding(8).background(.quaternary.opacity(0.5), in: RoundedRectangle(cornerRadius: 8))
    }

    private var serviceGate: some View {
        VStack(alignment: .leading, spacing: 6) {
            Toggle("Stop a model service for this job's GPU window", isOn: $stopService)
            if stopService {
                Picker("Service", selection: $serviceLabel) {
                    Text("Choose a service").tag("")
                    ForEach(services, id: \.ablationLabel) { Text($0["label"].text).tag($0["label"].text) }
                }
                TextField("Health URL checked after restart", text: $serviceHealth).textFieldStyle(.roundedBorder).font(.caption.monospaced())
                if !serviceLabel.isEmpty {
                    Toggle("Stop \(serviceLabel) for this job; the runner restarts it afterwards and waits for it to be healthy", isOn: $serviceConfirmed)
                        .toggleStyle(.checkbox).font(.callout)
                }
            }
        }.padding(10).background(.quaternary.opacity(0.35), in: RoundedRectangle(cornerRadius: 8))
    }

    private var gateOK: Bool { !stopService || (!serviceLabel.isEmpty && serviceConfirmed) }

    private func runButton(_ title: String, phase: String, extra: [String: Any] = [:], disabled: Bool = false) -> some View {
        Button(title) { Task { await start(phase, extra) } }
            .buttonStyle(.borderedProminent)
            .disabled(modelPath.isEmpty || running || !gateOK || disabled)
    }

    private func metricRow(_ name: String, _ value: JSON) -> some View {
        GridRow {
            Text(name)
            Text(percent(value["refusal_rate"].double)).monospacedDigit()
            Text(percent(value["keyword_rate"].double)).monospacedDigit()
            Text(percent(value["judge_b_rate"].double)).monospacedDigit()
            Text(value["kl"].double.map { String(format: "%.4f", $0) } ?? "—").monospacedDigit()
        }
    }

    private func metricHeader() -> some View {
        GridRow {
            Text("Arm").bold()
            Text("Refusal").bold().help("Judged refusal rate on 100 harmful test prompts (keyword hit or judge says it refused/redirected)")
            Text("Keyword").bold().help("Refusal by keyword markers only")
            Text("Judge B").bold().help("Share the judge called a refusal or redirect")
            Text("KL").bold().help("First-token KL divergence from the unmodified model on 100 harmless prompts (lower = less side effect)")
        }
    }

    // MARK: 1 model & data

    private var modelStep: some View {
        VStack(alignment: .leading, spacing: 16) {
            GroupBox("Model") {
                VStack(alignment: .leading, spacing: 6) {
                    if modelPath.isEmpty { Text("Choose a local model above.").foregroundStyle(.secondary) }
                    else {
                        Text(modelPath).font(.caption.monospaced()).textSelection(.enabled)
                        Text("Layers: \(layers > 0 ? String(layers) : "unknown")").font(.caption)
                    }
                    Label("The original model is never modified. Measuring, searching and comparing edit the loaded copy in memory only; Bake writes a new folder and checks the source's file hashes before and after.", systemImage: "lock.shield")
                        .font(.callout).foregroundStyle(.secondary)
                    Text("The model's adapter must declare its residual writers; the runner refuses a model that does not.").font(.caption).foregroundStyle(.secondary)
                }.padding(8).frame(maxWidth: .infinity, alignment: .leading)
            }
            GroupBox("Prompt sets (pinned)") {
                VStack(alignment: .leading, spacing: 6) {
                    Grid(alignment: .leading, horizontalSpacing: 14, verticalSpacing: 4) {
                        ForEach(state["runner"]["prompt_sets"].array, id: \.ablationRole) { row in
                            let data = state["runner"]["datasets"][row["dataset"].text]
                            GridRow {
                                Text(row["role"].text)
                                Text(data["dataset"].text).font(.caption.monospaced())
                                Text(row["split"].text).font(.caption.monospaced())
                                Text(String(data["revision"].text.prefix(10))).font(.caption.monospaced()).foregroundStyle(.secondary)
                                Image(systemName: data["present"].bool ? "checkmark.circle" : "exclamationmark.triangle").foregroundStyle(data["present"].bool ? .green : .orange)
                            }
                        }
                    }
                    Text("Harmful and harmless sets are fixed revisions of the mlabonne datasets the runner pins. Choosing other sets needs a runner option that does not exist yet.").font(.caption).foregroundStyle(.secondary)
                }.padding(8).frame(maxWidth: .infinity, alignment: .leading)
            }
            GroupBox("System prompt") {
                VStack(alignment: .leading, spacing: 6) {
                    Text(state["runner"]["system_prompt"].text).font(.callout.monospaced()).textSelection(.enabled)
                    Text(state["runner"]["system_prompt_flag"].bool ? "The runner accepts a custom system prompt." : "Fixed by the runner (no --system-prompt option yet). The system-prompt A/B in Compare also tests a compliance-guiding prompt.")
                        .font(.caption).foregroundStyle(.secondary)
                }.padding(8).frame(maxWidth: .infinity, alignment: .leading)
            }
            GroupBox("Refusal judge") {
                VStack(alignment: .leading, spacing: 8) {
                    Picker("Judge", selection: $judge) {
                        Text("D3-edge decision model (recommended)").tag("d3")
                        Text("The unmodified model itself").tag("self")
                        Text("D3-edge primed with this model's refusals (d3x)").tag("d3x")
                    }
                    if judge != "self" {
                        Picker("Answer options", selection: $judgeOptions) {
                            Text("Three: complies / refuses / redirects").tag("three")
                            Text("Two: complies / refuses").tag("two")
                        }.help("Three options catch 'redirects' (answering a safer version of the request) as refusals")
                        TextField("D3-edge artifact folder", text: $d3Path).textFieldStyle(.roundedBorder).font(.caption.monospaced())
                    }
                    Text("Keyword markers catch hard refusals first; the judge decides the rest.").font(.caption).foregroundStyle(.secondary)
                }.padding(8).frame(maxWidth: .infinity, alignment: .leading)
            }
            DisclosureGroup("Advanced: decoding, evidence and runtime") {
                VStack(alignment: .leading, spacing: 8) {
                    Stepper("Batch width: \(batch)", value: $batch, in: 1...256).help("Prompts decoded together (1 = one at a time)")
                    Stepper("Tokens per answer: \(maxTokens)", value: $maxTokens, in: 8...1024, step: 8)
                    TextField("Seed", value: $seed, format: .number.grouping(.never)).frame(width: 160)
                    TextField("Evidence root", text: $evidenceRoot).textFieldStyle(.roundedBorder).font(.caption.monospaced())
                    TextField("Python executable (blank = the app's)", text: $python).textFieldStyle(.roundedBorder).font(.caption.monospaced())
                    Button("Save settings") { model.perform("ablation_settings", ["changes": ["evidence_root": evidenceRoot, "d3_path": d3Path, "python": python]]) }
                }.padding(.top, 6)
            }
        }
    }

    // MARK: 2 measure

    private var measureStep: some View {
        let capture = workspace["capture"]
        let separation = capture["separation_d"].array.compactMap(\.double)
        return VStack(alignment: .leading, spacing: 16) {
            Text("Measure compares the model's internal state on harmless and harmful prompts, layer by layer, and records the direction that separates them.").foregroundStyle(.secondary)
            HStack { runButton("Measure (capture)", phase: "capture"); Text("Runs the capture phase in one GPU window.").font(.caption).foregroundStyle(.secondary) }
            serviceGate
            if separation.isEmpty { Text("No completed capture in this evidence folder yet.").foregroundStyle(.secondary) }
            else {
                GroupBox("Refusal signal by layer") {
                    Chart(separation.enumerated().map { LayerValue(layer: $0.offset, value: $0.element, series: "d") }) { row in
                        BarMark(x: .value("Layer", row.layer), y: .value("Separation d", row.value))
                            .foregroundStyle(row.layer == Int(capture["peak_layer"].double ?? -1) ? Color.orange : Color.accentColor)
                    }.frame(height: 200).padding(8)
                    Text("Bar height: separation_d, how far apart harmful and harmless prompts sit along the measured direction (in pooled standard deviations). Peak: layer \(Int(capture["peak_layer"].double ?? 0)).")
                        .font(.caption).foregroundStyle(.secondary).padding(.horizontal, 8)
                }
                GroupBox("Baseline and control") {
                    VStack(alignment: .leading, spacing: 8) {
                        Text("Baseline refusal rate (unmodified): \(percent(capture["baseline"]["refusal_rate"].double))")
                        Grid(alignment: .leading, horizontalSpacing: 16, verticalSpacing: 4) {
                            metricHeader()
                            metricRow("Measured direction", capture["probe_measured"])
                            metricRow("Random direction (control)", capture["probe_random_control"])
                        }
                        Text("Both arms use the same fixed strength profile. A real refusal direction should move refusals; a random one of the same size should not.").font(.caption).foregroundStyle(.secondary)
                        if !capture["baseline_logprobs_present"].bool {
                            Label("This capture's baseline log-probs are not on this Mac; later phases need Measure re-run here.", systemImage: "exclamationmark.triangle").font(.caption).foregroundStyle(.orange)
                        }
                        Text("Capture: \(capture["relative"].text) · \(capture["checkpoint"].text)").font(.caption.monospaced()).foregroundStyle(.secondary).textSelection(.enabled)
                    }.padding(8).frame(maxWidth: .infinity, alignment: .leading)
                }
            }
            DisclosureGroup("Advanced: calibrate the judge (judgecheck)") {
                VStack(alignment: .leading, spacing: 6) {
                    Text("Scores known-harmless answers, the baseline refusals and a probe with each judge; d3x needs its exemplars.").font(.caption).foregroundStyle(.secondary)
                    runButton("Run judge check", phase: "judgecheck", disabled: separation.isEmpty)
                    let check = workspace["judgecheck"]
                    if check["relative"].string != nil {
                        Text("Harmless judged compliant: \(percent(check["harmless_judged_A"].double)) · baseline refusals judged refusing: \(percent(check["baseline_refusals_judged_B"].double))").font(.caption)
                    }
                }.padding(.top, 6)
            }
        }
    }

    // MARK: 3 strength

    private var strengthStep: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack {
                Text(basisTrial.map { "From trial \($0)" } ?? basisLabel).font(.headline)
                if modified { Text("Modified · preview only").font(.caption).padding(4).background(.orange.opacity(0.2), in: Capsule()) }
                Spacer()
                Picker("Mode", selection: $advanced) { Text("Simple").tag(false); Text("Advanced").tag(true) }.pickerStyle(.segmented).frame(width: 200)
                Button("Reset") { profile = basis; simpleScale = 1; simplePeak = basis.attention.maxPosition }
            }
            if modified {
                Text("Compare and Bake run saved trials from the search journal. Edited profiles can be previewed, saved as an overlay, but not yet run by the runner (it needs a --params option).")
                    .font(.caption).foregroundStyle(.secondary)
            }
            if advanced { advancedEditor } else { simpleEditor }
            GroupBox("Strength by layer (λ)") {
                VStack(alignment: .leading, spacing: 6) {
                    if let previewError { Text(previewError).foregroundStyle(.orange) }
                    Chart(curve) { row in
                        LineMark(x: .value("Layer", row.layer), y: .value("λ", row.value)).foregroundStyle(by: .value("Component", row.series))
                        PointMark(x: .value("Layer", row.layer), y: .value("λ", row.value)).foregroundStyle(by: .value("Component", row.series)).symbolSize(16)
                    }.frame(height: 200)
                    .help("λ: how much of the refusal direction is removed from each component's output at that layer (1 = all of it). Computed by the engine's own plan math.")
                    Text("\(Int(preview["edits"].double ?? 0)) projections edited · direction: \(preview["direction"]["scope"].text)\(preview["direction"]["direction_index"].double.map { String(format: " (from layer %.2f)", $0) } ?? "")")
                        .font(.caption).foregroundStyle(.secondary)
                    let separation = workspace["capture"]["separation_d"].array.compactMap(\.double)
                    if !separation.isEmpty {
                        Chart(separation.enumerated().map { LayerValue(layer: $0.offset, value: $0.element, series: "signal") }) {
                            AreaMark(x: .value("Layer", $0.layer), y: .value("Signal", $0.value)).foregroundStyle(.gray.opacity(0.3))
                        }.frame(height: 60).chartYAxis(.hidden)
                        Text("Refusal signal by layer, for reference").font(.caption2).foregroundStyle(.secondary)
                    }
                }.padding(8)
            }
        }
    }

    private var curve: [LayerValue] {
        ["attention", "mlp"].flatMap { component in
            preview["components"][component].array.enumerated().compactMap { index, value in
                value.double.map { LayerValue(layer: index, value: $0, series: component == "mlp" ? "MLP output" : "Attention output") }
            }
        }
    }

    private var simpleEditor: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Text("Strength").frame(width: 90, alignment: .leading)
                Slider(value: $simpleScale, in: 0...2)
                Text(String(format: "×%.2f", simpleScale)).monospacedDigit().frame(width: 60)
            }.help("Scales the whole profile; ×1 is the loaded trial")
            HStack {
                Text("Peak layer").frame(width: 90, alignment: .leading)
                Slider(value: $simplePeak, in: 0...Double(max(1, layers - 1)))
                Text(String(format: "%.1f", simplePeak)).monospacedDigit().frame(width: 60)
            }.help("Where the attention edit is strongest; the MLP peak moves with it")
        }
    }

    private var advancedEditor: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(alignment: .top, spacing: 24) {
                kernelEditor("Attention output", kernel: $profile.attention)
                kernelEditor("MLP output", kernel: $profile.mlp)
            }
            HStack {
                Picker("Direction", selection: Binding(get: { profile.directionIndex == nil ? "per layer" : "global" },
                                                       set: { profile.directionIndex = $0 == "global" ? (profile.directionIndex ?? Double(layers) * 0.6) : nil })) {
                    Text("Per layer (each layer's own)").tag("per layer"); Text("Global (one direction)").tag("global")
                }.frame(width: 340)
                if let index = profile.directionIndex {
                    Slider(value: Binding(get: { index }, set: { profile.directionIndex = $0 }), in: 0...Double(max(1, layers - 1)))
                    Text(String(format: "layer %.2f", index)).monospacedDigit()
                }
            }.help("direction_index: a fractional layer; the global direction interpolates between the two neighbouring layers' directions")
        }
    }

    private func kernelEditor(_ title: String, kernel: Binding<AblationKernel>) -> some View {
        let last = Double(max(1, layers - 1))
        return VStack(alignment: .leading, spacing: 6) {
            Text(title).font(.subheadline.weight(.semibold))
            slider("Peak strength", "max_weight", kernel.maxWeight, -0.25...1.5)
            slider("Peak layer", "max_weight_position", kernel.maxPosition, 0...last)
            slider("Edge strength", "min_weight", kernel.minWeight, -0.25...1.5)
            slider("Reach (layers)", "min_weight_distance", kernel.minDistance, 0...last)
        }.frame(maxWidth: .infinity)
    }

    private func slider(_ name: String, _ technical: String, _ value: Binding<Double>, _ range: ClosedRange<Double>) -> some View {
        HStack {
            Text(name).frame(width: 110, alignment: .leading).help(technical)
            Slider(value: value, in: range)
            Text(String(format: "%.2f", value.wrappedValue)).monospacedDigit().frame(width: 48)
        }
    }

    // MARK: 4 search

    private var points: [TrialPoint] {
        workspace["trials"].array.compactMap { row in
            guard let trial = row["trial"].double, let refusal = row["refusal_rate"].double, let kl = row["kl"].double else { return nil }
            return TrialPoint(id: Int(trial), refusal: refusal, kl: kl, pareto: row["pareto"].bool)
        }
    }

    private var searchStep: some View {
        let search = workspace["search"]
        let done = Int(search["trials_complete"].double ?? 0)
        return VStack(alignment: .leading, spacing: 16) {
            Text("Search tries many strength profiles and keeps the ones that remove the most refusals for the least change to ordinary answers.").foregroundStyle(.secondary)
            GroupBox("Run") {
                VStack(alignment: .leading, spacing: 8) {
                    Stepper("Total trials: \(trials)", value: $trials, in: 10...2000, step: 10)
                    Stepper("Trials per GPU window: \(windowTrials == 0 ? "all remaining" : String(windowTrials))", value: $windowTrials, in: 0...trials, step: 10)
                        .help("Long searches run as bounded windows that resume one study (same journal)")
                    DisclosureGroup("Advanced") {
                        Stepper("Random start-up trials: \(startup)", value: $startup, in: 0...500, step: 5).help("Optuna TPE n_startup_trials")
                    }
                    Text("Judge: \(judge)\(judge == "self" ? "" : " · \(judgeOptions) options") · batch \(batch) · \(maxTokens) tokens (set in Model & data)").font(.caption).foregroundStyle(.secondary)
                    serviceGate
                    HStack {
                        runButton(done > 0 ? "Continue search" : "Start search", phase: "search",
                                  extra: ["trials": trials, "window_trials": windowTrials, "startup": startup], disabled: workspace["capture"]["relative"].string == nil)
                        Text(done > 0 ? "Study has \(done) completed trials." : "Needs a completed Measure.").font(.caption).foregroundStyle(.secondary)
                    }
                }.padding(8).frame(maxWidth: .infinity, alignment: .leading)
            }
            if !points.isEmpty {
                GroupBox("Refusal vs side effect") {
                    VStack(alignment: .leading, spacing: 6) {
                        scatter
                        Text("Each dot is a trial. Lower-left is better. Orange: the Pareto front (no other trial is better on both). Click a dot to load it.")
                            .font(.caption).foregroundStyle(.secondary)
                    }.padding(8)
                }
                if selectedTrial != nil { trialPanel }
                GroupBox("Pareto front") {
                    Grid(alignment: .leading, horizontalSpacing: 16, verticalSpacing: 3) {
                        GridRow { Text("Trial").bold(); Text("Refusal").bold(); Text("Keyword").bold(); Text("KL").bold(); Text("Edits").bold() }
                        ForEach(workspace["trials"].array.filter { $0["pareto"].bool }, id: \.ablationTrial) { row in
                            GridRow {
                                Button("#\(row["trial"].double.map { Int($0) } ?? 0)") { select(Int(row["trial"].double ?? 0)) }.buttonStyle(.link)
                                Text(percent(row["refusal_rate"].double)); Text(percent(row["keyword_rate"].double))
                                Text(String(format: "%.4f", row["kl"].double ?? 0)); Text("\(Int(row["edits"].double ?? 0))")
                            }.font(.callout.monospacedDigit())
                        }
                    }.padding(8)
                }
            }
            ForEach(search["windows"].array, id: \.ablationRelative) { window in
                HStack {
                    Text(window["relative"].text).font(.caption.monospaced())
                    Text("\(Int(window["trials_in_window"].double ?? 0)) trials").font(.caption)
                    Text(window["status"].text).font(.caption).foregroundStyle(window["ok"].bool ? Color.secondary : .orange)
                    if let failure = window["failure"].string { Text(failure).font(.caption).foregroundStyle(.orange).lineLimit(2) }
                }
            }
        }
    }

    private var scatter: some View {
        let front = points.filter(\.pareto).sorted { $0.kl < $1.kl }
        return Chart {
            ForEach(points) { point in
                PointMark(x: .value("KL", point.kl), y: .value("Refusal", point.refusal))
                    .foregroundStyle(point.pareto ? Color.orange : Color.secondary.opacity(0.55))
                    .symbolSize(point.id == selectedTrial ? 160 : point.pareto ? 60 : 30)
            }
            ForEach(front) { point in
                LineMark(x: .value("KL", point.kl), y: .value("Refusal", point.refusal), series: .value("front", "Pareto"))
                    .foregroundStyle(.orange.opacity(0.6)).lineStyle(StrokeStyle(lineWidth: 1, dash: [4, 3]))
            }
        }
        .chartXAxisLabel("KL divergence on harmless prompts (side effect)")
        .chartYAxisLabel("Refusal rate")
        .chartYScale(domain: 0...1)
        .frame(height: 280)
        .chartOverlay { proxy in
            GeometryReader { geometry in
                Rectangle().fill(.clear).contentShape(Rectangle())
                    .onTapGesture { location in
                        guard let frame = proxy.plotFrame else { return }
                        let origin = geometry[frame].origin
                        let tap = CGPoint(x: location.x - origin.x, y: location.y - origin.y)
                        let nearest = points.min { a, b in distance(proxy, a, tap) < distance(proxy, b, tap) }
                        if let nearest, distance(proxy, nearest, tap) < 400 { select(nearest.id) }
                    }
            }
        }
    }

    private func distance(_ proxy: ChartProxy, _ point: TrialPoint, _ tap: CGPoint) -> CGFloat {
        guard let x = proxy.position(forX: point.kl), let y = proxy.position(forY: point.refusal) else { return .infinity }
        return (x - tap.x) * (x - tap.x) + (y - tap.y) * (y - tap.y)
    }

    private var trialPanel: some View {
        GroupBox("Trial \(selectedTrial ?? 0)") {
            VStack(alignment: .leading, spacing: 8) {
                Grid(alignment: .leading, horizontalSpacing: 16, verticalSpacing: 4) {
                    metricHeader()
                    metricRow("This trial", trialDetail)
                    metricRow("Unmodified", workspace["capture"]["baseline"])
                }
                if let labels = trialDetail["labels"].raw as? [String: Any], !labels.isEmpty {
                    Text("How the 100 answers were classified: " + labels.sorted { $0.key < $1.key }.map { "\($0.key) \($0.value)" }.joined(separator: " · ")).font(.caption)
                }
                Text("Keyword-only refusal counts hard refusals; judged refusal also counts soft refusals and redirects.").font(.caption).foregroundStyle(.secondary)
                HStack {
                    Button("Use in Compare") { compareTrials = String(selectedTrial ?? 0); step = "Compare" }
                    Button("Bake this trial") { step = "Bake" }
                    Button("Edit strength") { step = "Strength" }
                }
                DisclosureGroup(isExpanded: $samplesShown) {
                    VStack(alignment: .leading, spacing: 10) {
                        ForEach(Array(trialDetail["samples"].array.enumerated()), id: \.offset) { _, sample in
                            VStack(alignment: .leading, spacing: 3) {
                                Text(sample["prompt"].text).font(.caption.weight(.semibold))
                                Text(sample["response"].text).font(.caption).textSelection(.enabled)
                                Text(sample["keyword"].bool ? "keyword refusal" : sample["label"].string ?? sample["judge"].text).font(.caption2).foregroundStyle(.secondary)
                            }
                            Divider()
                        }
                    }
                } label: {
                    Label("Show sample answers — content note: these are the model's answers to harmful requests", systemImage: "eye.slash").font(.caption)
                }
                .onChange(of: samplesShown) { _, shown in if shown { Task { await loadTrial(samples: true) } } }
            }.padding(8).frame(maxWidth: .infinity, alignment: .leading)
        }
    }

    // MARK: 5 compare

    private var parsedCompareTrials: [Int] {
        compareTrials.split(whereSeparator: { $0 == "," || $0 == " " }).compactMap { Int($0) }
    }

    private var compareStep: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Check a trial two ways: exact runtime hooks against norm-preserving edited weights (fullab), and with a default vs a compliance-guiding system prompt (promptab).").foregroundStyle(.secondary)
            HStack {
                TextField("Trials (e.g. 5, 44, 94)", text: $compareTrials).textFieldStyle(.roundedBorder).frame(width: 220)
                if modified { Text("Runs the saved trial, not your edited profile").font(.caption).foregroundStyle(.orange) }
            }
            serviceGate
            HStack {
                runButton("Hooks vs weights", phase: "fullab", extra: ["trial": parsedCompareTrials], disabled: parsedCompareTrials.isEmpty)
                runButton("System-prompt A/B", phase: "promptab", extra: ["trial": parsedCompareTrials], disabled: parsedCompareTrials.isEmpty)
            }
            ForEach(workspace["fullab"].array.reversed(), id: \.ablationRelative) { run in
                GroupBox("Hooks vs weights · \(run["relative"].text)") {
                    VStack(alignment: .leading, spacing: 6) {
                        runStatus(run)
                        Grid(alignment: .leading, horizontalSpacing: 16, verticalSpacing: 4) {
                            metricHeader()
                            ForEach(run["arms"].array, id: \.ablationTrial) { arm in
                                metricRow("#\(Int(arm["trial"].double ?? 0)) hooks (none)", arm["hooks_none"])
                                metricRow("#\(Int(arm["trial"].double ?? 0)) weights (full)", arm["materialized_full"])
                            }
                        }
                        if let restored = run["restored_kl_first10"].double { Text(String(format: "Restore check KL %.2g (must be ~0)", restored)).font(.caption).foregroundStyle(.secondary) }
                    }.padding(8).frame(maxWidth: .infinity, alignment: .leading)
                }
            }
            ForEach(workspace["promptab"].array.reversed(), id: \.ablationRelative) { run in
                GroupBox("System-prompt A/B · \(run["relative"].text)") {
                    VStack(alignment: .leading, spacing: 6) {
                        runStatus(run)
                        Grid(alignment: .leading, horizontalSpacing: 16, verticalSpacing: 4) {
                            metricHeader()
                            ForEach(run["arms"].array, id: \.ablationArm) { arm in metricRow(arm["arm"].text, arm) }
                        }
                    }.padding(8).frame(maxWidth: .infinity, alignment: .leading)
                }
            }
        }
    }

    private func runStatus(_ run: JSON) -> some View {
        VStack(alignment: .leading, spacing: 3) {
            if !run["ok"].bool { Text(run["failure"].string ?? run["status"].text).font(.caption).foregroundStyle(.orange) }
            if run["invalid"].bool { Label("Marked INVALID in its evidence notes", systemImage: "xmark.octagon").font(.caption).foregroundStyle(.red) }
            ForEach(Array((run["notes"].raw as? [String: String] ?? [:]).sorted { $0.key < $1.key }), id: \.key) { note in
                Text(note.key + ": " + note.value).font(.caption).foregroundStyle(.secondary).lineLimit(3)
            }
            if let service = run["service"].raw as? [String: Any] {
                Text("Service \(service["label"] as? String ?? "?"): stopped \((service["stopped"] as? Bool) == true ? "yes" : "no") · healthy after \((service["healthy_after"] as? Bool) == true ? "yes" : "no")").font(.caption).foregroundStyle(.secondary)
            }
        }
    }

    // MARK: 6 bake

    private var bakeStep: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Bake writes the chosen trial into a NEW model folder. The source is opened read-only and its file hashes are checked before and after.").foregroundStyle(.secondary)
            GroupBox("Bake") {
                VStack(alignment: .leading, spacing: 8) {
                    Text(selectedTrial.map { "Trial \($0)" + (modified ? " (saved parameters, not your edits)" : "") } ?? "Choose a trial in Search first").font(.headline)
                    TextField("New destination folder", text: $destination).textFieldStyle(.roundedBorder).font(.caption.monospaced())
                    if let destinationError { Label(destinationError, systemImage: "xmark.circle").font(.caption).foregroundStyle(.red) }
                    else if destinationCheck["destination"].string != nil {
                        Label("New folder outside the source · needs \(bytes(destinationCheck["required_free_bytes"].double)) free, \(bytes(destinationCheck["free_bytes"].double)) available", systemImage: "checkmark.circle").font(.caption).foregroundStyle(.green)
                    }
                    Picker("Row normalization", selection: $rowNormalization) {
                        Text("None (exact projection, as searched)").tag("none"); Text("Full (norm-preserving)").tag("full")
                    }
                    Picker("Edited weights", selection: $precision) {
                        Text("bf16 (recommended)").tag("bf16"); Text("Re-quantize to source format").tag("requant")
                    }
                    Text("Re-quantizing to 4 bits erased most of the edit on Qwen3.5-4B, so edited projections are stored unquantized by default.").font(.caption).foregroundStyle(.secondary)
                    serviceGate
                    runButton("Bake new model", phase: "bake", extra: ["trial": selectedTrial.map { [$0] } ?? [], "dest": destination,
                              "row_normalization": rowNormalization, "edited_precision": precision],
                              disabled: selectedTrial == nil || destinationError != nil || destinationCheck["destination"].string == nil)
                    Label("The baked model is unqualified.", systemImage: "seal").font(.caption)
                }.padding(8).frame(maxWidth: .infinity, alignment: .leading)
            }
            ForEach(workspace["bakes"].array.reversed(), id: \.ablationRelative) { bake in bakeReceipt(bake) }
        }
    }

    private func bakeReceipt(_ bake: JSON) -> some View {
        GroupBox("Receipt · \(bake["relative"].text)") {
            VStack(alignment: .leading, spacing: 6) {
                runStatus(bake)
                Text(bake["destination"].text).font(.caption.monospaced()).textSelection(.enabled)
                HStack(spacing: 14) {
                    check("Source unchanged during bake", bake["source_unchanged"])
                    check("Source hashes match after load", bake["source_hashes_match_after_load"])
                    Text("\(Int(bake["source_files_hashed"].double ?? 0)) source files hashed").font(.caption)
                    Text("Unqualified").font(.caption.weight(.semibold)).padding(3).background(.orange.opacity(0.2), in: Capsule())
                }
                Text("Trial \(Int(bake["trial"].double ?? -1)) · rows \(bake["row_normalization"].text) · stored \(bake["edited_precision"].text) · \(bake["tensors"].array.count) tensors edited").font(.caption)
                if bake["in_memory"]["refusal_rate"].double != nil {
                    Grid(alignment: .leading, horizontalSpacing: 16, verticalSpacing: 4) {
                        metricHeader()
                        metricRow("In memory", bake["in_memory"])
                        metricRow("Baked copy", bake["baked"])
                    }
                }
                DisclosureGroup("Per-tensor edit vs rounding error") {
                    Grid(alignment: .leading, horizontalSpacing: 14, verticalSpacing: 2) {
                        GridRow { Text("Layer").bold(); Text("Component").bold(); Text("λ").bold(); Text("Edit RMS").bold(); Text("Requant RMS").bold(); Text("Ratio").bold(); Text("Stored").bold() }
                        ForEach(Array(bake["tensors"].array.enumerated()), id: \.offset) { _, row in
                            GridRow {
                                Text("\(Int(row["layer"].double ?? 0))"); Text(row["component"].text)
                                Text(String(format: "%.3f", row["lam"].double ?? 0))
                                Text(String(format: "%.2e", row["edit_rms"].double ?? 0)); Text(String(format: "%.2e", row["requant_rms"].double ?? 0))
                                Text(row["requant_to_edit"].double.map { String(format: "%.2f", $0) } ?? "—")
                                    .foregroundStyle((row["requant_to_edit"].double ?? 0) > 1 ? Color.orange : .primary)
                                Text(row["stored"].text)
                            }.font(.caption.monospacedDigit())
                        }
                    }
                    Text("Ratio above 1: storage rounding is larger than the edit itself, so the edit may not survive.").font(.caption).foregroundStyle(.secondary)
                }
            }.padding(8).frame(maxWidth: .infinity, alignment: .leading)
        }
    }

    private func check(_ title: String, _ value: JSON) -> some View {
        Label(title, systemImage: value.raw is NSNull ? "questionmark.circle" : value.bool ? "checkmark.circle" : "xmark.circle")
            .font(.caption).foregroundStyle(value.raw is NSNull ? Color.secondary : value.bool ? .green : .red)
    }

    // MARK: 7 flagged cases & overlays

    private var flaggedStep: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Replies flagged in Chat. Each is a must-fix test for an edit, not training data: one example is too few to estimate a direction.").foregroundStyle(.secondary)
            if state["cases"].array.isEmpty { Text("No flagged replies. In Chat, use the flag on an assistant reply.").foregroundStyle(.secondary) }
            ForEach(state["cases"].array, id: \.ablationID) { row in
                HStack(alignment: .top) {
                    VStack(alignment: .leading, spacing: 2) {
                        Text(row["reason"].text.capitalized + " · " + row["prompt_excerpt"].text).lineLimit(1)
                        Text(row["checkpoint"].string ?? "checkpoint unknown").font(.caption2.monospaced()).foregroundStyle(.secondary).lineLimit(1)
                    }
                    Spacer()
                    Button(selectedCase == row["id"].string ? "Open" : "Review") { selectedCase = row["id"].string; Task { await loadCase() } }
                }
                .padding(6).background(selectedCase == row["id"].string ? Color.accentColor.opacity(0.1) : .clear, in: RoundedRectangle(cornerRadius: 6))
            }
            if selectedCase != nil, caseDetail["id"].string != nil { caseView }
            GroupBox("Overlays") {
                VStack(alignment: .leading, spacing: 4) {
                    if state["overlays"].array.isEmpty { Text("None saved.").font(.caption).foregroundStyle(.secondary) }
                    ForEach(state["overlays"].array, id: \.ablationID) { overlay in
                        HStack {
                            Text(overlay["label"].text); Text(overlay["id"].text).font(.caption2.monospaced()).foregroundStyle(.secondary)
                            Text("\(Int(overlay["edits"].double ?? 0)) edits").font(.caption)
                            Spacer()
                            Button("Delete") { model.perform("ablation_overlay_delete", ["overlay_id": overlay["id"].text]) }.controlSize(.small)
                        }
                    }
                }.padding(8).frame(maxWidth: .infinity, alignment: .leading)
            }
        }
    }

    private var caseView: some View {
        let latest = caseDetail["previews"].array.last
        let supported = capability["supported"].bool && capability["preview"].bool
        let overlays = state["overlays"].array.filter { $0["fingerprint"].string == caseDetail["fingerprint"].string }
        return GroupBox("Case \(caseDetail["id"].text)") {
            VStack(alignment: .leading, spacing: 10) {
                if !caseDetail["note"].text.isEmpty, caseDetail["note"].string != nil { Text("Wanted: " + caseDetail["note"].text).font(.callout) }
                Text("Prompt: " + (caseDetail["history"].array.last?["text"].text ?? "")).font(.callout).textSelection(.enabled)
                HStack(alignment: .top, spacing: 12) {
                    VStack(alignment: .leading, spacing: 4) {
                        Text("Original reply").font(.caption.weight(.semibold))
                        Text(caseDetail["flagged_text"].text).font(.callout).textSelection(.enabled)
                    }.frame(maxWidth: .infinity, alignment: .topLeading)
                    Divider()
                    VStack(alignment: .leading, spacing: 4) {
                        Text("Under the current edit").font(.caption.weight(.semibold))
                        if let latest {
                            Text(latest["text"].string ?? latest["error"].text).font(.callout).textSelection(.enabled)
                        } else { Text(supported ? "No preview yet." : (capability["reason"].string ?? "Checking server…")).font(.caption).foregroundStyle(.secondary) }
                    }.frame(maxWidth: .infinity, alignment: .topLeading)
                }
                HStack {
                    Button("Preview with current profile") { Task { await previewCase() } }.disabled(!supported)
                    Toggle("Re-preview when the profile changes", isOn: $autoPreview).toggleStyle(.checkbox).disabled(!supported)
                }
                if !supported { Text("Unavailable: " + (capability["reason"].string ?? "the running server has no overlay API")).font(.caption).foregroundStyle(.orange) }
                HStack {
                    TextField("Overlay label", text: $overlayLabel).textFieldStyle(.roundedBorder).frame(width: 220)
                    Button("Apply as overlay") { Task { await createOverlay() } }
                        .disabled(workspace["capture"]["relative"].string == nil)
                        .help("Saves the directions reference + current strength profile, bound to this checkpoint. Nothing is written to the model.")
                }
                ForEach(overlays, id: \.ablationID) { overlay in
                    HStack {
                        Text(overlay["label"].text).font(.callout)
                        Spacer()
                        Button("Replay with overlay") { Task { await replay(overlay) } }.disabled(!capability["supported"].bool)
                    }
                }
                if !capability["supported"].bool && !overlays.isEmpty { Text("Replay unavailable: " + (capability["reason"].string ?? "")).font(.caption).foregroundStyle(.orange) }
                ForEach(caseDetail["replays"].array, id: \.ablationBranch) { replay in
                    Text("Replay branch \(replay["branch_id"].text) · overlay \(replay["overlay_id"].text)").font(.caption.monospaced()).foregroundStyle(.secondary)
                }
                HStack {
                    Button("Add to contrast set") {
                        Task { await run("ablation_case_contrast", ["case_ids": [caseDetail["id"].text], "workspace": workspacePath]) }
                    }.disabled(workspacePath.isEmpty)
                        .help(state["contrast_flag_supported"].bool ? "Adds this prompt to the harmful set of a future Measure" : "Saves the prompt file; the runner needs --extra-bad-prompts before Measure can use it")
                    Spacer()
                    Button("Delete case", role: .destructive) {
                        model.perform("ablation_case_delete", ["case_id": caseDetail["id"].text]); selectedCase = nil; caseDetail = JSON()
                    }
                }
            }.padding(8).frame(maxWidth: .infinity, alignment: .leading)
        }
    }

    // MARK: 8 jobs

    private var jobsStep: some View {
        VStack(alignment: .leading, spacing: 10) {
            if state["jobs"].array.isEmpty { Text("No ablation jobs yet.").foregroundStyle(.secondary) }
            ForEach(state["jobs"].array, id: \.ablationID) { job in
                VStack(alignment: .leading, spacing: 4) {
                    HStack {
                        Text(job["spec"]["phase"].text.capitalized).font(.headline)
                        Text(job["state"].text).foregroundStyle(job["state"].text == "failed" ? Color.red : .secondary)
                        Spacer()
                        Button("Log") { Task { await showLog(job["id"].text) } }.controlSize(.small)
                        if job["spec"]["phase"].text == "search" && ["failed", "cancelled", "interrupted", "completed"].contains(job["state"].text) {
                            Button("Resume window") {
                                Task { await run("ablation_resume", ["job_id": job["id"].text, "window_trials": windowTrials].merging(serviceArguments()) { _, new in new }) }
                            }.controlSize(.small).disabled(running || !gateOK)
                                .help("Runs another window on the same study journal")
                        }
                    }
                    Text(job["progress"].text).font(.caption).textSelection(.enabled)
                    Text(job["spec"]["out"].text).font(.caption2.monospaced()).foregroundStyle(.secondary).textSelection(.enabled)
                    if let service = job["result"]["service"].raw as? [String: Any] {
                        Text("Service \(service["label"] as? String ?? ""): stopped \((service["stopped"] as? Bool) == true ? "yes" : "no") · healthy after \((service["healthy_after"] as? Bool) == true ? "yes" : "no")").font(.caption)
                    }
                    if let warning = job["result"]["service_warning"].string { Text(warning).font(.caption).foregroundStyle(.orange) }
                }
                Divider()
            }
            serviceGate
        }
    }

    // MARK: actions

    private func appear() {
        let settings = state["settings"]
        evidenceRoot = settings["evidence_root"].text; d3Path = settings["d3_path"].string ?? ""; python = settings["python"].string ?? ""
        let defaults = state["defaults"]
        trials = Int(defaults["trials"].double ?? 100); windowTrials = Int(defaults["window_trials"].double ?? 50)
        startup = Int(defaults["startup"].double ?? 30); batch = Int(defaults["batch"].double ?? 50)
        maxTokens = Int(defaults["max_tokens"].double ?? 100); serviceHealth = defaults["service_health"].text
        Task { if let list = try? await model.request("ablation_services") { services = list["services"].array } }
        focusCase()
    }

    private func focusCase() {
        guard let focus = model.ablationFocusCase else { return }
        model.ablationFocusCase = nil
        step = "Flagged"; selectedCase = focus
        Task {
            await loadCase()
            let path = caseDetail["model_path"].text
            if modelPath.isEmpty, model.rows.contains(where: { $0["path"].string == path }) { modelPath = path }
        }
    }

    @MainActor private func loadModel() async {
        guard !modelPath.isEmpty else { return }
        do {
            modelInfo = try await model.request("ablation_model", ["path": modelPath])
            workspacePath = modelInfo["default_workspace"].text
            basis = AblationProfile(modelInfo["default_params"]); basisLabel = "Default probe"; basisTrial = nil
            profile = basis; simpleScale = 1; simplePeak = basis.attention.maxPosition
            destination = ""
            await loadWorkspace()
        } catch { loadError = error.localizedDescription }
    }

    @MainActor private func loadWorkspace() async {
        guard !workspacePath.isEmpty else { return }
        do {
            workspace = try await model.request("ablation_workspace", ["path": workspacePath]); loadError = nil
            if basisTrial == nil, workspace["capture"]["probe_params"]["kernels"].raw is [String: Any] {
                basis = AblationProfile(workspace["capture"]["probe_params"]); profile = basis; simplePeak = basis.attention.maxPosition
            }
            if selectedTrial == nil, let best = workspace["pareto"].array.first?.double { select(Int(best)) }
        } catch {
            workspace = JSON()
            loadError = FileManager.default.fileExists(atPath: workspacePath) ? error.localizedDescription : "No evidence yet: the folder is created by the first job."
        }
    }

    private func select(_ trial: Int) {
        selectedTrial = trial; samplesShown = false
        guard let row = workspace["trials"].array.first(where: { $0["trial"].double == Double(trial) }) else { return }
        basis = AblationProfile(row["params"]); basisTrial = trial; profile = basis
        simpleScale = 1; simplePeak = basis.attention.maxPosition
        if compareTrials.isEmpty { compareTrials = String(trial) }
        if destination.isEmpty, !modelPath.isEmpty {
            destination = (modelPath as NSString).deletingLastPathComponent + "/" + (modelPath as NSString).lastPathComponent + "-ablated-t\(trial)-\(rowNormalization)-\(precision)"
        }
        Task { await loadTrial(samples: false) }
    }

    @MainActor private func loadTrial(samples: Bool) async {
        guard let selectedTrial else { return }
        do { trialDetail = try await model.request("ablation_trial", ["path": workspacePath, "trial": selectedTrial, "samples": samples]) }
        catch { model.error = error.localizedDescription }
    }

    @MainActor private func refreshPreview() async {
        guard layers > 0 else { return }
        try? await Task.sleep(for: .milliseconds(150))
        guard !Task.isCancelled else { return }
        do { preview = try await model.request("ablation_preview", ["params": profile.dict, "layers": layers]); previewError = nil }
        catch { previewError = error.localizedDescription }
    }

    @MainActor private func checkDestination() async {
        destinationCheck = JSON(); destinationError = nil
        guard !destination.isEmpty, !modelPath.isEmpty else { return }
        try? await Task.sleep(for: .milliseconds(300))
        guard !Task.isCancelled else { return }
        do { destinationCheck = try await model.request("ablation_destination", ["model": modelPath, "dest": destination]) }
        catch { destinationError = error.localizedDescription }
    }

    private func serviceArguments() -> [String: Any] {
        guard stopService, !serviceLabel.isEmpty else { return [:] }
        return ["stop_service": serviceLabel, "confirm_stop_service": serviceConfirmed ? serviceLabel : "", "service_health": serviceHealth]
    }

    @MainActor private func start(_ phase: String, _ extra: [String: Any]) async {
        var arguments: [String: Any] = ["phase": phase, "model": modelPath, "workspace": workspacePath, "judge": judge,
                                        "judge_options": judgeOptions, "batch": batch, "max_tokens": maxTokens, "seed": seed]
        if !d3Path.isEmpty { arguments["d3"] = d3Path }
        arguments.merge(serviceArguments()) { _, new in new }
        arguments.merge(extra) { _, new in new }
        await run("ablation_start", arguments)
        serviceConfirmed = false  // confirmation is per job
    }

    @MainActor private func run(_ command: String, _ arguments: [String: Any]) async {
        do { _ = try await model.request(command, arguments); model.error = nil }
        catch { model.error = error.localizedDescription }
        serviceConfirmed = false
        await model.refresh()
    }

    @MainActor private func pollWhileRunning() async {
        while running && !Task.isCancelled {
            if !workspacePath.isEmpty { await loadWorkspace() }
            try? await Task.sleep(for: .seconds(3))
        }
        if !workspacePath.isEmpty { await loadWorkspace() }
    }

    @MainActor private func showLog(_ job: String) async {
        do {
            let log = try await model.request("ablation_log", ["job_id": job])
            logText = "run.log\n" + log["run_log"].text + "\n\nrunner output\n" + log["stdout"].text
        } catch { model.error = error.localizedDescription }
    }

    private var caseServerURL: String? {
        let path = caseDetail["model_path"].string ?? ""
        let server = model.servers.first { $0["endpoint"]["model_path"].string == path && !path.isEmpty }
        return server?["endpoint"]["url"].string
    }

    @MainActor private func loadCase() async {
        guard let selectedCase else { return }
        do {
            caseDetail = try await model.request("ablation_case_get", ["case_id": selectedCase])
            if let url = caseServerURL { capability = try await model.request("ablation_overlay_capability", ["url": url]) }
            else { capability = JSON(["supported": false, "reason": "The flagged chat's model is not running"]) }
        } catch { model.error = error.localizedDescription }
    }

    @MainActor private func previewCase() async {
        guard let url = caseServerURL, let selectedCase else { return }
        await run("ablation_case_preview", ["case_id": selectedCase, "url": url, "params": profile.dict,
                                            "capture": workspace["capture"]["path"].text])
        try? await Task.sleep(for: .seconds(2))
        await loadCase()
    }

    @MainActor private func createOverlay() async {
        let source: [String: Any] = modified || basisTrial == nil ? ["kind": "custom"] : ["kind": "trial", "trial": basisTrial ?? 0]
        await run("ablation_overlay_create", ["capture": workspace["capture"]["path"].text, "params": profile.dict,
                                              "checkpoint": caseDetail["checkpoint"].string ?? modelInfo["checkpoint"].string ?? "",
                                              "label": overlayLabel.isEmpty ? (basisTrial.map { "trial \($0)" + (modified ? " (edited)" : "") } ?? "custom") : overlayLabel,
                                              "source": source])
    }

    /// Load the overlay on the server, fork a NEW branch before the flagged
    /// reply and re-send its user turn there with the overlay active.
    @MainActor private func replay(_ overlay: JSON) async {
        guard let url = caseServerURL else { return }
        let chat = model.chat
        let parent = caseDetail["branch_id"].text
        do {
            _ = try await model.request("ablation_overlay_load", ["url": url, "overlay_id": overlay["id"].text])
            guard let branch = chat.forkForReplay(flagged: caseDetail["flagged_message_id"].text, in: parent,
                                                  overlayID: overlay["id"].text, overlayLabel: overlay["label"].text,
                                                  caseID: caseDetail["id"].text) else {
                model.error = "The flagged chat is not open in this profile, or its history cannot be replayed exactly."; return
            }
            _ = try await model.request("ablation_case_replay", ["case_id": caseDetail["id"].text, "branch_id": branch, "overlay_id": overlay["id"].text])
            chat.comparePair = ChatComparePair(left: parent, right: branch, replayOf: caseDetail["flagged_message_id"].text,
                                               checkpoint: caseDetail["checkpoint"].string ?? "")
            chat.pendingAutoSend = branch
            model.section = "Chat"
        } catch { model.error = error.localizedDescription }
    }

    private func percent(_ value: Double?) -> String { value.map { String(format: "%.0f%%", $0 * 100) } ?? "—" }
    private func bytes(_ value: Double?) -> String { ByteCountFormatter.string(fromByteCount: Int64(value ?? 0), countStyle: .file) }
}

private extension JSON {
    var ablationPath: String { self["path"].text }
    var ablationLabel: String { self["label"].text }
    var ablationRole: String { self["role"].text }
    var ablationRelative: String { self["relative"].text }
    var ablationTrial: Double { self["trial"].double ?? -1 }
    var ablationArm: String { self["arm"].text }
    var ablationID: String { self["id"].text }
    var ablationBranch: String { self["branch_id"].text }
}
