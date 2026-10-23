import SwiftUI

struct ConversionView: View {
    @EnvironmentObject var model: DesktopModel
    @State private var source = ""
    @State private var recipe = "q4"
    @State private var python = ""
    @State private var tooling = ""
    @State private var reserve = "10"
    @State private var plan = JSON()
    private var state: JSON { model.snapshot["media"]["conversions"] }
    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                Text("Model Conversion").font(.largeTitle.weight(.semibold))
                Text("Convert original BF16/FP16 weights into a separate model. Inputs remain untouched. Converted artifacts begin unqualified.").foregroundStyle(.secondary)
                GroupBox("Conversion") {
                    VStack(alignment: .leading, spacing: 12) {
                        Picker("Source model", selection: $source) {
                            Text("Choose a model").tag("")
                            ForEach(model.rows, id: \.conversionPath) { row in Text(row["name"].text).tag(row["path"].text) }
                        }
                        if model.rows.isEmpty { Text("No models in the library. Download or add an original BF16/FP16 model to convert.").font(.caption).foregroundStyle(.secondary) }
                        Picker("Quantization", selection: $recipe) {
                            ForEach(state["recipes"].array, id: \.conversionID) { row in Text(row["label"].text).tag(row["id"].text) }
                        }
                        Text("oQ measures layer sensitivity and chooses a model-specific mixed precision plan. Its isolated pinned converter needs a compatible Python environment. Uniform Q uses a bounded CPU affine converter; adapter compatibility is verified by subsequent qualification.").font(.caption).foregroundStyle(.secondary)
                        Text("Free-space admission reserves temporary weights and the configured reserve. Existing destinations are refused; partial failures remain outside the model library.").font(.caption).foregroundStyle(.secondary)
                        HStack {
                            Button("Check disk & source") {
                                Task {
                                    do { plan = try await model.request("conversion_plan", ["path": source, "recipe": recipe]) }
                                    catch { plan = JSON(); model.error = error.localizedDescription }
                                }
                            }.disabled(source.isEmpty)
                            Spacer()
                            if state["running"].bool { Button("Cancel") { model.perform("conversion_cancel") } }
                            Button("Convert") { model.perform("conversion_start", ["path": source, "recipe": recipe]) }
                                .buttonStyle(.borderedProminent).disabled(source.isEmpty || state["running"].bool)
                        }
                        if plan["destination"].string != nil {
                            Text("Destination: \(plan["destination"].text)").font(.caption).textSelection(.enabled)
                            Text("Required free: \(bytes(plan["required_free_bytes"].double)) · available: \(bytes(plan["free_bytes"].double)) · estimated output: \(bytes(plan["estimated_output_bytes"].double))").font(.caption)
                        }
                    }.padding(8)
                }
                GroupBox("Conversion runtime") {
                    VStack(alignment: .leading, spacing: 10) {
                        TextField("Python executable", text: $python)
                        TextField("Pinned oMLX source directory", text: $tooling)
                        TextField("Free-space reserve (GiB)", text: $reserve)
                        Text("oMLX revision: \(state["omlx_revision"].text)").font(.system(.caption, design: .monospaced)).textSelection(.enabled)
                        Text("Prepare fetches conversion source only. It installs no serving backend or dependencies. Select a compatible isolated Python environment to run it.").font(.caption).foregroundStyle(.secondary)
                        HStack {
                            Button("Prepare pinned converter") { model.perform("conversion_prepare") }.disabled(state["running"].bool)
                            Spacer()
                            Button("Save runtime") {
                                guard let value = Double(reserve), value.isFinite, value >= 1, value < 1024 else { model.error = "Reserve must be between 1 and 1024 GiB."; return }
                                model.perform("preferences", ["changes": ["conversion_python": python, "conversion_omlx_source": tooling, "conversion_reserve_bytes": Int(value * 1_073_741_824)]])
                            }
                        }
                    }.padding(8)
                }
                GroupBox("Progress & records") {
                    VStack(alignment: .leading, spacing: 12) {
                        if state["jobs"].array.isEmpty { Text("No conversions queued.").foregroundStyle(.secondary) }
                        ForEach(state["jobs"].array, id: \.conversionID) { job in
                            VStack(alignment: .leading, spacing: 6) {
                                HStack { Text(job["spec"]["recipe"].text.uppercased()).font(.headline); Text(job["state"].text).foregroundStyle(.secondary); Spacer() }
                                Text(job["spec"]["path"].text).font(.caption).textSelection(.enabled)
                                if let percent = job["result"]["percent"].double { ProgressView(value: percent, total: 100) }
                                Text(job["progress"].text).font(.caption).textSelection(.enabled)
                                if let path = job["result"]["destination"].string { Text(path).font(.caption).textSelection(.enabled) }
                            }
                            Divider()
                        }
                    }.padding(8)
                }
            }.padding(24)
        }.onChange(of: source) { _, _ in plan = JSON() }
        .onChange(of: recipe) { _, _ in plan = JSON() }
        .onAppear {
            python = state["python"].text
            tooling = state["omlx_source"].text
            reserve = String(format: "%.0f", (state["reserve_bytes"].double ?? 10_737_418_240) / 1_073_741_824)
        }
    }
}
private extension JSON {
    var conversionID: String { self["id"].text }
    var conversionPath: String { self["path"].text }
}
