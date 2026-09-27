// Standalone mlx2 research probe, 2026-09-25.
// Build: swiftc -O fp16_partial.swift -o /tmp/mlx2-fp16-partial
import Foundation
import Metal

struct Params { var rows: UInt32; var k: UInt32 }
struct Shape { let label: String; let rows: Int; let k: Int; let batch: Int }
struct CaseData { let name: String; let x: [Float]; let w: [Float] }

func fail(_ message: String) -> Never {
    fputs(message + "\n", stderr)
    exit(1)
}

func option(_ flag: String, _ fallback: String) -> String {
    let args = CommandLine.arguments
    if let index = args.firstIndex(of: flag), index + 1 < args.count {
        return args[index + 1]
    }
    return fallback
}

func median(_ values: [Double]) -> Double {
    let ordered = values.sorted()
    return ordered[ordered.count / 2]
}

// Round Float32 to BF16 using round-to-nearest-even, retaining the result in
// a Float32 container so every GPU arm receives identical BF16 values.
func bf16(_ value: Float) -> Float {
    let bits = value.bitPattern
    let rounded = bits &+ 0x7fff &+ ((bits >> 16) & 1)
    return Float(bitPattern: rounded & 0xffff0000)
}

func hash01(_ value: UInt64) -> Float {
    var z = value &+ 0x9e3779b97f4a7c15
    z = (z ^ (z >> 30)) &* 0xbf58476d1ce4e5b9
    z = (z ^ (z >> 27)) &* 0x94d049bb133111eb
    z ^= z >> 31
    return Float(z & 0x00ffffff) / Float(0x01000000)
}

func makeCase(_ name: String, rows: Int, k: Int) -> CaseData {
    var x = [Float](repeating: 0, count: rows * k)
    var w = [Float](repeating: 0, count: rows * k)
    for row in 0..<rows {
        for j in 0..<k {
            let index = row * k + j
            let u = hash01(UInt64(index) &* 2 &+ 11)
            let v = hash01(UInt64(index) &* 2 &+ 29)
            switch name {
            case "realistic_q4":
                // Activation-like bounded input and affine-Q4-like weights.
                let activation = (u + v + hash01(UInt64(index) &+ 91) - 1.5) * 1.25
                let q = Float(Int(v * 16).clamped(to: 0...15))
                let scale = Float((j / 64) % 15 + 1) / 512.0
                x[index] = bf16(activation)
                w[index] = bf16((q - 7.5) * scale)
            case "random_unit":
                x[index] = bf16(2 * u - 1)
                w[index] = bf16(2 * v - 1)
            case "cancellation":
                // Equal-magnitude terms cancel across each lane stream. A tiny
                // BF16-visible residual prevents an all-zero oracle.
                let lanePosition = j / 32
                x[index] = bf16((lanePosition & 1) == 0 ? 1.0 : -1.0)
                let residual: Float = (j % 67 == 0) ? 0.0078125 : 0.0
                w[index] = bf16(1.0 + residual)
            case "adversarial_small_after_large":
                // Within each lane stream, a large term precedes many values
                // smaller than one FP16 ULP at that partial's magnitude.
                let lanePosition = j / 32
                x[index] = bf16(1.0)
                if lanePosition % 16 == 0 {
                    w[index] = bf16((lanePosition / 16) % 2 == 0 ? 64.0 : -64.0)
                } else {
                    w[index] = bf16((j & 1) == 0 ? 0.0078125 : -0.00390625)
                }
            case "range":
                // Individual products fit FP16, but a same-signed partial can
                // overflow unless it is scaled before FP16 accumulation.
                x[index] = bf16(256.0)
                w[index] = bf16(((j / 32) % 32) < 16 ? 240.0 : -240.0)
            default:
                fail("Unknown data case: \(name)")
            }
        }
    }
    // Canonicalize onto the BF16/FP16 intersection. Near zero, a general BF16
    // value need not align with the FP16 subnormal grid; the extra round keeps
    // this benchmark's product-vs-accumulator isolation honest.
    for index in x.indices {
        x[index] = bf16(Float(Float16(x[index])))
        w[index] = bf16(Float(Float16(w[index])))
        if Float(Float16(x[index])) != x[index] || Float(Float16(w[index])) != w[index] {
            fail("Case \(name) generated a value that is not exactly FP16-representable")
        }
    }
    return CaseData(name: name, x: x, w: w)
}

extension Comparable {
    func clamped(to limits: ClosedRange<Self>) -> Self {
        return min(max(self, limits.lowerBound), limits.upperBound)
    }
}

func errorMetrics(_ actual: [Float], oracle: [Double]) -> [String: Any] {
    var maxAbs = 0.0
    var sumSquared = 0.0
    var referenceSquared = 0.0
    var nonfinite = 0
    for (a, reference) in zip(actual, oracle) {
        referenceSquared += reference * reference
        if !a.isFinite {
            nonfinite += 1
            continue
        }
        let difference = Double(a) - reference
        maxAbs = max(maxAbs, abs(difference))
        sumSquared += difference * difference
    }
    let valid = nonfinite == 0
    return [
        "max_abs": valid ? maxAbs : NSNull(),
        "rmse": valid ? sqrt(sumSquared / Double(max(1, actual.count))) : NSNull(),
        "relative_l2": valid ? sqrt(sumSquared / max(referenceSquared, 1e-300)) : NSNull(),
        "error_metrics_valid": valid,
        "nonfinite_count": nonfinite,
        "checked_outputs": actual.count,
    ]
}

func deltaMetrics(_ actual: [Float], baseline: [Float]) -> [String: Any] {
    var maxAbs = 0.0
    var different = 0
    var nonfinite = 0
    for (a, b) in zip(actual, baseline) {
        if !a.isFinite { nonfinite += 1 }
        if a.bitPattern != b.bitPattern { different += 1 }
        if a.isFinite && b.isFinite { maxAbs = max(maxAbs, abs(Double(a) - Double(b))) }
    }
    return ["max_abs": maxAbs, "bitwise_mismatch_count": different,
            "nonfinite_count": nonfinite, "checked_outputs": actual.count]
}

let sourcePath = option("--source", "scripts/research/m5_next_math/fp16_partial.metal")
let outputPath = option("--output", "docs/research/m5-next-math-20260925/results/fp16-partial.json")
let samples = Int(option("--samples", "9")) ?? 9
guard samples >= 5 else { fail("At least five timing samples are required") }
guard let device = MTLCreateSystemDefaultDevice(),
      let queue = device.makeCommandQueue() else { fail("No Metal GPU") }

let metalSource = try String(contentsOfFile: sourcePath, encoding: .utf8)
let compileOptions = MTLCompileOptions()
compileOptions.mathMode = .safe
let library = try device.makeLibrary(source: metalSource, options: compileOptions)
let arms = ["fp32_fma", "half_partial1_fp32_acc", "half_partial4",
            "half_partial8", "half_partial16", "half_partial32",
            "half_partial16_scaled"]
var pipelines: [String: MTLComputePipelineState] = [:]
for arm in arms {
    guard let function = library.makeFunction(name: arm) else { fail("Missing kernel \(arm)") }
    pipelines[arm] = try device.makeComputePipelineState(function: function)
}
if CommandLine.arguments.contains("--compile-only") {
    print("Compiled \(pipelines.count) kernels; submitted no command buffers.")
    exit(0)
}

let shapes = [
    Shape(label: "q4word_decode", rows: 1, k: 640, batch: 1024),
    Shape(label: "projection_decode_k2048", rows: 1, k: 2048, batch: 512),
    Shape(label: "projection_decode_k4096", rows: 1, k: 4096, batch: 256),
    Shape(label: "q4word_verify8", rows: 8, k: 640, batch: 256),
    Shape(label: "projection_verify8_k2048", rows: 8, k: 2048, batch: 128),
    Shape(label: "projection_verify8_k4096", rows: 8, k: 4096, batch: 64),
    Shape(label: "q4word_prefill128", rows: 128, k: 640, batch: 32),
    Shape(label: "projection_prefill128_k2048", rows: 128, k: 2048, batch: 16),
    Shape(label: "projection_prefill128_k4096", rows: 128, k: 4096, batch: 8),
]
let caseNames = ["realistic_q4", "random_unit", "cancellation",
                 "adversarial_small_after_large", "range"]
var results: [[String: Any]] = []
let started = Date()

for shape in shapes {
    for caseName in caseNames {
        let data = makeCase(caseName, rows: shape.rows, k: shape.k)
        let byteCount = data.x.count * MemoryLayout<Float>.stride
        guard let xBuffer = device.makeBuffer(bytes: data.x, length: byteCount, options: .storageModeShared),
              let wBuffer = device.makeBuffer(bytes: data.w, length: byteCount, options: .storageModeShared),
              let outputBuffer = device.makeBuffer(length: shape.rows * MemoryLayout<Float>.stride,
                                                   options: .storageModeShared) else {
            fail("Buffer allocation failed")
        }
        var params = Params(rows: UInt32(shape.rows), k: UInt32(shape.k))

        func encode(_ encoder: MTLComputeCommandEncoder, arm: String) {
            encoder.setComputePipelineState(pipelines[arm]!)
            encoder.setBuffer(xBuffer, offset: 0, index: 0)
            encoder.setBuffer(wBuffer, offset: 0, index: 1)
            encoder.setBuffer(outputBuffer, offset: 0, index: 2)
            encoder.setBytes(&params, length: MemoryLayout<Params>.stride, index: 3)
            encoder.dispatchThreadgroups(MTLSize(width: shape.rows, height: 1, depth: 1),
                                         threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        }

        func run(_ arm: String, count: Int) -> Double {
            guard let command = queue.makeCommandBuffer(),
                  let encoder = command.makeComputeCommandEncoder() else { fail("Command allocation failed") }
            command.label = "fp16-partial/\(shape.label)/\(caseName)/\(arm)"
            for _ in 0..<count { encode(encoder, arm: arm) }
            encoder.endEncoding()
            command.commit()
            command.waitUntilCompleted()
            if let error = command.error { fail("GPU command failed: \(error)") }
            let elapsed = command.gpuEndTime - command.gpuStartTime
            if elapsed <= 0 { fail("GPU timestamps unavailable") }
            return elapsed * 1e6 / Double(count)
        }

        var oracle = [Double](repeating: 0, count: shape.rows)
        for row in 0..<shape.rows {
            var sum = 0.0
            for j in 0..<shape.k {
                let index = row * shape.k + j
                sum += Double(data.x[index]) * Double(data.w[index])
            }
            oracle[row] = sum
        }

        var outputs: [String: [Float]] = [:]
        var timings: [String: [Double]] = [:]
        for arm in arms {
            _ = run(arm, count: 2)
            let pointer = outputBuffer.contents().bindMemory(to: Float.self, capacity: shape.rows)
            outputs[arm] = Array(UnsafeBufferPointer(start: pointer, count: shape.rows))
            timings[arm] = []
        }

        var warmupSeconds = 0.0
        while warmupSeconds < 0.10 {
            warmupSeconds += run("fp32_fma", count: shape.batch) * Double(shape.batch) / 1e6
        }
        for sample in 0..<samples {
            let shift = sample % arms.count
            var order = Array(arms[shift...]) + Array(arms[..<shift])
            if sample % 2 == 1 { order.reverse() }
            for arm in order { timings[arm]!.append(run(arm, count: shape.batch)) }
        }

        let baselineTiming = median(timings["fp32_fma"]!)
        let baselineOutput = outputs["fp32_fma"]!
        var armResults: [[String: Any]] = []
        for arm in arms {
            let values = timings[arm]!
            let middle = median(values)
            let mad = median(values.map { abs($0 - middle) })
            let pipeline = pipelines[arm]!
            armResults.append([
                "name": arm,
                "gpu_us_median": middle,
                "gpu_us_mad": mad,
                "gpu_us_samples": values,
                "speedup_vs_fp32": baselineTiming / middle,
                "errors_vs_fp64_oracle": errorMetrics(outputs[arm]!, oracle: oracle),
                "delta_vs_fp32_kernel": deltaMetrics(outputs[arm]!, baseline: baselineOutput),
                "thread_execution_width": pipeline.threadExecutionWidth,
                "max_total_threads_per_threadgroup": pipeline.maxTotalThreadsPerThreadgroup,
                "static_threadgroup_memory_bytes": pipeline.staticThreadgroupMemoryLength,
            ])
        }
        results.append([
            "shape": shape.label, "rows": shape.rows, "k": shape.k,
            "data_case": caseName, "batch_dispatches": shape.batch,
            "sustained_warmup_gpu_seconds": warmupSeconds,
            "input_storage": "Float32 containers holding round-to-nearest-even BF16 values",
            "arms": armResults,
        ])
        print("\(shape.label) \(caseName) complete")
        fflush(stdout)
    }
}

let document: [String: Any] = [
    "schema": "mlx2.m5-next-math.fp16-partial.v1",
    "started_at": ISO8601DateFormatter().string(from: started),
    "finished_at": ISO8601DateFormatter().string(from: Date()),
    "device": device.name,
    "registry_id": String(device.registryID),
    "os": ProcessInfo.processInfo.operatingSystemVersionString,
    "samples": samples,
    "compile_fast_math": false,
    "timing": "MTLCommandBuffer GPU end-start divided by repeated dispatch count; conversions, scale selection and dispatch overhead included",
    "ordering": "rotated round-robin arms, reverse on odd samples",
    "source_path": sourcePath,
    "cli": CommandLine.arguments,
    "limitations": [
        "Synthetic dot products, not model or serving qualification.",
        "One simdgroup computes one output; this measures arithmetic choices under fixed lane ownership, not a competitive tiled GEMM.",
        "Inputs are canonicalized to the BF16/FP16 representable intersection and stored in Float32, so conversion cost remains visible but general BF16 conversion error is outside this probe.",
        "FP16 partial arms intentionally alter arithmetic and can overflow or erase cancellation residuals.",
        "The scaled arm includes a conservative per-block maximum scan and exponent-only downscale; it protects range only.",
        "Batched timing repeats identical inputs and allocations, so buffers can be cache-hot; this is an arithmetic mechanism probe, not a bandwidth claim.",
        "No tensor, simdgroup-matrix, NAX, or ANE operations.",
        "Pipeline limits are reported, but register count, spills and occupancy are unavailable through this API.",
    ],
    "results": results,
]
let json = try JSONSerialization.data(withJSONObject: document, options: [.prettyPrinted, .sortedKeys])
let destination = URL(fileURLWithPath: outputPath)
try FileManager.default.createDirectory(at: destination.deletingLastPathComponent(),
                                        withIntermediateDirectories: true)
try json.write(to: destination, options: .atomic)
print("Saved \(outputPath)")
