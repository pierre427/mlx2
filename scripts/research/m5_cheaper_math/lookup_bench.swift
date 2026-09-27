import Foundation
import Metal
import Darwin

struct Params { var n: UInt32; var op: UInt32; var mode: UInt32; var tableSize: UInt32 }
func emit(_ value: [String: Any]) {
    let data = try! JSONSerialization.data(withJSONObject: value, options: [.sortedKeys])
    print(String(data: data, encoding: .utf8)!)
    fflush(stdout)
}
func median(_ a: [Double]) -> Double { a.sorted()[a.count / 2] }
func check(_ cb: MTLCommandBuffer) {
    cb.waitUntilCompleted()
    if let e = cb.error { fatalError("GPU error: \(e)") }
}
let args = CommandLine.arguments
guard args.count == 3, args[2] == "--owned-run" else {
    fatalError("Usage: lookup_bench source.metal --owned-run (under owned_exec.py and cpg_job)")
}
guard let d = MTLCreateSystemDefaultDevice(), let q = d.makeCommandQueue() else { fatalError("Metal unavailable") }
let options = MTLCompileOptions()
options.mathMode = .safe
options.mathFloatingPointFunctions = .precise
let lib = try d.makeLibrary(source: String(contentsOfFile: args[1], encoding: .utf8), options: options)
let build = try d.makeComputePipelineState(function: lib.makeFunction(name: "make_exact_table")!)
let pipe = try d.makeComputePipelineState(function: lib.makeFunction(name: "lookup_bench")!)
let table = d.makeBuffer(length: 65536 * 4, options: .storageModeShared)!
let nMax = 1 << 20
let input = d.makeBuffer(length: nMax * 2, options: .storageModeShared)!
let output = d.makeBuffer(length: nMax * 4, options: .storageModeShared)!
let inp = input.contents().bindMemory(to: UInt16.self, capacity: nMax)
let out = output.contents().bindMemory(to: Float.self, capacity: nMax)
let td = MTLTextureDescriptor()
td.textureType = .type1D; td.pixelFormat = .r32Float
td.width = 4096; td.height = 1; td.depth = 1; td.mipmapLevelCount = 1
td.storageMode = .shared; td.usage = .shaderRead
let texture = d.makeTexture(descriptor: td)!
emit(["type": "metadata", "device": d.name, "os": ProcessInfo.processInfo.operatingSystemVersionString,
      "thread_execution_width": pipe.threadExecutionWidth, "max_threads": pipe.maxTotalThreadsPerThreadgroup,
      "math_mode": "safe", "fp_functions": "precise", "table_bytes": 262144,
      "supports_32bit_float_filtering": d.supports32BitFloatFiltering,
      "texture_bytes": 16384, "warmup": 3, "samples": 11, "dispatches_per_sample": 64,
      "sustained_baseline_warmup_gpu_seconds_per_case": 0.3,
      "timing": "command buffer GPU duration divided by dispatches", "acceleration": "scalar/SIMD ALU and texture sampler only"])

func run(_ n: Int, _ op: Int, _ mode: Int, _ repeats: Int) -> Double {
    let cb = q.makeCommandBuffer()!
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(pipe)
    enc.setBuffer(input, offset: 0, index: 0)
    enc.setBuffer(output, offset: 0, index: 1)
    enc.setBuffer(table, offset: 0, index: 2)
    enc.setTexture(texture, index: 0)
    var p = Params(n: UInt32(n), op: UInt32(op), mode: UInt32(mode), tableSize: 4096)
    enc.setBytes(&p, length: MemoryLayout<Params>.size, index: 3)
    for _ in 0..<repeats {
        enc.dispatchThreads(MTLSize(width: n, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        enc.memoryBarrier(scope: .buffers)
    }
    enc.endEncoding(); cb.commit(); check(cb)
    return (cb.gpuEndTime - cb.gpuStartTime) * 1e6 / Double(repeats)
}
let names = ["precise", "fast", "bf16_exact_lut", "texture_linear"]
for op in 0..<2 {
    let cb = q.makeCommandBuffer()!, enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(build); enc.setBuffer(table, offset: 0, index: 0)
    var v = UInt32(op); enc.setBytes(&v, length: 4, index: 1)
    enc.dispatchThreads(MTLSize(width: 65536, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    enc.endEncoding(); cb.commit(); check(cb)
    let buildUS = (cb.gpuEndTime - cb.gpuStartTime) * 1e6
    let opName = op == 0 ? "sigmoid" : "softplus_stable_log1pexp"
    var tex = (0..<4096).map { i -> Float in
        let x = -12.0 + Double(i) * 24.0 / 4095.0
        return Float(op == 0 ? 1.0 / (1.0 + exp(-x)) : max(x, 0) + log1p(exp(-abs(x))))
    }
    tex.withUnsafeMutableBytes { bytes in
        texture.replace(region: MTLRegionMake1D(0, 4096), mipmapLevel: 0, withBytes: bytes.baseAddress!, bytesPerRow: 0)
    }
    for i in 0..<65536 { inp[i] = UInt16(i) }
    _ = run(65536, op, 0, 1)
    let reference = Array(UnsafeBufferPointer(start: out, count: 65536))
    _ = run(65536, op, 2, 1)
    var mismatches = 0, finiteCount = 0, specialMismatches = 0
    for i in 0..<65536 {
        let x = Float(bitPattern: UInt32(i) << 16)
        if x.isFinite { finiteCount += 1; if out[i].bitPattern != reference[i].bitPattern { mismatches += 1 } }
        else if !(out[i].isNaN && reference[i].isNaN) && out[i].bitPattern != reference[i].bitPattern { specialMismatches += 1 }
    }
    emit(["type": "exhaustive", "operation": opName, "finite_input_count": finiteCount,
          "finite_bit_mismatches": mismatches, "special_value_mismatches_ignoring_nan_payload": specialMismatches,
          "table_build_gpu_us": buildUS])
    precondition(mismatches == 0 && specialMismatches == 0, "Exact LUT parity failure")
    for distribution in ["bf16_uniform_minus8_plus8", "all_finite_bf16_bits"] {
        var seed: UInt64 = 1234567
        for i in 0..<nMax {
            seed = seed &* 6364136223846793005 &+ 1
            if distribution == "bf16_uniform_minus8_plus8" {
                let x = Float(Double(UInt32(truncatingIfNeeded: seed >> 32)) / Double(UInt32.max) * 16.0 - 8.0)
                // Round to nearest even BF16.
                let bits = x.bitPattern; inp[i] = UInt16(truncatingIfNeeded: (bits &+ 0x7fff &+ ((bits >> 16) & 1)) >> 16)
            } else {
                var bits = UInt16(truncatingIfNeeded: seed >> 32)
                if bits & 0x7f80 == 0x7f80 { bits &= 0xff7f }
                inp[i] = bits
            }
        }
        for n in [4096, nMax] {
            _ = run(n, op, 0, 1)
            let ref = Array(UnsafeBufferPointer(start: out, count: n))
            var errors: [[String: Any]] = []
            for mode in 0..<4 {
                _ = run(n, op, mode, 1)
                var maxAbs = 0.0, square = 0.0, bitDiff = 0, nonfinite = 0
                for i in 0..<n {
                    if out[i].bitPattern != ref[i].bitPattern { bitDiff += 1 }
                    if !out[i].isFinite { nonfinite += 1; continue }
                    let e = abs(Double(out[i]) - Double(ref[i])); maxAbs = max(maxAbs, e); square += e * e
                }
                errors.append(["max_abs_vs_precise": maxAbs, "rms_vs_precise": sqrt(square / Double(n)),
                               "bit_mismatches_vs_precise": bitDiff, "nonfinite_outputs": nonfinite])
                for _ in 0..<3 { _ = run(n, op, mode, 16) }
            }
            var samples = Array(repeating: [Double](), count: 4)
            var warmupSeconds = 0.0
            while warmupSeconds < 0.3 {
                warmupSeconds += run(n, op, 0, 128) * 128 / 1e6
            }
            for rep in 0..<11 {
                let order = rep % 2 == 0 ? [0,1,2,3] : [3,2,1,0]
                for mode in order { samples[mode].append(run(n, op, mode, 64)) }
            }
            for mode in 0..<4 {
                var row = errors[mode]
                row.merge(["type": "result", "operation": opName, "distribution": distribution,
                           "elements": n, "arm": names[mode], "median_gpu_us": median(samples[mode]),
                           "samples_gpu_us": samples[mode], "speedup_vs_precise": median(samples[0]) / median(samples[mode])]) { _, new in new }
                emit(row)
            }
        }
    }
}
