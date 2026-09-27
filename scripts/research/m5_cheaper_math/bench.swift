// Original, standalone research harness. Compile: swiftc -O bench.swift -o /tmp/m5math
import Foundation
import Metal

struct Params { var m: UInt32; var n: UInt32; var k: UInt32; var iterations: UInt32 }
struct Spec {
    var name: String, kind: String
    var m: Int, n: Int, k: Int, iterations: Int = 1, batch: Int = 8
    var arms: [String]
    var note: String = ""
}
func fail(_ s: String) -> Never { fputs(s+"\n",stderr); exit(1) }
func median(_ a: [Double]) -> Double { let s=a.sorted(); return s[s.count/2] }
func errors(_ actual: [Double], _ expected: [Double]) -> [String: Any] {
    var maxAbs=0.0, sumSq=0.0, refSq=0.0, maxScaled=0.0, finite=true
    for (a,b) in zip(actual,expected) {
        if !a.isFinite {finite=false;continue}
        let d=abs(a-b);maxAbs=max(maxAbs,d);sumSq+=d*d;refSq+=b*b
        maxScaled=max(maxScaled,d/max(abs(b),1e-6))
    }
    return ["max_abs":maxAbs,"rmse":sqrt(sumSq/Double(max(1,actual.count))),
            "relative_l2":sqrt(sumSq/max(refSq,1e-30)),"max_relative_floor_1e_6":maxScaled,
            "all_finite":finite,"checked_outputs":actual.count]
}
let args=CommandLine.arguments
func option(_ flag: String, _ defaultValue: String) -> String {
    if let i=args.firstIndex(of:flag),i+1<args.count{return args[i+1]};return defaultValue
}
let sourcePath=option("--source","scripts/research/m5_cheaper_math/kernels.metal")
let outputPath=option("--output","docs/research/m5-cheaper-math-20260925/results/microbench.json")
let samples=Int(option("--samples","9"))!
let only=option("--only","")
guard samples>=3 else {fail("At least three samples required")}
guard let device=MTLCreateSystemDefaultDevice(),let queue=device.makeCommandQueue() else {fail("No Metal GPU")}
let options=MTLCompileOptions();options.mathMode = .safe
let source=try String(contentsOfFile:sourcePath,encoding:.utf8)
let library=try device.makeLibrary(source:source,options:options)
var pipes:[String:MTLComputePipelineState]=[:]
for name in library.functionNames {
    pipes[name]=try device.makeComputePipelineState(function:library.makeFunction(name:name)!)
}
if args.contains("--compile-only") {print("Compiled \(pipes.count) Metal kernels; no GPU command submitted.");exit(0)}
let specs:[Spec]=[
    Spec(name:"reduction_1024",kind:"reduction",m:1,n:2048,k:1024,arms:["reduce_simd","reduce_shuffle","reduce_shared","reduce_serial"],note:"Serial arm changes row parallelism; collectives use identical 256-thread geometry."),
    Spec(name:"reduction_4096",kind:"reduction",m:1,n:2048,k:4096,arms:["reduce_simd","reduce_shuffle","reduce_shared","reduce_serial"]),
    Spec(name:"invroot_single",kind:"invroot",m:1,n:1048576,k:1,iterations:1,arms:["sqrt_div","rsqrt_precise","rsqrt_fast"],note:"Memory-heavy single operation; inputs logarithmically cover 1e-6 through 1e6."),
    Spec(name:"invroot_chain",kind:"invroot",m:1,n:262144,k:1,iterations:64,arms:["sqrt_div","rsqrt_precise","rsqrt_fast"],note:"Dependent arithmetic chain; not RMSNorm end-to-end."),
    Spec(name:"softmax_decode",kind:"softmax",m:1,n:32,k:4096,arms:["softmax_precise","softmax_fast","softmax_exp2"]),
    Spec(name:"softmax_prefill",kind:"softmax",m:1,n:512,k:4096,arms:["softmax_precise","softmax_fast","softmax_exp2"],note:"Stable full softmax with max and sum collectives, two exponent evaluations per element; synthetic logits span -80 to 0."),
    Spec(name:"dot_decode",kind:"dot",m:1,n:4096,k:4096,arms:["dot_acc1","dot_acc2","dot_acc4","dot_acc8","dot_acc16"]),
    Spec(name:"dot_verify8",kind:"dot",m:8,n:1024,k:4096,arms:["dot_acc1","dot_acc2","dot_acc4","dot_acc8","dot_acc16"]),
    Spec(name:"dot_prefill128",kind:"dot",m:128,n:256,k:2048,batch:4,arms:["dot_acc1","dot_acc2","dot_acc4","dot_acc8","dot_acc16"],note:"Independent dot rows; not competitive tiled GEMM; tests accumulator sensitivity at multirow shape."),
    Spec(name:"q4_decode_cache_hot",kind:"q4",m:1,n:4096,k:4096,arms:["q4_affine","q4_factored","q4_precomputed","q4_rounded_half"]),
    Spec(name:"q4_verify8",kind:"q4",m:8,n:1024,k:4096,arms:["q4_affine","q4_factored","q4_precomputed","q4_rounded_half"]),
    Spec(name:"q4_prefill128",kind:"q4",m:128,n:256,k:2048,batch:4,arms:["q4_affine","q4_factored","q4_precomputed","q4_rounded_half"]),
    Spec(name:"q4_decode_large_working_set",kind:"q4",m:1,n:32768,k:16384,batch:2,arms:["q4_affine","q4_factored","q4_precomputed","q4_rounded_half"],note:"320 MiB weights plus scale and bias, exceeding 256 MiB. Repeated same allocation, not an assumed DRAM bandwidth measurement."),
    Spec(name:"packed_fma",kind:"packed",m:1,n:65536,k:1,iterations:256,batch:4,arms:["fma_float2","fma_half2"],note:"8 independent two-component accumulators; identical half-representable input/coefficient; FP16 accumulation approximate. Register count is not available from this API.")
]
var results:[[String:Any]]=[]
let started=Date()
for sp in specs where only.isEmpty || sp.name.contains(only) {
    let nx = (sp.kind=="reduction" || sp.kind=="softmax") ? sp.n*sp.k : ((sp.kind=="dot" || sp.kind=="q4") ? sp.m*sp.k : sp.n)
    let nw = sp.kind=="dot" ? sp.n*sp.k : 1
    let nq = sp.kind=="q4" ? sp.n*sp.k/2 : 1
    let ng = sp.kind=="q4" ? sp.n*sp.k/64 : 1
    let no = sp.kind=="softmax" ? sp.n*sp.k : (sp.kind=="packed" ? sp.n*2 : ((sp.kind=="dot" || sp.kind=="q4") ? sp.m*sp.n : sp.n))
    func buffer(_ bytes:Int)->MTLBuffer{device.makeBuffer(length:max(4,bytes),options:.storageModeShared)!}
    let bx=buffer(nx*4),bw=buffer(nw*4),bq=buffer(nq),bs=buffer(ng*4),bb=buffer(ng*4),bo=buffer(no*4),bxs=buffer(max(1,sp.m*sp.k/64)*4)
    let x=bx.contents().bindMemory(to:Float.self,capacity:nx),w=bw.contents().bindMemory(to:Float.self,capacity:nw)
    let q=bq.contents().bindMemory(to:UInt8.self,capacity:nq),sc=bs.contents().bindMemory(to:Float.self,capacity:ng),bi=bb.contents().bindMemory(to:Float.self,capacity:ng)
    let output=bo.contents().bindMemory(to:Float.self,capacity:no)
    for i in 0..<nx {
        let v=Float((i &* 1664525 &+ 1013904223) & 65535)/65536
        if sp.kind=="invroot" {x[i]=pow(10,12*v-6)}
        else if sp.kind=="softmax" {x[i]=(-80)*v}
        else if sp.kind=="packed" {x[i]=floor(v*128)/256}
        else {x[i]=(v-0.5)*2}
    }
    for i in 0..<nw {w[i]=Float((i &* 1103515245 &+ 12345)&65535)/32768-1}
    for i in 0..<nq {q[i]=UInt8(truncatingIfNeeded:(i &* 37) ^ (i>>5) ^ 173)}
    for i in 0..<ng {sc[i]=Float((i*13)%31+1)*0.0031;bi[i] = -7.3*sc[i]}
    var p=Params(m:UInt32(sp.m),n:UInt32(sp.n),k:UInt32(sp.k),iterations:UInt32(sp.iterations))
    func encode(_ enc:MTLComputeCommandEncoder,_ arm:String) {
        for (index,b) in [bx,bw,bq,bs,bb,bo].enumerated(){enc.setBuffer(b,offset:0,index:index)}
        enc.setBytes(&p,length:MemoryLayout<Params>.stride,index:6);enc.setBuffer(bxs,offset:0,index:7)
        if arm=="q4_precomputed" {
            enc.setComputePipelineState(pipes["activation_group_sums"]!)
            enc.dispatchThreadgroups(MTLSize(width:(sp.m*sp.k/64+255)/256,height:1,depth:1),threadsPerThreadgroup:MTLSize(width:256,height:1,depth:1))
            enc.memoryBarrier(resources:[bxs])
        }
        enc.setComputePipelineState(pipes[arm]!)
        var groups=sp.n,threads=256
        if sp.kind=="dot" || sp.kind=="q4" {groups=sp.m*sp.n;threads=32}
        if sp.kind=="invroot" || sp.kind=="packed" || arm=="reduce_serial" {groups=(sp.n+255)/256}
        enc.dispatchThreadgroups(MTLSize(width:groups,height:1,depth:1),threadsPerThreadgroup:MTLSize(width:threads,height:1,depth:1))
        enc.memoryBarrier(resources:[bo,bxs])
    }
    func run(_ arm:String,_ batch:Int)->Double {
        let cb=queue.makeCommandBuffer()!;cb.label="\(sp.name)/\(arm)"
        let enc=cb.makeComputeCommandEncoder()!
        for _ in 0..<batch {encode(enc,arm)}
        enc.endEncoding();cb.commit();cb.waitUntilCompleted()
        if let e=cb.error {fail("GPU command failed: \(e)")}
        let elapsed=cb.gpuEndTime-cb.gpuStartTime
        if elapsed<=0 {fail("Missing GPU timestamps")}
        return elapsed*1e6/Double(batch)
    }
    let indices:[Int]
    if sp.kind=="dot" || sp.kind=="q4" {indices=Array(Set((0..<min(128,no)).map{($0*(no-1))/max(1,min(128,no)-1)})).sorted()}
    else if sp.kind=="softmax" {indices=Array(0..<no)}
    else {indices=Array(stride(from:0,to:no,by:max(1,no/16384)))}
    var refs:[Double]=[],roundedRefs:[Double]=[]
    if sp.kind=="softmax" {
        for r in 0..<sp.n {
            let mx=(0..<sp.k).map{Double(x[r*sp.k+$0])}.max()!
            var denominator=0.0;for j in 0..<sp.k {denominator+=exp(Double(x[r*sp.k+j])-mx)}
            for j in 0..<sp.k {refs.append(exp(Double(x[r*sp.k+j])-mx)/denominator)}
        }
    } else {
        for i in indices {
            var s=0.0,sh=0.0
            if sp.kind=="reduction" {for j in 0..<sp.k{s+=Double(x[i*sp.k+j])}}
            if sp.kind=="invroot" {s=Double(x[i]);for _ in 0..<sp.iterations{s=1/sqrt(s)+0.125}}
            if sp.kind=="packed" {
                let row=i/2,sign=(i%2==0 ? 1.0 : -1.0)
                for a in 0..<8 {var v=Double(x[row])+sign*Double(a)*0.03125;for _ in 0..<sp.iterations{v=v*0.9990234375+0.0009765625};s+=v}
            }
            if sp.kind=="dot" {let m=i/sp.n,n=i%sp.n;for j in 0..<sp.k{s+=Double(x[m*sp.k+j])*Double(w[n*sp.k+j])}}
            if sp.kind=="q4" {
                let m=i/sp.n,n=i%sp.n
                for j in 0..<sp.k {
                    let packed=q[n*sp.k/2+j/2],qi=Float(j%2==0 ? packed&15 : packed>>4)
                    let g=n*sp.k/64+j/64,xd=Double(x[m*sp.k+j])
                    let exact=Double(sc[g])*Double(qi)+Double(bi[g]);s+=xd*exact
                    let rounded=Float16(fma(sc[g],qi,bi[g]));sh+=xd*Double(rounded)
                }
            }
            refs.append(s);roundedRefs.append(sh)
        }
    }
    var timings:[String:[Double]]=[:],armErrors:[String:[String:Any]]=[:],observed:[String:[Double]]=[:]
    var warmupGPUSeconds=0.0
    let warmBatch=sp.name.contains("large_working_set") ? 8 : 64
    while warmupGPUSeconds<0.3 {warmupGPUSeconds += run(sp.arms[0],warmBatch)*Double(warmBatch)/1e6}
    for arm in sp.arms {
        _=run(arm,2);_=run(arm,2)
        let actual=indices.map{Double(output[$0])};observed[arm]=actual
        var e=errors(actual,refs)
        e["all_outputs_finite"]=(0..<no).allSatisfy{output[$0].isFinite}
        if sp.kind=="q4" {e["versus_rounded_half_dequant_reference"]=errors(actual,roundedRefs)}
        if sp.kind=="softmax" {
            var worst=0.0;for r in 0..<sp.n {var s=0.0;for j in 0..<sp.k{s+=Double(output[r*sp.k+j])};worst=max(worst,abs(s-1))};e["max_row_sum_error"]=worst
        }
        armErrors[arm]=e;timings[arm]=[]
    }
    for sample in 0..<samples {
        let shift=sample%sp.arms.count
        var order=Array(sp.arms[shift...])+Array(sp.arms[..<shift])
        if sample%2==1 {order.reverse()}
        for arm in order {timings[arm]!.append(run(arm,sp.batch))}
    }
    let baseline=median(timings[sp.arms[0]]!)
    var arms:[[String:Any]]=[]
    for arm in sp.arms {
        let times=timings[arm]!,med=median(times),mad=median(times.map{abs($0-med)})
        let pipeline=pipes[arm]!
        arms.append(["name":arm,"gpu_us_median":med,"gpu_us_mad":mad,"gpu_us_samples":times,"speedup_vs_first":baseline/med,"errors_vs_fp64_math":armErrors[arm]!,"error_vs_first_arm":errors(observed[arm]!,observed[sp.arms[0]]!),"thread_execution_width":pipeline.threadExecutionWidth,"max_total_threads_per_threadgroup":pipeline.maxTotalThreadsPerThreadgroup,"static_threadgroup_memory_bytes":pipeline.staticThreadgroupMemoryLength])
        print("\(sp.name) \(arm): \(String(format:"%.3f",med)) us, \(String(format:"%.3f",baseline/med))x")
    }
    fflush(stdout)
    var caseResult:[String:Any]=["case":sp.name,"kind":sp.kind,"m":sp.m,"n":sp.n,"k":sp.k,"iterations":sp.iterations,"batch_dispatches":sp.batch,"sustained_warmup_gpu_seconds":warmupGPUSeconds,"weight_and_metadata_bytes":nw*4+nq+ng*8,"note":sp.note,"arms":arms]
    if sp.kind=="softmax" {
        let fast=observed["softmax_fast"]!,base2=observed["softmax_exp2"]!
        caseResult["fast_vs_exp2_bitwise_mismatch_count"]=zip(fast,base2).reduce(0){$0+($1.0.bitPattern==$1.1.bitPattern ? 0:1)}
        caseResult["fast_vs_exp2_errors"]=errors(fast,base2)
    }
    results.append(caseResult)
}
let document:[String:Any]=[
    "schema":"mlx2.m5-cheaper-math.microbench.v1","started_at":ISO8601DateFormatter().string(from:started),"finished_at":ISO8601DateFormatter().string(from:Date()),
    "device":device.name,"registry_id":String(device.registryID),"has_unified_memory":device.hasUnifiedMemory,"os":ProcessInfo.processInfo.operatingSystemVersionString,
    "compile_fast_math":false,"samples":samples,"warmup":"at least 0.3 GPU seconds of baseline per case, then two commands of two dispatches per arm","ordering":"round-robin rotated arms, reverse on odd samples",
    "timer":"MTLCommandBuffer GPU end minus start divided by batched dispatch count; includes dispatch/barrier overhead",
    "source_path":sourcePath,"cli":args,"limitations":["Synthetic mechanism probes; not model or serving qualification.","Reduction inputs are bounded dyadic fractions; exact sums here do not establish general reduction-order parity.","The reciprocal-square-root chain converges: final errors do not bound transient or adversarial errors; single-operation case provides separate range coverage.","Algebraically equivalent arithmetic can reassociate FP32 results; half and fast math are approximate.","Q4 affine factorization does not preserve dequant-to-half rounding; report both references.","No register count/occupancy or native disassembly captured. Pipeline thread limits are not occupancy.","No tensor, simdgroup matrix, NAX, or ANE operations. Repeated buffers can be cache-hot; only the designated large Q4 case exceeds 256 MiB."],
    "results":results]
let data=try JSONSerialization.data(withJSONObject:document,options:[.prettyPrinted,.sortedKeys])
try FileManager.default.createDirectory(at:URL(fileURLWithPath:outputPath).deletingLastPathComponent(),withIntermediateDirectories:true)
try data.write(to:URL(fileURLWithPath:outputPath),options:.atomic)
print("Saved \(outputPath)")
