import Foundation
import Metal
// Original isolated GDN scalar-decay representation benchmark.
struct Params { var heads:UInt32, steps:UInt32, unused0:UInt32=0, unused1:UInt32=0 }
func fail(_ s:String)->Never{fputs(s+"\n",stderr);exit(1)}
func median(_ a:[Double])->Double{a.sorted()[a.count/2]}
func error(_ a:[Float],_ b:[Double])->[String:Any]{
 var ma=0.0,ss=0.0,rr=0.0,finite=true
 for(i,x)in a.enumerated(){let d=Double(x)-b[i];if !x.isFinite{finite=false;continue};ma=max(ma,abs(d));ss+=d*d;rr+=b[i]*b[i]}
 return ["max_abs":ma,"relative_l2":sqrt(ss/max(rr,1e-30)),"rmse":sqrt(ss/Double(a.count)),"all_finite":finite,"checked":a.count]
}
let args=CommandLine.arguments
func opt(_ k:String,_ d:String)->String{if let i=args.firstIndex(of:k),i+1<args.count{return args[i+1]};return d}
let outPath=opt("--output","docs/research/m5-cheaper-math-20260925/results/gdn-run1.json")
let samples=Int(opt("--samples","9"))!
let d=MTLCreateSystemDefaultDevice()!,queue=d.makeCommandQueue()!,opts=MTLCompileOptions();opts.mathMode = .safe
let lib=try d.makeLibrary(source:String(contentsOfFile:"scripts/research/m5_cheaper_math/gdn_kernels.metal",encoding:.utf8),options:opts)
let names=["gdn_direct","gdn_lazy8","gdn_lazy32"]
var pipes:[String:MTLComputePipelineState]=[:];for n in names{pipes[n]=try d.makeComputePipelineState(function:lib.makeFunction(name:n)!)}
if args.contains("--compile-only"){print("Compiled three GDN kernels, no command submitted");exit(0)}
func random(_ i:Int,_ salt:Int)->Float {var z=UInt32(truncatingIfNeeded:i &+ salt);z=(z^(z>>16)) &* 0x7feb352d;z=(z^(z>>15)) &* 0x846ca68b;z=z^(z>>16);return Float(z>>8)/8388608-1}
var cases:[[String:Any]]=[]
for (steps,snapshotAll) in [(1,false),(8,false),(128,false),(8,true)] {
 let heads=32,stateCount=heads*16384,vectorCount=steps*heads*128,gateCount=steps*heads
 let snapshotCount=snapshotAll ? steps*stateCount : 1
 let counts=[stateCount,vectorCount,vectorCount,vectorCount,gateCount,gateCount,stateCount,vectorCount,snapshotCount]
 let buffers=counts.map{d.makeBuffer(length:$0*4,options:.storageModeShared)!}
 let ptrs=zip(buffers,counts).map{$0.0.contents().bindMemory(to:Float.self,capacity:$0.1)}
 for i in 0..<stateCount{ptrs[0][i]=random(i,91)*0.02}
 for i in 0..<vectorCount{ptrs[1][i]=random(i,17);ptrs[2][i]=random(i,1337);ptrs[3][i]=random(i,19937)*0.25}
 for base in stride(from:0,to:vectorCount,by:128){
  var kn=0.0,qn=0.0;for j in 0..<128{kn+=Double(ptrs[1][base+j])*Double(ptrs[1][base+j]);qn+=Double(ptrs[2][base+j])*Double(ptrs[2][base+j])}
  for j in 0..<128{ptrs[1][base+j]/=Float(sqrt(kn));ptrs[2][base+j]/=Float(sqrt(qn))}
 }
 for i in 0..<gateCount{ptrs[4][i]=exp(-0.025-0.015*random(i,721));ptrs[5][i]=0.1+0.05*random(i,233)}
 var p=Params(heads:UInt32(heads),steps:UInt32(steps),unused0:snapshotAll ? 1:0)
 func run(_ name:String,_ batch:Int)->Double {
  let cb=queue.makeCommandBuffer()!,e=cb.makeComputeCommandEncoder()!
  e.setComputePipelineState(pipes[name]!);for(i,b)in buffers.enumerated(){e.setBuffer(b,offset:0,index:i==8 ? 9:i)};e.setBytes(&p,length:MemoryLayout<Params>.stride,index:8)
  for _ in 0..<batch{e.dispatchThreadgroups(MTLSize(width:heads,height:1,depth:1),threadsPerThreadgroup:MTLSize(width:256,height:1,depth:1));e.memoryBarrier(resources:[buffers[6],buffers[7],buffers[8]])}
  e.endEncoding();cb.commit();cb.waitUntilCompleted();if let err=cb.error{fail("\(err)")}
  let elapsed=cb.gpuEndTime-cb.gpuStartTime;if elapsed<=0{fail("No GPU timer")};return elapsed*1e6/Double(batch)
 }
 // FP64 direct recurrence oracle for all elements in two heads. All heads are
 // additionally compared against the FP32 direct GPU arm, not just sampled.
 let oracleHeads=2
 var oracleState=Array(repeating:0.0,count:oracleHeads*16384),oracleOut=Array(repeating:0.0,count:steps*oracleHeads*128)
 for i in 0..<oracleState.count{oracleState[i]=Double(ptrs[0][i])}
 for t in 0..<steps{for h in 0..<oracleHeads{
  let vb=(t*heads+h)*128,g=Double(ptrs[4][t*heads+h]),beta=Double(ptrs[5][t*heads+h])
  for r in 0..<128{
   let sb=h*16384+r*128;var pred=0.0
   for j in 0..<128{oracleState[sb+j]*=g;pred+=oracleState[sb+j]*Double(ptrs[1][vb+j])}
   let delta=beta*(Double(ptrs[3][vb+r])-pred);var o=0.0
   for j in 0..<128{oracleState[sb+j]+=delta*Double(ptrs[1][vb+j]);o+=oracleState[sb+j]*Double(ptrs[2][vb+j])}
   oracleOut[(t*oracleHeads+h)*128+r]=o
  }
 }}
 var warm=0.0;while warm<0.3{warm+=run(names[0],32)*32/1e6}
 var actualState:[String:[Float]]=[:],actualOut:[String:[Float]]=[:],actualSnapshots:[String:[Float]]=[:],times:[String:[Double]]=[:]
 for n in names{_=run(n,4);actualState[n]=Array(UnsafeBufferPointer(start:ptrs[6],count:stateCount));actualOut[n]=Array(UnsafeBufferPointer(start:ptrs[7],count:vectorCount));actualSnapshots[n]=Array(UnsafeBufferPointer(start:ptrs[8],count:snapshotCount));times[n]=[]}
 for s in 0..<samples{let sh=s%3;var order=Array(names[sh...])+Array(names[..<sh]);if s%2==1{order.reverse()};for n in order{times[n]!.append(run(n,steps==128 ? 4:16))}}
 let base=median(times[names[0]]!),baseState=actualState[names[0]]!.map(Double.init),baseOut=actualOut[names[0]]!.map(Double.init)
 var arms:[[String:Any]]=[]
 for n in names{
  let vals=times[n]!,med=median(vals),st=actualState[n]!,ou=actualOut[n]!
  let subsetState=Array(st[..<(oracleHeads*16384)]);var subsetOut:[Float]=[]
  for t in 0..<steps{for h in 0..<oracleHeads{let b=(t*heads+h)*128;subsetOut+=Array(ou[b..<(b+128)])}}
  let es=error(st,baseState),eo=error(ou,baseOut)
  var arm:[String:Any]=["name":n,"gpu_us_median":med,"gpu_us_mad":median(vals.map{abs($0-med)}),"gpu_us_samples":vals,"speedup_vs_direct":base/med,"state_vs_direct":es,"output_vs_direct":eo,"state_vs_fp64_two_heads":error(subsetState,oracleState),"output_vs_fp64_two_heads":error(subsetOut,oracleOut)]
  if snapshotAll {arm["snapshots_vs_direct"]=error(actualSnapshots[n]!,actualSnapshots[names[0]]!.map(Double.init))}
  arms.append(arm)
  print("steps=\(steps) snapshots=\(snapshotAll) \(n) \(String(format:"%.3f",med)) us \(String(format:"%.3f",base/med))x stateError=\(es["max_abs"]!) outputError=\(eo["max_abs"]!)")
 }
 var minScale:[String:Double]=[:]
 for interval in [8,32] {var lowest=1.0;for h in 0..<heads{var c=1.0;for t in 0..<steps{c*=Double(ptrs[4][t*heads+h]);lowest=min(lowest,c);if(t+1)%interval==0{c=1}}};minScale[String(interval)]=lowest}
 fflush(stdout);cases.append(["heads":heads,"key_dim":128,"value_dim":128,"steps":steps,"snapshot_all":snapshotAll,"input_derived_min_scale_fp64":minScale,"warmup_gpu_seconds":warm,"arms":arms])
}
let obj:[String:Any]=["schema":"mlx2.m5-cheaper-math.gdn.v1","device":d.name,"timestamp":ISO8601DateFormatter().string(from:Date()),"compile_math_mode":"safe","samples":samples,"normalization":"k and q unit L2 per head-step; g=exp(-u), u in [0.01,0.04]; beta in [0.05,0.15]; v in [-0.25,0.25]","timer":"GPU command duration divided by batched dispatches; includes initialization, all token outputs and final materialized state","limits":["Synthetic FP32 direct recurrence, not current production GDN or full-model qualification.","Lazy representation changes rounding and division; every final state is materialized for comparison.","Only two heads use FP64 oracle; every element and output in all32 heads compares with direct GPU reference.","One256-threadgroup per head,16value rows per SIMD group; no occupancy counters captured.","No NAX, ANE, simdgroup matrices or tensor instructions."],"results":cases]
try JSONSerialization.data(withJSONObject:obj,options:[.prettyPrinted,.sortedKeys]).write(to:URL(fileURLWithPath:outPath),options:.atomic)
print("Saved \(outPath)")
