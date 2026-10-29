# SPDX-License-Identifier: MIT
"""Bounded reference-PID Metal timeline over already-warm private AST tiles.

No trace/GPU/notification action occurs on import. Parent owns model and CPG.
Only execute_batched_tree's normal completion gate belongs in frame brackets;
raw comparisons, receipt writes, trace startup/flush and telemetry are outside.
"""
from pathlib import Path
import ctypes
import hashlib
import json
import os
import signal
import subprocess
import time
import uuid

_QUARANTINE=[]


def _write(path,value):
    tmp=path.with_name(path.name+'.tmp')
    with tmp.open('x')as f:json.dump(value,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)


def clock_bridge():
    """Explicit perf↔mach bracket; later trace mabs epoch uses same timebase."""
    class Timebase(ctypes.Structure):_fields_=[('numer',ctypes.c_uint32),('denom',ctypes.c_uint32)]
    lib=ctypes.CDLL('/usr/lib/libSystem.B.dylib');info=Timebase()
    lib.mach_timebase_info.argtypes=[ctypes.POINTER(Timebase)];lib.mach_timebase_info.restype=ctypes.c_int
    lib.mach_absolute_time.argtypes=[];lib.mach_absolute_time.restype=ctypes.c_uint64
    if lib.mach_timebase_info(ctypes.byref(info))!=0 or info.denom==0:raise RuntimeError('Mach timebase unavailable')
    before=time.perf_counter_ns();ticks=int(lib.mach_absolute_time());after=time.perf_counter_ns()
    return dict(perf_before_ns=before,perf_after_ns=after,mach_absolute_ticks=ticks,
        timebase_numerator=int(info.numer),timebase_denominator=int(info.denom),
        calibration_uncertainty_ns=after-before,
        formula='trace_ns=(perf_ns-perf_midpoint_ns)+mach_absolute_ticks*numer/denom-trace_mabs_epoch_ticks*numer/denom')


class _StartedNotification:
    def __init__(self,name):
        self.lib=ctypes.CDLL('/usr/lib/libSystem.B.dylib');self.token=ctypes.c_int();self.closed=False
        self.lib.notify_register_check.argtypes=[ctypes.c_char_p,ctypes.POINTER(ctypes.c_int)]
        self.lib.notify_register_check.restype=ctypes.c_uint32
        self.lib.notify_check.argtypes=[ctypes.c_int,ctypes.POINTER(ctypes.c_int)];self.lib.notify_check.restype=ctypes.c_uint32
        self.lib.notify_cancel.argtypes=[ctypes.c_int];self.lib.notify_cancel.restype=ctypes.c_uint32
        if self.lib.notify_register_check(name.encode(),ctypes.byref(self.token))!=0:raise RuntimeError('Darwin trace-start registration failed')
        self.ready()  # consume initial registration state BEFORE child launch
    def ready(self):
        value=ctypes.c_int()
        if self.lib.notify_check(self.token,ctypes.byref(value))!=0:raise RuntimeError('Darwin trace-start check failed')
        return value.value!=0
    def close(self):
        if not self.closed:self.lib.notify_cancel(self.token);self.closed=True


class _Trace:
    def __init__(self,unit,live):
        self.unit,self.live=Path(unit),live;self.process=self.log=self.note=None
        self.record=dict(target_pid=os.getpid(),template='Metal System Trace',trace='native-ast-metal.trace',
            log='native-ast-xctrace.log',time_limit_seconds=30,rolling_window_seconds=10,
            startup_timeout_seconds=20,available=False,ready=False,rate_eligible=False,
            export_deferred_until_after_GPU_release=True)
    def start(self):
        output=self.unit/self.record['trace'];log=self.unit/self.record['log']
        if output.exists()or log.exists():raise FileExistsError('Trace evidence exists')
        try:
            name='mlx2.ast.trace.started.'+uuid.uuid4().hex;self.note=_StartedNotification(name)
            self.log=log.open('xb');command=['xcrun','xctrace','record','--template','Metal System Trace',
                '--attach',str(os.getpid()),'--time-limit','30s','--window','10s','--notify-tracing-started',name,
                '--no-prompt','--output',str(output)]
            self.record.update(command=command,startup_begin_perf_ns=time.perf_counter_ns())
            self.process=subprocess.Popen(command,stdout=self.log,stderr=subprocess.STDOUT,start_new_session=True)
            self.record['trace_pid']=self.process.pid;deadline=time.monotonic()+20
            while time.monotonic()<deadline:
                self.live()
                if self.process.poll()is not None:raise RuntimeError('xctrace exited before ready notification')
                if self.note.ready():
                    self.record.update(ready=True,ready_perf_ns=time.perf_counter_ns(),
                        readiness_barrier='Darwin --notify-tracing-started notification for this child')
                    return
                time.sleep(.05)
            raise TimeoutError('xctrace did not signal readiness within20s')
        except BaseException as error:
            self.record.update(startup_error=dict(type=type(error).__name__,message=str(error)),
                evidence_scope='wall_only; no GPU occupancy/busy-union inference')
            self.stop()
    def stop(self):
        cleanup_error=None
        try:
            if self.process is not None and self.process.poll()is None:
                os.killpg(self.process.pid,signal.SIGINT);deadline=time.monotonic()+45
                while self.process.poll()is None and time.monotonic()<deadline:
                    self.live();time.sleep(.1)
                if self.process.poll()is None:raise TimeoutError('xctrace flush exceeded45s')
        except BaseException as error:cleanup_error=error
        finally:
            if self.process is not None and self.process.poll()is None:
                try:os.killpg(self.process.pid,signal.SIGKILL)
                except ProcessLookupError:pass
                try:self.process.wait(timeout=5)
                except BaseException as error:cleanup_error=cleanup_error or error
            if self.process is not None:self.record['exit_code']=self.process.returncode
            if self.note is not None:self.note.close();self.note=None
            if self.log is not None:self.log.close();self.log=None
            self.record['available']=bool(self.record['ready']and self.record.get('exit_code')==0 and(self.unit/self.record['trace']).exists())
            if cleanup_error is not None:self.record['cleanup_error']=str(cleanup_error)
        return self.record


def run(adapter,unit,live,prefix,anchor,paths,discovery,discovery_states,*,tiles=8):
    """Called after successful warm tile while the same fixed bank is attached."""
    import mlx.core as mx
    from mlx2.adapters.deepseek_v41_ast_batched_target import execute_batched_tree,RESIDENT_EXPERT_LAW,PROFILE
    from ds41_resident_prefill_model_gate import _array_compare,_cache_compare,_counters
    from ds41_ast_batched_model_probe import _close,_delta
    if type(tiles)is not int or tiles not in(8,16):raise ValueError('Timeline supports8or16 warm tiles')
    if not callable(live)or live()is not True or mx.default_device()!=mx.gpu:raise ValueError('Timeline requires owned GPU guard')
    unit=Path(unit);path=unit/'ast-batched-timeline.json'
    if path.exists()or path.with_name(path.name+'.tmp').exists():raise FileExistsError('Timeline evidence exists')
    bound=json.loads((unit/'spec.json').read_text())
    record=dict(schema='mlx2.ds41-native-ast-gpu-timeline.v1',target_pid=os.getpid(),
        checkpoint=bound['checkpoint'],source_revision=bound['source_revision'],source_snapshot_sha256=bound['source_snapshot_sha256'],
        profile=PROFILE,expert_law=RESIDENT_EXPERT_LAW,tiles=tiles,frames=[],status='trace_startup',
        rate_eligible=False,qualified=False,controlled_performance_claim=False,APCv2_publication=False,
        clock_scope='perf_counter_ns bracket calibrated to mach_absolute_time; export trace time-info after release',
        normal_completion_only=True,no_extra_per_frame_eval_or_synchronize=True)
    trace=_Trace(unit,live);frames=[];all_states=[];drained=True;_write(path,record)
    try:
        record['clock_before']=clock_bridge();trace.start();record['trace']=dict(trace.record);record['status']='frames';_write(path,record)
        before=_counters(adapter)
        for index in range(tiles):
            live();drained=False;start_ns=time.perf_counter_ns()
            frame=execute_batched_tree(adapter.model,prefix,anchor,paths,expert_law=RESIDENT_EXPERT_LAW)
            end_ns=time.perf_counter_ns();drained=True
            # The returned tile already passed its required whole-tile gate.
            # Abort is outside the frame bracket and performs no GPU evaluation.
            frames.append(frame);frame.transaction.abort()
            record['frames'].append(dict(index=index,target_pid=os.getpid(),input_rows=16,
                start_perf_ns=start_ns,end_perf_ns=end_ns,wall_seconds=(end_ns-start_ns)/1e9,receipt=frame.receipt))
        record['clock_after']=clock_bridge();record['counter_delta']=_delta(_counters(adapter),before)
        record['trace']=trace.stop();record['status']='post_trace_functional_checks';_write(path,record)
        for index,frame in enumerate(frames):
            live();a=_array_compare(discovery.logits,frame.logits,mx);b=_array_compare(discovery.features,frame.features,mx)
            # frontiers retain descriptors after transaction.abort; materialize
            # through their descriptor once, only AFTER trace has stopped.
            descriptors=[v.descriptor for v in frame.frontiers[0]]
            states=[[segment.materialize(node)for segment in descriptors]for node in range(16)]
            all_states.extend(states);drained=False;mx.eval([[r.state for r in rows]for rows in states]);drained=True
            comparisons=[_cache_compare(discovery_states[node],states[node],mx)for node in range(16)]
            passed=a['equal']and b['equal']and all(v['equal']for v in comparisons)
            record['frames'][index]['functional']=dict(logits=a,features=b,all16_cache_frontiers_exact=all(v['equal']for v in comparisons),
                cache_comparison_sha256=hashlib.sha256(json.dumps(comparisons,sort_keys=True).encode()).hexdigest(),passed=passed)
            if not passed:raise RuntimeError('Trace frame differs from source-bound batched/M1 reference')
            for rows in states:_close(rows)
            all_states=[]
        delta=record['counter_delta'];zero=all(delta.get(k,0)==0 for k in('expert_bytes_read','expert_uploads','engram_bytes_read','host_route_materializations','ordinary_replays'))
        record.update(status='passed',passed=zero,zero_io_host_routes_or_replay=zero,
            timeline_methodology_passed=False,timeline_capture_available=record['trace']['available'],
            timeline_validation_pending=['offline export clock bridge and per-frame GPU interval coverage'],
            evidence_scope='bounded instrumented GPU timeline; no kernel occupancy/counters or controlled rate claim'if record['trace']['available']else'wall_only; trace unavailable, no GPU busy/occupancy claim')
        if not zero:raise RuntimeError('Trace resident cohort performed I/O/host routing/replay')
        return record
    except BaseException as error:
        record.update(status='failed',passed=False,error=dict(type=type(error).__name__,message=str(error)))
        error.partial_result=record;raise
    finally:
        record['trace']=trace.stop()
        if not drained:
            try:mx.synchronize();drained=True
            except BaseException:_QUARANTINE.append((adapter,tuple(frames),tuple(all_states)))
        if drained:
            for frame in frames:frame.transaction.abort()
            for rows in all_states:_close(rows)
        _write(path,record)
