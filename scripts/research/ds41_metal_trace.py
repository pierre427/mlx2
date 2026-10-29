"""Optional owned-process Metal System Trace; no inference or permission changes."""
from contextlib import contextmanager
import os
import signal
import subprocess
import time


def recording_started(text):
    """Recognize observed xctrace versions after their attach completes."""
    return 'Recording started' in text or 'Ctrl-C to stop the recording' in text


def stop_recorder(trace):
    """Give Instruments time to save; bound cleanup of our own recorder."""
    if trace.poll() is not None:
        return
    trace.send_signal(signal.SIGINT)
    try:
        trace.wait(timeout=30)
    except subprocess.TimeoutExpired:
        trace.terminate()
        try:
            trace.wait(timeout=3)
        except subprocess.TimeoutExpired:
            trace.kill()
            trace.wait(timeout=3)


@contextmanager
def trace_component(out,report,atomic):
    """Best-effort capture. Failed tracing does not establish shader use."""
    receipt={'scope':'Native Metal System Trace of the bounded component process; timings are instrumented',
             'status':'starting','target_pid':os.getpid(),'recording_ready':False}
    report['metal_trace']=receipt
    trace=None;log=None
    path=out/'metal.trace';logpath=out/'xctrace.log'
    if path.exists()or logpath.exists():raise FileExistsError('Refuse overwrite Metal evidence')
    try:
        log=logpath.open('xb')
        trace=subprocess.Popen(['xcrun','xctrace','record','--template','Metal System Trace',
            '--attach',str(os.getpid()),'--time-limit','10m','--no-prompt','--output',str(path)],
            stdout=log,stderr=subprocess.STDOUT)
        receipt['recorder_pid']=trace.pid
        deadline=time.monotonic()+30
        while time.monotonic()<deadline and trace.poll()is None:
            if recording_started(logpath.read_text(errors='replace')):
                receipt.update(status='recording',recording_ready=True);break
            time.sleep(.25)
        if not receipt['recording_ready']:
            receipt.update(status='unavailable',startup_returncode=trace.poll())
            stop_recorder(trace)
    except (OSError,subprocess.SubprocessError)as error:
        receipt.update(status='unavailable',error_type=type(error).__name__,error=str(error))
    try:
        yield receipt
    finally:
        if trace is not None:
            stop_recorder(trace)
            receipt['recorder_returncode']=trace.returncode
        if log is not None:log.close()
        receipt.update(trace_exists=path.exists(),finished_at=time.time(),recorder_stopped=trace is None or trace.poll()is not None)
        if receipt['recording_ready']:
            receipt['status']='captured'if path.exists() and trace.returncode==0 else'incomplete_output'
        atomic(out/'metal-trace-receipt.json',receipt)
