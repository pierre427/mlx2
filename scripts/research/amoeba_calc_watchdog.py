# SPDX-License-Identifier: LicenseRef-Pierre-Lamy-Calculator-NC-ND-1.0
# Copyright (c) 2026 Pierre Lamy
# Scope and earlier grants: provenance/LICENSE-calculator-scope.json
# Terms: provenance/LICENSE-calculator.txt
"""Run the unchanged calculator benchmark with a bounded inactivity watchdog.

Diagnostic wrapper only: no model, token, gate, parser or answer changes. The
append-only journal identifies a pending GPU calculation if a driver wait stalls.
It is not suitable for reporting performance numbers.
"""
import argparse
import dataclasses
import faulthandler
import json
import os
from pathlib import Path
import runpy
import sys
import threading
import time


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--journal',type=Path,required=True)
    ap.add_argument('--stall-seconds',type=float,default=30)
    args,forward=ap.parse_known_args()
    # Must precede importing mlx2 (whose process default enables fast fences).
    os.environ['MLX_METAL_FAST_SYNCH']='0'
    if args.stall_seconds<10:raise ValueError('watchdog requires at least 10 seconds')
    args.journal.parent.mkdir(parents=True,exist_ok=True)
    journal=args.journal.open('x',buffering=1)
    stacks=args.journal.with_suffix('.stacks.txt').open('x',buffering=1)
    faulthandler.enable(file=stacks)
    progress=[time.monotonic()]
    lock=threading.Lock()
    def record(kind,**data):
        with lock:
            journal.write(json.dumps(dict(time=time.time(),kind=kind,**data),default=lambda x:dataclasses.asdict(x) if dataclasses.is_dataclass(x) else str(x))+'\n')
            progress[0]=time.monotonic()
        # CPython's native watchdog also works while an extension holds the GIL.
        faulthandler.dump_traceback_later(args.stall_seconds,file=stacks,exit=True)
    def monitor():
        while True:
            time.sleep(1)
            if time.monotonic()-progress[0]>args.stall_seconds:
                faulthandler.dump_traceback(file=stacks,all_threads=True)
                record('watchdog_timeout',inactivity_seconds=args.stall_seconds)
                os._exit(124)
    record('start',argv=forward,metal_fast_synch='0')
    threading.Thread(target=monitor,daemon=True).start()
    from mlx2.runtime import amoeba_calc_gpu as G
    evaluate=G.evaluate
    def traced_evaluate(expr,backend='gpu'):
        record('calculator_start',expr=expr,backend=backend)
        result=evaluate(expr,backend)
        record('calculator_done',result=result)
        return result
    G.evaluate=traced_evaluate
    from mlx_lm.generate import BatchGenerator
    next_generated=BatchGenerator.next_generated
    calls=[0]
    def next_with_progress(self):
        result=next_generated(self)
        progress[0]=time.monotonic()
        calls[0]+=1
        if calls[0]%32==0 or not result:record('batch_progress',calls=calls[0])
        return result
    BatchGenerator.next_generated=next_with_progress
    sys.argv=[str(Path(__file__).with_name('amoeba_calc_eval.py')),*forward]
    runpy.run_path(sys.argv[0],run_name='__main__')
    record('complete')
    faulthandler.cancel_dump_traceback_later()


if __name__=='__main__':main()
