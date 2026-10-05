"""Research host quantum gate. Only evaluated terminal-proved states may pause.

The caller verifies both GPU locks for every grant. This gate never publishes
bootstrap state and never treats a paused private graph as a public checkpoint.
"""
from __future__ import annotations
from threading import Condition
import time

class PhaseController:
    def __init__(self,verify):
        if not callable(verify):raise TypeError('dual lease verifier required')
        self.verify=verify;self.condition=Condition();self.owner=None;self.remaining=0
        self.paused=True;self.aborted=False;self.finished=False;self.events=[];self.owners=[];self.wait_seconds=0.;self.active_since=None;self.active_seconds=0.
    def sample_clock(self):
        """Host completion clock excluding only terminal-proved operator pauses."""
        with self.condition:
            tick=time.monotonic_ns()
            active=self.active_seconds
            if self.active_since is not None:active+=tick/1e9-self.active_since
            return dict(monotonic_ns=tick,owned_elapsed_seconds=active,
                        paused=self.paused,lease_owner=dict(self.owner) if self.owner is not None else None,
                        clock_scope='host elapsed while lease active; includes CPU and GPU completion; not device-only')
    def grant(self,owner,budget):
        if type(budget) is not int or not 1<=budget<=16:raise ValueError('bounded1..16 evaluated phases per quantum')
        self.verify(owner)
        with self.condition:
            if not self.paused or self.finished or self.aborted:raise RuntimeError('grant requires paused live private work')
            key=(owner['session'],owner['lease_id'])
            if key in self.owners:raise RuntimeError('each quantum requires fresh dual lease')
            self.owners.append(key);self.owner=dict(owner);self.remaining=budget;self.paused=False;self.active_since=time.monotonic();self.condition.notify_all()
    def enter(self):
        with self.condition:
            tick=time.monotonic()
            while self.paused and not self.aborted:self.condition.wait()
            self.wait_seconds+=time.monotonic()-tick
            if self.aborted:raise RuntimeError('root-owned phase cancellation')
            self.verify(self.owner)
    def boundary(self,event):
        if type(event) is not dict or event.get('materialized') is not True or event.get('native_terminals_drained') is not True or type(event.get('public_state_published')) is not bool:
            raise RuntimeError('refuse ambiguous phase before yielding GPU lease')
        self.verify(self.owner)
        with self.condition:
            self.events.append(dict(event));self.remaining-=1
            if self.remaining<=0:
                self.active_seconds+=time.monotonic()-self.active_since;self.active_since=None;self.paused=True;self.condition.notify_all()
        self.enter()
    def finish(self):
        self.verify(self.owner)
        with self.condition:
            if self.active_since is not None:self.active_seconds+=time.monotonic()-self.active_since
            self.active_since=None;self.finished=True;self.paused=True;self.condition.notify_all()
    def release_completed(self,owner):
        self.verify(owner)
        with self.condition:
            if not self.paused:raise RuntimeError('completion release requires quiescence')
            self.finished=True;self.paused=False;self.condition.notify_all()
    def cancel(self,owner):
        self.verify(owner)
        with self.condition:
            if not self.paused:raise RuntimeError('cancel only at terminal-proved paused boundary')
            self.owner=dict(owner);self.aborted=True;self.paused=False;self.condition.notify_all()
    def wake(self):
        with self.condition:self.condition.notify_all()
    def seal_completed(self,owner,event):
        if type(event) is not dict or event.get('materialized') is not True or event.get('native_terminals_drained') is not True or event.get('product_idle') is not True:raise RuntimeError('completion lacks materialized terminal/product idle proof')
        self.verify(owner)
        with self.condition:
            if self.active_since is not None:self.active_seconds+=time.monotonic()-self.active_since
            self.active_since=None;self.paused=True;self.events.append(dict(event));self.condition.notify_all()
    def wait_quantum(self,timeout=90,*,completion=None):
        if not 0<timeout<=90:raise ValueError('bounded quantum wait<=90s')
        if completion is not None and not callable(completion):raise TypeError('completion predicate must be callable')
        with self.condition:
            if not self.condition.wait_for(lambda:self.paused or self.finished or (completion is not None and completion()),timeout):raise TimeoutError('no terminal-proved phase within quantum deadline')
            return dict(paused=self.paused,finished=self.finished,phase_count=len(self.events),last_phase=self.events[-1] if self.events else None,owned_compute_seconds=self.active_seconds,paused_wait_seconds=self.wait_seconds)
