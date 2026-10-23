"""Durable session-bound wakeups. The native host delivers turns, never a shell."""
from __future__ import annotations

import datetime as dt
import json
import math
import sqlite3
import threading
import time
import uuid


class Wakeups:
    def __init__(self, controller, *, clock=time.time, scheduler=True):
        self.controller, self.clock = controller, clock
        self.lock = threading.RLock()
        self.closed = threading.Event()
        self.error = None
        self.resume_retry_after = {}
        self.delivery_retry_after = {}
        self.db_closed = False
        path = controller.home / 'wakeups.sqlite3'
        self.db = sqlite3.connect(path, check_same_thread=False)
        path.chmod(0o600)
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('CREATE TABLE IF NOT EXISTS wakeups (id TEXT PRIMARY KEY, profile TEXT NOT NULL, session TEXT NOT NULL, spec TEXT NOT NULL, due REAL NOT NULL, state TEXT NOT NULL, fence TEXT, error TEXT, updated REAL NOT NULL)')
        self.db.execute('CREATE INDEX IF NOT EXISTS wakeups_due ON wakeups(state,due)')
        # A crashed delivery may have reached the model. Never replay it silently.
        self.db.execute("UPDATE wakeups SET state='interrupted',error='Delivery interrupted; review before retrying' WHERE state='delivering'")
        self.db.commit()
        self.thread = None
        if scheduler:
            self.thread = threading.Thread(target=self._loop, daemon=True, name='mlx2-wakeups')
            self.thread.start()

    def _loop(self):
        while not self.closed.wait(0.5):
            try:
                self.step()
            except Exception as error:  # noqa: BLE001 -- retain scheduler health without logging private messages
                self.error = type(error).__name__

    def _profile(self):
        return self.controller.user_profiles.active

    def _rows(self, clause, values=(), limit=200, offset=0, *, fair=False):
        order = 'updated,due,id' if fair else 'due,id'
        records = self.db.execute('SELECT id,profile,session,spec,due,state,fence,error FROM wakeups WHERE ' + clause + ' ORDER BY ' + order + ' LIMIT ? OFFSET ?', (*values, limit, offset)).fetchall()
        return [{'id': r[0], 'profile_id': r[1], 'session_id': r[2], **json.loads(r[3]), 'next_due': r[4], 'state': r[5], 'delivery_fence': r[6], 'error': r[7]} for r in records]

    def snapshot(self, *, offset=0, limit=200, session=None):
        if type(offset) is not int or offset < 0 or type(limit) is not int or not 1 <= limit <= 200:
            raise ValueError('Choose a nonnegative offset and 1–200 results')
        with self.lock:
            clause, values = 'profile=?', [self._profile()]
            if session is not None:
                clause += ' AND session=?'
                values.append(session)
            rows = self._rows(clause, values, limit, offset)
            for row in rows:
                raw = row["message"].encode("utf8")
                row["message_bytes"] = len(raw)
                row["message_truncated"] = len(raw) > 2048
                row["message"] = raw[:2048].decode("utf8", errors="ignore")
            count = self.db.execute('SELECT COUNT(*) FROM wakeups WHERE ' + clause, values).fetchone()[0]
            return {'schedules': rows, 'total': count, 'offset': offset, 'limit': limit,
                    'deliveries': [{key: value for key, value in row.items() if key != "message"} for row in self._rows(clause + " AND state='ready'", values) if self.delivery_retry_after.get(row["id"], 0) <= self.clock()],
                    'requires_running_app': True, 'error': self.error}

    def _update(self, identity, state, *, fence=None, error=None, due=None):
        if due is None:
            self.db.execute('UPDATE wakeups SET state=?,fence=?,error=?,updated=? WHERE id=?', (state, fence, error, self.clock(), identity))
        else:
            self.db.execute('UPDATE wakeups SET state=?,fence=?,error=?,due=?,updated=? WHERE id=?', (state, fence, error, due, self.clock(), identity))
        self.db.commit()

    def schedule(self, request, *, session, target):
        if self.closed.is_set():
            raise ValueError('Wakeup scheduler is closing')
        if not isinstance(session, str) or not 1 <= len(session) <= 512:
            raise ValueError('A wakeup needs an exact saved conversation branch')
        message = request.get('message')
        if not isinstance(message, str) or not message.strip() or len(message.encode()) > 65536:
            raise ValueError('Wakeup messages must contain 1–65536 bytes')
        every, at = request.get('every_seconds'), request.get('at')
        if (every is None) == (at is None):
            raise ValueError('Choose either every_seconds or an ISO date/time with timezone')
        if every is not None:
            if type(every) not in (int, float) or not math.isfinite(every) or every < 1:
                raise ValueError('Wakeup interval must be at least one second')
            due = self.clock() + every
        else:
            if not isinstance(at, str):
                raise ValueError('Supply an ISO date/time with an explicit timezone')
            date = dt.datetime.fromisoformat(at)
            if date.tzinfo is None:
                raise ValueError('Wakeup date/time must include a timezone')
            due = date.timestamp()
            if not math.isfinite(due) or due <= self.clock():
                raise ValueError('Wakeup date/time must be in the future')
        if not math.isfinite(due):
            raise ValueError("Wakeup time must be finite")
        suspend = request.get('suspend_after', True)
        if type(suspend) is not bool:
            raise ValueError('suspend_after must be boolean')
        matches = [(identity, engine) for identity, engine in self.controller.engines.items()
                   if engine.worker and engine.worker.get('url') == target.get('url')
                   and engine.worker.get('resume_lineage_id') and not engine.worker['resume_lineage_id'].startswith('pending:')
                   and engine.worker.get('execution_kind') == 'autoregressive' and engine.worker.get('apc_persist_dir')
                   and (target.get('resume_lineage_id') is None or target['resume_lineage_id'] == engine.worker['resume_lineage_id'])
                   and engine.worker.get('checkpoint') == target.get('checkpoint')]
        if len(matches) != 1:
            raise ValueError('Wakeups require an app-managed model with suspend/resume support')
        engine_id, engine = matches[0]
        spec = {'message': message, 'every_seconds': every, 'at': at, 'suspend_after': suspend,
                'engine_id': engine_id, 'checkpoint': target['checkpoint'],
                'resume_lineage_id': engine.worker['resume_lineage_id']}
        identity = uuid.uuid4().hex
        with self.lock:
            if self.closed.is_set():
                raise ValueError("Wakeup scheduler is closing")
            self.db.execute('INSERT INTO wakeups VALUES(?,?,?,?,?,?,?,?,?)', (identity, self._profile(), session, json.dumps(spec, allow_nan=False), due, 'scheduled', None, None, self.clock()))
            self.db.commit()
        return {'wakeup_id': identity, 'next_due': due, 'session_id': session, 'requires_running_app': True}

    def step(self):
        with self.lock:
            if self.closed.is_set():
                return
            profile = self._profile()
            rows = self._rows("profile=? AND state IN ('scheduled','waiting_model','ready') AND due<=?", (profile, self.clock()), fair=True)
        for row in rows:
            engine = self.controller.engines.get(row['engine_id'])
            worker = getattr(engine, 'worker', None) or {}
            lineage = worker.get('resume_lineage_id')
            if worker.get('state') == 'Resuming':
                lineage = ((getattr(getattr(engine, 'suspension', None), 'record', None) or {}).get('worker') or {}).get('resume_lineage_id', lineage)
            error = None
            if not engine or worker.get('checkpoint') != row['checkpoint'] or lineage != row['resume_lineage_id']:
                state, error = 'blocked', 'Original model lineage changed; review this schedule'
            elif worker.get('state') == 'Suspended':
                if self.resume_retry_after.get(row['id'], 0) > self.clock():
                    continue
                self.resume_retry_after[row['id']] = self.clock() + 30
                try:
                    # Resume hashes and admission can block; never hold the UI snapshot lock.
                    self.controller.dispatch({'command': 'engine_resume', 'profile_id': row['engine_id']})
                except (ValueError, RuntimeError):
                    error = 'Waiting for model ownership or resume admission'
                state = 'waiting_model'
            elif worker.get('state') == 'Ready':
                state = 'ready'
            elif worker.get('state') in {'Preparing', 'Loading', 'Resuming', 'Suspending', 'Busy'}:
                state = 'waiting_model'
            else:
                state, error = 'blocked', 'Original model is stopped or could not resume'
            with self.lock:
                if self.closed.is_set():
                    return
                current = self._rows('id=? AND profile=?', (row['id'], profile), 1)
                if self._profile() != profile or not current or current[0]['state'] != row['state'] or current[0]['delivery_fence'] != row['delivery_fence']:
                    continue
                if state != row['state'] or error != row['error']:
                    self._update(row['id'], state, error=error)

    def dispatch(self, request, *, session=None, target=None):
        if self.closed.is_set():
            raise ValueError("Wakeup scheduler is closing")
        command = request['command']
        if command == 'wakeup_schedule':
            return self.schedule(request, session=session or request.get('session_id'), target=target or request.get('_target', {}))
        if command == 'wakeup_list':
            return self.snapshot(offset=request.get('offset', 0), limit=request.get('limit', 200), session=session)
        with self.lock:
            rows = self._rows('id=? AND profile=?', (request.get('wakeup_id'), self._profile()), 1)
            if not rows or (session is not None and rows[0]['session_id'] != session):
                raise ValueError('Unknown wakeup in this session')
            row = rows[0]
            if command == 'wakeup_cancel':
                self._update(row['id'], 'cancelled')
                return {'cancelled': row['id']}
            if command == 'wakeup_retry':
                if row['state'] not in {'blocked', 'interrupted', 'failed'}:
                    raise ValueError('Only a blocked or interrupted wakeup can be retried')
                self._update(row['id'], 'scheduled', due=self.clock())
                return {'retried': row['id']}
            if command == 'wakeup_claim':
                if row['state'] != 'ready' or self.delivery_retry_after.get(row['id'], 0) > self.clock():
                    raise ValueError('Wakeup is no longer ready')
                engine = self.controller.engines.get(row['engine_id'])
                worker = getattr(engine, 'worker', None) or {}
                if worker.get('state') != 'Ready' or worker.get('checkpoint') != row['checkpoint'] or worker.get('resume_lineage_id') != row['resume_lineage_id']:
                    self._update(row['id'], 'waiting_model')
                    raise ValueError('The scheduled model is no longer ready with its original lineage')
                fence = uuid.uuid4().hex
                self._update(row['id'], 'delivering', fence=fence)
                return {**row, 'delivery_fence': fence}
            if command == 'wakeup_complete':
                if row['state'] != 'delivering' or row['delivery_fence'] != request.get('delivery_fence'):
                    raise ValueError('Stale wakeup completion')
                if type(request.get('success')) is not bool or type(request.get('attempted', True)) is not bool:
                    raise ValueError('Wakeup completion requires boolean success and attempted flags')
                if request.get('success') is not True and not request.get('attempted', True):
                    self.delivery_retry_after[row['id']] = self.clock() + 1
                    self._update(row['id'], 'ready')
                    return {'state': 'ready'}
                if request.get('success') is not True:
                    self._update(row['id'], 'failed', error='Scheduled turn failed or was cancelled; review before retrying')
                    return {'state': 'failed'}
                repeat = row['every_seconds']
                next_due = row['next_due'] + (max(0, math.floor((self.clock() - row['next_due']) / repeat)) + 1) * repeat if repeat else row['next_due']
                self._update(row['id'], 'scheduled' if repeat else 'completed', due=next_due)
                if row['suspend_after'] and request.get('allow_suspend', True) is True:
                    engine = self.controller.engines.get(row['engine_id'])
                    worker = getattr(engine, 'worker', None) or {}
                    if worker.get('resume_lineage_id') == row['resume_lineage_id'] and worker.get('state') == 'Ready':
                        try:
                            self.controller.dispatch({'command': 'engine_suspend', 'profile_id': row['engine_id']})
                        except (ValueError, RuntimeError):
                            warning = 'Scheduled turn completed; automatic suspension was not admitted'
                            self._update(row['id'], 'scheduled' if repeat else 'completed', due=next_due, error=warning)
                            return {'state': 'scheduled' if repeat else 'completed', 'warning': warning}
                return {'state': 'scheduled' if repeat else 'completed'}
            raise ValueError('Unknown wakeup operation')

    def profile_departing(self, profile_id):
        """Invalidate deliveries before switching the UI's active profile.

        A turn may already have reached inference. It needs operator review,
        never automatic replay into the newly selected profile.
        """
        with self.lock:
            self.db.execute("UPDATE wakeups SET state='interrupted',fence=NULL,error='User profile changed during delivery; review before retrying',updated=? WHERE profile=? AND state='delivering'", (self.clock(), profile_id))
            self.db.commit()

    def close(self):
        if self.db_closed:
            return
        self.closed.set()
        if self.thread and self.thread is not threading.current_thread():
            self.thread.join(timeout=10)
            if self.thread.is_alive():
                raise RuntimeError('Wakeup scheduler is still stopping')
        with self.lock:
            if not self.db_closed:
                self.db.close()
                self.db_closed = True
