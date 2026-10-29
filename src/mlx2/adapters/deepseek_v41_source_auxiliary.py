# SPDX-License-Identifier: MIT
"""Measured auxiliary admission beside Source's advisory demand expert cache.

Only eager auxiliary payload/preparation/staging, future execution workspace,
and imminent streamed progress are reserved. The configured expert-cache
maximum and the full target dense estimate never enter this requirement.
"""


def _bytes(value, name):
    if type(value) is not int or value < 0:
        raise ValueError('Source auxiliary ' + name + ' must be nonnegative integer bytes')
    return value


def stage_auxiliary_plans(admission, plans):
    """Publish complete CPU plans before constructing any auxiliary owner."""
    if type(plans) is not dict:
        raise ValueError('Source auxiliary plans must be a dictionary')
    staged = {}
    for name, plan in plans.items():
        if type(name) is not str or not name or type(plan) is not dict:
            raise ValueError('Source auxiliary plan must have a named dictionary')
        row = dict(plan)
        reservation = _bytes(row.get('reservation_bytes'), 'reservation')
        workspace = _bytes(row.get('workspace_bound_bytes', 0), 'workspace')
        components = sum(_bytes(row.get(field, 0), field) for field in
                         ('payload_bytes', 'prepared_bound_bytes', 'staging_bound_bytes'))
        if reservation < components + workspace:
            raise ValueError('Source auxiliary reservation omits eager payload or workspace')
        staged[name] = row
    admission['auxiliary_plans'] = staged
    admission['auxiliary_reservation_bytes'] = sum(row['reservation_bytes'] for row in staged.values())
    admission['auxiliary_workspace_bound_bytes'] = sum(row.get('workspace_bound_bytes', 0) for row in staged.values())
    for name, row in staged.items():
        admission[name + '_reservation_bytes'] = row['reservation_bytes']
    return admission


def auxiliary_memory_requirement(admission, *, loaded=()):
    plans = admission.get('auxiliary_plans', {})
    loaded = tuple(loaded)
    if len(set(loaded)) != len(loaded) or any(name not in plans for name in loaded):
        raise ValueError('Source auxiliary loaded inventory differs from staged plans')
    remaining = sum(row.get('workspace_bound_bytes', 0) if name in loaded else row['reservation_bytes']
                    for name, row in plans.items())
    return remaining + _bytes(admission.get('imminent_working_reserve_bytes', 0), 'imminent progress')


def check_auxiliary_memory(admission, *, loaded=(), active_bytes, available_bytes):
    """Require real headroom for eager owners; loaded payload is charged once.

    Call after optional expert reclamation, then remeasure. Active bytes already
    include retained target and loaded auxiliary weights. Unknown host availability
    cannot establish an eager reservation; it does not change ordinary admission.
    """
    active = _bytes(active_bytes, 'active memory')
    if type(available_bytes) is not int or available_bytes < 0:
        raise MemoryError('Source auxiliary reservation requires measured available memory')
    required = auxiliary_memory_requirement(admission, loaded=loaded)
    limit = admission.get('admission_limit_bytes')
    if limit is not None:
        _bytes(limit, 'process limit')
    receipt = dict(schema='mlx2.source-auxiliary-admission.v1', loaded=list(loaded),
                   active_bytes=active, available_bytes=available_bytes,
                   future_allocation_and_workspace_bytes=required,
                   process_limit_bytes=limit, expert_cache_maximum_reserved=False,
                   full_target_dense_estimate_reserved=False, qualification='unqualified')
    if available_bytes < required or (limit is not None and active + required > limit):
        raise MemoryError('Source auxiliary payload and future workspace exceed measured headroom')
    return receipt
