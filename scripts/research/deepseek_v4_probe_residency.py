"""Bound direct probes to the production weight-wiring mechanism."""


def wire_preloaded_weights(mx, receipt):
    from mlx2.runtime.weight_residency import wire_serving_weights
    # Store restore authority immediately after the mutating API succeeds.
    receipt['weight_residency'] = wire_serving_weights()
    receipt['weight_residency']['device_working_set_cap_bytes'] = mx.device_info().get('max_recommended_working_set_size')
    receipt['weight_residency']['observation'] = (
        'configured residency request and allocator active bytes; '
        'not independent measurement of physical resident/nonresident pages')
    return receipt['weight_residency']


def restore_probe_wired_limit(mx, receipt):
    state = receipt.get('weight_residency', {})
    if state.get('wired') and not state.get('prior_limit_restored'):
        previous = state['previous_wired_limit_bytes']
        displaced = mx.set_wired_limit(previous)
        state['restore_displaced_limit_bytes'] = int(displaced)
        state['prior_limit_restored'] = True
