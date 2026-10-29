# SPDX-License-Identifier: MIT
"""Fail closed at Source profile boundaries before constructing device owners."""


def validate_source_profiles(policy):
    """Keep independently implemented state/forward laws in their proven slices.

    This is capability admission, not a qualification or performance verdict.
    The ordinary adaptive path remains available with every individual extension.
    """
    if type(policy) is not dict:
        raise ValueError('Source V4.1 execution policy must be a dictionary')
    extensions = []
    for name, reference in (
        ('deepseek_kv_storage', 'reference'),
        ('deepseek_candidate_score', 'reference'),
        ('deepseek_ced', 'reference'),
        ('deepseek_vision', 'reference'),
        ('deepseek_physical_batch', 'b1'),
    ):
        if policy.get(name, reference) != reference:
            extensions.append(name)
    if policy.get('draft_model') == 'embedded_dspark':
        extensions.append('embedded_dspark')
    incoming = []
    for name, reference in (
        ('deepseek_source_dspark_ast', 'off'),
        ('deepseek_source_dspark_prepare', 'off'),
        ('deepseek_source_resident_gather', 'reference'),
        ('deepseek_source_resident_prefill', 'reference'),
        ('deepseek_source_prefill_partition', 'adaptive_v1'),
    ):
        if policy.get(name, reference) != reference:
            incoming.append(name)
    if extensions and incoming:
        raise ValueError('Source profile combination is unsupported: '
                         + ', '.join(extensions) + ' with ' + ', '.join(incoming))
