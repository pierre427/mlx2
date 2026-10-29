"""Owned actual-coefficient paired screen for Source resident kernel variants."""
from ds41_resident_bank_components import plan as bank_plan,run as bank_run

def plan():
    return dict(bank_plan(),schema='mlx2.ds41-resident-kernel-components.plan.v1',
        kernel_variants=['down8_v1','route_fused_v1','route_down8_v1'],timing_pairs=12,
        timing_scope='resident complete FFN host dispatch plus GPU completion; guards outside pair; bank build/input QAT excluded')

def run(mx,check_live=None):
    return bank_run(mx,check_live,kernel_variants=('down8_v1','route_fused_v1','route_down8_v1'),timing_repeats=12)
