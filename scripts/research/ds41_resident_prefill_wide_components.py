"""Parent-owned exact native-stock raw oracle for wide resident PP banks."""
from ds41_resident_prefill_components import plan as pp_plan,run as pp_run

def plan():return pp_plan(wide=True)

def run(mx,check_live=None):return pp_run(mx,check_live,wide=True)
