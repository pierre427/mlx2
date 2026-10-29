# SPDX-License-Identifier: MIT
"""Parent-owned native-stock oracle for explicit persistent PP task striding."""
from ds41_resident_prefill_components import plan as pp_plan,run as pp_run

def plan():return pp_plan(wide=True,task_policy='persistent_8_2_v1')

def run(mx,check_live=None):return pp_run(mx,check_live,wide=True,task_policy='persistent_8_2_v1')
