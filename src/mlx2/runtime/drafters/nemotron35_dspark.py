# SPDX-License-Identifier: Apache-2.0
"""Super VL DSpark: causal proposal block, Markov bias and trained confidence.

Source mechanisms: vllm-project/speculators; provenance/nemotron35-super-vl.json.
The model-independent implementation lives in :mod:`.dspark`; this module
keeps the historical import path of the causal (speculators) variant.
"""
from .dspark import (  # noqa: F401
    ConfidenceHead,
    DSparkAttention,
    DSparkDraftModel,
    DSparkLayer,
    MarkovHead,
    proposal_mask,
)
