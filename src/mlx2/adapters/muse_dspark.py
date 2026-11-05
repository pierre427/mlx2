# SPDX-License-Identifier: Apache-2.0
"""Strict binding of the released Muse mask-row DSpark checkpoint."""
from dataclasses import asdict
import json
from pathlib import Path

from . import muse_dflash


def is_dspark_artifact(path):
    config = json.loads((Path(path).expanduser() / 'config.json').read_text())
    return config.get('architectures') == ['Qwen3DSparkModel']


def config_from_hf(config, *, target_text):
    from ..runtime.drafters.muse_dflash import MuseDFlashConfig
    from ..runtime.drafters.muse_dspark import MuseDSparkConfig

    if (config.get('architectures') != ['Qwen3DSparkModel']
        or config.get('sample_from_anchor') is not False
        or type(config.get('markov_rank')) is not int or config['markov_rank'] <= 0
        or config.get('enable_confidence_head', False) is not False
        or config.get('quantization') or config.get('quantization_config')):
        raise ValueError('Unsupported Muse DSpark mask-row/Markov contract')
    normalized = {**config, 'architectures': ['MuseGlimmerAssistantModel'],
                  'model_type': 'muse_glimmer_assistant'}
    args = MuseDFlashConfig.from_hf(normalized, target_text=target_text)
    return MuseDSparkConfig(**asdict(args), markov_rank=config['markov_rank'])


def expected_weights(args):
    from ..runtime.drafters.muse_dflash import expected_weights as assistant_weights

    return {**assistant_weights(args),
            'markov_head.markov_w1.weight': ('float', [args.vocab_size, args.markov_rank]),
            'markov_head.markov_w2.weight': ('float', [args.vocab_size, args.markov_rank])}


def inspect_drafter(path, target):
    return muse_dflash.inspect_drafter(path, target, config_parser=config_from_hf,
                                     weight_schema=expected_weights)


def load_drafter(record, target_model):
    from ..runtime.drafters.muse_dspark import MuseDSparkDraftModel

    return muse_dflash.load_drafter(record, target_model, model_class=MuseDSparkDraftModel)
