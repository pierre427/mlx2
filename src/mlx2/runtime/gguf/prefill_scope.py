"""Explicit prompt-only scope for distinct packed dense NAX arithmetic."""
from contextlib import contextmanager
from contextvars import ContextVar
from .reader import GGUFError

_DENSE_PREFILL=ContextVar('mlx2_deepseek_dense_prefill',default=None)


@contextmanager
def dense_prefill_context(policy,*,tokens):
    if policy not in ('reference','nax_fp16','nax_fp16_v2','nax_fp16_v2d'):
        raise GGUFError('unknown dense prefill arithmetic policy')
    if type(tokens) is not int or tokens<=0:
        raise GGUFError('dense prefill context requires positive token count')
    token=_DENSE_PREFILL.set((policy,tokens))
    try:yield
    finally:_DENSE_PREFILL.reset(token)


def dense_prefill_active(rows):
    """Ordinary decode and slab verification never infer scope from width."""
    active=_DENSE_PREFILL.get()
    return type(rows) is int and rows>=16 and active in (('nax_fp16',rows),('nax_fp16_v2',rows),('nax_fp16_v2d',rows))
