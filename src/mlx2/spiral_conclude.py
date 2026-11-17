"""Opt-in client policy: detect a deliberation spiral in streamed reasoning and make the model conclude.

A deliberation spiral (seen on Qwen3.8-27B at xhigh effort on ambiguous GSM8K items) is not a
repetition loop: the model states a candidate answer early, then keeps re-weighing interpretations
without closing its thinking until the token cap. Exact-cycle breakers (vLLM #52677) never see it and
mlx2's server-side ``thinking_guard`` run-on alarm trips far too early on ordinary math reasoning.

This policy is client-side and serving-neutral. ``SpiralDetector`` watches the reasoning text as it
streams. When it fires, the client stops the stream and asks the server for one continuation through
``/v1/completions``: the rendered chat prompt, the reasoning so far, and ``CONCLUDE_SUFFIX`` (a short
"commit to the most natural reading" line followed by ``</think>``). The model then writes its own
answer. Nothing in the server changes.

Status (2026-11-13): implemented, opt-in, off by default, not qualified. A/B on
Qwen3.8-27B-oQ4e-mtp@fb827b65 (200 GSM8K test problems, greedy, xhigh; mlx2 a02a81eb1): 0.950 -> 0.960
accuracy, 515 -> 376 tokens per problem, caps 2 -> 0, and 0 broken runs out of 13 test and 5 train fires
on runs that were correct without it. The server-side guard on the same problems scored 0.940 (it broke
5 correct runs). This client policy applies when callers choose high reasoning effort.
"""

from __future__ import annotations

import collections
import json
import re
import urllib.request
from dataclasses import dataclass, field

HEDGE = re.compile(
    r"\b(maybe|perhaps|hmm+|alternatively|ambiguous|ambiguity|interpret\w*|could be|might be|not sure|"
    r"unclear|but if|what if|suppose|likely|unless|either)\b", re.I)
# A stated candidate answer: "answer 5", "answer is 5", "answer would be $18,400", "#### 2", "=> 39".
CANDIDATE = re.compile(
    r"(?:answer(?:\s+(?:is|would be|might be|could be|likely|maybe|probably|=|:))*|####|=>|→)\s*:?\s*\$?\s*"
    r"(-?\d[\d,]*(?:\.\d+)?)", re.I)

DEFAULTS = {"min_tokens": 2048, "min_candidates": 3, "stall_tokens": 1024, "min_hedges": 6, "window": 512}

# Appended to the reasoning at the fire point; the model then closes its thinking and answers.
CONCLUDE_SUFFIX = ("\n\nI have weighed the interpretations enough. I will commit to the most natural reading "
                   "of the problem and give that answer.\n</think>\n\n")

RECEIPT_SCHEMA = "mlx2.spiral-conclude.v1"


def norm_number(s: str) -> str:
    s = s.replace(",", "")
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s


@dataclass
class SpiralDetector:
    """Causal, streaming. Fires once, when all of these hold:

    tokens >= min_tokens; stated candidates >= min_candidates; tokens since the first candidate
    >= stall_tokens; hedges in the last ``window`` tokens >= min_hedges.
    """

    min_tokens: int = DEFAULTS["min_tokens"]
    min_candidates: int = DEFAULTS["min_candidates"]
    stall_tokens: int = DEFAULTS["stall_tokens"]
    min_hedges: int = DEFAULTS["min_hedges"]
    window: int = DEFAULTS["window"]
    text: str = ""
    tokens: int = 0
    candidates: list = field(default_factory=list)   # (token position, value)
    hedges: list = field(default_factory=list)       # token positions
    fired_at: int | None = None
    _scan: int = 0

    def feed(self, delta: str, tokens: int) -> bool:
        """Append ``delta`` (reasoning text) with ``tokens`` = completion tokens so far; True once it fires."""
        self.text += delta
        self.tokens = tokens
        # Rescan only the tail (a match may straddle a delta boundary); keep 64 chars of overlap.
        start = max(0, self._scan - 64)
        for m in CANDIDATE.finditer(self.text, start):
            if m.end() > self._scan:
                self.candidates.append((tokens, norm_number(m.group(1))))
        for m in HEDGE.finditer(self.text, start):
            if m.end() > self._scan:
                self.hedges.append(tokens)
        self._scan = len(self.text)
        if self.fired_at is None and self.should_fire():
            self.fired_at = tokens
            return True
        return False

    def should_fire(self) -> bool:
        if self.tokens < self.min_tokens or len(self.candidates) < self.min_candidates:
            return False
        if self.stall_tokens and self.tokens - self.candidates[0][0] < self.stall_tokens:
            return False
        recent = sum(1 for t in self.hedges if t > self.tokens - self.window)
        return recent >= self.min_hedges

    def modal_candidate(self):
        """Most-stated candidate (ties: the most recent). Diagnostic only; the model writes its own answer."""
        if not self.candidates:
            return None
        c = collections.Counter(v for _, v in self.candidates)
        top = max(c.values())
        for _, v in reversed(self.candidates):
            if c[v] == top:
                return v

    def receipt(self):
        return {"fired_at": self.fired_at, "tokens": self.tokens, "candidates": len(self.candidates),
                "first_candidate_token": self.candidates[0][0] if self.candidates else None,
                "modal_candidate": self.modal_candidate(),
                "recent_hedges": sum(1 for t in self.hedges if t > self.tokens - self.window),
                "params": {k: getattr(self, k) for k in DEFAULTS}}


def continuation_prompt(rendered_prompt: str, reasoning: str) -> str:
    """The /v1/completions prompt that makes the model conclude from ``reasoning``.

    ``rendered_prompt`` is the chat template rendered with the generation prompt (it ends where the
    assistant's thinking begins); ``reasoning`` is the streamed reasoning text up to the fire point.
    """
    return rendered_prompt + reasoning + CONCLUDE_SUFFIX


def _sse_events(response):
    for raw in response:
        line = raw.decode("utf-8").strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            return
        yield json.loads(data)


def _default_post(url, body, api_key, timeout, stream):
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers, method="POST")
    response = urllib.request.urlopen(request, timeout=timeout)
    if stream:
        return response
    with response:
        return json.load(response)


def chat_with_spiral_conclude(base_url, model, messages, rendered_prompt, count_tokens, *,
                              max_tokens=8192, conclude_tokens=1024, detector=None, api_key=None,
                              timeout=1800.0, extra_body=None, post=_default_post):
    """Stream one chat completion; if a spiral is detected, stop and request a concluding continuation.

    ``rendered_prompt`` must be the same chat template rendering the server uses for ``messages``
    (with the generation prompt and the same effort / thinking options); ``count_tokens(text)`` must
    count tokens with the model's tokenizer. Both come from the caller because the policy is
    client-side. Returns ``{"reasoning", "content", "finish_reason", "receipt"}``; the receipt says
    whether the policy fired and where.
    """
    detector = detector or SpiralDetector()
    body = {"model": model, "messages": messages, "max_tokens": max_tokens, "stream": True,
            **(extra_body or {})}
    reasoning, content, finish = [], [], None
    fired = False
    response = post(base_url.rstrip("/") + "/v1/chat/completions", body, api_key, timeout, True)
    try:
        for event in _sse_events(response):
            choice = (event.get("choices") or [{}])[0]
            delta = choice.get("delta") or {}
            finish = choice.get("finish_reason") or finish
            if delta.get("content"):
                content.append(delta["content"])
            piece = delta.get("reasoning_content")
            if piece and not content:
                reasoning.append(piece)
                if detector.feed(piece, count_tokens("".join(reasoning))):
                    fired = True
                    break
    finally:
        close = getattr(response, "close", None)
        if close:
            close()
    receipt = {"schema": RECEIPT_SCHEMA, "policy": "spiral_conclude", "qualified": False,
               "fired": fired, "detector": detector.receipt()}
    if not fired:
        return {"reasoning": "".join(reasoning), "content": "".join(content), "finish_reason": finish,
                "receipt": receipt}
    text = "".join(reasoning)
    res = post(base_url.rstrip("/") + "/v1/completions",
               {"model": model, "prompt": continuation_prompt(rendered_prompt, text),
                "max_tokens": conclude_tokens, "temperature": (extra_body or {}).get("temperature", 0)},
               api_key, timeout, False)
    choice = res["choices"][0]
    receipt["conclude_tokens"] = (res.get("usage") or {}).get("completion_tokens")
    return {"reasoning": text, "content": choice.get("text") or "", "finish_reason": choice.get("finish_reason"),
            "receipt": receipt}
