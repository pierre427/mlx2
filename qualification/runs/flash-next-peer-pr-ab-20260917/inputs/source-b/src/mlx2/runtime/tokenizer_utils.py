# SPDX-License-Identifier: MIT
# Adapted tokenizer primitives; see provenance/.
import copy
import importlib
import json
from functools import partial
from json import JSONDecodeError
from typing import Any, Dict, Optional
from transformers import AutoTokenizer


class StreamingDetokenizer:
    """The streaming detokenizer interface so that we can detokenize one token at a time.

    Example usage is as follows:

        detokenizer = ...

        # Reset the tokenizer state
        detokenizer.reset()

        for token in generate(...):
            detokenizer.add_token(token.item())

            # Contains the whole text so far. Some tokens may not be included
            # since it contains whole words usually.
            detokenizer.text

            # Contains the printable segment (usually a word) since the last
            # time it was accessed
            detokenizer.last_segment

            # Contains all the tokens added so far
            detokenizer.tokens

        # Make sure that we detokenize any remaining tokens
        detokenizer.finalize()

        # Now detokenizer.text should match tokenizer.decode(detokenizer.tokens)
    """

    __slots__ = ("text", "tokens", "offset")

    def reset(self):
        raise NotImplementedError()

    def add_token(self, token):
        raise NotImplementedError()

    def finalize(self):
        raise NotImplementedError()

    @property
    def last_segment(self):
        """Return the last segment of readable text since last time this property was accessed."""
        text = self.text
        segment = text[self.offset :]
        self.offset = len(text)
        return segment


class NaiveStreamingDetokenizer(StreamingDetokenizer):
    """NaiveStreamingDetokenizer relies on the underlying tokenizer
    implementation and should work with every tokenizer.

    Its complexity is O(T^2) where T is the longest line since it will
    repeatedly detokenize the same tokens until a new line is generated.
    """

    def __init__(self, tokenizer):
        self._tokenizer = tokenizer
        self._tokenizer.decode([0])
        probe = tokenizer.encode("a ,b", add_special_tokens=False)
        self._clean_spaces = " ," not in tokenizer.decode(probe)
        self.reset()

    def reset(self):
        self.offset = 0
        self.tokens = []
        self._text = ""
        self._current_tokens = []
        self._current_text = ""

    def add_token(self, token):
        self._current_tokens.append(token)
        self.tokens.append(token)

    def finalize(self):
        self._text += self._tokenizer.decode(self._current_tokens)
        self._current_tokens = []
        self._current_text = ""

    @property
    def text(self):
        if self._current_tokens:
            self._current_text = self._tokenizer.decode(self._current_tokens)
            if self._current_text.endswith("�") or (
                self._clean_spaces
                and len(self._current_text) > 0
                and (self._current_text[-1] == " ")
            ):
                self._current_text = self._current_text[:-1]
        if self._current_text and self._current_text[-1] == "\n":
            self._text += self._current_text
            self._current_tokens.clear()
            self._current_text = ""
        return self._text + self._current_text


class SPMStreamingDetokenizer(StreamingDetokenizer):
    """A streaming detokenizer for SPM models.

    It adds tokens to the text if the next token starts with the special SPM
    underscore which results in linear complexity.
    """

    def __init__(self, tokenizer, trim_space=True):
        self.trim_space = trim_space
        self._sep = "▁".encode()
        self.tokenmap = [""] * (max(tokenizer.vocab.values()) + 1)
        for value, tokenid in tokenizer.vocab.items():
            if value.startswith("<0x"):
                self.tokenmap[tokenid] = bytes([int(value[3:5], 16)])
            else:
                self.tokenmap[tokenid] = value.encode()
        self.reset()

    def reset(self):
        self.offset = 0
        self._unflushed = b""
        self.text = ""
        self.tokens = []

    def _try_flush(self, force=False):
        text = self._unflushed.replace(self._sep, b" ").decode("utf-8", "replace")
        if not force and text.endswith("�"):
            return
        if not self.text and self.trim_space and text and (text[0] == " "):
            text = text[1:]
        self.text += text
        self._unflushed = b""

    def add_token(self, token):
        self.tokens.append(token)
        v = self.tokenmap[token]
        self._unflushed += v
        self._try_flush()

    def finalize(self):
        self._try_flush(force=True)
        self._unflushed = b""


class BPEStreamingDetokenizer(StreamingDetokenizer):
    """A streaming detokenizer for OpenAI style BPE models.

    It adds tokens to the text if the next token starts with a space similar to
    the SPM detokenizer.
    """

    _byte_decoder = None

    def __init__(self, tokenizer):
        self.tokenmap = [None] * len(tokenizer.vocab)
        for value, tokenid in tokenizer.vocab.items():
            self.tokenmap[tokenid] = value
        self.reset()
        self.make_byte_decoder()

    def reset(self):
        self.offset = 0
        self._unflushed = ""
        self.text = ""
        self.tokens = []

    def _decode_bytes(self, seq):
        barr = bytearray()
        for c in seq:
            res = self._byte_decoder.get(c, False)
            if res:
                barr.append(res)
            else:
                barr.extend(bytes(c, "utf-8"))
        return barr.decode("utf-8", "replace")

    def _maybe_trim_space(self, current_text):
        if len(current_text) == 0:
            return current_text
        elif current_text[0] != " ":
            return current_text
        elif not self.text:
            return current_text[1:]
        return current_text

    def add_token(self, token):
        self.tokens.append(token)
        v = self.tokenmap[token] if token < len(self.tokenmap) else "!"
        self._unflushed += v
        text = self._decode_bytes(self._unflushed)
        if not text.endswith("�") and (
            not (len(v) == 1 and self._byte_decoder.get(v[0]) == 32)
        ):
            self.text += self._maybe_trim_space(text)
            self._unflushed = ""

    def finalize(self):
        current_text = bytearray(
            (self._byte_decoder[c] for c in self._unflushed)
        ).decode("utf-8", "replace")
        self.text += self._maybe_trim_space(current_text)
        self._unflushed = ""

    @classmethod
    def make_byte_decoder(cls):
        """See https://github.com/openai/gpt-2/blob/master/src/encoder.py for the rationale."""
        if cls._byte_decoder is not None:
            return
        char_to_bytes = {}
        limits = [
            0,
            ord("!"),
            ord("~") + 1,
            ord("¡"),
            ord("¬") + 1,
            ord("®"),
            ord("ÿ") + 1,
        ]
        n = 0
        for i, (start, stop) in enumerate(zip(limits, limits[1:])):
            if i % 2 == 0:
                for b in range(start, stop):
                    char_to_bytes[chr(2**8 + n)] = b
                    n += 1
            else:
                for b in range(start, stop):
                    char_to_bytes[chr(b)] = b
        cls._byte_decoder = char_to_bytes


def _infer_thinking(tokenizer):
    vocab = tokenizer.get_vocab()
    THINK_TOKENS = [
        ("<think>", "</think>"),
        ("<think:opensource>", "</think:opensource>"),
        ("<longcat_think>", "</longcat_think>"),
    ]
    for think_start, think_end in THINK_TOKENS:
        if think_start in vocab and think_end in vocab:
            return (think_start, think_end, (vocab[think_start],), (vocab[think_end],))
    if "<|channel>" in vocab and "<channel|>" in vocab:
        think_start = "<|channel>thought"
        think_end = "<channel|>"
        return (
            think_start,
            think_end,
            tuple(tokenizer.encode(think_start, add_special_tokens=False)),
            tuple(tokenizer.encode(think_end, add_special_tokens=False)),
        )
    return (None, None, None, None)


class TokenizerWrapper:
    """A wrapper that combines an HF tokenizer and a detokenizer.

    Accessing any attribute other than the ``detokenizer`` is forwarded to the
    huggingface tokenizer.
    """

    def __init__(
        self,
        tokenizer,
        detokenizer_class=NaiveStreamingDetokenizer,
        eos_token_ids=None,
        chat_template=None,
        tool_call_start=None,
        tool_call_end=None,
        tool_parser=None,
    ):
        self._tokenizer = tokenizer
        self._detokenizer_class = detokenizer_class
        self._eos_token_ids = (
            set(eos_token_ids)
            if eos_token_ids is not None
            else {tokenizer.eos_token_id}
        )
        (
            self._think_start,
            self._think_end,
            self._think_start_tokens,
            self._think_end_tokens,
        ) = _infer_thinking(tokenizer)
        self._chat_template = chat_template
        self.has_chat_template = (
            tokenizer.chat_template is not None or chat_template is not None
        )
        self._tool_parser = tool_parser
        self._tool_call_start = tool_call_start
        self._tool_call_end = tool_call_end
        self._tool_call_start_tokens = None
        self._tool_call_end_tokens = None
        if tool_call_start is not None:
            self._tool_call_start_tokens = tuple(
                tokenizer.encode(tool_call_start, add_special_tokens=False)
            )
            self._tool_call_end_tokens = tuple(
                tokenizer.encode(tool_call_end, add_special_tokens=False)
            )

    def apply_chat_template(self, *args, tokenize=True, **kwargs):
        if "enable_thinking" not in kwargs:
            kwargs["enable_thinking"] = self.has_thinking
        if self._chat_template is not None:
            out = self._chat_template(*args, **kwargs)
            if tokenize:
                out = self._tokenizer.encode(out, add_special_tokens=False)
            return out
        kwargs["return_dict"] = False
        return self._tokenizer.apply_chat_template(*args, tokenize=tokenize, **kwargs)

    def add_eos_token(self, token: str):
        token_id = None
        try:
            token_id = int(token)
        except ValueError:
            token_id = self._tokenizer.convert_tokens_to_ids(token)
        if token_id is None:
            raise ValueError(f"'{token}' is not a token for this tokenizer")
        self._eos_token_ids.add(token_id)

    @staticmethod
    def _find(tokens, sequence, start=None, end=None, reverse=False):
        start = max(start or 0, 0)
        end = end or len(tokens)
        outer_loop = (
            range(end - len(sequence), start - 1, -1)
            if reverse
            else range(start, end - len(sequence) + 1)
        )
        for i in outer_loop:
            if tokens[i] == sequence[0]:
                if all((tokens[i + j] == sequence[j] for j in range(1, len(sequence)))):
                    return i
        return -1

    def find_think_start(self, tokens, start=None, end=None):
        return self._find(tokens, self._think_start_tokens, start=start, end=end)

    def rfind_think_start(self, tokens, start=None, end=None):
        return self._find(
            tokens, self._think_start_tokens, start=start, end=end, reverse=True
        )

    def find_think_end(self, tokens, start=None, end=None):
        return self._find(tokens, self._think_end_tokens, start=start, end=end)

    def rfind_think_end(self, tokens, start=None, end=None):
        return self._find(
            tokens, self._think_end_tokens, start=start, end=end, reverse=True
        )

    @property
    def has_thinking(self):
        return self._think_start is not None

    @property
    def think_start(self):
        return self._think_start

    @property
    def think_start_id(self):
        if self._think_start_tokens is None:
            return None
        if len(self._think_start_tokens) > 1:
            raise ValueError("The start thinking sequence is more than 1 token")
        return self._think_start_tokens[0]

    @property
    def think_start_tokens(self):
        return self._think_start_tokens

    @property
    def think_end(self):
        return self._think_end

    @property
    def think_end_id(self):
        if self._think_end_tokens is None:
            return None
        if len(self._think_end_tokens) > 1:
            raise ValueError("The end thinking sequence is more than 1 token")
        return self._think_end_tokens[0]

    @property
    def think_end_tokens(self):
        return self._think_end_tokens

    @property
    def has_tool_calling(self):
        return self._tool_call_start is not None

    @property
    def tool_call_start(self):
        return self._tool_call_start

    @property
    def tool_call_start_tokens(self):
        return self._tool_call_start_tokens

    @property
    def tool_call_end(self):
        return self._tool_call_end

    @property
    def tool_call_end_tokens(self):
        return self._tool_call_end_tokens

    @property
    def tool_parser(self):
        return self._tool_parser

    @property
    def detokenizer(self):
        """
        Get a stateful streaming detokenizer.
        """
        prototype = self.__dict__.get("_detokenizer_prototype")
        if prototype is None:
            prototype = self._detokenizer_class(self)
            self.__dict__["_detokenizer_prototype"] = prototype
        detokenizer = object.__new__(type(prototype))
        detokenizer.__dict__.update(prototype.__dict__)
        detokenizer.reset()
        return detokenizer

    def __getattr__(self, attr):
        if attr == "detokenizer":
            return self._detokenizer
        elif attr == "eos_token_ids":
            return self._eos_token_ids
        elif attr.startswith("_"):
            return self.__getattribute__(attr)
        else:
            return getattr(self._tokenizer, attr)

    def __setattr__(self, attr, value):
        if attr in {"detokenizer", "eos_token_ids"}:
            if attr == "detokenizer":
                raise AttributeError("Cannot set the detokenizer.")
            elif attr == "eos_token_ids":
                self._eos_token_ids = set(value) if value is not None else set()
        elif attr.startswith("_"):
            super().__setattr__(attr, value)
        else:
            setattr(self._tokenizer, attr, value)
