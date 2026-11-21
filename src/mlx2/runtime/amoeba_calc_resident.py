# SPDX-License-Identifier: LicenseRef-Pierre-Lamy-Calculator-NC-ND-1.0
# Copyright (c) 2026 Pierre Lamy
# Terms and scope: provenance/LICENSE-calculator.txt; provenance/LICENSE-calculator-scope.json
"""Device-resident calculator: bytes -> checked postfix -> exact result -> token IDs.

All admission decisions are device data. Python only supplies immutable vocabulary
tables, request UTF-8 at the API boundary, array shapes and kernel dispatches. The
portable header is also compiled as C++ by the functional oracle; it is never a
CPU fallback in serving. Unsupported *complete* candidates leave logits unchanged.
"""

from functools import lru_cache
from pathlib import Path

PARSER = "calc-parse-v5-resident"
SOURCE_BYTES = 4096
PROGRAM_WORDS = 1 + 256 * 4
STATE_WORDS = 288
_TABLES = {}
_ANSWER_PIECE = 4  # Shared with RC_ANSWER_PIECE in the Metal header.
_MAX_ANSWER_PIECE_BYTES = 16


def vocabulary_tables(tokenizer):
    """CPU initialization only. Preserve byte BPE and SentencePiece spellings."""
    import numpy as np
    import mlx.core as mx
    from mlx2.structured_automaton import token_bytes_value

    key = (id(tokenizer), len(tokenizer))
    if key not in _TABLES:
        pool = bytearray()
        offsets = []
        lengths = []
        cues = []
        characters = [-1] * 128
        boundary = []
        special_ids = frozenset(tokenizer.all_special_ids)
        for i in range(len(tokenizer)):
            offsets.append(len(pool))
            try:
                piece = token_bytes_value(tokenizer, i)
            except (ValueError, TypeError, UnicodeError):
                piece = b"\xff"
            pool.extend(piece)
            lengths.append(len(piece))
            # An equals sign, closing delimiter or command can straddle BPE
            # pieces. A token with only trailing space may expose the cue too.
            # This is a hint. Metal checks the actual multi-token byte suffix.
            tail = piece.rstrip(b" \t\r")
            cues.append(
                int(
                    bool(piece)
                    and (
                        not tail
                        or tail[-1] in b"=)]$es"
                        or b"=" in piece
                        or b"\\hline" in piece
                    )
                )
            )
            if len(piece) == 1 and piece[0] < 128 and i not in special_ids:
                if characters[piece[0]] < 0 and tokenizer.decode(
                    [i], clean_up_tokenization_spaces=False
                ) == chr(piece[0]):
                    characters[piece[0]] = i
            trimmed = piece.lstrip(b" \t")
            boundary_kind = (
                2
                if piece and not trimmed
                else int(
                    bool(trimmed)
                    and (
                        48 <= trimmed[0] <= 57
                        or trimmed.startswith(
                            (
                                b".",
                                b",",
                                b"+",
                                b"-",
                                b"*",
                                b"/",
                                b"^",
                                b"%",  # "0.25" then "%" would read as another value
                                b"\\times",
                                b"\\cdot",
                                b"\\div",
                                "×".encode(),
                                "÷".encode(),
                            )
                        )
                    )
                )
            )
            if (i not in special_ids and 0 < len(piece) <= _MAX_ANSWER_PIECE_BYTES
                    and all(byte in b" \t0123456789.-" for byte in piece)):
                boundary_kind |= _ANSWER_PIECE
            boundary.append(boundary_kind)
        if any(characters[ord(c)] < 0 for c in "0123456789.-"):
            raise ValueError(
                "resident calculator requires exact single-byte numeric tokens"
            )
        with mx.stream(mx.gpu):
            arrays = (
                mx.array(np.frombuffer(pool, dtype=np.uint8)),
                mx.array(offsets, dtype=mx.int32),
                mx.array(lengths, dtype=mx.int32),
                mx.array(cues, dtype=mx.int32),
                mx.array(characters, dtype=mx.int32),
                mx.array(boundary, dtype=mx.int32),
            )
        if len(_TABLES) >= 4:
            _TABLES.pop(next(iter(_TABLES)))
        _TABLES[key] = (tokenizer, arrays)
    return _TABLES[key][1]


def header():
    return Path(__file__).with_name("amoeba_calc_resident.h").read_text()


@lru_cache(maxsize=5)
def kernel(kind):
    import mlx.core as mx
    from .amoeba_calc_gpu import _HEADER

    bodies = {
        "parse": (
            ["source", "length", "mode"],
            ["program", "normalized"],
            r"""
if(thread_position_in_grid.x)return;
resident_parse(source,length[0],mode[0],program,normalized);
""",
        ),
        "evaluate": (
            ["program", "normalized"],
            ["result"],
            r"""
if(thread_position_in_grid.x)return;
resident_evaluate(program,normalized,result);
""",
        ),
        "prepare": (
            [
                "tokens",
                "pool",
                "offsets",
                "lengths",
                "cues",
                "request",
                "meta",
                "state",
            ],
            ["source", "length"],
            r"""
if(thread_position_in_grid.x)return;
length[0]=0;
if(!meta[2] || state[0]>state[1] || state[4] || state[12])return;
if(meta[1]){
 int a=rc_skip(request,0,meta[3]);
 if(rc_word(request,a,meta[3],"please "))a=rc_skip(request,a+7,meta[3]);
 if(rc_word(request,a,meta[3],"repeat ")||rc_word(request,a,meta[3],"copy ")||rc_word(request,a,meta[3],"quote ")||rc_word(request,a,meta[3],"transcribe ")){length[0]=-4;return;}
 if(meta[3]>0&&meta[3]<=4096){for(int i=0;i<meta[3];i++)source[i]=request[i];length[0]=meta[3];}return;
}
int nt=tokens_shape[0],last=nt?int(tokens[nt-1]):-1;
if(last<0||last>=cues_shape[0]||!cues[last])return;
uchar suffix[32];int suffix_size=0;
for(int t=nt-1;t>=meta[0]&&suffix_size<32;t--){
 int id=int(tokens[t]);if(id<0||id>=offsets_shape[0])return;
 for(int j=lengths[id]-1;j>=0&&suffix_size<32;j--){
  uchar c=pool[offsets[id]+j];
  if(suffix_size==0&&(c==' '||c=='\t'||c=='\r'))continue;
  suffix[suffix_size++]=c;
 }
}
bool column=suffix_size>=6&&suffix[0]=='e'&&suffix[1]=='n'&&suffix[2]=='i'&&suffix[3]=='l'&&suffix[4]=='h'&&suffix[5]=='\\';
int cue=0;
if(suffix_size>=2&&(suffix[0]==')'||suffix[0]==']')&&suffix[1]=='\\')cue=2;
else if(suffix_size&&suffix[0]=='$')cue=suffix_size>=2&&suffix[1]=='$'?2:1;
while(cue<suffix_size&&rc_space(suffix[cue]))cue++;
bool eq=cue<suffix_size&&suffix[cue]=='=';
bool word=cue+6<=suffix_size&&suffix[cue]=='s'&&suffix[cue+1]=='l'&&suffix[cue+2]=='a'&&suffix[cue+3]=='u'&&suffix[cue+4]=='q'&&suffix[cue+5]=='e';
if(!eq&&!word&&!column)return;
// Reconstruct only generated tokens. A line longer than the bounded window is
// refused, never truncated into a valid-looking suffix. No host token decoding.
uchar reversed[4096];int used=0;bool boundary=false;
for(int t=nt-1;t>=meta[0];t--){
 int id=int(tokens[t]);if(id<0||id>=offsets_shape[0])return;
 for(int j=lengths[id]-1;j>=0;j--){
  uchar c=pool[offsets[id]+j];if(c==255)return;
  if(c=='\n' && !column){boundary=true;break;}
  if(used>=4096)return;reversed[used++]=c;
 }if(boundary)break;
}
// Generated text that continues a line begun in the prompt is not a complete
// line: refuse unless the prompt ends with a newline (or there is no prompt).
if(!boundary&&!column&&meta[0]>0){
 if(meta[0]>nt)return;
 int id=int(tokens[meta[0]-1]);
 if(id<0||id>=offsets_shape[0]||lengths[id]<1||pool[offsets[id]+lengths[id]-1]!='\n')return;
}
for(int i=0;i<used;i++)source[i]=reversed[used-1-i];length[0]=used;
""",
        ),
        "plan": (
            [
                "state",
                "result",
                "program",
                "length",
                "characters",
                "tokens",
                "logits",
                "meta",
                "boundary",
                "pool",
                "offsets",
                "lengths",
            ],
            ["next_state"],
            r"""
if(thread_position_in_grid.x)return;
for(int i=0;i<288;i++)next_state[i]=state[i];next_state[10]=-1;
if(length[0]==-4){next_state[12]=1;return;}
int nt=tokens_shape[0];
if(state[2]>=0 && nt>state[3]){
 int last=int(tokens[nt-1]);
 int used = state[11]!=RC_FORCE_TOKENS
     ? rc_match_answer_piece(state,last,pool,offsets,lengths,boundary,boundary_shape[0])
     : (last==state[2] ? 0 : -1);
 if(used<0){
  next_state[0]=next_state[1]=0;next_state[2]=-1;next_state[7]++;return;
 }
 if(state[11]!=RC_FORCE_TOKENS){
  next_state[1]+=used;
  next_state[6]+=used>0;
  next_state[11]=RC_MATCH_ANSWER_BYTES;
 }
}
bool active=next_state[0]>next_state[1];
if(!meta[2])return;
if(state[4]){int last=nt?int(tokens[nt-1]):-1;next_state[4]=(last>=0&&last<boundary_shape[0]&&(boundary[last]&RC_BOUNDARY_MASK)==RC_SPACE_BOUNDARY);next_state[10]=next_state[4]?-2:-1;return;}
if(!active&&next_state[0]>0){next_state[0]=next_state[1]=0;next_state[2]=-1;next_state[4]=1;next_state[10]=-2;return;}
if(!active&&length[0]>0){
 if(program[0]<1||result[0]){next_state[8]++;next_state[9]=program[0]<1?program[0]:result[0];return;}
 int used=result[3],places=result[2],count=0,integer=used-places;
 if(used<1||used>264||places<0||places>200||integer<1)return;
 if(result[1]<0)next_state[16+count++]=characters['-'];
 for(int i=used-1;i>=0;i--){
  if(i==places-1)next_state[16+count++]=characters['.'];
  int d=result[4+i];if(d<0||d>9)return;next_state[16+count++]=characters['0'+d];
 }
 if(meta[1]){if(meta[4]<0)return;next_state[16+count++]=meta[4];}
 if(count>272)return;
 for(int i=0;i<count;i++)if(next_state[16+i]<0||next_state[16+i]>=logits_shape[1])return;
 int first=next_state[16];if(float(logits[first])<=-5e8f)return;
 next_state[0]=count;next_state[1]=0;next_state[5]++;active=true;
 next_state[11]=meta[1]?RC_FORCE_TOKENS:RC_ALLOW_LEADING_SPACE;
}
if(active){
 int cursor=next_state[1],target=next_state[16+cursor];
 if(target<0||target>=logits_shape[1]||float(logits[target])<=-5e8f){next_state[0]=next_state[1]=0;next_state[2]=-1;next_state[7]++;return;}
 // Generated answers consume the sampled token's bytes on the next prefix.
 // Whole-request answers keep the forced token sequence, including EOS.
 if(next_state[11]==RC_FORCE_TOKENS){next_state[1]++;next_state[6]++;}
 next_state[2]=target;next_state[3]=nt;next_state[10]=target;
}
""",
        ),
        "mask": (
            ["logits", "state", "boundary", "pool", "offsets", "lengths"],
            ["output"],
            r"""
uint i=thread_position_in_grid.x;if(i>=logits_shape[1])return;
int target=state[10];
bool ban=false;
if(target>=0 && int(i)!=target)
 ban=state[11]==RC_FORCE_TOKENS || rc_match_answer_piece(state,int(i),pool,offsets,lengths,boundary,boundary_shape[0])<0;
else if(target==-2 && i<boundary_shape[0])
 ban=(boundary[i]&RC_BOUNDARY_MASK)==RC_NUMBER_BOUNDARY;
output[i]=ban?static_cast<T>(float(logits[i])-1e9f):logits[i];
""",
        ),
    }
    inputs, outputs, body = bodies[kind]
    # Keep exact arithmetic helpers out of the parse kernel and prevent massive
    # optimizer inlining of the bounded rational VM in the evaluation kernel.
    common = header()
    if kind == "evaluate":
        common = (
            "#define RESIDENT_ARITHMETIC 1\n"
            + _HEADER.replace("inline ", "__attribute__((noinline)) ")
            + common
        )
    return mx.fast.metal_kernel(
        name="amoeba_resident_" + kind,
        input_names=inputs,
        output_names=outputs,
        header=common,
        source=body,
        compile_options={"math_mode": "safe"},
    )


def parse_device(source, length, mode=0):
    """mode 0: complete request; mode 1: complete generated formula ending '='."""
    import mlx.core as mx

    with mx.stream(mx.gpu):
        # MLX emits a constant pointer for inputs smaller than eight elements.
        # The parser uses a device pointer for both production and short probes.
        if source.size < 8:
            source = mx.pad(source, (0, 8 - source.size))
        return kernel("parse")(
            inputs=[source, length, mx.array([mode], dtype=mx.int32)],
            grid=(1, 1, 1),
            threadgroup=(1, 1, 1),
            output_shapes=[(PROGRAM_WORDS,), (SOURCE_BYTES,)],
            output_dtypes=[mx.int32, mx.uint8],
        )


def evaluate_device(program, normalized):
    import mlx.core as mx

    with mx.stream(mx.gpu):
        return kernel("evaluate")(
            inputs=[program, normalized],
            grid=(1, 1, 1),
            threadgroup=(1, 1, 1),
            output_shapes=[(268,)],
            output_dtypes=[mx.int32],
        )[0]


class ResidentCalculatorProcessor:
    """Per-lane device state. No result/gate/history readback during generation."""

    def __init__(
        self, tileset, tile_id, tokenizer, *, prompt_length=0, request_text=None
    ):
        import mlx.core as mx

        self.tiles = tileset
        self.tile_id = tile_id
        self.row = 0
        self.forward = None
        self.prompt_length = int(prompt_length)
        self.first = True
        self.tables = vocabulary_tables(tokenizer)
        payload = (request_text or "").encode("utf-8")
        self.request_length = len(payload)
        if len(payload) > SOURCE_BYTES:
            self.request_length = 0
            payload = b""
        self.eos = getattr(tokenizer, "eos_token_id", None)
        if self.eos is None:
            raise ValueError("resident calculator requires an EOS token")
        initial = [0] * STATE_WORDS
        initial[2] = -1
        initial[10] = -1
        with mx.stream(mx.gpu):
            self.request = mx.array(list(payload.ljust(8, b"\0")), dtype=mx.uint8)
            self.state = mx.array(initial, dtype=mx.int32)
        self.counter_key = self.tiles.__dict__.get("_resident_serial", 0) + 1
        self.tiles._resident_serial = self.counter_key
        # Hold only the latest small state per active/completed request until a
        # receipt drains it; no per-token history or Python result strings.
        self.entry = {"state": self.state, "seen": [0, 0, 0, 0], "finished": False}
        self.tiles.__dict__.setdefault("_resident_states", {})[
            self.counter_key
        ] = self.entry

    def __del__(self):
        if hasattr(self, "entry"):
            self.entry["finished"] = True

    def bind_row(self, row):
        self.row = int(row)
        self.forward = self.tiles.forward_ids.get(self.tile_id)

    def _binding_valid(self):
        rows, positions = self.tiles.stash_shape.get(self.tile_id, (0, 0))
        return rows == 1 or (
            positions == 1
            and self.forward == self.tiles.forward_ids.get(self.tile_id)
            and 0 <= self.row < rows
        )

    def __call__(self, tokens, logits):
        import mlx.core as mx

        # Shapes and binding IDs are host metadata, never a device gate readback.
        if not self._binding_valid():
            return logits
        with mx.stream(mx.gpu):
            tokens = (
                mx.array(tokens, dtype=mx.int32).reshape(-1)
                if not isinstance(tokens, mx.array)
                else tokens.reshape(-1)
            )
            pool, offsets, lengths, cues, characters, boundary = self.tables
            meta = mx.array(
                [
                    self.prompt_length,
                    int(self.first),
                    1,
                    self.request_length,
                    int(self.eos),
                ],
                dtype=mx.int32,
            )
            source, length = kernel("prepare")(
                inputs=[
                    tokens,
                    pool,
                    offsets,
                    lengths,
                    cues,
                    self.request,
                    meta,
                    self.state,
                ],
                grid=(1, 1, 1),
                threadgroup=(1, 1, 1),
                output_shapes=[(SOURCE_BYTES,), (1,)],
                output_dtypes=[mx.uint8, mx.int32],
            )
            program, normalized = parse_device(source, length, 0 if self.first else 1)
            result = evaluate_device(program, normalized)
            self.state = kernel("plan")(
                inputs=[
                    self.state,
                    result,
                    program,
                    length,
                    characters,
                    tokens,
                    logits,
                    meta,
                    boundary,
                    pool,
                    offsets,
                    lengths,
                ],
                grid=(1, 1, 1),
                threadgroup=(1, 1, 1),
                output_shapes=[(STATE_WORDS,)],
                output_dtypes=[mx.int32],
            )[0]
            output = kernel("mask")(
                inputs=[logits, self.state, boundary, pool, offsets, lengths],
                template=[("T", logits.dtype)],
                grid=(logits.shape[-1], 1, 1),
                threadgroup=(256, 1, 1),
                output_shapes=[logits.shape],
                output_dtypes=[logits.dtype],
            )[0]
            self.entry["state"] = self.state
            self.first = False
            return output


def collect_counters(tileset):
    """Receipt boundary only: synchronize counters, never tokens or parser output."""
    states = tileset.__dict__.get("_resident_states", {})
    calc = tileset.counters["calc"]
    for key, entry in list(states.items()):
        counts = entry["state"][5:9].tolist()
        delta = [a - b for a, b in zip(counts, entry["seen"])]
        entry["seen"] = counts
        calc["plans"] += delta[0]
        calc["gate_open"] += delta[0]
        calc["tokens_written"] += delta[1]
        calc["plan_aborted"] += delta[2]
        calc["resident_refusals"] = calc.get("resident_refusals", 0) + delta[3]
        executions = calc["plan_executions"]
        executions["gpu_resident_metal"] = (
            executions.get("gpu_resident_metal", 0) + delta[0]
        )
        if entry["finished"]:
            states.pop(key, None)
