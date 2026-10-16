"""Build the pinned MiniLM Core ML encoder bundle (``mlx2.coreml-encoder.v1``).

The graph is written directly with the Core ML MIL builder from the verified,
pinned safetensors bytes; no Torch tracing is involved. Every bucket is a
fixed-shape ``[1, S]`` mlprogram compiled to ``models/seq{S}.mlmodelc``.
The build fails closed: the manifest is written last, atomically, and only
after every bucket passed the cosine gate against the ordinary MLX float32 CPU
reference on both ``CPU_ONLY`` and ``CPU_AND_NE``.

Compute-plan counts are anticipated placement, never observed Neural Engine
execution. Importing this module imports no Core ML, MLX or NumPy.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import os
import statistics
import struct
import subprocess
import tempfile
import time
from pathlib import Path

from ..adapters import minilm
from .ane_projection import compiled_model_digest

SCHEMA = "mlx2.coreml-encoder.v1"
MANIFEST_NAME = "mlx2-coreml-encoder.json"
TOKENIZER_NAME = "tokenizer.json"
MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_BUCKETS = (32, 64, 128, 256)
MAX_BUCKETS = 8
MIN_BUCKET = 8
MIN_COSINE = 0.999
NORM_TOLERANCE = 1e-2
# fp16-safe additive key mask: exp(-1e4) is exactly zero in float16/float32.
MASK_NEGATIVE = -1.0e4
MAX_COMPILED_BYTES = 64 << 20
HIDDEN, HEADS, HEAD_DIM, LAYERS, INTERMEDIATE = 384, 12, 32, 6, 1536
SOURCE_LN_EPS = 1e-12
# 1e-12 underflows to zero in float16. Use the smallest *normal* fp16 value
# (2**-14) so the fp16 LayerNorm epsilon survives conversion and is not a
# subnormal that a backend may flush; the parity gate judges the effect.
LN_EPS = 2.0 ** -14
LAYER_NORM_RECORD = {"precision": "float16", "epsilon": LN_EPS, "source_epsilon": SOURCE_LN_EPS}
# Ops counted for the placement fraction; reshape/transpose/cast/gather/
# split/concat/elementwise and const are excluded as data movement.
PLACEMENT_COMPUTE_OPS = frozenset({"matmul", "conv", "linear", "softmax", "layer_norm", "gelu"})
MIN_NE_COMPUTE_FRACTION = 0.9
LAYOUTS = ("sequence_major", "channels_first")
# Both layouts were built and compared (2026-10-16): each plans 73/74 resp.
# 271/272 compute ops on the Neural Engine at every bucket (only the
# embedding LayerNorm after the CPU gather is CPU-planned). The simpler
# sequence-major graph had equal placement, equal-or-better warm CPU_AND_NE
# latency and slightly better parity, so it is the default.
DEFAULT_LAYOUT = "sequence_major"
GPU_LOCK_OWNER = Path("/Users/Shared/mlxuag/gpu.lock/owner.json")
PLAN_NOTE = "anticipated placement, not observed execution"
_HEX = frozenset("0123456789abcdef")

# Realistic, varied parity inputs (never repeated-string filler). Lengths span
# every default bucket; the last three exceed 256 tokens and truncate.
DEFAULT_PARITY_TEXTS = (
    "How do I reset my router?",
    "The museum opens at nine on weekdays.",
    "Quelle est la meilleure saison pour visiter Lisbonne ?",
    "Patients with type 2 diabetes should monitor fasting glucose.",
    "git rebase --onto main feature~3 feature",
    "Ein schneller Zug fährt heute Abend nach München.",
    "Mitochondria convert nutrients into ATP through oxidative phosphorylation.",
    ("The orchestra rehearsed Mahler's fifth symphony until the strings finally agreed on "
    "the tempo of the adagietto, and the conductor sent everyone home early."),
    ("Quarterly revenue rose 7.4% to $3.1 billion, driven by subscription growth in Asia, "
    "while operating margin narrowed because of higher logistics costs."),
    ("To make a quick tomato sauce, soften garlic in olive oil, add crushed tomatoes and a "
    "pinch of salt, then simmer for twenty minutes before stirring in torn basil."),
    ("The appellate court held that the contract's arbitration clause did not cover tort "
    "claims arising after the agreement had expired."),
    ("Saturn's rings are made mostly of water ice, with particles ranging from tiny grains "
    "to chunks as large as a house."),
    ("def median(values):\n    ordered = sorted(values)\n    middle = len(ordered) // 2\n"
    "    return ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2"),
    ("The 1918 influenza pandemic spread along troop transport routes at the end of the "
    "First World War. Public health officials in several American cities closed schools, "
    "theaters and churches, and the cities that acted earliest generally recorded lower peak "
    "death rates. Historians still debate how much these measures changed the total toll."),
    ("When you replace a bicycle chain, count the links on the old chain first, or wrap the "
    "new chain around the largest chainring and the largest rear cog without threading it "
    "through the derailleur and add two links. A chain that is too short can tear the "
    "derailleur off the frame when you shift onto the big cogs."),
    ("Coral reefs cover less than one percent of the ocean floor but support roughly a "
    "quarter of all marine species. Rising sea temperatures cause corals to expel the "
    "symbiotic algae that feed them, a process called bleaching. If temperatures fall "
    "quickly enough, the coral can recover; prolonged heat stress kills it."),
    ("Our onboarding checklist for new engineers: request laptop access on day one, pair "
    "with a buddy on the deployment pipeline during the first week, read the incident "
    "postmortems from the last quarter, and ship one small change to production before the "
    "end of the second week so the whole release path is familiar."),
    ("La photosynthèse transforme l'énergie lumineuse en énergie chimique. Dans les "
    "chloroplastes, la chlorophylle absorbe surtout la lumière bleue et rouge, ce qui "
    "explique pourquoi les feuilles nous apparaissent vertes. Le dioxyde de carbone et "
    "l'eau sont convertis en glucose, et l'oxygène est libéré comme sous-produit."),
    ("The marathon course starts in the old harbor district, climbs gradually along the "
    "river for the first twelve kilometers, and crosses three bridges before the halfway "
    "mark. Runners who start too fast on the opening descent often struggle on the long "
    "exposed stretch beside the airport, where there is no shade and the wind usually "
    "blows straight into their faces. The final two kilometers run downhill through the "
    "park to a finish line beside the cathedral. Water stations appear every five "
    "kilometers, and medical tents are placed at the twenty-five and thirty-five kilometer "
    "points, where organizers say most runners who need help tend to stop."),
    ("Version 4.2 of the scheduling service changes how retries are handled. Previously a "
    "failed job was retried immediately up to five times, which could overload a database "
    "that was already struggling. Retries now use exponential backoff with jitter, starting "
    "at two seconds and capped at five minutes, and the retry budget is shared across all "
    "jobs owned by the same tenant. Operators can inspect the remaining budget through the "
    "admin endpoint, and a new metric reports how many jobs were dropped because their "
    "budget ran out. Existing configuration files keep working, but the old "
    "immediate_retry flag is ignored and logs a deprecation warning at startup."),
    ("My grandmother kept bees behind the house for almost forty years. Every spring she "
    "checked each hive for a laying queen, scraped the propolis from the frames, and added "
    "supers when the colonies grew strong. She never wore gloves, because she said the bees "
    "could feel her hesitation through the leather. In late summer we spun the honey in a "
    "hand-cranked extractor in the cellar, and the whole house smelled of wax for weeks. "
    "The darkest honey came from the buckwheat field across the road; the lightest came "
    "from the clover in June. She labeled every jar with the month and the field."),
    ("Glaciers in the Alps have lost more than half of their volume since the middle of the "
    "nineteenth century, and the pace of loss has accelerated since the 1980s. Researchers "
    "measure the change with stakes drilled into the ice, aerial photographs and satellite "
    "altimetry. In hot summers, a glacier can thin by several meters in a single season. "
    "Meltwater temporarily increases river flow, which helps hydropower and agriculture "
    "downstream, but once a glacier shrinks past a certain point the summer flow declines. "
    "Mountain communities are already adapting by building reservoirs, changing irrigation "
    "schedules and relocating hiking routes that crossed unstable moraines. Some villages "
    "cover parts of the ice with reflective fleece to slow melting near ski lifts, although "
    "scientists stress that this protects only small areas and cannot change the overall "
    "trend. Long-term projections depend strongly on future emissions."),
    ("The committee reviewed three proposals for the new library. The first kept the "
    "historic facade and added a glass reading room behind it; the second demolished the "
    "annex to make space for a children's wing; the third moved the entire collection to "
    "the former railway depot near the station. Residents who spoke at the public meeting "
    "were divided. Several parents supported the children's wing, pointing out that story "
    "time sessions were regularly turned away for lack of room. Older residents worried "
    "that the depot was too far from the bus lines they rely on. The heritage society "
    "objected to any change to the facade, while the finance office noted that the depot "
    "option was cheapest because the building is already owned by the city and its roof "
    "was replaced two years ago. After four hours of discussion, the committee asked staff "
    "to prepare detailed cost estimates for the first two options, including heating, "
    "accessibility upgrades and the temporary relocation of the archive during "
    "construction, and to return with a recommendation before the budget vote in March. "
    "The mayor thanked everyone for attending and promised a second hearing."),
    ("Sourdough baking depends on a stable culture of wild yeast and lactic acid bacteria. "
    "A healthy starter roughly doubles within four to eight hours after feeding at room "
    "temperature, smells pleasantly sour, and is full of small bubbles. To mix the dough, "
    "combine flour and most of the water and let it rest for an hour; this autolyse lets "
    "the flour hydrate and makes the dough easier to stretch. Then add the starter and the "
    "salt with the remaining water. Over the next three to four hours, perform sets of "
    "stretches and folds every thirty minutes, watching for the dough to become smoother, "
    "airier and more elastic. Shape it gently into a tight round, place it seam side up in "
    "a floured basket, and refrigerate it overnight. Cold retarding develops flavor and "
    "makes scoring easier. Bake in a preheated covered pot at a high temperature for twenty "
    "minutes, then uncover it and continue until the crust is deeply browned. Let the loaf "
    "cool for at least an hour before slicing, because the crumb is still setting as steam "
    "escapes. If the bread is dense and gummy, the dough was probably underproofed; if it "
    "spreads flat and tears, it may have overproofed or been shaped too loosely."),
    ("Incident summary: at 02:14 UTC the primary database in the eu-west region stopped "
    "accepting writes after its storage volume filled up. Automated failover promoted the "
    "replica within ninety seconds, but the replica inherited the same disk-usage alarm "
    "threshold and the cleanup job that normally removes old write-ahead logs had been "
    "disabled during a migration two days earlier. Customer-facing impact lasted "
    "thirty-eight minutes: checkout requests failed with a generic error, and roughly "
    "eleven thousand orders had to be retried by the payment service once writes resumed. "
    "No data was lost. Root causes were the disabled cleanup job, which nobody re-enabled "
    "after the migration, and an alert that paged only during business hours. Action items "
    "include re-enabling the job with a test that fails if it is ever disabled for more "
    "than six hours, paging the on-call engineer for storage alerts at any time of day, "
    "adding a runbook step to verify maintenance jobs after every migration, and running a "
    "quarterly game day in which a replica is deliberately filled to confirm that failover "
    "and alerting behave as documented. The incident review will be held on Thursday with "
    "the database, payments and platform teams. Until then, storage alerts are routed to "
    "the platform team's shared pager rotation, and someone checks the cleanup job by hand "
    "every morning before the European business day starts."),
    ("The history of the printing press is often told as the story of a single inventor, but "
    "the technology grew out of many older crafts. Goldsmiths already knew how to cut "
    "punches and cast small metal pieces with great precision. Wine and olive presses "
    "supplied the mechanical principle of applying steady, even pressure. Papermaking had "
    "travelled from China through the Islamic world to Spain and Italy, and by the "
    "fifteenth century paper mills in Europe were producing sheets far cheaper than "
    "parchment. What Gutenberg and his partners contributed in Mainz was a practical system "
    "that combined these pieces: an adjustable hand mould for casting uniform type from an "
    "alloy of lead, tin and antimony, an oil-based ink that would cling to metal, and a "
    "workshop routine that could set, print, distribute and reset type quickly. The first "
    "large project, a Latin Bible of about thirteen hundred pages, took several years and "
    "a great deal of borrowed money, and the partnership ended in a lawsuit. Within fifty "
    "years, however, printing shops had opened in more than two hundred European towns. "
    "Venice became a major center because it had capital, paper supplies, skilled "
    "craftsmen and a trading network that could carry books to distant markets. Printers "
    "experimented with smaller formats, page numbers, title pages and indexes, features "
    "that readers now take for granted. Scholars argue about how quickly the press changed "
    "ideas, since manuscripts continued to be copied for decades and literacy spread "
    "slowly, but almost all agree that cheaper, identical copies made it easier to compare "
    "texts, correct errors and circulate arguments across borders."),
    ("Here is how the home energy audit went last weekend. The technician started outside, "
    "walking around the house with a thermal camera while it was still cold in the "
    "morning, and pointed out a bright streak along the top of the north wall where warm "
    "air was escaping under the eaves. Inside, she closed every window and exterior door, "
    "mounted a large fan in the front doorway, and depressurized the house to measure how "
    "many air changes per hour leaked through gaps. Our result was a little over seven, "
    "which she said is typical for a house built in the 1970s but well above what a "
    "recent build would allow. With the fan running, she used a smoke pencil to find "
    "drafts around the attic hatch, the electrical outlets on outside walls, the dryer "
    "vent and the gap where the plumbing stack passes through the ceiling. The attic "
    "insulation turned out to be only about fifteen centimeters of old fiberglass, "
    "compressed in places where boxes had been stored. Her report recommends air sealing "
    "first, because insulation works poorly when air moves through it, then adding blown "
    "cellulose to reach roughly forty centimeters. She also suggested replacing the "
    "weatherstripping on the back door, insulating the first meter of the hot water pipes, "
    "and having the furnace serviced before winter. The estimated savings are around "
    "twenty percent of our heating bill, and the provincial rebate program covers part of "
    "the cost if the follow-up test shows a measurable improvement. We booked the "
    "contractor for next month and moved everything out of the attic in the meantime."),
)


def validate_buckets(buckets):
    """Strictly ascending unique positive ints, at most 8, ending at 256."""
    if isinstance(buckets, (str, bytes)) or not hasattr(buckets, "__iter__"):
        raise TypeError("buckets must be a sequence of integers")
    values = tuple(buckets)
    if not 1 <= len(values) <= MAX_BUCKETS:
        raise ValueError(f"buckets must contain 1 to {MAX_BUCKETS} sequence lengths")
    if any(type(value) is not int for value in values):
        raise TypeError("bucket sequence lengths must be plain integers")
    if any(not MIN_BUCKET <= value <= minilm.MAX_SEQUENCE_LENGTH for value in values):
        raise ValueError(
            f"bucket sequence lengths must be in [{MIN_BUCKET}, {minilm.MAX_SEQUENCE_LENGTH}]"
        )
    if any(left >= right for left, right in itertools.pairwise(values)):
        raise ValueError("buckets must be strictly ascending and unique")
    if values[-1] != minilm.MAX_SEQUENCE_LENGTH:
        raise ValueError("the largest bucket must equal the 256-token truncation length")
    return values


def bucket_for(token_count, buckets):
    """Smallest bucket that holds ``token_count`` tokens."""
    if type(token_count) is not int or token_count < 1:
        raise ValueError("token count must be a positive integer")
    for size in buckets:
        if token_count <= size:
            return size
    raise ValueError("token count exceeds the largest bucket")


def prepare_output_dir(output_dir):
    """Create a fresh bundle directory; refuse any existing path except an empty dir."""
    path = Path(output_dir).expanduser().absolute()
    if path.is_symlink():
        raise FileExistsError(f"output path is a symlink: {path}")
    if path.exists():
        if not path.is_dir() or any(path.iterdir()):
            raise FileExistsError(f"refusing to reuse a nonempty or non-directory output: {path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.mkdir()
    return path


def _sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def _is_hex64(value):
    return isinstance(value, str) and len(value) == 64 and set(value) <= _HEX


def macos_version():
    try:
        product = subprocess.run(
            ["sw_vers", "-productVersion"], capture_output=True, text=True, check=True, timeout=10,
        ).stdout.strip()
        build = subprocess.run(
            ["sw_vers", "-buildVersion"], capture_output=True, text=True, check=True, timeout=10,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return f"{product} ({build})" if build else product


def builder_source_sha256():
    return _sha256_bytes(Path(__file__).read_bytes())


def gpu_lock_state(path=GPU_LOCK_OWNER):
    """Read the shared GPU lock owner; benchmark-class holders forbid latency runs."""
    try:
        owner = json.loads(Path(path).read_text())
    except (FileNotFoundError, NotADirectoryError):
        # The lock is held while gpu.lock is a directory holding owner.json.
        return {"held": False, "label": None, "latency_allowed": True}
    except (OSError, ValueError):
        return {"held": None, "label": None, "latency_allowed": False}
    label = str(owner.get("label", "")) if isinstance(owner, dict) else ""
    lowered = label.lower()
    blocked = any(word in lowered for word in ("e2e", "bench", "ladder"))
    return {"held": True, "label": label, "latency_allowed": not blocked}


# ---------------------------------------------------------------- source ---

def load_verified_source(model_path):
    """Verify the pinned artifact and return the exact verified bytes."""
    artifact = minilm.inspect_artifact(model_path)
    path = Path(artifact["path"])
    weights = minilm._read_pinned(path, "model.safetensors")
    import io

    minilm.validate_weight_header(io.BytesIO(weights), len(weights))
    tokenizer = minilm._read_pinned(path, TOKENIZER_NAME)
    return artifact, weights, tokenizer


def numpy_weights(data):
    """Float32 tensors decoded from already-validated safetensors bytes."""
    import numpy as np

    length = struct.unpack("<Q", data[:8])[0]
    header = json.loads(data[8:8 + length])
    header.pop("__metadata__", None)
    base = 8 + length
    result = {}
    for name, tensor in header.items():
        if tensor["dtype"] != "F32":
            continue
        start, end = tensor["data_offsets"]
        array = np.frombuffer(data, dtype="<f4", count=(end - start) // 4, offset=base + start)
        array = array.reshape(tensor["shape"]).astype(np.float32)
        if not np.isfinite(array).all():
            raise ValueError(f"MiniLM weight contains nonfinite values: {name}")
        result[name] = array
    return result


def make_tokenizer(tokenizer_bytes, max_length=minilm.MAX_SEQUENCE_LENGTH):
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_str(tokenizer_bytes.decode())
    tokenizer.enable_truncation(max_length, direction="right")
    tokenizer.no_padding()
    return tokenizer


# ----------------------------------------------------------------- graphs ---

def _program(weights, sequence_length, layout):
    if layout == "sequence_major":
        return _program_sequence_major(weights, sequence_length)
    if layout == "channels_first":
        return _program_channels_first(weights, sequence_length)
    raise ValueError(f"unknown graph layout {layout!r}")


def _layer_weights(weights, layer):
    prefix = f"encoder.layer.{layer}."
    scale = HEAD_DIM ** -0.5
    get = weights.__getitem__
    return {
        # Fold the 1/sqrt(d) score scale into the query projection.
        "wq": get(prefix + "attention.self.query.weight") * scale,
        "bq": get(prefix + "attention.self.query.bias") * scale,
        "wk": get(prefix + "attention.self.key.weight"),
        "bk": get(prefix + "attention.self.key.bias"),
        "wv": get(prefix + "attention.self.value.weight"),
        "bv": get(prefix + "attention.self.value.bias"),
        "wo": get(prefix + "attention.output.dense.weight"),
        "bo": get(prefix + "attention.output.dense.bias"),
        "ln1_g": get(prefix + "attention.output.LayerNorm.weight"),
        "ln1_b": get(prefix + "attention.output.LayerNorm.bias"),
        "wi": get(prefix + "intermediate.dense.weight"),
        "bi": get(prefix + "intermediate.dense.bias"),
        "wo2": get(prefix + "output.dense.weight"),
        "bo2": get(prefix + "output.dense.bias"),
        "ln2_g": get(prefix + "output.LayerNorm.weight"),
        "ln2_b": get(prefix + "output.LayerNorm.bias"),
    }


def _embedding_constants(weights, sequence_length):
    import numpy as np

    # token_type ids are all zero and positions are 0..S-1 for every request,
    # so position + type-0 embeddings fold into one constant addend.
    fixed = (
        weights["embeddings.position_embeddings.weight"][:sequence_length]
        + weights["embeddings.token_type_embeddings.weight"][0]
    ).astype(np.float32)
    return weights["embeddings.word_embeddings.weight"], fixed


def _pool_and_normalize(mb, np, hidden_rows, mask_f, sequence_length, *, channels_first):
    # Mean weights mask/count keep the fp16 reduction bounded, then L2.
    count = mb.reduce_sum(x=mask_f, axes=[1], keep_dims=True)  # [1,1]
    share = mb.real_div(x=mask_f, y=count)  # [1,S]
    if channels_first:
        column = mb.reshape(x=share, shape=[1, sequence_length, 1])
        pooled = mb.matmul(x=hidden_rows, y=column)  # [1,H,S] @ [1,S,1] -> [1,H,1]
    else:
        row = mb.reshape(x=share, shape=[1, 1, sequence_length])
        pooled = mb.matmul(x=row, y=hidden_rows)  # [1,1,S] @ [1,S,H] -> [1,1,H]
    pooled = mb.reshape(x=pooled, shape=[1, HIDDEN])
    norm = mb.reduce_l2_norm(x=pooled, axes=[-1], keep_dims=True)
    norm = mb.maximum(x=norm, y=np.float32(1e-6))
    return mb.real_div(x=pooled, y=norm, name="embedding")


def _input_specs(mb, types, sequence_length):
    return [
        mb.TensorSpec(shape=(1, sequence_length), dtype=types.int32),
        mb.TensorSpec(shape=(1, sequence_length), dtype=types.int32),
    ]


def _program_sequence_major(weights, sequence_length):
    """Straightforward [B,S,C] graph: linear, batched heads, LN over last axis."""
    import coremltools as ct
    import numpy as np
    from coremltools.converters.mil import Builder as mb
    from coremltools.converters.mil.mil import types

    S = sequence_length
    table, fixed = _embedding_constants(weights, S)
    layers = [_layer_weights(weights, layer) for layer in range(LAYERS)]

    @mb.program(input_specs=_input_specs(mb, types, S), opset_version=ct.target.macOS14)
    def program(input_ids, attention_mask):
        x = mb.gather(x=table, indices=input_ids, axis=0)  # [1,S,H]
        x = mb.add(x=x, y=fixed[None])
        x = mb.layer_norm(
            x=x, axes=[-1], gamma=weights["embeddings.LayerNorm.weight"],
            beta=weights["embeddings.LayerNorm.bias"], epsilon=LN_EPS,
        )
        mask_f = mb.cast(x=attention_mask, dtype="fp32")
        key_bias = mb.mul(x=mb.sub(x=np.float32(1.0), y=mask_f), y=np.float32(MASK_NEGATIVE))
        key_bias = mb.reshape(x=key_bias, shape=[1, 1, 1, S])

        def heads(value):
            return mb.transpose(x=mb.reshape(x=value, shape=[1, S, HEADS, HEAD_DIM]), perm=[0, 2, 1, 3])

        for w in layers:
            q = heads(mb.linear(x=x, weight=w["wq"], bias=w["bq"]))
            k = heads(mb.linear(x=x, weight=w["wk"], bias=w["bk"]))
            v = heads(mb.linear(x=x, weight=w["wv"], bias=w["bv"]))
            scores = mb.add(x=mb.matmul(x=q, y=k, transpose_y=True), y=key_bias)
            probabilities = mb.softmax(x=scores, axis=-1)
            attended = mb.matmul(x=probabilities, y=v)  # [1,NH,S,D]
            attended = mb.reshape(x=mb.transpose(x=attended, perm=[0, 2, 1, 3]), shape=[1, S, HIDDEN])
            x = mb.layer_norm(
                x=mb.add(x=mb.linear(x=attended, weight=w["wo"], bias=w["bo"]), y=x),
                axes=[-1], gamma=w["ln1_g"], beta=w["ln1_b"], epsilon=LN_EPS,
            )
            hidden = mb.gelu(x=mb.linear(x=x, weight=w["wi"], bias=w["bi"]), mode="EXACT")
            x = mb.layer_norm(
                x=mb.add(x=mb.linear(x=hidden, weight=w["wo2"], bias=w["bo2"]), y=x),
                axes=[-1], gamma=w["ln2_g"], beta=w["ln2_b"], epsilon=LN_EPS,
            )
        return _pool_and_normalize(mb, np, x, mask_f, S, channels_first=False)

    return program


def _program_channels_first(weights, sequence_length):
    """ANE-oriented [B,C,1,S] graph: 1x1 conv, split heads, LN over channels.

    Follows the layout guidance of Apple's "Deploying Transformers on the Apple
    Neural Engine" (2022); re-expressed here in MIL, no code copied.
    """
    import coremltools as ct
    import numpy as np
    from coremltools.converters.mil import Builder as mb
    from coremltools.converters.mil.mil import types

    S = sequence_length
    table, fixed = _embedding_constants(weights, S)
    layers = [_layer_weights(weights, layer) for layer in range(LAYERS)]
    fixed_cf = fixed.T.reshape(1, HIDDEN, 1, S)

    def conv(x, weight, bias):
        return mb.conv(x=x, weight=weight[:, :, None, None], bias=bias)

    def channel_norm(x, gamma, beta):
        return mb.layer_norm(x=x, axes=[1], gamma=gamma, beta=beta,
                             epsilon=LN_EPS)

    @mb.program(input_specs=_input_specs(mb, types, S), opset_version=ct.target.macOS14)
    def program(input_ids, attention_mask):
        x = mb.gather(x=table, indices=input_ids, axis=0)  # [1,S,H]
        x = mb.transpose(x=x, perm=[0, 2, 1])  # [1,H,S]
        x = mb.reshape(x=x, shape=[1, HIDDEN, 1, S])
        x = mb.add(x=x, y=fixed_cf)
        x = channel_norm(x, weights["embeddings.LayerNorm.weight"], weights["embeddings.LayerNorm.bias"])
        mask_f = mb.cast(x=attention_mask, dtype="fp32")
        key_bias = mb.mul(x=mb.sub(x=np.float32(1.0), y=mask_f), y=np.float32(MASK_NEGATIVE))
        key_bias = mb.reshape(x=key_bias, shape=[1, 1, S])

        for w in layers:
            q = mb.split(x=conv(x, w["wq"], w["bq"]), num_splits=HEADS, axis=1)
            k = mb.split(x=conv(x, w["wk"], w["bk"]), num_splits=HEADS, axis=1)
            v = mb.split(x=conv(x, w["wv"], w["bv"]), num_splits=HEADS, axis=1)
            outputs = []
            for head in range(HEADS):
                qh = mb.reshape(x=q[head], shape=[1, HEAD_DIM, S])
                kh = mb.reshape(x=k[head], shape=[1, HEAD_DIM, S])
                vh = mb.reshape(x=v[head], shape=[1, HEAD_DIM, S])
                scores = mb.matmul(x=qh, y=kh, transpose_x=True)  # [1,Sq,Sk]
                probabilities = mb.softmax(x=mb.add(x=scores, y=key_bias), axis=-1)
                outputs.append(mb.matmul(x=vh, y=probabilities, transpose_y=True))  # [1,D,Sq]
            attended = mb.reshape(x=mb.concat(values=outputs, axis=1), shape=[1, HIDDEN, 1, S])
            x = channel_norm(mb.add(x=conv(attended, w["wo"], w["bo"]), y=x), w["ln1_g"], w["ln1_b"])
            hidden = mb.gelu(x=conv(x, w["wi"], w["bi"]), mode="EXACT")
            x = channel_norm(mb.add(x=conv(hidden, w["wo2"], w["bo2"]), y=x), w["ln2_g"], w["ln2_b"])
        rows = mb.reshape(x=x, shape=[1, HIDDEN, S])
        return _pool_and_normalize(mb, np, rows, mask_f, S, channels_first=True)

    return program


def convert_and_compile(program, destination, workdir):
    """fp16 mlprogram (macOS14), saved then compiled to ``destination``."""
    import coremltools as ct

    model = ct.convert(
        program,
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.macOS14,
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_ONLY,
        skip_model_load=True,
    )
    package = Path(workdir) / (Path(destination).stem + ".mlpackage")
    model.save(str(package))
    compiled = Path(ct.models.utils.compile_model(str(package), destination_path=str(destination)))
    if compiled.resolve() != Path(destination).resolve():
        raise RuntimeError("Core ML compiler wrote outside the requested bucket path")
    return compiled


# ----------------------------------------------------------- inspection ---

_DEVICE_NAMES = {"MLCPUComputeDevice": "cpu", "MLNeuralEngineComputeDevice": "neural_engine"}


def compute_plan_counts(model_path):
    """Anticipated CPU_AND_NE per-op placement; ``const`` ops are not counted."""
    import coremltools as ct
    from coremltools.models.compute_plan import MLComputePlan

    plan = MLComputePlan.load_from_path(
        path=str(model_path), compute_units=ct.ComputeUnit.CPU_AND_NE,
    )
    program = plan.model_structure.program
    if program is None:
        raise ValueError("compute plan is not an ML Program")
    counts = {"neural_engine": 0, "cpu": 0, "unknown": 0}
    compute = {"neural_engine": 0, "cpu": 0, "unknown": 0}
    by_operator = {}
    constants = 0

    def visit(block, depth=0):
        nonlocal constants
        if depth > 16:
            raise ValueError("compute plan exceeds nesting limit")
        for operation in block.operations:
            # Plans name ops with their opset prefix ("ios17.matmul").
            operator = operation.operator_name.rsplit(".", 1)[-1]
            if operator == "const":
                constants += 1
            else:
                usage = plan.get_compute_device_usage_for_mlprogram_operation(operation)
                device = None if usage is None else usage.preferred_compute_device
                name = type(device).__name__
                if name == "MLGPUComputeDevice":
                    raise ValueError("CPU_AND_NE compute plan unexpectedly prefers GPU")
                anticipated = _DEVICE_NAMES.get(name, "unknown")
                counts[anticipated] += 1
                if operator in PLACEMENT_COMPUTE_OPS:
                    compute[anticipated] += 1
                slot = by_operator.setdefault(operator, {})
                slot[anticipated] = slot.get(anticipated, 0) + 1
            for child in operation.blocks:
                visit(child, depth + 1)

    for function in program.functions.values():
        visit(function.block)
    total = sum(compute.values())
    counts["neural_engine_compute_fraction"] = (
        round(compute["neural_engine"] / total, 6) if total else 0.0
    )
    return counts, {"by_operator": by_operator, "const_ops_excluded": constants,
                    "compute_ops": compute, "compute_op_set": sorted(PLACEMENT_COMPUTE_OPS)}


def placement_gate(per_bucket, minimum=MIN_NE_COMPUTE_FRACTION):
    """Recorded, non-failing gate: buckets whose compute ops are CPU-planned."""
    passed = sorted(int(size) for size, row in per_bucket.items()
                    if row["neural_engine_compute_fraction"] >= minimum)
    planned_cpu = sorted(int(size) for size in per_bucket if int(size) not in passed)
    return {"min_neural_engine_compute_fraction": minimum,
            "passed_buckets": passed, "cpu_planned_buckets": planned_cpu}


def load_compiled(model_path, compute_unit):
    import coremltools as ct

    units = {"cpu_only": ct.ComputeUnit.CPU_ONLY, "cpu_and_ne": ct.ComputeUnit.CPU_AND_NE}
    if compute_unit not in units:  # never ALL / CPU_AND_GPU
        raise ValueError("compute unit must be cpu_only or cpu_and_ne")
    return ct.models.CompiledMLModel(str(model_path), compute_units=units[compute_unit])


def padded_inputs(token_ids, sequence_length):
    import numpy as np

    count = len(token_ids)
    if not 1 <= count <= sequence_length:
        raise ValueError("token ids do not fit the bucket")
    ids = np.zeros((1, sequence_length), dtype=np.int32)
    mask = np.zeros((1, sequence_length), dtype=np.int32)
    ids[0, :count] = token_ids
    mask[0, :count] = 1
    return {"input_ids": ids, "attention_mask": mask}


def checked_embedding(result):
    """Contract output check: exact [1,384], finite, unit norm within 1e-2."""
    import numpy as np

    if not isinstance(result, dict) or set(result) != {"embedding"}:
        raise ValueError("Core ML encoder returned unexpected output features")
    output = np.asarray(result["embedding"])
    if output.dtype not in (np.float16, np.float32):
        raise ValueError(f"Core ML encoder output dtype {output.dtype} is not float16/float32")
    if output.shape != (1, minilm.DIMENSIONS):
        raise ValueError(f"Core ML encoder output shape {output.shape} is not [1, 384]")
    output = output.astype(np.float64)
    if not np.isfinite(output).all():
        raise ValueError("Core ML encoder output is nonfinite")
    if abs(float(np.linalg.norm(output)) - 1.0) > NORM_TOLERANCE:
        raise ValueError("Core ML encoder output is not unit-normalized")
    return output[0]


def reference_embeddings(weight_bytes, encodings):
    """Ordinary reference: MLX BertModel + normalized_masked_mean, CPU float32."""
    import io

    import mlx.core as mx

    mx.set_default_device(mx.cpu)
    import numpy as np

    from .models.bert import (
        BertArgs,
        BertModel,
        encoder_weights,
        normalized_masked_mean,
    )

    with io.BytesIO(weight_bytes) as stream:
        weights = mx.load(stream, format="safetensors")
    model = BertModel(BertArgs())
    model.load_weights(encoder_weights(weights), strict=True)
    model.eval()
    mx.eval(model.parameters())
    vectors = []
    for encoded in encodings:
        ids = mx.array([encoded.ids], dtype=mx.int32)
        mask = mx.array([encoded.attention_mask], dtype=mx.bool_)
        types = mx.array([encoded.type_ids], dtype=mx.int32)
        vector = normalized_masked_mean(model(ids, mask, types), mask)
        mx.eval(vector)
        vectors.append(np.array(vector[0], dtype=np.float64))
    return vectors


def _cosine(left, right):
    import numpy as np

    return float(np.dot(left, right) / (np.linalg.norm(left) * np.linalg.norm(right)))


def run_parity(bucket_paths, encodings, references, *, compute_units=("cpu_only", "cpu_and_ne")):
    """Each text runs in its own bucket and every larger bucket (padding exercised)."""
    import numpy as np

    per_bucket, rows = {}, []
    for size, path in bucket_paths.items():
        summary = {"cases": 0, "min_cosine": 1.0, "max_abs_error": 0.0}
        for unit in compute_units:
            model = load_compiled(path, unit)
            for index, (encoded, reference) in enumerate(zip(encodings, references)):
                count = len(encoded.ids)
                if count > size:
                    continue
                output = checked_embedding(model.predict(padded_inputs(encoded.ids, size)))
                cosine = _cosine(output, reference)
                error = float(np.max(np.abs(output - reference)))
                rows.append({"bucket": size, "compute_unit": unit, "case": index,
                             "tokens": count, "cosine": cosine, "max_abs_error": error})
                summary["min_cosine"] = min(summary["min_cosine"], cosine)
                summary["max_abs_error"] = max(summary["max_abs_error"], error)
                if unit == compute_units[0]:
                    summary["cases"] += 1
            del model
        if summary["cases"] < 1:
            raise ValueError(f"parity texts leave bucket {size} without cases")
        per_bucket[str(size)] = summary
    return per_bucket, rows


# --------------------------------------------------------------- manifest ---

_TOP_KEYS = {
    "schema", "family", "source", "tokenizer", "representation", "precision",
    "compute_units", "buckets", "builder", "build_parity", "compute_plan",
}
_BUCKET_KEYS = {"sequence_length", "batch", "path", "tree_sha256", "inputs", "output", "output_shape"}


def validate_manifest(manifest):
    """Structural self-check of a manifest before it is published."""
    if not isinstance(manifest, dict) or set(manifest) != _TOP_KEYS:
        raise ValueError("manifest keys differ from mlx2.coreml-encoder.v1")
    if manifest["schema"] != SCHEMA or manifest["family"] != "minilm":
        raise ValueError("manifest schema/family mismatch")
    source = manifest["source"]
    if (
        not isinstance(source, dict)
        or set(source) != {"model", "revision", "artifact_fingerprint", "model_safetensors_sha256"}
        or source["model"] != MODEL_ID or source["revision"] != minilm.MODEL_REVISION
        or not _is_hex64(source["artifact_fingerprint"])
        or source["model_safetensors_sha256"] != minilm.ARTIFACT_FILES["model.safetensors"][1]
    ):
        raise ValueError("manifest source does not name the pinned MiniLM artifact")
    size, digest = minilm.ARTIFACT_FILES[TOKENIZER_NAME]
    if manifest["tokenizer"] != {"path": TOKENIZER_NAME, "bytes": size, "sha256": digest}:
        raise ValueError("manifest tokenizer does not name the pinned tokenizer bytes")
    if manifest["representation"] != minilm.representation_contract():
        raise ValueError("manifest representation differs from the MiniLM contract")
    if manifest["precision"] != "float16" or manifest["compute_units"] != "cpu_and_ne":
        raise ValueError("manifest precision/compute units mismatch")
    buckets = manifest["buckets"]
    if not isinstance(buckets, list) or not buckets:
        raise ValueError("manifest buckets must be a nonempty list")
    lengths = []
    for bucket in buckets:
        if not isinstance(bucket, dict) or set(bucket) != _BUCKET_KEYS:
            raise ValueError("manifest bucket keys mismatch")
        length = bucket["sequence_length"]
        lengths.append(length)
        if (
            bucket["batch"] != 1 or bucket["path"] != f"models/seq{length}.mlmodelc"
            or not _is_hex64(bucket["tree_sha256"])
            or bucket["inputs"] != {"input_ids": "int32", "attention_mask": "int32"}
            or bucket["output"] != "embedding" or bucket["output_shape"] != [1, minilm.DIMENSIONS]
        ):
            raise ValueError(f"manifest bucket {length} is malformed")
    validate_buckets(lengths)
    builder = manifest["builder"]
    if (
        not isinstance(builder, dict)
        or set(builder) != {"tool", "source_sha256", "coremltools", "macos",
                            "minimum_deployment_target", "graph_layout", "layer_norm"}
        or builder["graph_layout"] not in LAYOUTS
        or builder["layer_norm"] != LAYER_NORM_RECORD
        or builder["tool"] != __name__ or not _is_hex64(builder["source_sha256"])
        or builder["minimum_deployment_target"] != "macOS14"
    ):
        raise ValueError("manifest builder record is malformed")
    parity = manifest["build_parity"]
    if (
        not isinstance(parity, dict)
        or set(parity) != {"reference", "threshold_min_cosine", "compute_units", "cases",
                           "per_bucket", "min_cosine", "max_abs_error", "passed"}
        or parity["reference"] != "mlx_bert_cpu_float32"
        or parity["threshold_min_cosine"] != MIN_COSINE
        or parity["compute_units"] != ["cpu_only", "cpu_and_ne"]
        or parity["passed"] is not True
        or not parity["min_cosine"] >= MIN_COSINE
        or set(parity["per_bucket"]) != {str(length) for length in lengths}
        or any(
            set(row) != {"cases", "min_cosine", "max_abs_error"} or row["cases"] < 1
            or not row["min_cosine"] >= MIN_COSINE
            for row in parity["per_bucket"].values()
        )
    ):
        raise ValueError("manifest build parity does not record a passing gate")
    plan = manifest["compute_plan"]
    if (
        not isinstance(plan, dict)
        or set(plan) != {"available", "note", "per_bucket", "placement_gate"}
        or type(plan["available"]) is not bool or plan["note"] != PLAN_NOTE
        or (plan["available"] and set(plan["per_bucket"]) != {str(length) for length in lengths})
        or any(set(row) != {"neural_engine", "cpu", "unknown", "neural_engine_compute_fraction"}
               or not 0.0 <= row["neural_engine_compute_fraction"] <= 1.0
               for row in plan["per_bucket"].values())
        or (plan["available"] and plan["placement_gate"] != placement_gate(plan["per_bucket"]))
        or (not plan["available"] and plan["placement_gate"] is not None)
    ):
        raise ValueError("manifest compute plan record is malformed")
    return manifest


def write_manifest_atomic(bundle_dir, manifest):
    validate_manifest(manifest)
    target = Path(bundle_dir) / MANIFEST_NAME
    if target.exists():
        raise FileExistsError("bundle manifest already exists")
    payload = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    descriptor, temporary = tempfile.mkstemp(prefix=".manifest-", dir=str(bundle_dir))
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
    return target


# ------------------------------------------------------------------ build ---

def build_minilm_bundle(model_path, output_dir, *, buckets=DEFAULT_BUCKETS, parity_texts=None,
                        layout=DEFAULT_LAYOUT, details=None):
    """Build, verify and publish the bundle; returns the manifest.

    ``details`` (optional dict) receives non-manifest evidence: timings,
    per-case parity rows and per-operator anticipated placement.
    """
    buckets = validate_buckets(buckets)
    if layout not in LAYOUTS:
        raise ValueError(f"layout must be one of {LAYOUTS}")
    texts = tuple(DEFAULT_PARITY_TEXTS if parity_texts is None else parity_texts)
    if not texts or any(not isinstance(text, str) or not text for text in texts):
        raise ValueError("parity texts must be nonempty strings")
    bundle = prepare_output_dir(output_dir)
    started = time.perf_counter()
    details = {} if details is None else details
    details.update({"layout": layout, "timings_s": {}})

    import coremltools as ct

    artifact, weight_bytes, tokenizer_bytes = load_verified_source(model_path)
    (bundle / TOKENIZER_NAME).write_bytes(tokenizer_bytes)
    if _sha256_bytes((bundle / TOKENIZER_NAME).read_bytes()) != minilm.ARTIFACT_FILES[TOKENIZER_NAME][1]:
        raise RuntimeError("tokenizer copy is not byte-identical")
    weights = numpy_weights(weight_bytes)
    models_dir = bundle / "models"
    models_dir.mkdir()
    bucket_paths = {}
    with tempfile.TemporaryDirectory(prefix="mlx2-coreml-build-") as workdir:
        for size in buckets:
            tick = time.perf_counter()
            program = _program(weights, size, layout)
            bucket_paths[size] = convert_and_compile(program, models_dir / f"seq{size}.mlmodelc", workdir)
            details["timings_s"][f"convert_compile_seq{size}"] = round(time.perf_counter() - tick, 3)
    digests = {size: compiled_model_digest(path, max_bytes=MAX_COMPILED_BYTES)
               for size, path in bucket_paths.items()}

    tick = time.perf_counter()
    tokenizer = make_tokenizer(tokenizer_bytes)
    encodings = [tokenizer.encode(text, add_special_tokens=True) for text in texts]
    references = reference_embeddings(weight_bytes, encodings)
    details["timings_s"]["reference"] = round(time.perf_counter() - tick, 3)
    tick = time.perf_counter()
    per_bucket, rows = run_parity(bucket_paths, encodings, references)
    details["timings_s"]["parity"] = round(time.perf_counter() - tick, 3)
    details["parity_rows"] = rows
    details["parity_token_counts"] = [len(encoded.ids) for encoded in encodings]
    details["truncated_cases"] = sum(
        1 for encoded in encodings if len(encoded.ids) == minilm.MAX_SEQUENCE_LENGTH
        and encoded.overflowing
    )
    min_cosine = min(row["min_cosine"] for row in per_bucket.values())
    max_error = max(row["max_abs_error"] for row in per_bucket.values())
    passed = min_cosine >= MIN_COSINE
    details["parity_passed"] = passed
    if not passed:
        raise RuntimeError(
            f"Core ML parity gate failed: min cosine {min_cosine:.6f} < {MIN_COSINE}; no manifest written"
        )

    tick = time.perf_counter()
    plan_per_bucket, plan_detail, plan_available = {}, {}, True
    try:
        for size, path in bucket_paths.items():
            counts, extra = compute_plan_counts(path)
            plan_per_bucket[str(size)] = counts
            plan_detail[str(size)] = extra
    except Exception as exc:  # noqa: BLE001 - plan is optional evidence
        plan_available, plan_per_bucket = False, {}
        details["compute_plan_error"] = repr(exc)
    details["compute_plan_detail"] = plan_detail
    details["timings_s"]["compute_plan"] = round(time.perf_counter() - tick, 3)

    # The bytes that passed parity must be the bytes we publish.
    for size, path in bucket_paths.items():
        if compiled_model_digest(path, max_bytes=MAX_COMPILED_BYTES) != digests[size]:
            raise RuntimeError(f"compiled bucket {size} changed during verification")

    size_bytes, tokenizer_digest = minilm.ARTIFACT_FILES[TOKENIZER_NAME]
    manifest = {
        "schema": SCHEMA,
        "family": "minilm",
        "source": {
            "model": MODEL_ID,
            "revision": minilm.MODEL_REVISION,
            "artifact_fingerprint": artifact["identity"]["fingerprint"],
            "model_safetensors_sha256": minilm.ARTIFACT_FILES["model.safetensors"][1],
        },
        "tokenizer": {"path": TOKENIZER_NAME, "bytes": size_bytes, "sha256": tokenizer_digest},
        "representation": minilm.representation_contract(),
        "precision": "float16",
        "compute_units": "cpu_and_ne",
        "buckets": [
            {
                "sequence_length": size, "batch": 1, "path": f"models/seq{size}.mlmodelc",
                "tree_sha256": digests[size],
                "inputs": {"input_ids": "int32", "attention_mask": "int32"},
                "output": "embedding", "output_shape": [1, minilm.DIMENSIONS],
            }
            for size in buckets
        ],
        "builder": {
            "tool": __name__,
            "source_sha256": builder_source_sha256(),
            "coremltools": str(ct.__version__),
            "macos": macos_version(),
            "minimum_deployment_target": "macOS14",
            "graph_layout": layout,
            "layer_norm": dict(LAYER_NORM_RECORD),
        },
        "build_parity": {
            "reference": "mlx_bert_cpu_float32",
            "threshold_min_cosine": MIN_COSINE,
            "compute_units": ["cpu_only", "cpu_and_ne"],
            "cases": sum(row["cases"] for row in per_bucket.values()),
            "per_bucket": per_bucket,
            "min_cosine": min_cosine,
            "max_abs_error": max_error,
            "passed": True,
        },
        "compute_plan": {
            "available": plan_available, "note": PLAN_NOTE, "per_bucket": plan_per_bucket,
            "placement_gate": placement_gate(plan_per_bucket) if plan_available else None,
        },
    }
    write_manifest_atomic(bundle, manifest)
    details["timings_s"]["total"] = round(time.perf_counter() - started, 3)
    return manifest


# ---------------------------------------------------------------- latency ---

def measure_bundle_latency(bundle_dir, manifest, *, text, warmup=5, iterations=20):
    """Exploratory warm latency; not a performance claim.

    Load time is reported separately. Warm calls are interleaved CPU_ONLY /
    CPU_AND_NE with the order alternated per iteration (ABBA), so neither
    unit systematically runs first or cold.
    """
    bundle = Path(bundle_dir)
    tokenizer_bytes = (bundle / TOKENIZER_NAME).read_bytes()
    units = ("cpu_only", "cpu_and_ne")
    result = {}
    for bucket in manifest["buckets"]:
        size = bucket["sequence_length"]
        encoded = make_tokenizer(tokenizer_bytes, max_length=size).encode(text, add_special_tokens=True)
        inputs = padded_inputs(encoded.ids, size)
        models, row = {}, {"tokens": len(encoded.ids)}
        for unit in units:
            tick = time.perf_counter()
            models[unit] = load_compiled(bundle / bucket["path"], unit)
            row[unit] = {"load_s": round(time.perf_counter() - tick, 3)}
        samples = {unit: [] for unit in units}
        for index in range(warmup + iterations):
            order = units if index % 2 == 0 else units[::-1]
            for unit in order:
                tick = time.perf_counter()
                checked_embedding(models[unit].predict(inputs))
                if index >= warmup:
                    samples[unit].append((time.perf_counter() - tick) * 1000)
        for unit in units:
            values = samples[unit]
            row[unit].update({
                "warm_median_ms": round(statistics.median(values), 3),
                "warm_min_ms": round(min(values), 3),
                "warm_max_ms": round(max(values), 3),
                "iterations": iterations, "warmup": warmup,
            })
        row["schedule"] = "interleaved_abba"
        models.clear()
        result[str(size)] = row
    return result


def compare_layouts(model_path, *, buckets=DEFAULT_BUCKETS, layouts=LAYOUTS, parity_texts=None):
    """Build every layout in a scratch directory; plan counts and parity only."""
    buckets = validate_buckets(buckets)
    texts = tuple(DEFAULT_PARITY_TEXTS if parity_texts is None else parity_texts)
    _, weight_bytes, tokenizer_bytes = load_verified_source(model_path)
    weights = numpy_weights(weight_bytes)
    tokenizer = make_tokenizer(tokenizer_bytes)
    encodings = [tokenizer.encode(text, add_special_tokens=True) for text in texts]
    references = reference_embeddings(weight_bytes, encodings)
    result = {}
    with tempfile.TemporaryDirectory(prefix="mlx2-coreml-compare-") as workdir:
        for layout in layouts:
            target = Path(workdir) / layout
            target.mkdir()
            paths = {size: convert_and_compile(_program(weights, size, layout),
                                               target / f"seq{size}.mlmodelc", workdir)
                     for size in buckets}
            plans = {}
            for size, path in paths.items():
                counts, extra = compute_plan_counts(path)
                plans[str(size)] = {**counts, "compute_ops": extra["compute_ops"],
                                    "by_operator": extra["by_operator"]}
            parity, _rows = run_parity(paths, encodings, references)
            result[layout] = {"compute_plan": plans, "parity": parity,
                              "placement_gate": placement_gate(
                                  {size: {"neural_engine_compute_fraction":
                                          row["neural_engine_compute_fraction"]}
                                   for size, row in plans.items()})}
    return result


def bundle_bytes(bundle_dir):
    return sum(item.stat().st_size for item in Path(bundle_dir).rglob("*") if item.is_file())

