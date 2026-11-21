"""Bug sweep 2026-11-21, lane amoeba-corpus (AMCO-1 .. AMCO-4). CPU only: no model, a loopback stub server."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


# ----------------------------------------------------------------------------- AMCO-1: fit slice

def _parity_split_items(n=4000):
    """items split the way amoeba_topics_corpus.py splits them: sha256(group) parity"""
    items = []
    for i in range(n):
        g = f"src/doc-{i}"
        fit = int(hashlib.sha256(g.encode()).hexdigest()[:8], 16) % 2 == 0
        items.append({"id": f"p{i}", "group": g, "split": "fit" if fit else "held"})
    return items


@pytest.mark.parametrize("n", [2, 3, 4, 8])
def test_fit_slice_is_one_in_n_of_the_fit_partition(n):
    import amoeba_topics_donor_gap as G
    items = _parity_split_items()
    fit = [it for it in items if it["split"] == "fit"]
    seen = set()
    for k in range(n):
        class A:
            split, fit_slice, exclude_manifest, min_tokens = "fit", f"{k}/{n}", "", None
        got = G.select_split(items, A)
        share = len(got) / len(fit)
        assert got, f"slice {k}/{n} is empty"
        assert abs(share - 1 / n) < 0.05, f"slice {k}/{n} holds {share:.3f} of the fit partition"
        assert not seen & {it["id"] for it in got}
        seen |= {it["id"] for it in got}
        rec = G.split_record(got, A)
        assert rec["fit_slice"] == f"{k}/{n}" and rec["fit_slice_law"] == G.FIT_SLICE_LAW
    assert seen == {it["id"] for it in fit}            # the N slices partition the fit texts


def test_fit_slice_law_is_recorded_only_for_fit_and_compare_tells_the_laws_apart(tmp_path):
    import amoeba_topics_donor_gap as G

    class A:
        split, fit_slice, exclude_manifest, min_tokens = "held", "", "", None
    assert G.split_record([], A)["fit_slice_law"] is None
    rows = [{"id": f"t{i}", "cond": c, "topics": ["prog.python"], "group": f"g{i}", "nats": 70.0 + i, "bytes": 100,
             "tokens": 60} for i in range(12) for c in ("bare", "prefix")]
    old = {"split": "fit", "fit_slice": "0/4", "exclude_manifest": None, "min_tokens": 64}     # no law: v1 output
    for name, sp in (("host", old), ("donor", {**old, "fit_slice_law": G.FIT_SLICE_LAW})):
        d = tmp_path / name
        d.mkdir()
        (d / "scores.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        (d / "score.json").write_text(json.dumps({"checkpoint": name + "@0", "corpus_sha256": "x", "split": sp}))
    r = subprocess.run([sys.executable, str(ROOT / "scripts/amoeba_topics_donor_gap.py"), "compare", "--boot", "50",
                        "--host", str(tmp_path / "host"), "--donor", str(tmp_path / "donor"), "--out",
                        str(tmp_path / "out")], capture_output=True, text=True)
    assert r.returncode != 0 and "different splits" in (r.stderr + r.stdout)


def test_score_with_an_empty_selection_fails_before_any_backend_loads(tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text("".join(json.dumps({"id": f"h{i}", "split": "held", "group": f"g{i}", "topics": ["math"],
                                          "text": f"held text number {i}"}) + "\n" for i in range(6)))
    out = tmp_path / "out"
    # --model names nothing loadable: reaching the backend would fail with another error
    r = subprocess.run([sys.executable, str(ROOT / "scripts/amoeba_topics_donor_gap.py"), "score", "--backend", "torch",
                        "--model", str(tmp_path / "no-such-model"), "--corpus", str(corpus), "--out", str(out),
                        "--split", "fit", "--fit-slice", "1/4"], capture_output=True, text=True)
    assert r.returncode != 0 and "no texts selected" in r.stderr, r.stderr[-1500:]
    assert not (out / "scores.jsonl").exists() and not (out / "score.json").exists()


# ----------------------------------------------------------------------------- AMCO-2: on-policy resume

class _Stub(BaseHTTPRequestHandler):
    tag = "MODEL_A"
    calls = 0

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        type(self).calls += 1
        out = json.dumps({"choices": [{"text": f" <{type(self).tag} max_tokens={body['max_tokens']}>"}],
                          "usage": {"completion_tokens": 2}}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def log_message(self, *a):
        pass


@pytest.fixture()
def stub():
    _Stub.tag, _Stub.calls = "MODEL_A", 0
    srv = HTTPServer(("127.0.0.1", 0), _Stub)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()


OPENINGS = ["The harbour master logged every vessel that crossed the bar before dawn and compared the tally with",
            "A binary search halves the candidate interval at each step, which is why its running time grows",
            "Granite weathers slowly in a dry climate, yet the same rock crumbles within decades when frost",
            "The committee postponed its vote on the transit levy after two members asked for a revised",
            "Sourdough starters differ from one kitchen to the next because the wild yeasts that settle",
            "When the tide turns in a narrow estuary the current can reverse within minutes and carry"]
TAIL = (" several further remarks that the author added in a later edition, where the argument is laid out again "
        "with more examples, a longer discussion of the exceptions, and a closing note on what remains unsettled "
        "for readers who want to follow the question into the specialist literature on the subject")


def _write_corpus(path, texts, split="fit"):
    path.write_text("".join(json.dumps({"text": t, "domain": "prose", "source": f"{path.stem}-{k}", "split": split})
                            + "\n" for k, t in enumerate(texts)))


def _gen(url, out, corpora, checkpoint="modelA@aaaa", extra=()):
    cmd = [sys.executable, str(ROOT / "scripts/amoeba_onpolicy_gen.py"), "--base-url", url, "--checkpoint", checkpoint,
           "--out", str(out), "--prompt-words", "12", "--workers", "1", "--samples", "1"]
    for c in corpora:
        cmd += ["--corpus", str(c)]
    return subprocess.run(cmd + list(extra), capture_output=True, text=True)


def _interrupt(out, keep):
    """leave --out as a run that died part-way: no gen.jsonl, the first `keep` partial rows"""
    (out / "gen.jsonl").unlink()
    (out / "manifest.json").unlink()
    part = out / "gen.partial.jsonl"
    lines = part.read_text().splitlines()
    part.write_text("".join(x + "\n" for x in lines[:keep]))
    return len(lines)


def test_onpolicy_resume_refuses_other_checkpoint_or_sampling(tmp_path, stub):
    c = tmp_path / "carve.jsonl"
    _write_corpus(c, [o + TAIL for o in OPENINGS])
    out = tmp_path / "gen"
    r = _gen(stub, out, [c])
    assert r.returncode == 0, r.stderr[-1500:]
    assert _interrupt(out, 8) == 12
    _Stub.tag = "MODEL_B"                                 # the server now holds another checkpoint
    before = _Stub.calls
    r = _gen(stub, out, [c], checkpoint="modelB@bbbb", extra=["--max-tokens", "512"])
    assert r.returncode != 0 and "refusing to resume" in r.stderr
    assert "checkpoint: recorded 'modelA@aaaa', now 'modelB@bbbb'" in r.stderr
    assert "max_tokens: recorded 64, now 512" in r.stderr
    assert _Stub.calls == before and not (out / "gen.jsonl").exists()      # no request, nothing published
    r = _gen(stub, out, [c], extra=["--top-p", "0.9"])
    assert r.returncode != 0 and "top_p: recorded None, now 0.9" in r.stderr
    # the settings it was started with still resume; the manifest and every row are checkpoint A's
    _Stub.tag = "MODEL_A"
    r = _gen(stub, out, [c])
    assert r.returncode == 0, r.stderr[-1500:]
    rows = [json.loads(x) for x in (out / "gen.jsonl").read_text().splitlines()]
    man = json.loads((out / "manifest.json").read_text())
    assert len(rows) == 12 and man["rows"] == 12 and man["resumed_rows"] == 8 and "resume_adopted" not in man
    assert all("<MODEL_A max_tokens=64>" in x["text"] for x in rows)


def test_onpolicy_resume_on_another_port_is_the_same_run(tmp_path):
    """The local gen backend serves the same checkpoint on a fresh port at every invocation: base_url is
    recorded, never compared."""
    c = tmp_path / "carve.jsonl"
    _write_corpus(c, [o + TAIL for o in OPENINGS[:3]])
    out = tmp_path / "gen"
    urls = []
    for _ in range(2):
        srv = HTTPServer(("127.0.0.1", 0), _Stub)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        try:
            url = f"http://127.0.0.1:{srv.server_address[1]}"
            urls.append(url)
            r = _gen(url, out, [c])
            assert r.returncode == 0, r.stderr[-1500:]
            if len(urls) == 1:
                _interrupt(out, 4)
        finally:
            srv.shutdown()
    assert urls[0] != urls[1]
    man = json.loads((out / "manifest.json").read_text())
    assert man["rows"] == 6 and man["resumed_rows"] == 4 and "resume_adopted" not in man


def test_onpolicy_partial_without_recorded_settings_is_resumed_and_recorded(tmp_path, stub):
    c = tmp_path / "carve.jsonl"
    _write_corpus(c, [o + TAIL for o in OPENINGS[:3]])
    out = tmp_path / "gen"
    assert _gen(stub, out, [c]).returncode == 0
    _interrupt(out, 4)
    (out / "gen.identity.json").unlink()                  # a partial written before the identity record existed
    r = _gen(stub, out, [c])                              # callers that resume such a partial cannot pass a flag
    assert r.returncode == 0, r.stderr[-1500:]
    man = json.loads((out / "manifest.json").read_text())
    assert man["rows"] == 6 and man["resumed_rows"] == 4 and man["resume_adopted"]


def test_onpolicy_resume_keeps_duplicate_prompts(tmp_path, stub):
    """the same documents in two corpus files give jobs with one resume key: a resumed run must publish as many
    rows as a clean one, each with its own corpus row's source and split"""
    texts = [o + TAIL for o in OPENINGS[:3]]
    c1, c2 = tmp_path / "first.jsonl", tmp_path / "second.jsonl"
    _write_corpus(c1, texts, split="fit")
    _write_corpus(c2, texts, split="held")
    clean = tmp_path / "clean"
    assert _gen(stub, clean, [c1, c2]).returncode == 0
    want = sorted((x["source"], x["split"], x["mode"]) for x in map(json.loads, (clean / "gen.jsonl").read_text().splitlines()))
    greedy = [w for w in want if w[2] == "greedy"]
    assert len(want) == 12 and len(greedy) == 6 and len(set(greedy)) == 6
    for keep in (12, 5):                                  # resume from a whole partial, and from a cut one
        out = tmp_path / f"resumed{keep}"
        assert _gen(stub, out, [c1, c2]).returncode == 0
        _interrupt(out, keep)
        r = _gen(stub, out, [c1, c2])
        assert r.returncode == 0, r.stderr[-1500:]
        got = sorted((x["source"], x["split"], x["mode"]) for x in map(json.loads, (out / "gen.jsonl").read_text().splitlines()))
        assert got == want
        assert json.loads((out / "manifest.json").read_text())["resumed_rows"] == keep


# ----------------------------------------------------------------------------- AMCO-3: eval guard

def _shingles(text):
    import re
    w = re.findall(r"\w+", text.lower())
    return {hashlib.blake2b(" ".join(w[k:k + 13]).encode(), digest_size=8).hexdigest() for k in range(max(1, len(w) - 12))}


def test_eval_guard_catches_a_13_word_overlap_at_every_offset():
    import amoeba_category_build as B
    eval_text = ("Which planet in the solar system has the shortest day, and how long does one full rotation take "
                 "when measured against the distant stars rather than the sun")
    guard = _shingles(eval_text)                          # as amoeba_category_sources.py writes it: every shingle
    shared = eval_text.split()[3:16]                      # exactly 13 consecutive eval words
    lead = ("Our reading group met on Thursday evening to talk through last month's astronomy quiz before the "
            "club newsletter went to print").split()
    tail = "and nobody at the table could agree on an answer until somebody fetched the almanac".split()
    missed = [k for k in range(len(lead) + 1) if not B.shares_guard_shingle(" ".join(lead[:k] + shared + tail), guard)]
    assert missed == []                                   # stride 4 missed every offset that is not a multiple of 4
    assert not B.shares_guard_shingle(" ".join(lead + shared[:12] + tail), guard)     # 12 shared words: kept
    assert not B.shares_guard_shingle(" ".join(lead + tail), guard)
    assert not B.shares_guard_shingle(eval_text, set())


# ----------------------------------------------------------------------------- AMCO-4: source revisions

def test_cti_radio_revision_is_never_a_working_directory_file_hash(tmp_path, monkeypatch):
    import amoeba_topics_corpus as C
    (tmp_path / "README.md").write_text("an unrelated project's readme\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(C, "CTI_RADIO_REV", {})
    monkeypatch.setattr(C, "_REV_CACHE", {})
    for src in ("cti-radio:README.md#3", "cti-radio:threat-model.md#0"):
        rev = C.source_revision(src)
        assert rev == f"cti-radio checkout {C.UNKNOWN_REV}"
        assert "file-sha256" not in rev and "missing" not in rev
    monkeypatch.setattr(C, "CTI_RADIO_REV", {"rev": "git:0123abcde"})
    assert C.source_revision("cti-radio:README.md#3") == "cti-radio checkout git:0123abcde"


def test_checkout_revision_reads_only_the_checkouts_own_repository(tmp_path):
    import amoeba_topics_corpus as C
    env = {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.invalid", "GIT_COMMITTER_NAME": "t",
           "GIT_COMMITTER_EMAIL": "t@example.invalid", "PATH": "/usr/bin:/bin", "HOME": str(tmp_path),
           "GIT_CONFIG_NOSYSTEM": "1"}
    repo = tmp_path / "notes"
    repo.mkdir()
    (repo / "threat-model.md").write_text("assets, actors and trust boundaries\n")
    for cmd in (["init", "-q"], ["add", "."], ["-c", "commit.gpgsign=false", "commit", "-q", "-m", "first"]):
        subprocess.run(["git", *cmd], cwd=repo, check=True, env=env, capture_output=True)
    head = subprocess.run(["git", "rev-parse", "--short=9", "HEAD"], cwd=repo, capture_output=True, text=True).stdout.strip()
    assert C.checkout_revision(repo) == "git:" + head
    (repo / "draft.md").write_text("not committed yet\n")
    assert C.checkout_revision(repo) == "git:" + head + "+dirty"
    (repo / "sub").mkdir()
    assert C.checkout_revision(repo / "sub") is None       # a directory inside some repository is not a checkout
    assert C.checkout_revision(tmp_path / "absent") is None


def test_annotate_keeps_the_builds_revisions_and_never_writes_none(tmp_path, monkeypatch):
    import amoeba_topics_corpus as C
    monkeypatch.setattr(C, "CTI_RADIO_REV", {})
    rows = [{"id": "tp00000", "sha256": "a1", "split": "fit", "topics": ["swe.git"], "source": "gitlog:mlx2#12",
             "license": "generated (own)", "source_revision": "mlx2 git log at 0f9908ec9"},
            {"id": "tp00001", "sha256": "b2", "split": "held", "topics": ["sci.physics"], "source": "corev2:physics#4",
             "license": "generated (own)"},                # an older corpus row without its own record
            {"id": "tp00002", "sha256": "c3", "split": "fit", "topics": ["swe.git"], "source": "gitlog:mlx2#40",
             "license": "generated (own)"}]
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text("".join(json.dumps(r) + "\n" for r in rows))
    C.annotate(corpus)
    got = [json.loads(x) for x in (tmp_path / "provenance.jsonl").read_text().splitlines()]
    assert got[0]["source_revision"] == "mlx2 git log at 0f9908ec9"
    assert got[1]["source_revision"] == f"core corpus v2 {C.UNKNOWN_REV}"
    assert got[2]["source_revision"] == f"mlx2 git log at {C.UNKNOWN_REV}"
    assert not any("None" in g["source_revision"] for g in got)
