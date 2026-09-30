"""Rebuild the source-bound mlx2 architecture review PDF.

Run with the Codex PDF runtime (ReportLab 4.x) or any Python with reportlab.
The public narrative contains no raw prompts, hostnames, credentials or model files.
"""

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    HRFlowable,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

OUT = Path(__file__).with_name("mlx2_deep_dive_feature_architecture_review.pdf")
SOURCE = "343cc092"
DATE = "30 September 2026"
INK = colors.HexColor("#14212c")
TEAL = colors.HexColor("#087f80")
MUTED = colors.HexColor("#5c6871")
PALE = colors.HexColor("#eaf4f2")
RULE = colors.HexColor("#d7e2e5")

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="CoverKicker", parent=styles["Normal"], fontName="Helvetica-Bold", fontSize=11, leading=15, textColor=TEAL, spaceAfter=22))
styles.add(ParagraphStyle(name="CoverTitle", parent=styles["Normal"], fontName="Helvetica-Bold", fontSize=43, leading=48, textColor=INK, spaceAfter=20))
styles.add(ParagraphStyle(name="CoverLead", parent=styles["Normal"], fontSize=15, leading=23, textColor=MUTED, spaceAfter=22))
styles.add(ParagraphStyle(name="Chapter", parent=styles["Normal"], fontName="Helvetica-Bold", fontSize=23, leading=29, textColor=INK, spaceAfter=12))
styles.add(ParagraphStyle(name="Deck", parent=styles["Normal"], fontSize=11, leading=16, textColor=MUTED, spaceAfter=17))
styles.add(ParagraphStyle(name="Sub", parent=styles["Normal"], fontName="Helvetica-Bold", fontSize=12.5, leading=17, textColor=TEAL, spaceBefore=12, spaceAfter=6))
styles.add(ParagraphStyle(name="BodyX", parent=styles["Normal"], fontSize=9.6, leading=14.5, textColor=INK, spaceAfter=8))
styles.add(ParagraphStyle(name="SmallX", parent=styles["Normal"], fontSize=8, leading=11.5, textColor=MUTED, spaceAfter=6))
styles.add(ParagraphStyle(name="CellX", parent=styles["Normal"], fontSize=8, leading=11, textColor=INK))
styles.add(ParagraphStyle(name="CellHeadX", parent=styles["Normal"], fontName="Helvetica-Bold", fontSize=8, leading=11, textColor=colors.white))

story = []


def para(text, style="BodyX"):
    return Paragraph(text, styles[style])


def chapter(number, title, deck):
    story.append(para(f"MLX2 / FEATURE &amp; ARCHITECTURE REVIEW / {number:02d}", "CoverKicker"))
    story.append(para(title, "Chapter"))
    story.append(para(deck, "Deck"))
    story.append(HRFlowable(width="100%", thickness=1, color=RULE, spaceAfter=15))


def section(title, body):
    story.extend([para(title, "Sub"), para(body)])


def bullets(items):
    for item in items:
        story.append(para("&#8226; " + item))


def table(headers, rows, widths):
    data = [[para(x, "CellHeadX") for x in headers]]
    data.extend([[para(str(x), "CellX") for x in row] for row in rows])
    t = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), TEAL),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, PALE]),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ("LINEBELOW", (0, -1), (-1, -1), 0.5, RULE),
    ]))
    story.append(t)
    story.append(Spacer(1, 12))


def page():
    story.append(PageBreak())


story.extend([
    Spacer(1, 0.8 * inch),
    para("APPLE SILICON / INFERENCE SYSTEMS", "CoverKicker"),
    para("mlx2<br/>Feature &amp;<br/>architecture review", "CoverTitle"),
    para("Stateful local inference, from exact prefix reuse to agent APIs and explicit execution evidence.", "CoverLead"),
    HRFlowable(width="100%", thickness=2, color=TEAL, spaceAfter=28),
])
table(["REQUEST", "PLAN", "EXECUTE", "EXPLAIN"], [["API + identity", "Capability + memory", "Batch + state", "Receipt + metrics"]], [1.2 * inch] * 4)
story.append(Spacer(1, 35))
story.append(para("The product in one sentence", "Sub"))
story.append(para("mlx2 is an experimental Apple Silicon inference runtime that puts model-specific acceleration, exact state ownership, physical-memory admission, and agent-facing APIs into one request lifecycle."))
story.append(para(f"Reviewed {DATE}. Private mlx2 main source: {SOURCE}. Historical benchmark receipts retain their own source identities. This review distinguishes implemented, qualified, selected and observed-used behavior.", "SmallX"))
page()

chapter(1, "A coherent inference system", "The strongest feature is the relationship between the parts.")
section("One lifecycle", "Adapters own model tensor math and topology. The common runtime owns request admission, cancellation, APCv2 leases, scheduler cohorts, speculative transactions, output constraints, and route receipts. OpenAI and Anthropic client dialects enter this same lifecycle rather than independent backends.")
table(["Layer", "Responsibility"], [
    ("Model adapters", "Declare topology, cache shape, capabilities and model-specific tensor operations."),
    ("APCv2", "Own exact reusable prefix state, immutable segments, leased branches and revision-bound publication."),
    ("Scheduler", "Admit real memory, choose physical cohorts, share prefill/decode resources and handle backpressure."),
    ("Execution", "Run ordinary decode or a qualified optimized route, then verify and commit state at boundaries."),
    ("Evidence", "Bind artifacts, source and settings to receipts; record what mechanisms actually engaged."),
], [1.25 * inch, 3.55 * inch])
section("A strict vocabulary", "<b>Implemented</b> means code exists. <b>Qualified</b> means matching source-bound evidence passed its defined gate. <b>Selected</b> means policy chose a route. <b>Observed-used</b> means execution counters and a receipt show engagement. These are separate facts, and a configuration flag alone proves none of the later three.")
page()

chapter(2, "Exact state and APCv2", "One prefix-cache authority, with identity and ownership visible at every transition.")
section("Immutable prefixes, mutable branches", "APCv2 publishes exact prefix segments under a revision-bound identity. Active requests hold leases; branch tails remain private until an explicit successful publication. Target and draft sidecars, persistent blocks, sessions, junction snapshots and rolling checkpoints use this ownership model rather than separate cache engines.")
section("Speculation is a transaction", "Embedded MTP, prompt or copy lookup, and external drafters propose tokens. The target verifies them; accepted prefixes and associated KV or recurrent state commit together. Cancellation, mismatch and stale revisions roll back the proposal. Approximate state requires a separately qualified operation and cannot masquerade as an exact APCv2 segment.")
section("Why the 32K gate matters", "In an M5 Qwen3.8 oQ4e MTP width-four context ladder, an 8 GiB APCv2 cap gave only three of four warm reuse hits in each of three repetitions, despite zero swapouts and complete needle accuracy. A 16 GiB cap passed four of four warm lanes in all three repetitions, all 30 needles, and zero swapouts. This qualifies that bounded 32K configuration; it does not settle longer-context admission.")
story.append(para("Evidence: benchmark_results/2026-09-29/qwen38.md; benchmark_results/2026-09-29/README.md. The two capacity profiles have separate source-bound receipts.", "SmallX"))
page()

chapter(3, "Scheduling and memory", "The physical batch is what fits and executes, not merely the number of open HTTP requests.")
section("Admit before compute", "Host availability, process footprint, MLX allocation, cache commitments and a protected service reserve are considered before optional work enters GPU lanes. Preemption, cancellation and deferral preserve useful checkpoints. The memory policy must be read with its model, context and cache cap; a pass on one configuration cannot license a larger one.")
section("Batching and APCv2 together", "A shared scheduler forms compatible cohorts and records actual execution width. Prefix hits avoid repeated prefill while multiple requests contend for compute and memory. The September domain workload sent 20 rounds of 20 graded requests; it exercises concurrency, backpressure and cache behavior, not just isolated decode speed.")
table(["Host / artifact", "20x20 result", "Observed width", "Median aggregate rate"], [
    ("M5 Max 128 GB / Qwen3.8 MLX 4-bit", "400/400; zero HTTP errors; APCv2 reuse; zero swapouts", "4", "46.8 generated tokens/s"),
    ("M3 Pro 36 GB / Qwen3.8 MLX 4-bit", "400/400; zero HTTP errors; APCv2 reuse; zero swapouts", "4", "13.8 generated tokens/s"),
], [1.32 * inch, 2.15 * inch, 0.55 * inch, 0.78 * inch])
story.append(para("The rates are aggregate mixed-workload medians from separately source-bound runs. They are not single-prompt decode rates or an apples-to-apples silicon benchmark.", "SmallX"))
page()

chapter(4, "Acceleration with an ordinary reference", "Optimizations are useful only when their mechanism engages and the target verifies their result.")
table(["Mechanism", "Contract and boundary"], [
    ("Native MTP", "Embedded draft heads propose tokens. Relevant sections latch on; accepted depth and target verification are observed per request."),
    ("Prompt/copy lookup", "Drafts copied from prompt or accepted history are checked against the target. Latches prevent work outside useful spans."),
    ("External drafting", "Separate draft models and TensorFold-style paths need compatible artifacts, memory budget and independent route evidence."),
    ("PLE / model math", "Model-specific recurrent and attention math belongs behind adapters. Offload and fusion choices need host- and artifact-specific gates."),
    ("Approximate KV", "Quantized or reduced-fidelity state is an explicit operation with its own identity; it cannot be silently substituted into exact reuse."),
], [1.2 * inch, 3.6 * inch])
section("Evidence before a default", "Ordinary decode remains available for each model family. Feature qualification can prove correctness and engagement without proving speed benefit. A default is chosen only after the relevant workload, fidelity, memory and thermal gates pass for the model and host. Candidate and partial results are kept visible but not promoted into deployment claims.")
page()

chapter(5, "Product surface and observability", "Agent-facing APIs and operational telemetry share the same state and qualification machinery.")
section("Requests", "Chat and text completions, Responses and Anthropic Messages use bounded request lifecycles. Streaming, disconnect cancellation, output backpressure, tool parsing, JSON/regex/schema constraints and explicit unsupported-feature failures are first-class behavior. Model-specific parsers translate tool formats into the common contract.")
section("Identity and receipts", "Health and status expose readiness, queue depth, memory, APC leases, selected route and qualification identity. Terminal request receipts report actual batching and mechanism use. Source, native dependencies, artifact metadata and serving settings are bound to qualification evidence; changing any relevant part requires new qualification.")
section("mlx2-top", "The read-only Apple Silicon monitor added at source 0d6c4c66 displays logical CPU busy time, available AGX GPU counters, ANE residency, memory and swap, thermal state and mlx2 route receipts. It labels unavailable hardware counters honestly and does not confuse observed NAX calls or rows with system-wide NAX utilization. Six focused tests and a one-shot local run passed; the monitor itself is not a model benchmark.")
page()

chapter(6, "Model support is scoped", "A registry entry, a smoke pass and a fully qualified route are different milestones.")
section("Adapters", "The source includes ordinary paths across dense, MoE, sliding-window and hybrid families, with model-specific tensor math isolated from the scheduler. The current GPT-OSS Harmony adapter separates analysis and commentary from final-channel content, and its thinking guard uses a bounded multi-token close sequence; Puzzle can run hidden low-effort analysis when it cannot skip that channel. Flash-Next, Qwen3.8, Muse-Glimmer, Nemotron Lightning, Laguna and other campaign targets have distinct artifacts and mechanisms. The qualification ledger and per-model pages state which workload and host actually passed.")
section("September campaign", "M5 Qwen3.8 MLX 4-bit passed ordinary smoke and the 20x20 domain workload; M3 repeated the 20x20 on current source. Muse-Glimmer original passed its M5 smoke and 20x20 gate with 398/400 graded correct and zero HTTP errors, but this does not qualify all feature routes. Qwen3.8 oQ4e MTP achieved the bounded 16 GiB / 32K warm APCv2 ladder; the 8 GiB profile failed its warm-reuse gate. Other model pages record their own results and open items.")
section("Offline and candidate paths", "Media, diarization and experimental model capabilities have separate scopes. For example, a narrow offline Nemotron diarization qualification is not a text-serving or streaming qualification. A candidate trace or CPU contract is useful implementation evidence, but it cannot stand in for a served GPU route receipt.")
page()

chapter(7, "How to read the evidence", "The document is a source review and a route to receipts, not a blanket performance guarantee.")
bullets([
    "<b>Smoke</b> checks model load, default route and sane generated output; it does not establish a long-context or feature verdict.",
    "<b>20x20</b> checks heavy concurrent domain traffic, graded answers, HTTP errors, physical width, APCv2 reuse and swap behavior.",
    "<b>Context ladders</b> require repeated cells, thermal control and source-bound policy identity. Cold TTFT and per-stream decode are reported separately from aggregate traffic.",
    "<b>Feature checks</b> need both correctness and nonzero mechanism engagement. A partial combined run may leave individual features unqualified.",
    "<b>Selection and production use</b> require policy and live route receipts beyond a successful qualification experiment.",
])
section("Primary in-repository references", "README.md; docs/ARCHITECTURE.md; docs/SERVING.md; docs/QUALIFICATION-EXPERIMENTS.md; docs/PROVENANCE.md; benchmark_results/2026-09-29/README.md; benchmark_results/2026-09-29/ (per-model pages and scripts); tests/test_system_monitor.py. Historical receipts name the exact source and artifact they tested.")
story.append(para("Review source: private mlx2 main 0d6c4c66 (30 September 2026). Public projections have independent ancestry and are published as reviewed individual commits. This PDF contains no model weights, raw prompts, private paths or credentials.", "SmallX"))


def footer(canvas, doc):
    canvas.saveState()
    w, _ = letter
    canvas.setStrokeColor(RULE)
    canvas.line(0.7 * inch, 0.57 * inch, w - 0.7 * inch, 0.57 * inch)
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(MUTED)
    canvas.drawString(0.7 * inch, 0.39 * inch, f"MLX2 / SOURCE-BOUND REVIEW / {DATE.upper()}")
    canvas.drawRightString(w - 0.7 * inch, 0.39 * inch, str(doc.page))
    canvas.restoreState()


def main():
    doc = SimpleDocTemplate(str(OUT), pagesize=letter, rightMargin=0.78 * inch,
                            leftMargin=0.78 * inch, topMargin=0.72 * inch,
                            bottomMargin=0.78 * inch, title="mlx2: Feature & Architecture Review",
                            author="mlx2", subject=f"Source-bound review {DATE}; source {SOURCE}")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(OUT)


if __name__ == "__main__":
    main()
