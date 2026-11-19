#!/usr/bin/env python3
# SPDX-License-Identifier: LicenseRef-Pierre-Lamy-Calculator-NC-ND-1.0
# Copyright (c) 2026 Pierre Lamy
"""Render the current high-school-level calculator tile one-pager."""

import argparse
from pathlib import Path

from reportlab import rl_config
from reportlab.lib.colors import HexColor, white
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "output/pdf/calculator-tile-explained-resident.pdf"
RELEASE = "https://github.com/pierre427/mlx2/blob/62742aeda49576f543df2ac0c6a3525c008cc559"
RESEARCH = f"{RELEASE}/docs/research/2026-11-17-calculator"
WIDTH, HEIGHT = letter
MARGIN = 36
BODY = WIDTH - 2 * MARGIN
PAPER = HexColor("#F8F8F2")
INK = HexColor("#142F31")
MUTED = HexColor("#466061")
TEAL = HexColor("#087968")
MINT = HexColor("#94F0D3")
SOFT = HexColor("#E6F2EB")
TRACK = HexColor("#E8ECE7")
STOCK = HexColor("#788591")
HAIRLINE = HexColor("#CBD7D0")
BOLD = "Helvetica-Bold"
REGULAR = "Helvetica"


def render(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pdf = canvas.Canvas(str(path), pagesize=letter, invariant=True)
    pdf.setTitle("Exact math inside an AI response")
    pdf.setAuthor("Pierre Lamy")
    pdf.setCreator("mlx2 calculator one-pager")
    pdf.setSubject("GPU-resident calculator tile and three-model evaluation")

    def rect(x, top, w, h, fill, radius=0):
        pdf.setFillColor(fill)
        if radius:
            pdf.roundRect(x, HEIGHT - top - h, w, h, radius, stroke=0, fill=1)
        else:
            pdf.rect(x, HEIGHT - top - h, w, h, stroke=0, fill=1)

    def label(x, baseline, value, size=10, font=REGULAR, color=INK, *, right=False):
        pdf.setFont(font, size)
        pdf.setFillColor(color)
        draw = pdf.drawRightString if right else pdf.drawString
        draw(x, HEIGHT - baseline, value)

    def paragraph(x, top, w, value, size=10.2, leading=13.1, color=INK):
        style = ParagraphStyle(
            "Body", fontName=REGULAR, fontSize=size, leading=leading, textColor=color
        )
        block = Paragraph(value, style)
        _, h = block.wrap(w, HEIGHT)
        block.drawOn(pdf, x, HEIGHT - top - h)
        return top + h

    def line(x1, top, x2, color, weight=0.7):
        pdf.setStrokeColor(color)
        pdf.setLineWidth(weight)
        pdf.line(x1, HEIGHT - top, x2, HEIGHT - top)

    def arrow(x, top, length, color):
        line(x, top, x + length - 4, color, 1.5)
        head = pdf.beginPath()
        head.moveTo(x + length, HEIGHT - top)
        head.lineTo(x + length - 5, HEIGHT - top + 3)
        head.lineTo(x + length - 5, HEIGHT - top - 3)
        head.close()
        pdf.setFillColor(color)
        pdf.drawPath(head, fill=1, stroke=0)

    def link(x, baseline, value, url, size=8.5):
        label(x, baseline, value, size, BOLD, TEAL)
        w = pdf.stringWidth(value, BOLD, size)
        pdf.linkURL(url, (x, HEIGHT - baseline - 3, x + w, HEIGHT - baseline + size), thickness=0)
        return x + w

    rect(0, 0, WIDTH, HEIGHT, PAPER)
    label(MARGIN, 37, "MLX2 RESEARCH  /  CALCULATOR TILE", 9.4, BOLD, TEAL)
    label(WIDTH - MARGIN, 37, "17 NOVEMBER 2026  /  APPLE M5 MAX", 8.6, BOLD, MUTED, right=True)
    label(MARGIN, 83, "Exact math inside an AI response", 33, BOLD)
    paragraph(
        MARGIN, 96, 514,
        "A language model writes the response. A small program called a <b>calculator tile</b> "
        "reads supported math, computes the exact result and helps the model write it.",
        11.6, 15.6,
    )

    # The worked example and the three stages all sit inside the GPU boundary.
    rect(MARGIN, 145, BODY, 143, INK, 12)
    label(51, 167, "INSIDE THE GPU TILE", 8.8, BOLD, MINT)
    label(51, 196, "Calculate (18.75 × 24) ÷ 3", 18, BOLD, white)
    arrow(300, 189, 29, MINT)
    rect(344, 176, 64, 27, MINT, 6)
    label(360, 195, "150", 17, BOLD)
    label(422, 194, "exact answer", 10, REGULAR, white)
    line(51, 216, 561, HexColor("#466461"))

    stages = (
        ("Read", "Spot a math cue or a complete request. Check the whole expression."),
        ("Calculate", "Use Metal to do exact arithmetic with digits and fractions."),
        ("Write", "Turn the result into the model's tokens and guide it to emit them."),
    )
    for i, (title, body) in enumerate(stages):
        x = 51 + 172 * i
        label(x, 237, title, 13, BOLD, MINT)
        paragraph(x, 245, 148, body, 9.8, 12.7, white)
        if i < 2:
            arrow(x + 139, 232, 17, MINT)
    rect(MARGIN, 295, BODY, 23, SOFT, 6)
    label(51, 310, "On the CPU", 9, BOLD, TEAL)
    label(116, 310, "Set up the vocabulary, schedule GPU work, and handle requests and responses.", 9.1)

    label(MARGIN, 350, "More correct answers on three models", 19.5, BOLD)
    label(MARGIN, 373, "Correct answers on the same questions within each row", 9.1, REGULAR, MUTED)
    rect(445, 365.5, 10, 7, STOCK, 2)
    label(460, 373, "Stock", 8.8, BOLD, MUTED)
    rect(501, 365.5, 10, 7, TEAL, 2)
    label(516, 373, "With tile", 8.8, BOLD, TEAL)

    tested = (
        ("Qwen3 0.6B", "Stock vs repaired v4 tile", 3, 605, 800),
        ("Qwen3.8 27B", "DFlash stock vs ordinary + v4 tile", 62, 104, 112),
        ("Gemma 4 26B-A4B", "Ordinary stock vs ordinary + resident tile", 59, 103, 112),
    )
    for i, (model, description, stock, tile, total) in enumerate(tested):
        y = 385 + i * 42
        line(MARGIN, y, WIDTH - MARGIN, HAIRLINE)
        label(MARGIN, y + 16, model, 11.2, BOLD)
        label(MARGIN, y + 31, description, 8.5, REGULAR, MUTED)
        for offset, correct, bar_color, text_color in (
            (7, stock, STOCK, MUTED), (23, tile, TEAL, TEAL)
        ):
            rect(246, y + offset, 238, 9, TRACK, 3)
            length = 238 * correct / total
            rect(246, y + offset, length, 9, bar_color, min(3, length / 2))
            label(570, y + offset + 8, f"{correct}/{total}", 9.9, BOLD, text_color, right=True)
    line(MARGIN, 511, WIDTH - MARGIN, HAIRLINE)
    paragraph(
        MARGIN, 519, BODY,
        "The 0.6B result uses 800 unchanged questions from a set reused after repairs. "
        "The larger models used the same 112-question diagnostic set.",
        9.2, 11.7, MUTED,
    )

    # Evidence that explains what is special about the implementation.
    card_top, card_width = 552, 264
    rect(MARGIN, card_top, card_width, 97, SOFT, 9)
    rect(312, card_top, card_width, 97, SOFT, 9)
    label(48, 573, "One engine, two model families", 12.3, BOLD)
    paragraph(
        48, 583, 240,
        "The same GPU engine runs on Qwen and Gemma without retraining. Gemma solved "
        "<b>32/32</b> hard expressions and <b>16/16</b> large-number or decimal cases. "
        "The tile also reads some LaTeX and counts letters.",
    )
    label(324, 573, "Works with DFlash drafts", 12.3, BOLD)
    paragraph(
        324, 583, 240,
        "DFlash proposes several tokens at once. The tile saves its place and rolls back rejected "
        "steps. Both runs scored <b>31/32</b> on a GSM8K sample and committed about <b>8 tokens</b> "
        "per check.",
    )

    label(MARGIN, 674, "Where it still needs help", 12.4, BOLD)
    paragraph(
        MARGIN, 684, BODY,
        "Unsupported math and verbatim copy requests leave token scores alone. Hard word problems "
        "still need the model to pick the equation. On all 1,319 GSM8K word problems, Qwen 0.6B "
        "got <b>750 right with v4 vs 741 stock</b> at the same token limit. Median prose speed went from "
        "<b>323 to 284 tokens/s</b>.",
        10.1, 12.7,
    )
    line(MARGIN, 735, WIDTH - MARGIN, HAIRLINE)
    label(MARGIN, 751, "Calculator tile by Pierre Lamy  /  Opt-in research prototype", 8.5, BOLD)
    label(MARGIN, 768, "Free noncommercial use with credit. Commercial use and sharing modified copies need permission.", 8.1, REGULAR, MUTED)
    x = link(398, 751, "0.6B", f"{RESEARCH}-results.md")
    x = link(x + 10, 751, "27B", f"{RESEARCH}-27b-transfer.md")
    x = link(x + 10, 751, "Gemma", f"{RESEARCH}-gemma4.md")
    x = link(x + 10, 751, "DFlash", f"{RESEARCH}-speculative.md")
    link(x + 10, 751, "License", f"{RELEASE}/provenance/LICENSE-calculator.txt")
    link(507, 768, "Implementation", f"{RESEARCH}-tile-architecture.md")
    pdf.showPage()
    pdf.save()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resident", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    rl_config.useA85 = False
    render(args.output)
    print(args.output)


if __name__ == "__main__":
    main()
