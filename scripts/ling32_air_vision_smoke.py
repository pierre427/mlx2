#!/usr/bin/env python3
"""Ling-3.2-air vision-mode smoke client (drives a running loopback server).

``images``  : greedy chat requests on generated still images with exact functional
             oracles (a rendered 4-digit code, the colour and shape of one figure,
             a two-image ordering question), plus the route/vision receipts.
``text``    : fixed text-only prompts with their token traces (``return_token_trace``),
             run against a vision-mode and a text-mode server and compared offline.
``refusal`` : requests that must be refused (video part; image to a text-mode server).
``compare`` : token-identity of two ``text`` outputs.

Every output names the checkpoint passed in ``--checkpoint``; no GPU work happens here.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

TEXT_PROMPTS = [
    "List the first five prime numbers, separated by commas.",
    "Translate 'the cat sleeps on the warm windowsill' into French.",
    "In one sentence, what does a compiler do?",
]


def call(base, path, payload=None, timeout=900):
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(base.rstrip("/") + path, data=data,
                                     headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        body = error.read()
        try:
            return error.code, json.loads(body)
        except ValueError:
            return error.code, {"raw": body.decode(errors="replace")[:2000]}


def data_url(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def code_image(code):
    from PIL import Image, ImageDraw, ImageFont

    image = Image.new("RGB", (640, 320), "white")
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 150)
    except OSError:
        font = ImageFont.load_default(size=150)
    draw.text((60, 70), code, fill="black", font=font)
    return image


def shape_image(colour, shape):
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (448, 448), "white")
    draw = ImageDraw.Draw(image)
    box = (104, 104, 344, 344)
    if shape == "circle":
        draw.ellipse(box, fill=colour)
    else:
        draw.rectangle(box, fill=colour)
    return image


def find_trace(body):
    stack = [body]
    while stack:
        item = stack.pop()
        if isinstance(item, dict):
            if "token_trace" in item:
                return item["token_trace"]
            stack.extend(item.values())
        elif isinstance(item, list):
            stack.extend(item)
    return None


def dots_image(count):
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (512, 256), "white")
    draw = ImageDraw.Draw(image)
    for i in range(count):
        x = 60 + i * 150
        draw.ellipse((x, 88, x + 80, 168), fill="red")
    return image


def chat(model, parts, max_tokens=64):
    return {"model": model, "temperature": 0, "max_tokens": max_tokens, "stream": False,
            "enable_thinking": False, "messages": [{"role": "user", "content": parts}]}


def image_part(image):
    return {"type": "image_url", "image_url": {"url": data_url(image)}}


def served_model(base):
    status, body = call(base, "/v1/models")
    if status != 200 or not body.get("data"):
        raise SystemExit(f"/v1/models failed: {status} {body}")
    return body["data"][0]["id"]


def run_images(args):
    model = served_model(args.base_url)
    cases = [
        ("code_4729", [image_part(code_image("4729")),
                       {"type": "text", "text": "What number is written in the image? Reply with the digits only."}],
         lambda text: text.strip().replace(" ", "").startswith("4729")),
        ("red_square", [image_part(shape_image("red", "square")),
                        {"type": "text", "text": "What colour is the shape? Reply with one word."}],
         lambda text: "red" in text.lower()),
        ("blue_circle", [image_part(shape_image("blue", "circle")),
                         {"type": "text", "text": "Is the shape a circle or a square? Reply with one word."}],
         lambda text: "circle" in text.lower()),
        ("two_images", [{"type": "text", "text": "First image:"}, image_part(shape_image("green", "square")),
                        {"type": "text", "text": "Second image:"}, image_part(shape_image("yellow", "circle")),
                        {"type": "text", "text": "What colour is the shape in the second image? One word."}],
         lambda text: "yellow" in text.lower()),
    ]
    if args.extended:
        cases += [
            ("code_3081", [image_part(code_image("3081")),
                           {"type": "text", "text": "Read the number in the image. Digits only."}],
             lambda text: text.strip().replace(" ", "").startswith("3081")),
            ("count_dots", [image_part(dots_image(3)),
                            {"type": "text", "text": "How many red dots are in the image? Reply with a digit."}],
             lambda text: text.strip().startswith("3") or "three" in text.lower()),
            ("word_planet", [image_part(code_image("PLANET")),
                             {"type": "text", "text": "What word is written in the image? One word."}],
             lambda text: "planet" in text.lower()),
        ]
    rows = []
    for name, parts, oracle in cases * args.repeat:
        payload = chat(model, parts)
        payload["return_token_trace"] = True
        status, body = call(args.base_url, "/v1/chat/completions", payload)
        text = ""
        if status == 200:
            text = body["choices"][0]["message"].get("content") or ""
        trace = find_trace(body) if status == 200 else None
        rows.append({"case": name, "status": status, "text": text,
                     "token_ids": (trace or {}).get("token_ids"),
                     "passed": status == 200 and oracle(text),
                     "usage": body.get("usage"), "mlx2": body.get("mlx2"),
                     "error": None if status == 200 else body})
        print(json.dumps({"case": name, "status": status, "text": text[:120], "passed": rows[-1]["passed"]}),
              flush=True)
    _, status_body = call(args.base_url, "/v1/status")
    out = {"checkpoint": args.checkpoint, "mode": "images", "model": model, "rows": rows,
           "passed": all(row["passed"] for row in rows), "status": status_body}
    Path(args.output).write_text(json.dumps(out, indent=1) + "\n")
    return 0 if out["passed"] else 1


def run_text(args):
    model = served_model(args.base_url)
    rows = []
    for prompt in TEXT_PROMPTS:
        payload = chat(model, prompt, max_tokens=48)
        payload["return_token_trace"] = True
        status, body = call(args.base_url, "/v1/chat/completions", payload)
        trace = None
        stack = [body]
        while stack:
            item = stack.pop()
            if isinstance(item, dict):
                if "token_trace" in item:
                    trace = item["token_trace"]
                    break
                stack.extend(item.values())
        rows.append({"prompt": prompt, "status": status,
                     "text": (body.get("choices") or [{}])[0].get("message", {}).get("content"),
                     "token_ids": (trace or {}).get("token_ids")})
    out = {"checkpoint": args.checkpoint, "mode": "text", "model": model, "rows": rows}
    Path(args.output).write_text(json.dumps(out, indent=1) + "\n")
    return 0 if all(row["status"] == 200 and row["token_ids"] for row in rows) else 1


def run_refusal(args):
    model = served_model(args.base_url)
    cases = {"video_part": [{"type": "video_url", "video_url": {"url": "data:video/mp4;base64,AAAA"}},
                            {"type": "text", "text": "describe"}],
             "image": [image_part(shape_image("red", "square")), {"type": "text", "text": "colour?"}]}
    rows = []
    for name in args.cases.split(","):
        status, body = call(args.base_url, "/v1/chat/completions", chat(model, cases[name]))
        rows.append({"case": name, "status": status, "refused": status >= 400, "body": body})
        print(json.dumps({"case": name, "status": status}), flush=True)
    out = {"checkpoint": args.checkpoint, "mode": "refusal", "rows": rows,
           "passed": all(row["refused"] for row in rows)}
    Path(args.output).write_text(json.dumps(out, indent=1) + "\n")
    return 0 if out["passed"] else 1


def run_compare(args):
    a = json.loads(Path(args.reference).read_text())["rows"]
    b = json.loads(Path(args.candidate).read_text())["rows"]
    same = [x["token_ids"] == y["token_ids"] and x["token_ids"] is not None for x, y in zip(a, b)]
    out = {"identical": same, "passed": all(same) and len(a) == len(b)}
    Path(args.output).write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out))
    return 0 if out["passed"] else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("images", "text", "refusal", "compare"))
    parser.add_argument("--base-url")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--cases", default="video_part")
    parser.add_argument("--extended", action="store_true", help="add 3 more image oracles")
    parser.add_argument("--repeat", type=int, default=1,
                        help="run the case list N times (later passes replay APCv2-warm)")
    parser.add_argument("--reference")
    parser.add_argument("--candidate")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    return {"images": run_images, "text": run_text, "refusal": run_refusal,
            "compare": run_compare}[args.mode](args)


if __name__ == "__main__":
    sys.exit(main())
