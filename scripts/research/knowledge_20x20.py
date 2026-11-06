#!/usr/bin/env python3
"""20x20 domain-knowledge check against a running mlx2 server, with live Prometheus.

20 domains x 20 questions with machine-checkable answers: 8 computed domains
(exact answers generated here) and 12 factual lookup domains. Questions are
sent batched (``--concurrency`` clients), greedy, short answers. ``/metrics``
is scraped every ``--scrape`` seconds into ``metrics.jsonl``; the report adds
throughput, queueing, lane use, TTFT / inter-token / end-to-end latency
(histogram means and p50/p95), prefix-cache hits, memory and HTTP status.

Usage: knowledge_20x20.py OUT_DIR [--url http://127.0.0.1:8282] [--concurrency 8]
"""

from __future__ import annotations

import argparse
import datetime
import json
import random
import re
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
from math import gcd
from pathlib import Path

SYSTEM = "You are a helpful assistant. Answer with just the answer, no explanation."

# --------------------------------------------------------------- question sets


def computed(rng):
    domains = {}

    def add(name, items):
        domains[name] = items

    add("arithmetic", [(f"What is {a} * {b}?", [str(a * b)])
                       for a, b in ((rng.randint(12, 99), rng.randint(12, 99)) for _ in range(20))])
    add("modular_arithmetic", [(f"What is {a} mod {m}?", [str(a % m)])
                               for a, m in ((rng.randint(100, 9999), rng.randint(3, 29)) for _ in range(20))])
    primes = [n for n in range(50, 400) if all(n % d for d in range(2, int(n ** 0.5) + 1))]
    nums = rng.sample(primes, 10) + rng.sample([n for n in range(50, 400) if n not in primes], 10)
    rng.shuffle(nums)
    add("primes", [(f"Is {n} a prime number? Answer yes or no.", ["yes" if n in primes else "no"]) for n in nums])
    add("gcd", [(f"What is the greatest common divisor of {a} and {b}?", [str(gcd(a, b))])
                for a, b in ((rng.randint(2, 40) * k, rng.randint(2, 40) * k)
                             for k in (rng.randint(2, 12) for _ in range(20)))])
    add("base_conversion", [(f"Write the decimal number {n} in binary.", [bin(n)[2:]])
                            for n in (rng.randint(5, 255) for _ in range(20))])
    units = [("kilometers", "meters", 1000), ("meters", "centimeters", 100), ("hours", "minutes", 60),
             ("minutes", "seconds", 60), ("kilograms", "grams", 1000), ("days", "hours", 24),
             ("liters", "milliliters", 1000), ("feet", "inches", 12), ("yards", "feet", 3),
             ("weeks", "days", 7)]
    add("unit_conversion", [(f"How many {b} are in {n} {a}?", [str(n * f)])
                            for (a, b, f), n in ((rng.choice(units), rng.randint(2, 50)) for _ in range(20))])
    days = []
    for _ in range(20):
        d = datetime.date(rng.randint(1950, 2030), rng.randint(1, 12), rng.randint(1, 28))
        days.append((f"What day of the week was {d.strftime('%B')} {d.day}, {d.year}?", [d.strftime("%A")]))
    add("day_of_week", days)
    fractions = []
    for _ in range(20):
        den = rng.choice([2, 4, 5, 8, 10, 16, 20, 25])
        num = rng.randint(1, den * 3)
        value = Fraction(num, den)
        text = f"{float(value):.4f}".rstrip("0").rstrip(".")
        fractions.append((f"Write {num}/{den} as a decimal number.", [text]))
    add("fraction_to_decimal", fractions)
    return domains


FACTS = {
    "country_capitals": ("What is the capital of {}?", [
        ("France", "Paris"), ("Japan", "Tokyo"), ("Canada", "Ottawa"), ("Australia", "Canberra"),
        ("Brazil", "Brasilia|Brasília"), ("Egypt", "Cairo"), ("Kenya", "Nairobi"), ("Norway", "Oslo"),
        ("Peru", "Lima"), ("Turkey", "Ankara"), ("Vietnam", "Hanoi"), ("Argentina", "Buenos Aires"),
        ("Poland", "Warsaw"), ("Portugal", "Lisbon"), ("Thailand", "Bangkok"), ("Morocco", "Rabat"),
        ("New Zealand", "Wellington"), ("Chile", "Santiago"), ("Finland", "Helsinki"), ("Ireland", "Dublin")]),
    "element_symbols": ("What is the chemical symbol for {}?", [
        ("gold", "Au"), ("silver", "Ag"), ("iron", "Fe"), ("sodium", "Na"), ("potassium", "K"),
        ("lead", "Pb"), ("tin", "Sn"), ("mercury", "Hg"), ("copper", "Cu"), ("tungsten", "W"),
        ("helium", "He"), ("neon", "Ne"), ("carbon", "C"), ("oxygen", "O"), ("calcium", "Ca"),
        ("zinc", "Zn"), ("chlorine", "Cl"), ("magnesium", "Mg"), ("silicon", "Si"), ("nitrogen", "N")]),
    "atomic_numbers": ("What is the atomic number of {}?", [
        ("hydrogen", "1"), ("helium", "2"), ("lithium", "3"), ("carbon", "6"), ("nitrogen", "7"),
        ("oxygen", "8"), ("fluorine", "9"), ("neon", "10"), ("sodium", "11"), ("magnesium", "12"),
        ("aluminium", "13"), ("silicon", "14"), ("phosphorus", "15"), ("sulfur", "16"), ("chlorine", "17"),
        ("argon", "18"), ("potassium", "19"), ("calcium", "20"), ("iron", "26"), ("gold", "79")]),
    "us_state_capitals": ("What is the capital of the US state of {}?", [
        ("California", "Sacramento"), ("Texas", "Austin"), ("New York", "Albany"), ("Florida", "Tallahassee"),
        ("Illinois", "Springfield"), ("Washington", "Olympia"), ("Oregon", "Salem"), ("Nevada", "Carson City"),
        ("Arizona", "Phoenix"), ("Colorado", "Denver"), ("Georgia", "Atlanta"), ("Ohio", "Columbus"),
        ("Michigan", "Lansing"), ("Pennsylvania", "Harrisburg"), ("Massachusetts", "Boston"),
        ("Vermont", "Montpelier"), ("Alaska", "Juneau"), ("Hawaii", "Honolulu"), ("Utah", "Salt Lake City"),
        ("Kentucky", "Frankfort")]),
    "book_authors": ("Who wrote the novel '{}'?", [
        ("Pride and Prejudice", "Austen"), ("1984", "Orwell"), ("Moby-Dick", "Melville"),
        ("War and Peace", "Tolstoy"), ("The Great Gatsby", "Fitzgerald"), ("Don Quixote", "Cervantes"),
        ("Crime and Punishment", "Dostoevsky|Dostoyevsky"), ("Frankenstein", "Shelley"),
        ("Jane Eyre", "Brontë|Bronte"), ("The Hobbit", "Tolkien"), ("Ulysses", "Joyce"),
        ("To Kill a Mockingbird", "Lee"), ("One Hundred Years of Solitude", "García Márquez|Garcia Marquez|Márquez"),
        ("The Catcher in the Rye", "Salinger"), ("Brave New World", "Huxley"), ("Dracula", "Stoker"),
        ("Great Expectations", "Dickens"), ("Les Misérables", "Hugo"), ("The Odyssey", "Homer"),
        ("Madame Bovary", "Flaubert")]),
    "painters": ("Who painted '{}'?", [
        ("The Starry Night", "van Gogh|Van Gogh"), ("Mona Lisa", "Leonardo|da Vinci"), ("The Scream", "Munch"),
        ("Guernica", "Picasso"), ("The Persistence of Memory", "Dalí|Dali"), ("Girl with a Pearl Earring", "Vermeer"),
        ("The Night Watch", "Rembrandt"), ("The Birth of Venus", "Botticelli"), ("Water Lilies", "Monet"),
        ("The Last Supper", "Leonardo|da Vinci"), ("American Gothic", "Wood"), ("The Kiss (1908)", "Klimt"),
        ("Las Meninas", "Velázquez|Velazquez"), ("The Garden of Earthly Delights", "Bosch"),
        ("Liberty Leading the People", "Delacroix"), ("Nighthawks", "Hopper"), ("The Great Wave off Kanagawa", "Hokusai"),
        ("Impression, Sunrise", "Monet"), ("The School of Athens", "Raphael"), ("Sunflowers", "van Gogh|Van Gogh")]),
    "composers": ("Who composed '{}'?", [
        ("The Four Seasons", "Vivaldi"), ("The Magic Flute", "Mozart"), ("Symphony No. 9 'Choral'", "Beethoven"),
        ("The Nutcracker", "Tchaikovsky"), ("The Rite of Spring", "Stravinsky"), ("Boléro", "Ravel"),
        ("Clair de lune", "Debussy"), ("The Messiah (oratorio)", "Handel|Händel"), ("Brandenburg Concertos", "Bach"),
        ("The Ring of the Nibelung", "Wagner"), ("La traviata", "Verdi"), ("Carmen (opera)", "Bizet"),
        ("Rhapsody in Blue", "Gershwin"), ("The Planets (suite)", "Holst"), ("New World Symphony", "Dvořák|Dvorak"),
        ("Swan Lake", "Tchaikovsky"), ("Madama Butterfly", "Puccini"), ("Peer Gynt Suite", "Grieg"),
        ("The Barber of Seville (opera)", "Rossini"), ("Gymnopédies", "Satie")]),
    "currencies": ("What is the official currency of {}?", [
        ("Japan", "yen"), ("the United Kingdom", "pound"), ("India", "rupee"), ("Switzerland", "franc"),
        ("Mexico", "peso"), ("South Korea", "won"), ("China", "yuan|renminbi"), ("Russia", "ruble|rouble"),
        ("Brazil", "real"), ("South Africa", "rand"), ("Sweden", "krona"), ("Poland", "złoty|zloty"),
        ("Turkey", "lira"), ("Israel", "shekel"), ("Thailand", "baht"), ("Vietnam", "dong|đồng"),
        ("Germany", "euro"), ("Canada", "dollar"), ("Saudi Arabia", "riyal"), ("Hungary", "forint")]),
    "astronomy": ("{}", [
        ("Which planet is closest to the Sun?", "Mercury"), ("Which planet is the largest in the Solar System?", "Jupiter"),
        ("Which planet is known as the Red Planet?", "Mars"), ("Which planet has the most prominent rings?", "Saturn"),
        ("What is the name of Earth's natural satellite?", "Moon"), ("Which planet is third from the Sun?", "Earth"),
        ("Which is the hottest planet in the Solar System?", "Venus"), ("Which planet rotates on its side?", "Uranus"),
        ("What is the largest moon of Saturn?", "Titan"), ("What is the largest moon of Jupiter?", "Ganymede"),
        ("Which dwarf planet was reclassified in 2006?", "Pluto"), ("What star is at the centre of the Solar System?", "Sun"),
        ("What galaxy contains the Solar System?", "Milky Way"), ("What is the nearest large galaxy to the Milky Way?", "Andromeda"),
        ("Which planet is farthest from the Sun?", "Neptune"), ("How many planets are in the Solar System?", "8|eight"),
        ("What is the brightest star in the night sky?", "Sirius"), ("Which planet has the Great Red Spot?", "Jupiter"),
        ("What is Mars's larger moon called?", "Phobos"), ("Which planet is second from the Sun?", "Venus")]),
    "inventors": ("Who is credited with inventing {}?", [
        ("the telephone", "Bell"), ("the light bulb (practical incandescent)", "Edison"), ("the World Wide Web", "Berners-Lee"),
        ("the printing press with movable type in Europe", "Gutenberg"), ("dynamite", "Nobel"),
        ("the airplane (first powered flight)", "Wright"), ("the radio (wireless telegraphy)", "Marconi"),
        ("the steam engine improvements with a separate condenser", "Watt"), ("the polio vaccine (injected)", "Salk"),
        ("penicillin (discovery)", "Fleming"), ("the theory of general relativity", "Einstein"),
        ("the periodic table", "Mendeleev|Mendeleyev"), ("the phonograph", "Edison"), ("alternating current induction motor", "Tesla"),
        ("the cotton gin", "Whitney"), ("the telegraph code with dots and dashes", "Morse"),
        ("the analytical engine", "Babbage"), ("the laws of motion and universal gravitation", "Newton"),
        ("the theory of evolution by natural selection", "Darwin"), ("the dynamo-based electromagnetic induction discovery", "Faraday")]),
    "history_years": ("In what year {}?", [
        ("did World War II end", "1945"), ("did humans first land on the Moon", "1969"), ("did the Berlin Wall fall", "1989"),
        ("did World War I begin", "1914"), ("was the US Declaration of Independence signed", "1776"),
        ("did the French Revolution begin", "1789"), ("did Columbus first reach the Americas", "1492"),
        ("did the Titanic sink", "1912"), ("was the Magna Carta sealed", "1215"), ("did the Soviet Union dissolve", "1991"),
        ("did the Battle of Hastings take place", "1066"), ("did the American Civil War end", "1865"),
        ("did the Wall Street Crash happen", "1929"), ("was Nelson Mandela released from prison", "1990"),
        ("did the Chernobyl disaster happen", "1986"), ("was the euro introduced as cash", "2002"),
        ("did India gain independence", "1947"), ("did the September 11 attacks occur", "2001"),
        ("was the United Nations founded", "1945"), ("did Napoleon lose the Battle of Waterloo", "1815")]),
    "programming_languages": ("Who created the programming language {}?", [
        ("Python", "van Rossum|Rossum"), ("C", "Ritchie"), ("C++", "Stroustrup"), ("Java", "Gosling"),
        ("Linux kernel (not a language, but who created it)", "Torvalds"), ("Ruby", "Matsumoto|Matz"),
        ("Perl", "Wall"), ("JavaScript", "Eich"), ("Lisp", "McCarthy"), ("Haskell (main designers' committee member Simon)", "Peyton Jones|Peyton-Jones"),
        ("Pascal", "Wirth"), ("Swift", "Lattner"), ("Go (one of the three creators besides Pike and Thompson)", "Griesemer"),
        ("Rust (original creator)", "Hoare"), ("PHP", "Lerdorf"), ("Erlang", "Armstrong"), ("Scala", "Odersky"),
        ("Smalltalk (lead designer)", "Kay"), ("COBOL (key influence, FLOW-MATIC author)", "Hopper"), ("Fortran (team lead)", "Backus")]),
}


def questions(seed):
    rng = random.Random(seed)
    out = []
    for domain, items in computed(rng).items():
        for question, answers in items:
            out.append({"domain": domain, "question": question, "answers": answers, "kind": "computed"})
    for domain, (template, rows) in FACTS.items():
        assert len(rows) == 20, domain
        for subject, answer in rows:
            out.append({"domain": domain, "question": template.format(subject), "answers": answer.split("|"),
                        "kind": "fact"})
    assert len(out) == 400 and len({q["domain"] for q in out}) == 20
    return out


def grade(row):
    text = row.get("response") or ""
    norm = re.sub(r"\s+", " ", text.replace("*", "")).strip().lower()
    if row["kind"] == "computed":
        if row["domain"] == "primes":
            first = re.findall(r"\b(yes|no)\b", norm)
            return bool(first) and first[0] == row["answers"][0]
        if row["domain"] == "day_of_week":
            return row["answers"][0].lower() in norm
        compact = norm.replace(",", "").replace(" ", "")
        return any(re.search(rf"(?<![0-9.]){re.escape(a.lower())}(?![0-9])", compact) for a in row["answers"])
    return any(a.lower() in norm for a in row["answers"])


# ------------------------------------------------------------------ metrics


def scrape(url):
    with urllib.request.urlopen(url + "/metrics", timeout=5) as response:
        text = response.read().decode()
    values = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        name_labels, _, value = line.rpartition(" ")
        try:
            values[name_labels] = float(value)
        except ValueError:
            pass
    return values


def series(values, name):
    return {k: v for k, v in values.items() if k == name or k.startswith(name + "{")}


def total(values, name):
    return sum(series(values, name).values())


def histogram(before, after, name):
    count = total(after, name + "_count") - total(before, name + "_count")
    summ = total(after, name + "_sum") - total(before, name + "_sum")
    buckets = {}
    for key, value in series(after, name + "_bucket").items():
        match = re.search(r'le="([^"]+)"', key)
        if match:
            le = float("inf") if match.group(1) == "+Inf" else float(match.group(1))
            buckets[le] = buckets.get(le, 0) + value - before.get(key, 0)
    def quantile(q):
        target = q * count
        for le in sorted(buckets):
            if buckets[le] >= target:
                return le
        return None
    return {"count": count, "mean": summ / count if count else None, "p50_le": quantile(0.5), "p95_le": quantile(0.95)}


# ---------------------------------------------------------------------- run


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", type=Path)
    ap.add_argument("--url", default="http://127.0.0.1:8282")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--scrape", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=20261105)
    args = ap.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit("refusing nonempty evidence directory")
    args.out.mkdir(parents=True, exist_ok=True)
    rows = questions(args.seed)
    model = json.loads(urllib.request.urlopen(args.url + "/v1/models", timeout=10).read())["data"][0]["id"]
    stop = threading.Event()
    samples = []

    def scraper():
        with (args.out / "metrics.jsonl").open("w") as handle:
            while not stop.is_set():
                try:
                    values = scrape(args.url)
                    sample = {"t": time.time(), **{k: total(values, k) for k in (
                        "mlx2_generation_tokens_total", "mlx2_prompt_tokens_total", "mlx2_num_requests_running",
                        "mlx2_num_requests_waiting", "mlx2_active_lanes", "mlx2_memory_active_bytes",
                        "mlx2_memory_peak_bytes", "mlx2_memory_headroom_bytes", "mlx2_cached_prompt_tokens_total",
                        "mlx2_prefix_cache_hits_total")}}
                    samples.append(sample)
                    handle.write(json.dumps(sample) + "\n")
                    handle.flush()
                except Exception as error:  # a scrape failure is itself a finding
                    samples.append({"t": time.time(), "scrape_error": str(error)})
                stop.wait(args.scrape)

    before = scrape(args.url)
    thread = threading.Thread(target=scraper, daemon=True)
    thread.start()

    def ask(row):
        body = {"model": model, "temperature": 0, "max_tokens": args.max_tokens,
                "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": row["question"]}]}
        started = time.time()
        try:
            request = urllib.request.Request(args.url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                             headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(request, timeout=600) as response:
                reply = json.loads(response.read())
            row["response"] = reply["choices"][0]["message"].get("content") or ""
            row["finish_reason"] = reply["choices"][0].get("finish_reason")
            receipt = reply.get("mlx2") or {}
            row["route"], row["qualification"] = receipt.get("route"), receipt.get("qualification")
            row["status"] = 200
        except urllib.error.HTTPError as error:
            row["status"], row["error"] = error.code, error.read().decode()[:300]
        except Exception as error:
            row["status"], row["error"] = None, f"{type(error).__name__}: {error}"
        row["seconds"] = time.time() - started
        row["correct"] = row.get("status") == 200 and grade(row)
        return row

    t0 = time.time()
    with ThreadPoolExecutor(args.concurrency) as pool:
        results = list(pool.map(ask, rows))
    wall = time.time() - t0
    time.sleep(min(args.scrape, 5))
    stop.set()
    thread.join(timeout=15)
    after = scrape(args.url)

    by_domain = {}
    for row in results:
        entry = by_domain.setdefault(row["domain"], {"correct": 0, "total": 0, "errors": 0})
        entry["total"] += 1
        entry["correct"] += bool(row["correct"])
        entry["errors"] += row.get("status") != 200
    http = {}
    for key in series(after, "mlx2_http_requests_total"):
        delta = after[key] - before.get(key, 0)
        if delta:
            http[key] = delta
    running = [s.get("mlx2_num_requests_running") for s in samples if "mlx2_num_requests_running" in s]
    waiting = [s.get("mlx2_num_requests_waiting") for s in samples if "mlx2_num_requests_waiting" in s]
    lanes = [s.get("mlx2_active_lanes") for s in samples if "mlx2_active_lanes" in s]
    rates = []
    for a, b in zip(samples, samples[1:]):
        if "mlx2_generation_tokens_total" in a and "mlx2_generation_tokens_total" in b and b["t"] > a["t"]:
            rates.append((b["mlx2_generation_tokens_total"] - a["mlx2_generation_tokens_total"]) / (b["t"] - a["t"]))
    report = {
        "model": model, "url": args.url, "concurrency": args.concurrency, "max_tokens": args.max_tokens,
        "seed": args.seed, "wall_seconds": wall, "questions": len(results),
        "accuracy": sum(bool(r["correct"]) for r in results) / len(results),
        "errors": sum(r.get("status") != 200 for r in results),
        "by_domain": by_domain,
        "routes": sorted({str(r.get("route")) for r in results}),
        "qualification": sorted({str(r.get("qualification")) for r in results}),
        "finish_reasons": {k: sum(r.get("finish_reason") == k for r in results)
                           for k in {r.get("finish_reason") for r in results}},
        "metrics": {
            "generation_tokens": total(after, "mlx2_generation_tokens_total") - total(before, "mlx2_generation_tokens_total"),
            "prompt_tokens": total(after, "mlx2_prompt_tokens_total") - total(before, "mlx2_prompt_tokens_total"),
            "cached_prompt_tokens": total(after, "mlx2_cached_prompt_tokens_total") - total(before, "mlx2_cached_prompt_tokens_total"),
            "prefix_cache_hits": total(after, "mlx2_prefix_cache_hits_total") - total(before, "mlx2_prefix_cache_hits_total"),
            "generation_tok_per_s_wall": (total(after, "mlx2_generation_tokens_total")
                                          - total(before, "mlx2_generation_tokens_total")) / wall,
            "generation_tok_per_s_scrape_max": max(rates) if rates else None,
            "running_max": max(running) if running else None, "waiting_max": max(waiting) if waiting else None,
            "active_lanes_max": max(lanes) if lanes else None,
            "active_lanes_mean": sum(lanes) / len(lanes) if lanes else None,
            "memory_peak_gb": total(after, "mlx2_memory_peak_bytes") / 1e9,
            "memory_headroom_gb_min": min((s["mlx2_memory_headroom_bytes"] for s in samples
                                           if "mlx2_memory_headroom_bytes" in s), default=0) / 1e9,
            "ttft_s": histogram(before, after, "mlx2_time_to_first_token_seconds"),
            "inter_token_s": histogram(before, after, "mlx2_inter_token_latency_seconds"),
            "e2e_s": histogram(before, after, "mlx2_e2e_request_latency_seconds"),
            "http_status_deltas": http,
            "scrape_errors": sum("scrape_error" in s for s in samples),
            "scrapes": len(samples),
        },
    }
    (args.out / "responses.jsonl").write_text("".join(json.dumps(r) + "\n" for r in results))
    (args.out / "report.json").write_text(json.dumps(report, indent=1, default=str))
    print(json.dumps({k: report[k] for k in ("accuracy", "errors", "wall_seconds")}, default=str))


if __name__ == "__main__":
    main()
