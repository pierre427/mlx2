"""Task sets for core and box carving (capitals, arithmetic, French, recall + completion splits).

Ported unchanged from scripts/amoeba27b_lib.py at claude/amoeba-27b-80bc6a@37d75f144 (lines 456-627),
so every donor is scored on the same items, phrasings and fit/held splits as the 27B runs.
Tokenizer-generic: items are built with the donor's own tokenizer through host.encode.
"""
from __future__ import annotations

import random

COUNTRIES = [
    ("France", "Paris"), ("Germany", "Berlin"), ("Italy", "Rome"), ("Spain", "Madrid"), ("Japan", "Tokyo"),
    ("China", "Beijing"), ("Russia", "Moscow"), ("Egypt", "Cairo"), ("Canada", "Ottawa"), ("Australia", "Canberra"),
    ("Brazil", "Brasília"), ("Argentina", "Buenos Aires"), ("Mexico", "Mexico City"), ("India", "New Delhi"),
    ("Kenya", "Nairobi"), ("Greece", "Athens"), ("Portugal", "Lisbon"), ("Poland", "Warsaw"), ("Austria", "Vienna"),
    ("Hungary", "Budapest"), ("Sweden", "Stockholm"), ("Norway", "Oslo"), ("Finland", "Helsinki"),
    ("Denmark", "Copenhagen"), ("Ireland", "Dublin"), ("Belgium", "Brussels"), ("Netherlands", "Amsterdam"),
    ("Turkey", "Ankara"), ("Iran", "Tehran"), ("Iraq", "Baghdad"), ("Thailand", "Bangkok"), ("Vietnam", "Hanoi"),
    ("Indonesia", "Jakarta"), ("Peru", "Lima"), ("Chile", "Santiago"), ("Colombia", "Bogotá"), ("Cuba", "Havana"),
    ("Ukraine", "Kyiv"), ("Romania", "Bucharest"), ("Bulgaria", "Sofia"), ("Serbia", "Belgrade"),
    ("Croatia", "Zagreb"), ("Czechia", "Prague"), ("Switzerland", "Bern"), ("Morocco", "Rabat"),
    ("Ethiopia", "Addis Ababa"), ("Nigeria", "Abuja"), ("Ghana", "Accra"), ("Pakistan", "Islamabad"),
    ("Afghanistan", "Kabul"), ("Nepal", "Kathmandu"), ("Mongolia", "Ulaanbaatar"), ("Philippines", "Manila"),
    ("Malaysia", "Kuala Lumpur"), ("Venezuela", "Caracas"), ("Uruguay", "Montevideo"), ("Israel", "Jerusalem"),
    ("Jordan", "Amman"), ("Syria", "Damascus"), ("Lebanon", "Beirut"), ("Iceland", "Reykjavik"),
    ("Estonia", "Tallinn"), ("Latvia", "Riga"), ("Lithuania", "Vilnius"), ("Slovakia", "Bratislava"),
    ("Slovenia", "Ljubljana"), ("Albania", "Tirana"), ("Belarus", "Minsk"), ("Moldova", "Chișinău"),
    ("Georgia", "Tbilisi"), ("Armenia", "Yerevan"), ("Azerbaijan", "Baku"), ("Kazakhstan", "Astana"),
    ("Uzbekistan", "Tashkent"), ("Kyrgyzstan", "Bishkek"), ("Tajikistan", "Dushanbe"),
    ("Turkmenistan", "Ashgabat"), ("Saudi Arabia", "Riyadh"), ("Yemen", "Sanaa"), ("Oman", "Muscat"),
    ("Qatar", "Doha"), ("Kuwait", "Kuwait City"), ("Bahrain", "Manama"), ("Bangladesh", "Dhaka"),
    ("Sri Lanka", "Colombo"), ("Myanmar", "Naypyidaw"), ("Cambodia", "Phnom Penh"), ("Laos", "Vientiane"),
    ("South Korea", "Seoul"), ("North Korea", "Pyongyang"), ("New Zealand", "Wellington"), ("Fiji", "Suva"),
    ("Algeria", "Algiers"), ("Tunisia", "Tunis"), ("Libya", "Tripoli"), ("Sudan", "Khartoum"),
    ("Senegal", "Dakar"), ("Mali", "Bamako"), ("Niger", "Niamey"), ("Chad", "N'Djamena"),
    ("Cameroon", "Yaoundé"), ("Uganda", "Kampala"), ("Tanzania", "Dodoma"), ("Rwanda", "Kigali"),
    ("Zambia", "Lusaka"), ("Zimbabwe", "Harare"), ("Angola", "Luanda"), ("Mozambique", "Maputo"),
    ("Namibia", "Windhoek"), ("Botswana", "Gaborone"), ("Madagascar", "Antananarivo"), ("Somalia", "Mogadishu"),
    ("Ecuador", "Quito"), ("Bolivia", "Sucre"), ("Paraguay", "Asunción"), ("Guatemala", "Guatemala City"),
    ("Honduras", "Tegucigalpa"), ("Nicaragua", "Managua"), ("Costa Rica", "San José"), ("Panama", "Panama City"),
    ("Jamaica", "Kingston"), ("Haiti", "Port-au-Prince"), ("Luxembourg", "Luxembourg"), ("Malta", "Valletta"),
    ("Cyprus", "Nicosia"), ("Montenegro", "Podgorica"), ("Bosnia and Herzegovina", "Sarajevo"),
    ("North Macedonia", "Skopje"), ("Bhutan", "Thimphu"), ("Maldives", "Malé"), ("Brunei", "Bandar Seri Begawan"),
    ("Papua New Guinea", "Port Moresby"), ("Eritrea", "Asmara"), ("Gabon", "Libreville"), ("Benin", "Porto-Novo"),
    ("Togo", "Lomé"), ("Liberia", "Monrovia"), ("Sierra Leone", "Freetown"), ("Guinea", "Conakry"),
    ("Burkina Faso", "Ouagadougou"), ("Malawi", "Lilongwe"), ("Lesotho", "Maseru"), ("Mauritania", "Nouakchott"),
]

FRENCH = [("dog", "chien"), ("cat", "chat"), ("house", "maison"), ("water", "eau"), ("book", "livre"),
          ("bread", "pain"), ("cheese", "fromage"), ("apple", "pomme"), ("tree", "arbre"), ("sun", "soleil"),
          ("moon", "lune"), ("star", "étoile"), ("sea", "mer"), ("river", "rivière"), ("mountain", "montagne"),
          ("city", "ville"), ("street", "rue"), ("car", "voiture"), ("boat", "bateau"),
          ("bird", "oiseau"), ("fish", "poisson"), ("horse", "cheval"), ("cow", "vache"), ("mouse", "souris"),
          ("flower", "fleur"), ("garden", "jardin"), ("door", "porte"), ("window", "fenêtre"),
          ("chair", "chaise"), ("bed", "lit"), ("kitchen", "cuisine"), ("friend", "ami"), ("mother", "mère"),
          ("father", "père"), ("brother", "frère"), ("sister", "sœur"), ("child", "enfant"), ("man", "homme"),
          ("woman", "femme"), ("king", "roi"), ("queen", "reine"), ("school", "école"), ("teacher", "professeur"),
          ("money", "argent"), ("day", "jour"), ("night", "nuit"), ("week", "semaine"),
          ("year", "année"), ("morning", "matin"), ("evening", "soir"), ("winter", "hiver"), ("summer", "été"),
          ("spring", "printemps"), ("snow", "neige"), ("rain", "pluie"), ("wind", "vent"), ("fire", "feu"),
          ("milk", "lait"), ("egg", "œuf"), ("meat", "viande"), ("wine", "vin"), ("beer", "bière"),
          ("red", "rouge"), ("blue", "bleu"), ("green", "vert"), ("black", "noir"), ("white", "blanc"),
          ("yellow", "jaune"), ("big", "grand"), ("small", "petit"), ("old", "vieux"),
          ("good", "bon"), ("bad", "mauvais"), ("happy", "heureux"), ("sad", "triste"), ("hot", "chaud"),
          ("cold", "froid"), ("hand", "main"), ("head", "tête"), ("eye", "œil"), ("heart", "cœur"),
          ("road", "route"), ("bridge", "pont"), ("church", "église"), ("market", "marché"), ("shop", "magasin"),
          ("letter", "lettre"), ("paper", "papier"), ("pen", "stylo"), ("song", "chanson"), ("game", "jeu"),
          ("love", "amour"), ("war", "guerre"), ("peace", "paix"), ("country", "pays"), ("island", "île"),
          ("knife", "couteau"), ("spoon", "cuillère"), ("forest", "forêt"), ("cloud", "nuage"), ("sky", "ciel"),
          ("butter", "beurre"), ("sugar", "sucre"), ("salt", "sel"), ("bottle", "bouteille"), ("glass", "verre")]
FR_SHOTS = "English: shoe\nFrench: chaussure\nEnglish: hat\nFrench: chapeau\n"
CAP_TEMPLATES = ["The capital of {} is", "{}'s capital city is", "The seat of government of {} is the city of"]
CAP_VAL_TEMPLATE = "In {}, the capital city is"
RECALL_FILLERS = ["What color is the sky? A: Blue.", "How many legs does a cat have? A: Four.",
                  "What do bees make? A: Honey.", "Which season is coldest? A: Winter."]


def tf_items(host, rows):
    """Teacher-forced items: ids = prompt + answer; pos = positions predicting each answer token."""
    items = []
    for p, a, label in rows:
        pi, ai = host.encode(p), host.encode(a)
        items.append({"ids": pi + ai, "pos": list(range(len(pi) - 1, len(pi) + len(ai) - 1)),
                      "tgt": ai, "label": label, "prompt": p, "answer": a})
    return items


def build_tasks(host, seed=0):
    """Capitals (fit / held / fresh countries, templates 0-1 for fit+held, template 2 = new phrasing),
    3-digit sums, English->French words, and multi-turn number recall. Returns tasks, splits."""
    rng = random.Random(seed)
    tasks, splits = {}, {}
    countries = list(COUNTRIES)
    random.Random(seed + 7).shuffle(countries)
    fresh = countries[:40]                          # never used to fit or choose anything
    pool = countries[40:]
    caps_rows = [(CAP_TEMPLATES[i % 2].format(c), " " + k, c) for i, (c, k) in enumerate(pool)]
    tasks["capitals"] = tf_items(host, caps_rows)
    tasks["capitals_fresh"] = tf_items(host, [(CAP_TEMPLATES[0].format(c), " " + k, c) for c, k in fresh])
    tasks["capitals_phrasing"] = tf_items(host, [(CAP_TEMPLATES[2].format(c), " " + k, c) for c, k in pool])
    pairs = set()
    while len(pairs) < 240:
        pairs.add((rng.randint(100, 999), rng.randint(100, 999)))
    tasks["arithmetic"] = tf_items(host, [(f"{a} + {b} =", f" {a + b}", f"{a}+{b}") for a, b in sorted(pairs)])
    fr = list(FRENCH)
    tasks["french"] = tf_items(host, [(FR_SHOTS + f"English: {e}\nFrench:", f" {f}", e) for e, f in fr])
    # completion splits (2026-11-14, Pierre: a box is complete when core + box matches the full model):
    # the same facts in two further phrasings; one chooses, the other reports
    tasks["capitals_val"] = tf_items(host, [(CAP_VAL_TEMPLATE.format(c), " " + k, c) for c, k in pool])
    tasks["french_val"] = tf_items(host, [(f"English: {e}\nFrench:", f" {f}", e) for e, f in fr])
    tasks["french_phrasing"] = tf_items(host, [(f"The French word for \"{e}\" is", f" \"{f}", e) for e, f in fr])
    nums = rng.sample(range(10, 99), 80)
    tasks["recall"] = tf_items(host, [(f"Remember the number {n}. Q: {RECALL_FILLERS[i % 4]} Q: What number did I ask "
                                       f"you to remember? A: The number is", f" {n}", str(n)) for i, n in enumerate(nums)])
    for name, items in tasks.items():
        idx = list(range(len(items)))
        if name in ("capitals_fresh", "capitals_phrasing", "capitals_val", "french_val", "french_phrasing"):
            splits[name] = {"fit": [], "held": idx}
            continue
        random.Random(seed + len(name)).shuffle(idx)
        h = len(idx) // 2
        splits[name] = {"fit": sorted(idx[:h]), "held": sorted(idx[h:])}
    # capitals_phrasing reports only on held-out countries of the capitals split
    held_c = {tasks["capitals"][i]["label"] for i in splits["capitals"]["held"]}
    splits["capitals_phrasing"]["held"] = [i for i, it in enumerate(tasks["capitals_phrasing"]) if it["label"] in held_c]
    return tasks, splits


def groups_of(items, idx, cap=16):
    by = {}
    for i in idx:
        by.setdefault(len(items[i]["tgt"]), []).append(i)
    out = []
    for g in by.values():
        out += [g[j:j + cap] for j in range(0, len(g), cap)]
    return out


def batch_of(items, grp, pad=0):
    import mlx.core as mx
    T = max(len(items[i]["ids"]) for i in grp)
    ids = mx.array([items[i]["ids"] + [pad] * (T - len(items[i]["ids"])) for i in grp])
    pos = mx.array([items[i]["pos"] for i in grp])
    return ids, pos


def task_eval(host, items, idx, cfg_fn=None, clean=None, key=None):
    """Teacher-forced exact match of the whole answer, and mean KL vs clean (if clean cache given)."""
    import mlx.core as mx

    from .carving import kl_rows
    correct, kl_tot, n_pos = 0, 0.0, 0
    for grp in groups_of(items, idx):
        ids, pos = batch_of(items, grp)
        if cfg_fn:
            cfg_fn()
        lg = host.logits_at(ids, pos)
        host.ctl.reset()
        tgt = mx.array([items[i]["tgt"] for i in grp])
        ok = (lg.argmax(-1) == tgt).all(-1).sum()
        if clean is not None:
            kl = kl_rows(clean[(key, tuple(grp))], lg).sum()
            mx.eval(ok, kl)
            kl_tot += float(kl)
        else:
            mx.eval(ok)
        correct += int(ok)
        n_pos += int(pos.size)
        mx.clear_cache()
    return correct / max(1, len(idx)), (kl_tot / max(1, n_pos) if clean is not None else None)


def cache_clean(host, items, idx, cache, key, cap=16):
    import mlx.core as mx
    for grp in groups_of(items, idx, cap=cap):
        k = (key, tuple(grp))
        if k in cache:
            continue
        ids, pos = batch_of(items, grp)
        host.ctl.reset()
        lg = host.logits_at(ids, pos)
        mx.eval(lg)
        cache[k] = lg
