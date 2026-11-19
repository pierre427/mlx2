#!/usr/bin/env python3
"""General-prose language corpus for an aMoEba firing map (Pierre 2026-11-16: "just normal regular chinese text, NOT
expert-related, so not AI related, not medical related. Just prose. Anything from 2k to 10k documents ... statistically
significant neural firing maps, not one offs"; "exclude too-similar corpus material"). CPU only.

Sources (repeatable --source KIND:NAME:PATH): KIND wiki (wikimedia/wikipedia parquet: title, text) or web (FineWeb /
FineWeb-2 / FineWeb-edu parquet: text, url). Per document:
  1. clean: wiki text is cut at its reference / link sections; whitespace normalized;
  2. quality: --min-chars .. (cut to --max-chars at a sentence end), share of letters in the target script
     (Han for cmn, Latin for eng), sentence punctuation density, repeated-line share, digit/URL share, and a spam
     word list (gambling, adult, SEO);
  3. topic exclusion: AI / computing and medical / health words (per language list); a document with any title hit
     or >= --topic-hits body hits is dropped;
  4. de-duplication: exact (normalized text hash), then near-duplicates by MinHash over character 5-grams (Han) or word
     3-grams (Latin) with LSH (--bands x --rows); a pair at estimated Jaccard >= --near keeps the earlier document only.
     This also removes templated pages (e.g. "X is a town in Bulgaria" stubs) that fire the same units again and again;
  5. sampling: --per-source documents per source (seeded), then a document-level split fit / val / held
     (--val / --held / --held2 shares, stratified by source x script).
Writes OUT/corpus.jsonl (text, id, doc_id, topics [lang.<code>], split, origin natural, source, source_url,
source_revision, license, redistribute, script, chars) and OUT/manifest.json (filters, drop counts per reason,
sha256, counts by split / source / script). Needs pyarrow (e.g. the mlx-uag venv) and numpy.

Usage: amoeba_lang_prose_build.py --lang cmn --out DIR --source wiki:wikipedia-zh:FILE --source web:fineweb2:FILE
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import unicodedata
import zlib
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

SPAM = {
    "cmn": ["博彩", "娱乐城", "娛樂城", "彩票", "六合彩", "赌场", "賭場", "赌博", "賭博", "投注", "百家乐", "百家樂", "老虎机",
            "威尼斯人", "葡京", "开户", "開戶", "时时彩", "時時彩", "pk10", "北京赛车", "体育在线", "色情", "成人视频",
            "成人影片", "约炮", "裸聊", "av女优", "代孕", "代开发票", "发票", "贷款", "網賺", "网赚", "传奇私服", "私服",
            "棋牌", "真人视讯", "博彩网", "足球博彩", "下注", "外围", "微信号", "加微信", "QQ群", "点击进入", "免费下载"],
    "eng": ["casino", "betting", "viagra", "cialis", "porn", "xxx", "payday loan", "escort", "click here",
            "free download", "bitcoin casino", "slot machine", "jackpot"],
}
TOPIC = {
    "cmn": ["人工智能", "机器学习", "機器學習", "深度学习", "深度學習", "神经网络", "神經網路", "神經網絡", "算法", "演算法",
            "大模型", "语言模型", "語言模型", "ChatGPT", "GPT", "AI", "编程", "程式設計", "程序员", "程式設計師", "软件",
            "軟體", "軟件", "计算机", "計算機", "電腦", "电脑", "数据库", "資料庫", "服务器", "伺服器", "代码", "程式碼",
            "芯片", "晶片", "处理器", "處理器", "操作系统", "作業系統", "互联网", "網際網路", "互聯網",
            "医院", "醫院", "医生", "醫生", "疾病", "病人", "患者", "治疗", "治療", "症状", "症狀", "药物", "藥物", "医学",
            "醫學", "癌", "手术", "手術", "临床", "臨床", "疫苗", "病毒", "细菌", "細菌", "感染", "诊断", "診斷", "药品",
            "藥品", "保健", "养生", "養生", "减肥", "減肥", "中医", "中醫", "护理", "護理", "健康", "血压", "血壓", "糖尿病"],
    "eng": ["artificial intelligence", "machine learning", "deep learning", "neural network", "algorithm", "software",
            "computer", "programming", "database", "server", "chatgpt", "language model", " ai ", "semiconductor",
            "hospital", "doctor", "disease", "patient", "treatment", "symptom", "medication", "medical", "cancer",
            "surgery", "clinical", "vaccine", "virus", "infection", "diagnosis", "therapy", "health", "nutrition"],
}
SPAM_COMMON = ["casino", "viagra", "cialis", "porn", "xxx", "escort", "bitcoin casino", "slot machine", "jackpot"]
SPAM.update({
    "fra": SPAM_COMMON + ["paris sportifs", "pariez", "cliquez ici", "téléchargement gratuit", "rencontre coquine"],
    "deu": SPAM_COMMON + ["sportwetten", "hier klicken", "kostenlos herunterladen", "online wetten"],
    "spa": SPAM_COMMON + ["apuestas", "haga clic aquí", "descarga gratis", "tragamonedas"],
    "jpn": SPAM_COMMON + ["カジノ", "出会い系", "アダルト", "無料ダウンロード", "パチンコ", "副業で稼ぐ", "エロ"],
    "rus": SPAM_COMMON + ["казино", "ставки на спорт", "порно", "скачать бесплатно", "интим", "букмекер"],
    "arb": SPAM_COMMON + ["كازينو", "مراهنات", "اباحي", "تحميل مجاني", "اضغط هنا"],
    "hin": SPAM_COMMON + ["कैसीनो", "सट्टा", "अश्लील", "मुफ्त डाउनलोड", "यहाँ क्लिक करें"],
    "swh": SPAM_COMMON + ["kamari", "bonyeza hapa", "pakua bure", "betting"],
    "vie": SPAM_COMMON + ["cá cược", "nhà cái", "sòng bài", "tải miễn phí", "nhấn vào đây", "lô đề"],
})
_AI = ["ChatGPT", "GPT", "AI"]
TOPIC.update({
    "fra": _AI + ["intelligence artificielle", "apprentissage automatique", "réseau de neurones", "algorithme", "logiciel",
                  "ordinateur", "informatique", "programmation", "base de données", "serveur", "hôpital", "médecin",
                  "maladie", "patient", "traitement", "symptôme", "médicament", "médical", "cancer", "chirurgie",
                  "clinique", "vaccin", "virus", "infection", "diagnostic", "thérapie", "santé"],
    "deu": _AI + ["künstliche intelligenz", "maschinelles lernen", "neuronales netz", "algorithmus", "software",
                  "computer", "programmierung", "datenbank", "server", "krankenhaus", "klinik", "arzt", "ärztin",
                  "krankheit", "patient", "patientin", "behandlung", "symptom", "medikament", "medizin", "medizinisch",
                  "krebs", "operation", "impfstoff", "virus", "infektion", "diagnose", "therapie", "gesundheit"],
    "spa": _AI + ["inteligencia artificial", "aprendizaje automático", "red neuronal", "algoritmo", "software",
                  "computadora", "ordenador", "informática", "programación", "base de datos", "servidor", "hospital",
                  "médico", "enfermedad", "paciente", "tratamiento", "síntoma", "medicamento", "cáncer", "cirugía",
                  "clínica", "vacuna", "virus", "infección", "diagnóstico", "terapia", "salud"],
    "jpn": _AI + ["人工知能", "機械学習", "深層学習", "ニューラルネットワーク", "アルゴリズム", "ソフトウェア", "コンピュータ",
                  "プログラミング", "データベース", "サーバー", "半導体", "病院", "医師", "医者", "病気", "疾患", "患者",
                  "治療", "症状", "薬", "医療", "医学", "がん", "癌", "手術", "臨床", "ワクチン", "ウイルス", "感染",
                  "診断", "健康", "サプリ"],
    "rus": _AI + ["искусственный интеллект", "машинное обучение", "нейронн", "алгоритм", "программн", "компьютер",
                  "программирован", "база данных", "сервер", "больниц", "врач", "болезн", "заболеван", "пациент",
                  "лечени", "симптом", "лекарств", "медицин", "рак", "операци", "клиник", "вакцин", "вирус",
                  "инфекци", "диагноз", "терапи", "здоровь"],
    "arb": _AI + ["الذكاء الاصطناعي", "تعلم الآلة", "التعلم الآلي", "الشبكات العصبية", "خوارزمية", "برمجيات", "برنامج",
                  "حاسوب", "كمبيوتر", "برمجة", "قاعدة بيانات", "خادم", "مستشفى", "طبيب", "مرض", "المرض", "مريض", "علاج",
                  "أعراض", "دواء", "أدوية", "طبي", "سرطان", "جراحة", "لقاح", "فيروس", "عدوى", "تشخيص", "صحة"],
    "hin": _AI + ["कृत्रिम बुद्धिमत्ता", "मशीन लर्निंग", "एल्गोरिदम", "सॉफ्टवेयर", "कंप्यूटर", "प्रोग्रामिंग", "डेटाबेस",
                  "सर्वर", "अस्पताल", "डॉक्टर", "चिकित्सक", "बीमारी", "रोग", "मरीज", "रोगी", "इलाज", "उपचार", "लक्षण",
                  "दवा", "दवाई", "चिकित्सा", "कैंसर", "सर्जरी", "टीका", "वैक्सीन", "वायरस", "संक्रमण", "स्वास्थ्य"],
    "swh": _AI + ["akili bandia", "kompyuta", "programu", "algorithm", "hospitali", "daktari", "ugonjwa", "magonjwa",
                  "mgonjwa", "wagonjwa", "matibabu", "dalili", "dawa", "saratani", "upasuaji", "chanjo", "virusi",
                  "maambukizi", "afya", "kliniki"],
    "vie": _AI + ["trí tuệ nhân tạo", "học máy", "mạng nơ-ron", "thuật toán", "phần mềm", "máy tính", "lập trình",
                  "cơ sở dữ liệu", "máy chủ", "bệnh viện", "bác sĩ", "bệnh nhân", "căn bệnh", "điều trị", "triệu chứng",
                  "thuốc", "y tế", "y học", "ung thư", "phẫu thuật", "lâm sàng", "vắc xin", "virus", "nhiễm trùng",
                  "chẩn đoán", "sức khỏe"],
})
CJK = {"cmn", "jpn"}           # no spaces between words: character shingles, character-level filters
SCRIPT = {"cmn": "Hani", "jpn": "Jpan", "eng": "Latn", "fra": "Latn", "deu": "Latn", "spa": "Latn", "swh": "Latn",
          "vie": "Latn", "rus": "Cyrl", "arb": "Arab", "hin": "Deva"}
WIKI_CUT_MORE = ["Notes et références", "Références", "Liens externes", "Voir aussi", "Einzelnachweise", "Weblinks",
                 "Siehe auch", "Literatur", "Referencias", "Enlaces externos", "Véase también", "脚注", "参考文献",
                 "関連項目", "外部リンク", "出典", "Примечания", "Литература", "Ссылки", "См. также", "المراجع", "مراجع",
                 "وصلات خارجية", "انظر أيضا", "सन्दर्भ", "संदर्भ", "बाहरी कड़ियाँ", "इन्हें भी देखें", "Marejeo",
                 "Viungo vya nje", "Tazama pia", "Tham khảo", "Liên kết ngoài", "Xem thêm", "Chú thích"]
WIKI_CUT = ["参考资料", "參考資料", "参考文献", "參考文獻", "外部链接", "外部連結", "外部連接", "参见", "參見", "注释",
            "註釋", "相关条目", "相關條目", "延伸阅读", "延伸閱讀", "References", "External links", "See also"]
SENT_END = {"cmn": "。！？；…", "jpn": "。！？…", "eng": ".!?", "fra": ".!?", "deu": ".!?", "spa": ".!?", "swh": ".!?",
            "vie": ".!?", "rus": ".!?", "arb": ".!?؟", "hin": "।.!?"}
COMMAS = {"cmn": "，,", "jpn": "、，,"}
# characters that exist only in one script (a small, common set; enough to label a document)
SIMP_ONLY = set("们这对说时来会国过还发后为见长门问头写书车东马鸟鱼龙区学觉开关间万与业丝丢两严丧个丰临丽举义乌乐乔习乡")
TRAD_ONLY = set("們這對說時來會國過還發後為見長門問頭寫書車東馬鳥魚龍區學覺開關間萬與業絲丟兩嚴喪個豐臨麗舉義烏樂喬習鄉")


def han(c):
    o = ord(c)
    return 0x4E00 <= o <= 0x9FFF or 0x3400 <= o <= 0x4DBF or 0xF900 <= o <= 0xFAFF


def in_script(c, sc):
    o = ord(c)
    if sc == "Hani":
        return han(c)
    if sc == "Jpan":
        return han(c) or 0x3040 <= o <= 0x30FF
    if sc == "Latn":
        return c.isalpha() and (o < 0x250 or 0x1E00 <= o <= 0x1EFF)
    if sc == "Cyrl":
        return 0x400 <= o <= 0x52F
    if sc == "Arab":
        return 0x600 <= o <= 0x6FF or 0x750 <= o <= 0x77F or 0xFB50 <= o <= 0xFEFF
    if sc == "Deva":
        return 0x900 <= o <= 0x97F
    raise ValueError(sc)


def script_of(t, lang):
    if lang != "cmn":
        return SCRIPT[lang]
    s = sum(c in SIMP_ONLY for c in t)
    tr = sum(c in TRAD_ONLY for c in t)
    return "Hans" if s > 2 * tr else "Hant" if tr > 2 * s else "mixed"


def clean(text, kind):
    lines = [ln.strip() for ln in text.replace("\r", "").split("\n")]
    if kind == "wiki":
        out = []
        for ln in lines:
            if any(ln == w or ln.startswith(w) and len(ln) <= len(w) + 4 for w in WIKI_CUT + WIKI_CUT_MORE):
                break
            out.append(ln)
        lines = out
    lines = [re.sub(r"[ \t　]+", " ", ln) for ln in lines if ln]
    return "\n".join(lines)


def cut(text, max_chars, lang):
    if len(text) <= max_chars:
        return text
    t = text[:max_chars]
    ends = [t.rfind(c) for c in SENT_END[lang]]
    e = max(ends)
    return t[:e + 1] if e > max_chars // 2 else t


def quality(t, lang, a):
    """-> reason string or None."""
    if len(t) < a.min_chars:
        return "short"
    letters = [c for c in t if not c.isspace()]
    if not letters:
        return "empty"
    if lang == "eng":
        share = sum(c.isascii() and c.isalpha() for c in letters) / len(letters)
    else:
        share = sum(in_script(c, SCRIPT[lang]) for c in letters) / len(letters)
    if share < a.min_script_share:
        return "script_share"
    ends = sum(t.count(c) for c in SENT_END[lang])
    per = len(t) / max(1, ends)
    if per > (a.max_chars_per_sentence if lang in CJK else 4 * a.max_chars_per_sentence):
        return "few_sentences"
    lines = [ln for ln in t.split("\n") if ln]
    if lines:
        c = Counter(lines)
        if sum(v for v in c.values() if v > 1) / len(lines) > 0.2:
            return "repeated_lines"
        if lang in CJK and sum(len(ln) < 12 for ln in lines) / len(lines) > 0.5:
            return "list_like"
    if lang in CJK:         # headline salad / spun SEO pages: unrelated Han phrases joined by spaces
        sc = SCRIPT[lang]
        sp = sum(1 for i in range(1, len(t) - 1) if t[i] == " " and in_script(t[i - 1], sc) and in_script(t[i + 1], sc))
        if sp * 100 / len(t) > a.max_han_spaces:
            return "headline_salad"
        if sum(t.count(c) for c in COMMAS[lang]) < len(t) / 80:
            return "few_clauses"
        odd = sum(1 for c in t if not (in_script(c, sc) or c.isascii() or 0x30FC == ord(c) or 0x3000 <= ord(c) <= 0x303F or 0xFF00 <= ord(c) <= 0xFFEF
                                      or c in "“”‘’…—·《》〈〉【】「」『』•°％"))
        if odd > 3 or "&n" in t or "&#" in t:
            return "odd_chars_or_html"
    if sum(c.isdigit() for c in t) / len(t) > 0.12 or len(re.findall(r"https?://|www\.", t)) > 2:
        return "digits_urls"
    low = t.lower()
    # Latin-script spam words match whole words only ("xxx" was hitting "XXXe siècle", "escort" "escorte")
    if any((re.search(r"(?<!\w)" + re.escape(w.lower()) + r"(?!\w)", low) if re.match(r"^[a-z0-9 ]+$", w.lower())
            else w.lower() in low) for w in SPAM[lang]):
        return "spam"
    if lang not in CJK and ("&nbsp" in t or "&#" in t):
        return "odd_chars_or_html"
    return None


def topic_hit(title, t, lang, a):
    low = t.lower() if lang not in CJK else t
    words = TOPIC[lang]
    def count(w, s):        # Latin-script words match whole words only (review: "server" in "observer"), and Russian
        # stems (no trailing word boundary) as word starts; Han / kana / Arabic / Devanagari as substrings
        if re.search(r"[A-Za-z]", w):
            return len(re.findall(r"(?<![A-Za-z])" + re.escape(w.strip()) + r"(?![A-Za-z])", s, flags=re.I))
        if lang in ("fra", "deu", "spa", "swh", "vie"):
            return len(re.findall(r"(?<!\w)" + re.escape(w) + r"(?!\w)", s.lower()))
        if lang == "rus":
            return len(re.findall(r"(?<!\w)" + re.escape(w), s.lower()))
        return s.count(w)
    if title and any(count(w, title) for w in words):
        return "topic_title"
    n = sum(count(w, low) for w in words if w.strip())
    return "topic" if n >= a.topic_hits else None


def shingles(t, lang):
    if lang in CJK:
        s = re.sub(r"\s+", "", t)
        g = [s[i:i + 5] for i in range(max(1, len(s) - 4))]
    else:
        w = re.findall(r"\w+", t.lower())
        g = [" ".join(w[i:i + 3]) for i in range(max(1, len(w) - 2))]
    return np.unique(np.array([zlib.crc32(x.encode()) for x in g], dtype=np.uint64))


def minhash(sh, A, B, P=(1 << 61) - 1):
    return ((np.outer(sh, A) + B) % P).min(axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", required=True, choices=sorted(SPAM))
    ap.add_argument("--out", required=True)
    ap.add_argument("--source", action="append", required=True, help="KIND:NAME:PATH (KIND wiki | web)")
    ap.add_argument("--per-source", type=int, default=3000)
    ap.add_argument("--min-chars", type=int, default=0, help="0 = 600 for CJK, 1200 otherwise (a 256-token window)")
    ap.add_argument("--max-chars", type=int, default=3000)
    ap.add_argument("--min-script-share", type=float, default=0.7)
    ap.add_argument("--max-chars-per-sentence", type=int, default=120)
    ap.add_argument("--topic-hits", type=int, default=2)
    ap.add_argument("--max-han-spaces", type=float, default=0.5, help="spaces between Han characters per 100 chars")
    ap.add_argument("--near", type=float, default=0.5, help="estimated Jaccard at or above which a document is a near-duplicate")
    ap.add_argument("--perms", type=int, default=128)
    ap.add_argument("--bands", type=int, default=32, help="4 rows per band: candidate threshold about 0.42 (review)")
    ap.add_argument("--val", type=float, default=0.1)
    ap.add_argument("--held", type=float, default=0.1)
    ap.add_argument("--held2", type=float, default=0.05, help="a second held split, never used to steer data (Pierre's "
                    "held-miss repair loop uses held to find missing sub-topics; held2 stays the clean final test)")
    ap.add_argument("--scan", type=int, default=60000, help="rows read per source (seeded order) before filtering")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    a.min_chars = a.min_chars or (600 if a.lang in CJK else 1200)
    import pyarrow.parquet as pq
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(a.seed)
    A = rng.integers(1, (1 << 61) - 1, a.perms, dtype=np.uint64)
    B = rng.integers(0, (1 << 61) - 1, a.perms, dtype=np.uint64)
    rows_per_band = a.perms // a.bands
    drops = defaultdict(Counter)
    kept, seen_exact, buckets, sigs = [], set(), defaultdict(list), []
    srcinfo = {}
    got_by = Counter()          # a NAME given twice (two shards of one source) shares one --per-source cap (Pierre
                                # 2026-11-16: balance the Wikipedia share where one shard is mostly stubs)
    for spec in a.source:
        kind, name, path = spec.split(":", 2)
        p = Path(path).expanduser()
        sj = next((q / "SOURCE.json" for q in [p.parent, *p.parents] if (q / "SOURCE.json").exists()), None)
        meta = json.loads(sj.read_text()) if sj else {}
        info = {"kind": kind, "path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                "url": meta.get("url"), "revision": meta.get("revision"), "license": meta.get("license"),
                "redistribute": meta.get("redistribute")}
        if name in srcinfo:
            prev = srcinfo[name]
            info = {**prev, "shards": prev.get("shards", [{k: prev[k] for k in ("path", "sha256", "revision")}]) +
                    [{k: info[k] for k in ("path", "sha256", "revision")}]}
        srcinfo[name] = info
        t = pq.read_table(p, columns=[c for c in ("title", "text", "url", "id") if c in pq.read_schema(p).names])
        n = t.num_rows
        order = np.random.default_rng(a.seed + zlib.crc32(name.encode())).permutation(n)[:a.scan]
        cols = {c: t.column(c).to_pylist() for c in t.column_names}
        got = got_by[name]
        for i in order:
            if got >= a.per_source:
                break
            title = cols.get("title", [None] * n)[i]
            text = unicodedata.normalize("NFC", cols["text"][i] or "")
            text = cut(clean(text, kind), a.max_chars, a.lang)
            why = quality(text, a.lang, a) or topic_hit(title, text, a.lang, a)
            if why:
                drops[name][why] += 1
                continue
            key = hashlib.sha256(re.sub(r"\s+", "", text).encode()).hexdigest()
            if key in seen_exact:
                drops[name]["exact_dup"] += 1
                continue
            sig = minhash(shingles(text, a.lang), A, B)
            cand = set()
            for b in range(a.bands):
                h = hash(sig[b * rows_per_band:(b + 1) * rows_per_band].tobytes())
                cand.update(buckets[(b, h)])
            if any(float((sigs[j] == sig).mean()) >= a.near for j in cand):
                drops[name]["near_dup"] += 1
                continue
            j = len(sigs)
            sigs.append(sig)
            for b in range(a.bands):
                buckets[(b, hash(sig[b * rows_per_band:(b + 1) * rows_per_band].tobytes()))].append(j)
            seen_exact.add(key)
            src_id = cols.get("id", [None] * n)[i] or cols.get("url", [None] * n)[i] or str(i)
            kept.append({"text": text, "source": name, "source_row": int(i), "source_id": str(src_id),
                         "title": title, "url": cols.get("url", [None] * n)[i], "script": script_of(text, a.lang)})
            got += 1
        got_by[name] = got
        print(name, "kept", got, "of", sum(drops[name].values()) + got, "scanned;", dict(drops[name]), flush=True)
    # document-level split, stratified by source x script (seeded)
    strata = defaultdict(list)
    for r in kept:
        strata[(r["source"], r["script"])].append(r)
    R = random.Random(a.seed + 1)
    rows = []
    for k in sorted(strata):
        rs = strata[k]
        R.shuffle(rs)
        nv, nh, n2 = round(len(rs) * a.val), round(len(rs) * a.held), round(len(rs) * a.held2)
        for j, r in enumerate(rs):
            split = "val" if j < nv else "held" if j < nv + nh else "held2" if j < nv + nh + n2 else "fit"
            s = srcinfo[r["source"]]
            doc = hashlib.sha256(r["text"].encode()).hexdigest()[:16]
            rows.append({"text": r["text"], "id": f"{a.lang}-{doc}", "doc_id": doc, "topics": [f"lang.{a.lang}"],
                         "split": split, "origin": "natural", "group": f"{r['source']}:{r['source_id']}",
                         "source": r["source"], "source_id": r["source_id"], "source_row": r["source_row"],
                         "title": r["title"], "url": r["url"], "source_url": s["url"], "source_revision": s["revision"],
                         "license": s["license"], "redistribute": s["redistribute"], "script": r["script"],
                         "chars": len(r["text"]), "method": "amoeba_lang_prose_build"})
    rows.sort(key=lambda r: r["id"])
    cp = out / "corpus.jsonl"
    cp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    man = {"schema": "mlx2.amoeba-lang-prose.v1", "lang": a.lang, "args": vars(a), "sources": srcinfo,
           "corpus_sha256": hashlib.sha256(cp.read_bytes()).hexdigest(), "documents": len(rows),
           "drops": {k: dict(v) for k, v in drops.items()},
           "counts": {"split": Counter(r["split"] for r in rows), "source": Counter(r["source"] for r in rows),
                      "script": Counter(r["script"] for r in rows),
                      "split_source": Counter(f"{r['split']}:{r['source']}" for r in rows)},
           "chars": {"total": sum(r["chars"] for r in rows),
                     "median": sorted(r["chars"] for r in rows)[len(rows) // 2] if rows else 0},
           "exclusions": {"spam": SPAM[a.lang], "topic": TOPIC[a.lang]}}
    (out / "manifest.json").write_text(json.dumps(man, indent=1, ensure_ascii=False, default=dict))
    print(json.dumps({k: man[k] for k in ("documents", "counts", "chars", "corpus_sha256")}, ensure_ascii=False, default=dict))


if __name__ == "__main__":
    sys.exit(main())
