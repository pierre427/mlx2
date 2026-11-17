"""Build the geo.capitals topic spec (closed answer set, so chance is known).

Countries split 60/40 into `extract` (used to find the box) and `heldout`
(used for necessity), so a box that only memorised its probe prompts fails.
Scoring is restricted to the set of candidate first tokens; chance = 1/N.
A neighbouring control topic (official language) checks that a box is about
capitals rather than "facts about countries".
"""
import json, random, sys
from pathlib import Path
from transformers import AutoTokenizer
PAIRS = """Afghanistan:Kabul;Albania:Tirana;Algeria:Algiers;Andorra:Andorra la Vella;Angola:Luanda;Argentina:Buenos Aires;Armenia:Yerevan;Australia:Canberra;Austria:Vienna;Azerbaijan:Baku;Bahamas:Nassau;Bahrain:Manama;Bangladesh:Dhaka;Barbados:Bridgetown;Belarus:Minsk;Belgium:Brussels;Belize:Belmopan;Benin:Porto-Novo;Bhutan:Thimphu;Bolivia:Sucre;Botswana:Gaborone;Brazil:Brasília;Bulgaria:Sofia;Burundi:Gitega;Cambodia:Phnom Penh;Cameroon:Yaoundé;Canada:Ottawa;Chad:N'Djamena;Chile:Santiago;China:Beijing;Colombia:Bogotá;Croatia:Zagreb;Cuba:Havana;Cyprus:Nicosia;Denmark:Copenhagen;Ecuador:Quito;Egypt:Cairo;Eritrea:Asmara;Estonia:Tallinn;Ethiopia:Addis Ababa;Fiji:Suva;Finland:Helsinki;France:Paris;Gabon:Libreville;Georgia:Tbilisi;Germany:Berlin;Ghana:Accra;Greece:Athens;Guinea:Conakry;Guyana:Georgetown;Haiti:Port-au-Prince;Honduras:Tegucigalpa;Hungary:Budapest;Iceland:Reykjavik;India:New Delhi;Indonesia:Jakarta;Iran:Tehran;Iraq:Baghdad;Ireland:Dublin;Israel:Jerusalem;Italy:Rome;Jamaica:Kingston;Japan:Tokyo;Jordan:Amman;Kazakhstan:Astana;Kenya:Nairobi;Kyrgyzstan:Bishkek;Laos:Vientiane;Latvia:Riga;Lebanon:Beirut;Lesotho:Maseru;Liberia:Monrovia;Libya:Tripoli;Lithuania:Vilnius;Madagascar:Antananarivo;Malawi:Lilongwe;Malaysia:Kuala Lumpur;Mali:Bamako;Malta:Valletta;Mauritania:Nouakchott;Moldova:Chisinau;Mongolia:Ulaanbaatar;Montenegro:Podgorica;Morocco:Rabat;Mozambique:Maputo;Myanmar:Naypyidaw;Namibia:Windhoek;Nepal:Kathmandu;Netherlands:Amsterdam;New Zealand:Wellington;Nicaragua:Managua;Niger:Niamey;Nigeria:Abuja;North Korea:Pyongyang;Norway:Oslo;Oman:Muscat;Pakistan:Islamabad;Paraguay:Asunción;Peru:Lima;Philippines:Manila;Poland:Warsaw;Portugal:Lisbon;Qatar:Doha;Romania:Bucharest;Russia:Moscow;Rwanda:Kigali;Samoa:Apia;Senegal:Dakar;Serbia:Belgrade;Slovakia:Bratislava;Slovenia:Ljubljana;Somalia:Mogadishu;South Africa:Pretoria;South Korea:Seoul;Spain:Madrid;Sri Lanka:Colombo;Sudan:Khartoum;Suriname:Paramaribo;Sweden:Stockholm;Switzerland:Bern;Syria:Damascus;Taiwan:Taipei;Tajikistan:Dushanbe;Tanzania:Dodoma;Thailand:Bangkok;Togo:Lomé;Tonga:Nuku'alofa;Tunisia:Tunis;Turkey:Ankara;Turkmenistan:Ashgabat;Uganda:Kampala;Ukraine:Kyiv;United Kingdom:London;United States:Washington;Uruguay:Montevideo;Uzbekistan:Tashkent;Venezuela:Caracas;Vietnam:Hanoi;Yemen:Sanaa;Zambia:Lusaka;Zimbabwe:Harare"""
LANG = {"France": "French", "Germany": "German", "Spain": "Spanish", "Italy": "Italian", "Japan": "Japanese", "China": "Chinese",
        "Russia": "Russian", "Brazil": "Portuguese", "Portugal": "Portuguese", "Mexico": "Spanish", "Argentina": "Spanish",
        "Greece": "Greek", "Turkey": "Turkish", "Poland": "Polish", "Sweden": "Swedish", "Norway": "Norwegian", "Finland": "Finnish",
        "Denmark": "Danish", "Netherlands": "Dutch", "Hungary": "Hungarian", "Romania": "Romanian", "Iran": "Persian",
        "Egypt": "Arabic", "Iraq": "Arabic", "Thailand": "Thai", "Vietnam": "Vietnamese", "South Korea": "Korean",
        "Indonesia": "Indonesian", "Ukraine": "Ukrainian", "Israel": "Hebrew", "Bulgaria": "Bulgarian", "Czechia": "Czech",
        "Croatia": "Croatian", "Serbia": "Serbian", "Iceland": "Icelandic", "Chile": "Spanish", "Peru": "Spanish",
        "Colombia": "Spanish", "Cuba": "Spanish", "Lithuania": "Lithuanian", "Latvia": "Latvian", "Estonia": "Estonian"}
TEMPLATES = ["The capital of {c} is", "{c}'s capital city is", "Q: What is the capital of {c}?\nA:",
             "Fact: the capital city of {c} is called"]
LANG_TEMPLATES = ["The official language of {c} is", "In {c}, most people speak"]
def main(out):
    tok = AutoTokenizer.from_pretrained(sys.argv[2])
    pairs = [p.split(":", 1) for p in PAIRS.split(";")]
    first = {}
    for c, cap in pairs:
        t = tok(" " + cap, add_special_tokens=False)["input_ids"][0]
        first.setdefault(t, []).append((c, cap))
    keep = [v[0] for t, v in first.items() if len(v) == 1 and v[0][0] != v[0][1]]
    dropped = [v for t, v in first.items() if len(v) > 1]
    random.Random(1111).shuffle(keep)
    n = int(len(keep) * 0.6)
    split = {"extract": keep[:n], "heldout": keep[n:]}
    cands = {c: tok(" " + cap, add_special_tokens=False)["input_ids"][0] for c, cap in keep}
    langs = sorted(set(LANG.values()))
    lang_first = {l: tok(" " + l, add_special_tokens=False)["input_ids"][0] for l in langs}
    spec = {"name": "geo.capitals", "version": 1, "tokenizer": "Qwen/Qwen3-0.6B",
            "scoring": "restricted argmax over candidate first tokens; chance = 1/N",
            "candidates": {cap: cands[c] for c, cap in keep}, "chance": 1 / len(keep),
            "templates": TEMPLATES,
            "items": {k: [{"country": c, "capital": cap, "answer_token": cands[c]} for c, cap in v] for k, v in split.items()},
            "dropped_first_token_collisions": dropped,
            "control_topic": {"name": "geo.language", "templates": LANG_TEMPLATES,
                              "candidates": lang_first, "chance": 1 / len(lang_first),
                              "items": [{"country": c, "language": l, "answer_token": lang_first[l]} for c, l in LANG.items()]}}
    Path(out).write_text(json.dumps(spec, indent=1, ensure_ascii=False))
    print(len(keep), "countries kept;", len(dropped), "collision groups dropped;", {k: len(v) for k, v in split.items()})
    print("sample:", keep[:6]); print("language candidates", len(lang_first))
if __name__ == "__main__":
    main(sys.argv[1])
