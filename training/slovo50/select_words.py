"""Pick the 50 most frequent everyday Russian words from the Slovo vocabulary.

Frequency comes from wordfreq (Zipf scale). Only content words are kept
(nouns, verbs, adjectives, adverbs, numerals, interjections, predicatives;
pronominal adjectives like "этот"/"мой" are dropped),
plus the answers "да"/"нет". Function words (prepositions, conjunctions,
particles, pronoun case forms), fingerspelled letters and multi-word phrases
are skipped, and labels sharing a lemma are deduplicated.
Output: classes.json with original Slovo class ids and the no_event class.
"""
import json
import re
from pathlib import Path

import pymorphy3
from wordfreq import zipf_frequency

ROOT = Path(__file__).resolve().parent
LABELS = ROOT.parents[1] / "web_app" / "models" / "slovo" / "labels.json"
COUNT = 50
NO_EVENT_ID = 1000
CONTENT_POS = {"NOUN", "INFN", "VERB", "ADJF", "ADJS", "ADVB", "NUMR", "INTJ", "PRED", "COMP"}
ALWAYS = {"да", "нет"}


def main():
    morph = pymorphy3.MorphAnalyzer()
    labels = json.loads(LABELS.read_text(encoding="utf-8"))
    scored = []
    for class_id, label in enumerate(labels[:NO_EVENT_ID]):
        text = re.sub(r"[^\w\s-]", "", label).strip().lower()
        if len(text) <= 1 or " " in text:
            continue
        parse = morph.parse(text)[0]
        if text not in ALWAYS and (parse.tag.POS not in CONTENT_POS or "Apro" in parse.tag):
            continue
        scored.append((zipf_frequency(text, "ru"), class_id, label, parse.normal_form))
    scored.sort(key=lambda item: (-item[0], item[1]))
    chosen, lemmas = [], set()
    for item in scored:
        if item[3] in lemmas:
            continue
        lemmas.add(item[3])
        chosen.append(item)
        if len(chosen) == COUNT:
            break
    classes = [{"slovoId": cid, "label": label, "zipf": round(z, 2)}
               for z, cid, label, _ in sorted(chosen, key=lambda item: item[1])]
    classes.append({"slovoId": NO_EVENT_ID, "label": "---", "zipf": None})
    (ROOT / "classes.json").write_text(json.dumps(classes, ensure_ascii=False, indent=2), encoding="utf-8")
    print(", ".join(f"{label} ({z:.2f})" for z, _, label, _ in chosen))


if __name__ == "__main__":
    main()
