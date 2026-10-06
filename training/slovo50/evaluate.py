"""Compare models on the official Slovo test split of the selected classes.

1. official-1000: current app model, argmax over all 1001 classes;
2. official-restricted: same probabilities, argmax over our 51 classes only;
3. finetuned: finetuned.pt from train.py.
All use the app's clip protocol (centred 63-frame window, stride 2).
"""
import json
from collections import Counter

import torch

from common import OFFICIAL_PT, ROOT, ClipDataset, SlovoMViT, load_classes, load_rows, split_rows, to_input


def score(preds, rows, labels):
    top1 = sum(p[0] == r["cls"] for p, r in zip(preds, rows))
    top5 = sum(r["cls"] in p[:5] for p, r in zip(preds, rows))
    words = [(p, r) for p, r in zip(preds, rows) if r["text"] != "no_event"]
    per_class = Counter(r["cls"] for _, r in words)
    hit = Counter(r["cls"] for p, r in words if p[0] == r["cls"])
    errors = Counter((labels[r["cls"]], labels[p[0]] if p[0] >= 0 else "(вне 50)")
                     for p, r in words if p[0] != r["cls"])
    return {
        "top1": top1 / len(rows), "top5": top5 / len(rows),
        "wordTop1": sum(hit.values()) / len(words),
        "macroWordTop1": sum(hit[c] / n for c, n in per_class.items()) / len(per_class),
        "noEventTop1": sum(p[0] == r["cls"] for p, r in zip(preds, rows) if r["text"] == "no_event")
        / max(1, sum(r["text"] == "no_event" for r in rows)),
        "perClass": {labels[c]: f"{hit[c]}/{n}" for c, n in sorted(per_class.items())},
        "topErrors": [f"{a} -> {b} ({n})" for (a, b), n in errors.most_common(15)],
    }


def main():
    device = torch.device("cuda")
    classes = load_classes()
    labels = [c["label"] for c in classes]
    ids = torch.tensor([c["slovoId"] for c in classes])
    _, _, test_rows = split_rows(load_rows())
    loader = torch.utils.data.DataLoader(ClipDataset(test_rows, False), 4, num_workers=4)

    official = torch.jit.load(str(OFFICIAL_PT), map_location=device).eval()
    tuned = SlovoMViT(ids.tolist()).to(device)
    tuned.load_state_dict(torch.load(ROOT / "finetuned.pt", map_location=device))
    tuned.eval()
    to_local = {int(s): i for i, s in enumerate(ids)}

    full, restricted, finetuned = [], [], []
    with torch.no_grad():
        for clips, _ in loader:
            x = to_input(clips, device)
            # The traced graph averages over the batch dim, so feed one clip at a time.
            probs = torch.cat([official(clip[None, None]) for clip in x]).cpu()  # [B, 1001]
            for row in probs:
                full.append([to_local.get(int(i), -1) for i in row.topk(5).indices])
                restricted.append(row[ids].topk(5).indices.tolist())
            finetuned += tuned(x).topk(5, 1).indices.cpu().tolist()

    result = {
        "testClips": len(test_rows),
        "testSigners": len({r["user_id"] for r in test_rows}),
        "official-1000": score(full, test_rows, labels),
        "official-restricted": score(restricted, test_rows, labels),
        "finetuned": score(finetuned, test_rows, labels),
    }
    (ROOT / "test-results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    for name in ("official-1000", "official-restricted", "finetuned"):
        r = result[name]
        print(f"{name:20s} top1 {r['top1']:.3f}  top5 {r['top5']:.3f}  words {r['wordTop1']:.3f}  "
              f"macro {r['macroWordTop1']:.3f}  no_event {r['noEventTop1']:.3f}")


if __name__ == "__main__":
    main()
