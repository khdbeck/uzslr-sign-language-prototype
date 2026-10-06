"""Official-test accuracy for every saved epoch (ckpt/epoch<N>.pt), for the chart.

Epoch 0 is the official model with the 51-class head (no training). This is
for monitoring only: the shipped model is always the last epoch, so the test
split is never used to pick a checkpoint.
"""
import json
import re

import torch

from common import ROOT, ClipDataset, SlovoMViT, load_classes, load_rows, split_rows, to_input
from evaluate import score


def main():
    device = torch.device("cuda")
    classes = load_classes()
    labels = [c["label"] for c in classes]
    _, _, test_rows = split_rows(load_rows())
    loader = torch.utils.data.DataLoader(ClipDataset(test_rows, False), 4, num_workers=4)
    model = SlovoMViT([c["slovoId"] for c in classes]).to(device).eval()
    official = {k: v.clone() for k, v in model.state_dict().items()}
    checkpoints = sorted(ROOT.glob("ckpt/epoch*.pt"), key=lambda p: int(re.findall(r"\d+", p.stem)[0]))
    history = {h["epoch"]: h for h in json.loads((ROOT / "history.json").read_text())["history"]}
    rows = []
    for epoch, state in [(0, official), *((int(re.findall(r"\d+", p.stem)[0]), p) for p in checkpoints)]:
        model.load_state_dict(state if isinstance(state, dict) else torch.load(state, map_location=device))
        preds = []
        with torch.no_grad():
            for clips, _ in loader:
                preds += model(to_input(clips, device)).topk(5, 1).indices.cpu().tolist()
        result = score(preds, test_rows, labels)
        row = {"epoch": epoch, **{k: round(result[k], 4) for k in ("top1", "top5", "wordTop1", "noEventTop1")},
               "trainLoss": history.get(epoch, {}).get("loss"), "val": history.get(epoch, {}).get("val")}
        rows.append(row)
        print(row, flush=True)
    (ROOT / "epochs-test.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
