"""Shared data, sampling and model code for the 50-word Slovo fine-tune."""
import csv
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
CACHE = ROOT / "cache"
OFFICIAL_PT = ROOT.parents[1] / "web_app" / "models" / "slovo" / "pt" / "mvit32-2.pt"
FRAMES, STRIDE, WINDOW = 32, 2, 63
MEAN = torch.tensor([123.675, 116.28, 103.53]).view(1, 3, 1, 1, 1)
STD = torch.tensor([58.395, 57.12, 57.375]).view(1, 3, 1, 1, 1)


def load_classes():
    return json.loads((ROOT / "classes.json").read_text(encoding="utf-8"))


def load_rows():
    """Annotation rows of the selected classes with a 'cls' index (0..50)."""
    index = {("no_event" if c["label"] == "---" else c["label"]): i for i, c in enumerate(load_classes())}
    with (DATA / "annotations.csv").open(encoding="utf-8") as stream:
        rows = [row for row in csv.DictReader(stream, delimiter="\t") if row["text"] in index]
    for row in rows:
        row["cls"] = index[row["text"]]
    return rows


def split_rows(rows, val_per_class=2, seed=0):
    """Official test stays untouched; validation holds out a few train clips per class.

    Signers record many classes, so holding out whole signers would empty the
    train split. Validation is only used to pick the epoch; the reported
    numbers come from the official test split (new signers).
    """
    test = [r for r in rows if r["train"] != "True"]
    train = sorted((r for r in rows if r["train"] == "True"), key=lambda r: r["attachment_id"])
    rng = random.Random(seed)
    by_class = {}
    for row in train:
        by_class.setdefault(row["cls"], []).append(row)
    val_ids = set()
    for items in by_class.values():
        take = val_per_class if items[0]["text"] != "no_event" else val_per_class * 10
        val_ids.update(r["attachment_id"] for r in rng.sample(items, take))
    val = [r for r in train if r["attachment_id"] in val_ids]
    train = [r for r in train if r["attachment_id"] not in val_ids]
    return train, val, test


def clip_indices(count, rng=None):
    """App protocol: centred 63-frame window, every 2nd frame, repeat the last.

    With rng (training) the window start is random and the stride varies 1..3
    to simulate faster and slower signing.
    """
    if rng is None:
        start = max(0, (count - WINDOW) // 2)
        idx = list(range(start, min(count, start + WINDOW), STRIDE))
        return idx + [idx[-1]] * (FRAMES - len(idx))
    stride = rng.choices([1, 2, 3], weights=[1, 3, 1])[0]
    start = rng.randint(0, max(0, count - (FRAMES - 1) * stride - 1))
    return [min(start + i * stride, count - 1) for i in range(FRAMES)]


class ClipDataset(torch.utils.data.Dataset):
    def __init__(self, rows, train):
        self.rows, self.train = rows, train

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        frames = np.load(CACHE / f"{row['attachment_id']}.npy", mmap_mode="r")
        rng = random.Random() if self.train else None
        clip = np.ascontiguousarray(frames[clip_indices(len(frames), rng)])  # T,H,W,C
        return torch.from_numpy(clip), row["cls"]


def to_input(batch, device):
    """uint8 [B,T,H,W,C] -> normalized float [B,3,T,H,W] on device."""
    x = batch.to(device, non_blocking=True).permute(0, 4, 1, 2, 3).float()
    return (x - MEAN.to(device)) / STD.to(device)


def augment(x):
    """Per-clip random affine + brightness/contrast, same for every frame of a clip."""
    b, c, t, h, w = x.shape
    angle = torch.empty(b, device=x.device).uniform_(-math.radians(6), math.radians(6))
    scale = torch.empty(b, device=x.device).uniform_(0.88, 1.12)
    shift = torch.empty(b, 2, device=x.device).uniform_(-0.08, 0.08)
    cos, sin = torch.cos(angle) / scale, torch.sin(angle) / scale
    theta = torch.stack([torch.stack([cos, -sin, shift[:, 0]], 1),
                         torch.stack([sin, cos, shift[:, 1]], 1)], 1)
    grid = F.affine_grid(theta, (b, c * t, h, w), align_corners=False)
    pad = ((114 - MEAN.to(x.device)) / STD.to(x.device)).view(1, 3, 1, 1, 1)
    out = F.grid_sample((x - pad).reshape(b, c * t, h, w), grid, align_corners=False)
    x = out.view(b, c, t, h, w) + pad
    contrast = torch.empty(b, 1, 1, 1, 1, device=x.device).uniform_(0.85, 1.15)
    bright = torch.empty(b, 1, 1, 1, 1, device=x.device).uniform_(-0.25, 0.25)
    return x * contrast + bright


class SlovoMViT(nn.Module):
    """Official Slovo MViTv2-small-32-2 backbone with a new 51-way head.

    The TorchScript blocks are called one by one so the first `frozen` blocks
    run without gradients (activation checkpointing breaks on the traced blocks).
    """

    def __init__(self, class_ids, frozen=8, dropout=0.3):
        super().__init__()
        official = torch.jit.load(str(OFFICIAL_PT), map_location="cpu")
        backbone = official.backbone
        self.patch_embed, self.norm3 = backbone.patch_embed, backbone.norm3
        self.blocks = nn.ModuleList(getattr(backbone.blocks, str(i)) for i in range(16))
        self.cls_token = nn.Parameter(backbone.cls_token.detach().clone())
        self.frozen = frozen
        fc = official.cls_head.fc_cls
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(fc.weight.shape[1], len(class_ids))
        with torch.no_grad():  # start from the official classifier rows
            self.head.weight.copy_(fc.weight[class_ids])
            self.head.bias.copy_(fc.bias[class_ids])
        for module in [self.patch_embed, *self.blocks[:frozen]]:
            for p in module.parameters():
                p.requires_grad_(False)
        if frozen > 0:
            self.cls_token.requires_grad_(False)

    def features(self, x):
        with torch.set_grad_enabled(torch.is_grad_enabled() and self.frozen == 0):
            out = self.patch_embed(x)
            tokens = torch.cat([self.cls_token.expand(x.shape[0], -1, -1), out[0]], 1)
            state = (tokens, *out[1:])
            for block in self.blocks[:self.frozen]:
                state = block(*state)
        for block in self.blocks[self.frozen:]:
            state = block(*state)
        return self.norm3(state[0])[:, 0]

    def forward(self, x):
        return self.head(self.dropout(self.features(x)))


class ExportWrapper(nn.Module):
    """App-compatible signature: [1, clips, 3, T, H, W] -> softmax mean over clips."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, inputs):
        b, n = inputs.shape[:2]
        logits = self.model(inputs.flatten(0, 1))
        return torch.softmax(logits.view(b, n, -1), 2).mean(1)
