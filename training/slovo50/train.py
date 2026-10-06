"""Fine-tune the official Slovo MViTv2-small-32-2 on 50 words + no_event.

Train: official train split minus a few held-out clips per class (validation).
The official model has already seen those clips, so validation is only a sanity
check; the last epoch is kept. The official test split is never used here;
see evaluate.py.
"""
import argparse
import json
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from common import ROOT, ClipDataset, SlovoMViT, augment, load_classes, load_rows, split_rows, to_input


def run_eval(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for clips, labels in loader:
            pred = model(to_input(clips, device)).argmax(1).cpu()
            correct += int((pred == labels).sum())
            total += len(labels)
    return correct / max(1, total)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--accum", type=int, default=4)
    parser.add_argument("--frozen", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--limit", type=int, default=0, help="debug: steps per epoch")
    args = parser.parse_args()

    torch.manual_seed(0), random.seed(0), np.random.seed(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda")
    classes = load_classes()
    train_rows, val_rows, _ = split_rows(load_rows())
    print(f"train {len(train_rows)}  val {len(val_rows)}", flush=True)

    # Balance classes: no_event has ~15x more clips than each word.
    counts = np.bincount([r["cls"] for r in train_rows], minlength=len(classes))
    weights = [1.0 / counts[r["cls"]] for r in train_rows]
    per_epoch = len(classes) * int(np.median(counts))
    sampler = torch.utils.data.WeightedRandomSampler(weights, per_epoch, replacement=True)
    train_loader = torch.utils.data.DataLoader(
        ClipDataset(train_rows, True), args.batch, sampler=sampler,
        num_workers=4, pin_memory=True, drop_last=True, persistent_workers=True)
    val_loader = torch.utils.data.DataLoader(
        ClipDataset(val_rows, False), 4, num_workers=2, pin_memory=True)

    model = SlovoMViT([c["slovoId"] for c in classes], frozen=args.frozen).to(device)
    head = list(model.head.parameters())
    body = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("head.")]
    print(f"trainable: body {sum(p.numel() for p in body) / 1e6:.1f}M, head {sum(p.numel() for p in head)}",
          flush=True)
    optimizer = torch.optim.AdamW([{"params": body, "lr": args.lr},
                                   {"params": head, "lr": args.head_lr}], weight_decay=0.05)
    steps = args.epochs * (args.limit or len(train_loader)) // args.accum
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=[args.lr, args.head_lr], total_steps=steps, pct_start=0.1)

    print(f"epoch 0  val {run_eval(model, val_loader, device):.3f} (official weights)", flush=True)
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        start, seen, loss_sum, correct = time.time(), 0, 0.0, 0
        optimizer.zero_grad(set_to_none=True)
        for step, (clips, labels) in enumerate(train_loader, 1):
            labels = labels.to(device)
            logits = model(augment(to_input(clips, device)))
            loss = F.cross_entropy(logits, labels, label_smoothing=0.1)
            (loss / args.accum).backward()
            if step % args.accum == 0:
                torch.nn.utils.clip_grad_norm_([*body, *head], 1.0)
                optimizer.step(), scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            seen += len(labels)
            loss_sum += loss.item() * len(labels)
            correct += int((logits.argmax(1) == labels).sum())
            if step % 50 == 0:
                print(f"  {step}/{len(train_loader)} loss {loss_sum / seen:.3f} "
                      f"{(time.time() - start) / seen:.2f}s/clip "
                      f"mem {torch.cuda.max_memory_allocated() / 1e9:.1f}GB", flush=True)
            if args.limit and step >= args.limit:
                break
        val_acc = run_eval(model, val_loader, device)
        history.append({"epoch": epoch, "loss": loss_sum / seen, "train": correct / seen, "val": val_acc})
        print(f"epoch {epoch}  loss {loss_sum / seen:.3f}  train {correct / seen:.3f}  "
              f"val {val_acc:.3f}  {time.time() - start:.0f}s", flush=True)
        torch.save(model.state_dict(), ROOT / "finetuned.pt")
        (ROOT / "history.json").write_text(json.dumps({"args": vars(args), "history": history}, indent=2))


if __name__ == "__main__":
    main()
