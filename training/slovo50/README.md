# Slovo 50-word fine-tune

Fine-tunes the official Slovo **MViTv2-small-32-2** on the 50 most frequent Russian words in the Slovo vocabulary plus `no_event`, and installs it into the web app as `slovo50`.

## Words

`select_words.py` ranks Slovo labels by `wordfreq` Zipf frequency and keeps content words only (nouns, verbs, adjectives, adverbs, numerals, interjections, plus да/нет). Prepositions, conjunctions, particles, pronouns, single letters and multi-word phrases are dropped; labels sharing a lemma are deduplicated. Result: `classes.json`.

## Pipeline

Run from this folder with the training environment (`..\..\.venv-train`, Python 3.12, PyTorch CUDA, onnx, onnxruntime, opencv, wordfreq, pymorphy3):

| Step | Script | Output |
|---|---|---|
| 1 | `select_words.py` | `classes.json` |
| 2 | `download_subset.py` | `data/{train,test}/*.mp4` — 1,400 clips (~1 GB) fetched from the official 15.9 GB zip by HTTP range requests, CRC-checked |
| 3 | `cache_frames.py` | `cache/*.npy` — letterboxed 224×224 frames (~9 GB) |
| 4 | `train.py --epochs 10 --frozen 4` | `finetuned.pt`, `history.json` |
| 5 | `evaluate.py`, `evaluate_epochs.py` | `test-results.json`, `epochs-test.json` |
| 6 | `export_onnx.py` | `web_app/models/slovo/slovo50.onnx`, `labels50.json`, `manifest50.json` |
| 7 | `make_examples.py` | `web_app/frontend/slovo-images/slovo50/<i>.{mp4,jpg}` — one train clip per word for `/slovo.html` |

Needs the official PyTorch weights at `web_app/models/slovo/pt/mvit32-2.pt` (https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/slovo/models/mvit/pt/mvit32-2.pt).

## Training

- Start: official weights; new 51-way head initialized from the official classifier rows.
- Patch embedding and blocks 0–3 frozen; blocks 4–15, norm and head trained (31.5M params). fp32 (the traced graph breaks under autocast), batch 2 × 4 accumulation, AdamW, one-cycle LR 3e-5 / head 1e-3, label smoothing 0.1, class-balanced sampling.
- Augmentation: random temporal window with stride 1–3, per-clip affine (±6°, scale 0.88–1.12, shift ±8%), brightness/contrast.
- ~4.5 min/epoch on an RTX 4060 Laptop (4.4 GB VRAM).

## Results — official test split

350 clips (250 words + 100 no_event) from 90 signers, app clip protocol (centred 63 frames, stride 2):

| Model | Top-1 | Words only | no_event |
|---|---|---|---|
| Official, argmax over 1,000 words (previous app) | 70.3% | 61.6% | 92% |
| Official, restricted to the 51 classes | 93.7% | 92.4% | 97% |
| Fine-tuned, epoch 10 (shipped) | **94.6%** | **92.8%** | **99%** |

Most of the gain comes from restricting the vocabulary. Fine-tuning adds about 1 point Top-1 (3 clips of 350, within noise) and mostly improves no_event. Test accuracy and training loss both plateau from epoch 8, so longer training is not expected to help; more signers per word would.

The validation split (2 held-out train clips per word) is not informative: the official model was trained on those clips. The test split was never used for training; per-epoch test numbers are for monitoring only, and the shipped model is the last epoch.
