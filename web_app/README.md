# Isora-AI — Slovo recognition

The web app recognizes Russian Sign Language with **Slovo MViTv2-small-32-2** RGB models. By default it uses **slovo50**: the official model fine-tuned on the **50 most frequent words** plus `no_event` (training code, word list and test results: [training/slovo50](../training/slovo50/README.md)). If `web_app/models/slovo/slovo50.onnx` is missing, or with `SLOVO_MODEL=slovo1000`, it uses the official model with **1,000 signs** plus `no_event`. Russian labels are shown directly; Uzbek recognition accuracy is unmeasured. The former 50-sign recognition backend and weights were removed. Existing Uzbek reference videos and reply generation remain available separately.

## Local startup

Python 3.12 is recommended. From uzslr-isolated-dynamic:

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r web_app/requirements-local.txt
.venv\Scripts\python.exe -m web_app.setup_slovo
.venv\Scripts\python.exe -m uvicorn web_app.backend.main:app --host 127.0.0.1 --port 8000
```

Open http://localhost:8000. On this workstation the environment and weights are installed; ./start-web.ps1 starts the server again.

Windows uses ONNX Runtime DirectML with a high-performance GPU preference; other platforms use CPU. Set SLOVO_DEVICE=cpu to force CPU. Do not install different ONNX Runtime distributions together in one environment.

Setup downloads ~141 MB, verifies SHA-256, and reads ordered class labels from a pinned upstream commit without executing downloaded Python. Weights are ignored by Git. Setup does not download the ~16 GB dataset.

## Recognition

- Webcam: keep face, upper body and hands visible; perform one sign in about two seconds. Single-handed signs work. Original unmirrored JPEGs are sent to the local backend at 15 FPS (30 FPS source with frame interval 2).
- Each prediction uses 32 RGB frames with official aspect-preserving 224px padding and normalization. A fresh clip starts after each prediction to avoid temporal gaps on slow CPUs.
- Video test: select MP4/WebM/MOV/AVI under the camera, up to 50 MB. Camera inference pauses. Use “Kameraga qaytish” to resume. Uploaded temporary files are deleted after inference.
- Results show the most likely sign at confidence of 10% or higher; below 10% the result is uncertain. Five candidates remain visible. Confidence is not test accuracy. The no_event class is displayed as “Ishora yo‘q” and never enters word history.
- /slovo.html lists the labels of the active model (50 or 1,000). /signs.html is the separate gallery of existing Uzbek reply/reference videos.

## Official test evaluation

Download the **trimmed** dataset from https://github.com/hukenovs/slovo. Keep official annotations.csv and the train=False split. Run from this repository:

```powershell
.venv\Scripts\python.exe -m web_app.evaluate_slovo --annotations C:\datasets\slovo\annotations.csv --videos C:\datasets\slovo --output slovo-test-results.json
```

The command evaluates only test videos, does not train/tune, and fails on missing samples. --limit 10 explicitly runs the first ten rows. --sample-size 500 --seed 42 selects a reproducible uniform random subset without replacement. Use --endpoint http://127.0.0.1:8000 to test the running application's video endpoint with exactly the same preprocessing as uploads. Reports contain per-video Top-1, Top-5, mean accuracy over represented classes, class coverage, timing and sampling metadata. Subset results must not be presented as full test accuracy.

Upstream reports **64.09% mean accuracy**. This app has not reproduced it on the complete 5,100-video test set. Integration tests exercise actual ONNX inference, binary JPEG WebSocket transport and video uploads, not benchmark accuracy.

On 2026-10-06 the running application was evaluated on a fixed random sample of **500 official test videos** (seed 42, 399 represented classes). Measured Top-1: **324/500 = 64.8%**; Top-5: **449/500 = 89.8%**; mean accuracy over observed classes: **64.35%**. On 178 clips from users entirely absent in train, Top-1 was **61.24%**. This remains a subset evaluation. The reports, predictions, annotations and downloaded test clips are in the parent workspace's tmp/slovo-evaluation directory, with methodology in REPORT.md.

The existing Uzbek reference clips performed much worse: strict comparison against predeclared Russian semantic equivalents gave **3/21 = 14.3%** Top-1 and **7/21 = 33.3%** Top-5 on gestures with a matching class. Another 28 of the 49 gesture names have no corresponding class in Slovo; the separate rest clip was recognized as no_event. This local diagnostic is not a labelled RSL benchmark and sign variants may differ. Twenty test clips sent through browser-equivalent 640px JPEG WebSocket transport matched the video endpoint's first prediction in all twenty cases.

## Docker

Run python -m web_app.setup_slovo before creating an image. The existing web_app/Dockerfile and Dockerfile.llm include Slovo assets and run CPU inference in Linux; the LLM image retains its optional Ollama integration.

## Sources

- Model, labels, preprocessing and downloads: https://github.com/hukenovs/slovo
- Dataset paper: https://arxiv.org/abs/2305.14527
- Upstream license: https://github.com/hukenovs/slovo/tree/main/license
