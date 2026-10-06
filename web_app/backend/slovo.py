"""RGB inference for Slovo MViTv2-small-32-2 ONNX models.

Two models share the same input contract:
- slovo50: fine-tuned on the 50 most frequent words + no_event
  (training/slovo50), the default when its files are present;
- slovo1000: the official 1000 words + no_event.
Select explicitly with SLOVO_MODEL=slovo50|slovo1000.

Preprocessing follows hukenovs/slovo demo.py and config_example.yaml.
Model input is [batch, clips, RGB, time, height, width], already normalized.
The last label is always the no_event class "---".
"""
import json
import os
import logging
from time import perf_counter
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

ASSET_DIR = Path(__file__).resolve().parents[1] / "models" / "slovo"
MODELS = {
    "slovo50": (ASSET_DIR / "slovo50.onnx", ASSET_DIR / "labels50.json"),
    "slovo1000": (ASSET_DIR / "mvit32-2.onnx", ASSET_DIR / "labels.json"),
}


def default_model_name():
    name = os.getenv("SLOVO_MODEL")
    if name:
        if name not in MODELS:
            raise ValueError(f"SLOVO_MODEL must be one of {sorted(MODELS)}")
        return name
    return "slovo50" if all(p.is_file() for p in MODELS["slovo50"]) else "slovo1000"
MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)


def preprocess_frame(bgr):
    """Preserve aspect ratio; pad with 114; convert BGR to normalized RGB."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    ratio = min(224 / h, 224 / w)
    width, height = round(w * ratio), round(h * ratio)
    rgb = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_LINEAR)
    dw, dh = (224 - width) / 2, (224 - height) / 2
    rgb = cv2.copyMakeBorder(
        rgb, round(dh - 0.1), round(dh + 0.1),
        round(dw - 0.1), round(dw + 0.1), cv2.BORDER_CONSTANT,
        value=(114, 114, 114),
    )
    return np.transpose((rgb.astype(np.float32) - MEAN) / STD, (2, 0, 1))


class SlovoRecognizer:
    frame_count = 32
    sample_fps = 15  # 30 FPS source sampled with the official frame interval of 2

    def __init__(self, name=None):
        self.name = name or default_model_name()
        model_path, label_path = MODELS[self.name]
        if not model_path.is_file() or not label_path.is_file():
            raise FileNotFoundError(f"Slovo assets missing for {self.name}. Run: python -m web_app.setup_slovo")
        self.labels = json.loads(label_path.read_text(encoding="utf-8"))
        if len(self.labels) < 2 or self.labels[-1] != "---":
            raise ValueError("Slovo label map must end with the no_event class '---'")
        self.no_event_id = len(self.labels) - 1
        options = ort.SessionOptions()
        options.log_severity_level = 3
        options.intra_op_num_threads = max(1, min(4, os.cpu_count() or 1))
        providers = ["CPUExecutionProvider"]
        if "DmlExecutionProvider" in ort.get_available_providers() and os.getenv("SLOVO_DEVICE", "auto") != "cpu":
            options.enable_mem_pattern = False
            options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            providers.insert(0, ("DmlExecutionProvider", {
                "performance_preference": "high_performance", "device_filter": "gpu",
            }))
        try:
            self.session = ort.InferenceSession(str(model_path), sess_options=options, providers=providers)
        except Exception:
            if len(providers) == 1:
                raise
            logging.getLogger(__name__).warning("DirectML initialization failed; using CPU", exc_info=True)
            self.session = ort.InferenceSession(str(model_path), sess_options=options, providers=["CPUExecutionProvider"])
        self.provider = self.session.get_providers()[0]
        self.input = self.session.get_inputs()[0]
        if self.input.shape != [1, 1, 3, 32, 224, 224]:
            raise ValueError(f"Unexpected Slovo input shape: {self.input.shape}")

    def predict(self, frames):
        if len(frames) != self.frame_count:
            raise ValueError("Slovo needs exactly 32 preprocessed frames")
        tensor = np.ascontiguousarray(np.stack(frames, axis=1)[None, None], dtype=np.float32)
        start = perf_counter()
        output = self.session.run(None, {self.input.name: tensor})[0].reshape(-1)
        inference_ms = round((perf_counter() - start) * 1000)
        if output.size != len(self.labels) or not np.isfinite(output).all():
            raise ValueError("Invalid Slovo model output")
        # Exported graph may contain Softmax; never apply it twice.
        if np.all(output >= 0) and np.isclose(output.sum(), 1, atol=1e-4):
            probs = output
        else:
            probs = np.exp(output - output.max())
            probs /= probs.sum()
        top = np.argsort(probs)[-5:][::-1]
        index = int(top[0])
        return {
            "prediction": None if index == self.no_event_id else self.labels[index],
            "classId": index,
            "confidence": float(probs[index]),
            "noEvent": index == self.no_event_id,
            "inferenceMs": inference_ms,
            "top5": [{"classId": int(i), "label": self.labels[int(i)],
                      "confidence": float(probs[i])} for i in top],
        }

    def predict_video(self, path):
        """Use a centered 32-frame clip, interval 2, repeat last frame if short."""
        cap = cv2.VideoCapture(str(path))
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        start = max(0, (count - 63) // 2)
        frames = []
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start)
            for index in range(63):
                ok, frame = cap.read()
                if not ok:
                    break
                if index % 2 == 0:
                    frames.append(preprocess_frame(frame))
        finally:
            cap.release()
        if not frames:
            raise ValueError(f"Cannot read video: {path}")
        frames.extend([frames[-1]] * (self.frame_count - len(frames)))
        return self.predict(frames)
