"""Decode every downloaded video once into letterboxed 224x224 RGB uint8 frames.

Letterboxing matches web_app/backend/slovo.py (keep aspect, pad with 114),
so training sees exactly what the app feeds the model. Normalization is done
later on the GPU. Output: cache/<attachment_id>.npy, shape [T, 224, 224, 3].
"""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
CACHE = ROOT / "cache"


def letterbox(bgr):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    ratio = min(224 / h, 224 / w)
    width, height = round(w * ratio), round(h * ratio)
    rgb = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_LINEAR)
    dw, dh = (224 - width) / 2, (224 - height) / 2
    return cv2.copyMakeBorder(
        rgb, round(dh - 0.1), round(dh + 0.1),
        round(dw - 0.1), round(dw + 0.1), cv2.BORDER_CONSTANT,
        value=(114, 114, 114),
    )


def convert(path):
    target = CACHE / f"{path.stem}.npy"
    if target.is_file():
        return path.stem, None
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(letterbox(frame))
    cap.release()
    if not frames:
        return path.stem, "unreadable"
    np.save(target, np.stack(frames))
    return path.stem, None


def main():
    CACHE.mkdir(exist_ok=True)
    videos = sorted(DATA.glob("*/*.mp4"))
    errors = []
    with ProcessPoolExecutor(max_workers=8) as pool:
        for done, (stem, error) in enumerate(pool.map(convert, videos, chunksize=4), 1):
            if error:
                errors.append(stem)
            if done % 100 == 0 or done == len(videos):
                print(f"{done}/{len(videos)}", flush=True)
    print(f"errors: {errors}")


if __name__ == "__main__":
    main()
