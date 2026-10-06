"""Download the official model and ordered labels, without downloading Slovo videos."""
import ast
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

ASSET_DIR = Path(__file__).resolve().parent / "models" / "slovo"
MODEL_URL = "https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/slovo/models/mvit/onnx/mvit32-2.onnx"
SOURCE_COMMIT = "14289a822a51faaecf65b5f228264b05db3e380e"
LABEL_URL = f"https://raw.githubusercontent.com/hukenovs/slovo/{SOURCE_COMMIT}/constants.py"
MODEL_SHA256 = "3414b1cca3619ba9a92994c261505d14361657fcc7a66f5a13696de526bc4975"


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    target = ASSET_DIR / "mvit32-2.onnx"
    if not target.is_file() or sha256(target) != MODEL_SHA256:
        partial = target.with_suffix(".onnx.part")
        print("Downloading Slovo MViTv2-small-32-2 (~141 MB)...")
        try:
            with urlopen(MODEL_URL, timeout=60) as response, partial.open("wb") as output:
                while block := response.read(1024 * 1024):
                    output.write(block)
            if sha256(partial) != MODEL_SHA256:
                raise ValueError("Model checksum mismatch; refusing to install")
            partial.replace(target)
        finally:
            partial.unlink(missing_ok=True)
    with urlopen(LABEL_URL, timeout=30) as response:
        source = response.read().decode("utf-8")
    # Read a literal dictionary, never execute downloaded Python.
    classes = None
    for statement in ast.parse(source).body:
        if isinstance(statement, ast.Assign) and any(
            isinstance(name, ast.Name) and name.id == "classes" for name in statement.targets
        ):
            classes = ast.literal_eval(statement.value)
    if not isinstance(classes, dict) or set(classes) != set(range(1001)) or classes[1000] != "---":
        raise ValueError("Invalid upstream label map")
    (ASSET_DIR / "labels.json").write_text(
        json.dumps([classes[i] for i in range(1001)], ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (ASSET_DIR / "manifest.json").write_text(json.dumps({
        "model": "MViTv2-small-32-2", "modelUrl": MODEL_URL,
        "sha256": MODEL_SHA256, "labelUrl": LABEL_URL, "classes": 1001,
        "reportedTestMeanAccuracy": 64.09,
        "source": "https://github.com/hukenovs/slovo",
    }, indent=2), encoding="utf-8")
    print(f"Ready: {target}; 1000 signs + no_event")


if __name__ == "__main__":
    main()
