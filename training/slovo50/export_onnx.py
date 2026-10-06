"""Export finetuned.pt to the app's ONNX contract and install it.

The fine-tuned weights are copied back into the official TorchScript model
(same parameter names). Classifier rows outside our 51 classes get bias -1e4,
so their softmax probability is 0; a Gather node then keeps only our 51
outputs. Tracing our own wrapper is not possible because the tracer rejects
TorchScript submodules taken from another model.

Writes web_app/models/slovo/slovo50.onnx (+ labels50.json, manifest50.json).
Input [1, 1, 3, 32, 224, 224] normalized RGB, output [1, 51] softmax with
"---" (no_event) last.
"""
import hashlib
import json

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnx import helper, numpy_helper

from common import OFFICIAL_PT, ROOT, ExportWrapper, SlovoMViT, load_classes

ASSET_DIR = ROOT.parents[1] / "web_app" / "models" / "slovo"


def main():
    classes = load_classes()
    ids = [c["slovoId"] for c in classes]
    tuned = torch.load(ROOT / "finetuned.pt", map_location="cpu")
    official = torch.jit.load(str(OFFICIAL_PT), map_location="cpu").eval()
    state = official.state_dict()
    for key, value in tuned.items():
        if not key.startswith("head."):
            state[f"backbone.{key}"] = value
    weight = torch.zeros_like(state["cls_head.fc_cls.weight"])
    bias = torch.full_like(state["cls_head.fc_cls.bias"], -1e4)
    weight[ids], bias[ids] = tuned["head.weight"], tuned["head.bias"]
    state["cls_head.fc_cls.weight"], state["cls_head.fc_cls.bias"] = weight, bias
    official.load_state_dict(state)

    example = torch.randn(1, 1, 3, 32, 224, 224)
    target = ASSET_DIR / "slovo50.onnx"
    partial = target.with_suffix(".onnx.part")
    with torch.no_grad():
        torch.onnx.export(official, (example,), str(partial), input_names=["input"],
                          output_names=["probs1001"], opset_version=17, dynamo=False)
    graph = onnx.load(str(partial))
    graph.graph.initializer.append(numpy_helper.from_array(np.array(ids, dtype=np.int64), "class_ids"))
    graph.graph.node.append(helper.make_node("Gather", ["probs1001", "class_ids"], ["probs"], axis=1))
    del graph.graph.output[:]
    graph.graph.output.append(helper.make_tensor_value_info("probs", onnx.TensorProto.FLOAT, [1, len(ids)]))
    onnx.checker.check_model(graph)
    onnx.save(graph, str(partial))

    # Reference: the PyTorch fine-tuned model with the app's output contract.
    model = SlovoMViT(ids)
    model.load_state_dict(tuned)
    with torch.no_grad():
        expected = ExportWrapper(model.eval())(example).numpy()
    session = ort.InferenceSession(str(partial), providers=["CPUExecutionProvider"])
    actual = session.run(None, {"input": example.numpy()})[0]
    diff = float(np.abs(actual - expected).max())
    assert actual.shape == (1, len(ids)) and diff < 1e-4, (actual.shape, diff)
    assert list(session.get_inputs()[0].shape) == [1, 1, 3, 32, 224, 224]
    partial.replace(target)
    print(f"ONNX matches PyTorch (max diff {diff:.2e}, prob sum {actual.sum():.6f})")

    labels = [c["label"] for c in classes]
    (ASSET_DIR / "labels50.json").write_text(json.dumps(labels, ensure_ascii=False, indent=2), encoding="utf-8")
    results = json.loads((ROOT / "test-results.json").read_text(encoding="utf-8"))
    (ASSET_DIR / "manifest50.json").write_text(json.dumps({
        "model": "MViTv2-small-32-2 fine-tuned on 50 frequent Slovo words + no_event",
        "base": "https://github.com/hukenovs/slovo (mvit32-2.pt)",
        "training": "training/slovo50",
        "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        "classes": len(labels),
        "testClips": results["testClips"],
        "officialTest": {k: {m: round(results[k][m], 4) for m in ("top1", "top5", "wordTop1", "noEventTop1")}
                         for k in ("official-1000", "official-restricted", "finetuned")},
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Installed {target} ({target.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
