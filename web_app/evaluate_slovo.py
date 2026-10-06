"""Evaluate only official train=False clips; never fit or tune the model here."""
import argparse
import csv
import json
import random
import urllib.request
from collections import defaultdict
from pathlib import Path
from time import perf_counter

from web_app.backend.slovo import SlovoRecognizer


def test_rows(annotation):
    with Path(annotation).open(encoding="utf-8-sig", newline="") as source:
        delimiter = "\t" if "\t" in source.readline() else ","
        source.seek(0)
        rows = list(csv.DictReader(source, delimiter=delimiter))
    if not rows or not {"attachment_id", "train", "text", "user_id"}.issubset(rows[0]):
        raise ValueError("Expected official Slovo annotations (attachment_id, train, text, user_id)")
    if any(row["train"].lower() not in {"true", "false"} for row in rows):
        raise ValueError("Unrecognized train/test flags")
    # Keep the published split; do not reshuffle samples or augment the test set.
    return [row for row in rows if row["train"].lower() == "false"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--videos", type=Path, required=True, help="Unpacked TRIMMED Slovo root or test folder")
    sampling = parser.add_mutually_exclusive_group()
    sampling.add_argument("--limit", type=int, help="Explicitly run only the first N test rows")
    sampling.add_argument("--sample-size", type=int, help="Uniform random test subset, without replacement")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--endpoint", help="Use the running application's /api/infer-video instead of a new model session")
    parser.add_argument("--output", type=Path, default=Path("slovo-test-results.json"))
    args = parser.parse_args()
    rows = test_rows(args.annotations)
    if not rows:
        parser.error("No official test samples found")
    total_test = len(rows)
    if args.limit is not None:
        if args.limit < 1:
            parser.error("--limit must be positive")
        rows = rows[:args.limit]
    if args.sample_size is not None:
        if not 1 <= args.sample_size <= total_test:
            parser.error("--sample-size must be between 1 and the official test count")
        rows = random.Random(args.seed).sample(sorted(rows, key=lambda row: row["attachment_id"]), args.sample_size)
        rows.sort(key=lambda row: row["attachment_id"])
    recognizer = None if args.endpoint else SlovoRecognizer()
    if args.endpoint:
        with urllib.request.urlopen(args.endpoint.rstrip("/") + "/api/recognition-signs", timeout=30) as response:
            labels = json.load(response)["signs"] + ["---"]
    else:
        labels = recognizer.labels
    classes = {label: index for index, label in enumerate(labels)}
    classes["no_event"] = len(labels) - 1
    # slovo50 knows only 50 words: clips of other words cannot be scored and are counted separately.
    out_of_vocabulary = sum(row["text"].strip() not in classes for row in rows)
    rows = [row for row in rows if row["text"].strip() in classes]
    counts, correct = defaultdict(int), defaultdict(int)
    records = []
    for row in rows:
        name = row["attachment_id"] + ".mp4"
        candidates = [args.videos / "test" / name, args.videos / "slovo" / "test" / name, args.videos / name]
        path = next((candidate for candidate in candidates if candidate.is_file()), None)
        if path is None:
            raise FileNotFoundError(f"Test sample missing: {name}; no missing clips are silently skipped")
        label = row["text"].strip()
        if label not in classes:
            raise ValueError(f"Unknown official label: {label}")
        expected = classes[label]
        start = perf_counter()
        if args.endpoint:
            boundary = "slovo-evaluation-clip"
            payload = (f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="clip.mp4"\r\n'
                       'Content-Type: video/mp4\r\n\r\n').encode() + path.read_bytes() + f'\r\n--{boundary}--\r\n'.encode()
            request = urllib.request.Request(args.endpoint.rstrip("/") + "/api/infer-video", data=payload,
                                             headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
            with urllib.request.urlopen(request, timeout=120) as response:
                prediction = json.load(response)
        else:
            prediction = recognizer.predict_video(path)
        hit = prediction["classId"] == expected
        counts[expected] += 1
        correct[expected] += int(hit)
        records.append({"attachment_id": row["attachment_id"], "expected": label,
                        "correct": hit, "top5Correct": expected in [p["classId"] for p in prediction["top5"]],
                        "elapsedMs": round((perf_counter() - start) * 1000),
                        **prediction})
        print(f"{len(records)}/{len(rows)}: {label} -> {prediction['prediction'] or 'no_event'}", flush=True)
    result = {
        "split": "official_test", "evaluated": len(records), "officialTestCount": total_test,
        "isSubset": len(records) != total_test, "classCount": len(labels),
        "outOfVocabularySkipped": out_of_vocabulary,
        "sampling": "uniform_random_without_replacement" if args.sample_size is not None else "official_row_order",
        "seed": args.seed if args.sample_size is not None else None,
        "inferencePath": args.endpoint or "direct_model",
        "top1Accuracy": sum(r["correct"] for r in records) / len(records),
        "top5Accuracy": sum(r["top5Correct"] for r in records) / len(records),
        "meanAccuracyObservedClasses": sum(correct[i] / counts[i] for i in counts) / len(counts),
        "observedClassCount": len(counts), "records": records,
    }
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "records"}, indent=2))


if __name__ == "__main__":
    main()
