"""Fetch only the videos of the selected classes from the official Slovo zip.

The archive (~15.9 GB) is read with HTTP range requests: the central
directory once, then each member's bytes in parallel. Every file is
CRC-checked. Output: data/{train,test}/<attachment_id>.mp4 + annotations.csv.
"""
import csv
import json
import struct
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
URL = "https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/slovo/slovo.zip"
WORKERS = 12


def fetch(start, end):
    request = Request(URL, headers={"Range": f"bytes={start}-{end}"})
    for attempt in range(5):
        try:
            with urlopen(request, timeout=120) as response:
                data = response.read()
            if len(data) == end - start + 1:
                return data
        except OSError:
            if attempt == 4:
                raise
    raise IOError(f"Short read for bytes {start}-{end}")


class RemoteFile:
    """Seekable read-only view of the remote zip, enough for zipfile's index."""

    def __init__(self):
        with urlopen(Request(URL, method="HEAD"), timeout=60) as response:
            self.size = int(response.headers["Content-Length"])
        self.pos = 0

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = {0: offset, 1: self.pos + offset, 2: self.size + offset}[whence]
        return self.pos

    def read(self, n=-1):
        end = self.size if n < 0 else min(self.size, self.pos + n)
        if end <= self.pos:
            return b""
        data = fetch(self.pos, end - 1)
        self.pos = end
        return data


def extract(info, target):
    header = fetch(info.header_offset, info.header_offset + 29)
    name_len, extra_len = struct.unpack("<HH", header[26:30])
    start = info.header_offset + 30 + name_len + extra_len
    raw = fetch(start, start + info.compress_size - 1)
    if info.compress_type == zipfile.ZIP_DEFLATED:
        raw = zlib.decompress(raw, -15)
    elif info.compress_type != zipfile.ZIP_STORED:
        raise ValueError(f"Unsupported compression in {info.filename}")
    if zlib.crc32(raw) != info.CRC or len(raw) != info.file_size:
        raise ValueError(f"CRC mismatch: {info.filename}")
    partial = target.with_suffix(".part")
    partial.write_bytes(raw)
    partial.replace(target)
    return len(raw)


def main():
    classes = json.loads((ROOT / "classes.json").read_text(encoding="utf-8"))
    # Annotations call the background class "no_event"; the model labels call it "---".
    wanted = {"no_event" if c["label"] == "---" else c["label"] for c in classes}
    archive = zipfile.ZipFile(RemoteFile())
    members = {Path(info.filename).name: info for info in archive.infolist()}
    annotation = next(info for name, info in members.items() if name == "annotations.csv")
    DATA.mkdir(parents=True, exist_ok=True)
    extract(annotation, DATA / "annotations.csv")
    with (DATA / "annotations.csv").open(encoding="utf-8") as stream:
        rows = [row for row in csv.DictReader(stream, delimiter="\t") if row["text"] in wanted]
    jobs = []
    for row in rows:
        split = "train" if row["train"] == "True" else "test"
        target = DATA / split / f"{row['attachment_id']}.mp4"
        if not target.is_file():
            jobs.append((members[f"{row['attachment_id']}.mp4"], target))
    for split in ("train", "test"):
        (DATA / split).mkdir(exist_ok=True)
    print(f"{len(rows)} videos for {len(wanted)} classes; {len(jobs)} to download")
    done = total = 0
    with ThreadPoolExecutor(WORKERS) as pool:
        futures = [pool.submit(extract, info, target) for info, target in jobs]
        for future in as_completed(futures):
            total += future.result()
            done += 1
            if done % 50 == 0 or done == len(jobs):
                print(f"{done}/{len(jobs)}  {total / 1e6:.0f} MB", flush=True)


if __name__ == "__main__":
    main()
