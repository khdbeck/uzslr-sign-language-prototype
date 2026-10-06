"""One example video per word for the dictionary page (/slovo.html).

For each of the 50 words the highest-resolution *train* video is used (test
videos stay unseen). The original H.264 MP4 is copied as is (browsers play it)
and its middle frame - mid-gesture - is saved as the JPEG poster.
Output: web_app/frontend/slovo-images/slovo50/<class index>.{mp4,jpg}, where
the index matches labels50.json. no_event gets no example.
"""
import shutil

import cv2

from common import DATA, ROOT, load_rows

OUT = ROOT.parents[1] / "web_app" / "frontend" / "slovo-images" / "slovo50"
WIDTH = 270


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    best = {}
    for row in load_rows():
        if row["train"] != "True" or row["text"] == "no_event":
            continue
        pixels = int(row["height"]) * int(row["width"])
        if row["cls"] not in best or pixels > best[row["cls"]][0]:
            best[row["cls"]] = (pixels, row)
    for cls, (_, row) in sorted(best.items()):
        video = DATA / "train" / f"{row['attachment_id']}.mp4"
        shutil.copyfile(video, OUT / f"{cls}.mp4")
        cap = cv2.VideoCapture(str(video))
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) // 2)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            raise ValueError(f"Cannot read {row['attachment_id']}")
        h, w = frame.shape[:2]
        frame = cv2.resize(frame, (WIDTH, round(h * WIDTH / w)), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(OUT / f"{cls}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
    print(f"{len(best)} videos + posters -> {OUT}")


if __name__ == "__main__":
    main()
