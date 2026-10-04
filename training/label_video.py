"""
Auto-label real video of the bullseye page (no hand labeling).

For each frame: find the purple center dot, then the bright white page that
contains it, and box the page. Frames where the page outline isn't a clean
four-sided shape (page blending into a hand / table) are skipped.

Writes labeled frames to training/real/{images,labels}, cut-out pages
(with transparent background) to training/real/cutouts for make_dataset.py,
and a preview sheet training/real/preview.jpg so you can eyeball the boxes.

Run:
  /Users/anmolbhatt/.pyenv/versions/3.10.13/bin/python training/label_video.py
"""
import glob
import os
import shutil

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "real")
EVERY = 3          # use every Nth frame
WORK = 960         # analyse at this width (faster, and kernels stay a sane size)


def find_dot(hsv):
    ink = ((hsv[..., 0] > 105) & (hsv[..., 0] < 165) & (hsv[..., 1] > 55) & (hsv[..., 2] > 40)).astype(np.uint8)
    n, lab, st, cen = cv2.connectedComponentsWithStats(ink)
    best, score = None, 0
    for k in range(1, n):
        x, y, w, h, a = st[k]
        fill = a / max(1, w * h)
        # a solid, roughly round blob = the center dot (rings are thin and hollow)
        if a >= 12 and fill > 0.5 and 0.5 < w / max(1, h) < 2 and a > score:
            best, score = cen[k], a
    return best


def find_page(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    dot = find_dot(hsv)
    if dot is None:
        return None
    v, s = hsv[..., 2].astype(np.int16), hsv[..., 1]
    # paper = the brightest, greyest stuff; threshold relative to this frame's paper brightness
    dx, dy = int(dot[0]), int(dot[1])
    ref = np.percentile(v[max(0, dy - 40):dy + 40, max(0, dx - 40):dx + 40], 75)
    paper = ((v > ref - 35) & (s < 45)).astype(np.uint8) * 255
    paper = cv2.morphologyEx(paper, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))   # bridge the ink lines
    n, lab, st, _ = cv2.connectedComponentsWithStats(paper)
    # the component containing the dot (look in a small window, the dot itself is ink)
    win = lab[max(0, dy - 25):dy + 25, max(0, dx - 25):dx + 25]
    ids, counts = np.unique(win[win > 0], return_counts=True)
    if not len(ids):
        return None
    comp = (lab == ids[np.argmax(counts)]).astype(np.uint8) * 255
    cs, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(cs, key=cv2.contourArea)
    area = cv2.contourArea(c)
    hull = cv2.convexHull(c)
    (_, _), (rw, rh), _ = cv2.minAreaRect(c)
    solidity = area / max(1, cv2.contourArea(hull))
    rect_fill = area / max(1, rw * rh)
    aspect = min(rw, rh) / max(1, max(rw, rh))
    # a clean page: fills its rotated rectangle, no big bites or blobs stuck on
    if solidity < 0.93 or rect_fill < 0.85 or aspect < 0.55 or area < 1500:
        return None
    x, y, w, h = cv2.boundingRect(c)
    # page must contain the dot reasonably centrally (rules out "page" = sock / wall)
    if not (x + 0.2 * w < dx < x + 0.8 * w and y + 0.2 * h < dy < y + 0.8 * h):
        return None
    mask = np.zeros_like(comp)
    cv2.drawContours(mask, [hull], -1, 255, -1)   # fill the ink holes
    return (x, y, x + w, y + h), mask


def main():
    shutil.rmtree(OUT, ignore_errors=True)
    os.makedirs(f"{OUT}/images")
    os.makedirs(f"{OUT}/labels")
    os.makedirs(f"{OUT}/cutouts")
    kept, seen, tiles = 0, 0, []
    for video in sorted(glob.glob(os.path.join(HERE, "test_videos", "*"))):
        name = os.path.splitext(os.path.basename(video))[0]
        cap = cv2.VideoCapture(video)
        i = 0
        while True:
            ok, f = cap.read()
            if not ok:
                break
            if i % EVERY == 0:
                seen += 1
                s = WORK / f.shape[1]
                f = cv2.resize(f, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
                found = find_page(f)
                if found:
                    (x1, y1, x2, y2), mask = found
                    H, W = f.shape[:2]
                    stem = f"{name}_{i:05d}"
                    cv2.imwrite(f"{OUT}/images/{stem}.jpg", f)
                    cut = cv2.cvtColor(f[y1:y2, x1:x2], cv2.COLOR_BGR2BGRA)
                    cut[..., 3] = mask[y1:y2, x1:x2]
                    cv2.imwrite(f"{OUT}/cutouts/{stem}.png", cut)
                    with open(f"{OUT}/labels/{stem}.txt", "w") as fh:
                        fh.write(f"0 {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} {(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}\n")
                    kept += 1
                    if kept % 8 == 1 and len(tiles) < 24:
                        p = f.copy()
                        cv2.rectangle(p, (x1, y1), (x2, y2), (0, 255, 0), 4)
                        tiles.append(cv2.resize(p, (320, int(320 * H / W))))
            i += 1
        cap.release()
    print(f"labeled {kept} of {seen} frames -> {OUT}")
    if tiles:
        h = max(t.shape[0] for t in tiles)
        tiles = [cv2.copyMakeBorder(t, 0, h - t.shape[0], 0, 0, cv2.BORDER_CONSTANT) for t in tiles]
        while len(tiles) % 4:
            tiles.append(np.zeros_like(tiles[0]))
        cv2.imwrite(f"{OUT}/preview.jpg", np.vstack([np.hstack(tiles[k:k + 4]) for k in range(0, len(tiles), 4)]))


if __name__ == "__main__":
    main()
