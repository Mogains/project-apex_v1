"""
Build a YOLO dataset of your bullseye page without labeling anything by hand.

Run label_video.py first. It gives us:
  - real/images + real/labels   real video frames with the page boxed
  - real/cutouts                the page cut out of those frames

This script then makes:
  - the real frames themselves (the last 20% of each video is held back to test on)
  - thousands of generated images: real cut-out pages pasted at random sizes
    (tiny to big), angles, lighting, blur and noise onto real frames, your
    backgrounds/ photos/videos, or random clutter with decoy circles and
    blank paper. Boxes are known because we placed the pages ourselves.

Run:
  /Users/anmolbhatt/.pyenv/versions/3.10.13/bin/python training/make_dataset.py
"""
import argparse
import glob
import os
import random
import shutil

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REAL = os.path.join(HERE, "real")
OUT = os.path.join(HERE, "dataset")
SIZE = 640
HOLDOUT = 0.2                     # last part of each video, kept for testing only
REAL_REPEAT = 3                   # real frames are worth more than generated ones

MIN_PX, MAX_PX = 14, 600          # page size range in a 640 image (small ones matter most)


# ---------------- inputs ----------------
def load_templates():
    cuts = sorted(glob.glob(os.path.join(REAL, "cutouts", "*.png")))
    if not cuts:
        raise SystemExit("No cut-outs found. Put a video in training/test_videos/ and run label_video.py first.")
    return [cv2.imread(p, cv2.IMREAD_UNCHANGED) for p in cuts]


def load_real():
    """Real labeled frames, split per video into (train, holdout) by time."""
    by_video = {}
    for p in sorted(glob.glob(os.path.join(REAL, "images", "*.jpg"))):
        im = cv2.imread(p)
        H, W = im.shape[:2]
        boxes = []
        for line in open(p.replace("images", "labels").replace(".jpg", ".txt")):
            _, cx, cy, w, h = map(float, line.split())
            boxes.append(((cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H))
        by_video.setdefault(os.path.basename(p).rsplit("_", 1)[0], []).append((os.path.basename(p), im, boxes))
    train, hold = [], []
    for frames in by_video.values():
        cut = int(len(frames) * (1 - HOLDOUT))
        train += frames[:cut]
        hold += frames[cut:]
    return train, hold


def load_backgrounds():
    """Photos / videos of the room without the target, from training/backgrounds/."""
    bgs = []
    for p in glob.glob(os.path.join(HERE, "backgrounds", "*")):
        ext = p.lower().rsplit(".", 1)[-1]
        if ext in ("jpg", "jpeg", "png", "webp"):
            im = cv2.imread(p)
            if im is not None:
                bgs.append(im)
        elif ext in ("mov", "mp4", "m4v", "avi"):
            cap = cv2.VideoCapture(p)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
            for i in np.linspace(0, total - 1, min(total, 200)).astype(int):
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
                ok, f = cap.read()
                if ok:
                    bgs.append(f)
            cap.release()
    return bgs


# ---------------- backgrounds ----------------
def rand_color():
    return tuple(int(v) for v in np.random.randint(0, 256, 3))


def synthetic_background():
    bg = np.zeros((SIZE, SIZE, 3), np.uint8)
    bg[:] = rand_color()
    g = np.linspace(random.uniform(0.5, 1), random.uniform(0.5, 1.3), SIZE, dtype=np.float32)
    bg = np.clip(bg * (g[:, None, None] if random.random() < .5 else g[None, :, None]), 0, 255).astype(np.uint8)
    for _ in range(random.randint(3, 25)):
        kind = random.random()
        c, t = rand_color(), random.randint(1, 6)
        p = (random.randint(0, SIZE), random.randint(0, SIZE))
        if kind < .3:
            cv2.rectangle(bg, p, (random.randint(0, SIZE), random.randint(0, SIZE)), c, -1 if random.random() < .5 else t)
        elif kind < .55:
            cv2.line(bg, p, (random.randint(0, SIZE), random.randint(0, SIZE)), c, t)
        else:
            # decoy circles / ellipses, sometimes in pen-purple, so the model can't just find "a circle"
            if random.random() < .4:
                c = (random.randint(120, 200), random.randint(30, 90), random.randint(60, 130))
            ax = (random.randint(4, 150), random.randint(4, 150))
            cv2.ellipse(bg, p, ax, random.uniform(0, 180), 0, 360, c, -1 if random.random() < .3 else t)
    return bg


def paper_decoy(w, h):
    """A white / lined sheet with a scribble or single circle on it (a hard 'not a target')."""
    tone = random.randint(205, 250)
    paper = np.full((h, w, 4), (tone, tone, tone, 255), np.uint8)
    if random.random() < .7:
        gap = max(3, h // 25)
        for y in range(gap, h, gap):
            cv2.line(paper, (0, y), (w, y), (190, 180, 175, 255), 1)
    ink = (random.randint(120, 200), random.randint(30, 90), random.randint(60, 130), 255)
    t = max(1, w // 120)
    r = random.random()
    if r < .4:
        cv2.circle(paper, (w // 2, h // 2), random.randint(w // 8, w // 3), ink, t)
    elif r < .7:
        pts = np.random.randint(0, min(w, h), (random.randint(3, 8), 1, 2))
        cv2.polylines(paper, [pts], False, ink, t)
    return paper


def crop_real(frame, boxes):
    """Random 640 crop of a real frame, carrying its boxes. None if a page would be cut in half."""
    H, W = frame.shape[:2]
    s = random.uniform(SIZE / min(H, W), 1.8 * SIZE / min(H, W))
    x0, y0 = random.uniform(0, W * s - SIZE), random.uniform(0, H * s - SIZE)
    out = []
    for x1, y1, x2, y2 in boxes:
        b = np.array([x1, y1, x2, y2]) * s - [x0, y0, x0, y0]
        c = np.clip(b, 0, SIZE)
        area = (b[2] - b[0]) * (b[3] - b[1])
        vis = max(0, c[2] - c[0]) * max(0, c[3] - c[1]) / area
        if vis < .999 and vis > 0:
            return None
        if vis > 0:
            out.append(tuple(c))
    M = np.float32([[s, 0, -x0], [0, s, -y0]])
    return cv2.warpAffine(frame, M, (SIZE, SIZE), flags=cv2.INTER_AREA), out


def shrink_real(frame, boxes):
    """Whole real frame shrunk into a corner of a clutter background (page looks far away)."""
    H, W = frame.shape[:2]
    s = random.uniform(.25, 1) * SIZE / W
    w, h = int(W * s), int(H * s)
    x0, y0 = random.randint(0, SIZE - w), random.randint(0, SIZE - h)
    bg = synthetic_background()
    bg[y0:y0 + h, x0:x0 + w] = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
    return bg, [(x1 * s + x0, y1 * s + y0, x2 * s + x0, y2 * s + y0) for x1, y1, x2, y2 in boxes]


def background(real_train, room):
    r = random.random()
    if real_train and r < .35:
        frame, boxes = random.choice(real_train)[1:]
        for _ in range(10):
            got = crop_real(frame, boxes)
            if got:
                return got
    if real_train and r < .45:
        return shrink_real(*random.choice(real_train)[1:])
    if room and r < .75:
        im = random.choice(room)
        h, w = im.shape[:2]
        s = random.uniform(SIZE / min(h, w), 2 * SIZE / min(h, w))
        im = cv2.resize(im, None, fx=s, fy=s)
        h, w = im.shape[:2]
        x, y = random.randint(0, w - SIZE), random.randint(0, h - SIZE)
        return im[y:y + SIZE, x:x + SIZE].copy(), []
    return synthetic_background(), []


# ---------------- pasting ----------------
def paste(canvas, img, size, avoid):
    """Warp a BGRA cut-out onto canvas at a random spot. Returns its box (x1,y1,x2,y2) or None."""
    img = np.rot90(img, random.randint(0, 3)).copy()
    h, w = img.shape[:2]
    scale = size / max(h, w)
    tw, th = w * scale, h * scale
    for _ in range(20):
        cx, cy = random.uniform(tw / 2, SIZE - tw / 2), random.uniform(th / 2, SIZE - th / 2)
        ang = np.radians(random.uniform(-30, 30))
        ca, sa = np.cos(ang), np.sin(ang)
        corners = []
        for dx, dy in ((-tw / 2, -th / 2), (tw / 2, -th / 2), (tw / 2, th / 2), (-tw / 2, th / 2)):
            jx, jy = np.random.uniform(-.12, .12, 2) * (tw, th)   # perspective tilt
            x, y = dx + jx, dy + jy
            corners.append((cx + x * ca - y * sa, cy + x * sa + y * ca))
        dst = np.float32(corners)
        x1, y1 = dst.min(0)
        x2, y2 = dst.max(0)
        if x1 < 0 or y1 < 0 or x2 > SIZE or y2 > SIZE:
            continue
        if any(not (x2 < a1 or x1 > a2 or y2 < b1 or y1 > b2) for a1, b1, a2, b2 in avoid):
            continue
        src = img
        if scale < .5:   # pre-shrink so tiny pages look like a real camera would see them (no aliasing)
            src = cv2.resize(img, (max(2, int(w * scale * 2)), max(2, int(h * scale * 2))), interpolation=cv2.INTER_AREA)
        sh, sw = src.shape[:2]
        M = cv2.getPerspectiveTransform(np.float32([(0, 0), (sw, 0), (sw, sh), (0, sh)]), dst)
        warped = cv2.warpPerspective(src, M, (SIZE, SIZE))
        alpha = warped[..., 3]
        ys, xs = np.nonzero(alpha > 127)
        if not len(xs):
            continue
        # per-page lighting: brightness, tint, and a soft shadow across it
        lit = warped[..., :3].astype(np.float32) * random.uniform(.55, 1.15) + np.random.uniform(-15, 15, 3)
        if random.random() < .4:
            g = np.linspace(random.uniform(.5, 1), random.uniform(.8, 1.2), SIZE, dtype=np.float32)
            lit *= g[None, :, None] if random.random() < .5 else g[:, None, None]
        m = (alpha.astype(np.float32) / 255)[..., None]
        canvas[:] = np.clip(canvas * (1 - m) + lit * m, 0, 255).astype(np.uint8)
        return (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)
    return None


def finish(im):
    """Camera-ish degradation over the whole frame."""
    im = im.astype(np.float32)
    im = im * random.uniform(.7, 1.3) + random.uniform(-30, 30)
    if random.random() < .5:
        im += np.random.normal(0, random.uniform(2, 10), im.shape)
    im = np.clip(im, 0, 255).astype(np.uint8)
    r = random.random()
    if r < .3:
        im = cv2.GaussianBlur(im, (0, 0), random.uniform(.5, 1.8))
    elif r < .45:
        k = random.randint(3, 9)
        kern = np.zeros((k, k), np.float32)
        kern[k // 2, :] = 1 / k
        M = cv2.getRotationMatrix2D((k / 2, k / 2), random.uniform(0, 180), 1)
        im = cv2.filter2D(im, -1, cv2.warpAffine(kern, M, (k, k)))
    ok, buf = cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, random.randint(40, 95)])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def make_image(templates, real_train, room):
    canvas, boxes = background(real_train, room)
    boxes = list(boxes)
    # decoys first, so real pages can land beside them
    for _ in range(random.choice([0, 0, 1, 2])):
        s = int(np.exp(random.uniform(np.log(MIN_PX * 2), np.log(MAX_PX * .7))))
        paste(canvas, paper_decoy(s, int(s * random.uniform(.9, 1.3))), s, boxes)
    r = random.random()
    n = 0 if r < .15 else 1 if r < .8 else random.randint(2, 3)
    for _ in range(n):
        s = np.exp(random.uniform(np.log(MIN_PX), np.log(MAX_PX)))   # log-uniform: lots of small ones
        b = paste(canvas, random.choice(templates), s, boxes)
        if b:
            boxes.append(b)
    return finish(canvas), boxes


def write(split, stem, im, boxes):
    H, W = im.shape[:2]
    cv2.imwrite(f"{OUT}/images/{split}/{stem}.jpg", im)
    with open(f"{OUT}/labels/{split}/{stem}.txt", "w") as f:
        for x1, y1, x2, y2 in boxes:
            f.write(f"0 {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} {(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=3000, help="generated training images")
    ap.add_argument("--val", type=int, default=200, help="generated test images (on top of the held-out real frames)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    real_train, real_hold = load_real()
    # cut-outs only from the training part, so the held-out frames stay a fair test
    train_names = {n for n, _, _ in real_train}
    templates = [t for p, t in zip(sorted(glob.glob(os.path.join(REAL, "cutouts", "*.png"))), load_templates())
                 if os.path.basename(p).replace(".png", ".jpg") in train_names]
    room = load_backgrounds()
    print(f"real frames: {len(real_train)} train, {len(real_hold)} held out | page cut-outs: {len(templates)} | room backgrounds: {len(room)}")

    shutil.rmtree(OUT, ignore_errors=True)
    for split in ("train", "val"):
        os.makedirs(f"{OUT}/images/{split}")
        os.makedirs(f"{OUT}/labels/{split}")

    for name, im, boxes in real_train:
        for k in range(REAL_REPEAT):
            write("train", f"real_{k}_{name[:-4]}", im, boxes)
    for name, im, boxes in real_hold:
        write("val", f"real_{name[:-4]}", im, boxes)

    for split, count in (("train", args.train), ("val", args.val)):
        pool = real_train if split == "train" else []   # never leak held-out frames into training
        for i in range(count):
            im, boxes = make_image(templates, pool, room)
            write(split, f"gen_{i:05d}", im, boxes)
            if (i + 1) % 500 == 0:
                print(f"{split}: {i + 1}/{count}")

    with open(f"{OUT}/data.yaml", "w") as f:
        f.write(f"path: {OUT}\ntrain: images/train\nval: images/val\nnames:\n  0: bullseye\n")
    print(f"done -> {OUT}/data.yaml")


if __name__ == "__main__":
    main()
