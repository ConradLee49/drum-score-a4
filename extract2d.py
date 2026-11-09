"""Extract the drum staff from a video whose view also scrolls vertically.

The percussion clef is used as a landmark, so every stable segment contributes
the same staff even when the page sits at a different vertical offset.
"""
import json
import sys

import cv2
import numpy as np

from stitch import find_segments, profile, best_shift, STATIC_MAE  # noqa: E402

CLEF_TPL = None


def load_clef(video, t=100.0):
    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
    ok, fr = cap.read()
    cap.release()
    g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
    return g[884:942, 2:20].astype(np.float32)


def clef_match(gray):
    r = cv2.matchTemplate(gray.astype(np.float32), CLEF_TPL, cv2.TM_CCOEFF_NORMED)
    _, mx, _, loc = cv2.minMaxLoc(r)
    return loc[1], float(mx)


def main():
    global CLEF_TPL
    video = sys.argv[1]
    out = sys.argv[2]
    pad_top, pad_bot = 39, 97          # band around the clef anchor
    fixed_y = None
    skip_before = 0.0
    for i, a in enumerate(sys.argv):
        if a == "--fixed-y":
            fixed_y = int(sys.argv[i + 1])
        if a == "--skip-before":
            skip_before = float(sys.argv[i + 1])
    CLEF_TPL = load_clef(video)

    segs, fps, total = find_segments(video, 96, 1000)
    kept = [s for s in segs if s["dur"] >= 1.2 and s["t0"] >= skip_before]
    print(f"segments {len(segs)} kept {len(kept)}")

    cap = cv2.VideoCapture(video)
    rows, last_y = [], None
    for s in kept:
        frame_ids = np.linspace(s["start"], s["end"], min(7, len(s["frames"]))).astype(int)
        ys, imgs = [], []
        for f in frame_ids:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(f))
            ok, fr = cap.read()
            if not ok:
                continue
            g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
            if fixed_y is None:
                y, sc = clef_match(g)
                if sc >= 0.82:
                    ys.append(y)
                top = (y if sc >= 0.82 else (ys[-1] if ys else last_y))
            else:
                top = fixed_y
            if top is None:
                continue
            top = int(top)
            band = g[max(0, top - pad_top): top + pad_bot]
            if band.shape[0] == pad_top + pad_bot:
                imgs.append(band)
        if not imgs:
            print(f"  skip t={s['t0']:.1f} (no clef)")
            rows.append(None)
            continue
        y = int(np.median(ys)) if ys else last_y
        if fixed_y is not None:
            y = fixed_y
        last_y = y
        stack = np.stack(imgs).astype(np.float32)
        img = np.median(stack, axis=0).astype(np.uint8)
        rows.append({"t0": s["t0"], "t1": s["t1"], "y": y, "img": img, "views": len(imgs)})
        print(f"  t={s['t0']:7.2f} y={y:4d} views={len(imgs)} clefs={len(ys)}")
    cap.release()

    good = [r for r in rows if r]

    def staff_top(img):
        bright = (img > 55)
        frac = bright.mean(axis=1)
        cand = [y for y in range(len(frac)) if frac[y] > 0.5]
        groups = []
        for y in cand:
            if groups and y - groups[-1][-1] <= 2:
                groups[-1].append(y)
            else:
                groups.append([y])
        if len(groups) < 5:
            return None
        return int(round(np.mean(groups[0])))

    tops = [staff_top(r["img"]) for r in good]
    valid = [t for t in tops if t is not None]
    target = int(round(np.median(valid))) if valid else None
    print(f"staff tops: median={target} spread={min(valid) if valid else '-'}.."
          f"{max(valid) if valid else '-'}")
    if target is not None:
        for r, t in zip(good, tops):
            if t is None:
                continue
            dy = target - t
            if dy:
                r["img"] = np.roll(r["img"], dy, axis=0)
                if dy > 0:
                    r["img"][:dy] = r["img"][dy]
                else:
                    r["img"][dy:] = r["img"][dy - 1]
                r["dy"] = int(dy)

    images = [r["img"] for r in good]
    h, w = images[0].shape
    offsets = [0]
    links = []
    for i in range(1, len(images)):
        dx, corr = best_shift(profile(images[i - 1]), profile(images[i]), w - 200)
        if dx >= 0:
            a, b = images[i - 1][:, dx:], images[i][:, : w - dx]
        else:
            a, b = images[i - 1][:, : w + dx], images[i][:, -dx:]
        mae = float(np.abs(a.astype(np.float32) - b.astype(np.float32)).mean()) if a.size else 999
        ok = corr > 0.55 and mae < 14
        links.append({"i": i, "t": good[i]["t0"], "dx": int(dx), "corr": round(corr, 3),
                      "mae": round(mae, 2), "ok": ok})
        offsets.append(offsets[-1] + (int(dx) if ok else 0))
        print(f"  link {i:3d} t={good[i]['t0']:7.2f} dx={int(dx):5d} corr={corr:5.3f} "
              f"mae={mae:6.2f} {'ok' if ok else 'REJECT'}")

    canvas = np.full((h, offsets[-1] + w), 255, np.uint8)
    end = 0
    for img, off in zip(images, offsets):
        if end == 0:
            canvas[:, :w] = img
            end = w
        elif off + w > end:
            canvas[:, end:off + w] = img[:, max(0, end - off):]
            end = off + w
    cv2.imwrite(out, canvas)
    json.dump({"strip": out, "width": int(canvas.shape[1]), "rows": [
        {k: r[k] for k in ("t0", "t1", "y", "views")} for r in good], "links": links},
        open(out.replace(".png", ".json"), "w"), indent=1)
    print(f"strip {canvas.shape[1]}x{h} -> {out}")


if __name__ == "__main__":
    main()
