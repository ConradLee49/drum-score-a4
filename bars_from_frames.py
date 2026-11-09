"""Locate bar lines using every staff system in a source frame.

In a Guitar Pro score the bar lines of all systems share the same x, while
stems and clefs belong to a single system.  Requiring a column to cut through
several systems therefore separates真小节线 from stems.
"""
import json
import sys

import cv2
import numpy as np

from stitch import find_segments  # noqa: E402


def staff_groups(gray, bright=False):
    """Return (top, bottom) rows of every five-line staff in the frame."""
    ink = gray > 60 if bright else gray < 210
    frac = ink.mean(axis=1)
    rows = [y for y in range(len(frac)) if frac[y] > 0.45]
    lines = []
    for y in rows:
        if lines and y - lines[-1][-1] <= 2:
            lines[-1].append(y)
        else:
            lines.append([y])
    centres = [float(np.mean(g)) for g in lines]
    if len(centres) < 5:
        return []
    out = []
    i = 0
    while i + 4 < len(centres):
        window = centres[i:i + 5]
        gaps = np.diff(window)
        if gaps.min() >= 4 and gaps.max() <= 2.0 * gaps.min():
            out.append((int(round(window[0])), int(round(window[-1]))))
            i += 5
        else:
            i += 1
    return out


def frame_bars(gray, need_ratio=0.5, bright=False):
    groups = staff_groups(gray, bright)
    if len(groups) < 2:
        return []
    dark = gray > 60 if bright else gray < 210
    h = gray.shape[0]
    cols = np.zeros(gray.shape[1], np.int16)
    for top, bot in groups:
        inner = dark[top:bot + 1]
        full = inner.mean(axis=0) > 0.92
        up = dark[max(0, top - 4):max(1, top - 1)]
        dn = dark[min(h, bot + 2):min(h, bot + 5)]
        above = up.mean(axis=0) > 0.4 if up.size else np.zeros(gray.shape[1], bool)
        below = dn.mean(axis=0) > 0.4 if dn.size else np.zeros(gray.shape[1], bool)
        cols += (full & ~above & ~below)
    need = max(2, int(round(len(groups) * need_ratio)))
    good = cols >= need
    bars, x = [], 0
    while x < len(good):
        if good[x]:
            s = x
            while x + 1 < len(good) and good[x + 1]:
                x += 1
            bars.append((s + x) // 2)
        x += 1
    return bars


def main():
    video, out = sys.argv[1], sys.argv[2]
    bright = "--bright" in sys.argv
    segs, fps, total = find_segments(video, 96, 1000, verbose=False)
    kept = [s for s in segs if s["dur"] >= 1.2]
    cap = cv2.VideoCapture(video)
    hits = []
    for s in kept:
        f = int((s["start"] + s["end"]) // 2)
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, fr = cap.read()
        if not ok:
            continue
        g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        bars = frame_bars(g, bright=bright)
        print(f"t={s['t0']:7.2f} bars={len(bars)} {bars}")
        hits.append({"t0": s["t0"], "bars": bars})
    cap.release()
    json.dump(hits, open(out, "w"), indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main()
