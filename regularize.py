"""Fill in bar lines that the detector missed, using the fact that the measures
of one section all have (nearly) the same width.

Detected positions are kept as anchors.  Any run between two anchors that is
1.4x or more of the local measure width is assumed to hold round(run/width)
measures, and the missing dividers are placed on that even grid, then snapped
onto the nearest column that still looks like a stroke (the detector can be
sure of a bar without the photo letting all seven staves show it).
"""
import argparse
import json

import cv2
import numpy as np

import system_layout as sl


def score_profile(g):
    """How many of the seven staves have a full-height stroke in each column."""
    st = sl.find_staves(g > 20)
    bg = cv2.GaussianBlur(g, (0, 0), 18)
    ink = ((g - bg) > 15).astype(np.uint8)
    score = np.zeros(g.shape[1], np.int16)
    for s in st:
        hh = s[-1] - s[0] + 1
        k = max(8, int(round(0.7 * hh)))
        op = cv2.morphologyEx(ink[max(0, s[0] - 2):s[-1] + 3] * 255,
                              cv2.MORPH_OPEN, np.ones((k, 1), np.uint8))
        score += (op.max(axis=0) > 0)
    return score, st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("strip")
    ap.add_argument("--bars", required=True)
    ap.add_argument("--out", "-o", required=True)
    ap.add_argument("--ratio", type=float, default=1.4,
                    help="a run this many times the local width holds 2+ measures")
    ap.add_argument("--snap", type=float, default=0.18,
                    help="search +-this fraction of a measure for the real stroke")
    ap.add_argument("--min-score", type=int, default=2)
    args = ap.parse_args()

    g = cv2.imread(args.strip, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    score, staves = score_profile(g)
    w = g.shape[1]
    bars = sorted(int(x) for x in json.load(open(args.bars)) if 0 < x < w)

    def local_width(i):
        lo = max(0, i - 12)
        hi = min(len(bars) - 1, i + 12)
        d = np.diff(bars[lo:hi + 1])
        d = d[(d > 0.6 * np.median(d)) & (d < 1.4 * np.median(d))]
        return float(np.median(d)) if len(d) else float(np.median(np.diff(bars)))

    out = [bars[0]]
    added = 0
    for i in range(len(bars) - 1):
        a, b = bars[i], bars[i + 1]
        gap = b - a
        wloc = local_width(i)
        n = int(round(gap / wloc))
        if n < 2 or gap / wloc < args.ratio:
            out.append(b)
            continue
        for k in range(1, n):
            ideal = a + gap * k / n
            win = max(3, int(round(args.snap * gap / n)))
            lo, hi = int(ideal) - win, int(ideal) + win + 1
            seg = score[max(0, lo):hi]
            if len(seg) and seg.max() >= args.min_score:
                x = int(max(0, lo)) + int(np.argmax(seg))
                # keep it away from the anchors so measures stay non-degenerate
                x = min(max(x, a + gap // (2 * n)), b - gap // (2 * n))
            else:
                x = int(round(ideal))
            out.append(x)
            added += 1
        out.append(b)
    out = sorted(set(out))
    d = np.diff(out)
    print(f"{len(bars)} -> {len(out)} bars (+{added}); "
          f"spacing median {np.median(d):.0f}px "
          f"min {d.min()} max {d.max()} "
          f"hist {np.histogram(d, bins=[0, 150, 250, 330, 450, 700, 1200])[0]}")
    json.dump(out, open(args.out, "w"))


if __name__ == "__main__":
    main()
