"""Cut the drum staff out of a stitched band score and binarise it for a4layout.

In this render the notation is the only thing brighter than ~150; the staff
lines sit at grey ~50 and the translucent photo/playhead live in 80..145, so a
plain bright threshold already yields a clean staff.  Bar lines are erased (the
layout draws its own) and the five staff lines are forced to 1 px so the
engraved result is crisp.
"""
import argparse
import json

import cv2
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("strip")
    ap.add_argument("--bars", required=True, help="bar-line x list (json)")
    ap.add_argument("--out", "-o", required=True)
    ap.add_argument("--rows", required=True,
                    help="drum unit rows in the stitched strip, e.g. 707:792")
    ap.add_argument("--lines", default=None,
                    help="comma separated staff-line rows (default: auto)")
    ap.add_argument("--bright", type=float, default=150.0)
    ap.add_argument("--pad", type=int, default=0)
    args = ap.parse_args()

    g = cv2.imread(args.strip, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    t, b = (int(v) for v in args.rows.split(":"))
    t = max(0, t - args.pad)
    b = min(g.shape[0], b + args.pad)
    sub = g[t:b]
    mask = sub > args.bright

    if args.lines:
        lines = [int(v) - t for v in args.lines.split(",")]
    else:
        frac = mask.mean(axis=1)
        rows = np.flatnonzero(frac > 0.5)
        groups = []
        for y in rows:
            if groups and y - groups[-1][-1] <= 2:
                groups[-1].append(int(y))
            else:
                groups.append([int(y)])
        lines = [int(round(np.mean(x))) for x in groups]
    print(f"crop rows {t}..{b} ({b - t}px), staff lines at {[l + t for l in lines]}")

    out = np.where(mask, 0, 255).astype(np.uint8)
    bars = [int(x) for x in json.load(open(args.bars)) if 0 < x < sub.shape[1]]
    for x in bars:
        out[:, max(0, x - 1):x + 2] = 255
    for y in lines:
        out[y, :] = 0
    cv2.imwrite(args.out, out)

    m = out < 160
    print(f"strip {out.shape} ink {m.mean():.3f}  bars erased {len(bars)}")
    print("top rows", [int(m[i].sum()) for i in range(3)],
          "bottom rows", [int(m[out.shape[0] - 1 - i].sum()) for i in range(3)])


if __name__ == "__main__":
    main()
