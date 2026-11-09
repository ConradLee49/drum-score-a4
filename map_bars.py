"""Map frame-space bar lines into stitched-strip coordinates."""
import json
import sys

import numpy as np


def cluster(vals, tol=4):
    vals = sorted(vals)
    out = []
    for v in vals:
        if out and v - out[-1][-1] <= tol:
            out[-1].append(v)
        else:
            out.append([v])
    return [int(round(float(np.median(g)))) for g in out]


def main():
    stitch = json.load(open(sys.argv[1]))
    framebars = json.load(open(sys.argv[2]))
    out = sys.argv[3]
    segs = stitch.get("segments") or stitch.get("rows")
    offs = stitch.get("offsets")
    if offs is None:
        offs = [0]
        for link in stitch.get("links", []):
            offs.append(offs[-1] + (link["dx"] if link["ok"] else 0))
        print(f"replayed {len(offs)} offsets")
    by_t = {round(h["t0"], 2): h["bars"] for h in framebars}
    mapped = []
    for s, off in zip(segs, offs):
        bars = by_t.get(round(s["t0"], 2))
        if not bars:
            print(f"  no bars for t={s['t0']:.2f}")
            continue
        for b in bars:
            mapped.append(off + b)
    bars = cluster(mapped)
    w = stitch.get("width")
    if w:
        bars = [b for b in bars if 0 < b < w]
    if len(bars) > 4:
        typical = float(np.median(np.diff(bars)))
        kept = [bars[0]]
        for b in bars[1:]:
            if b - kept[-1] >= 0.45 * typical:
                kept.append(b)
        if len(kept) != len(bars):
            print(f"dropped {len(bars) - len(kept)} stray bars")
        bars = kept
    print(f"{len(mapped)} hits -> {len(bars)} unique bars")
    gaps = np.diff(bars)
    print(f"spacing min={gaps.min()} med={np.median(gaps):.0f} max={gaps.max()}")
    json.dump(bars, open(out, "w"))
    print("wrote", out)


if __name__ == "__main__":
    main()
