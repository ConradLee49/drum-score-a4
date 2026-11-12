"""Montage a binarised staff strip as rows of N measures.

This is the "cut and paste" alternative to `a4layout.py`: it does not re-engrave
anything, it just cuts the strip at the bar lines and pastes the pieces onto A4.
Because nothing is re-drawn, no note can be lost.

Three width modes, because a scrolling score (Guitar Pro and friends) does not
have evenly wide bars -- a bar of 16ths covers ~1.5x the width of a bar of
8ths:

* (default) `--cap 0.85` -- every bar may be squeezed/stretched by at most 15%,
  the leftovers become space between bars.  Rows of equally dense bars come out
  perfectly even, the staff size only drops where a whole row is dense.
* `--equal` -- give every bar the same slot: bar lines are perfectly even, but
  a dense bar's noteheads get squashed (down to ~0.70 in this corpus).
* `--cap 0` -- natural widths, i.e. exactly what the video shows; bar lines are
  then as uneven as they are on screen.
"""
import argparse
import json

import cv2
import numpy as np
from PIL import Image, ImageDraw

A4 = (595.0, 842.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("strip", help="binary strip: 0 = ink, 255 = paper")
    ap.add_argument("--bars", required=True)
    ap.add_argument("--out", "-o", required=True)
    ap.add_argument("--per-line", type=int, default=4)
    ap.add_argument("--equal", action="store_true",
                    help="give every measure the same slot width (bar lines then "
                         "sit on an even grid; the notes inside keep their order "
                         "but are spaced tighter in dense bars)")
    ap.add_argument("--space-mm", type=float, default=1.6,
                    help="staff space used by --equal / --cap")
    ap.add_argument("--cap", type=float, default=0.85,
                    help="allow at most this much horizontal squeeze per measure "
                         "and absorb the rest as spacing between measures "
                         "(0 = keep the natural widths)")
    ap.add_argument("--max-space-mm", type=float, default=1.7,
                    help="do not enlarge a short system past this staff space")
    ap.add_argument("--lines", default=None, help="staff-line rows, e.g. 16,26,37,46,55")
    ap.add_argument("--dpi", type=int, default=400)
    ap.add_argument("--margin-mm", type=float, default=12.0)
    ap.add_argument("--gap-mm", type=float, default=5.0)
    ap.add_argument("--bar-pt", type=float, default=1.3)
    ap.add_argument("--staff-pt", type=float, default=0.35)
    args = ap.parse_args()

    g = cv2.imread(args.strip, cv2.IMREAD_GRAYSCALE)
    h, w = g.shape
    bars = sorted(int(b) for b in json.load(open(args.bars)) if 0 < b < w)
    edges = [0] + bars + [w]
    # a stub of a measure at the very end (the video cut off mid-bar) would wreck
    # the even grid; merge it into the previous measure
    med = float(np.median(np.diff(edges)))
    while len(edges) > 2 and edges[-1] - edges[-2] < 0.5 * med:
        print(f"dropping trailing stub {edges[-1] - edges[-2]}px")
        edges.pop(-2)
    print(f"strip {w}x{h}, {len(edges) - 1} measures ({len(bars)} bar lines)")

    if args.lines:
        staff = [int(v) for v in args.lines.split(",")]
    else:
        dark = (g < 160).mean(axis=1)
        staff = [int(y) for y in np.flatnonzero(dark > 0.5)]
    print(f"staff rows {staff}")
    space_px = float(np.median(np.diff(staff))) if len(staff) > 1 else 9.5
    print(f"staff space {space_px:.2f}px")

    n = max(1, args.per_line)
    groups = [(edges[i], edges[min(i + n, len(edges) - 1)])
              for i in range(0, len(edges) - 1, n)]
    print(f"{len(groups)} systems of up to {n} measures")

    dpi_scale = args.dpi / 72.0
    page_w, page_h = A4
    cw, ch = int(round(page_w * dpi_scale)), int(round(page_h * dpi_scale))
    margin = args.margin_mm / 25.4 * 72 * dpi_scale
    usable_w = cw - 2 * margin
    usable_h = ch - 2 * margin
    gap = args.gap_mm / 25.4 * 72 * dpi_scale

    y = margin
    pages = []
    img = Image.new("L", (cw, ch), 255)
    draw = ImageDraw.Draw(img)
    bar_px = max(2, int(round(args.bar_pt * dpi_scale)))
    staff_px = max(1, int(round(args.staff_pt * dpi_scale)))
    spaces = []

    for a, b in groups:
        piece = g[:, a:b]
        if args.cap > 0:
            # Compromise between "natural widths" and "all bars equal": a
            # measure may be squeezed or stretched by at most `cap`, and only as
            # much as it must be -- the rest of the width becomes space between
            # measures.  Rows of equally dense bars therefore end up perfectly
            # even, and the staff size only drops where a whole row is dense.
            s_y = args.space_mm / 25.4 * 72 * dpi_scale / space_px
            idxs = [i for i in range(len(edges) - 1) if a <= edges[i] < b]
            ws = [edges[i + 1] - edges[i] for i in idxs]
            # a row that mixes very narrow and very wide bars cannot be made
            # even within the cap; loosen it just enough for this row to
            # equalise, no further (sqrt keeps the squeeze symmetric)
            c_row = args.cap
            if len(ws) > 1:
                c_row = min(c_row, max(0.70, float(np.sqrt(min(ws) / max(ws)))))
            lo = [c_row * w for w in ws]
            hi = [w / c_row for w in ws]
            budget = usable_w / s_y
            if sum(lo) > budget:                 # a whole dense row
                s_row = s_y * budget / sum(lo)
                budget = sum(lo)
            else:
                s_row = s_y
            t_lo, t_hi = min(lo), max(hi)
            for _ in range(50):                  # water-fill the slot width
                t = 0.5 * (t_lo + t_hi)
                if sum(min(max(t, x), h) for x, h in zip(lo, hi)) > budget:
                    t_hi = t
                else:
                    t_lo = t
            slots = [min(max(0.5 * (t_lo + t_hi), x), h) for x, h in zip(lo, hi)]
            slack = budget - sum(slots)
            npad = max(1, len(slots) - 1)
            pad = max(0.0, slack) / npad if len(slots) > 1 else 0.0
            pad = min(pad, 0.45 * float(np.median(ws)))
            th = max(1, int(round(piece.shape[0] * s_row)))
            if y + th > ch - margin and y > margin:
                pages.append(img)
                img = Image.new("L", (cw, ch), 255)
                draw = ImageDraw.Draw(img)
                y = margin
            top = y + staff[0] * s_row
            bot = y + staff[-1] * s_row
            x = float(margin)
            right = x
            for k, i in enumerate(idxs):
                tw = max(1, int(round(slots[k] * s_row)))
                cell = Image.fromarray(g[:, edges[i]:edges[i + 1]])
                img.paste(cell.resize((tw, th), Image.LANCZOS), (int(round(x)), int(y)))
                draw.line((int(round(x)), top, int(round(x)), bot), fill=0,
                          width=bar_px)
                right = x + tw
                x += slots[k] * s_row + pad * s_row
            draw.line((int(round(right)), top, int(round(right)), bot), fill=0,
                      width=bar_px)
            for r in staff:
                draw.line((margin, y + r * s_row, right, y + r * s_row), fill=0,
                          width=staff_px)
            spaces.append(space_px * s_row / args.dpi * 25.4)
            y += th + gap
            continue
        if args.equal:
            # One slot per measure.  Guitar Pro justifies a system by note
            # count, so a bar of 16ths is 1.5x wider on screen than a bar of
            # 8ths; a printed chart instead keeps every bar the same width and
            # lets the dense bars get tighter.  The grid of bar lines is then
            # perfectly even, which is what a reader expects.
            s_y = args.space_mm / 25.4 * 72 * dpi_scale / space_px
            th = max(1, int(round(piece.shape[0] * s_y)))
            if y + th > ch - margin and y > margin:
                pages.append(img)
                img = Image.new("L", (cw, ch), 255)
                draw = ImageDraw.Draw(img)
                y = margin
            slot = int(round(usable_w / args.per_line))
            idxs = [i for i in range(len(edges) - 1) if a <= edges[i] < b]
            top = y + staff[0] * s_y
            bot = y + staff[-1] * s_y
            for k, i in enumerate(idxs):
                cell = g[:, edges[i]:edges[i + 1]]
                tile = Image.fromarray(cell).resize((slot, th), Image.LANCZOS)
                xpos = int(round(margin + k * slot))
                img.paste(tile, (xpos, int(y)))
                draw.line((xpos, top, xpos, bot), fill=0, width=bar_px)
            right = int(round(margin + slot * max(1, len(idxs))))
            draw.line((right - 1, top, right - 1, bot), fill=0, width=bar_px)
            for r in staff:
                draw.line((margin, y + r * s_y, right, y + r * s_y), fill=0,
                          width=staff_px)
            spaces.append(args.space_mm)
            y += th + gap
            continue
        s = min(usable_w / piece.shape[1],
                args.max_space_mm / 25.4 * 72 * dpi_scale / space_px)
        tw = int(round(piece.shape[1] * s))
        th = int(round(piece.shape[0] * s))
        if y + th > ch - margin and y > margin:
            pages.append(img)
            img = Image.new("L", (cw, ch), 255)
            draw = ImageDraw.Draw(img)
            y = margin
        tile = Image.fromarray(piece).resize((tw, th), Image.LANCZOS)
        img.paste(tile, (int(margin), int(y)))
        # re-draw the staff and bar lines crisply at the output resolution
        top = y + staff[0] * s
        bot = y + staff[-1] * s
        for r in staff:
            draw.line((margin, y + r * s, margin + tw, y + r * s), fill=0, width=staff_px)
        for e in [x for x in edges if a < x <= b]:
            xs = margin + (e - a) * s
            draw.line((xs, top, xs, bot), fill=0, width=bar_px)
        # the slice starts *at* a bar line, and that column was erased from the
        # strip when it was cut -- without this every system after the first
        # would hang in the air on its left side
        if a > 0:
            draw.line((margin, top, margin, bot), fill=0, width=bar_px)
        spaces.append(space_px * s / args.dpi * 25.4)   # staff space on paper, mm
        y += th + gap

    pages.append(img)
    print(f"{len(pages)} pages")
    print("staff space per system (mm):", [round(v, 2) for v in spaces])
    print(f"min {min(spaces):.2f}mm max {max(spaces):.2f}mm")
    pages[0].save(args.out, "PDF", resolution=args.dpi, save_all=True,
                  append_images=pages[1:])
    print("->", args.out)


if __name__ == "__main__":
    main()
