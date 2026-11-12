"""Typeset a stitched multi-staff (band) score strip into printable A4 pages.

`a4layout.py` assumes one drum staff.  A band score has several staves per
system (vocal, guitar + tab, bass + tab, drums ...), so this tool keeps the
whole band together and lays out complete systems.

* the strip is turned into pure white paper with solid black ink,
* the staves are found automatically and give the mm scale,
* each output line starts with the clef block cropped from the strip head,
* line breaks happen only on bar lines.
"""
import argparse
import json

import cv2
import numpy as np
from PIL import Image, ImageDraw

A4_PORTRAIT = (595.28, 841.89)
A4_LANDSCAPE = (841.89, 595.28)


def frac_rows(mask, thr=0.7):
    return np.flatnonzero(mask.mean(axis=1) > thr)


def group(rows, gap=3):
    out = []
    for y in rows:
        if out and y - out[-1][-1] <= gap:
            out[-1].append(int(y))
        else:
            out.append([int(y)])
    return [int(round(np.mean(g))) for g in out]


def find_staves(ink):
    """Rows carrying staff lines -> list of staves, each a list of line rows."""
    rows = group(frac_rows(ink, 0.7))
    staves, cur = [], []
    for y in rows:
        if cur and y - cur[-1] > 22:          # a bigger gap ends the staff
            if len(cur) >= 4:
                staves.append(cur)
            cur = []
        cur.append(y)
    if len(cur) >= 4:
        staves.append(cur)
    return staves


def bar_columns(ink, staves, cover=0.7, gap=10):
    """Columns that cut through every staff = real bar lines.

    A stem or a beam lives inside one staff, a bar line crosses them all, so
    requiring every staff to be filled is what separates the two.  The test is
    a *continuous* run of at least `cover` x the staff height (a morphological
    opening with a vertical kernel) rather than a mean: mean fill is fooled by
    a fat chord that happens to line up with a stem in the next staff, while a
    real bar line is one uninterrupted stroke from above the top line to below
    the bottom one.
    """
    score = np.zeros(ink.shape[1], np.int16)
    for st in staves:
        band = (ink[st[0]:st[-1] + 1] > 0.5).astype(np.uint8) * 255
        k = max(6, int(round(cover * band.shape[0])))
        op = cv2.morphologyEx(band, cv2.MORPH_OPEN, np.ones((k, 1), np.uint8))
        score += (op.max(axis=0) > 0)
    cols = np.flatnonzero(score >= len(staves))
    out = []
    for c in cols:
        if out and c - out[-1] <= gap:
            continue
        out.append(int(c))
    return out


def mask_overlay(ink, g, staves, thr=0.35, keep=0.45):
    """Blank the semi-transparent vertical band that follows the playhead.

    Between two staves a clean score is empty, so any column that is lit in the
    gaps belongs to the overlay rather than to the music.
    """
    gaps = []
    for a, b in zip(staves[:-1], staves[1:]):
        y0, y1 = a[-1] + 6, b[0] - 6
        if y1 - y0 >= 8:
            gaps.append((y0, y1))
    if not gaps:
        return ink, []
    lit = np.stack([(g[y0:y1] > 20).mean(axis=0) for y0, y1 in gaps]).mean(axis=0)
    lit = np.convolve(lit, np.ones(5) / 5.0, mode="same")
    on = lit > thr
    bands, i = [], 0
    while i < len(on):
        if on[i]:
            j = i
            while j + 1 < len(on) and on[j + 1]:
                j += 1
            if j - i + 1 >= 10:
                bands.append((i, j))
            i = j + 1
        else:
            i += 1
    for a, b in bands:
        ink[:, a:b + 1] = np.minimum(ink[:, a:b + 1], keep)
    return ink, bands


def clean(path, gmin, gdim, gbright, blob_cut=5.5):
    """Band-reject ink mask.

    In this render the staff lines sit at grey ~50 and the notes/bar lines are
    near white, while the photo overlay behind the score lives in the mid
    greys.  Keeping the dim band plus the bright band and dropping the middle
    is what erases the picture without touching the notation.
    """
    g = cv2.imread(path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    ink = np.zeros_like(g)
    dim = (g > gmin) & (g < gdim)
    ink[dim] = 0.85 + 0.15 * (g[dim] - gmin) / max(1.0, gdim - gmin)
    ink[g > gbright] = 1.0
    # local contrast: where the translucent photo sits on top of the score it
    # pulls a bar line down into the mid greys, i.e. into the band we reject --
    # comparing against a blurred copy puts those strokes back
    bg = cv2.GaussianBlur(g, (0, 0), 18)
    ink[(g - bg) > 15] = 1.0
    if blob_cut > 0:
        thick = cv2.distanceTransform((g > 60).astype(np.uint8), cv2.DIST_L2, 5)
        core = (thick > blob_cut).astype(np.uint8)
        if core.any():
            ink[cv2.dilate(core, np.ones((11, 11), np.uint8)) > 0] = 0.0
    ink = cv2.GaussianBlur(ink, (0, 0), 0.6)
    struct = g > 20
    return ink, struct


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("strip")
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--bars-json", default=None)
    ap.add_argument("--measures-per-line", type=int, default=0,
                    help="0 = fill each line to the page width")
    ap.add_argument("--systems-per-page", type=int, default=2)
    ap.add_argument("--staff-space-mm", type=float, default=1.15)
    ap.add_argument("--margin-mm", type=float, default=11.0)
    ap.add_argument("--gap-mm", type=float, default=6.0)
    ap.add_argument("--top-mm", type=float, default=6.0)
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--portrait", action="store_true")
    ap.add_argument("--page-numbers", action="store_true")
    ap.add_argument("--thresh", type=float, default=0.16,
                    help="unused placeholder kept for CLI compatibility")
    ap.add_argument("--gmin", type=float, default=35.0,
                    help="dimmest value counted as ink (staff lines ~50)")
    ap.add_argument("--gdim", type=float, default=80.0,
                    help="upper edge of the dim band (staff lines)")
    ap.add_argument("--gbright", type=float, default=160.0,
                    help="lower edge of the bright band (notes, bar lines)")
    ap.add_argument("--blob-cut", type=float, default=5.5,
                    help="drop filled blobs thicker than this (photo highlights)")
    ap.add_argument("--overlay-thr", type=float, default=0.35,
                    help="gap-fill fraction that marks the playhead overlay band")
    ap.add_argument("--overlay-keep", type=float, default=0.45,
                    help="ink cap inside an overlay band (1 = leave untouched)")
    ap.add_argument("--highpass", type=float, default=0.0)
    ap.add_argument("--gain", type=float, default=25.0)
    ap.add_argument("--hp-floor", type=float, default=0.0)
    ap.add_argument("--clef-width", type=int, default=0,
                    help="px taken from the strip head as the clef block (0 = auto)")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    ink, struct = clean(args.strip, args.gmin, args.gdim, args.gbright,
                        args.blob_cut)
    h, w = ink.shape
    staves = find_staves(struct)
    if not staves:
        raise SystemExit("no staves found - tune --thresh")
    grid = ink.copy()                     # unblanked copy, used for bar lines
    ink, bands = mask_overlay(ink, cv2.imread(args.strip, cv2.IMREAD_GRAYSCALE),
                              staves, args.overlay_thr, args.overlay_keep)
    if bands:
        print(f"blanked {len(bands)} overlay bands "
              f"({sum(b - a + 1 for a, b in bands)}px)")
    spaces = [np.median(np.diff(st)) for st in staves if len(st) == 5]
    if not spaces:
        spaces = [np.median(np.diff(st)) for st in staves if len(st) >= 5]
    space = float(np.median(spaces))
    top, bot = staves[0][0], staves[-1][-1]
    print(f"strip {w}x{h} staves={len(staves)} space={space:.1f}px "
          f"band={top}..{bot}")

    bars = bar_columns(grid, staves)
    if args.bars_json:
        bars = [int(b) for b in json.load(open(args.bars_json)) if 0 < b < w]
    print(f"bar lines: {len(bars)}")
    if len(bars) < 4:
        raise SystemExit("too few bar lines")

    page_w, page_h = A4_PORTRAIT if args.portrait else A4_LANDSCAPE
    margin = args.margin_mm / 25.4 * 72
    gap = args.gap_mm / 25.4 * 72
    usable_w = page_w - 2 * margin
    usable_h = page_h - 2 * margin - args.top_mm / 25.4 * 72
    scale = args.staff_space_mm / 25.4 * 72 / space      # pt per px

    # clef block: everything left of the first bar line (min 60 px so the
    # clefs are always whole)
    clef_w = args.clef_width or max(60, bars[0])
    lead = clef_w * scale

    # group measures into lines
    edges = bars + [w]
    measures = [(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]
    lines, cur, cur_w = [], [], 0.0
    limit = usable_w - lead
    for a, b in measures:
        mw = (b - a) * scale
        if args.measures_per_line:
            if len(cur) >= args.measures_per_line:
                lines.append(cur)
                cur, cur_w = [], 0.0
        elif cur and cur_w + mw > limit:
            lines.append(cur)
            cur, cur_w = [], 0.0
        cur.append((a, b))
        cur_w += mw
    if cur:
        lines.append(cur)

    band_top = max(0, top - int(0.55 * space))
    band_bot = min(h, bot + int(0.55 * space))
    band_h = (band_bot - band_top) * scale
    fits = max(1, int((usable_h + gap) // (band_h + gap)))
    per_page = min(args.systems_per_page, fits)
    if per_page < args.systems_per_page:
        print(f"note: only {fits} systems fit per page, using {per_page}")
    print(f"scale={scale:.3f}pt/px band_h={band_h:.1f}pt lead={lead:.1f}pt "
          f"{len(lines)} lines -> {-(-len(lines) // per_page)} pages")

    dpi_scale = args.dpi / 72.0
    cw, ch = int(round(page_w * dpi_scale)), int(round(page_h * dpi_scale))
    pages = []
    for page_no, start in enumerate(range(0, len(lines), per_page), start=1):
        img = Image.new("L", (cw, ch), 255)
        draw = ImageDraw.Draw(img)
        for row, line in enumerate(lines[start:start + per_page]):
            y = int(round((margin + args.top_mm / 25.4 * 72
                           + row * (band_h + gap)) * dpi_scale))
            x = margin * dpi_scale
            if lead > 0:
                tile = ink[band_top:band_bot, 0:clef_w]
                pil = Image.fromarray((tile * 255).astype(np.uint8))
                tw = max(1, int(round(lead * dpi_scale)))
                th = max(1, int(round(band_h * dpi_scale)))
                pil = pil.resize((tw, th), Image.LANCZOS)
                img.paste(pil.convert("L").point(lambda v: 255 - v), (int(x), y))
                x += tw
            line_w = sum((b - a) for a, b in line) * scale
            stretch = 1.0
            if line_w + lead > usable_w:
                stretch = (usable_w - lead) / line_w
            for a, b in line:
                piece = ink[band_top:band_bot, a:b]
                tw = max(1, int(round((b - a) * scale * stretch * dpi_scale)))
                th = max(1, int(round(band_h * dpi_scale)))
                pil = Image.fromarray((piece * 255).astype(np.uint8))
                pil = pil.resize((tw, th), Image.LANCZOS)
                img.paste(pil.convert("L").point(lambda v: 255 - v), (int(x), y))
                x += tw
            if args.debug:
                print(f"  page {page_no} row {row}: measures "
                      f"{line[0][0]}..{line[-1][1]} bars={len(line)} "
                      f"stretch={stretch:.3f}")
        if args.page_numbers:
            draw.text((cw - int(margin * dpi_scale), ch - int(12 * dpi_scale)),
                      str(page_no), fill=140, anchor="ra")
        pages.append(img)
        img.save(f"{args.output}.p{page_no:02d}.png")

    pages[0].save(args.output, "PDF", resolution=args.dpi, save_all=True,
                  append_images=pages[1:])
    print(f"wrote {args.output} ({len(pages)} pages, {page_w:.0f}x{page_h:.0f}pt)")


if __name__ == "__main__":
    main()
