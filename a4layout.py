"""Paginate a stitched score strip into a clean, portrait A4 drum chart.

* the strip is re-levelled to pure white paper with solid black ink,
* line breaks happen only on bar lines (a measure is never split),
* every line is typeset at one scale and justified by widening the gaps at
  bar lines, so the result looks engraved rather than screenshotted.
"""
import argparse

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

A4_PORTRAIT = (595.28, 841.89)
A4_LANDSCAPE = (841.89, 595.28)

_FONTS = {}


def _font(size):
    if size in _FONTS:
        return _FONTS[size]
    for path in ("/System/Library/Fonts/Supplemental/Arial.ttf",
                 "/System/Library/Fonts/Helvetica.ttc",
                 "/Library/Fonts/Arial.ttf",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            _FONTS[size] = ImageFont.truetype(path, size)
            return _FONTS[size]
        except Exception:
            continue
    _FONTS[size] = ImageFont.load_default()
    return _FONTS[size]


# ---------------------------------------------------------------- cleaning
def clean(gray, mode, ink_v, paper_v, highpass=0.0, gain=25.0, floor=0.0):
    """Return ink-on-white uint8 image (0 = ink, 255 = paper)."""
    g = gray.astype(np.float32)
    if highpass:
        blur = cv2.GaussianBlur(g, (0, 0), highpass)
        detail = g - blur
        frac = np.clip((detail - floor) / gain, 0.0, 1.0)
        return (255.0 * (1.0 - frac)).astype(np.uint8)
    span = abs(paper_v - ink_v)
    if mode == "bright":          # white notation on a dark background
        frac = (g - paper_v) / span
    else:                          # dark notation on a light background
        frac = (paper_v - g) / span
    frac = np.clip(frac, 0.0, 1.0)
    return (255.0 * (1.0 - frac)).astype(np.uint8)


def staff_rows(ink):
    h = ink.shape[0]
    frac = (ink < 200).mean(axis=1)
    cand = [y for y in range(h) if frac[y] > 0.5]
    groups = []
    for y in cand:
        if groups and y - groups[-1][-1] <= 2:
            groups[-1].append(y)
        else:
            groups.append([y])
    return [int(round(np.mean(g))) for g in groups]


def bar_columns(ink, lines, keep_ratio=0.55):
    """Columns that look like bar lines: ink across the whole staff, none beyond.

    Stems and the percussion clef also satisfy the strict test, so candidates
    closer than ~half a typical measure to the previous accepted bar line are
    dropped (a clef splits into two bars a few pixels apart).
    """
    h, w = ink.shape
    top, bot = lines[0], lines[-1]
    dark = ink < 160
    inside = dark[top:bot + 1].mean(axis=0)
    above = dark[max(0, top - 6):top - 1].mean(axis=0)
    below = dark[bot + 2:min(h, bot + 7)].mean(axis=0)
    good = (inside > 0.92) & (above < 0.25) & (below < 0.25)
    raw = []
    x = 0
    while x < w:
        if good[x]:
            start = x
            while x + 1 < w and good[x + 1]:
                x += 1
            raw.append((start + x) // 2)
        x += 1
    if len(raw) < 4:
        return raw
    gaps = np.diff(raw)
    typical = float(np.median(gaps))
    kept = [raw[0]]
    for b in raw[1:]:
        if b - kept[-1] >= keep_ratio * typical:
            kept.append(b)
    return kept


# ---------------------------------------------------------------- layout
def find_clef(ink, lines, search=220):
    """Bounds of the first clef glyph (a stroke crossing the whole staff)."""
    top, bot = lines[0], lines[-1]
    dark = ink < 160
    ext = dark[max(0, top - 4):bot + 5].mean(axis=0)
    cols = np.flatnonzero(ext[6:search] > 0.9) + 6
    if cols.size == 0:
        return None
    return max(0, int(cols.min()) - 2), int(cols.max()) + 3


def plain_bar(ink, lines, x, span=16):
    """True when the bar at x has no companion stroke (i.e. not a repeat bar).

    Stems near a bar used to defeat a plain stroke count, so only a second
    full-height stroke 3--12 px away marks a repeat/double bar.
    """
    h, w = ink.shape
    if x <= 0 or x >= w:
        return True
    top, bot = lines[0], lines[-1]
    dark = ink < 160
    lo, hi = max(0, x - span), min(w, x + span + 1)
    runs = []
    for xx in range(lo, hi):
        if dark[top:bot + 1, xx].mean() > 0.9:
            if runs and xx - runs[-1][-1] <= 1:
                runs[-1].append(xx)
            else:
                runs.append([xx])
    if len(runs) <= 1:
        return True
    own = min(runs, key=lambda r: min(abs(r[0] - x), abs(r[-1] - x)))
    for r in runs:
        if r is own:
            continue
        gap = max(r[0] - own[-1], own[0] - r[-1])
        if 3 <= gap <= 12:
            return False
    return True


def plan_lines(width, bars, budget):
    """Split the strip into lines, cutting only at bar lines."""
    edges = [0] + [b for b in bars if 0 < b < width - 1] + [width]
    edges = sorted(set(edges))
    lines, start = [], 0
    for i in range(1, len(edges)):
        if edges[i] - start > budget and i - 1 > 0 and edges[i - 1] > start:
            lines.append((start, edges[i - 1]))
            start = edges[i - 1]
    lines.append((start, width))
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("strip")
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--mode", choices=["bright", "dark"], default="dark")
    ap.add_argument("--ink", type=float, default=None, help="value that prints solid black")
    ap.add_argument("--paper", type=float, default=None, help="value that prints paper white")
    ap.add_argument("--staff-space-mm", type=float, default=1.55)
    ap.add_argument("--margin-mm", type=float, default=13.0)
    ap.add_argument("--gap-mm", type=float, default=2.4)
    ap.add_argument("--top-mm", type=float, default=6.0)
    ap.add_argument("--dpi", type=int, default=220)
    ap.add_argument("--landscape", action="store_true")
    ap.add_argument("--page-numbers", action="store_true")
    ap.add_argument("--bars-json", default=None,
                    help="JSON list of bar-line x positions in strip pixels")
    ap.add_argument("--clef-px", default=None, help="x0,x1 of the clef glyph")
    ap.add_argument("--bar-width-pt", type=float, default=1.2,
                    help="thickness of the drawn bar lines (0 disables)")
    ap.add_argument("--measures-per-line", type=int, default=0,
                    help="fixed number of measures per line (0 = auto width)")
    ap.add_argument("--rest-ink", type=float, default=120.0,
                    help="ink budget (px, staff lines excluded) for a measure to count as empty")
    ap.add_argument("--no-merge-rests", action="store_true",
                    help="keep every empty measure instead of a multi-measure rest")
    ap.add_argument("--repeat-dist", type=float, default=0.06,
                    help="profile distance below which two measures count as the same")
    ap.add_argument("--phrase-len", type=int, default=1,
                    help="treat this many measures as one repeated unit (e.g. 4)")
    ap.add_argument("--collapse", action="append", default=[],
                    help="explicit fold, format 'a-b:L' (measures a..b, L bars shown once)")
    ap.add_argument("--repeat-min", type=int, default=3,
                    help="minimum run of identical measures to print as xN")
    ap.add_argument("--uniform", action="store_true",
                    help="keep measures at natural width, justify with gaps")
    ap.add_argument("--balance", action="store_true",
                    help="break lines by natural width (not a fixed count) and pick "
                         "the largest staff size that fits in --max-pages pages")
    ap.add_argument("--max-pages", type=int, default=2,
                    help="page budget used by --balance")
    ap.add_argument("--debug", action="store_true",
                    help="print empty-measure runs and repeat groups")
    ap.add_argument("--highpass", type=float, default=0.0,
                    help="gaussian sigma for a high-pass clean (bright-on-dark)")
    ap.add_argument("--gain", type=float, default=25.0)
    ap.add_argument("--hp-floor", type=float, default=0.0)
    args = ap.parse_args()

    raw = cv2.imread(args.strip, cv2.IMREAD_GRAYSCALE)
    ink_v = args.ink if args.ink is not None else (205 if args.mode == "bright" else 185)
    paper_v = args.paper if args.paper is not None else (140 if args.mode == "bright" else 233)
    ink = clean(raw, args.mode, ink_v, paper_v, args.highpass, args.gain, args.hp_floor)
    h, w = ink.shape

    lines = staff_rows(ink)
    if len(lines) < 2:
        raise SystemExit(f"staff lines not found: {lines}")
    space = float(np.median(np.diff(lines)))
    print(f"strip {w}x{h} staff rows={lines} space={space:.2f}px")

    bars = bar_columns(ink, lines)
    if args.bars_json:
        import json as _json
        bars = [int(b) for b in _json.load(open(args.bars_json)) if 0 < b < w]
        print(f"bar lines from {args.bars_json}: {len(bars)}")
    # the head of the strip is the clef block, and the tail is often a stub cut
    # off by the video: neither is a measure, so drop them instead of printing
    # a blank bar at the start / end of the chart
    if len(bars) > 4:
        med = float(np.median(np.diff(bars)))
        if bars[0] < 0.55 * med:
            print(f"dropping leading clef block ({bars[0]}px)")
            bars = bars[1:]
        while len(bars) > 4 and (bars[-1] - bars[-2]) < 0.5 * med:
            print(f"dropping trailing stub ({bars[-1] - bars[-2]}px)")
            bars = bars[:-1]
    print(f"bar lines used: {len(bars)}")
    if args.clef_px and args.clef_px.strip().lower() in ("none", "0", "off"):
        clef = None
    elif args.clef_px:
        a_, b_ = (int(v) for v in args.clef_px.split(","))
        clef = (a_, b_)
    else:
        clef = find_clef(ink, lines)
    if clef:
        print(f"clef glyph at x={clef[0]}..{clef[1]}")
        clef_tile = ink[:, clef[0]:clef[1]]
    else:
        clef_tile = None

    page_w, page_h = A4_LANDSCAPE if args.landscape else A4_PORTRAIT
    margin = args.margin_mm / 25.4 * 72
    gap = args.gap_mm / 25.4 * 72
    top_pad = args.top_mm / 25.4 * 72
    usable_w = page_w - 2 * margin
    usable_h = page_h - 2 * margin - top_pad

    scale = args.staff_space_mm / 25.4 * 72 / space      # pt per px
    line_h = h * scale
    per_page = max(1, int((usable_h + gap) // (line_h + gap)))
    budget = int(usable_w / scale)
    clef_px = (clef[1] - clef[0]) if clef else 0
    print(f"scale={scale:.3f}pt/px line_h={line_h:.1f}pt per_page={per_page} "
          f"width budget={budget}px clef={clef_px}px")

    # ---- items: real measures, or one multi-measure rest per empty run
    edges = [0] + [b for b in bars if 0 < b < w] + [w]
    dark = ink < 160
    keep = np.ones(ink.shape[0], bool)
    for y in lines:
        keep[max(0, y - 1):y + 2] = False
    span = float(lines[-1] - lines[0])

    def is_empty(a, b):
        sub = dark[keep, a:b]
        total = int(sub.sum())
        if total == 0:
            return True
        rr = np.flatnonzero(sub.any(axis=1))
        return total <= args.rest_ink and (rr[-1] - rr[0]) <= 0.9 * span
    def same(pa, pb):
        """Same music test: normalized horizontal ink profile, shift-tolerant."""
        a_, b_ = measures[pa], measures[pb]
        profs = []
        for x0, x1 in (a_, b_):
            col = dark[:, x0:x1].mean(axis=0)
            if col.max() <= 0:
                return False
            q = np.interp(np.linspace(0, len(col) - 1, 40), np.arange(len(col)), col)
            profs.append(q / q.max())
        x, y = profs
        best = 9.0
        for dx in range(-2, 3):
            xs = x[2:38]
            ys = np.interp(np.linspace(2 + dx, 37 + dx, 36), np.arange(40), y)
            best = min(best, float(np.abs(xs - ys).mean()))
        return best <= args.repeat_dist

    measures = [(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]
    empty = [is_empty(a, b) for a, b in measures]
    collapse = {}
    for spec_ in args.collapse:
        rng, _, ln = spec_.partition(":")
        c0, c1 = (int(v) for v in rng.split("-"))
        collapse[c0] = (c1, max(1, int(ln or 1)))
    if args.debug:
        runs = []
        k = 0
        while k < len(empty):
            if empty[k]:
                j2 = k
                while j2 + 1 < len(empty) and empty[j2 + 1]:
                    j2 += 1
                runs.append((k, j2, j2 - k + 1))
                k = j2 + 1
            else:
                k += 1
        print(f"n_measures={len(measures)} empty={sum(empty)} "
              f"empty_runs={runs}")

    items = []
    i = 0
    while i < len(measures):
        if not args.no_merge_rests and empty[i]:
            j = i
            while j + 1 < len(measures) and empty[j + 1]:
                j += 1
            if j - i + 1 >= 2:
                items.append({"kind": "rest", "a": measures[i][0],
                              "b": measures[j][1], "n": j - i + 1})
                i = j + 1
                continue
        if i in collapse:
            c1, L = collapse[i]
            c1 = min(c1, len(measures) - 1)
            reps = max(1, int(round((c1 - i + 1) / float(L))))
            if args.debug:
                print(f"collapse: measures {i}..{c1} -> {L} bars x{reps}")
            for k in range(min(L, c1 - i + 1)):
                items.append({"kind": "bar", "a": measures[i + k][0],
                              "b": measures[i + k][1],
                              "rep": reps if k == 0 else 1})
            i = c1 + 1
            continue
        L = max(1, args.phrase_len)
        if L > 1:
            def block_ok(a, b):
                if b + L - 1 >= len(measures):
                    return False
                for k in range(L):
                    if empty[a + k] or empty[b + k] or not same(a + k, b + k):
                        return False
                return True

            reps = 1
            while block_ok(i, i + reps * L):
                reps += 1
            if reps >= 2:
                if args.debug:
                    print(f"phrase x{reps}: measures {i}..{i + reps * L - 1} "
                          f"({L} bars per unit)")
                for k in range(L):
                    items.append({"kind": "bar", "a": measures[i + k][0],
                                  "b": measures[i + k][1],
                                  "rep": reps if k == 0 else 1})
                i += reps * L
                continue
        j = i
        while (j + 1 < len(measures) and not empty[j + 1]
               and same(j, j + 1)):
            j += 1
        rep = j - i + 1
        if rep < max(2, args.repeat_min):
            rep = 1
            for k in range(i, j + 1):
                items.append({"kind": "bar", "a": measures[k][0],
                              "b": measures[k][1], "rep": 1})
            i = j + 1
            continue
        if args.debug:
            print(f"repeat group: measures {i}..{j} x{rep}")
        items.append({"kind": "bar", "a": measures[i][0],
                      "b": measures[i][1], "rep": rep})
        i = j + 1
    print(f"{len(items)} items ({sum(1 for x in items if x['kind'] == 'rest')} rests, "
          f"{sum(1 for x in items if x.get('rep', 1) > 1)} repeated groups)")

    mpl = args.measures_per_line
    if args.balance:
        # ---- line breaking by natural width -------------------------------
        # Measures in a scrolling-score strip are as wide as their content
        # (a bar full of 16ths is three times a bar of quarters), so a fixed
        # count per line either overflows the worst line (and everything gets
        # squeezed) or leaves most lines half empty.  Pack by width instead and
        # keep the largest staff size that still fits the page budget.
        med_px_b = float(np.median(np.diff(bars))) if len(bars) > 2 else 0.0

        def nat_px(it):
            return med_px_b if it["kind"] == "rest" else (it["b"] - it["a"])

        def pack(s):
            budget = usable_w - (clef_px if clef_tile is not None else 0)
            out, cur, curpx = [], [], 0.0
            for it in items:
                w = nat_px(it) * s
                if cur and curpx + w > budget:
                    out.append(cur)
                    cur, curpx = [], 0.0
                cur.append(it)
                curpx += w
            if cur:
                out.append(cur)
            return out

        def layout_at(s):
            pl = pack(s)
            lh = h * s
            pp = max(1, int((usable_h + gap) // (lh + gap)))
            return pl, pp, -(-len(pl) // pp)

        s = scale
        plan, per_page, n_pages = layout_at(s)
        for _ in range(90):
            if n_pages <= args.max_pages:
                break
            s *= 0.97
            plan, per_page, n_pages = layout_at(s)
        # Shrinking just enough to fit the page budget often leaves every line
        # 15-20% short (three wide measures overflow by a few points, so they
        # only ever fit two per line).  Trade a couple of percent of staff size
        # for lines that actually reach the right margin.
        budget_w = usable_w - (clef_px if clef_tile is not None else 0.0)
        n_pages0 = n_pages
        best = (float(np.mean([sum(nat_px(it) for it in ln) * s / budget_w
                               for ln in plan])), s, plan, per_page, n_pages)
        for k in range(1, 12):
            ks = s * (0.98 ** k)
            if ks < s * 0.895:
                break
            kp, kpp, knp = layout_at(ks)
            # keep the same page count: a slightly smaller staff that suddenly
            # fits on one page is not an improvement, it is a smaller chart
            if knp != n_pages0:
                break
            fill = float(np.mean([sum(nat_px(it) for it in ln) * ks / budget_w
                                  for ln in kp]))
            if fill > best[0] + 0.02:
                best = (fill, ks, kp, kpp, knp)
        fill, s, plan, per_page, n_pages = best
        print(f"balance: line fill {fill * 100:.0f}%")
        # a tail line with a single measure looks like a mistake: pull measures
        # down from the previous line while they still fit
        if len(plan) > 1:
            budget = usable_w - (clef_px if clef_tile is not None else 0)
            while len(plan[-2]) > 1:
                cand = plan[-2][-1]
                used = sum(nat_px(it) for it in plan[-1]) * s + nat_px(cand) * s
                if used > budget or len(plan[-1]) >= len(plan[-2]) - 1:
                    break
                plan[-1].insert(0, plan[-2].pop())
        args.uniform = True
        scale = s
        line_h = h * scale
        print(f"balance: staff space {scale * space / 72.0 * 25.4:.2f}mm, "
              f"{len(plan)} lines, {per_page}/page -> {n_pages} pages "
              f"(measures/line {min(len(l) for l in plan)}..{max(len(l) for l in plan)})")
        mpl = 0
    elif mpl <= 0:
        widths_px = [x["b"] - x["a"] for x in items]
        mpl = max(1, int(round(usable_w / scale / float(np.median(widths_px)))))
        plan = [items[k:k + mpl] for k in range(0, len(items), mpl)]
    else:
        plan = [items[k:k + mpl] for k in range(0, len(items), mpl)]
    if args.uniform and mpl:
        med_px = float(np.median(np.diff(bars))) if len(bars) > 2 else 0.0
        widest = 0.0
        for line in plan:
            wpx = sum((it["b"] - it["a"]) if it["kind"] == "bar" else med_px
                      for it in line)
            widest = max(widest, wpx)
        if widest > 0:
            cap = usable_w / (widest + clef_px * (1 if clef_tile is not None else 0))
            if cap < scale:
                print(f"uniform fit: staff space {args.staff_space_mm:.2f}mm -> "
                      f"{cap * space / 72.0 * 25.4:.2f}mm")
                scale = cap
                line_h = h * scale
                per_page = max(1, int((usable_h + gap) // (line_h + gap)))
    print(f"{len(plan)} lines -> {-(-len(plan) // per_page)} pages (mpl={mpl})")

    dpi_scale = args.dpi / 72.0
    cw, ch = int(round(page_w * dpi_scale)), int(round(page_h * dpi_scale))
    pages = []
    for page_no, start in enumerate(range(0, len(plan), per_page), start=1):
        img = Image.new("L", (cw, ch), 255)
        draw = ImageDraw.Draw(img)
        for row, line_items in enumerate(plan[start:start + per_page]):
            add_clef = clef_tile is not None and not (start == 0 and row == 0)
            lead = clef_px * scale if add_clef else 0.0
            pad = 0.0
            adv = None
            med_px = float(np.median(np.diff(edges[:-1]))) if len(edges) > 2 else 0.0

            def slot_px(it):
                # a multi-measure rest is written once, so it takes one measure
                return med_px if it["kind"] == "rest" else (it["b"] - it["a"])

            if args.uniform and mpl:
                # fixed count per line, but every measure keeps its own width:
                # a line that would overflow is scaled down as a whole, so no
                # measure ever overlaps its neighbour (that is what used to eat
                # notes) and no head is ever stretched
                widths = [slot_px(it) * scale for it in line_items]
                slack = usable_w - lead - sum(widths)
                pad = slack / (len(line_items) - 1) if len(line_items) > 1 else 0.0
                if pad < 0:
                    pad = 0.0
                # a short last line must not be blown up to full width, but a
                # full line is still justified to the right margin
                if len(line_items) < mpl:
                    pad = min(pad, 0.40 * med_px * scale)
            elif args.uniform:
                widths = [slot_px(it) * scale for it in line_items]
                slack = max(0.0, usable_w - lead - sum(widths))
                pad = slack / (len(line_items) - 1) if len(line_items) > 1 else 0.0
                # a line that is naturally short (usually the last one) must not
                # be spread over the page: cap the gap and leave it ragged
                pad = min(pad, 0.45 * med_px * scale)
            else:
                slot = (usable_w - lead) / max(1, mpl)
                widths = [slot] * len(line_items)
            plain = [plain_bar(ink, lines, it["a"]) for it in line_items]
            plain_end = plain_bar(ink, lines, line_items[-1]["b"])
            squeeze = 1.0
            line_w = lead + (sum(adv) if adv else
                             sum(widths) + pad * max(0, len(line_items) - 1))
            if args.debug:
                print(f"  line p{page_no}r{row}: items={len(line_items)} "
                      f"lead={lead:.1f} widths={[round(w, 1) for w in widths]} "
                      f"pad={pad:.1f} line_w={line_w:.1f} usable={usable_w:.1f}")
            if line_w > usable_w:
                squeeze = usable_w / line_w
                widths = [x * squeeze for x in widths]
                if adv:
                    adv = [x * squeeze for x in adv]
                lead *= squeeze
            line_h_eff = line_h * squeeze
            y = int(round((margin + top_pad + row * (line_h + gap)) * dpi_scale))
            if squeeze < 1.0:
                y += int(round((line_h - line_h_eff) * 0.5 * dpi_scale))
            total_w = line_w * squeeze if squeeze else line_w
            # bridge the widened gaps so the staff reads as one continuous line
            x0 = int(round(margin * dpi_scale))
            x1 = int(round((margin + total_w) * dpi_scale))
            for r in lines:
                yy = y + int(round(r * scale * squeeze * dpi_scale))
                draw.line((x0, yy, x1, yy), fill=0, width=max(1, int(round(dpi_scale))))
            th = max(1, int(round(line_h_eff * dpi_scale)))
            if add_clef:
                tile = Image.fromarray(clef_tile).resize(
                    (max(1, int(round(lead * dpi_scale))), th), Image.LANCZOS)
                img.paste(tile, (x0, y))
            x = lead
            bar_px = max(1, int(round(args.bar_width_pt * dpi_scale)))
            y_top = y + int(round(lines[0] * scale * squeeze * dpi_scale))
            y_bot = y + int(round(lines[-1] * scale * squeeze * dpi_scale))
            for idx, (it, wp) in enumerate(zip(line_items, widths)):
                bx = x0 + int(round(x * dpi_scale))
                if it["kind"] == "bar":
                    piece = ink[:, it["a"]:it["b"]]
                    tw = max(1, int(round(wp * dpi_scale)))
                    pil = Image.fromarray(piece).resize((tw, th), Image.LANCZOS)
                    img.paste(pil, (bx, y))
                    if it.get("rep", 1) > 1:
                        font = _font(max(10, int(round(2.0 * space * scale * squeeze
                                                       * dpi_scale))))
                        num_y = y_top - int(round(0.9 * space * scale * squeeze * dpi_scale))
                        draw.text((bx + tw // 2, num_y), f"\u00d7{it['rep']}",
                                  font=font, fill=0, anchor="ms")
                else:
                    # multi-measure rest: a thin bar hanging under the 4th
                    # staff line with short end serifs, count printed above
                    inset = 0.26 * wp
                    left = bx + int(round(inset * dpi_scale))
                    right = bx + int(round((wp - inset) * dpi_scale))
                    hang = y + int(round(lines[3] * scale * squeeze * dpi_scale))
                    thick = max(2, int(round(0.20 * space * scale * squeeze * dpi_scale)))
                    draw.rectangle((left, hang, right, hang + thick), fill=0)
                    serif = max(3, int(round(1.1 * space * scale * squeeze * dpi_scale)))
                    draw.line((left, hang, left, hang + serif), fill=0,
                              width=max(1, thick))
                    draw.line((right, hang, right, hang + serif), fill=0,
                              width=max(1, thick))
                    if it["n"] > 1:
                        font = _font(max(9, int(round(1.9 * space * scale * squeeze
                                                       * dpi_scale))))
                        num_y = y_top - int(round(0.55 * space * scale * squeeze * dpi_scale))
                        draw.text(((left + right) // 2, num_y), str(it["n"]),
                                  font=font, fill=0, anchor="ms")
                if args.bar_width_pt > 0 and plain[idx]:
                    draw.line((bx, y_top, bx, y_bot), fill=0, width=bar_px)
                x += adv[idx] if adv else wp + pad
            if args.bar_width_pt > 0 and plain_end:
                bx = x0 + int(round((x - pad) * dpi_scale))
                draw.line((bx, y_top, bx, y_bot), fill=0, width=bar_px)
        if args.page_numbers:
            draw.text((cw - int(margin * dpi_scale), ch - int(14 * dpi_scale)),
                      str(page_no), fill=120,
                      anchor="ra")
        pages.append(img)
        img.save(f"{args.output}.p{page_no:02d}.png")

    pages[0].save(args.output, "PDF", resolution=args.dpi, save_all=True,
                  append_images=pages[1:])
    print(f"wrote {args.output} ({len(pages)} pages, {page_w:.0f}x{page_h:.0f}pt)")


if __name__ == "__main__":
    main()
