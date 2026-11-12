"""Stitch a horizontally scrolling score band into one long strip.

The band (fixed y range) is assumed to hold one staff line.  Stable segments
are detected first, each segment is rebuilt as a per-column median (which
erases the moving playhead and highlight), and consecutive segments are then
aligned by 1-D normalised cross-correlation on their column profiles.
"""
import json
import sys

import cv2
import numpy as np

STATIC_MAE = 0.8
MIN_SEGMENT_S = 1.2


def band_gray(frame, top, bottom, width=None):
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)[top:bottom]
    if width and width != g.shape[1]:
        g = cv2.resize(g, (width, max(8, int(round(g.shape[0] * width / g.shape[1])))),
                       interpolation=cv2.INTER_AREA)
    return g


def ink_mask(g, thresh=150.0):
    """Notation only.

    The notation of these videos is the only thing near white; the background
    photo and the playhead glow live in the mid greys.  Profiling the mask
    instead of the raw grayscale is what stops an *animated* background from
    vetoing correct shifts (and a rejected link silently loses that stretch of
    music, because the strip only ever advances by accepted shifts).
    """
    return (g > thresh).astype(np.uint8)


def profile(g, thresh=150.0):
    p = ink_mask(g, thresh).sum(axis=0).astype(np.float32)
    k = 9
    pad = np.pad(p, (k // 2, k // 2), mode="edge")
    med = np.median(np.lib.stride_tricks.sliding_window_view(pad, k), axis=1)
    return p - med


def best_shift(a, b, max_shift, min_overlap=300):
    best = (-2.0, 0)
    for s in range(-max_shift, max_shift + 1):
        if s >= 0:
            x, y = a[s:], b[: len(b) - s]
        else:
            x, y = a[: len(a) + s], b[-s:]
        if len(x) < min_overlap:
            continue
        nx, ny = np.linalg.norm(x), np.linalg.norm(y)
        if nx == 0 or ny == 0:
            continue
        c = float(np.dot(x, y) / (nx * ny))
        # prefer more overlap when scores tie
        c += 1e-6 * len(x) / 1000.0
        if c > best[0]:
            best = (c, s)
    return best[1], best[0]


def find_segments(video, top, bottom, sample_fps=10.0, verbose=True):
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, int(round(fps / sample_fps)))
    prev = None
    segments = []
    cur = None
    idx = 0
    while idx < total:
        ok = cap.grab()
        if not ok:
            break
        if idx % step == 0:
            ok, frame = cap.retrieve()
            if not ok:
                break
            g = band_gray(frame, top, bottom, 640)
            if prev is None:
                cur = {"start": idx, "end": idx, "frames": [idx]}
                segments.append(cur)
            else:
                mae = float(np.abs(g.astype(np.float32) - prev).mean())
                if mae < STATIC_MAE:
                    cur["end"] = idx
                    cur["frames"].append(idx)
                else:
                    cur = {"start": idx, "end": idx, "frames": [idx]}
                    segments.append(cur)
            prev = g.astype(np.float32)
        idx += 1
    cap.release()
    for s in segments:
        s["t0"] = s["start"] / fps
        s["t1"] = s["end"] / fps
        s["dur"] = s["t1"] - s["t0"]
    if verbose:
        print(f"fps={fps:.2f} frames={total} segments={len(segments)}")
    return segments, fps, total


def segment_image(video, top, bottom, frames, views=9):
    """Per-column median of several frames inside one static segment."""
    cap = cv2.VideoCapture(video)
    picks = np.linspace(frames[0], frames[-1], min(views, max(1, len(frames)))).astype(int)
    imgs = []
    for f in picks:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(f))
        ok, frame = cap.read()
        if not ok:
            continue
        imgs.append(band_gray(frame, top, bottom))
    cap.release()
    if not imgs:
        return None
    stack = np.stack(imgs, axis=0)
    return np.median(stack, axis=0).astype(np.uint8), len(imgs)


def main():
    video = sys.argv[1]
    top, bottom = int(sys.argv[2]), int(sys.argv[3])
    out_prefix = sys.argv[4]
    segments, fps, total = find_segments(video, top, bottom)
    kept = [s for s in segments if s["dur"] >= MIN_SEGMENT_S]
    print(f"kept {len(kept)} stable segments of {len(segments)}")
    for s in kept:
        print(f"  t={s['t0']:7.2f}-{s['t1']:7.2f} dur={s['dur']:5.2f} frames={len(s['frames'])}")

    images, profs = [], []
    for s in kept:
        got = segment_image(video, top, bottom, s["frames"])
        if got is None:
            continue
        img, n = got
        images.append(img)
        profs.append(profile(img))
        s["views"] = n

    height, width = images[0].shape
    # An intro overlay can duplicate the first view without moving it: keep the
    # later, cleaner observation instead of the decorated one.
    if len(images) > 1:
        dx0, _ = best_shift(profs[0], profs[1], width - 200)
        if abs(dx0) <= 5:
            print(f"dropping duplicate first segment (t={kept[0]['t0']:.2f})")
            kept, images, profs = kept[1:], images[1:], profs[1:]
    offsets = [0]
    links = []
    masks = [ink_mask(im) for im in images]
    for i in range(1, len(images)):
        max_shift = width - 200
        dx, corr = best_shift(profs[i - 1], profs[i], max_shift)
        # residual check on the overlap, on masks: a background change is not a
        # reason to throw a correct shift away
        if dx >= 0:
            a, b = masks[i - 1][:, dx:], masks[i][:, : width - dx]
        else:
            a, b = masks[i - 1][:, : width + dx], masks[i][:, -dx:]
        imae = float(np.abs(a.astype(np.float32) - b.astype(np.float32)).mean())
        ok = corr > 0.55 and imae < 0.10
        links.append({"i": i, "dx": int(dx), "corr": round(corr, 3),
                      "overlap_ink_mae": round(imae, 4),
                      "accepted": ok, "t0": kept[i]["t0"]})
        offsets.append(offsets[-1] + (int(dx) if ok else 0))
        print(f"  link {i:3d} t={kept[i]['t0']:7.2f} dx={int(dx):5d} "
              f"corr={corr:5.3f} ink_mae={imae:6.4f} {'ok' if ok else 'REJECT'}")

    total_w = offsets[-1] + width
    canvas = np.full((height, total_w), 255, np.uint8)
    end = 0
    for i, (img, off) in enumerate(zip(images, offsets)):
        if end == 0:
            canvas[:, :width] = img
            end = width
        elif off + width > end:
            fresh = max(0, end - off)
            piece = img[:, fresh:]
            canvas[:, end:off + width] = piece
            end = off + width
    cv2.imwrite(f"{out_prefix}_strip.png", canvas)
    info = {"video": video, "band": [top, bottom], "height": int(height), "width": int(total_w),
            "segments": [{k: s[k] for k in ("t0", "t1", "dur")} for s in kept],
            "links": links, "offsets": offsets}
    json.dump(info, open(f"{out_prefix}_stitch.json", "w"), indent=1)
    print(f"strip {total_w}x{height} -> {out_prefix}_strip.png")


if __name__ == "__main__":
    main()
