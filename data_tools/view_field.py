import os
import argparse
import numpy as np
import cv2


import field_filter as ff


def overlay_mask(img: np.ndarray, mask_small: np.ndarray, alpha: float = 0.35) -> np.ndarray:
    """Overlay della maschera (verde) + contorno (rosso). mask_small è alla scala 'scale'."""
    h, w = img.shape[:2]
    mask = cv2.resize(mask_small, (w, h), interpolation=cv2.INTER_NEAREST)

    out = img.copy()
    tint = np.zeros_like(out)
    tint[:, :, 1] = 255  # verde (canale G)

    m = mask > 0
    out[m] = (out[m] * (1 - alpha) + tint[m] * alpha).astype(np.uint8)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        c = max(cnts, key=cv2.contourArea)
        cv2.drawContours(out, [c], -1, (0, 0, 255), 2)  # bordo rosso

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracking-file", required=True, help="Predictions_folder/tracking_<id>.txt")
    ap.add_argument("--videos-root", required=True, help=".../test_set/videos (cartella che contiene le folder 1,2,3...)")
    ap.add_argument("--out", default="debug_field.mp4")
    ap.add_argument("--fps", type=float, default=25.0)

    # stessi parametri del filtro
    ap.add_argument("--scale", type=float, default=0.5)
    ap.add_argument("--hsv-low", type=str, default="25,40,40")
    ap.add_argument("--hsv-high", type=str, default="90,255,255")
    ap.add_argument("--close", type=int, default=11)
    ap.add_argument("--open", type=int, default=7)
    ap.add_argument("--dilate", type=int, default=9)
    ap.add_argument("--line-tol", type=int, default=6)
    ap.add_argument("--mask-every", type=int, default=5)
    ap.add_argument("--area-min", type=float, default=0.08)
    ap.add_argument("--area-max", type=float, default=0.98)
    ap.add_argument("--conf-keep-outside", type=float, default=0.0)

    
    ap.add_argument("--max-frames", type=int, default=0, help="0 = tutti, altrimenti limita i frame per debug veloce")
    args = ap.parse_args()

    hsv_low = tuple(int(x) for x in args.hsv_low.split(","))
    hsv_high = tuple(int(x) for x in args.hsv_high.split(","))

    base = os.path.basename(args.tracking_file)
    vid = ff.extract_video_id_from_filename(base)
    if vid is None:
        raise ValueError(f"Nome tracking non compatibile: {base} (atteso tracking_<id>...)")

    img1_dir = os.path.join(args.videos_root, vid, "img1")
    if not os.path.isdir(img1_dir):
        raise FileNotFoundError(f"Non trovo img1_dir: {img1_dir}")

    df = ff.load_mot_txt(args.tracking_file).sort_values(["frame", "id"])
    if df.empty:
        print("tracking vuoto")
        return

    first_frame = int(df["frame"].min())
    img0 = cv2.imread(ff.frame_path(img1_dir, first_frame))
    if img0 is None:
        raise RuntimeError("Non riesco a leggere il primo frame")

    h, w = img0.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(args.out, fourcc, args.fps, (w, h))

    last_mask = None
    last_mask_frame = None
    last_shape = None

    written = 0

    for frame_idx, group in df.groupby("frame", sort=True):
        frame_idx = int(frame_idx)
        img = cv2.imread(ff.frame_path(img1_dir, frame_idx))
        if img is None:
            continue

        recompute = True
        if last_mask is not None and last_mask_frame is not None and args.mask_every > 1:
            if (frame_idx - last_mask_frame) < args.mask_every and last_shape == img.shape[:2]:
                recompute = False

        if recompute:
            field_mask = ff.build_field_mask(
                img, args.scale, hsv_low, hsv_high,
                args.close, args.open, args.dilate,
                args.area_min, args.area_max
            )
            last_mask = field_mask
            last_mask_frame = frame_idx
            last_shape = img.shape[:2]
        else:
            field_mask = last_mask

        vis = img.copy()
        if field_mask is not None:
            vis = overlay_mask(vis, field_mask, alpha=0.35)

        # bbox -> xyxy
        x1 = group["x"].to_numpy(dtype=float)
        y1 = group["y"].to_numpy(dtype=float)
        x2 = x1 + group["w"].to_numpy(dtype=float)
        y2 = y1 + group["h"].to_numpy(dtype=float)
        xyxy = np.stack([x1, y1, x2, y2], axis=1)

        if field_mask is not None:
            keep = ff.keep_by_field(img, xyxy, args.scale, field_mask, args.line_tol)
            if args.conf_keep_outside > 0:
                conf = group["conf"].to_numpy(dtype=float)
                keep = keep | ((~keep) & (conf >= args.conf_keep_outside))
        else:
            keep = np.ones((len(xyxy),), dtype=bool)

        # disegna bbox + footpoint
        for (bb, k) in zip(xyxy, keep):
            x1, y1, x2, y2 = bb
            x1i, y1i, x2i, y2i = map(int, [x1, y1, x2, y2])
            color = (0, 255, 0) if k else (0, 0, 255)
            cv2.rectangle(vis, (x1i, y1i), (x2i, y2i), color, 2)

            fx = int((x1 + x2) * 0.5)
            fy = int(y2 - 2.0)
            cv2.circle(vis, (fx, fy), 3, color, -1)

        cv2.putText(vis, f"video {vid} | frame {frame_idx}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

        vw.write(vis)
        written += 1

        if args.max_frames > 0 and written >= args.max_frames:
            break

    vw.release()
    print(f"Creato: {args.out} (frame scritti: {written})")


if __name__ == "__main__":
    main()
