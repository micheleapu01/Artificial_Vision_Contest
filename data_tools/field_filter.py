import os
import re
import glob
import argparse
import numpy as np
import pandas as pd
import cv2


def load_mot_txt(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, header=None, sep=",")
    if df.shape[1] == 1:
        df = pd.read_csv(path, header=None, sep=r"\s+", engine="python")

    if df.shape[1] < 10:
        for i in range(df.shape[1], 10):
            df[i] = -1

    df = df.iloc[:, :10]
    df.columns = ["frame", "id", "x", "y", "w", "h", "conf", "x3d", "y3d", "z3d"]
    return df


def frame_path(img1_dir: str, frame_idx: int) -> str:
    return os.path.join(img1_dir, f"{frame_idx:06d}.jpg")


def extract_video_id_from_filename(fname: str) -> str | None:
    m = re.search(r"^tracking_(\d+)", fname)
    return m.group(1) if m else None


def build_field_mask(
    bgr: np.ndarray,
    scale: float,
    hsv_low: tuple[int, int, int],
    hsv_high: tuple[int, int, int],
    k_close: int,
    k_open: int,
    k_dilate: int,
    area_min_ratio: float,
    area_max_ratio: float,
) -> np.ndarray | None:
    """
    Campo = TUTTE le aree realmente verdi (anche fuori dalle linee bianche),
    escludendo cemento/grigio e falsi verdi.
    Strategia:
      - HSV green (ma con S più severa)
      - + indice ExG (green-dominant) per eliminare grigi/verdi finti
      - scegli la componente che TOCCA il bordo basso (evita cartelloni)
    """
    h0, w0 = bgr.shape[:2]
    if scale != 1.0:
        img = cv2.resize(bgr, (int(w0 * scale), int(h0 * scale)), interpolation=cv2.INTER_AREA)
    else:
        img = bgr

    hs, ws = img.shape[:2]
    total = float(hs * ws)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)

    # ----------------------------
    # 1) HSV GREEN (stringi S per buttare fuori cemento/grigi)
    # ----------------------------
    hL, sL, vL = hsv_low
    hH, sH, vH = hsv_high

    # Critico: non usare sL troppo basso (40 è troppo poco)
    s_min = max(int(sL), 70)   # <- chiave anti-cemento
    v_min = max(int(vL), 35)

    hsv_mask = cv2.inRange(
        hsv,
        np.array([hL, s_min, v_min], np.uint8),
        np.array([hH, 255, 255], np.uint8),
    )

    # ----------------------------
    # 2) ExG (Excess Green) per eliminare falsi verdi/grigi illuminati
    # ExG = 2G - R - B
    # ----------------------------
    b, g, r = cv2.split(img)
    exg = (2 * g.astype(np.int16) - r.astype(np.int16) - b.astype(np.int16))
    exg = np.clip(exg, -255, 255)

    # soglia: abbastanza permissiva da tenere prato in ombra, ma tagliare cemento
    exg_mask = (exg > 20).astype(np.uint8) * 255

    green = cv2.bitwise_and(hsv_mask, exg_mask)

    # ----------------------------
    # 3) Morfologia (NON spalmare sul cemento!)
    # ----------------------------
    if k_open > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_open, k_open))
        green = cv2.morphologyEx(green, cv2.MORPH_OPEN, k)
    if k_close > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_close, k_close))
        green = cv2.morphologyEx(green, cv2.MORPH_CLOSE, k)
    # dilate solo se proprio serve (meglio disattivarlo = 1)
    if k_dilate > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_dilate, k_dilate))
        green = cv2.dilate(green, k, iterations=1)

    green_bin = (green > 0).astype(np.uint8)

    # ----------------------------
    # 4) Connected Components: prendi quella che tocca il bordo basso
    # ----------------------------
    num, lab, stats, _ = cv2.connectedComponentsWithStats(green_bin, 8)
    if num <= 1:
        return None

    bottom_labels = set(lab[hs - 1, :].tolist())
    bottom_labels.discard(0)

    # se per qualche motivo non tocca l'ultimo pixel, guarda poco sopra
    if not bottom_labels:
        yb = int(0.95 * (hs - 1))
        bottom_labels = set(lab[yb, :].tolist())
        bottom_labels.discard(0)

    candidates = bottom_labels if bottom_labels else set(range(1, num))

    best = None
    best_area = -1
    for lid in candidates:
        area = int(stats[lid, cv2.CC_STAT_AREA])
        ratio = area / max(total, 1.0)
        if ratio < area_min_ratio or ratio > area_max_ratio:
            continue
        if area > best_area:
            best_area = area
            best = lid

    if best is None:
        return None

    field = (lab == best).astype(np.uint8) * 255

    # chiudi piccoli buchi
    ksmall = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    field = cv2.morphologyEx(field, cv2.MORPH_CLOSE, ksmall)

    return field


def keep_by_field(
    bgr: np.ndarray,
    xyxy: np.ndarray,
    scale: float,
    field_mask: np.ndarray,
    line_tol_px: int,
    # tuning:
    bottom_strip_frac: float = 0.22,   # usa ultimo ~22% bbox per overlap
    min_strip_overlap: float = 0.06,   # >=6% strip su campo => keep
    dilate_px: int = 3,                # dilata la maschera (pixel su scala ridotta)
) -> np.ndarray:
    """
    Keep robusto:
    - multi-footpoint (3 punti sul fondo bbox)
    - fallback overlap su strip bassa bbox
    - tolleranza con dilatazione leggera della field_mask
    """

    h0, w0 = bgr.shape[:2]
    hs, ws = field_mask.shape[:2]
    sx = ws / max(w0, 1)
    sy = hs / max(h0, 1)

    # 1) dilata un pelo la maschera per non perdere giocatori su linee/bordi/rumore
    if dilate_px and dilate_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*dilate_px+1, 2*dilate_px+1))
        fm = cv2.dilate(field_mask, k, iterations=1)
    else:
        fm = field_mask

    # 2) distance transform sul complementare
    non_field = (fm == 0).astype(np.uint8)
    dt = cv2.distanceTransform(non_field, cv2.DIST_L2, 3)

    keep = np.zeros((len(xyxy),), dtype=bool)

    for i, (x1, y1, x2, y2) in enumerate(xyxy):
        # clamp bbox in immagine
        x1 = float(np.clip(x1, 0, w0 - 1))
        x2 = float(np.clip(x2, 0, w0 - 1))
        y1 = float(np.clip(y1, 0, h0 - 1))
        y2 = float(np.clip(y2, 0, h0 - 1))

        # --- A) multi-footpoint ---
        fy = y2 - 2.0
        xs = [x1 + 0.50*(x2-x1), x1 + 0.25*(x2-x1), x1 + 0.75*(x2-x1)]

        inside_any = False
        near_any = False
        for fx in xs:
            mx = int(np.clip(round(fx * sx), 0, ws - 1))
            my = int(np.clip(round(fy * sy), 0, hs - 1))

            if fm[my, mx] > 0:
                inside_any = True
                break
            if dt[my, mx] <= float(line_tol_px):
                near_any = True

        if inside_any:
            keep[i] = True
            continue

        # --- B) fallback overlap su strip bassa ---
        if near_any:
            x1m = int(np.clip(round(x1 * sx), 0, ws - 1))
            x2m = int(np.clip(round(x2 * sx), 0, ws - 1))
            y1m = int(np.clip(round(y1 * sy), 0, hs - 1))
            y2m = int(np.clip(round(y2 * sy), 0, hs - 1))

            if x2m <= x1m or y2m <= y1m:
                keep[i] = False
                continue

            hbb = max(1, y2m - y1m)
            ys = max(0, y2m - int(bottom_strip_frac * hbb))
            ye = y2m

            strip = fm[ys:ye, x1m:x2m]
            if strip.size == 0:
                keep[i] = False
                continue

            overlap = float(np.mean(strip > 0))
            keep[i] = (overlap >= float(min_strip_overlap))
        else:
            keep[i] = False

    return keep



def filter_one_file(
    pred_path: str,
    videos_root: str,
    scale: float,
    hsv_low: tuple[int, int, int],
    hsv_high: tuple[int, int, int],
    k_close: int,
    k_open: int,
    k_dilate: int,
    line_tol_px: int,
    area_min_ratio: float,
    area_max_ratio: float,
    conf_keep_outside: float,
    mask_every: int,
) -> None:
    base = os.path.basename(pred_path)
    vid = extract_video_id_from_filename(base)
    if vid is None:
        print(f" SKIP {base}: nome file non compatibile (atteso tracking_<id>...).")
        return

    img1_dir = os.path.join(videos_root, vid, "img1")
    if not os.path.isdir(img1_dir):
        print(f" SKIP {base}: non trovo {img1_dir}")
        return

    df = load_mot_txt(pred_path)
    if df.empty:
        print(f" {base} vuoto, skip.")
        return

    df = df.sort_values(["frame", "id"])

    kept_rows = []
    removed = 0

    last_mask = None
    last_mask_frame = None
    last_shape = None

    for frame_idx, group in df.groupby("frame", sort=True):
        frame_idx = int(frame_idx)
        img = cv2.imread(frame_path(img1_dir, frame_idx))
        if img is None:
            kept_rows.append(group)
            continue

        recompute = True
        if last_mask is not None and last_mask_frame is not None and mask_every > 1:
            if (frame_idx - last_mask_frame) < mask_every and last_shape == img.shape[:2]:
                recompute = False

        if recompute:
            field_mask = build_field_mask(
                img, scale, hsv_low, hsv_high,
                k_close, k_open, k_dilate,
                area_min_ratio, area_max_ratio
            )
            last_mask = field_mask
            last_mask_frame = frame_idx
            last_shape = img.shape[:2]
        else:
            field_mask = last_mask

        if field_mask is None:
            kept_rows.append(group)
            continue

        x1 = group["x"].to_numpy(dtype=float)
        y1 = group["y"].to_numpy(dtype=float)
        x2 = x1 + group["w"].to_numpy(dtype=float)
        y2 = y1 + group["h"].to_numpy(dtype=float)
        xyxy = np.stack([x1, y1, x2, y2], axis=1)

        keep = keep_by_field(img, xyxy, scale, field_mask, line_tol_px)

        if conf_keep_outside > 0:
            conf = group["conf"].to_numpy(dtype=float)
            keep = keep | ((~keep) & (conf >= conf_keep_outside))

        removed += int((~keep).sum())
        kept_rows.append(group.loc[keep])

    out = pd.concat(kept_rows, axis=0).sort_values(["frame", "id"])
    out[["x", "y", "w", "h"]] = out[["x", "y", "w", "h"]].round(2)

    tmp = pred_path + ".tmp"
    out.to_csv(tmp, header=False, index=False, sep=",")
    os.replace(tmp, pred_path)

    print(f" Field-filter: {base} | rimossi {removed} bbox fuori campo")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-folder", required=True, help="Cartella con tracking*.txt (Predictions_folder)")
    ap.add_argument("--videos-root", required=True, help="Cartella test_set/videos")

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
    args = ap.parse_args()

    hsv_low = tuple(int(x) for x in args.hsv_low.split(","))
    hsv_high = tuple(int(x) for x in args.hsv_high.split(","))

    pred_files = sorted(glob.glob(os.path.join(args.pred_folder, "tracking*.txt")))
    if not pred_files:
        print(" Nessun tracking*.txt trovato in pred-folder")
        return

    print(f" Trovati {len(pred_files)} file tracking*.txt. (behavior ignorati)")

    for pred_path in pred_files:
        filter_one_file(
            pred_path=pred_path,
            videos_root=args.videos_root,
            scale=args.scale,
            hsv_low=hsv_low,
            hsv_high=hsv_high,
            k_close=args.close,
            k_open=args.open,
            k_dilate=args.dilate,
            line_tol_px=args.line_tol,
            area_min_ratio=args.area_min,
            area_max_ratio=args.area_max,
            conf_keep_outside=args.conf_keep_outside,
            mask_every=max(1, args.mask_every),
        )


if __name__ == "__main__":
    main()
