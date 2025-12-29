import os
import re
import glob
import argparse
import numpy as np
import pandas as pd
import cv2


def load_mot_txt(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, header=None, sep=",")
        if df.shape[1] == 1:
            df = pd.read_csv(path, header=None, sep=r"\s+", engine="python")
    except pd.errors.EmptyDataError:
        return pd.DataFrame()

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
    Strategia "UNIVERSAL / BLIND RUN":
    1. HSV Range molto largo (passato da args) per catturare ombre e luci forti.
    2. Excess Green (ExG): Filtro fisico (2G > R+B) per distinguere erba da cemento
       anche quando la saturazione è bassa.
    3. Largest Component: Prende solo la massa verde più grande.
    """
    h0, w0 = bgr.shape[:2]
    if scale != 1.0:
        img = cv2.resize(bgr, (int(w0 * scale), int(h0 * scale)), interpolation=cv2.INTER_AREA)
    else:
        img = bgr

    hs, ws = img.shape[:2]
    total_pixels = float(hs * ws)

    # --- 1) HSV Range (Permissivo) ---
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask_hsv = cv2.inRange(hsv, np.array(hsv_low, dtype=np.uint8), np.array(hsv_high, dtype=np.uint8))

    # --- 2) Excess Green (Refinement) ---
    # Formula: ExG = 2*G - R - B. 
    # Se > 0 (o soglia bassa), il verde domina. 
    # Il cemento ha R~G~B, quindi 2G-R-B ~ 0. L'erba ha G alto.
    # Questo ci salva se il range HSV include grigi o bianchi.
    
    # Separiamo i canali (img è BGR in OpenCV)
    B, G, R = cv2.split(img)
    
    # Calcolo vettorizzato veloce usando float per evitare overflow/underflow
    # Usiamo una soglia bassa (es. 10) per non perdere erba in ombra scura
    exg_mask = ((2.0 * G) > (R.astype(float) + B.astype(float) + 5.0)).astype(np.uint8) * 255
    
    # Combina HSV e ExG
    mask = cv2.bitwise_and(mask_hsv, exg_mask)

    # --- 3) Pulizia Morfologica ---
    if k_open > 1:
        # Rimuove rumore (coriandoli, linee sottili spurie)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_open, k_open))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    
    if k_close > 1:
        # Unisce le zolle d'erba separate da linee bianche o gambe
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_close, k_close))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

    # --- 4) Largest Component (Il Campo) ---
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None

    # Ordina per area
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    
    best_cnt = None
    for c in cnts[:3]:
        area = cv2.contourArea(c)
        ratio = area / max(total_pixels, 1.0)
        if ratio < area_min_ratio: continue
        if ratio > area_max_ratio: continue
        best_cnt = c
        break

    if best_cnt is None:
        return None

    # Disegna il campo pieno
    field_mask = np.zeros_like(mask)
    cv2.drawContours(field_mask, [best_cnt], -1, 255, thickness=cv2.FILLED)

    if k_dilate > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_dilate, k_dilate))
        field_mask = cv2.dilate(field_mask, k, iterations=1)

    return field_mask


def keep_by_field(
    bgr: np.ndarray,
    xyxy: np.ndarray,
    scale: float,
    field_mask: np.ndarray,
    line_tol_px: int,
    bottom_strip_frac: float = 0.20,
    min_strip_overlap: float = 0.05,
) -> np.ndarray:
    """
    Logica robusta "Inside or Touch":
    1. Controlla 3 punti (sx, centro, dx) sul lato inferiore del box.
    2. Fallback: Se vicini ma non dentro, controlla overlap striscia inferiore.
    """
    h0, w0 = bgr.shape[:2]
    hs, ws = field_mask.shape[:2]
    sx = ws / max(w0, 1)
    sy = hs / max(h0, 1)

    # Distance Transform solo se necessario (più lento ma preciso per 'line_tol')
    if line_tol_px > 0:
        non_field = (field_mask == 0).astype(np.uint8)
        dt = cv2.distanceTransform(non_field, cv2.DIST_L2, 3)
    else:
        dt = None

    keep = np.zeros((len(xyxy),), dtype=bool)

    for i, (x1, y1, x2, y2) in enumerate(xyxy):
        # Clip
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w0 - 1, x2), min(h0 - 1, y2)

        # A) Multi-point check sul fondo (piedi)
        fy = y2 - 1.0 
        # Check su 25%, 50%, 75% della larghezza
        xs_check = [x1 + 0.25*(x2-x1), x1 + 0.50*(x2-x1), x1 + 0.75*(x2-x1)]

        is_inside = False
        is_close = False

        for fx in xs_check:
            mx = int(fx * sx)
            my = int(fy * sy)
            
            # Bound check sulla mask
            mx = min(max(mx, 0), ws - 1)
            my = min(max(my, 0), hs - 1)

            if field_mask[my, mx] > 0:
                is_inside = True
                break 
            
            if dt is not None and dt[my, mx] <= line_tol_px:
                is_close = True

        if is_inside:
            keep[i] = True
            continue

        # B) Fallback: Strip Overlap
        # Se i punti esatti falliscono (es. piede alzato o scarpa bianca su linea),
        # guardiamo se la parte bassa del box è "abbastanza verde".
        if is_close or True: # Fallback sempre attivo per sicurezza
            x1m = int(x1 * sx)
            x2m = int(x2 * sx)
            y1m = int(y1 * sy)
            y2m = int(y2 * sy)
            
            # Validità box scalato
            if x2m > x1m and y2m > y1m:
                h_box = y2m - y1m
                # Ultimo 20% del box
                y_start = int(y2m - max(1, h_box * bottom_strip_frac))
                strip = field_mask[y_start:y2m, x1m:x2m]
                
                if strip.size > 0:
                    overlap = np.count_nonzero(strip) / strip.size
                    if overlap >= min_strip_overlap:
                        keep[i] = True

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
        print(f" SKIP {base}: pattern tracking_<id> non trovato.")
        return

    img1_dir = os.path.join(videos_root, vid, "img1")
    if not os.path.isdir(img1_dir):
        print(f" SKIP {base}: dir {img1_dir} non trovata.")
        return

    df = load_mot_txt(pred_path)
    if df.empty:
        print(f" SKIP {base}: file vuoto.")
        return

    df = df.sort_values(["frame", "id"])
    kept_rows = []
    removed = 0

    last_mask = None
    last_mask_frame = -999
    last_shape = None

    for frame_idx, group in df.groupby("frame", sort=True):
        frame_idx = int(frame_idx)
        
        # Logica cache maschera
        recompute = True
        if last_mask is not None and (frame_idx - last_mask_frame) < mask_every:
             recompute = False

        img = None
        if recompute:
            img_path = frame_path(img1_dir, frame_idx)
            img = cv2.imread(img_path)
            
            if img is None:
                # Se manca frame, teniamo i dati (safe)
                kept_rows.append(group)
                continue
            
            if last_shape and img.shape[:2] != last_shape:
                last_mask = None # Reset se cambia risoluzione
            
            last_shape = img.shape[:2]
            
            field_mask = build_field_mask(
                img, scale, hsv_low, hsv_high,
                k_close, k_open, k_dilate,
                area_min_ratio, area_max_ratio
            )
            last_mask = field_mask
            last_mask_frame = frame_idx
        else:
            field_mask = last_mask
            # Se non ricomputiamo, img serve solo per keep_by_field (che usa le dimensioni)
            # Possiamo caricare un placeholder o usare le dimensioni salvate, 
            # ma per semplicità ricarichiamo l'img solo se img è None e serve.
            # In realtà keep_by_field usa img.shape. Possiamo ottimizzare:
            if img is None and last_shape is not None:
                # Creiamo dummy array per passare shape senza I/O disco (ottimizzazione)
                img = np.zeros((last_shape[0], last_shape[1], 3), dtype=np.uint8) 
            elif img is None:
                img = cv2.imread(frame_path(img1_dir, frame_idx))
                if img is None: 
                    kept_rows.append(group)
                    continue

        if field_mask is None:
            # Fallback safe: se non trovo il campo, tengo tutto
            kept_rows.append(group)
            continue

        xyxy = group[["x", "y", "w", "h"]].to_numpy()
        xyxy[:, 2] += xyxy[:, 0] # w -> x2
        xyxy[:, 3] += xyxy[:, 1] # h -> y2
        
        keep = keep_by_field(img, xyxy, scale, field_mask, line_tol_px)

        if conf_keep_outside > 0:
            conf = group["conf"].to_numpy()
            keep = keep | ((~keep) & (conf >= conf_keep_outside))

        removed += int((~keep).sum())
        kept_rows.append(group.loc[keep])

    if kept_rows:
        out = pd.concat(kept_rows, axis=0).sort_values(["frame", "id"])
        # Arrotonda e salva
        out[["x", "y", "w", "h"]] = out[["x", "y", "w", "h"]].round(2)
        tmp = pred_path + ".tmp"
        out.to_csv(tmp, header=False, index=False, sep=",")
        os.replace(tmp, pred_path)
    else:
        open(pred_path, 'w').close()

    print(f" Done {base} | Rimossi: {removed}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-folder", required=True)
    ap.add_argument("--videos-root", required=True)
    ap.add_argument("--scale", type=float, default=0.5)

    # --- SOGLIE "UNIVERSALI" (Blind Run) ---
    # Hue 20-100: Prende dal verde-giallo al verde-bluastro.
    # Sat 25-255: Prende anche colori molto spenti (ombre), ma taglia il grigio puro (<25).
    # Val 20-255: Prende anche ombre scure.
    # Il vero filtro è "ExG" all'interno del codice.
    ap.add_argument("--hsv-low", type=str, default="20,25,20")
    ap.add_argument("--hsv-high", type=str, default="100,255,255")

    # Morfologia Aggressiva per unire tutto
    ap.add_argument("--close", type=int, default=35) # Molto alto per chiudere linee grandi
    ap.add_argument("--open", type=int, default=5)
    ap.add_argument("--dilate", type=int, default=5)

    ap.add_argument("--line-tol", type=int, default=15)
    ap.add_argument("--mask-every", type=int, default=5)
    ap.add_argument("--area-min", type=float, default=0.10)
    ap.add_argument("--area-max", type=float, default=1.0)
    ap.add_argument("--conf-keep-outside", type=float, default=0.0)

    args = ap.parse_args()
    
    h_low = tuple(int(x) for x in args.hsv_low.split(","))
    h_high = tuple(int(x) for x in args.hsv_high.split(","))

    files = sorted(glob.glob(os.path.join(args.pred_folder, "tracking*.txt")))
    print(f" Trovati {len(files)} file.")

    for p in files:
        filter_one_file(
            p, args.videos_root, args.scale,
            h_low, h_high,
            args.close, args.open, args.dilate,
            args.line_tol, args.area_min, args.area_max,
            args.conf_keep_outside, max(1, args.mask_every)
        )

if __name__ == "__main__":
    main()