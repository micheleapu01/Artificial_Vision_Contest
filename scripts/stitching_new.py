import os
import glob
import math
import argparse
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def foot_point(x: float, y: float, w: float, h: float) -> Tuple[float, float]:
    """Restituisce il punto a terra (metà base) della bounding box."""
    return (x + w / 2.0, y + h)


def near_border(x: float, y: float, w: float, h: float, W: int, H: int, margin: int) -> bool:
    """Controlla se la box è vicina al bordo immagine."""
    return min(x, y, W - (x + w), H - (y + h)) < margin


def linreg_predict(frames: List[int], values: List[float], t: int) -> float:
    """Predizione lineare semplice basata sugli ultimi frame."""
    n = len(frames)
    if n <= 1:
        return values[-1]
    mt = sum(frames) / n
    mv = sum(values) / n
    num = sum((frames[i] - mt) * (values[i] - mv) for i in range(n))
    den = sum((frames[i] - mt) ** 2 for i in range(n))
    if den == 0:
        return values[-1]
    a = num / den
    b = mv - a * mt
    return a * t + b


def load_mot_file(file_path: str) -> pd.DataFrame:
    """Carica file standard MOT."""
    try:
        df = pd.read_csv(file_path, header=None, sep=",")
        # Fallback per separatori spazi se necessario
        if df.shape[1] == 1:
            df = pd.read_csv(file_path, header=None, sep=r"\s+", engine="python")
    except pd.errors.EmptyDataError:
        return pd.DataFrame()

    if df.shape[1] < 7:
        # Se il file è vuoto o corrotto, ritorna DF vuoto
        return pd.DataFrame()

    # Riempimento colonne mancanti fino a 10 (standard MOT)
    if df.shape[1] < 10:
        for i in range(df.shape[1], 10):
            df[i] = -1

    df = df.iloc[:, :10]
    df.columns = ["frame", "id", "x", "y", "w", "h", "conf", "x3d", "y3d", "z3d"]

    df["frame"] = df["frame"].astype(int)
    df["id"] = df["id"].astype(int)
    for c in ["x", "y", "w", "h", "conf"]:
        df[c] = df[c].astype(float)

    return df


def save_mot_file_inplace(df: pd.DataFrame, file_path: str) -> None:
    """Salva il file sovrascrivendo l'originale."""
    if df.empty:
        return
    df = df.copy()
    df[["x", "y", "w", "h"]] = df[["x", "y", "w", "h"]].round(2)
    df["conf"] = df["conf"].round(5)
    df = df.sort_values(["frame", "id"])

    tmp = file_path + ".tmp"
    df.to_csv(tmp, header=False, index=False, sep=",")
    os.replace(tmp, file_path)


def build_tracklets(df: pd.DataFrame) -> Dict[int, pd.DataFrame]:
    """Raggruppa le detection in tracklets per ID."""
    tracks = {}
    if df.empty:
        return tracks
    for tid, g in df.groupby("id"):
        # Rimuove duplicati nello stesso frame (prende confidenza maggiore)
        g = g.sort_values(["frame", "conf"], ascending=[True, False]).drop_duplicates("frame", keep="first")
        tracks[int(tid)] = g.sort_values("frame").reset_index(drop=True)
    return tracks


def summarize_tracklets(tracks: Dict[int, pd.DataFrame], W: int, H: int, border_frac: float, K: int):
    """Estrae feature chiave da ogni traccia (inizio, fine, velocità coda)."""
    margin = int(border_frac * min(W, H))
    summ = {}
    starts_by_frame = defaultdict(list)

    for tid, g in tracks.items():
        if g.empty: continue
        
        f_start = int(g.loc[0, "frame"])
        f_end = int(g.loc[len(g) - 1, "frame"])

        x0, y0, w0, h0 = map(float, g.loc[0, ["x", "y", "w", "h"]])
        xe, ye, we, he = map(float, g.loc[len(g) - 1, ["x", "y", "w", "h"]])

        # Analisi della "coda" (ultimi K frame) per predizione
        tail = g.tail(min(K, len(g)))
        tail_frames = tail["frame"].astype(int).tolist()
        tail_pts = [foot_point(*map(float, r)) for r in tail[["x", "y", "w", "h"]].values]
        tail_xs = [p[0] for p in tail_pts]
        tail_ys = [p[1] for p in tail_pts]

        # Altezza mediana (più robusta di h finale)
        hs = tail["h"].astype(float).tolist()
        h_med = float(np.median(hs)) if hs else float(he)
        h_med = max(1.0, h_med)

        start_near = near_border(x0, y0, w0, h0, W, H, margin)
        end_near = near_border(xe, ye, we, he, W, H, margin)

        summ[tid] = {
            "start": f_start,
            "end": f_end,
            "first_bbox": (x0, y0, w0, h0),
            "last_bbox": (xe, ye, we, he),
            "tail_frames": tail_frames,
            "tail_xs": tail_xs,
            "tail_ys": tail_ys,
            "h_med": h_med,
            "len": int(len(g)),
            "start_near": start_near,
            "end_near": end_near,
        }
        starts_by_frame[f_start].append(tid)

    return summ, starts_by_frame


def stitch_tracklets(
    df: pd.DataFrame,
    W: int,
    H: int,
    border_frac: float,
    max_gap: int,
    K: int,
    dist_base: float,
    dist_per_gap: float,
    size_log_gate: float,
    size_weight: float,
    min_len: int,
    allow_border: bool,
):
    tracks = build_tracklets(df)
    if not tracks:
        return df, []
        
    summ, starts_by_frame = summarize_tracklets(tracks, W=W, H=H, border_frac=border_frac, K=K)

    edges: List[Tuple[float, int, int, int]] = []

    # --- MODIFICA 1: Tetto massimo alla soglia di ricerca ---
    # Non cercare mai oltre 3.5 volte l'altezza, anche se il gap è enorme.
    # Questo previene unioni attraverso tutto il campo.
    MAX_THRESH_CAP = 3.5 

    for a, sa in summ.items():
        if sa["len"] < min_len:
            continue
        # Se non permettiamo border merge, salta chi finisce fuori campo
        if (not allow_border) and sa["end_near"]:
            continue

        t_end = sa["end"]
        hA_med = sa["h_med"]

        for gap in range(1, max_gap + 1):
            t_b = t_end + gap
            
            # Cerca tracce che iniziano esattamente al frame t_b
            for b in starts_by_frame.get(t_b, []):
                sb = summ[b]
                if sb["len"] < min_len:
                    continue
                if (not allow_border) and sb["start_near"]:
                    continue

                # --- MODIFICA 2: Gate Dimensionale Adattivo ---
                # Se il gap è piccolo, tolleriamo cambi di dimensione.
                # Se il gap è grande (>10 frame), le dimensioni devono essere MOLTO simili
                # per evitare di unire primo piano con sfondo.
                current_size_gate = size_log_gate
                if gap > 10:
                    current_size_gate *= 0.6  # Diventa più severo del 40%

                # Check dimensione (Altezza)
                hA = sa["last_bbox"][3]
                hB = sb["first_bbox"][3]
                d_size = abs(math.log(max(1e-6, hB) / max(1e-6, hA)))
                
                if d_size > current_size_gate:
                    continue

                # Predizione Movimento (Regressione Lineare)
                x_pred = linreg_predict(sa["tail_frames"], sa["tail_xs"], t_b)
                y_pred = linreg_predict(sa["tail_frames"], sa["tail_ys"], t_b)

                xb, yb, wb, hb = sb["first_bbox"]
                xbf, ybf = foot_point(xb, yb, wb, hb)

                # Distanza normalizzata sull'altezza
                dpos = math.hypot(xbf - x_pred, ybf - y_pred) / hA_med

                # Calcolo soglia dinamica con CAP
                raw_thr = dist_base + dist_per_gap * gap
                thr = min(raw_thr, MAX_THRESH_CAP) # Il tetto massimo di sicurezza

                if dpos > thr:
                    continue

                # Costo dell'arco
                # Aggiungiamo una piccola penalità per il gap temporale (meglio unire subito che dopo)
                gap_penalty = 1.0 + (0.05 * gap)
                cost = (dpos * gap_penalty) + (size_weight * d_size)
                
                edges.append((cost, a, b, gap))

    # Ordina per costo (Greedy Matching)
    edges.sort(key=lambda e: e[0])

    # Unione disgiunta (Union-Find like)
    used_succ = set()
    used_pred = set()
    parent = {tid: tid for tid in summ.keys()}
    links = []

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union_keep_a(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra # B diventa figlio di A (mantiene ID di A)

    for cost, a, b, gap in edges:
        # Se A ha già un successore o B ha già un predecessore, salta
        if a in used_succ or b in used_pred:
            continue
        # Se sono già uniti (cicli), salta
        if find(a) == find(b):
            continue
            
        used_succ.add(a)
        used_pred.add(b)
        union_keep_a(a, b)
        links.append((a, b, gap, cost))

    # Riscrivi gli ID nel DataFrame originale
    out = df.copy()
    out["id"] = out["id"].astype(int).apply(find)
    out = out.sort_values(["frame", "id"])
    return out, links


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--folder", required=True, help="Cartella con predizioni MOT")
    p.add_argument("--w", type=int, default=1920)
    p.add_argument("--h", type=int, default=1080)
    p.add_argument("--pattern", default="tracking*.txt")
    
    # Parametri Default ottimizzati per Calcio
    p.add_argument("--max-gap", type=int, default=30)
    p.add_argument("--min-len", type=int, default=10)
    p.add_argument("--K", type=int, default=8)
    p.add_argument("--border-frac", type=float, default=0.05)
    p.add_argument("--allow-border", action="store_true")

    p.add_argument("--dist-base", type=float, default=1.5) 
    p.add_argument("--dist-per-gap", type=float, default=0.15)
    p.add_argument("--size-log-gate", type=float, default=0.5)
    p.add_argument("--size-weight", type=float, default=0.2)

    args = p.parse_args()

    if not os.path.isdir(args.folder):
        raise SystemExit(f"Cartella non trovata: {args.folder}")

    files = glob.glob(os.path.join(args.folder, args.pattern))
    print(f"--- STITCHING AVANZATO (Soccer Optimized) ---")
    print(f"Target: {len(files)} files in {args.folder}")
    print(f"Params: MaxGap={args.max_gap}, DistBase={args.dist_base}, MinLen={args.min_len}")

    for f in files:
        base = os.path.basename(f)
        df = load_mot_file(f)
        if df.empty:
            print(f"Skipping empty file: {base}")
            continue
            
        n0 = df["id"].nunique()

        stitched, links = stitch_tracklets(
            df,
            W=args.w,
            H=args.h,
            border_frac=args.border_frac,
            max_gap=args.max_gap,
            K=args.K,
            dist_base=args.dist_base,
            dist_per_gap=args.dist_per_gap,
            size_log_gate=args.size_log_gate,
            size_weight=args.size_weight,
            min_len=args.min_len,
            allow_border=args.allow_border,
        )

        n1 = stitched["id"].nunique()
        save_mot_file_inplace(stitched, f)
        print(f"✅ {base}: IDs {n0} -> {n1} | Merged: {len(links)}")


if __name__ == "__main__":
    main()