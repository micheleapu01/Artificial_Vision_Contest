import argparse
import json
from pathlib import Path
import re
import statistics

IMG_W, IMG_H = 1920, 1080

def pick_video_id(folder: Path, fallback_idx: int) -> int:
    if folder.name.isdigit():
        return int(folder.name)
    m = re.search(r"(\d+)", folder.name)
    return int(m.group(1)) if m else (fallback_idx + 1)

def load_roi_json(roi_path: Path):
    data = json.loads(roi_path.read_text(encoding="utf-8"))

    def read_roi(key: str):
        r = data[key]
        return float(r["x"]), float(r["y"]), float(r["width"]), float(r["height"])

    return read_roi("roi1"), read_roi("roi2")

def read_tracking(track_path: Path):
    per_frame = {}
    with track_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.strip():
                continue
            parts = line.strip().split(",")
            if len(parts) < 6:
                continue
            fr = int(float(parts[0]))
            x = float(parts[2]); y = float(parts[3])
            w = float(parts[4]); h = float(parts[5])
            per_frame.setdefault(fr, []).append((x, y, w, h))
    return per_frame

def point_in_rect(px, py, rect):
    x, y, w, h = rect
    return (x <= px <= x + w) and (y <= py <= y + h)

def apply_median_filter(data_list, window_size):
    """
    Applica un filtro mediano a una lista di numeri interi.
    La finestra scorre sui dati e calcola la mediana locale.
    """
    if window_size < 3:
        return data_list
    
    # Assicuriamoci che la finestra sia dispari
    if window_size % 2 == 0:
        window_size += 1
    
    half_w = window_size // 2
    filtered = []
    length = len(data_list)
    
    for i in range(length):
        # Definisce i bordi della finestra (gestisce l'inizio e la fine della lista)
        start = max(0, i - half_w)
        end = min(length, i + half_w + 1)
        
        chunk = data_list[start:end]
        # Calcola la mediana e la forza a intero
        # (median_high o int(median) va bene per i conteggi)
        med_val = int(statistics.median(chunk))
        filtered.append(med_val)
        
    return filtered

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="Test folder containing video folders")
    ap.add_argument("--tracking-dir", required=True, help="Folder with tracking_K_XX.txt files")
    ap.add_argument("--out", required=True, help="Output folder for behavior_K_XX.txt")
    ap.add_argument("--team", required=True, help="Team id with 2 digits, e.g. 01, 12")
    ap.add_argument("--roi-name", default="roi.json", help="ROI filename inside each video folder")
    ap.add_argument("--frames", type=int, default=750)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--window", type=int, default=5, help="Window size for median filter (must be odd, e.g. 5)")
    args = ap.parse_args()

    test_folder = Path(args.source)
    tracking_dir = Path(args.tracking_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    video_folders = sorted([p for p in test_folder.iterdir() if p.is_dir()])
    if args.limit is not None:
        video_folders = video_folders[:args.limit]

    for idx, vf in enumerate(video_folders):
        K = pick_video_id(vf, idx)

        roi_file = vf / args.roi_name
        if not roi_file.exists():
            print(f"[WARN] Missing ROI file in {vf}, skipping...")
            continue

        r1_rel, r2_rel = load_roi_json(roi_file)

        roi1 = (r1_rel[0] * IMG_W, r1_rel[1] * IMG_H, r1_rel[2] * IMG_W, r1_rel[3] * IMG_H)
        roi2 = (r2_rel[0] * IMG_W, r2_rel[1] * IMG_H, r2_rel[2] * IMG_W, r2_rel[3] * IMG_H)

        track_path = tracking_dir / f"tracking_{K}_{args.team}.txt"
        if not track_path.exists():
            print(f"[WARN] Missing tracking file: {track_path}, skipping...")
            continue

        per_frame = read_tracking(track_path)

        # 1. Raccogliamo i dati grezzi in liste
        raw_counts_1 = []
        raw_counts_2 = []

        for fr in range(1, args.frames + 1):
            boxes = per_frame.get(fr, [])
            c1 = 0
            c2 = 0
            for (x, y, w, h) in boxes:
                cx = x + w / 2.0
                cy = y + h
                if point_in_rect(cx, cy, roi1):
                    c1 += 1
                if point_in_rect(cx, cy, roi2):
                    c2 += 1
            
            raw_counts_1.append(c1)
            raw_counts_2.append(c2)

        # 2. Applichiamo il Median Filter alle liste complete
        smooth_counts_1 = apply_median_filter(raw_counts_1, args.window)
        smooth_counts_2 = apply_median_filter(raw_counts_2, args.window)

        # 3. Scriviamo i risultati filtrati
        out_path = out_dir / f"behavior_{K}_{args.team}.txt"
        with out_path.open("w", encoding="utf-8") as f:
            for i in range(args.frames):
                frame_idx = i + 1
                # Recuperiamo il valore filtrato
                val1 = smooth_counts_1[i]
                val2 = smooth_counts_2[i]
                
                f.write(f"{frame_idx},1,{val1}\n")
                f.write(f"{frame_idx},2,{val2}\n")

        print(f"[OK] {out_path} (Filter window: {args.window})")

if __name__ == "__main__":
    main()