import argparse
import json
from pathlib import Path
import sys

# Dimensioni standard
IMG_W, IMG_H = 1920, 1080

def load_roi_json(roi_path: Path):
    """Carica le ROI dal json locale e le converte in pixel assoluti."""
    try:
        data = json.loads(roi_path.read_text(encoding="utf-8"))
        def get_rect(key):
            r = data[key]
            return (r["x"] * IMG_W, r["y"] * IMG_H, r["width"] * IMG_W, r["height"] * IMG_H)
        return get_rect("roi1"), get_rect("roi2")
    except Exception as e:
        print(f"Errore leggendo {roi_path}: {e}")
        return None, None

def point_in_rect(px, py, rect):
    x, y, w, h = rect
    return (x <= px <= x + w) and (y <= py <= y + h)

def generate_gt_behavior(video_folder: Path):
    gt_track_file = video_folder / "gt" / "gt.txt"
    roi_file = video_folder / "roi.json"
    
    # Output: salviamo behavior_gt.txt nella cartella gt
    output_file = video_folder / "gt" / "behavior_gt.txt"

    if not gt_track_file.exists():
        return
    
    if not roi_file.exists():
        print(f"[SKIP] {video_folder.name}: Manca roi.json! Esegui prima distribute_roi.py")
        return

    # 1. Carica ROI
    roi1, roi2 = load_roi_json(roi_file)
    if roi1 is None: return

    # 2. Leggi GT Tracking
    per_frame_counts = {}
    
    with gt_track_file.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 6: continue
            
            try:
                frame = int(parts[0])
                # Parsing coordinate (MOT format: frame, id, left, top, width, height, ...)
                x, y, w, h = map(float, parts[2:6])
                
                # Calcolo punto di interesse (piedi = y + h, centro x = x + w/2)
                cx = x + w / 2.0
                cy = y + h

                if frame not in per_frame_counts:
                    per_frame_counts[frame] = {"roi1": 0, "roi2": 0}

                # Verifica conteggio
                if point_in_rect(cx, cy, roi1):
                    per_frame_counts[frame]["roi1"] += 1
                if point_in_rect(cx, cy, roi2):
                    per_frame_counts[frame]["roi2"] += 1
            except ValueError:
                continue

    # 3. Scrivi Behavior GT
    if not per_frame_counts:
        print(f"{video_folder.name}: GT letto ma nessun dato valido trovato.")
        return

    sorted_frames = sorted(per_frame_counts.keys())
    
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with output_file.open("w", encoding="utf-8") as out:
        for fr in sorted_frames:
            c1 = per_frame_counts[fr]["roi1"]
            c2 = per_frame_counts[fr]["roi2"]
            out.write(f"{fr},1,{c1}\n")
            out.write(f"{fr},2,{c2}\n")

    print(f"{video_folder.name}: Generato {output_file.name}")

def main():
    default_path = "SIMULATOR/lecture_example_from_training/test_set/videos"
    
    parser = argparse.ArgumentParser(description="Genera Ground Truth per il Behavior partendo dal Tracking GT.")
    parser.add_argument("--source", default=default_path, help="Percorso alla cartella 'videos' (padre delle cartelle numerate)")
    args = parser.parse_args()

    videos_path = Path(args.source)
    if not videos_path.exists():
        print(f"Errore: Il percorso '{videos_path}' non esiste.")
        return

    # Trova sottocartelle numerate
    subfolders = sorted([f for f in videos_path.iterdir() if f.is_dir() and f.name.isdigit()], key=lambda x: int(x.name))
    
    print(f"Trovate {len(subfolders)} cartelle video in: {videos_path}")
    print("-" * 50)
    
    for vf in subfolders:
        generate_gt_behavior(vf)
    
    print("-" * 50)
    print("Completato.")

if __name__ == "__main__":
    main()