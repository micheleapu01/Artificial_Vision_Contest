import argparse
from pathlib import Path
import cv2
import numpy as np
import torch
from tqdm import tqdm
from ultralytics import YOLO

def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def _harmonic_mean_cost(d1: np.ndarray, d2: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    # d1,d2 distanze in [0,1] (0=match perfetto). Evitiamo divisioni per zero.
    d1 = np.clip(d1, eps, 1.0)
    d2 = np.clip(d2, eps, 1.0)
    return 2.0 / (1.0 / d1 + 1.0 / d2)

def enable_hmsort_like_patch():
    """
    Sostituisce la fusione dei costi in BoT-SORT:
    dists = min(iou_dist, emb_dist)  -->  dists = harmonic_mean(iou_dist, emb_dist)
    """
    try:
        # Ultralytics (molte versioni)
        from ultralytics.trackers import bot_sort
        from ultralytics.trackers.utils import matching
    except Exception as e:
        raise RuntimeError(
            "Non riesco a importare i moduli tracker di Ultralytics. "
            "Probabile mismatch di versione. Errore: " + str(e)
        )

    def get_dists_hm(self, tracks, detections):
        # 1) distanza spaziale (IoU distance standard di Ultralytics)
        dists = matching.iou_distance(tracks, detections)
        dists_mask = dists > (1 - self.proximity_thresh)

        if getattr(self.args, "fuse_score", False):
            dists = matching.fuse_score(dists, detections)

        # 2) distanza appearance (ReID)
        if getattr(self.args, "with_reid", False) and getattr(self, "encoder", None) is not None:
            emb_dists = matching.embedding_distance(tracks, detections) / 2.0  # porta in [0,1] circa
            emb_dists[emb_dists > (1 - self.appearance_thresh)] = 1.0
            emb_dists[dists_mask] = 1.0

            # >>> HM-SORT-like: media armonica invece del min()
            dists = _harmonic_mean_cost(dists, emb_dists)

        return dists

    bot_sort.BOTSORT.get_dists = get_dists_hm
    print("[OK] HM-SORT-like patch attiva: harmonic mean fusion in BOTSORT.get_dists()")

def process_video(video_path, tracker, out_path, model, device, conf, iou, show, win_w, win_h):
    with open(out_path, "w", encoding="utf-8") as f:
        results = model.track(
            source=video_path,
            tracker=tracker,
            imgsz=1280,
            stream=True,
            persist=True,
            device=device,
            conf=conf,
            iou=iou,
            verbose=False,
        )

        pbar = tqdm(total=750, desc="Tracking frames", unit="frame")
        frame_idx = 0

        for r in results:
            frame_idx += 1

            if r.boxes is not None and r.boxes.id is not None and len(r.boxes) > 0:
                xyxy = r.boxes.xyxy.cpu().numpy()
                tids = r.boxes.id.cpu().numpy().astype(int)
                confs = r.boxes.conf.cpu().numpy() if r.boxes.conf is not None else np.ones(len(tids))

                for (x1, y1, x2, y2), tid, c in zip(xyxy, tids, confs):
                    w = x2 - x1
                    h = y2 - y1
                    f.write(f"{frame_idx},{tid},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},{c:.5f},-1,-1,-1\n")

            if show:
                frame_vis = r.plot()
                cv2.imshow("Tracking", frame_vis)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("[INFO] Stopped by user (q).")
                    break

            pbar.update(1)
            if frame_idx >= 750:
                break

        pbar.close()

    if show:
        cv2.destroyAllWindows()

    print(f"[OK] wrote: {out_path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="Path to test folder")
    ap.add_argument("--tracker", required=True, help="Path YAML tracker")
    ap.add_argument("--out", required=True, help="Output folder for txt files")
    ap.add_argument("--weights", default="weights/train_yolo11m_SoccerNet.pt")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.7)

    ap.add_argument("--show", action="store_true")
    ap.add_argument("--win-w", type=int, default=960)
    ap.add_argument("--win-h", type=int, default=540)

    
    ap.add_argument("--limit", type=int, default=None, help="Process only first N videos (default: all)")

    args = ap.parse_args()

    device = pick_device()
    print(f"[INFO] device = {device}")

    model = YOLO(args.weights)
    enable_hmsort_like_patch()
    out_base_path = Path(args.out)
    out_base_path.mkdir(parents=True, exist_ok=True)

    test_folder = Path(args.source)

    video_folders = sorted(
        [p for p in test_folder.iterdir() if p.is_dir() and p.name.isdigit()],
        key=lambda p: int(p.name)
    )

    
    if args.limit is not None:
        video_folders = video_folders[:args.limit]

    print(f"[INFO] Processing {len(video_folders)} videos (limit={args.limit if args.limit is not None else 'ALL'})")

    for video_folder in video_folders:
        vid = int(video_folder.name)
        video_path = video_folder / "img1"

        print(f"[INFO] Predicting from folder: {video_path}")

        output_txt_path = out_base_path / f"tracking_{vid}_12.txt"
        process_video(
            video_path, args.tracker, output_txt_path, model, device,
            args.conf, args.iou, args.show, args.win_w, args.win_h
        )

if __name__ == "__main__":
    main()

