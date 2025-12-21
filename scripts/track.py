import argparse
from pathlib import Path
import cv2
import numpy as np
import torch
from tqdm import tqdm
from ultralytics import YOLO
import os

def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def process_video(video_path, tracker, out_path, model, device, conf, iou, show, win_w, win_h):
    with open(out_path, "w", encoding="utf-8") as f:
        results = model.track(
            source=video_path,
            tracker=tracker,
            classes=[1, 2, 3],  # player, goalkeeper, referee
            imgsz=640,
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
    ap.add_argument("--weights", default="weights/yolov8m-640-football-players.pt")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.7)

    ap.add_argument("--show", action="store_true")
    ap.add_argument("--win-w", type=int, default=960)
    ap.add_argument("--win-h", type=int, default=540)

    args = ap.parse_args()

    device = pick_device()
    print(f"[INFO] device = {device}")

    model = YOLO(args.weights)

    out_base_path = Path(args.out)
    out_base_path.mkdir(parents=True, exist_ok=True)

    test_folder = Path(args.source)

    
    video_folders = sorted(
        [p for p in test_folder.iterdir() if p.is_dir() and p.name.isdigit()],
        key=lambda p: int(p.name)
    )

    for video_folder in video_folders:
        vid = int(video_folder.name)          
        video_path = video_folder / "img1"

        
        print(f"[INFO] Predicting from folder: {video_path}")

        output_txt_path = out_base_path / f"tracking_{vid}_12.txt"
        process_video(video_path, args.tracker, output_txt_path, model, device,
                      args.conf, args.iou, args.show, args.win_w, args.win_h)

if __name__ == "__main__":
    main()
