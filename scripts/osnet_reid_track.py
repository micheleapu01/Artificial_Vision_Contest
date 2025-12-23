# scripts/track_osnet.py
import argparse
from pathlib import Path
import cv2
import numpy as np
import torch
from tqdm import tqdm
from ultralytics import YOLO


# -----------------------------
# OSNet ReID Monkey-Patch (Ultralytics BoT-SORT)
# -----------------------------
from torchreid.utils import FeatureExtractor


def _to_xyxy_candidates(x, y, w, h):
    """
    Restituisce 2 interpretazioni:
    1) xywh con (x,y) = centro
    2) tlwh con (x,y) = top-left
    """
    # center-xywh
    c_x1 = x - w / 2.0
    c_y1 = y - h / 2.0
    c_x2 = x + w / 2.0
    c_y2 = y + h / 2.0

    # tlwh
    t_x1 = x
    t_y1 = y
    t_x2 = x + w
    t_y2 = y + h

    return (c_x1, c_y1, c_x2, c_y2), (t_x1, t_y1, t_x2, t_y2)


def _clip_xyxy(x1, y1, x2, y2, W, H):
    x1 = float(max(0, min(W - 1, x1)))
    y1 = float(max(0, min(H - 1, y1)))
    x2 = float(max(0, min(W - 1, x2)))
    y2 = float(max(0, min(H - 1, y2)))
    return x1, y1, x2, y2


def _area_xyxy(x1, y1, x2, y2):
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


class ReID_OSNet:
    """
    Replacement della classe ultralytics.trackers.bot_sort.ReID.
    Ultralytics chiamerà: encoder(img, dets) -> lista di embedding (uno per det).
    """

    def __init__(self, _model_str: str, weights_path: str, model_name: str, device: str, min_h_for_reid: int = 0):
        self.device = device
        self.min_h_for_reid = int(min_h_for_reid)

        self.extractor = FeatureExtractor(
            model_name=model_name,
            model_path=weights_path,
            device=device,
        )

    def __call__(self, img: np.ndarray, dets: np.ndarray):
        if dets is None or len(dets) == 0:
            return []

        H, W = img.shape[:2]
        dets = np.asarray(dets)
        boxes = dets[:, :4].astype(np.float32)

        crops = []
        for x, y, w, h in boxes:
            cand1, cand2 = _to_xyxy_candidates(x, y, w, h)

            x1a, y1a, x2a, y2a = _clip_xyxy(*cand1, W, H)
            x1b, y1b, x2b, y2b = _clip_xyxy(*cand2, W, H)

            area_a = _area_xyxy(x1a, y1a, x2a, y2a)
            area_b = _area_xyxy(x1b, y1b, x2b, y2b)
            x1, y1, x2, y2 = (x1a, y1a, x2a, y2a) if area_a >= area_b else (x1b, y1b, x2b, y2b)

            # Se vuoi ignorare ReID su box troppo piccoli (consigliato)
            if self.min_h_for_reid > 0 and (y2 - y1) < self.min_h_for_reid:
                crops.append(np.zeros((256, 128, 3), dtype=np.uint8))
                continue

            if _area_xyxy(x1, y1, x2, y2) < 4.0:
                crops.append(np.zeros((256, 128, 3), dtype=np.uint8))
                continue

            crop = img[int(y1):int(y2), int(x1):int(x2)]
            if crop.size == 0:
                crops.append(np.zeros((256, 128, 3), dtype=np.uint8))
                continue

            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            crops.append(crop_rgb)

        feats = self.extractor(crops)  # torch.Tensor (N, D)
        return [f.detach().cpu().numpy() for f in feats]


def patch_ultralytics_botsort_osnet(weights_path: str, model_name: str, device: str, min_h_for_reid: int = 0):
    """
    Monkey-patch: sostituisce ultralytics.trackers.bot_sort.ReID con OSNet.
    Chiamala PRIMA di iniziare il tracking.
    """
    import ultralytics.trackers.bot_sort as bot_sort_module

    def _reid_factory(model_str: str):
        return ReID_OSNet(model_str, weights_path=weights_path, model_name=model_name, device=device, min_h_for_reid=min_h_for_reid)

    bot_sort_module.ReID = _reid_factory


# -----------------------------
# Tracking script (tuo standard + OSNet)
# -----------------------------
def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def process_video(video_path, tracker, out_path, model, device, conf, iou, show, max_det):
    with open(out_path, "w", encoding="utf-8") as f:
        results = model.track(
            source=video_path,
            tracker=tracker,
            classes=[1, 2, 3],
            imgsz=640,
            stream=True,
            persist=True,
            device=device,
            conf=conf,
            iou=iou,
            max_det=max_det,
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
    ap.add_argument("--conf", type=float, default=0.15)
    ap.add_argument("--iou", type=float, default=0.55)

    # OSNet args
    ap.add_argument("--osnet-weights", required=True, help="Path OSNet .pth, e.g. weights/osnet_x0_25_msmt17.pth")
    ap.add_argument("--osnet-name", default="osnet_x0_25", help="TorchReID model name: osnet_x0_25 or osnet_x1_0")
    ap.add_argument("--reid-min-h", type=int, default=0, help="If >0, ignore ReID on boxes with height < this (px).")

    ap.add_argument("--max-det", type=int, default=300, help="Max detections per frame (reduce NMS spikes).")

    ap.add_argument("--show", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="Process only first N videos (default: all)")

    args = ap.parse_args()

    device = pick_device()
    print(f"[INFO] device = {device}")

    # Patch ReID BEFORE creating/loading detector model
    patch_ultralytics_botsort_osnet(
        weights_path=args.osnet_weights,
        model_name=args.osnet_name,
        device=device,
        min_h_for_reid=args.reid_min_h,
    )
    print(f"[INFO] OSNet ReID enabled: name={args.osnet_name}, weights={args.osnet_weights}")

    # Detector (unchanged)
    model = YOLO(args.weights)

    out_base_path = Path(args.out)
    out_base_path.mkdir(parents=True, exist_ok=True)

    test_folder = Path(args.source)
    video_folders = sorted([p for p in test_folder.iterdir() if p.is_dir() and p.name.isdigit()], key=lambda p: int(p.name))

    if args.limit is not None:
        video_folders = video_folders[:args.limit]

    print(f"[INFO] Processing {len(video_folders)} videos (limit={args.limit if args.limit is not None else 'ALL'})")

    for video_folder in video_folders:
        vid = int(video_folder.name)
        video_path = video_folder / "img1"
        print(f"[INFO] Predicting from folder: {video_path}")

        output_txt_path = out_base_path / f"tracking_{vid}_12.txt"
        process_video(
            video_path=video_path,
            tracker=args.tracker,
            out_path=output_txt_path,
            model=model,
            device=device,
            conf=args.conf,
            iou=args.iou,
            show=args.show,
            max_det=args.max_det,
        )


if __name__ == "__main__":
    main()
