import argparse
from pathlib import Path
import cv2
import numpy as np
import torch
from tqdm import tqdm
from ultralytics import YOLO
from torchreid.utils import FeatureExtractor


# -----------------------------
# Utils
# -----------------------------
def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def _clip_xyxy(x1, y1, x2, y2, W, H):
    x1 = float(max(0, min(W - 1, x1)))
    y1 = float(max(0, min(H - 1, y1)))
    x2 = float(max(0, min(W - 1, x2)))
    y2 = float(max(0, min(H - 1, y2)))
    return x1, y1, x2, y2

def _xywh_center_to_xyxy(x, y, w, h):
    return (x - w / 2.0, y - h / 2.0, x + w / 2.0, y + h / 2.0)

def _pad_xyxy(x1, y1, x2, y2, pad_frac):
    if pad_frac <= 0:
        return x1, y1, x2, y2
    w = (x2 - x1)
    h = (y2 - y1)
    px = w * pad_frac
    py = h * pad_frac
    return x1 - px, y1 - py, x2 + px, y2 + py

def _harmonic_mean_cost(d1: np.ndarray, d2: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    d1 = np.clip(d1, eps, 1.0)
    d2 = np.clip(d2, eps, 1.0)
    return 2.0 / (1.0 / d1 + 1.0 / d2)


# -----------------------------
# OSNet ReID 
# -----------------------------
class ReID_OSNet:
    """
    Replacement for ultralytics.trackers.bot_sort.ReID
    Ultralytics calls: encoder(img, dets) -> list[np.ndarray], one per det
    dets[:, :4] are xywh with center (x,y).
    """
    def __init__(self, weights_path: str, model_name: str, device: str, min_h: int, pad: float):
        self.device = device
        self.min_h = int(min_h)
        self.pad = float(pad)

        self.extractor = FeatureExtractor(
            model_name=model_name,
            model_path=weights_path,
            device=device,
        )

        # Determina feature dim  
        dummy = np.zeros((256, 128, 3), dtype=np.uint8)
        with torch.no_grad():
            feat = self.extractor([dummy])  # (1, D)
        self.feat_dim = int(feat.shape[1])

    def __call__(self, img: np.ndarray, dets: np.ndarray):
        if dets is None or len(dets) == 0:
            return []

        H, W = img.shape[:2]
        dets = np.asarray(dets)
        boxes_xywh = dets[:, :4].astype(np.float32)

        out = [None] * len(boxes_xywh)   # None => embedding mancante
        crops = []
        valid_idx = []

        for i, (x, y, w, h) in enumerate(boxes_xywh):
            # Skip ReID su box piccoli
            if self.min_h > 0 and h < self.min_h:
                continue

            # xywh(center) -> xyxy
            x1 = x - w / 2.0
            y1 = y - h / 2.0
            x2 = x + w / 2.0
            y2 = y + h / 2.0

            # padding (aiuta reid)
            if self.pad > 0:
                px = (x2 - x1) * self.pad
                py = (y2 - y1) * self.pad
                x1 -= px; y1 -= py; x2 += px; y2 += py

            # clip
            x1, y1, x2, y2 = _clip_xyxy(x1, y1, x2, y2, W, H)

            x1i, y1i, x2i, y2i = map(int, [x1, y1, x2, y2])
            if (x2i - x1i) < 2 or (y2i - y1i) < 2:
                continue

            crop = img[y1i:y2i, x1i:x2i]
            if crop is None or crop.size == 0:
                continue

            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            crops.append(crop_rgb)
            valid_idx.append(i)

        # Estrai feature per crop validi
        if len(crops) > 0:
            feats = self.extractor(crops)  # torch.Tensor (Nv, D)
            feats = feats.detach().cpu().numpy().astype(np.float32)
            for k, i in enumerate(valid_idx):
                out[i] = feats[k]

        # Sostituisci i None con NaN-vector 
        nanvec = np.full((self.feat_dim,), np.nan, dtype=np.float32)
        out = [nanvec.copy() if v is None else v for v in out]

        return out



def patch_ultralytics_botsort_osnet(weights_path: str, model_name: str, device: str, min_h: int, pad: float):
    import ultralytics.trackers.bot_sort as bot_sort_module

    def _reid_factory(_model_str: str):
        return ReID_OSNet(weights_path=weights_path, model_name=model_name, device=device, min_h=min_h, pad=pad)

    bot_sort_module.ReID = _reid_factory
    print("[OK] Patched Ultralytics BoT-SORT ReID -> OSNet (safe).")


# -----------------------------
# Patch BoT-SORT get_dists
# -----------------------------
def enable_botsort_sanitize_patch(use_hm: bool):
    from ultralytics.trackers import bot_sort
    from ultralytics.trackers.utils import matching

    def get_dists_patched(self, tracks, detections):
        dists = matching.iou_distance(tracks, detections)
        dists_mask = dists > (1 - self.proximity_thresh)

        if getattr(self.args, "fuse_score", False):
            dists = matching.fuse_score(dists, detections)

        if getattr(self.args, "with_reid", False) and getattr(self, "encoder", None) is not None:
            emb_dists = matching.embedding_distance(tracks, detections) / 2.0
            #  NaN/inf -> 1.0 così ReID è ignorato quando il crop è poco affidabile
            emb_dists = np.nan_to_num(emb_dists, nan=1.0, posinf=1.0, neginf=1.0)

            emb_dists[emb_dists > (1 - self.appearance_thresh)] = 1.0
            emb_dists[dists_mask] = 1.0

            if use_hm:
                dists = _harmonic_mean_cost(dists, emb_dists)
            else:
                dists = np.minimum(dists, emb_dists)

        return dists

    bot_sort.BOTSORT.get_dists = get_dists_patched
    print(f"[OK] BoT-SORT patched: sanitize NaNs + fusion={'HM' if use_hm else 'MIN'}")


# -----------------------------
# Tracking
# -----------------------------
def process_video(video_path, tracker, out_path, model, device, conf, iou, show, max_det):
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
    ap.add_argument("--source", required=True)
    ap.add_argument("--tracker", required=True)
    ap.add_argument("--out", required=True)

    ap.add_argument("--weights", default="weights/train_yolo11m_SoccerNet.pt")
    ap.add_argument("--conf", type=float, default=0.15)
    ap.add_argument("--iou", type=float, default=0.65)

    ap.add_argument("--osnet-weights", required=True)
    ap.add_argument("--osnet-name", default="osnet_x0_25")

    ap.add_argument("--reid-min-h", type=int, default=60, help="Skip ReID if bbox height < this px")
    ap.add_argument("--reid-pad", type=float, default=0.10, help="Pad bbox before cropping (fraction)")

    ap.add_argument("--max-det", type=int, default=300)
    ap.add_argument("--hm-fusion", action="store_true", help="Use HM fusion (after OSNet works)")

    ap.add_argument("--show", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    device_det = pick_device()
    print(f"[INFO] detector device = {device_det}")

    # OSNet on CPU if MPS
    device_reid = "cpu" if device_det == "mps" else device_det
    if device_det == "mps":
        print("[WARN] MPS detected: running OSNet ReID on CPU for stability.")

    patch_ultralytics_botsort_osnet(
        weights_path=args.osnet_weights,
        model_name=args.osnet_name,
        device=device_reid,
        min_h=args.reid_min_h,
        pad=args.reid_pad,
    )

    # Sanitize embeddings + choose fusion
    enable_botsort_sanitize_patch(use_hm=args.hm_fusion)

    model = YOLO(args.weights)

    out_base = Path(args.out)
    out_base.mkdir(parents=True, exist_ok=True)

    test_folder = Path(args.source)
    video_folders = sorted([p for p in test_folder.iterdir() if p.is_dir() and p.name.isdigit()],
                           key=lambda p: int(p.name))
    if args.limit is not None:
        video_folders = video_folders[:args.limit]

    print(f"[INFO] Processing {len(video_folders)} videos (limit={args.limit if args.limit is not None else 'ALL'})")
    print("[REMINDER] In YAML set: with_reid: True and model: osnet.")

    for vf in video_folders:
        vid = int(vf.name)
        video_path = vf / "img1"
        print(f"[INFO] Predicting from folder: {video_path}")

        out_path = out_base / f"tracking_{vid}_12.txt"
        process_video(video_path, args.tracker, out_path, model, device_det, args.conf, args.iou, args.show, args.max_det)

if __name__ == "__main__":
    main()
