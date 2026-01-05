import os
import glob
import subprocess
import sys

# ==============================================================================
# CONFIGURAZIONE 
# ==============================================================================
SOURCE_VIDEOS = "SIMULATOR/lecture_example_from_training/test_set/videos"
FULL_OUTPUT_DIR = "SIMULATOR/lecture_example_from_training/Predictions_folder"

SCRIPT_TRACK        = "scripts/osnet_reid_track.py"
SCRIPT_FIELD_FILTER = "data_tools/field_filter.py"
SCRIPT_INTERP       = "scripts/interpolate.py"
SCRIPT_STITCH       = "scripts/stitching.py"
SCRIPT_BEHAVIOR     = "scripts/behavior.py"
TRACKER_CONFIG      = "configs/botsort_with_reid_OsNet.yaml"

# ==============================================================================
# SETUP DA GARA
# ==============================================================================

# 1. TRACKING
TRACKING_ARGS = ["--source", SOURCE_VIDEOS, "--tracker", TRACKER_CONFIG, "--out", FULL_OUTPUT_DIR, "--conf", "0.15", "--iou", "0.6",
                 "--osnet-weights", "weights/osnet_x0_25_finetuned_backbone_only.pth", "--max-det", "300", "--reid-min-h", "60", "--reid-pad", "0.10", "--hm-fusion"]

# 2. FIELD FILTER 
FILTER_ARGS = [
    "--pred-folder", FULL_OUTPUT_DIR,
    "--videos-root", SOURCE_VIDEOS,
    "--mask-every", "5",
    "--line-tol", "2"      
]

# 3. STITCHING 
STITCH_ARGS = [
    "--max-gap", "30",
    "--dist-base", "1.5", 
    "--dist-per-gap", "0.15", 
    "--min-len", "6",      
    "--allow-border"
]

# 4. INTERPOLAZIONE 
INTERP_ARGS = ["--gap", "30", "--conf", "0.4", "--sigma", "1.5"] 

# 5. BEHAVIOR 
BEHAVIOR_ARGS = ["--source", SOURCE_VIDEOS, "--tracking-dir", FULL_OUTPUT_DIR, "--out", FULL_OUTPUT_DIR, "--window", "7"]

# ==============================================================================
# ESECUZIONE
# ==============================================================================
def run_command(script, args, name):
    print(f"\n{'='*60}\n {name}\nCMD: python {script} {' '.join(args)}\n{'='*60}")
    try:
        subprocess.run([sys.executable, script] + args, check=True)
        print(f" {name} OK.")
    except Exception as e:
        print(f" ERRORE: {e}")
        sys.exit(1)

def main():
    if not os.path.exists(FULL_OUTPUT_DIR): os.makedirs(FULL_OUTPUT_DIR)

    # 1. TRACKING (Salta se esiste)
    if not glob.glob(os.path.join(FULL_OUTPUT_DIR, "tracking_*.txt")):
        run_command(SCRIPT_TRACK, TRACKING_ARGS, "Tracking")
    
    # 2. FILTER 
    run_command(SCRIPT_FIELD_FILTER, FILTER_ARGS, "Field Filter ")

    # 3. PIPELINE da gara
    run_command(SCRIPT_STITCH, ["--folder", FULL_OUTPUT_DIR] + STITCH_ARGS, "1. Stitching")
    run_command(SCRIPT_INTERP, ["--folder", FULL_OUTPUT_DIR] + INTERP_ARGS, "2. Interpolazione")
    run_command(SCRIPT_BEHAVIOR, BEHAVIOR_ARGS, "3. Behavior")

    print("\n PIPELINE COMPLETATA.")

if __name__ == "__main__":
    main()