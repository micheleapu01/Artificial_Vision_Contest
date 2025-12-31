import os
import glob
import subprocess
import sys

# ==============================================================================
# CONFIGURAZIONE (RECORD 1.751 SETUP)
# ==============================================================================
SOURCE_VIDEOS = "SIMULATOR/lecture_example_from_training/test_set/videos"
FULL_OUTPUT_DIR = "SIMULATOR/lecture_example_from_training/Predictions_folder"

SCRIPT_TRACK        = "scripts/track_with_FineTuning.py"
SCRIPT_FIELD_FILTER = "data_tools/field_filter.py"
SCRIPT_INTERP       = "scripts/interpolate_new.py"
SCRIPT_STITCH       = "scripts/stitching_new.py"
SCRIPT_BEHAVIOR     = "scripts/behavior.py"
TRACKER_CONFIG      = "configs/botsort_hybrid.yaml"

# ==============================================================================
# PARAMETRI VINCENTI + FIELD TUNING
# ==============================================================================

# 1. TRACKING
TRACKING_ARGS = ["--source", SOURCE_VIDEOS, "--tracker", TRACKER_CONFIG, "--out", FULL_OUTPUT_DIR, "--conf", "0.15", "--iou", "0.6"]

# 2. FIELD FILTER (IL NUOVO TEST)
# Tolleranza 2 pixel. Taglia le riserve, salva il guardalinee.
FILTER_ARGS = [
    "--pred-folder", FULL_OUTPUT_DIR,
    "--videos-root", SOURCE_VIDEOS,
    "--mask-every", "5",
    "--line-tol", "2"      # <--- MODIFICA CHIRURGICA (Era 6)
]

# 3. STITCHING (Min-Len 6 confermato)
STITCH_ARGS = [
    "--max-gap", "30",
    "--dist-base", "1.5", 
    "--dist-per-gap", "0.15", 
    "--min-len", "6",      # Confermato vincente
    "--allow-border"
]

# 4. INTERPOLAZIONE (Sigma 1.5 confermato)
INTERP_ARGS = ["--gap", "30", "--conf", "0.4", "--sigma", "1.5"] 

# 5. BEHAVIOR (Window 7 confermato)
BEHAVIOR_ARGS = [
    "--source", SOURCE_VIDEOS, "--tracking-dir", FULL_OUTPUT_DIR, "--out", FULL_OUTPUT_DIR,
    "--team", "12", "--window", "7"
]

# ==============================================================================
# ESECUZIONE
# ==============================================================================
def run_command(script, args, name):
    print(f"\n{'='*60}\n▶️ {name}\nCMD: python {script} {' '.join(args)}\n{'='*60}")
    try:
        subprocess.run([sys.executable, script] + args, check=True)
        print(f"✅ {name} OK.")
    except Exception as e:
        print(f"❌ ERRORE: {e}")
        sys.exit(1)

def main():
    if not os.path.exists(FULL_OUTPUT_DIR): os.makedirs(FULL_OUTPUT_DIR)

    # 1. TRACKING (Salta se esiste)
    if not glob.glob(os.path.join(FULL_OUTPUT_DIR, "tracking_*.txt")):
        run_command(SCRIPT_TRACK, TRACKING_ARGS, "Tracking")
    
    # 2. FILTER (Strict 2px)
    run_command(SCRIPT_FIELD_FILTER, FILTER_ARGS, "Field Filter (Tol 2)")

    # 3. PIPELINE RECORD
    run_command(SCRIPT_STITCH, ["--folder", FULL_OUTPUT_DIR] + STITCH_ARGS, "1. Stitching (MinLen 6)")
    run_command(SCRIPT_INTERP, ["--folder", FULL_OUTPUT_DIR] + INTERP_ARGS, "2. Interpolazione (Sigma 1.5)")
    run_command(SCRIPT_BEHAVIOR, BEHAVIOR_ARGS, "3. Behavior (Win 7)")

    print("\n🏆 FINITO. Se il DetPr sale e l'AssA regge, andiamo verso 1.76.")

if __name__ == "__main__":
    main()