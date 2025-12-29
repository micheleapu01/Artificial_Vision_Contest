import os
import glob
import subprocess
import sys

# ==============================================================================
# CONFIGURAZIONE PERCORSI 
# ==============================================================================

# Cartella dei video originali
SOURCE_VIDEOS = "SIMULATOR/lecture_example_from_training/test_set/videos"

# Cartella dove salvare/leggere i risultati (Predictions)
FULL_OUTPUT_DIR = "SIMULATOR/lecture_example_from_training/Predictions_folder"

# Percorsi agli script Python
SCRIPT_TRACK        = "scripts/track_with_FineTuning.py"
SCRIPT_FIELD_FILTER = "data_tools/field_filter.py"    # <--- Aggiunto
SCRIPT_INTERP       = "scripts/interpolate.py"
SCRIPT_STITCH       = "scripts/stitching_advanced.py" # O stitch_tracks_advanced.py
SCRIPT_BEHAVIOR     = "scripts/behavior.py"

# Configurazione YAML del tracker
TRACKER_CONFIG      = "configs/botsort_with_gmc_with_FineTuning.yaml"


# ==============================================================================
# PARAMETRI DI OGNI FASE
# ==============================================================================

# 1. TRACKING
TRACKING_ARGS = [
    "--source", SOURCE_VIDEOS,
    "--tracker", TRACKER_CONFIG,
    "--out", FULL_OUTPUT_DIR,
    "--conf", "0.15",
    "--iou", "0.6"
]

# 2. FIELD FILTER (Pulizia bordo campo)
FILTER_ARGS = [
    "--pred-folder", FULL_OUTPUT_DIR,
    "--videos-root", SOURCE_VIDEOS,
    "--mask-every", "5",
    "--line-tol", "6"
]

# 3. POST-PROCESSING (Hybrid Sandwich)

# A. Interpolazione Preliminare
INTERP_1_ARGS = ["--gap", "15", "--conf", "0.6"]

# B. Stitching Avanzato
STITCH_ARGS = [
    "--max-gap", "60",
    "--min-len", "10",
    "--dist-base", "3.5",     # <--- Aggressivo per i pan
    "--dist-per-gap", "0.2",
    "--allow-border"
]

# C. Interpolazione Finale
INTERP_2_ARGS = ["--gap", "60", "--conf", "0.4"]

# 4. BEHAVIOR ANALYSIS (Calcolo metriche comportamentali)
BEHAVIOR_ARGS = [
    "--source", SOURCE_VIDEOS,
    "--tracking-dir", FULL_OUTPUT_DIR,
    "--out", FULL_OUTPUT_DIR,
    "--team", "12",
    "--window", "5"
]


# ==============================================================================
# 🚀 MOTORE DELLA PIPELINE
# ==============================================================================

def run_command(script_path, args, step_name):
    """Esegue uno script python gestendo l'interprete e gli errori."""
    print(f"\n{'='*60}")
    print(f"  STEP: {step_name}")
    
    # Verifica esistenza script
    if not os.path.exists(script_path):
        print(f" ERRORE: Script non trovato: {script_path}")
        # Tenta di cercare nella root se il path relativo fallisce
        if os.path.exists(os.path.basename(script_path)):
            script_path = os.path.basename(script_path)
            print(f"    Trovato nella root, uso: {script_path}")
        else:
            sys.exit(1)

    # Costruzione comando
    full_cmd = [sys.executable, script_path] + args
    
    # Debug print
    print(f"    CMD: python {script_path} {' '.join(args)}")
    print(f"{'='*60}")
    
    try:
        subprocess.run(full_cmd, check=True)
        print(f" {step_name} COMPLETATO.")
    except subprocess.CalledProcessError as e:
        print(f" ERRORE CRITICO in {step_name}: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n Interrotto dall'utente.")
        sys.exit(1)

def main():
    print(f" PIPELINE COMPLETA SOCCERNET")
    print(f" Output Dir: {FULL_OUTPUT_DIR}\n")

    # Creazione cartella output se manca
    if not os.path.exists(FULL_OUTPUT_DIR):
        os.makedirs(FULL_OUTPUT_DIR, exist_ok=True)

    # --- FASE 1: TRACKING ---
    txt_files = glob.glob(os.path.join(FULL_OUTPUT_DIR, "tracking_*.txt"))
    
    if len(txt_files) > 0:
        print(f" TROVATI {len(txt_files)} FILE DI TRACKING.")
        print("   Salto la generazione (Tracking).")
    else:
        print("  NESSUN FILE TROVATO. Avvio Tracking Locale...")
        run_command(SCRIPT_TRACK, TRACKING_ARGS, "1. Tracking (YOLO+BoT)")

    # --- FASE 2: FIELD FILTERING ---
    # Rimuove le detection fuori dal campo PRIMA di unirle
    run_command(SCRIPT_FIELD_FILTER, FILTER_ARGS, "2. Field Filter (Rimozione Pubblico)")

    # --- FASE 3: POST-PROCESSING (Sandwich) ---
    
    # A. Pulizia micro-gap
    run_command(SCRIPT_INTERP, ["--folder", FULL_OUTPUT_DIR] + INTERP_1_ARGS, "3. Interpolazione Preliminare")

    # B. Unione ID spezzati
    run_command(SCRIPT_STITCH, ["--folder", FULL_OUTPUT_DIR] + STITCH_ARGS, "4. Stitching Avanzato")

    # C. Chiusura macro-gap
    run_command(SCRIPT_INTERP, ["--folder", FULL_OUTPUT_DIR] + INTERP_2_ARGS, "5. Interpolazione Finale")

    # --- FASE 4: BEHAVIOR ---
    # Calcola le azioni sulle tracce ormai definitive
    run_command(SCRIPT_BEHAVIOR, BEHAVIOR_ARGS, "6. Behavior Analysis")

    print(f"\n{'='*60}")
    print(" PIPELINE TERMINATA CORRETTAMENTE!")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()