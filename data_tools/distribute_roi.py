import shutil
import argparse
from pathlib import Path

def main():

    default_source = "roi.json"
    default_videos_dir = "SIMULATOR/lecture_example_from_training/test_set/videos"

    parser = argparse.ArgumentParser(description="Distribuisce un file roi.json in tutte le sottocartelle video.")
    parser.add_argument("--roi", default=default_source, help="Il file roi.json sorgente da copiare")
    parser.add_argument("--dest", default=default_videos_dir, help="Cartella contenente le sottocartelle dei video (1, 2, ...)")
    
    args = parser.parse_args()

    source_file = Path(args.roi)
    videos_dir = Path(args.dest)

    # 1. Controlli preliminari
    if not source_file.exists():
        print(f"ERRORE: file sorgente non trovato: {source_file.resolve()}")
        print("Assicursi di aver creato il file 'roi.json' nella cartella corrente.")
        return

    if not videos_dir.exists():
        print(f"ERRORE: La cartella di destinazione non esiste: {videos_dir}")
        return

    print(f"File Sorgente: {source_file.name}")
    print(f"Destinazione: {videos_dir}")
    print("-" * 40)

    # 2. Trova le cartelle target (solo quelle numerate)
    target_folders = [p for p in videos_dir.iterdir() if p.is_dir() and p.name.isdigit()]
    target_folders.sort(key=lambda x: int(x.name)) # Ordina numericamente 

    if not target_folders:
        print("Nessuna cartella video numerata trovata!")
        return

    # 3. Esegui la copia
    count = 0
    for folder in target_folders:
        dest_path = folder / source_file.name
        try:
            shutil.copy(source_file, dest_path)
            print(f"Copiato in: {folder.name}/")
            count += 1
        except Exception as e:
            print(f"Errore copiando in {folder.name}: {e}")

    print("-" * 40)
    print(f"Operazione completata! File copiato in {count} cartelle.")

if __name__ == "__main__":
    main()