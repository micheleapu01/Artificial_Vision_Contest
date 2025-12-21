import os
import shutil
import csv

def remove_ball_from_gt_and_copy(gameinfo_file, video_folder_path, output_folder, new_video_id):
    print(f"Leggendo {gameinfo_file} per trovare l'ID della palla...")
    ball_id = None
    try:
        with open(gameinfo_file, 'r') as f:
            for line in f:
                if 'ball' in line:
                    parts = line.split('=')
                    ball_id = parts[0].strip().replace('trackletID_', '').strip()
                    print(f"Trovato ID della palla: {ball_id}")
                    break
    except Exception as e:
        print(f"Errore nell'aprire o leggere {gameinfo_file}: {e}")
        return False

    if not ball_id:
        print(f"ID della palla non trovato nel file {gameinfo_file}")
        return False

    # Crea una nuova cartella numerata
    new_video_folder = os.path.join(output_folder, str(new_video_id))

    # Se esiste già, evita errore (puoi anche scegliere di cancellare e ricreare)
    if os.path.exists(new_video_folder):
        print(f"[WARN] La cartella {new_video_folder} esiste già, la salto.")
        return False

    # Copia l'intera cartella del video (inclusi tutti i file)
    shutil.copytree(video_folder_path, new_video_folder)

    # Modifica il file gt.txt nella cartella copiata
    gt_file_path = os.path.join(new_video_folder, 'gt', 'gt.txt')
    try:
        with open(gt_file_path, 'r') as f:
            lines = f.readlines()

        # Filtra le righe che non corrispondono all'ID della palla (2a colonna = object_id)
        filtered_lines = [line for line in lines if line.split(',')[1].strip() != ball_id]

        with open(gt_file_path, 'w') as f:
            f.writelines(filtered_lines)

        print(f"File gt.txt modificato e salvato in {gt_file_path}")
        return True

    except Exception as e:
        print(f"Errore nell'elaborazione del file {gt_file_path}: {e}")
        return False


def process_all_videos_in_test(test_folder, output_folder):
    print(f"Iniziando il processo di rimozione della palla per tutti i video in {test_folder}...")

    # Prendo SOLO le cartelle di primo livello e le ordino
    dirs = [d for d in os.listdir(test_folder) if os.path.isdir(os.path.join(test_folder, d))]
    dirs.sort()  # <-- ORDINE ALFABETICO GARANTITO (SNMOT-116, SNMOT-117, ...)

    os.makedirs(output_folder, exist_ok=True)

    new_video_id = 1  # numerazione da 1

    # (opzionale) mapping per ricordare chi è chi
    mapping_path = os.path.join(output_folder, "mapping.csv")
    with open(mapping_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["new_id", "original_folder"])

        for dir_name in dirs:
            video_folder_path = os.path.join(test_folder, dir_name)
            gameinfo_path = os.path.join(video_folder_path, 'gameinfo.ini')

            if os.path.exists(gameinfo_path):
                print(f"Elaborando il video {dir_name} -> id {new_video_id}...")
                ok = remove_ball_from_gt_and_copy(gameinfo_path, video_folder_path, output_folder, new_video_id)

                if ok:
                    writer.writerow([new_video_id, dir_name])
                    new_video_id += 1
            else:
                print(f"[SKIP] gameinfo.ini non trovato in {video_folder_path}")

    print(f"[OK] mapping salvato in: {mapping_path}")


# Esegui il processo su tutta la cartella test
test_folder_path = 'data/tracking-2023/test'
output_folder_path = 'SIMULATOR/lecture_example_from_training/test_set/videos'

process_all_videos_in_test(test_folder_path, output_folder_path)
