import os
import shutil

def remove_ball_from_gt_and_copy(gameinfo_file, video_folder_path, output_folder, new_video_id):
    # Leggi il file gameinfo.ini per ottenere l'ID della palla
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
        return
    
    if not ball_id:
        print(f"ID della palla non trovato nel file {gameinfo_file}")
        return

    # Crea una nuova cartella numerata
    new_video_folder = os.path.join(output_folder, str(new_video_id))

    # Copia l'intera cartella del video (inclusi tutti i file)
    shutil.copytree(video_folder_path, new_video_folder)  # Copia tutta la cartella

    # Modifica il file gt.txt nella cartella copiata
    gt_file_path = os.path.join(new_video_folder, 'gt', 'gt.txt')
    try:
        with open(gt_file_path, 'r') as f:
            lines = f.readlines()

        # Filtra le righe che non corrispondono all'ID della palla
        filtered_lines = [line for line in lines if not line.split(',')[1].strip() == ball_id]

        # Scrivi il contenuto modificato nel nuovo file
        with open(gt_file_path, 'w') as f:
            f.writelines(filtered_lines)
        print(f"File gt.txt modificato e salvato in {gt_file_path}")
    except Exception as e:
        print(f"Errore nell'elaborazione del file {gt_file_path}: {e}")
        return


def process_all_videos_in_test(test_folder, output_folder):
    print(f"Iniziando il processo di rimozione della palla per tutti i video in {test_folder}...")  # Debug stampa
    new_video_id = 1  # Iniziamo a numerare i video a partire da 1
    for root, dirs, files in os.walk(test_folder):
        for dir_name in dirs:
            video_folder_path = os.path.join(root, dir_name)
            gameinfo_path = os.path.join(video_folder_path, 'gameinfo.ini')

            # Verifica la presenza del file gameinfo.ini nella cartella principale del video
            if os.path.exists(gameinfo_path):
                # Chiamata per copiare la cartella e modificare gt.txt
                print(f"Elaborando il video {dir_name}...")
                remove_ball_from_gt_and_copy(gameinfo_path, video_folder_path, output_folder, new_video_id)
                new_video_id += 1  # Incrementa l'ID del video
                
# Esegui il processo su tutta la cartella test
test_folder_path = 'data/tracking-2023/test'
output_folder_path = 'SIMULATOR/test_set/videos'

process_all_videos_in_test(test_folder_path, output_folder_path)
