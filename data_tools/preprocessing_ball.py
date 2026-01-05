import os
import shutil
import csv
import re
import argparse


TEST_FOLDER_PATH = "data/tracking-2023/test"
OUTPUT_FOLDER_PATH = "SIMULATOR/lecture_example_from_training/test_set/videos"

def get_ball_ids_from_gameinfo(gameinfo_file):
    ball_ids = set()
    with open(gameinfo_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or "=" not in line:
                continue

            left, right = line.split("=", 1)
            left = left.strip()      # es: trackletID_6
            right = right.strip()    # es: ball;1

            # Prendi solo la label prima del ';'
            label = right.split(";", 1)[0].strip().lower()
            if label != "ball":
                continue

            # Estrai il numero da trackletID_X
            m = re.match(r"trackletID_(\d+)$", left, flags=re.IGNORECASE)
            if m:
                ball_ids.add(m.group(1))  # tienilo come stringa

    return ball_ids

def remove_ball_from_gt_and_copy(gameinfo_file, video_folder_path, output_folder, new_video_id):
    ball_ids = get_ball_ids_from_gameinfo(gameinfo_file)

    if not ball_ids:
        print(f"[WARN] Nessun ball_id trovato in {gameinfo_file}")
        

    new_video_folder = os.path.join(output_folder, str(new_video_id))
    if os.path.exists(new_video_folder):
        print(f"[ERR] {new_video_folder} esiste già. Pulisci l'output e rilancia.")
        return False

    shutil.copytree(video_folder_path, new_video_folder)

    gt_file_path = os.path.join(new_video_folder, "gt", "gt.txt")
    try:
        with open(gt_file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        removed = 0
        filtered_lines = []
        for line in lines:
            parts = line.strip().split(",")
            if len(parts) < 2:
                filtered_lines.append(line)
                continue

            obj_id = parts[1].strip()
            if obj_id in ball_ids:
                removed += 1
                continue

            filtered_lines.append(line)

        with open(gt_file_path, "w", encoding="utf-8") as f:
            f.writelines(filtered_lines)

        print(f"[OK] {gt_file_path} | ball_ids={sorted(ball_ids)} | righe rimosse={removed}")
        return True

    except Exception as e:
        print(f"Errore nel processare {gt_file_path}: {e}")
        return False


def _folder_sort_key(name: str):
    m = re.search(r"(\d+)$", name)
    return int(m.group(1)) if m else name


def process_all_videos_in_test(test_folder, output_folder, limit=None):
    print(f"[INFO] test_folder  = {test_folder}")
    print(f"[INFO] out_folder   = {output_folder}")
    print(f"[INFO] limit        = {limit if limit is not None else 'ALL'}")


    dirs = [d for d in os.listdir(test_folder) if os.path.isdir(os.path.join(test_folder, d))]
    dirs.sort(key=_folder_sort_key)

    
    dirs = [d for d in dirs if os.path.exists(os.path.join(test_folder, d, "gameinfo.ini"))]

    
    if limit is not None:
        dirs = dirs[:limit]

    mapping_path = os.path.join(output_folder, "mapping.csv")
    with open(mapping_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["new_id", "original_folder"])

        new_video_id = 1
        for dir_name in dirs:
            video_folder_path = os.path.join(test_folder, dir_name)
            gameinfo_path = os.path.join(video_folder_path, "gameinfo.ini")

            print(f"[INFO] {dir_name} -> new_id {new_video_id}")
            ok = remove_ball_from_gt_and_copy(gameinfo_path, video_folder_path, output_folder, new_video_id)

            if ok:
                writer.writerow([new_video_id, dir_name])
                new_video_id += 1

    print(f"[OK] mapping salvato in: {mapping_path}")
    print(f"[OK] copiati: {new_video_id - 1} video")

def main():
    ap = argparse.ArgumentParser()
    
    ap.add_argument("--limit", type=int, default=None, help="Process only first N videos (default: all)")
    args = ap.parse_args()

    process_all_videos_in_test(TEST_FOLDER_PATH, OUTPUT_FOLDER_PATH, args.limit)

if __name__ == "__main__":
    main()
