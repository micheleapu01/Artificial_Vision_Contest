import os
import glob
import numpy as np
import pandas as pd
import argparse

def interpolate_file(file_path, max_gap=20, interp_conf=0.4):
    filename = os.path.basename(file_path)
    
    try:
        df = pd.read_csv(file_path, header=None, sep=',')
        if df.shape[1] == 1:
             df = pd.read_csv(file_path, header=None, sep=r'\s+', engine='python')
    except Exception as e:
        print(f"⚠️ [SKIP] Errore lettura {filename}: {e}")
        return

    if df.empty or df.shape[1] < 7:
        return

    if df.shape[1] < 10:
        for i in range(df.shape[1], 10):
            df[i] = -1

    df = df.iloc[:, :10] 
    df.columns = ['frame', 'id', 'x', 'y', 'w', 'h', 'conf', 'x3d', 'y3d', 'z3d']
    df = df.sort_values(by=['id', 'frame'])
    
    interpolated_list = []
    
    # Variabili per statistiche finali
    total_filled = 0
    affected_ids = 0

    print(f"\n--- Analisi file: {filename} ---")
    
    for track_id in df['id'].unique():
        track = df[df['id'] == track_id].copy()
        
        # 1. VISUALIZZA MIN E MAX
        min_f, max_f = track['frame'].min(), track['frame'].max()
        
        # Calcoliamo quanti frame ci sono "realmente" e quanti dovrebbero esserci
        real_frames_count = len(track)
        expected_frames_count = int(max_f - min_f + 1)
        missing_count = expected_frames_count - real_frames_count
        
        # Se non manca nulla, passa oltre veloce
        if missing_count == 0:
            interpolated_list.append(track)
            continue

        full_range = np.arange(min_f, max_f + 1)
        track = track.set_index('frame').reindex(full_range).reset_index()
        track['id'] = track_id
        
        # 2. IDENTIFICA I BUCHI PRIMA DELL'INTERPOLAZIONE
        # missing_mask è True dove c'è un buco
        missing_mask = track['x'].isna()
        
        # Lista dei frame che sono buchi (per stamparli)
        gaps_frames = track.loc[missing_mask, 'frame'].tolist()

        # Interpolazione
        cols_to_interp = ['x', 'y', 'w', 'h']
        track[cols_to_interp] = track[cols_to_interp].interpolate(
            method='linear', 
            limit=max_gap, 
            limit_area='inside'
        )

        track['conf'] = track['conf'].where(track['conf'].notna(), 0.0)
        track.loc[missing_mask, 'conf'] = interp_conf
        track[['x3d', 'y3d', 'z3d']] = -1
        
        # 3. VERIFICA COSA È STATO RIEMPITO
        # Se dopo l'interpolazione la 'x' non è più NaN, vuol dire che è stato riempito
        # Se è ancora NaN, vuol dire che il gap era > max_gap
        filled_mask = missing_mask & track['x'].notna()
        filled_frames = track.loc[filled_mask, 'frame'].tolist()
        
        num_filled = len(filled_frames)
        
        if num_filled > 0:
            affected_ids += 1
            total_filled += num_filled
            print(f"🔹 ID {int(track_id)}: Range [{int(min_f)} - {int(max_f)}]")
            print(f"   Mancanti ({len(gaps_frames)}): {gaps_frames}")
            print(f"   Riempiti ({num_filled}): {filled_frames}")
            if len(gaps_frames) > num_filled:
                skipped = [f for f in gaps_frames if f not in filled_frames]
                print(f"   Ignorati (Gap > {max_gap}): {skipped}")
        
        # Rimuove i residui NaN
        track = track.dropna(subset=['x'])
        interpolated_list.append(track)
    
    if not interpolated_list:
        return

    final_df = pd.concat(interpolated_list)
    final_df = final_df.sort_values(by=['frame', 'id'])
    final_df[['x', 'y', 'w', 'h']] = final_df[['x', 'y', 'w', 'h']].round(2)
    
    temp_file = file_path + ".tmp"
    final_df.to_csv(temp_file, header=False, index=False, sep=',')
    os.replace(temp_file, file_path)
    
    print(f" CONCLUSIONE {filename}: Modificati {affected_ids} ID, Creati {total_filled} nuovi frame.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", default="SIMULATOR/lecture_example_from_training/Predictions_folder", help="Cartella txt")
    parser.add_argument("--gap", type=int, default=20, help="Max frame gap to fill")
    parser.add_argument("--conf", type=float, default=0.4, help="Confidence value for interpolated frames (default: 0.4)")
    parser.add_argument("--pattern", default="tracking*.txt", help="Pattern file da processare (default: tracking*.txt)")
    args = parser.parse_args()

    target_dir = args.folder
    
    if not os.path.exists(target_dir):
        print(f" Cartella non trovata: {target_dir}")
        return

    txt_files = glob.glob(os.path.join(target_dir, args.pattern))
    print(f" Avvio interpolazione su {len(txt_files)} file...")
    print(f"    Params: Max Gap={args.gap}, Interp Conf={args.conf}")
    
    count = 0
    for f in txt_files:
        interpolate_file(f, max_gap=args.gap, interp_conf=args.conf)
        count += 1
        
    print(f" Finito. {count} file processati e sovrascritti.")

if __name__ == "__main__":
    main()