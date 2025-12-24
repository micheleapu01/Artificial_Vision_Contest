import os
import glob
import numpy as np
import pandas as pd
import argparse

def interpolate_file(file_path, max_gap=20, interp_conf=0.4):
    """
    Legge un file di tracking, applica l'interpolazione e lo SOVRASCRIVE.
    interp_conf: Valore di confidenza da assegnare ai frame creati artificialmente.
    """
    filename = os.path.basename(file_path)
    
    try:
        # Tenta di leggere con la virgola
        df = pd.read_csv(file_path, header=None, sep=',')
        
        # Se ha letto tutto in una sola colonna, prova con spazi
        if df.shape[1] == 1:
             df = pd.read_csv(file_path, header=None, sep=r'\s+', engine='python')

    except Exception as e:
        print(f" [SKIP] Errore lettura {filename}: {e}")
        return

    if df.empty:
        print(f" [SKIP] File vuoto: {filename}")
        return

    # --- CONTROLLO COLONNE ---
    num_cols = df.shape[1]
    if num_cols < 7:
        return

    # Standardizzazione a 10 colonne (MOT format)
    if num_cols < 10:
        for i in range(num_cols, 10):
            df[i] = -1

    df = df.iloc[:, :10] 
    df.columns = ['frame', 'id', 'x', 'y', 'w', 'h', 'conf', 'x3d', 'y3d', 'z3d']
    
    # Ordina per ID e Frame
    df = df.sort_values(by=['id', 'frame'])
    
    interpolated_list = []
    
    # Itera per ogni ID
    for track_id in df['id'].unique():
        track = df[df['id'] == track_id].copy()
        
        min_f, max_f = track['frame'].min(), track['frame'].max()
        full_range = np.arange(min_f, max_f + 1)
        
        # Reindicizza per creare i buchi (NaN) dove mancano i frame
        track = track.set_index('frame').reindex(full_range).reset_index()
        track['id'] = track_id
        
        # --- INTERPOLAZIONE LINEARE ---
        cols_to_interp = ['x', 'y', 'w', 'h']

        # segna quali righe erano buchi PRIMA dell'interpolazione
        missing_mask = track['x'].isna()

        # riempi SOLO i buchi interni (limit_area='inside')
        track[cols_to_interp] = track[cols_to_interp].interpolate(
            method='linear', 
            limit=max_gap, 
            limit_area='inside'
        )

        # conf: lascia invariata dove esiste, metti 0 dove manca (temporaneo)
        track['conf'] = track['conf'].where(track['conf'].notna(), 0.0)

        # Assegna la confidenza specifica passata da terminale alle righe interpolate
        track.loc[missing_mask, 'conf'] = interp_conf

        # Riempie le coordinate 3D fittizie con -1
        track[['x3d', 'y3d', 'z3d']] = -1
        
        # Rimuove le righe che sono rimaste NaN 
        track = track.dropna(subset=['x'])
        
        interpolated_list.append(track)
    
    if not interpolated_list:
        return

    final_df = pd.concat(interpolated_list)
    final_df = final_df.sort_values(by=['frame', 'id'])
    
    # Arrotondamento per pulizia
    final_df[['x', 'y', 'w', 'h']] = final_df[['x', 'y', 'w', 'h']].round(2)
    
    temp_file = file_path + ".tmp"
    final_df.to_csv(temp_file, header=False, index=False, sep=',')
    os.replace(temp_file, file_path)
    print(f" Interpolato: {filename}")

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