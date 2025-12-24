import os
import glob
import numpy as np
import pandas as pd
import argparse
from scipy.spatial.distance import cdist

def stitch_file(file_path, time_thresh=30, dist_thresh=50):
    """
    Collega tracce spezzate (ID switches) basandosi su prossimità spazio-temporale.
    time_thresh: Max frame tra la fine di traccia A e inizio traccia B
    dist_thresh: Max distanza pixel tra fine A e inizio B
    """
    filename = os.path.basename(file_path)
    
    try:
        df = pd.read_csv(file_path, header=None, sep=',')
        # Fallback separatore
        if df.shape[1] < 6:
            df = pd.read_csv(file_path, header=None, sep=r'\s+', engine='python')
    except Exception as e:
        print(f"⚠️ [SKIP] Errore lettura {filename}: {e}")
        return

    if df.shape[1] < 10:
        # Pad columns if needed
        for i in range(df.shape[1], 10):
            df[i] = -1
            
    df = df.iloc[:, :10]
    df.columns = ['frame', 'id', 'x', 'y', 'w', 'h', 'conf', 'x3d', 'y3d', 'z3d']
    
    # Ordina per frame
    df = df.sort_values(by=['frame', 'id'])
    
    # 1. Estrai info su inizio e fine di ogni traccia
    track_stats = {}
    unique_ids = df['id'].unique()
    
    for tid in unique_ids:
        track = df[df['id'] == tid]
        start_row = track.iloc[0]
        end_row = track.iloc[-1]
        
        # Salviamo: (start_frame, start_x, start_y), (end_frame, end_x, end_y)
        # Usiamo il centro del box per la distanza, è più robusto
        start_cx = start_row['x'] + start_row['w'] / 2
        start_cy = start_row['y'] + start_row['h'] / 2
        
        end_cx = end_row['x'] + end_row['w'] / 2
        end_cy = end_row['y'] + end_row['h'] / 2
        
        track_stats[tid] = {
            'start_f': start_row['frame'],
            'end_f': end_row['frame'],
            'start_pos': np.array([start_cx, start_cy]),
            'end_pos': np.array([end_cx, end_cy]),
            'count': len(track) # utile per non unire rumore
        }

    # 2. Algoritmo Greedy di unione
    # Ordiniamo le tracce per tempo di fine (chi finisce prima cerca chi inizia dopo)
    sorted_ids_by_end = sorted(track_stats.keys(), key=lambda x: track_stats[x]['end_f'])
    
    # Mappa per rinominare gli ID: old_id -> new_id
    id_map = {tid: tid for tid in unique_ids}
    
    merged_count = 0
    
    for i, id_a in enumerate(sorted_ids_by_end):
        # Se id_a è già stato unito a qualcun altro (come "secondo"), saltalo
        # Vogliamo la catena A -> B -> C. Se A->B, ora cerchiamo B->C
        curr_a = id_map[id_a] # Seguiamo la catena corrente
        
        stats_a = track_stats[id_a] # Usiamo le stats originali della coda di A
        
        best_match = None
        min_dist = float('inf')
        
        # Cerca tra i possibili successori (quelli che non sono già stati processati troppo)
        # Ottimizzazione: guarda solo quelli che iniziano DOPO che A finisce
        candidates = []
        for id_b in unique_ids:
            if id_b == id_a: continue
            
            # Se B è già stato mappato (è diventato parte di una traccia precedente), ignoralo come INIZIO
            if id_map[id_b] != id_b: continue
            
            stats_b = track_stats[id_b]
            
            # Delta temporale
            time_gap = stats_b['start_f'] - stats_a['end_f']
            
            # Deve iniziare DOPO e entro la soglia
            if 0 < time_gap <= time_thresh:
                candidates.append(id_b)
        
        # Tra i candidati temporali, cerca il più vicino spazialmente
        for id_b in candidates:
            dist = np.linalg.norm(stats_a['end_pos'] - track_stats[id_b]['start_pos'])
            
            if dist < dist_thresh:
                if dist < min_dist:
                    min_dist = dist
                    best_match = id_b
        
        # SE TROVATO MATCH
        if best_match is not None:
            # Unisci: ID B diventa ID A
            # Attenzione: se A era già stato rinominato, B prende il nome "root" di A
            root_id = id_map[id_a]
            id_map[best_match] = root_id
            
            # Aggiorniamo le statistiche "logiche" per permettere catene A->B->C
            # La "nuova" coda di A diventa la coda di B
            # Ma nel loop stiamo iterando sugli ID originali, quindi basta aggiornare la mappa
            merged_count += 1
            # print(f"   Link: {id_a} -> {best_match} (Dist: {min_dist:.1f}px, Time: {track_stats[best_match]['start_f'] - stats_a['end_f']}fr)")

    # 3. Applica la mappa al DataFrame
    if merged_count > 0:
        df['id'] = df['id'].map(id_map)
        
        # Riordina e salva
        df = df.sort_values(by=['frame', 'id'])
        
        # Arrotonda
        df[['x', 'y', 'w', 'h']] = df[['x', 'y', 'w', 'h']].round(2)
        
        # Sovrascrivi
        temp_file = file_path + ".tmp"
        df.to_csv(temp_file, header=False, index=False, sep=',')
        os.replace(temp_file, file_path)
        print(f"✅ Stitched {filename}: Unito {merged_count} frammenti.")
    else:
        print(f"🔹 {filename}: Nessuna unione necessaria.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", default="SIMULATOR/lecture_example_from_training/Predictions_folder")
    parser.add_argument("--time", type=int, default=60, help="Max frame gap (default: 60 = 2 secondi)")
    parser.add_argument("--dist", type=int, default=100, help="Max pixel distance (default: 100px)")
    args = parser.parse_args()

    target_dir = args.folder
    if not os.path.exists(target_dir):
        print("❌ Cartella non trovata")
        return

    files = glob.glob(os.path.join(target_dir, "*.txt"))
    print(f"🧵 Avvio Stitching su {len(files)} file...")
    print(f"   Parametri: MaxTime={args.time} frames, MaxDist={args.dist} pixels")
    
    for f in files:
        stitch_file(f, time_thresh=args.time, dist_thresh=args.dist)

if __name__ == "__main__":
    main()