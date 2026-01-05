import os
import glob
import numpy as np
import pandas as pd
import argparse
from scipy.ndimage import gaussian_filter1d

def smooth_tracks(df, sigma=1.0):
    """
    Applica Gaussian Smoothing alle coordinate.
    Sigma basso (0.5-1.0) = Toglie solo il tremolio.
    Sigma alto (>1.5) = Curve molto larghe (rischia di tagliare i dribbling).
    """
    for col in ['x', 'y', 'w', 'h']:
        df[col] = gaussian_filter1d(df[col].values, sigma=sigma)
    return df

def interpolate_file(file_path, max_gap=20, interp_conf=0.4, sigma=0.0):
    filename = os.path.basename(file_path)
    try:
        df = pd.read_csv(file_path, header=None, sep=',')
        if df.shape[1] == 1:
             df = pd.read_csv(file_path, header=None, sep=r'\s+', engine='python')
    except Exception as e:
        print(f" [SKIP] {filename}: {e}")
        return

    if df.empty or df.shape[1] < 7: return

    # Standardizzazione
    if df.shape[1] < 10:
        for i in range(df.shape[1], 10): df[i] = -1
    df = df.iloc[:, :10] 
    df.columns = ['frame', 'id', 'x', 'y', 'w', 'h', 'conf', 'x3d', 'y3d', 'z3d']
    df = df.sort_values(by=['id', 'frame'])
    
    interpolated_list = []
    
    for track_id, track in df.groupby('id'):
        track = track.copy()
        min_f, max_f = track['frame'].min(), track['frame'].max()
        full_range = np.arange(min_f, max_f + 1)
        track = track.set_index('frame').reindex(full_range).reset_index()
        track['id'] = track_id
        
        missing_mask = track['x'].isna()
        
        if not missing_mask.any():
            if sigma > 0 and len(track) > 5:
                track = smooth_tracks(track, sigma=sigma)
            interpolated_list.append(track)
            continue

        # Interpolazione Lineare
        cols = ['x', 'y', 'w', 'h']
        track[cols] = track[cols].interpolate(method='linear', limit=max_gap, limit_area='inside')
        track['conf'] = track['conf'].fillna(interp_conf)
        track[['x3d', 'y3d', 'z3d']] = track[['x3d', 'y3d', 'z3d']].fillna(-1)

        # Pulizia residui
        track = track.dropna(subset=['x'])

        # Smoothing Gaussiano
        if sigma > 0 and len(track) > 5:
            track = smooth_tracks(track, sigma=sigma)

        interpolated_list.append(track)
    
    if not interpolated_list: return

    final_df = pd.concat(interpolated_list).sort_values(by=['frame', 'id'])
    final_df[['x', 'y', 'w', 'h']] = final_df[['x', 'y', 'w', 'h']].round(2)
    final_df['frame'] = final_df['frame'].astype(int)
    final_df['id'] = final_df['id'].astype(int)
    
    temp = file_path + ".tmp"
    final_df.to_csv(temp, header=False, index=False, sep=',')
    os.replace(temp, file_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", required=True)
    parser.add_argument("--gap", type=int, default=20)
    parser.add_argument("--conf", type=float, default=0.4)
    parser.add_argument("--sigma", type=float, default=0.0, help="Intensità smoothing (0 = off)")
    parser.add_argument("--pattern", default="tracking*.txt")
    args = parser.parse_args()

    files = glob.glob(os.path.join(args.folder, args.pattern))
    print(f"🚀 Interpolazione (Gap {args.gap}) + Smoothing (Sigma {args.sigma}) su {len(files)} file.")
    
    for f in files:
        interpolate_file(f, max_gap=args.gap, interp_conf=args.conf, sigma=args.sigma)

if __name__ == "__main__":
    main()