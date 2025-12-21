Artificial Vision Contest — Team Repo

Repo di lavoro per il contest (tracking + behavior analysis) su SoccerNet.

Struttura progetto (WIP)

scripts/ — script eseguibili (tracking, ecc.)

tools/ — utility per gestione dati (download, unzip, ispezione)

src/ — codice riusabile del progetto (in costruzione)

configs/ — configurazioni tracker (ByteTrack/BoT-SORT) e altro

data/ — dataset (NON versionato)

outputs/ — risultati (NON versionato)

weights/ — pesi modelli (NON versionato)

Setup ambiente

Creazione venv dentro la repo:

python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt

Dati (SoccerNet Tracking 2023)

Dataset scaricato tramite SoccerNet Downloader, task tracking-2023.

Struttura tipica (train):

data/tracking-2023/train/SNMOT-XXX/img1/ (750 frame)

data/tracking-2023/train/SNMOT-XXX/gt/gt.txt

Nota: spesso i file arrivano come .zip e vanno estratti manualmente (poi si può cancellare lo zip per risparmiare spazio).

Tracking (baseline attuale)

Baseline:

Detector: YOLO (Ultralytics)

Tracker: ByteTrack / BoT-SORT via Ultralytics (model.track(...))

Output attuale: formato MOT (10 colonne) per debugging locale

Output contest: tracking_K_XX.txt (6 colonne) da implementare/allineare

Esempio run (una sequenza):

python scripts/track.py `
  --source "data/tracking-2023/train/SNMOT-060/img1" `
  --tracker "configs/bytetrack.yaml" `
  --out "outputs/mot/baseline/SNMOT-060.txt" `
  --show

Weights / Modelli utilizzati

I pesi sono salvati in weights/ (ignorata da Git).
Per cambiare modello usa --weights weights/<file>.pt.

YOLO weights

yolov8m.pt
Pesi ufficiali Ultralytics (download automatico tramite libreria ultralytics).

yolov8m-640-football-players.pt (football-specific, da testare/integrare)
Preso dalla repo:

Darkmyter/Football-Players-Tracking
https://github.com/Darkmyter/Football-Players-Tracking.git

Nota: quando si usano weights custom, controllare model.names e settare correttamente classes=[...] (escludendo eventuale “ball”).

Crediti / Fonti

Dataset: SoccerNet Tracking (sn-tracking)
https://github.com/SoccerNet/sn-tracking

Weights / riferimento pipeline football-specific:
https://github.com/Darkmyter/Football-Players-Tracking.git

TODO (breve)

 Implementare evaluation locale (HOTA) con TrackEval