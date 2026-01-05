# Artificial Vision Contest — Team Repo ⚽👁️

Repository di lavoro per l’**Artificial Vision Contest** su SoccerNet.
Il progetto implementa una pipeline completa per il **tracking** dei giocatori e la **behavior analysis** (conteggio giocatori in 2 specifiche ROI).

---

## 📂 Struttura del Progetto

- `notebooks/` — notebook di tracking/training e run end-to-end
  - `track.ipynb`
  - `train_yolo11m_SoccerNet.ipynb`
  - `Training_Osnet.ipynb`
  - `run_pipeline.ipynb` — esecuzione pipeline end-to-end (es. Colab) *(in arrivo / da commitare)*
- `data_tools/` — script per gestione dati (download, extract/unzip, ispezione, ecc.)
- `scripts/` — script eseguibili
  - `pipeline.py` — esegue l’intera pipeline end-to-end (tracking + behavior) in un solo comando
- `configs/` — configurazioni tracker (ByteTrack/BoT-SORT) e componenti correlati
- `data/` — dataset (**NON versionato**)
- `weights/` — pesi modelli (**NON versionato**)
- `SIMULATOR/` — cartella di supporto per la simulazione e la validazione locale
  - `Predictions_folder/` — file di output della pipeline 
  - `test_set/` — set di test per la simulazione
  - `results/` — risultati prodotti dal simulator

> Nota: `data/`, `weights/`, `Predictions_folder/`,`test_set/`  non vengono versionate su Git.

---

📊 Dataset & Pre-processing

### Dataset (SoccerNet Tracking 2023)
Il dataset di training è stato scaricato tramite **SoccerNet Downloader** (task `tracking-2023`).

**Struttura tipica (train):**

data/tracking-2023/train/SNMOT-XXX/img1/       # Cartella frame
data/tracking-2023/train/SNMOT-XXX/gt/gt.txt   # Ground Truth

**Preparazione Test Set**

I video presenti in test_set/videos/ sono stati inseriti dopo un flusso di pre-processing composto dai seguenti script:

preprocessing_ball.py, distribute_roi.py, generate_behavior.py.

Questo flusso prepara i video per l’analisi e li rinumera in modo consistente da 1 a N (dove N è il numero massimo di video), rendendoli pronti per il tracking e la behavior analysis.

**🚀 Esecuzione Pipeline**
1. Esecuzione da CLI:
Lo script principale è scripts/pipeline.py.

### 2. Esecuzione via Notebook (Colab)
È disponibile il notebook `run_pipeline.ipynb`, che permette di:
- Caricare una cartella con video di test direttamente da **Google Drive**.
- Scaricare e utilizzare i pesi contenuti nel Drive della consegna.
- Generare gli output finali ed esportarli in un archivio (es. `.zip`).

---

## 🎮 Simulazione e Convenzioni Output

### Avvio Simulazione (SIMULATOR)
1. Scaricare l’archivio dei risultati generato dalla pipeline (es. `.zip`).
2. Estrarlo nella cartella `Predictions_folder` del **SIMULATOR**.
3. **Input Video:** Assicurarsi che i video nella cartella `videos/` del simulatore siano numerati `1, 2, 3, ...`.

### Formato Output Richiesto
I risultati di tracking e behavior devono seguire rigorosamente il seguente formato di nomenclatura:

- **Tracking:** `tracking_K_XX.txt`
- **Behavior:** `behavior_K_XX.txt`

**Legenda:**
* `K` = Video ID (da 1 a 5)
* `XX` = Numero del team a due cifre (es. Team 1 → `01`)

**Esempi:**
`tracking_1_01.txt`, `behavior_1_01.txt`

### Specifiche File

#### 📍 Tracking (`tracking_K_XX.txt`)
Una riga per giocatore per frame.
**Formato:** `frame_id, object_id, top_left_x, top_left_y, width, height`

📋 Behavior (behavior_K_XX.txt)
Due righe per frame (una per ROI). Formato: frame_id, region_id, n_players

Nota: Un giocatore è considerato "dentro" una ROI se il centro della base della bounding box (footpoint) cade all'interno della ROI.

🧠 Weights / Modelli
I pesi vengono salvati nella cartella weights/ (ignorata da Git). Per cambiare modello durante l'esecuzione, usare l'opzione:

--weights weights/<file>.pt

🔗 Crediti / Fonti
Dataset / Tools SoccerNet Tracking: https://github.com/SoccerNet/sn-tracking

Repository di riferimento per i pesi football-specific: https://github.com/Darkmyter/Football-Players-Tracking