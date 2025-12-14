from pathlib import Path
import zipfile

def has_single_top_folder(zf: zipfile.ZipFile):
    names = [n for n in zf.namelist() if n and not n.endswith("/")]
    tops = {n.split("/", 1)[0] for n in names if "/" in n}
    return len(tops) == 1

def main():
    root = Path("data")
    for z in root.rglob("*.zip"):
        dest = z.with_suffix("")          # es: data/train
        dest.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(z) as zf:
            # se lo zip ha già una cartella root unica, estrai nel parent per evitare train/train
            out = z.parent if has_single_top_folder(zf) else dest
            print(f"[EXTRACT] {z} -> {out}")
            zf.extractall(out)

if __name__ == "__main__":
    main()

