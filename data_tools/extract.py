from pathlib import Path
import zipfile

def main():
    root = Path("data")
    zips = list(root.rglob("*.zip"))
    if not zips:
        print("Nessuno zip trovato in data/")
        return

    for z in zips:
        dest = z.with_suffix("")  # train.zip -> train/
        dest.mkdir(parents=True, exist_ok=True)

        print(f"[EXTRACT] {z} -> {dest}")
        with zipfile.ZipFile(z, "r") as zf:
            zf.extractall(dest)

    print("[OK] Done.")

if __name__ == "__main__":
    main()
