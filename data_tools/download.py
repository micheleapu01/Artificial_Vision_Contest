from pathlib import Path
from SoccerNet.Downloader import SoccerNetDownloader

def main():
    root = Path("data")
    root.mkdir(parents=True, exist_ok=True)

    d = SoccerNetDownloader(LocalDirectory=str(root))
    d.downloadDataTask(task="tracking-2023", split=["train", "challenge"])

if __name__ == "__main__":
    main()
