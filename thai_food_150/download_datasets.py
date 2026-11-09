"""Download original Thai food datasets to a local folder or Colab Drive.
Image archives are NOT committed to GitHub. Educational/non-commercial use only.
Requires: pip install gdown
"""
import argparse
from pathlib import Path
import gdown

SOURCES = {
    "thfood100_sample": ("1Yju46kBptcURRkNfQtgjWNXt_RvDqbIP", "THFOOD-100_sample.zip"),
    "thfood100_full": ("1ZHwErRGhLQIHpSyKYT44Lhxbxbk3Odcs", "THFOOD-100.zip"),
    "thdess20_full": ("16nvoQOusNCd0Nju6YTwbvvDV7yfRoTQg", "THDESS-20.zip"),
}

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="downloaded_archives")
    p.add_argument("--datasets", nargs="+", choices=list(SOURCES), default=["thfood100_sample"])
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for key in a.datasets:
        file_id, name = SOURCES[key]
        dest = out / name
        if dest.exists() and dest.stat().st_size > 0:
            print(f"Already exists, verify manually: {dest}")
            continue
        print(f"Downloading {key} to {dest}")
        result = gdown.download(id=file_id, output=str(dest), quiet=False, fuzzy=True)
        if result is None or not dest.is_file():
            raise RuntimeError(f"Failed: {key}. Open the dataset provider's Drive link in a browser.")
        print(f"Downloaded {dest} ({dest.stat().st_size:,} bytes)")
    print("Archives downloaded. Review licenses, extract and verify class labels before training.")

if __name__ == "__main__":
    main()
