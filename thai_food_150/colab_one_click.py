"""One-command Colab workflow: download source data, audit label coverage, save planning report.
Does NOT claim a trained 150-class model or silently invent extra image classes.
"""
import argparse
import csv
import json
import shutil
import sys
import zipfile
from collections import Counter
from pathlib import Path
from urllib.request import urlopen


def csv_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as fp:
        return list(csv.DictReader(fp))


def write_csv(path, rows, fields):
    with path.open("w", encoding="utf-8-sig", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def zip_audit(path):
    if not path.is_file() or not zipfile.is_zipfile(path):
        raise RuntimeError("Archive missing or invalid: " + str(path))
    counts = Counter()
    examples = {}
    with zipfile.ZipFile(path) as z:
        for info in z.infolist():
            if info.is_dir() or Path(info.filename).suffix.lower() not in (".jpg", ".jpeg", ".png", ".webp"):
                continue
            parts = Path(info.filename).parts
            # Full datasets: train/class/filename.jpeg, val/class/filename.jpeg, test/class/filename.jpeg
            # Sample dataset: THFOOD-100.sample/class/filename.jpg
            if len(parts) >= 3:
                if parts[0] in ("train", "val", "test"):
                    split, cls = parts[0], parts[1]
                else:
                    split, cls = "sample", parts[1]
                counts[(split, cls)] += 1
                examples.setdefault(split, info.filename)
    return {"images": sum(counts.values()),
            "classes": len(set(cls for split, cls in counts)),
            "per_split": dict(Counter({s: sum(n for (split, _), n in counts.items() if split == s)
                                       for s in sorted({s for s, _ in counts})})),
            "examples": examples}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drive-dir", default="/content/drive/MyDrive/ThaiFood150")
    ap.add_argument("--download-full", action="store_true")
    ap.add_argument("--no-download", action="store_true")
    args = ap.parse_args()
    root = Path(args.drive_dir)
    archive = root / "archives"
    archive.mkdir(parents=True, exist_ok=True)
    report_dir = root / "reports"
    report_dir.mkdir(exist_ok=True)
    repo = Path(__file__).resolve().parents[1]
    base = repo / "expansion120"
    mapped = csv_rows(base / "classes_120.csv")
    food100 = csv_rows(base / "thfood100_source_labels.csv")
    dessert20 = csv_rows(base / "thdess20_source_labels.csv")
    old = mapped[:50]
    if len(old) != 50 or len(mapped) != 120 or len(food100) != 100 or len(dessert20) != 20:
        raise ValueError("Unexpected number of classes in checked-in mapping files")

    # Preserve the existing, manually reviewed 120-class proposal without asserting it is verified.
    rows = []
    for row in mapped:
        rows.append({"target_index": row["target_index"], "class_id": row["class_id"],
                     "menu_th": row["menu_th"], "origin": row["origin"],
                     "source_class_id": row["source_class_id"],
                     "status": "candidate_needs_audit"})
    for idx in range(120, 150):
        rows.append({"target_index": str(idx), "class_id": "", "menu_th": "",
                     "origin": "", "source_class_id": "",
                     "status": "MISSING_SOURCE_AND_IMAGES"})
    write_csv(report_dir / "target_150_plan.csv", rows,
              ["target_index", "class_id", "menu_th", "origin", "source_class_id", "status"])

    # Known alias checks are not automatically accepted: different dish scope may matter.
    flagged = [
        ("Tom Yam", "ต้มยำ", "TomYumGoong", "ต้มยำกุ้ง"),
        ("Tom Kha", "ต้มข่า", "TomKhaGai", "ต้มข่าไก่"),
        ("Mu Yang", "หมูย่าง", "KorMooYang", "คอหมูย่าง"),
        ("Khai Chiao", "ไข่เจียว", "KaiJeowMooSaap", "ไข่เจียวหมูสับ"),
        ("Lap", "ลาบ", "LarbMoo", "ลาบหมู"),
        ("Suki Haeng", "สุกี้แห้ง", "Suki", "สุกี้"),
        ("Phat Pak Bung Fai Daeng", "ผัดผักบุ้งไฟแดง", "PadPakBung", "ผัดผักบุ้ง"),
        ("Khao Phat", "ข้าวผัด", "", "possible umbrella class"),
    ]
    write_csv(report_dir / "alias_conflicts_review.csv",
              [{"source_en": a, "source_th": b, "legacy_class": c, "legacy_th_or_note": d,
                "decision": "REVIEW_REQUIRED"} for a, b, c, d in flagged],
              ["source_en", "source_th", "legacy_class", "legacy_th_or_note", "decision"])

    # One original 50-class model is not enough to retrain all 50 categories:
    # source images must be available too.
    original_model_candidates = list((root.parent / "ThaiFoodAI_THFOOD50").glob("food_model_best.pth"))
    status = {"goal": 150, "existing_model_classes": 50, "source_thfood100_classes": 100,
              "source_thdess20_classes": 20, "mapped_candidate_classes": 120,
              "unfilled_class_slots": 30, "legacy_checkpoint_found": bool(original_model_candidates),
              "trained_150_model": False, "full_archives": {}, "warnings": []}

    if args.download_full and not args.no_download:
        try:
            import gdown
            downloads = [
                ("THFOOD-100.zip", "1ZHwErRGhLQIHpSyKYT44Lhxbxbk3Odcs"),
                ("THDESS-20.zip", "16nvoQOusNCd0Nju6YTwbvvDV7yfRoTQg"),
            ]
            for name, id_ in downloads:
                target = archive / name
                if target.is_file() and zipfile.is_zipfile(target):
                    print("Already downloaded:", target, flush=True)
                    continue
                if target.is_file():
                    raise RuntimeError(f"Existing file {target} is not a valid ZIP; inspect before retrying")
                print("Downloading", name, "to Google Drive (may take considerable space/time)", flush=True)
                result = gdown.download(id=id_, output=str(target), quiet=False, fuzzy=True)
                if not result or not zipfile.is_zipfile(target):
                    raise RuntimeError("Download failed or invalid ZIP: " + name)
        except Exception as exc:
            status["warnings"].append("Full download issue: " + str(exc))

    for filename in ("THFOOD-100_sample.zip", "THFOOD-100.zip", "THDESS-20.zip"):
        path = archive / filename
        if path.is_file():
            try:
                status["full_archives"][filename] = zip_audit(path)
                print(filename, status["full_archives"][filename], flush=True)
            except Exception as exc:
                status["warnings"].append(str(exc))

    if "THFOOD-100.zip" not in status["full_archives"]:
        status["warnings"].append("THFOOD-100 full images unavailable: do not start training")
    status["warnings"].append("Exactly 150 non-overlapping, image-backed classes have NOT been established")
    status["warnings"].append("Keep original train/val/test splits; verify image overlap and permissions")
    status["warnings"].append("Do not automatically deploy an untested model")
    dest = report_dir / "pipeline_status.json"
    dest.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n===== FINISHED PREPARATION =====")
    print("Status report:", dest)
    print("150-class taxonomy draft:", report_dir / "target_150_plan.csv")
    print("Alias review:", report_dir / "alias_conflicts_review.csv")
    print("Full archive ready:", "THFOOD-100.zip" in status["full_archives"])
    print("15O-class TRAINING NOT STARTED. 30 candidate slots need verified image sources.")
    if status["warnings"]:
        print("Warnings:")
        for warning in status["warnings"]:
            print("-", warning)


if __name__ == "__main__":
    main()
