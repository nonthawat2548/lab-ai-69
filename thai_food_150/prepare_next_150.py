"""Prepare the 9 genuinely new THDESS-20 classes not in the 120-class plan.
Creates an additional image ZIP, a 129-class mapping and a truthful 150-class gap report.
Does not change the trained 120-class checkpoint or invent 21 image classes.
"""
import argparse,csv,io,json,os,shutil,zipfile
from collections import Counter
from pathlib import Path
from PIL import Image,ImageOps

ROOT=Path(__file__).resolve().parents[1]
MAPPING=ROOT/"expansion120"
EXTS={".jpg",".jpeg",".png",".webp"}
LIMITS={"train":100,"val":20,"test":20}
EXCLUDE={"Mango_with_Sticky_Rice","Roti"}

def rows(path):
    with open(path,encoding="utf-8-sig",newline="") as f:return list(csv.DictReader(f))

def find_zip(base):
    options=[base/"archives/THDESS-20.zip",
             Path("/content/drive/MyDrive/ThaiFood150/archives/THDESS-20.zip"),
             Path("/content/gdrive/MyDrive/ThaiFood150/archives/THDESS-20.zip"),
             Path("/content/thaifood_drive/MyDrive/ThaiFood150/archives/THDESS-20.zip")]
    for p in options:
        if p.is_file() and zipfile.is_zipfile(p):return p
    target=base/"archives/THDESS-20.zip"
    if target.exists():
        raise ValueError(f"Existing dessert archive is invalid: {target}")
    print("THDESS-20 missing; trying automatic download...",flush=True)
    try:
        import gdown
    except ImportError as exc:
        raise RuntimeError("Install gdown: pip install gdown") from exc
    target.parent.mkdir(parents=True,exist_ok=True)
    result=gdown.download(id="16nvoQOusNCd0Nju6YTwbvvDV7yfRoTQg",output=str(target),quiet=False,fuzzy=True)
    if not result or not zipfile.is_zipfile(target):
        raise RuntimeError("THDESS-20 download failed; mount Drive containing the original archive")
    return target

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--base",default="/content/thaifood_drive/MyDrive/ThaiFood150")
    args=ap.parse_args()
    base=Path(args.base);out=base/"prepared";out.mkdir(parents=True,exist_ok=True)
    model=base/"training_120/best_model_120.pth"
    merged=out/"ThaiFood120_merged.zip"
    names_path=out/"class_names_120.json"
    for required in (model,merged,names_path):
        if not required.is_file():raise FileNotFoundError(str(required))
    existing=json.loads(names_path.read_text(encoding="utf-8"))
    if len(existing)!=120 or len(set(existing))!=120:raise ValueError("Not 120 unique classes")
    if not zipfile.is_zipfile(merged):raise ValueError("120-class merged ZIP invalid")
    src=find_zip(base)
    original=rows(MAPPING/"thdess20_source_labels.csv")
    selected=rows(MAPPING/"classes_120.csv")
    used={r["source_class_id"] for r in selected if r["origin"]=="THDESS20_source_label"}
    available=[r for r in original if r["Directory_Name"] not in used and r["Directory_Name"] not in EXCLUDE]
    if len(available)!=9:raise ValueError(f"Expected 9 new dessert classes, found {len(available)}")
    extra=[{"class_id":"THDESS20_"+r["Directory_Name"],
            "menu_th":r["Dish_Name_Thai"],"source_class_id":r["Directory_Name"],
            "origin":"THDESS20_source_label"} for r in available]
    if set(x["class_id"] for x in extra)&set(existing):raise ValueError("Class ID collisions")
    lookup={r["source_class_id"]:r["class_id"] for r in extra}
    counts=Counter()
    local=Path("/content/ThaiFood129_extra.tmp.zip")
    if local.exists():local.unlink()
    print("Selecting desserts:",[(e["source_class_id"],e["menu_th"]) for e in extra],flush=True)
    with zipfile.ZipFile(src) as z,zipfile.ZipFile(local,"w",compression=zipfile.ZIP_STORED,allowZip64=True) as target:
        for member in z.infolist():
            parts=Path(member.filename).parts
            if len(parts)!=3 or member.is_dir():continue
            split,cat,filename=parts
            if split not in LIMITS or cat not in lookup or Path(filename).suffix.lower() not in EXTS:continue
            cid=lookup[cat]
            if counts[(cid,split)]>=LIMITS[split]:continue
            data=z.read(member)
            try:
                with Image.open(io.BytesIO(data)) as im:
                    im=ImageOps.exif_transpose(im).convert("RGB");im.load()
                    buf=io.BytesIO();im.save(buf,format="JPEG",quality=90)
            except Exception as e:
                print("Skipped corrupt image",member.filename,e,flush=True);continue
            target.writestr(f"{split}/{cid}/{counts[(cid,split)]:05d}.jpg",buf.getvalue())
            counts[(cid,split)]+=1
    deficits=[(cid,split,counts[cid,split]) for cid in lookup.values() for split,n in LIMITS.items() if counts[cid,split]<n]
    report={"source":str(src),"candidate_existing_classes":120,"new_unique_dessert_classes":len(extra),
            "total_image_backed_candidates":120+len(extra),"target":150,
            "remaining_unsourced_classes":150-120-len(extra),
            "splits":{s:sum(v for (c,ss),v in counts.items() if ss==s) for s in LIMITS},
            "deficits":deficits,"model_150_trained":False,
            "warnings":["New categories need semantic boundary and duplicate-image review",
                        "Existing 120 categories are preliminary-audit candidates"]}
    (out/"class_names_129.json").write_text(json.dumps(existing+[x["class_id"] for x in extra],ensure_ascii=False,indent=2),encoding="utf-8")
    with (out/"candidate_classes_150.csv").open("w",encoding="utf-8-sig",newline="") as f:
        columns=["target_index","class_id","menu_th","origin","source_class_id","status"];w=csv.DictWriter(f,fieldnames=columns);w.writeheader()
        for i,r in enumerate(selected):
            w.writerow({"target_index":i,**{k:r.get(k,"") for k in ("class_id","menu_th","origin","source_class_id")},"status":"candidate_preliminary_audit"})
        for i,r in enumerate(extra,120):
            w.writerow({"target_index":i,**r,"status":"candidate_new_dessert_needs_review"})
        for i in range(129,150):w.writerow({"target_index":i,"status":"NO_VERIFIED_IMAGES_YET"})
    (out/"progress_150.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    if deficits:
        print("Insufficient source images:",deficits)
        print("Temporary ZIP:",local,"-- not marked ready")
    else:
        dest=out/"ThaiFood129_extra_desserts.zip"
        temp=out/"ThaiFood129_extra_desserts.uploading"
        shutil.copy2(local,temp)
        if not zipfile.is_zipfile(temp):raise ValueError("Copied zip invalid")
        os.replace(temp,dest)
        print("Saved nine-class ZIP:",dest,flush=True)
    print("Report:",out/"progress_150.json")
    print("150 CLASS MODEL NOT TRAINED: remaining new classes =",report["remaining_unsourced_classes"],flush=True)

if __name__=="__main__":main()
