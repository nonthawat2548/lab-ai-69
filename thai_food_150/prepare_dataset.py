"""Prepare 150 food classes (50 original + 100 new) for ImageFolder."""
import argparse, hashlib, json, random
from pathlib import Path
from PIL import Image, ImageOps
import torch

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--raw",default="raw_images")
    p.add_argument("--out",default="data")
    p.add_argument("--checkpoint",default="food_model_best.pth")
    p.add_argument("--min-images",type=int,default=20)
    a=p.parse_args()
    old=torch.load(a.checkpoint,map_location="cpu",weights_only=False)["class_names"]
    if len(old)!=50: raise ValueError("Expected 50 original classes")
    folders={d.name:d for d in Path(a.raw).iterdir() if d.is_dir()}
    missing=set(old)-set(folders)
    if missing: raise ValueError("Missing original class photos: "+", ".join(sorted(missing)))
    names=list(old)+sorted(set(folders)-set(old))
    if len(names)!=150: raise ValueError(f"Expected 150 unique classes, found {len(names)}. Resolve synonyms and duplicates.")
    out=Path(a.out)
    if out.exists() and any(out.iterdir()): raise FileExistsError(f"Output not empty: {out}")
    rng=random.Random(42); seen=set(); report={}
    for name in names:
        photos=[]
        for f in sorted(folders[name].rglob("*")):
            if f.suffix.lower() not in (".jpg",".jpeg",".png",".webp") or not f.is_file(): continue
            try:
                with Image.open(f) as im:
                    im=ImageOps.exif_transpose(im).convert("RGB")
                    if min(im.size)<96: continue
                    im.thumbnail((512,512))
                    digest=hashlib.sha256(im.tobytes()+str(im.size).encode()).hexdigest()
                if digest not in seen: seen.add(digest);photos.append(f)
            except Exception as e: print("Skipping",f,e)
        if len(photos)<a.min_images: raise ValueError(f"{name}: only {len(photos)} valid unique images")
        rng.shuffle(photos)
        n=len(photos);nt=max(1,round(n*.10));nv=max(1,round(n*.15))
        groups={"train":photos[:n-nt-nv],"val":photos[n-nt-nv:n-nt],"test":photos[n-nt:]}
        report[name]={k:len(v) for k,v in groups.items()}
        for split,items in groups.items():
            dest=out/split/name;dest.mkdir(parents=True,exist_ok=True)
            for i,src in enumerate(items):
                with Image.open(src) as im: ImageOps.exif_transpose(im).convert("RGB").save(dest/f"{i:06d}.jpg",quality=92)
    out.mkdir(parents=True,exist_ok=True)
    (out/"class_names.json").write_text(json.dumps(names,ensure_ascii=False,indent=2),encoding="utf-8")
    (out/"report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print("Prepared",len(names),"classes")
if __name__=="__main__":main()
