"""Discover reviewable Wikimedia Commons candidates for 21 distinct Thai dishes.
CPU-only. Does NOT train, auto-label, or claim 150 image-backed classes.
Saves license/provenance CSV and downloaded small previews for human review.
Usage: python discover_remaining_21.py --base /content/thaifood_drive/MyDrive/ThaiFood150
"""
import argparse,csv,html,json,re,time,urllib.parse
from pathlib import Path
import requests

DISHES=[
 ("TH150_KhanomChan","ขนมชั้น","khanom chan"),
 ("TH150_ThongYip","ทองหยิบ","thong yip Thai dessert"),
 ("TH150_ThongYod","ทองหยอด","thong yot Thai dessert"),
 ("TH150_FoiThong","ฝอยทอง","foi thong Thai dessert"),
 ("TH150_MetKhanoon","เม็ดขนุน","met khanun Thai dessert"),
 ("TH150_KhanomTan","ขนมตาล","khanom tan"),
 ("TH150_KhanomKluay","ขนมกล้วย","khanom kluay"),
 ("TH150_KhanomSaiSai","ขนมใส่ไส้","khanom sai sai"),
 ("TH150_KhanomPiakPoon","ขนมเปียกปูน","khanom piak pun"),
 ("TH150_KhanomMoKaeng","ขนมหม้อแกง","khanom mo kaeng"),
 ("TH150_KhaoNiewDurian","ข้าวเหนียวทุเรียน","durian sticky rice Thai"),
 ("TH150_SangkhayaFakThong","สังขยาฟักทอง","sankaya fak thong"),
 ("TH150_KhanomTom","ขนมต้ม","khanom tom"),
 ("TH150_KhanomTian","ขนมเทียน","khanom tian"),
 ("TH150_KhaoNiewNaKung","ข้าวเหนียวหน้ากุ้ง","khao niao na kung"),
 ("TH150_KhanomLa","ขนมลา","khanom la"),
 ("TH150_KhanomDokJok","ขนมดอกจอก","khanom dok jok"),
 ("TH150_KhanomWong","ขนมวง","khanom wong"),
 ("TH150_KhrongKhraeng","ครองแครง","khrong khraeng Thai dessert"),
 ("TH150_KhaoNiewSangkhaya","ข้าวเหนียวสังขยา","khao niao sangkhaya"),
 ("TH150_KhaoNiewPing","ข้าวเหนียวปิ้ง","khao niao ping")
]
API="https://commons.wikimedia.org/w/api.php"
HEADERS={"User-Agent":"ThaiFoodAIResearchDiscovery/0.1 (educational image dataset audit; contact via github.com/nonthawat2548/lab-ai-69)"}
def clean(value):
    return re.sub(r"<[^>]*>"," ",str(value or "")).strip()
def discover(session,term,limit):
    results=[];seen=set()
    # Commons search is candidate discovery, not a trustworthy class label.
    for q in (term,):
        cont={}
        while len(results)<limit:
            params={"action":"query","format":"json","generator":"search",
                "gsrsearch":q+" filetype:bitmap","gsrnamespace":"6","gsrlimit":"50",
                "prop":"imageinfo","iiprop":"url|extmetadata|mime",
                "iiurlwidth":"360",**cont}
            response=session.get(API,params=params,timeout=40)
            response.raise_for_status()
            payload=response.json()
            for page in payload.get("query",{}).get("pages",{}).values():
                info=(page.get("imageinfo") or [{}])[0]
                name=page.get("title","")
                if name in seen:continue
                mime=info.get("mime","")
                if mime not in ("image/jpeg","image/png","image/webp"):continue
                seen.add(name)
                ext=info.get("extmetadata",{})
                def md(k):return clean(ext.get(k,{}).get("value",""))
                results.append({"commons_title":name,"commons_page":info.get("descriptionurl",""),
                    "image_url":info.get("url",""),"preview_url":info.get("thumburl",""),
                    "license":md("LicenseShortName"),"license_url":md("LicenseUrl"),
                    "attribution":md("Artist"),"restrictions":md("Restrictions"),
                    "image_description":md("ImageDescription")[:250]})
                if len(results)>=limit:break
            cont=payload.get("continue",{})
            if not cont:break
            time.sleep(.2)
    return results

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--base",default="/content/thaifood_drive/MyDrive/ThaiFood150")
    parser.add_argument("--max-preview",type=int,default=8)
    parser.add_argument("--max-candidates",type=int,default=60)
    args=parser.parse_args()
    if not (1<=args.max_preview<=24 and 1<=args.max_candidates<=120):
        raise ValueError("Preview max 1..24, candidates max 1..120")
    base=Path(args.base)
    prepared=base/"prepared"
    required=prepared/"candidate_classes_150.csv"
    oldnames=prepared/"class_names_129.json"
    if not required.exists() or not oldnames.exists():
        raise FileNotFoundError("Need prepared/candidate_classes_150.csv and class_names_129.json")
    prior=json.loads(oldnames.read_text(encoding="utf-8"))
    if len(prior)!=129 or len(set(prior))!=129:raise ValueError("Expected 129 distinct prior class IDs")
    existing_rows=list(csv.DictReader(required.open(encoding="utf-8-sig",newline="")))
    prev_th={r["menu_th"].strip() for r in existing_rows[:129] if r["menu_th"]}
    overlap=[th for _,th,_ in DISHES if th in prev_th]
    if overlap:raise ValueError("Possible duplicate Thai dish names: "+str(overlap))
    out=base/"discovery_150"
    previews=out/"previews";previews.mkdir(parents=True,exist_ok=True)
    session=requests.Session();session.headers.update(HEADERS)
    records=[];counts=[]
    for idx,(cid,th,query) in enumerate(DISHES,130):
        print(f"[{idx-129}/21] Discovering {th}: {query}",flush=True)
        try:
            found=discover(session,query,args.max_candidates)
        except Exception as exc:
            print("Search failed:",exc,flush=True)
            found=[]
        downloaded=0
        folder=previews/cid;folder.mkdir(exist_ok=True)
        for j,item in enumerate(found):
            record={"class_id":cid,"menu_th":th,"english_search":query,
                    "human_review":"PENDING","usable_for_training":"NO",**item,
                    "local_preview":""}
            if downloaded<args.max_preview and item["preview_url"]:
                try:
                    r=session.get(item["preview_url"],timeout=35)
                    r.raise_for_status()
                    if r.headers.get("Content-Type","").startswith("image/") and len(r.content)<5_000_000:
                        ext=".png" if "png" in r.headers.get("Content-Type","") else ".jpg"
                        local=folder/f"{j:03d}{ext}"
                        local.write_bytes(r.content)
                        record["local_preview"]=str(local.relative_to(out))
                        downloaded+=1
                except Exception as exc:
                    print(" preview failed:",str(exc)[:150],flush=True)
                time.sleep(.12)
            records.append(record)
        counts.append({"class_id":cid,"menu_th":th,"search_candidates":len(found),
                       "preview_downloaded":downloaded,"verified_training_images":0})
        print(" candidates:",len(found),"previews:",downloaded,flush=True)
    columns=["class_id","menu_th","english_search","human_review","usable_for_training",
        "commons_title","commons_page","image_url","preview_url","license","license_url",
        "attribution","restrictions","image_description","local_preview"]
    with (out/"candidate_images.csv").open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=columns);w.writeheader();w.writerows(records)
    with (out/"candidate_counts.csv").open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=counts[0].keys());w.writeheader();w.writerows(counts)
    with (out/"candidate_21_classes.csv").open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.writer(f);w.writerow(["target_index","class_id","menu_th","english_search","status"])
        for i,(cid,th,q) in enumerate(DISHES,129):
            w.writerow([i,cid,th,q,"PENDING_IMAGE_AND_LICENSE_REVIEW"])
    cards=[]
    for record in records:
        if not record["local_preview"]:continue
        src=urllib.parse.quote(record["local_preview"],safe="/")
        page=html.escape(record["commons_page"],quote=True)
        caption=html.escape(record["commons_title"][:65])
        cards.append(f'<figure><img src="{src}"><figcaption>{html.escape(record["menu_th"])}<br><a href="{page}">{caption}</a><br>{html.escape(record["license"])}</figcaption></figure>')
    (out/"review_gallery.html").write_text("""<!doctype html><html lang="th"><meta charset="utf-8">
    <title>ThaiFood 150 - review only</title><style>body{font-family:Arial,sans-serif}
    main{display:grid;grid-template-columns:repeat(auto-fill,minmax(190px,1fr));gap:10px}
    img{width:100%;height:150px;object-fit:cover}figure{margin:0;padding:8px;border:1px solid #ddd}
    figcaption{font-size:12px;overflow-wrap:anywhere}</style><h2>ภาพผู้สมัคร - ไม่ใช่ชุดฝึกที่ตรวจผ่านแล้ว</h2><main>"""+
    "\n".join(cards)+"</main></html>",encoding="utf-8")
    print("\nREADY FOR HUMAN REVIEW ONLY, no labels are verified and no model training was run")
    print("Image candidates:",out/"candidate_images.csv")
    print("Counts:",out/"candidate_counts.csv")
    print("Gallery:",out/"review_gallery.html")
    print("Review licenses, label accuracy, duplicates, attribution and splits before training.")
if __name__=="__main__":main()
