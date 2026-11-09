"""Assemble a reproducible source-split subset; Forms photos are never read."""
import csv,hashlib,io,json,sys,zipfile
from collections import Counter
from pathlib import Path
from PIL import Image,ImageOps
ROOT=Path(__file__).resolve().parent
FIELDS=['image_path','class_id','split','source_url','collection_group','rights_status','human_reviewed','sha256','phash','source_sha256','label_review_basis']

def save(data,cid,split,key,source,out):
    dest=out/'images'/cid/split/(hashlib.sha256(key.encode()).hexdigest()+'.jpg')
    dest.parent.mkdir(parents=True,exist_ok=True)
    with Image.open(io.BytesIO(data)) as image:
        image=ImageOps.exif_transpose(image).convert('RGB');image.thumbnail((640,640))
        pixels=list(image.convert('L').resize((9,8)).getdata())
        dh=sum(int(pixels[r*9+c]>pixels[r*9+c+1])<<(r*8+c) for r in range(8) for c in range(8))
        image.save(dest,quality=95,subsampling=0)
    return dict(zip(FIELDS,[str(dest.relative_to(out)),cid,split,source,source+':'+hashlib.sha256(data).hexdigest(),'dataset_license_permits_education','true',hashlib.sha256(dest.read_bytes()).hexdigest(),f'{dh:016x}',hashlib.sha256(data).hexdigest(),'original_source_expert_label; not individually relabeled in this project']))

def build(out,source):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    rows=list(csv.DictReader((ROOT/'classes_120.csv').open()));manifest=[]
    if source=='legacy':
        import pyarrow.parquet as pq
        legacy=[r['class_id'] for r in rows[:50]]
        for split,cap in [('train',100),('val',20),('test',20)]:
            counts=Counter();position=0
            files=sorted(Path('/tmp/thfood50_parquet/data').glob(split+'-*.parquet'))
            if not files:raise ValueError('Download source parquet files first')
            for file in files:
                parquet=pq.ParquetFile(file)
                info=json.loads(parquet.schema_arrow.metadata[b'huggingface'])
                names=info['info']['features']['label']['names']
                if set(names)!=set(legacy):raise ValueError('Original dataset labels differ from checkpoint')
                for batch in parquet.iter_batches(batch_size=128):
                    for row in batch.to_pylist():
                        i=position;position+=1;cid=names[row['label']]
                        if counts[cid]>=cap:continue
                        data=row['image']['bytes']
                        if not data:raise ValueError('Expected embedded source image bytes')
                        manifest.append(save(data,cid,split,f'legacy/{split}/{i}','https://huggingface.co/datasets/thean/THFOOD-50',out));counts[cid]+=1
                        if len(manifest)%500==0:print('legacy saved',len(manifest),flush=True)
            print('legacy split',split,'total',sum(counts.values()),'minimum',min(counts.values()),flush=True)
    else:
        filename='/tmp/THFOOD-100.zip' if source=='thfood100' else '/tmp/THDESS-20.zip'
        selected=[r for r in rows if r['origin']==('THFOOD100_source_label' if source=='thfood100' else 'THDESS20_source_label')]
        lookup={r['source_class_id'].zfill(2) if source=='thfood100' else r['source_class_id']:r['class_id'] for r in selected}
        with zipfile.ZipFile(filename) as z:
            counts=Counter()
            for info in sorted(z.infolist(),key=lambda x:x.filename):
                path=Path(info.filename)
                if path.suffix.lower() not in ('.jpg','.jpeg','.png','.webp'):continue
                split=path.parts[0];src=path.parent.name
                if split not in ('train','val','test') or src not in lookup:continue
                cid=lookup[src];cap=100 if split=='train' else 20
                if counts[cid,split]>=cap:continue
                data=z.read(info);url='https://github.com/NawanolT/'+('THFOOD-100' if source=='thfood100' else 'THDESS-20')
                manifest.append(save(data,cid,split,source+'/'+info.filename,url,out));counts[cid,split]+=1
                if len(manifest)%500==0:print(source,'saved',len(manifest),flush=True)
    with (out/f'manifest_{source}.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=FIELDS);w.writeheader();w.writerows(manifest)
    print('DONE',source,len(manifest),flush=True)

if __name__=='__main__':build(sys.argv[1],sys.argv[2])
