"""Inspect the source THFOOD-100 sample without training or guessing reference labels."""
import argparse, csv, hashlib, io, json, zipfile
from collections import Counter, defaultdict
from pathlib import Path
from PIL import Image
ROOT=Path(__file__).resolve().parent
IMAGE_EXTENSIONS={'.jpg','.jpeg','.png','.webp'}
ALIASES={'ก๋วยจั๊บ':'กวยจั๊บ','ผัดผักบุ้งไฟแดง':'ผัดผักบุ้ง'}
# These are related labels, not interchangeable image classes.
BOUNDARY={'พะโล้':'ไข่พะโล้','หอยผัดน้ำพริกเผา':'หอยลายผัดพริกเผา','สุกี้แห้ง':'สุกี้',
          'ต้มข่า':'ต้มข่าไก่','ราดหน้า':'ราดหน้าหมู','ขนมจีน':'ขนมจีนน้ำยากะทิ',
          'ไข่เจียว':'ไข่เจียวหมูสับ','สะเต๊ะ':'หมูสะเต๊ะ','ลาบ':'ลาบหมู',
          'หมูย่าง':'คอหมูย่าง','ต้มยำ':'ต้มยำกุ้ง'}

def inspect(archive, output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    taxonomy=list(csv.DictReader((ROOT/'classes_200.csv').open(encoding='utf-8-sig')))
    names={x['menu_th']:x for x in taxonomy};mapping=[];inventory=[];errors=[]
    with zipfile.ZipFile(archive) as z:
        labels=list(csv.DictReader(io.StringIO(z.read('THFOOD-100.sample/class_labels.csv').decode('utf-8-sig'))))
        valid_labels={x['Directory_Name'] for x in labels}
        counts=Counter()
        for info in z.infolist():
            path=Path(info.filename)
            if path.suffix.lower() not in IMAGE_EXTENSIONS:continue
            source_id=path.parent.name
            if source_id not in valid_labels:
                errors.append('Unknown directory: '+source_id);continue
            data=z.read(info)
            try:
                with Image.open(io.BytesIO(data)) as image:
                    image.load();w,h=image.size
                    pixels=list(image.convert('L').resize((9,8)).getdata())
                    bits=sum((pixels[r*9+c]>pixels[r*9+c+1]) << (r*8+c) for r in range(8) for c in range(8))
            except Exception as e:
                errors.append(info.filename+': '+type(e).__name__);continue
            counts[source_id]+=1
            inventory.append({'archive_path':info.filename,'source_class_id':source_id,
                'sha256':hashlib.sha256(data).hexdigest(),'dhash':f'{bits:016x}',
                'width':w,'height':h,'split':'unassigned',
                'source_url':'https://github.com/NawanolT/THFOOD-100',
                'rights_status':'dataset_license_permits_education'})
        for row in labels:
            name=row['Dish_Name_Thai'];canonical=ALIASES.get(name,name)
            target=names.get(canonical)
            related=names.get(BOUNDARY.get(name,''))
            status='exact_label_match' if target and canonical==name else ('alias_label_match' if target else ('class_boundary_review' if related else 'outside_current_taxonomy'))
            mapping.append({'source_class_id':row['Directory_Name'],'source_name_th':name,
                'source_name_en':row['Dish_Name_English'],'sample_images':counts[row['Directory_Name']],
                'mapping_status':status,'target_class_id':target['class_id'] if target else '',
                'target_name_th':target['menu_th'] if target else '',
                'related_target_class_id':related['class_id'] if related else '',
                'note':'Source-label match only; photos and ambiguous class boundaries require review before merging.'})
    hashes=defaultdict(list)
    for row in inventory:hashes[row['sha256']].append(row['archive_path'])
    mapped={r['target_class_id'] for r in mapping if r['target_class_id']}
    coverage=[{'class_id':r['class_id'],'menu_th':r['menu_th'],'origin':r['origin'],
        'sample_source_match':r['class_id'] in mapped,
        'training_ready':False,'remaining_action':'Full image data, class-boundary review, deduplication and held-out splits required'} for r in taxonomy]
    report={'archive_bytes':Path(archive).stat().st_size,'archive_sha256':hashlib.sha256(Path(archive).read_bytes()).hexdigest(),
        'source_classes':len(labels),'decoded_sample_images':len(inventory),
        'images_per_class':dict(Counter(counts.values())),
        'mapping_counts':dict(Counter(x['mapping_status'] for x in mapping)),
        'mapped_legacy_classes':sum(r['class_id'] in mapped for r in taxonomy[:50]),
        'mapped_additional_target_classes':sum(r['class_id'] in mapped for r in taxonomy[50:]),
        'uncovered_target_classes_in_this_sample':200-len(mapped),
        'exact_duplicate_groups':[v for v in hashes.values() if len(v)>1],
        'errors':errors,'ready_for_training':False,
        'note':'Sample has only five images per source class. Mapping uses source text labels; it is not a reviewed training dataset or evidence of 200-class model accuracy.',
        'full_dataset_drive_id':'1ZHwErRGhLQIHpSyKYT44Lhxbxbk3Odcs',
        'full_dataset_bytes_from_drive_metadata':16894645941}
    for filename,rows in [('thfood100_mapping.csv',mapping),('thfood100_sample_inventory.csv',inventory),('target_coverage.csv',coverage)]:
        with (output/filename).open('w',encoding='utf-8',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (output/'thfood100_inspection.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('archive');p.add_argument('--output',default=str(ROOT));a=p.parse_args();inspect(a.archive,a.output)
