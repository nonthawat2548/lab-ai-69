"""Blind external image evaluation from private Google Forms responses.

Inference boundary: predict_images accepts only local file paths and image IDs.
No reference menu, spreadsheet row text, or gold label is passed to the model.
All response/image data and results stay in the user's Google Drive, never GitHub.
"""
from __future__ import annotations
import csv
import hashlib
import io
import json
import re
import unicodedata
from pathlib import Path
from urllib.parse import urlparse, parse_qs

LABEL_FIELDS = ['file_id', 'source_row', 'image_index', 'reported_menu',
                'gold_label', 'scope', 'human_verified', 'notes']
PIPELINE_VERSION = 'efficientnet_b0_resize256_centercrop224_imagenetnorm_cpu_v1'


def normalized(text):
    return re.sub(r'\s+', '', unicodedata.normalize('NFKC', str(text or '')).strip())


def file_ids(text):
    """Parse only supported Drive links, not arbitrary URL downloads."""
    found = []
    for url in re.findall(r'https://(?:drive\.google\.com|docs\.google\.com)/[^\s,]+', str(text or '')):
        p = urlparse(url)
        value = parse_qs(p.query).get('id', [''])[0]
        if not value:
            m = re.search(r'/d/([\w-]+)', p.path)
            value = m.group(1) if m else ''
        if re.fullmatch(r'[A-Za-z0-9_-]{10,}', value) and value not in found:
            found.append(value)
    return found


def menu_aliases(class_ids, thai_names):
    lookup = {normalized(c): c for c in class_ids}
    lookup.update({normalized(thai_names[c]): c for c in class_ids})
    # Only unambiguous spelling variants; do not infer unknown dishes from partial strings.
    aliases = {'ก๋วยจั๊บ': 'KuayJab', 'ข้าวกะเพรา': 'PhatKaphrao',
               'ข้าวผัดกะเพรา': 'PhatKaphrao', 'กระเพรา': 'PhatKaphrao',
               'แกงเขียวหวานไก่': 'GaengKeawWan'}
    lookup.update({normalized(k): v for k, v in aliases.items() if v in class_ids})
    return lookup


def parse_responses(rows, lookup):
    """Read C:I only: no timestamps, email addresses or names are requested."""
    if not rows:
        raise ValueError('ชีตคำตอบว่าง')
    headers = rows[0]
    def col(fragment):
        hits = [i for i, text in enumerate(headers) if fragment in str(text)]
        if len(hits) != 1:
            raise ValueError(f'ไม่พบหัวคอลัมน์ชัดเจน: {fragment}')
        return hits[0]
    consent = col('ยินยอมให้ใช้ภาพ')
    photo = col('อัปโหลดภาพ')
    menu = col('ชื่อเมนูอาหารในภาพ')
    rights = col('เป็นผู้ถ่ายภาพนี้เอง')
    manifest, labels, skipped = [], [], []
    seen = set()
    for rownum, row in enumerate(rows[1:], 2):
        at = lambda i: str(row[i]) if i < len(row) else ''
        if normalized(at(consent)) not in ('ยินยอม', 'ยินยอมให้ใช้ข้อมูล'):
            skipped.append({'source_row': rownum, 'reason': 'consent_not_granted'})
            continue
        if not at(rights).strip().startswith('ใช่'):
            skipped.append({'source_row': rownum, 'reason': 'image_rights_not_confirmed'})
            continue
        ids = file_ids(at(photo))
        if not ids:
            skipped.append({'source_row': rownum, 'reason': 'no_drive_image_link'})
        for image_index, fid in enumerate(ids, 1):
            if fid in seen:
                skipped.append({'source_row': rownum, 'reason': 'repeated_file_id'})
                continue
            seen.add(fid)
            # This manifest is the ONLY image input sent to the download/prediction stage.
            manifest.append({'file_id': fid, 'source_row': rownum, 'image_index': image_index})
            reported = at(menu).strip()
            candidate = lookup.get(normalized(reported), '')
            labels.append({'file_id': fid, 'source_row': rownum, 'image_index': image_index,
                           'reported_menu': reported, 'gold_label': candidate,
                           'scope': 'in_scope' if candidate else 'pending',
                           'human_verified': 'false', 'notes': ''})
    return manifest, labels, skipped


def read_csv(path):
    if not Path(path).exists():
        return []
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def save_labels(path, rows):
    with Path(path).open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=LABEL_FIELDS)
        w.writeheader()
        w.writerows({k: row.get(k, '') for k in LABEL_FIELDS} for row in rows)


def merge_labels(new_rows, saved_rows):
    """Preserve human annotations as Form responses grow; never fill from predictions."""
    old = {r['file_id']: r for r in saved_rows}
    merged = []
    for row in new_rows:
        prior = old.get(row['file_id'], {})
        updated = dict(row)
        for field in ('gold_label', 'scope', 'human_verified', 'notes'):
            if field in prior:
                updated[field] = prior[field]
        merged.append(updated)
    return merged


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def download_images(drive_api, manifest, folder):
    from googleapiclient.http import MediaIoBaseDownload
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    images, failures = [], []
    for row in manifest:
        fid = row['file_id']
        destination = folder / f'{fid}.img'
        try:
            if not destination.exists():
                temp = destination.with_suffix('.part')
                try:
                    with temp.open('wb') as out:
                        downloader = MediaIoBaseDownload(out, drive_api.files().get_media(fileId=fid), chunksize=1 << 20)
                        done = False
                        while not done:
                            _, done = downloader.next_chunk(num_retries=2)
                            if out.tell() > 40 * 1024 * 1024:
                                raise ValueError('ภาพใหญ่กว่า 40MB')
                    temp.replace(destination)
                except Exception:
                    temp.unlink(missing_ok=True)
                    raise
            images.append({'file_id': fid, 'path': str(destination)})
        except Exception as exc:
            failures.append({'file_id': fid, 'status': 'download_error', 'error_type': type(exc).__name__})
    return images, failures


def predict_images(model, transform, images, class_ids, checkpoint_hash, cache_file):
    """Blind inference. Deliberately rejects any record carrying a reference label."""
    for image in images:
        if set(image) != {"file_id", "path"}:
            raise ValueError("Blind inference accepts only file_id and image path, never gold labels")
    from PIL import Image, ImageOps
    import torch
    output = []
    existing = json.loads(Path(cache_file).read_text()) if Path(cache_file).exists() else []
    cache = {(r.get('checkpoint_sha256'), r.get('image_sha256'), r.get('pipeline')): r for r in existing}
    model.cpu().eval()
    for image in images:
        if set(image) != {'file_id', 'path'}:
            raise ValueError('Blind inference accepts only file_id and image path, never gold labels')
        file_id, path = image['file_id'], image['path']
        digest = sha256_file(path)
        key = (checkpoint_hash, digest, PIPELINE_VERSION)
        if key in cache:
            prediction = {**cache[key], 'file_id': file_id}
        else:
            prediction = {'file_id': file_id, 'image_sha256': digest,
                          'checkpoint_sha256': checkpoint_hash, 'pipeline': PIPELINE_VERSION}
            try:
                with Image.open(path) as img:
                    tensor = transform(ImageOps.exif_transpose(img).convert('RGB')).unsqueeze(0)
                with torch.inference_mode():
                    probabilities = torch.softmax(model(tensor), dim=1)[0].cpu()
                order = probabilities.topk(min(5, len(class_ids)))
                prediction.update({'status': 'ok', 'predicted_class': class_ids[int(order.indices[0])],
                    'confidence': float(order.values[0]),
                    'top5': [{'class_id': class_ids[int(i)], 'score': float(v)} for i, v in zip(order.indices, order.values)]})
            except Exception as exc:
                prediction.update({'status': 'inference_error', 'error_type': type(exc).__name__})
            cache[key] = prediction
        output.append(prediction)
        # Resume safely after a disconnect; never needs to predict old images again.
        Path(cache_file).write_text(json.dumps(list(cache.values()), ensure_ascii=False, indent=2))
    return output


def score_predictions(predictions, labels, class_ids):
    """Gold labels are accessed only AFTER blind predictions have been generated."""
    pred = {r['file_id']: r for r in predictions}
    class_set = set(class_ids)
    rows, truth, guessed, seen = [], [], [], set()
    counts = {'pending_human_review': 0, 'out_of_scope': 0, 'excluded_by_reviewer': 0,
              'download_or_inference_error': 0, 'duplicate_image_bytes': 0, 'scored': 0}
    for row in labels:
        result = {k: row.get(k, '') for k in LABEL_FIELDS}
        p = pred.get(row['file_id'], {})
        verified = str(row.get('human_verified', '')).lower() == 'true'
        scope, gold = row.get('scope', 'pending'), row.get('gold_label', '')
        if not verified:
            status = 'pending_human_review'
        elif scope == 'out_of_scope':
            status = 'out_of_scope'
        elif scope == 'exclude':
            status = 'excluded_by_reviewer'
        elif scope != 'in_scope' or gold not in class_set:
            status = 'pending_human_review'
        elif p.get('status') != 'ok':
            status = 'download_or_inference_error'
        elif p['image_sha256'] in seen:
            status = 'duplicate_image_bytes'
        else:
            status = 'scored'
            seen.add(p['image_sha256'])
            truth.append(gold)
            guessed.append(p['predicted_class'])
        counts[status] += 1
        result.update({'evaluation_status': status, 'predicted_class': p.get('predicted_class', ''),
                       'confidence': p.get('confidence', ''), 'correct': gold == p.get('predicted_class') if status == 'scored' else ''})
        rows.append(result)
    present = [c for c in class_ids if c in set(truth)]
    metrics = {'evaluation': 'external_form_photos_blind_inference', 'counts': counts,
               'evaluated_reference_classes': present, 'n_evaluated': len(truth),
               'accuracy': None, 'macro_f1_present_classes': None, 'macro_f1_all_50_classes': None,
               'note': 'Only human-verified, in-scope, unique images scored. Top-1 errors count even if confidence is low. Not RAG evaluation.'}
    matrix = None
    if truth:
        index = {c: i for i, c in enumerate(class_ids)}
        matrix = [[0 for _ in class_ids] for _ in class_ids]
        for gold, guess in zip(truth, guessed):
            matrix[index[gold]][index[guess]] += 1
        f1_values = {}
        for c, i in index.items():
            tp = matrix[i][i]
            fp = sum(matrix[j][i] for j in range(len(class_ids))) - tp
            fn = sum(matrix[i]) - tp
            denominator = 2 * tp + fp + fn
            f1_values[c] = 2 * tp / denominator if denominator else 0.0
        metrics.update({'accuracy': sum(a == b for a, b in zip(truth, guessed)) / len(truth),
                        'macro_f1_present_classes': sum(f1_values[c] for c in present) / len(present),
                        'macro_f1_all_50_classes': sum(f1_values.values()) / len(class_ids)})
    return metrics, rows, matrix


def connect_and_predict(spreadsheet_id, sheet_name, class_thai):
    """Colab entry point: private, CPU only, no training and no public upload."""
    from google.colab import auth, drive
    import google.auth
    from googleapiclient.discovery import build
    import torch
    from torchvision import models, transforms
    drive.mount('/content/drive', force_remount=False)
    auth.authenticate_user()
    credentials, _ = google.auth.default()
    sheets_api = build('sheets', 'v4', credentials=credentials, cache_discovery=False)
    drive_api = build('drive', 'v3', credentials=credentials, cache_discovery=False)
    area = "'" + sheet_name.replace("'", "''") + "'!C1:I10000"
    response = sheets_api.spreadsheets().values().get(spreadsheetId=spreadsheet_id, range=area).execute()
    project = Path('/content/drive/MyDrive/ThaiFoodAI_THFOOD50')
    checkpoint = project / 'food_model_best.pth'
    if not checkpoint.exists():
        raise FileNotFoundError('ไม่พบโมเดลที่ฝึกแล้วใน Google Drive; หยุดโดยไม่ฝึกใหม่')
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    classes = list(saved['class_names'])
    if len(classes) != 50 or set(classes) != set(class_thai):
        raise ValueError('Checkpoint และรายชื่อคลาสไม่ตรงกัน')
    workspace = project / 'external_form_test' / spreadsheet_id
    workspace.mkdir(parents=True, exist_ok=True)
    manifest, labels, skipped = parse_responses(response.get('values', []), menu_aliases(classes, class_thai))
    labels_path = workspace / 'gold_labels_private.csv'
    save_labels(labels_path, merge_labels(labels, read_csv(labels_path)))
    (workspace / 'source_manifest_private.json').write_text(json.dumps({'spreadsheet_id': spreadsheet_id,
        'sheet_name': sheet_name, 'images': manifest, 'skipped_rows': skipped}, ensure_ascii=False, indent=2))
    # Deliberate boundary: metadata/labels are never supplied to this model or its transform.
    images, errors = download_images(drive_api, manifest, workspace / 'image_cache')
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, len(classes))
    model.load_state_dict(saved['model'], strict=True)
    transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
        transforms.Normalize([.485, .456, .406], [.229, .224, .225])])
    predictions = predict_images(model, transform, images, classes, sha256_file(checkpoint), workspace / 'prediction_cache_private.json')
    predictions.extend(errors)
    (workspace / 'blind_predictions_private.json').write_text(json.dumps(predictions, ensure_ascii=False, indent=2))
    context = {'workspace': workspace, 'labels_path': labels_path, 'images': images,
               'predictions': predictions, 'classes': classes, 'class_thai': class_thai}
    print(f'นำเข้ารูป {len(manifest)} รูป | ทำนายได้ {sum(x.get("status")=="ok" for x in predictions)} รูป | ข้ามแถวไม่ยินยอม/ซ้ำ {len(skipped)} รายการ')
    print('โมเดลได้รับเฉพาะภาพแล้ว ผลถูกบันทึกก่อนอ่านเฉลยเพื่อคิดคะแนน ไม่ฝึกใหม่ ไม่ส่งรูปไปเว็บสาธารณะ')
    print('ไฟล์และรูปอยู่ใน Drive ส่วนตัว:', workspace)
    return context


def save_results(context):
    import pandas as pd
    metrics, rows, matrix = score_predictions(context['predictions'], read_csv(context['labels_path']), context['classes'])
    root = context['workspace']
    (root / 'external_image_metrics.json').write_text(json.dumps(metrics, ensure_ascii=False, indent=2))
    pd.DataFrame(rows).to_csv(root / 'external_image_results_private.csv', index=False, encoding='utf-8-sig')
    if matrix is not None:
        pd.DataFrame(matrix, index=context['classes'], columns=context['classes']).to_csv(root / 'external_confusion_matrix.csv', encoding='utf-8-sig')
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    if not metrics['n_evaluated']:
        print('ยังไม่มีภาพที่ยืนยันเฉลยครบ จึงไม่สร้างคะแนน Accuracy/F1 สมมติ')
    return metrics


def review_labels(context):
    """Human reference annotation screen; prediction results deliberately not displayed."""
    import ipywidgets as widgets
    from IPython.display import display, clear_output
    from PIL import Image, ImageOps
    labels = read_csv(context['labels_path'])
    paths = {i['file_id']: i['path'] for i in context['images']}
    choices = [('— ยังไม่ระบุเฉลย —', '')] + [(context['class_thai'][c], c) for c in context['classes']]
    controls = []
    for index, row in enumerate(labels, 1):
        thumb = widgets.HTML('เปิดรูปไม่สำเร็จ')
        if row['file_id'] in paths:
            try:
                with Image.open(paths[row['file_id']]) as raw:
                    im = ImageOps.exif_transpose(raw).convert('RGB')
                    im.thumbnail((400, 260))
                    buffer = io.BytesIO()
                    im.save(buffer, format='JPEG')
                thumb = widgets.Image(value=buffer.getvalue(), format='jpg')
            except Exception:
                pass
        reported = widgets.Label(f'ภาพ {index} — ข้อความชื่อเมนูจากฟอร์ม: {row["reported_menu"]}')
        gold = widgets.Dropdown(options=choices, value=row['gold_label'] if row['gold_label'] in context['classes'] else '', description='เฉลยจริง:')
        scope = widgets.Dropdown(options=[('รอตรวจ', 'pending'), ('อยู่ใน 50 เมนู', 'in_scope'),
            ('อยู่นอก 50 เมนู', 'out_of_scope'), ('ตัดออก (ใส่เหตุผล)', 'exclude')], value=row.get('scope', 'pending'), description='ขอบเขต:')
        verified = widgets.Checkbox(value=row['human_verified'].lower() == 'true', description='ตรวจเฉลยจากรูปนี้แล้ว (ไม่อิงผลที่ AI ทาย)')
        notes = widgets.Text(value=row.get('notes', ''), description='หมายเหตุ:')
        controls.append((row, gold, scope, verified, notes))
        display(widgets.VBox([reported, thumb, gold, scope, verified, notes]))
    status = widgets.Output()
    save = widgets.Button(description='บันทึกเฉลยแล้วคิดคะแนน', button_style='success', layout=widgets.Layout(width='240px'))
    def clicked(_):
        with status:
            clear_output()
            updated = []
            for row, gold, scope, verified, notes in controls:
                r = {**row, 'gold_label': gold.value, 'scope': scope.value,
                     'human_verified': str(verified.value).lower(), 'notes': notes.value}
                if verified.value and scope.value == 'in_scope' and not gold.value:
                    print('กรุณาระบุเฉลยจริงสำหรับภาพที่อยู่ใน 50 เมนูก่อน');return
                if verified.value and scope.value == 'exclude' and not notes.value.strip():
                    print('กรุณาใส่เหตุผลของภาพที่ตัดออกก่อน');return
                updated.append(r)
            save_labels(context['labels_path'], updated)
            save_results(context)
    save.on_click(clicked)
    display(widgets.HTML('<b>ไม่มีการแสดงผลคำทำนายในหน้าทำเฉลยนี้ เพื่อหลีกเลี่ยงการเฉลยตาม AI</b>'), save, status)
    return controls
