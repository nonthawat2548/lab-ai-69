"""Audit a prospective 120-class dataset before any paid model training.

Reads only a local CSV and image bytes; never downloads data, starts training,
changes the production app or treats recipe-page images as a ready dataset.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

SPLITS = ('train', 'val', 'test')
RIGHTS = {'owned_with_consent', 'dataset_license_permits_education', 'permission_granted', 'public_domain', 'cc_by', 'cc_by_sa', 'cc0'}


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def audit(taxonomy_path, manifest_path, image_root, minimums=None):
    taxonomy = read_csv(taxonomy_path)
    minimums = minimums or {'train': 100, 'val': 20, 'test': 20}
    errors, warnings = [], []
    classes = [r['class_id'] for r in taxonomy]
    if len(classes) != 120 or len(set(classes)) != 120:
        errors.append('Taxonomy must contain exactly 120 distinct class IDs')
    if len({r['menu_th'].strip() for r in taxonomy}) != 120:
        errors.append('Thai menu names must be unique')
    known = set(classes)
    counts = {cid: Counter() for cid in classes}
    hash_splits, phash_splits, group_splits = defaultdict(set), defaultdict(set), defaultdict(set)
    seen_paths = set()
    raw_rows = read_csv(manifest_path)
    for number, row in enumerate(raw_rows, 2):
        tag = f'row {number}'
        cid, split = row.get('class_id', ''), row.get('split', '')
        if cid not in known or split not in SPLITS:
            errors.append(f'{tag}: unknown class ID or split')
            continue
        if row.get('human_reviewed', '').lower() != 'true':
            errors.append(f'{tag}: reference label not reviewed by a human')
        if row.get('rights_status') not in RIGHTS:
            errors.append(f'{tag}: image usage rights not documented')
        if not row.get('source_url', '').strip():
            errors.append(f'{tag}: missing source provenance')
        if not row.get('collection_group', '').strip():
            errors.append(f'{tag}: missing collection group (original photo/session)')
        path = Path(image_root) / row.get('image_path', '')
        try:
            resolved = path.resolve()
            if not resolved.is_relative_to(Path(image_root).resolve()):
                raise ValueError('image is outside dataset root')
            if not path.is_file():
                raise FileNotFoundError('image file missing')
            if str(resolved) in seen_paths:
                errors.append(f'{tag}: image path appears twice')
                continue
            seen_paths.add(str(resolved))
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if row.get('sha256') and row['sha256'] != digest:
                errors.append(f'{tag}: SHA256 does not match actual bytes')
            if digest in hash_splits:
                errors.append(f'{tag}: exact duplicate image bytes; count only one copy')
            hash_splits[digest].add(split)
            if row.get('phash'):
                phash_splits[row['phash']].add(split)
            else:
                warnings.append(f'{tag}: no perceptual hash; near-duplicate review still required')
            if row.get('collection_group'):
                group_splits[row['collection_group']].add(split)
            counts[cid][split] += 1
        except (OSError, ValueError) as exc:
            errors.append(f'{tag}: {exc}')
    for group, label in ((hash_splits, 'same image'), (phash_splits, 'same perceptual hash'), (group_splits, 'same collection group')):
        if any(len(splits) > 1 for splits in group.values()):
            errors.append(f'Leakage: {label} occurs in more than one split')
    missing = []
    for row in taxonomy:
        cid = row['class_id']
        need = {split: max(0, minimums[split] - counts[cid][split]) for split in SPLITS}
        if any(need.values()):
            missing.append({'class_id': cid, 'menu_th': row['menu_th'], 'missing': need})
    if missing:
        errors.append(f'{len(missing)} of 120 classes below configured minimum image counts')
    return {'target_classes': 120, 'ready_for_training': not errors,
            'observed_manifest_rows': len(raw_rows), 'audited_class_count': sum(sum(c.values()) > 0 for c in counts.values()),
            'minimum_counts_per_class': minimums,
            'note': 'Minimum counts are a configurable planning floor, not a guarantee of accuracy. Review ambiguous class boundaries and near-duplicates before training.',
            'counts': {k: {split: v[split] for split in SPLITS} for k, v in counts.items()},
            'missing_classes': missing, 'errors': errors, 'warnings': warnings}


def main():
    parser = argparse.ArgumentParser()
    here = Path(__file__).resolve().parent
    parser.add_argument('--taxonomy', default=str(here / 'classes_120.csv'))
    parser.add_argument('--manifest', default=str(here / 'images_manifest_template.csv'))
    parser.add_argument('--image-root', default='.')
    parser.add_argument('--output')
    parser.add_argument('--min-train', type=int, default=100)
    parser.add_argument('--min-val', type=int, default=20)
    parser.add_argument('--min-test', type=int, default=20)
    args = parser.parse_args()
    if min(args.min_train, args.min_val, args.min_test) < 1:
        parser.error('All split minimums must be positive')
    report = audit(args.taxonomy, args.manifest, args.image_root,
                   {'train': args.min_train, 'val': args.min_val, 'test': args.min_test})
    if args.output:
        Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({k: report[k] for k in ('target_classes', 'ready_for_training', 'observed_manifest_rows', 'audited_class_count', 'errors')}, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report['ready_for_training'] else 2)

if __name__ == '__main__':
    main()
