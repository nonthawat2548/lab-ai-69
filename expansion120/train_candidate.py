"""Train an auditable 120-class EfficientNet candidate; never deploy automatically.

Input images must pass audit_dataset.py. Features are cached to avoid repeating the
expensive backbone pass. The legacy production checkpoint is read-only.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from audit_dataset import audit, read_csv

HERE = Path(__file__).resolve().parent


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def feature_fingerprint(rows, root, checkpoint, class_ids):
    """Bind cache to image content, labels, splits, model weights and preprocessing."""
    digest = hashlib.sha256()
    digest.update(b'efficientnet_b0_imagenet1k_v1_eval_resize256_crop224_v1')
    digest.update(sha256_file(checkpoint).encode())
    digest.update(json.dumps(class_ids, ensure_ascii=False).encode())
    for row in rows:
        digest.update(json.dumps([row['image_path'], row['class_id'], row['split'],
                                  sha256_file(Path(root) / row['image_path'])], ensure_ascii=False).encode())
    return digest.hexdigest()


def fit_head(train_x, train_y, val_x, val_y, class_count, epochs=20, seed=42):
    """Fit a linear softmax-like head, selecting using validation macro F1 only."""
    from copy import deepcopy
    if epochs < 1:
        raise ValueError('epochs must be positive')
    expected = np.arange(class_count)
    if not np.array_equal(np.unique(train_y), expected):
        raise ValueError('training split must cover every class')
    if set(np.unique(val_y)) != set(expected):
        raise ValueError('validation split must cover every class')
    model = SGDClassifier(loss='log_loss', alpha=0.0001, random_state=seed,
                          average=False, learning_rate='optimal')
    rng = np.random.default_rng(seed)
    best, best_f1, history = None, -1.0, []
    for epoch in range(1, epochs + 1):
        order = rng.permutation(len(train_y))
        model.partial_fit(train_x[order], train_y[order], classes=expected)
        pred = model.predict(val_x)
        score = float(f1_score(val_y, pred, labels=expected, average='macro', zero_division=0))
        history.append({'epoch': epoch, 'val_accuracy': float(accuracy_score(val_y, pred)), 'val_macro_f1': score})
        if score > best_f1:
            best, best_f1 = deepcopy(model), score
    return best, history


def extract_features(rows, root, model, device, batch_size, workers):
    import torch
    from PIL import Image, ImageOps
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms
    # Match current_app.VALID_TF so the saved candidate uses the live app's preprocessing.
    transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([.485, .456, .406], [.229, .224, .225])])

    class Images(Dataset):
        def __len__(self):
            return len(rows)
        def __getitem__(self, index):
            with Image.open(Path(root) / rows[index]['image_path']) as image:
                return transform(ImageOps.exif_transpose(image).convert('RGB'))

    loader = DataLoader(Images(), batch_size=batch_size, shuffle=False,
                        num_workers=workers, pin_memory=device.type == 'cuda')
    model.eval().to(device)
    blocks = []
    with torch.inference_mode():
        for batch in loader:
            x = model.features(batch.to(device))
            x = model.avgpool(x).flatten(1)
            blocks.append(x.cpu().numpy())
            if len(blocks) % 20 == 0: print('feature batches', len(blocks), '/', len(loader), flush=True)
    return np.concatenate(blocks).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--image-root', required=True)
    parser.add_argument('--legacy-checkpoint', required=True, help='Read-only original 50-class checkpoint')
    parser.add_argument('--output-dir', required=True, help='Separate candidate output directory')
    parser.add_argument('--taxonomy', default=str(HERE / 'classes_120.csv'))
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--min-train', type=int, default=100)
    parser.add_argument('--min-val', type=int, default=10)
    parser.add_argument('--min-test', type=int, default=10)
    args = parser.parse_args()
    if min(args.min_train, args.min_val, args.min_test, args.epochs, args.batch_size) < 1 or args.workers < 0:
        parser.error('Minimum counts, epochs, and batch size must be positive; workers cannot be negative')
    report = audit(args.taxonomy, args.manifest, args.image_root,
                   {'train': args.min_train, 'val': args.min_val, 'test': args.min_test})
    if not report['ready_for_training']:
        raise SystemExit('Dataset not ready; no training performed: ' + '; '.join(report['errors']))
    taxonomy, rows = read_csv(args.taxonomy), read_csv(args.manifest)
    if [int(r['target_index']) for r in taxonomy] != list(range(120)):
        raise SystemExit('Taxonomy must be ordered by target_index 0..199')
    class_ids = [row['class_id'] for row in taxonomy]
    out = Path(args.output_dir).resolve()
    legacy = Path(args.legacy_checkpoint).resolve()
    if out == legacy.parent:
        raise SystemExit('Use a separate output directory to protect the existing checkpoint')
    out.mkdir(parents=True, exist_ok=True)
    # Heavy imports occur only after data readiness checks.
    import torch
    from torchvision import models
    torch.set_num_threads(8)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    original = torch.load(legacy, map_location='cpu', weights_only=False)
    legacy_names = list(original['class_names'])
    if legacy_names != class_ids[:50]:
        raise SystemExit('Legacy model class order differs from taxonomy; refusing to train')
    net = models.efficientnet_b0(weights=None)
    net.classifier[1] = torch.nn.Linear(net.classifier[1].in_features, len(legacy_names))
    net.load_state_dict(original['model'], strict=True)
    fingerprint = feature_fingerprint(rows, args.image_root, legacy, class_ids)
    feature_path, metadata_path = out / 'features.npy', out / 'features_metadata.json'
    cached = feature_path.exists() and metadata_path.exists() and json.loads(metadata_path.read_text()).get('fingerprint') == fingerprint
    if cached:
        features = np.load(feature_path, allow_pickle=False)
        if features.shape != (len(rows), net.classifier[1].in_features):
            raise SystemExit('Cached features have an unexpected shape; remove candidate cache to rebuild')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        features = extract_features(rows, args.image_root, net, device, args.batch_size, args.workers)
        temporary = out / 'features.tmp.npy'
        np.save(temporary, features, allow_pickle=False)
        os.replace(temporary, feature_path)
        metadata_path.write_text(json.dumps({'fingerprint': fingerprint, 'shape': list(features.shape)}, indent=2))
    index = {class_id: i for i, class_id in enumerate(class_ids)}
    labels = np.array([index[row['class_id']] for row in rows])
    splits = np.array([row['split'] for row in rows])
    tr, va, te = splits == 'train', splits == 'val', splits == 'test'
    head, history = fit_head(features[tr], labels[tr], features[va], labels[va], len(class_ids), args.epochs, args.seed)
    pred = head.predict(features[te])
    legacy_slice = labels[te] < 50
    metrics = {'model': 'efficientnet_b0_frozen_backbone_linear_logistic_head',
               'dataset_manifest_sha256': sha256_file(args.manifest), 'feature_fingerprint': fingerprint,
               'classes': len(class_ids), 'split': 'test', 'n': int(te.sum()),
               'accuracy': float(accuracy_score(labels[te], pred)),
               'macro_f1': float(f1_score(labels[te], pred, labels=np.arange(120), average='macro', zero_division=0)),
               'legacy_50_slice_n': int(legacy_slice.sum()),
               'legacy_50_slice_accuracy': float(accuracy_score(labels[te][legacy_slice], pred[legacy_slice])),
               'legacy_50_slice_macro_f1': float(f1_score(labels[te][legacy_slice], pred[legacy_slice], labels=np.arange(50), average='macro', zero_division=0)),
               'best_validation_epoch': max(history, key=lambda row: row['val_macro_f1'])['epoch'],
               'val_history': history, 'seed': args.seed,
               'caveat': 'Legacy slice is scored among all 120 candidate outputs; external Forms photos are not included.'}
    np.save(out / 'confusion_matrix.npy', confusion_matrix(labels[te], pred, labels=np.arange(120)), allow_pickle=False)
    (out / 'evaluation_candidate.json').write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding='utf-8')
    net.classifier[1] = torch.nn.Linear(net.classifier[1].in_features, len(class_ids))
    with torch.no_grad():
        net.classifier[1].weight.copy_(torch.from_numpy(head.coef_.astype(np.float32)))
        net.classifier[1].bias.copy_(torch.from_numpy(head.intercept_.astype(np.float32)))
    candidate = {'model': net.cpu().state_dict(), 'class_names': class_ids, 'training_method': metrics['model'],
                 'metadata': {'feature_fingerprint': fingerprint, 'requires_120_class_app_integration': True}}
    temp = out / 'food_model_candidate.tmp.pth'
    torch.save(candidate, temp)
    os.replace(temp, out / 'food_model_candidate.pth')
    print(json.dumps({k: metrics[k] for k in ('classes', 'n', 'accuracy', 'macro_f1', 'legacy_50_slice_accuracy')}, indent=2))
    print('Candidate saved separately. It has not been deployed to the live app.')


if __name__ == '__main__':
    main()
