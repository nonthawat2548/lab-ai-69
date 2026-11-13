import csv
import tempfile
import unittest
from pathlib import Path
from audit_dataset import audit, read_csv
ROOT = Path(__file__).resolve().parent

class DatasetPreparationTests(unittest.TestCase):
    def test_target_really_has_200_unique_labels_and_retains_legacy_50(self):
        rows = read_csv(ROOT / 'classes_200.csv')
        self.assertEqual(len(rows), 200)
        self.assertEqual(len({r['class_id'] for r in rows}), 200)
        self.assertEqual(len({r['menu_th'] for r in rows}), 200)
        self.assertEqual([int(r['target_index']) for r in rows], list(range(200)))
        self.assertTrue(all(r['origin'] == 'legacy_THFOOD50' and r['live_model_support'] == 'true' for r in rows[:50]))
        self.assertTrue(all(r['live_model_support'] == 'false' and r['image_availability'] == 'not_collected' for r in rows[50:]))
        self.assertTrue(all(r['recipe_reference_url'].startswith('https://nlovecooking.com/') for r in rows[50:]))

    def test_empty_dataset_is_blocked_not_reported_as_200_trained_classes(self):
        report = audit(ROOT / 'classes_200.csv', ROOT / 'images_manifest_template.csv', ROOT)
        self.assertFalse(report['ready_for_training'])
        self.assertEqual(report['audited_class_count'], 0)
        self.assertEqual(len(report['missing_classes']), 200)

    def test_reused_photo_across_train_test_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'a.jpg').write_bytes(b'same image content')
            (root / 'b.jpg').write_bytes(b'same image content')
            rows = [{'image_path': name, 'class_id': 'PadThai', 'split': split,
                'source_url': 'https://example.org/owned', 'collection_group': 'session-a',
                'human_reviewed': 'true', 'rights_status': 'owned_with_consent'}
                for name, split in [('a.jpg', 'train'), ('b.jpg', 'test')]]
            manifest = root / 'images.csv'
            with manifest.open('w', newline='') as out:
                writer = csv.DictWriter(out, fieldnames=list(rows[0]))
                writer.writeheader(); writer.writerows(rows)
            report = audit(ROOT / 'classes_200.csv', manifest, root)
            self.assertFalse(report['ready_for_training'])
            self.assertTrue(any('Leakage' in e for e in report['errors']))
            self.assertTrue(any('duplicate image' in e for e in report['errors']))

    def test_missing_image_and_unverified_rights_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest = root / 'bad.csv'
            manifest.write_text('image_path,class_id,split\nmissing.jpg,PadThai,train\n')
            report = audit(ROOT / 'classes_200.csv', manifest, root)
            self.assertTrue(any('usage rights' in e for e in report['errors']))
            self.assertTrue(any('not reviewed' in e for e in report['errors']))
            self.assertTrue(any('missing' in e for e in report['errors']))

if __name__ == '__main__':
    unittest.main(verbosity=2)
