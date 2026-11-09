import tempfile
import unittest
from pathlib import Path
from form_image_eval import file_ids, parse_responses, merge_labels, score_predictions, save_labels, read_csv, predict_images

HEADERS = ['ท่านยินยอมให้ใช้ภาพอาหารหรือไม่', 'กรุณาอัปโหลดภาพอาหารไทย', 'ชื่อเมนูอาหารในภาพคืออะไร',
           'อาหารประเภทใด', 'ซื้อหรือปรุงเอง', 'ปริมาณ', 'ท่านเป็นผู้ถ่ายภาพนี้เองและมีสิทธิหรือไม่']
IDS = ['a' * 28, 'b' * 28, 'c' * 28]
LINKS = [f'https://drive.google.com/open?id={fid}' for fid in IDS]

class FormImageTests(unittest.TestCase):
    def test_multiple_photos_never_inherit_a_fake_gold_label(self):
        manifest, labels, _ = parse_responses([HEADERS, ['ยินยอม', ', '.join(LINKS), 'หลายอย่าง', '', '', '', 'ใช่ เป็นเจ้าของ']], {'ผัดไทย': 'PadThai'})
        self.assertEqual(len(manifest), 3)
        self.assertTrue(all('gold_label' not in x and 'reported_menu' not in x for x in manifest))
        self.assertTrue(all(x['gold_label'] == '' and x['human_verified'] == 'false' for x in labels))

    def test_decline_or_no_rights_not_used(self):
        source = [HEADERS, ['ไม่ยินยอม', LINKS[0], 'ผัดไทย', '', '', '', 'ใช่'],
                  ['ยินยอม', LINKS[1], 'ผัดไทย', '', '', '', 'ไม่ใช่']]
        manifest, labels, skip = parse_responses(source, {'ผัดไทย': 'PadThai'})
        self.assertFalse(manifest)
        self.assertFalse(labels)
        self.assertEqual(len(skip), 2)

    def test_outside_model_and_known_labels_need_human_confirmation(self):
        source = [HEADERS, ['ยินยอม', LINKS[0], 'คิมบับ', '', '', '', 'ใช่'],
                  ['ยินยอม', LINKS[1], 'ผัดไทย', '', '', '', 'ใช่']]
        manifest, labels, _ = parse_responses(source, {'ผัดไทย': 'PadThai'})
        self.assertEqual(labels[0]['scope'], 'pending')
        self.assertEqual(labels[1]['gold_label'], 'PadThai')
        self.assertEqual(labels[1]['human_verified'], 'false')

    def test_predictor_rejects_ground_truth(self):
        # Reject before importing torch/loading model; never forwards reference text.
        with self.assertRaisesRegex(ValueError, 'never gold labels'):
            predict_images(None, None, [{'file_id': IDS[0], 'path': '/tmp/x', 'gold_label': 'PadThai'}], ['PadThai'], 'sha', 'cache.json')

    def test_scoring_after_predictions_counts_low_confidence_errors(self):
        labels = [{'file_id': IDS[0], 'gold_label': 'PadThai', 'scope': 'in_scope', 'human_verified': 'true'},
                  {'file_id': IDS[1], 'gold_label': 'KaoManGai', 'scope': 'in_scope', 'human_verified': 'true'}]
        predictions = [{'file_id': IDS[0], 'image_sha256': 'sha-a', 'predicted_class': 'PadThai', 'confidence': .8, 'status': 'ok'},
                       {'file_id': IDS[1], 'image_sha256': 'sha-b', 'predicted_class': 'PadThai', 'confidence': .2, 'status': 'ok'}]
        metrics, rows, matrix = score_predictions(predictions, labels, ['PadThai', 'KaoManGai'])
        self.assertEqual(metrics['accuracy'], .5)
        self.assertAlmostEqual(metrics['macro_f1_present_classes'], 1 / 3)
        self.assertEqual(matrix, [[1, 0], [1, 0]])
        self.assertFalse(rows[1]['correct'])
        self.assertEqual(predictions[1]['predicted_class'], 'PadThai')

    def test_missing_labels_does_not_create_perfect_score(self):
        metrics, _, matrix = score_predictions([], [{'file_id': IDS[0], 'scope': 'pending', 'human_verified': 'false'}], ['PadThai'])
        self.assertIsNone(metrics['accuracy'])
        self.assertIsNone(matrix)
        self.assertEqual(metrics['counts']['pending_human_review'], 1)

    def test_ood_not_in_primary_accuracy_and_duplicates_not_double_counted(self):
        labels = [{'file_id': fid, 'scope': scope, 'gold_label': 'PadThai', 'human_verified': 'true'}
                  for fid, scope in zip(IDS, ['in_scope', 'in_scope', 'out_of_scope'])]
        preds = [{'file_id': fid, 'status': 'ok', 'image_sha256': 'same-file-bytes', 'predicted_class': 'PadThai'} for fid in IDS]
        metrics, _, _ = score_predictions(preds, labels, ['PadThai'])
        self.assertEqual(metrics['n_evaluated'], 1)
        self.assertEqual(metrics['counts']['duplicate_image_bytes'], 1)
        self.assertEqual(metrics['counts']['out_of_scope'], 1)

    def test_annotations_persist_after_refresh(self):
        new = [{'file_id': IDS[0], 'gold_label': '', 'scope': 'pending', 'human_verified': 'false'}]
        old = [{'file_id': IDS[0], 'gold_label': 'PadThai', 'scope': 'in_scope', 'human_verified': 'true', 'notes': 'reviewed'}]
        result = merge_labels(new, old)
        self.assertEqual(result[0]['gold_label'], 'PadThai')
        self.assertEqual(result[0]['human_verified'], 'true')
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'gold.csv'
            save_labels(path, result)
            self.assertEqual(read_csv(path)[0]['notes'], 'reviewed')

    def test_url_parser_rejects_non_drive_and_deduplicates(self):
        self.assertEqual(file_ids(LINKS[0] + ', ' + LINKS[0]), [IDS[0]])
        self.assertFalse(file_ids('https://evil.invalid/download?id=' + IDS[1]))

if __name__ == '__main__':
    unittest.main(verbosity=2)
