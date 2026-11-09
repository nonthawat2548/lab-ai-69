import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_candidate import fit_head, feature_fingerprint

class CandidateTrainingTests(unittest.TestCase):
    def test_separable_features_fit_and_predict(self):
        rng = np.random.default_rng(4)
        centers = np.eye(4, dtype=np.float32) * 8
        x = np.repeat(centers, 8, axis=0) + rng.normal(0, .05, (32, 4))
        y = np.repeat(np.arange(4), 8)
        model, history = fit_head(x, y, x.copy(), y.copy(), 4, epochs=5)
        self.assertEqual(len(history), 5)
        self.assertGreater(np.mean(model.predict(x) == y), .99)
        self.assertEqual(model.coef_.shape, (4, 4))

    def test_missing_training_class_is_blocked(self):
        with self.assertRaises(ValueError):
            fit_head(np.ones((2, 2)), np.array([0, 0]), np.ones((2, 2)), np.array([0, 1]), 2)

    def test_cache_key_changes_when_image_changes(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'photo').write_bytes(b'first')
            (root / 'model').write_bytes(b'weights')
            rows = [{'image_path': 'photo', 'class_id': 'food', 'split': 'train'}]
            a = feature_fingerprint(rows, root, root / 'model', ['food'])
            (root / 'photo').write_bytes(b'second')
            b = feature_fingerprint(rows, root, root / 'model', ['food'])
            self.assertNotEqual(a, b)

if __name__ == '__main__':
    unittest.main(verbosity=2)
