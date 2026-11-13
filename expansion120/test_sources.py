import csv,json,sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from audit_dataset import audit, read_csv
class SourceTests(unittest.TestCase):
    def test_120_unique_labels_and_retained_legacy_order(self):
        rows=read_csv(ROOT/'classes_120.csv')
        old=read_csv(ROOT.parent/'expansion200/classes_200.csv')
        self.assertEqual(len(rows),120)
        self.assertEqual(len({x['menu_th'] for x in rows}),120)
        self.assertEqual([x['class_id'] for x in rows[:50]],[x['class_id'] for x in old[:50]])
    def test_additional_classes_exist_in_actual_archives(self):
        counts=json.loads((ROOT/'source_image_counts.json').read_text())
        for row in read_csv(ROOT/'classes_120.csv')[50:]:
            source='thfood100_full' if row['origin']=='THFOOD100_source_label' else 'thdess20'
            cid=row['source_class_id'].zfill(2) if source=='thfood100_full' else row['source_class_id']
            for split,minimum in [('train',100),('val',20),('test',20)]:
                self.assertGreaterEqual(counts[source]['split_class_counts'][split+'/'+cid],minimum)
    def test_empty_manifest_does_not_allow_training(self):
        r=audit(ROOT/'classes_120.csv',ROOT.parent/'expansion200/images_manifest_template.csv',ROOT)
        self.assertFalse(r['ready_for_training']);self.assertEqual(len(r['missing_classes']),120)
if __name__=='__main__':unittest.main(verbosity=2)
