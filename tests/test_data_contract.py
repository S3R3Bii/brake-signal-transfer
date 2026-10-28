"""Small synthetic fixtures test failure modes; they are not study observations."""
from pathlib import Path
import hashlib
import json
import sys
import tempfile
import unittest
from unittest.mock import patch
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import prepare

class DataContracts(unittest.TestCase):
    def fixture(self,root,rows):
        (root/'data').mkdir(exist_ok=True)
        path=root/'data/fixture.csv';pd.DataFrame(rows).to_csv(path,index=False)
        (root/'data/manifest.json').write_text(json.dumps([{'file':'data/fixture.csv',
            'rows':len(rows),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}]))

    def sample(self):
        return [dict(vehicle_id=1,frame_id=i,global_time=100000+i*100,v_vel=10+i*.1,
            v_acc=1,local_y=i,v_class=2,location='test-road') for i in range(30)]

    def test_gap_resets_the_history(self):
        rows=self.sample();rows=rows[:15]+rows[16:]
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);self.fixture(root,rows)
            with patch.object(prepare,'ROOT',root): f,c=prepare.build()
        self.assertEqual(c[0]['segments_before_history_filter'],2)
        self.assertEqual(f.frame_id.tolist(),[10,11,12,13,14,26,27,28,29])

    def test_future_speed_does_not_change_earlier_features(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);rows=self.sample();self.fixture(root,rows)
            with patch.object(prepare,'ROOT',root):before,_=prepare.build()
            rows[25]['v_vel']=1000;self.fixture(root,rows)
            with patch.object(prepare,'ROOT',root):after,_=prepare.build()
        pd.testing.assert_frame_equal(before[before.frame_id<25],after[after.frame_id<25])

    def test_constant_speed_has_zero_variance(self):
        rows=self.sample()
        for row in rows:row['v_vel']=12.5
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);self.fixture(root,rows)
            with patch.object(prepare,'ROOT',root):f,_=prepare.build()
        self.assertTrue((f.past_sd==0).all())

    def test_changed_source_hash_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);self.fixture(root,self.sample())
            with (root/'data/fixture.csv').open('a') as stream:stream.write('\n')
            with patch.object(prepare,'ROOT',root),self.assertRaises(AssertionError):prepare.build()

    def test_conflicting_keys_are_rejected_not_arbitrarily_chosen(self):
        rows=self.sample();duplicate=dict(rows[15]);duplicate['v_vel']=999;rows.append(duplicate)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);self.fixture(root,rows)
            with patch.object(prepare,'ROOT',root):f,c=prepare.build()
        self.assertEqual(c[0]['conflicting_rows_removed'],2)
        self.assertNotIn(15,f.frame_id.tolist())

if __name__=='__main__':unittest.main()
