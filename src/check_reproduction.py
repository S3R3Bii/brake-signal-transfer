"""Replay the frozen experiment from cached public data in a disposable folder."""
from pathlib import Path
import json
import math
import shutil
import subprocess
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]

def compare(expected,actual,path='root'):
    if isinstance(expected,dict):
        assert expected.keys()==actual.keys(),path
        for key in expected:
            if key not in {'frozen_utc','evaluated_utc'}:
                compare(expected[key],actual[key],f'{path}.{key}')
    elif isinstance(expected,list):
        assert len(expected)==len(actual),path
        for i,(left,right) in enumerate(zip(expected,actual)):
            compare(left,right,f'{path}[{i}]')
    elif isinstance(expected,float):
        assert math.isclose(expected,actual,rel_tol=1e-7,abs_tol=1e-9),(path,expected,actual)
    else:
        assert expected==actual,(path,expected,actual)

with tempfile.TemporaryDirectory(prefix='brake-reproduction-') as folder:
    replay=Path(folder)
    for directory in ['data','src']:
        shutil.copytree(ROOT/directory,replay/directory)
    for script,args in [('analyze.py',['validate']),('analyze.py',['test']),('verify.py',['test']),('figures.py',[])]:
        subprocess.run([sys.executable,str(replay/'src'/script),*args],cwd=replay,check=True,timeout=420)
    for name in ['selection.json','test_summary.json','data_quality.json']:
        compare(json.loads((ROOT/'results'/name).read_text()),json.loads((replay/'results'/name).read_text()),name)
    print('PASS: offline refit, frozen selection, independent metrics and figure generation match the released experiment.')
