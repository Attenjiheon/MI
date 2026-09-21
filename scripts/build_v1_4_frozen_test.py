"""Prepare and package the predeclared, all-seed frozen-test delivery."""
import argparse
import hashlib
import json
import shutil
import sys
import zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from interp_v1_4.frozen_test import FREEZE, FREEZE_SHA, CONTRACT, SUITES, expected_targets, check_inputs
from interp_v1_4.runtime import sha

FOLDER=ROOT/'experiment_v1_4/frozen_test_r1'
PREP=ROOT/'experiment_v1_4/results/frozen_test_preparation_r1'


def write(path,value):
    text=json.dumps(value,ensure_ascii=False,indent=2)+'\n'
    if path.exists():
        assert path.read_text()==text, 'Refusing to replace '+str(path)
    else:
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(text)


def prepare():
    assert sha(ROOT/FREEZE)==FREEZE_SHA
    freeze=json.loads((ROOT/FREEZE).read_text())
    paths={ROOT/FREEZE,ROOT/'experiment_v1_2/debug/sequences.json',FOLDER/'debug_pairs.jsonl.gz'}
    paths.update(ROOT/p for p in freeze['seeds'][1]['hashes']['code'])
    paths.update(ROOT/p for p in ('interp_v1_4/frozen_test.py','tests_v1_4/test_frozen_test.py',
                                  'scripts/build_v1_4_frozen_test.py','scripts/frozen_test_notebook.py'))
    paths.update((ROOT/'experiment_v1_4/configs').glob('*.json'))
    paths.update(ROOT/'experiment_v1_4'/p for p in ('design_config.json','corpus_rebuild.json','DESIGN.md'))
    sources=[]
    stamps=['20260921T064114169595','20260921T121619607352','20260921T134421434839']
    for item in freeze['seeds']:
        n=item['lm_seed'];audit=ROOT/item['audit_completion'];paths.add(audit)
        assert sha(audit)==item['audit_completion_sha256']
        verification=audit.parent/'archive_verification.json';paths.add(verification)
        record=json.loads(verification.read_text())
        archive=ROOT/f'experiment_v1_4/evidence/v1_4_seed{n}_evidence_{stamps[n]}.zip'
        digest=record.get('sha256',record.get('archive_sha256'))
        assert sha(archive)==digest
        checkpoint=FOLDER/f'checkpoints/seed{n}.pt';checkpoint.parent.mkdir(parents=True,exist_ok=True)
        member=f"deepwide12_read4_seed{n}/{item['selected_checkpoint']}"
        if not checkpoint.exists():
            with zipfile.ZipFile(archive) as z, z.open(member) as src, checkpoint.open('xb') as dst:
                shutil.copyfileobj(src,dst,1024*1024)
        assert sha(checkpoint)==item['selected_checkpoint_sha256']
        paths.add(checkpoint)
        sources.append(dict(seed=n,archive=str(archive.relative_to(ROOT)),archive_sha256=digest,member=member,
                            checkpoint_sha256=sha(checkpoint)))
    lock=ROOT/'experiment_v1_4/results/p3_audit_20260921_01/gpu_smoke/requirements.lock.txt';paths.add(lock)
    constraints='\n'.join(line for line in lock.read_text().splitlines() if '==' in line and not line.startswith('#'))+'\n'
    primary='\n'.join(line for line in constraints.splitlines() if line.split('==')[0] in {'torch','numpy','scipy','pandas','matplotlib','pytest'})+'\n'
    for name,content in [('requirements-primary.lock.txt',primary),('requirements-seed0.constraints.txt',constraints)]:
        path=FOLDER/name
        if path.exists():assert path.read_text()==content
        else:path.write_text(content)
        paths.add(path)
    data_root='data/language_v1_4/rebuild_01'
    manifest=ROOT/data_root/'manifest.json';paths.add(manifest)
    assert sha(manifest)==freeze['seeds'][0]['hashes']['corpus_manifest']
    cm=json.loads(manifest.read_text());suites=[]
    for spec in SUITES:
        path=ROOT/data_root/spec['path'];paths.add(path)
        assert sha(path)==cm['files'][spec['path']]['sha256']
        suites.append(dict(**spec,**expected_targets(path,spec),input_sha256=sha(path)))
    contract=dict(schema='v1.4-frozen-test-r1',validation_freeze_sha256=FREEZE_SHA,data_root=data_root,microbatch=16,
                  suites=suites,files={str(p.relative_to(ROOT)):sha(p) for p in sorted(paths)},sources=sources,
                  selection_or_gate=False,test_scores_observed_before_freeze=False,
                  excluded=['test/length (optional P10)'],
                  intervals='First/repeat: frozen paired within-cell bootstrap; other suites: 1000 sequence-cluster bootstrap draws, 95% percentile',
                  resume='Reuse completed results; saved measurements resume reporting only; a started unit without measurements blocks inference replay.')
    write(ROOT/CONTRACT,contract)
    check_inputs(ROOT)
    print(json.dumps(dict(status='prepared',contract_sha256=sha(ROOT/CONTRACT),suites=suites),indent=2))


def pack():
    contract,_=check_inputs(ROOT)
    smoke=PREP/'cpu/preflight.json';sm=json.loads(smoke.read_text())
    assert sm['status']=='passed' and sm['scope']=='cpu' and sm['contract_sha256']==sha(ROOT/CONTRACT)
    import xml.etree.ElementTree as ET
    cases=ET.parse(PREP/'tests.xml').getroot().findall('.//testcase')
    assert len(cases)>=6 and not any(c.find('failure') is not None or c.find('error') is not None or c.find('skipped') is not None for c in cases)
    paths=[ROOT/p for p in contract['files']]+[ROOT/CONTRACT,smoke,PREP/'tests.xml',PREP/'tests.log']
    inventory={str(p.relative_to(ROOT)):sha(p) for p in sorted(paths)}
    bundle=ROOT/'experiment_v1_4/bundles/v1_4_frozen_test_bundle_r1.zip'
    with zipfile.ZipFile(bundle,'x',zipfile.ZIP_DEFLATED,compresslevel=1) as z:
        for p in paths:z.write(p,str(p.relative_to(ROOT)))
        z.writestr('bundle_manifest.json',json.dumps(dict(schema='v1.4-frozen-test-input',files=inventory),indent=2))
    with zipfile.ZipFile(bundle) as z:
        assert len(z.namelist())==len(set(z.namelist()))==len(inventory)+1
        for name,digest in inventory.items():
            h=hashlib.sha256()
            with z.open(name) as f:
                for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
            assert h.hexdigest()==digest
    digest=sha(bundle);bundle.with_suffix('.zip.sha256').write_text(digest+'  '+bundle.name+'\n')
    from scripts.frozen_test_notebook import notebook
    nb=notebook(bundle.name,digest)
    for c in nb['cells']:
        if c['cell_type']=='code':compile(c['source'],'frozen test notebook','exec')
    notebook_path=ROOT/'experiment_v1_4/notebooks/P4_v1_4_frozen_test_r1.ipynb';write(notebook_path,nb)
    write(PREP/'verification.json',dict(status='passed_local_preparation',bundle=str(bundle.relative_to(ROOT)),
          bundle_sha256=digest,bytes=bundle.stat().st_size,members=len(inventory),all_member_hashes_verified=True,
          contract_sha256=sha(ROOT/CONTRACT),validation_freeze_sha256=FREEZE_SHA,tests=len(cases),
          notebook=str(notebook_path.relative_to(ROOT)),notebook_sha256=sha(notebook_path),
          notebook_code_cells_compiled=sum(c['cell_type']=='code' for c in nb['cells']),
          cpu_preflight_sha256=sha(smoke),test_scores_observed=False,gpu_test_status='not_run',p4_complete=False))
    print((PREP/'verification.json').read_text())


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','pack']);args=p.parse_args()
    prepare() if args.action=='prepare' else pack()
