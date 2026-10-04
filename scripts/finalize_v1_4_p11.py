"""Seal verified P11 artifacts; Git delivery is recorded separately after push."""
from pathlib import Path
import argparse,hashlib,json,xml.etree.ElementTree as ET


def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()


def finalize(root):
 p=Path(root);dest=p/'completion.json'
 if dest.exists():raise ValueError('Final completion is immutable')
 v=json.loads((p/'verification.json').read_text());r=json.loads((p/'reproduction.json').read_text())
 suites=ET.parse(p/'tests.xml').getroot().findall('.//testsuite')
 tests=sum(int(s.attrib['tests']) for s in suites)
 assert v['status']=='passed' and r['status']=='passed' and tests==39
 assert all(int(s.attrib.get('failures',0))+int(s.attrib.get('errors',0))==0 for s in suites)
 assert v['read_runs']==64 and v['update_runs']==12 and r['compared_tables']==12
 aggregate=json.loads((p/'aggregation.json').read_text());aggregate['status']='passed_final_validation';aggregate['verification']='verification.json'
 (p/'aggregation.json').write_text(json.dumps(aggregate,indent=2)+'\n')
 sources=[Path('interp_v1_4/p11.py'),Path('experiment_v1_4/p11_r1/contract.json'),Path('tests_v1_4/test_p11.py')]+sorted(Path('scripts').glob('*v1_4_p11*.py'))
 artifacts={str(f.relative_to(p)):sha(f) for f in sorted(p.rglob('*')) if f.is_file() and f.name not in ('completion.json','git_delivery.json')}
 out=dict(schema='p11-completion-v1.4-r1',status='passed',date='2026-10-05',p11_complete=True,experiment_complete=True,
  required_read_dictionaries=64,optional_update_dictionaries=12,passing_lm_seeds=[0,1,2],failed_required_runs=0,
  training_tokens=192017253,read_updates=320000,read_position_draws=163840000,update_updates=60000,update_position_draws=30720000,
  tests_passed=tests,verification_sha256=sha(p/'verification.json'),reproduction_sha256=sha(p/'reproduction.json'),
  code_and_contract_sha256={str(f):sha(f) for f in sources},artifacts_sha256=artifacts,
  skipped_optional=['additional dictionary layers','length GPU evaluation','m-to-m SAE','Update causal interventions'],
  interpretation_limits=['Linear/post-hoc supervised accessibility is distinct from causal use.','SAE h and TC m have different targets and intervention hooks.',
   'Three LM seeds; layers and sparse repetitions are not independent models.','Layer expansion followed P5 test observation; Update is exploratory.',
   'Fidelity and sparse initialization differences remain point estimates with descriptive ranges, not invented CIs.'],
  model_inference_repeated=False,checkpoint_or_feature_reselection=False,git_delivery_record='git_delivery.json')
 dest.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n');print('P11 scientific completion sealed:',dest)


if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);finalize(parser.parse_args().output)
