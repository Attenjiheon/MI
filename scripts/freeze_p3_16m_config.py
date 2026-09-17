"""Freeze executable config only after the complete immutable corpus audit."""
import copy
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from corpus.generate import file_hash,write_json


def run():
    data=ROOT/'data/language_v1_2'; out=ROOT/'experiment_v1_2/configs'
    if (out/'config_set_manifest.json').exists(): raise FileExistsError('Config already frozen')
    assert json.loads((data/'postwrite_audit.json').read_text())['status']=='passed'
    corpus=json.loads((data/'manifest.json').read_text())
    transformer=json.loads((ROOT/'experiment_v1_1/configs/transformer.json').read_text())
    transformer['spec_version']='experiment-spec-v1.2'; transformer['config_version']='p3-config-v1.2'
    for key in ('pilot_prediction_tokens','maximum_extended_prediction_tokens'): del transformer['training'][key]
    transformer['training'].update(nominal_prediction_tokens=16000000,budget_policy='fixed_no_early_gate_stop')
    for key in tuple(transformer['gate']):
        if key.startswith('extension_'): del transformer['gate'][key]
    write_json(out/'transformer.json',transformer)
    run=dict(schema='p3-run-v1.2',seed=0,budget=16000000,validation_interval=100000,effective_batch=64,data_root='data/language_v1_2',
             frozen_training=corpus['frozen_training'],data_hashes={name:file_hash(data/name) for name in ('manifest.json','cpu_validation.json','postwrite_audit.json','corpus_statistics.json')},
             design_sha256=file_hash(ROOT/'experiment_v1_2/design_config.json'),selection='minimum_general_validation_answer_ce_earlier_update_tie',
             milestones=[1000000,3000000,8000000,16000000],persistent_policy='immutable_files_incremental_copy_atomic_completion_index')
    write_json(out/'run.json',run)
    files={str(p.relative_to(ROOT)):file_hash(p) for p in sorted(out.glob('*.json'))}
    files['experiment_v1_2/design_config.json']=run['design_sha256']
    write_json(out/'config_set_manifest.json',dict(files=files,combined_sha256=hashlib.sha256(json.dumps(files,sort_keys=True,separators=(',',':')).encode()).hexdigest()))
if __name__=='__main__': run()
