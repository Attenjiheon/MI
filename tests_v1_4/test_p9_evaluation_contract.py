from pathlib import Path
import json
import numpy as np
import pytest
import torch
from interp_v1_4 import p9_sae_evaluation as sae,p9_tc_evaluation as tc
from interp_v1_4.p5 import read,derived
from interp_v1_4.runtime import sha
from interp_v1_4.p9_comparison import compare_rows
ROOT=Path(__file__).resolve().parents[1]

@pytest.mark.parametrize('module,tool,kind',[(sae,'sae','saes'),(tc,'transcoder','tcs')])
def test_frozen_selection_and_checkpoint_stats(module,tool,kind):
    c=module.verify(ROOT);rules=read(ROOT/'experiment_v1_4/p9_r1/evaluation_rules.json')
    assert len(c[kind])==8 and len(c['models'])==1 and c['models'][0]['lm_seed']==0
    assert c['baselines']==rules['baselines'][tool]
    for key,value in rules['shared'].items():assert c[key]==value
    assert c['frozen_rules_sha256']==sha(ROOT/'experiment_v1_4/p9_r1/evaluation_rules.json')
    rng=read(ROOT/c['patch_rng'])
    for entry in c[kind]:
        r=entry['run'];assert r['tool']==tool and r['sparse_seed']==1 and r['lm_seed']==0
        d,stats=module.model_entry(ROOT,entry)
        state=torch.load(ROOT/entry['checkpoint'],map_location='cpu',weights_only=False)
        if tool=='sae':
            assert torch.equal(stats['mu'],state['stats']['h']['mu']);assert stats['scale']==state['stats']['h']['scale']
        else:
            for hook in ('u','m'):assert torch.equal(stats[hook]['mu'],state['stats'][hook]['mu'])
        x=np.random.default_rng(8).normal(size=(5,256)).astype('float32')
        _,z=module.encode(d,stats,x,'cpu');assert z.shape==(5,512) and (z>0).sum(1).max()<=r['k']
        for key in list(rng[r['name']])[:3]:
            assert rng[r['name']][key]['latent']==derived('random_patch',r['run_key']+'|'+key)
        assert c['bootstrap_keys'][r['name']]==rules['bootstrap_keys'][r['name']]
        assert '|layer=' not in c['bootstrap_keys'][r['name']]


def test_seed_comparison_preserves_metric_and_missing_coverage():
    old=[dict(lm_seed=0,sparse_seed=0,layer=0,k=4,metric='delta_margin',mean=2),dict(lm_seed=1,sparse_seed=0,layer=0,k=4,metric='delta_margin',mean=99)]
    new=[dict(lm_seed=0,sparse_seed=1,layer=0,k=4,metric='delta_margin',mean=3),dict(lm_seed=0,sparse_seed=1,layer=0,k=4,metric='matched',mean=1)]
    rows=compare_rows(old,new,['layer','k','metric'],['mean'])
    assert rows[0]['metric']=='delta_margin' and rows[0]['statistic']=='mean' and rows[0]['difference_seed1_minus_seed0']==1
    assert rows[1]['status']=='NA' and rows[1]['difference_seed1_minus_seed0'] is None


def test_csv_resume(tmp_path):
    from interp_v1_4.p9_sae_aggregate import csvfile
    p=tmp_path/'table.csv';csvfile(p,[dict(value=1)]);before=p.read_bytes();csvfile(p,[dict(value=1)])
    assert p.read_bytes()==before
    with pytest.raises(ValueError):csvfile(p,[dict(value=2)])


def test_sae_selection_train_val_only(tmp_path,monkeypatch):
    from interp_v1_4.dictionary import Dictionary
    from interp_v1_4.p5 import write
    ev=sae;write(tmp_path/ev.CONTRACT,{});write(tmp_path/ev.RECEIPT,dict(config_sha256='cache'))
    entry=dict(run=dict(name='debug',lm_seed=0,layer=0),sha256='sae')
    reuse={f'seed0_l0_full_current_{domain}':'full.json' for domain in ('iid','transfer')}
    write(tmp_path/'full.json',dict(result=dict(status='NA',reason='fixture')))
    monkeypatch.setattr(ev,'verify',lambda root:dict(reused_probes=reuse));monkeypatch.setattr(ev,'LABELS',{'current':2})
    monkeypatch.setattr(ev,'model_entry',lambda *a:(Dictionary('sae',4,1,width=256),dict(mu=torch.zeros(256),scale=torch.tensor(2.))))
    monkeypatch.setattr(ev,'directions',lambda *a:(np.eye(256,512),dict(projection_seed=1,subsets={r:list(range(128)) for r in ('coordinate','random','sae')})))
    calls=[]
    def cache(source,seed,kind,split,layer,hook,identity):
        assert split in ('train','val') and hook=='h';calls.append(split);return np.ones((4,256),np.float32)
    monkeypatch.setattr(ev,'cache_array',cache)
    for split in ('train','val'):write(tmp_path/'source/labels'/f'{split}.json',[dict(sequence_id=str(i),current=i%2,query=i%4,previous=0) for i in range(4)])
    monkeypatch.setattr(ev,'domains',lambda rows,*a:np.ones(len(rows),bool));monkeypatch.setattr(ev.probe,'fit',lambda *a:dict(status='NA',reason='fixture'))
    result=ev.selection(tmp_path,tmp_path/'source',tmp_path/'out',entry)
    assert set(calls)=={'train','val'} and result['test_used_for_selection'] is False
