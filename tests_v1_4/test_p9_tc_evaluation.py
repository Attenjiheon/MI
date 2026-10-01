import json
from pathlib import Path
import numpy as np
import pytest
import torch
from interp_v1_4.dictionary import Dictionary,normalize,denormalize
from interp_v1_4 import p9_tc_evaluation as ev
from interp_v1_4 import p9_tc_causal as pc
from interp_v1_4.p5 import read,write


@pytest.mark.parametrize('layer',[0,3,7,11])
@pytest.mark.parametrize('k',[4,16])
def test_causal_m_output_u_encoding_and_skip(layer,k):
    from interp_v1_4.model import Transformer
    from interp_v1_4.patching import capture
    torch.set_num_threads(2);torch.manual_seed(304)
    model=Transformer().eval().requires_grad_(False)
    d=Dictionary('transcoder',k,44,width=256).eval().requires_grad_(False)
    p=ev.records(Path('data/language_v1_4/rebuild_01/causal_pairs/val_changed_memory.jsonl.gz'))[0]
    prefixes=[p['original_prefix_ids'],p['counterfactual_prefix_ids']]
    original,m,u=pc.forward(model,prefixes,layer)
    stats={'u':dict(mu=torch.ones(256)*.2,scale=torch.tensor(.3)), 'm':dict(mu=torch.ones(256)*-.4,scale=torch.tensor(2.1))}
    z=d.encode(normalize(u,stats['u']));J=torch.argsort((z[1]-z[0]).abs(),descending=True)[:4].tolist()
    selection=dict(fits={f'{rep}_current_iid':dict(status='passed',selected={size:dict(columns=J[:1] if size=='single' else J) for size in ('single','up_to_four')}) for rep in ('transcoder','random')})
    selection['fits']['coordinate_current_iid']=dict(status='passed',selected={size:dict(columns=[0] if size=='single' else [0,1,2,3]) for size in ('single','up_to_four')})
    match=dict(rules={size:dict(rule=dict(h_edges=[.1,.5,1.,2.],z_edges=[.1,.5,1.,2.])) for size in ('single','up_to_four')})
    R=np.random.default_rng(4).normal(size=(256,512));R/=np.linalg.norm(R,axis=0)
    result=pc.causal_unit(model,d,stats,R,dict(layer=layer,run_key='test'),p,selection,match,dict(random_candidates=3,matched_limit=2))
    for direction in (0,1):
        donor=1-direction
        values={'selected':m[direction:direction+1]+2.1*((z[donor:donor+1,J]-z[direction:direction+1,J])@d.decoder[:,J].T),
                'approximation':denormalize(d(normalize(u[direction:direction+1],stats['u']))[0],stats['m']),
                'mean':stats['m']['mu'][None],'full_donor':m[donor:donor+1], 'identity':m[direction:direction+1]}
        for control,value in values.items():
            logits,_,_=pc.forward(model,[prefixes[direction]],layer,value)
            row=next(r for r in result['rows'] if r['size']=='up_to_four' and r['direction']==direction and r['control']==control)
            np.testing.assert_allclose(row['patched_logits'],logits[0].numpy(),atol=1e-5,rtol=1e-4)
        # Hook only m: resulting h equals original residual midpoint + proposed m.
        from interp_v1_4.model import batch
        ids,mask,_=batch([prefixes[direction]],'cpu',shift=False);pos=len(prefixes[direction])-1
        with torch.no_grad(),capture(model) as cached:model(ids,mask)
        mid=cached[f'blocks.{layer}.resid_mid'][:,pos]
        from interp_v1_4.patching import patch
        with torch.no_grad(),capture(model) as after,patch(model,f'blocks.{layer}.mlp_out',[pos],values['selected']):model(ids,mask)
        torch.testing.assert_close(after[f'blocks.{layer}.resid_post'][:,pos],mid+values['selected'])


def test_encode_uses_u_statistics():
    d=Dictionary('transcoder',4,31,width=256).eval()
    stats={'u':dict(mu=torch.ones(256),scale=torch.tensor(.2)),'m':dict(mu=torch.zeros(256),scale=torch.tensor(7.))}
    raw=np.random.default_rng(1).normal(size=(7,256)).astype('float32')
    y,z=ev.encode(d,stats,raw,'cpu')
    with torch.no_grad():expected,latent=d(normalize(torch.tensor(raw),stats['u']))
    np.testing.assert_array_equal(y,expected.numpy());np.testing.assert_array_equal(z,latent.numpy())


def test_selection_reads_only_train_val_and_m_baselines(tmp_path,monkeypatch):
    contract=tmp_path/ev.CONTRACT;write(contract,{})
    write(tmp_path/ev.RECEIPT,dict(config_sha256='cache'))
    run=dict(name='debug',lm_seed=0,layer=0);entry=dict(run=run,sha256='tc')
    reused={}
    for rep in ('fullu','fullm'):
        name=f'{rep}_current_iid';file=f'{rep}.json';write(tmp_path/file,dict(result=dict(status='NA',reason=rep)))
        reused['seed0_l0_'+name]=file
    config=dict(reused_probes=reused)
    monkeypatch.setattr(ev,'verify',lambda root:config);monkeypatch.setattr(ev,'LABELS',{'current':2})
    # A/B/C -> D transfer is also checked using same reuse fixture.
    for rep in ('fullu','fullm'):reused[f'seed0_l0_{rep}_current_transfer']=f'{rep}.json'
    d=Dictionary('transcoder',4,1,width=256)
    stats={h:dict(mu=torch.zeros(256),scale=torch.tensor(2. if h=='m' else .25)) for h in 'um'}
    monkeypatch.setattr(ev,'model_entry',lambda *a:(d,stats))
    R=np.eye(256,512);subsets={r:list(range(128)) for r in ('coordinate','random','transcoder')}
    monkeypatch.setattr(ev,'directions',lambda *a:(R,dict(projection_seed=1,subsets=subsets)))
    calls=[]
    def cache(source,seed,kind,split,layer,hook,identity):
        assert split in ('train','val');calls.append((split,hook))
        return np.ones((4,256),np.float32)*(3 if hook=='u' else 9)
    monkeypatch.setattr(ev,'cache_array',cache)
    for split in ('train','val'):
        write(tmp_path/'source/labels'/f'{split}.json',[dict(sequence_id=str(i),current=i%2,query=i%4,previous=0) for i in range(4)])
    monkeypatch.setattr(ev,'domains',lambda rows,*a:np.ones(len(rows),bool))
    seen=[]
    def fit(x,*a):seen.append(x.copy());return dict(status='NA',reason='fixture')
    monkeypatch.setattr(ev.probe,'fit',fit)
    result=ev.selection(tmp_path,tmp_path/'source',tmp_path/'out',entry)
    assert set(calls)=={('train','m'),('val','m'),('train','u'),('val','u')}
    assert np.all(seen[0]==9) # coordinate source is m, not u
    assert result['fits']['fullu_current_iid']['reason']=='fullu'
    assert result['fits']['fullm_current_iid']['reason']=='fullm'
    assert result['test_used_for_selection'] is False
