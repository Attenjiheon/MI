"""Immutable v1.2 extension: verify inherited bytes, append fresh indexed shards, audit all."""
import copy
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import corpus.audit as auditor
from corpus.generate import Builder, file_hash, write_json, self_test
from corpus.language import digest
from corpus.replay import replay


def run():
    start=time.monotonic()
    design=json.loads((ROOT/'experiment_v1_2/design_config.json').read_text())
    base=ROOT/design['data']['base_root']; out=ROOT/design['data']['planned_root']
    if out.exists(): raise FileExistsError('Preserve existing corpus; inspect or resume explicitly, never overwrite')
    assert file_hash(base/'manifest.json')==design['data']['base_manifest_sha256']
    bm=json.loads((base/'manifest.json').read_text())
    reports=ROOT/'experiment_v1_2/results'
    # Redirect only the report destination, leaving all original corpus bytes intact.
    original_writer=auditor.write_json
    auditor.write_json=lambda path,result:write_json(reports/'p1_base_reaudit.json',result)
    auditor.audit(base)
    auditor.write_json=original_writer
    hashes=set((base/'all_sequence_hashes.txt').read_text().splitlines())
    prefixes=set((base/'causal_prefix_hashes.txt').read_text().splitlines())
    # Recompute both reserved full origin forms (the legacy audit checks only origin).
    for path in sorted((base/'causal_pairs').glob('*.origins.jsonl.gz')):
        pair_path=Path(str(path).replace('.origins.jsonl.gz','.jsonl.gz'))
        with gzip.open(path,'rt') as origins,gzip.open(pair_path,'rt') as pairs:
            for eline,pline in zip(origins,pairs,strict=True):
                e,p=json.loads(eline),json.loads(pline)
                assert e['canonical_hash']==p['origin_hash']
                t=e['token_ids'].copy(); t[p['modified_token_index']]=27-t[p['modified_token_index']]
                h=digest(replay(t,check_answers=False)['serialized'])
                assert h==p['counterfactual_origin_hash'] and h in hashes
    bstats=json.loads((base/'corpus_statistics.json').read_text())
    # Independent token totals and first-crossing cursors, including the short 00006 shard.
    base_lengths=[len(json.loads(line))-1 for p in sorted((base/'train_shards').glob('*.tokens.jsonl')) for line in p.open()]
    assert len(base_lengths)==26048 and sum(base_lengths)==3004531
    assert len(list((base/'train_shards').glob('*.tokens.jsonl')))==7
    # Small two-build determinism with the same reservation sets and actual shard-7 seed.
    samples=[]
    for _ in range(2):
        with tempfile.TemporaryDirectory() as temp:
            b=Builder(Path(temp)); b.hashes=hashes.copy(); b.causal_prefixes=prefixes.copy()
            b.produce('train_shards/00007','lm_train',7,64)
            raw=gzip.decompress((Path(temp)/'train_shards/00007.jsonl.gz').read_bytes())
            samples.append(dict(metadata_sha256=hashlib.sha256(raw).hexdigest(),tokens_sha256=file_hash(Path(temp)/'train_shards/00007.tokens.jsonl'),rng=b.manifest))
    assert samples[0]==samples[1]
    out.mkdir(parents=True)
    # Preserve all data-bearing bytes; old summary files are retained under provenance/.
    summaries={'manifest.json','corpus_statistics.json','cpu_validation.json','postwrite_audit.json','all_sequence_hashes.txt'}
    inherited={}
    for p in sorted(base.rglob('*')):
        if not p.is_file(): continue
        rel=p.relative_to(base); dest=out/('provenance/base/'+str(rel) if str(rel) in summaries else rel)
        dest.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(p,dest)
        assert file_hash(p)==file_hash(dest)
        inherited[str(rel)]={'destination':str(dest.relative_to(out)),'sha256':file_hash(p)}
    b=Builder(out); b.hashes=hashes; b.causal_prefixes=prefixes
    b.manifest=copy.deepcopy(bm['splits']); b.statistics=bstats
    total=sum(base_lengths); seqs=len(base_lengths); index=7; budget=16_000_000
    while total<budget:
        tokens,n=b.produce(f'train_shards/{index:05d}','lm_train',index,4096,prediction_budget=budget-total)
        total+=tokens; seqs+=n; index+=1
        # No new causal pairs are created, so ordinary prefix storage is unnecessary.
        b.prefixes.clear()
    lengths=[len(json.loads(line))-1 for p in sorted((out/'train_shards').glob('*.tokens.jsonl')) for line in p.open()]
    assert len(lengths)==seqs and sum(lengths)==total and seqs%64==0
    assert total-sum(lengths[-64:])<budget<=total
    train=[v for k,v in b.statistics.items() if k.startswith('train_shards/')]
    coverage={k:len(set().union(*(set(v['histograms'][k]) for v in train))) for k in ('states','commands','truth_table','adjacent_operators')}
    assert coverage==dict(states=16,commands=48,truth_table=12,adjacent_operators=25)
    for name,info in b.manifest.items():
        if name.startswith('train_shards/'):
            idx=info['index']; n=info['accepted']
            assert n==4096 or idx in (6,index-1)
        elif info.get('requested_sequences') is not None: assert info['accepted']==info['requested_sequences']
        elif info.get('requested_pairs') is not None: assert info['accepted']==info['requested_pairs']
    frozen=dict(nominal_prediction_tokens=budget,actual_prediction_tokens=total,overshoot=total-budget,final_cursor=seqs,final_update=seqs//64,shards=index,prefix_sequences=26048,prefix_prediction_tokens=3004531)
    milestones={}; cumulative=0
    for cursor in range(64,seqs+1,64):
        cumulative+=sum(lengths[cursor-64:cursor])
        for boundary in design['training']['milestones_prediction_tokens']:
            if cumulative>=boundary and str(boundary) not in milestones:
                milestones[str(boundary)]=dict(update=cursor//64,cursor=cursor,prediction_tokens=cumulative)
    frozen['milestones']=milestones
    validation=dict(**self_test(),status='passed',train_coverage=coverage,train_prediction_tokens=total,train_sequences=seqs,batch_multiple_64=True,shard_7_two_build_determinism=samples[0],inherited_files=inherited,frozen_training=frozen)
    write_json(out/'corpus_statistics.json',b.statistics)
    write_json(out/'cpu_validation.json',validation)
    (out/'all_sequence_hashes.txt').write_text(''.join(h+'\n' for h in sorted(b.hashes)))
    manifest=copy.deepcopy(bm)
    manifest.update(generator_version='1.2.0',experiment_version='v1.2',base_manifest_sha256=file_hash(base/'manifest.json'),splits=b.manifest,all_sequence_hash_count=len(b.hashes),numpy_version=__import__('numpy').__version__,frozen_training=frozen,
                    code_sha256={str(p.relative_to(ROOT)):file_hash(p) for p in list((ROOT/'corpus').glob('*.py'))+[Path(__file__)]},
                    inherited_files=inherited,extension_elapsed_seconds=time.monotonic()-start)
    manifest['files']={str(p.relative_to(out)):{'sha256':file_hash(p),'bytes':p.stat().st_size} for p in sorted(out.rglob('*')) if p.is_file()}
    write_json(out/'manifest.json',manifest)
    auditor.audit(out)
    write_json(reports/'p1_frozen_training.json',frozen)
    write_json(reports/'p1_generation.json',dict(status='passed',manifest_sha256=file_hash(out/'manifest.json'),audit_sha256=file_hash(out/'postwrite_audit.json'),elapsed_seconds=time.monotonic()-start,**frozen))
    print(json.dumps(frozen,indent=2),flush=True)

if __name__=='__main__': run()
