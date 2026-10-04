"""Independent coverage/numeric checks for the final synthesis, without inference."""
from pathlib import Path
import argparse,csv,hashlib,itertools,json,platform,sys,zipfile
import numpy as np
import pandas as pd


def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
def require(condition,message):
 if not condition:raise AssertionError(message)


def verify(root):
 p=Path(root);manifest=json.loads((p/'input_manifest.json').read_text());zips={};checked=0
 for ref,value in manifest['files'].items():
  if '::' in ref:
   path,member=ref.split('::',1)
   if path not in zips:zips[path]=zipfile.ZipFile(path)
   actual=hashlib.sha256(zips[path].read(member)).hexdigest()
  else:actual=sha(ref)
  require(actual==value['sha256'],'Input changed: '+ref);checked+=1
 for ref,value in manifest['archives'].items():require(sha(ref)==value['sha256'],'Archive changed: '+ref)
 b=pd.read_csv(p/'behavior.csv')
 empty=b[(b.metric=='count')&(pd.to_numeric(b.value,errors='coerce')==0)]
 for row in empty.itertuples():
  q=b[(b.lm_seed==row.lm_seed)&(b.suite==row.suite)&(b.stratum==row.stratum)&(b.cell==row.cell)&b.metric.isin(['accuracy','answer_ce'])]
  require(len(q)==2 and q.value.isna().all() and q.na_reason.notna().all(),'Empty behavior stratum must have explicit NA')
 gaps=b[b.suite=='first_minus_repeat'];require(len(gaps)==6 and gaps.ci_low.notna().all(),'Paired first/repeat macro gap')
 r=pd.read_csv(p/'dictionary_registry.csv');s=pd.read_csv(p/'semantic_metrics.csv',low_memory=False);f=pd.read_csv(p/'fidelity_metrics.csv');c=pd.read_csv(p/'causal_metrics.csv');cov=pd.read_csv(p/'matching_coverage.csv');ld=pd.read_csv(p/'layer_differences.csv',low_memory=False);su=pd.read_csv(p/'lm_seed_summary.csv',low_memory=False)
 expected={(lm,l,t,k,0) for lm in (0,1,2) for l in (0,3,7,11) for t in ('sae','tc') for k in (4,16)}|{(0,l,t,k,1) for l in (0,3,7,11) for t in ('sae','tc') for k in (4,16)}
 read=r[r.position_type=='READ'];require(len(read)==64,'READ count');require(set(map(tuple,read[['lm_seed','layer','tool','k','sparse_seed']].to_numpy()))==expected,'READ grid')
 update=r[r.position_type=='UPDATE'];require(len(update)==12 and set(update.layer)=={3},'Update grid');require(r.name.is_unique,'Duplicate runs');require((r.updates==5000).all(),'Training updates')
 require(int(read.position_draws.sum())==163840000 and int(update.position_draws.sum())==30720000,'Actual draw counts')
 required_controls={'selected','selected_matched','identity','mean','approximation','full_donor','random_unmatched','random_matched','coordinate','random_direction_selected','random_direction_pure'}
 for row in read.itertuples():
  cs=c[c.name==row.name];require(set(cs.control)==required_controls,'Missing causal control '+row.name)
  require(set(cs['size'])=={'single','up_to_four'},'Feature sizes')
  require(set(cs.condition)=={'memory','composition','all'},'Conditions')
  ss=s[(s.name==row.name)&(s.representation==row.tool)&(s.label=='current')&(s.metric=='balanced_accuracy')]
  require(set(ss.domain)=={'iid','transfer'} and set(ss.subset)=={'all','current_ne_previous'},'Semantic domains/subsets '+row.name)
  require(set(ss['size'])=={'single','up_to_four'},'Semantic feature sizes')
  require(set(ss.candidates)=={512},'Latent candidates')
  eq=s[(s.name==row.name)&(s.representation==row.tool+'_128')];require(len(eq)>0 and set(eq.candidates)=={128},'128 candidate controls')
  ff=f[f.name==row.name];require(set(ff[ff.category=='fidelity'].split)=={'train','val','test'},'Fidelity splits')
  require(set(ff[ff.category=='replacement'].metric)=={'original_ce','patched_ce','delta_ce','original_accuracy','patched_accuracy','delta_accuracy'},'Replacement metrics')
  dead=ff[ff.metric=='dead_fraction'].value.to_numpy();train=ff[(ff.metric=='inactive_fraction')&(ff.split=='train')].value.iloc[0];require(np.allclose(dead,train),'Dead definition')
  # Every matched comparison uses the same successfully matched origins on both sides.
  keys=['changed','condition','size','subset','metric'];sel=cs[cs.control=='selected_matched'].set_index(keys);ran=cs[cs.control=='random_matched'].set_index(keys)
  require(sel.index.equals(ran.index) and np.array_equal(sel.origins,ran.origins),'Matching subset denominators')
 require(set(s[(s.phase=='P10')&(s.representation=='latent')].tool)=={'sae','tc'},'Update tool identity')
 require(set(s[s.representation.isin(['token','token_position'])].candidates)=={17},'Token/position baseline dimension')
 require((cov.matched+cov.unmatched==cov.pairs).all(),'Coverage denominators')
 require(np.allclose(cov.matched_fraction,cov.matched/cov.pairs),'Coverage fractions')
 for name,df in [('causal',c),('layers',ld)]:
  require((df.bootstrap_requested==1000).all(),name+' draws');require(((df.bootstrap_valid>=0)&(df.bootstrap_valid<=1000)).all(),name+' valid')
  require((df.ci_low<=df.ci_high).all(),name+' intervals');require(np.isfinite(df.value).all(),name+' nonfinite')
 require(set(ld.layer_high-ld.layer_low)>={3,4,7,8,11},'Layer contrasts')
 require(len(ld[ld.task=='replacement'])==192,'Additional paired contrasts')
 defined=s[s.value.notna()];require(np.isfinite(defined.value).all(),'Semantic nonfinite')
 require(((defined.value>=0)&(defined.value<=1)).all(),'Semantic metric range')
 ba=s[s.metric=='balanced_accuracy'];require(ba.value.notna().all(),'Unreported fitting/support failures')
 require((ba.bootstrap_requested==1000).all() and (ba.bootstrap_valid==1000).all(),'BA CI coverage')
 auroc=s[(s.metric=='binary_auroc')&s.value.isna()];require((auroc.bootstrap_valid==0).all() and auroc.na_reason.notna().all(),'AUROC NA transparency')
 require((su.n_lm==3).all(),'LM repetition groups must each have three seeds')
 require(np.allclose(su['mean'],su[['seed0','seed1','seed2']].mean(axis=1)),'LM means')
 require(np.allclose(su.minimum,su[['seed0','seed1','seed2']].min(axis=1)) and np.allclose(su.maximum,su[['seed0','seed1','seed2']].max(axis=1)),'LM ranges')
 # Recompute reported confusion-matrix BA independently for every applicable row.
 max_error=0.
 for row in ba.itertuples():
  cm=np.asarray(json.loads(row.confusion_matrix),float);den=cm.sum(1)
  if np.all(den>0):max_error=max(max_error,abs(float(np.mean(np.diag(cm)/den))-row.value))
 require(max_error<1e-12,'Confusion matrix BA mismatch')
 require(len(list((p/'figures').glob('*.png')))==13,'Required figures')
 require((p/'report.md').exists() and (p/'update_summary.csv').exists(),'Report missing')
 for filename in ('behavior.csv','semantic_metrics.csv','fidelity_metrics.csv','causal_metrics.csv','run_registry.csv','dictionary_registry.csv','environment_index.csv','reproduction_inputs.csv'):
  require((p/filename).stat().st_size>0,'Empty artifact '+filename)
 validation=dict(status='passed',input_files_verified=checked,archive_hashes_verified=len(manifest['archives']),read_runs=64,update_runs=12,
  causal_controls=11,empty_behavior_strata=len(empty),paired_behavior_gap_rows=len(gaps),paired_replacement_contrasts=192,ba_reports=len(ba),ba_max_confusion_error=max_error,undefined_auroc_reports=len(auroc),
  matching_rows=len(cov),unmatched_range=[int(cov.unmatched.min()),int(cov.unmatched.max())],figures=13,
  lm_summary_groups=len(su),all_lm_summary_groups_have_three_seeds=True,python=sys.version,platform=platform.platform(),
  numpy=np.__version__,pandas=pd.__version__,model_inference=False,feature_reselection=False,
  limitations=['Fidelity point estimates retain descriptive seed ranges, not invented bootstrap CIs.','Sparse seed differences are point estimates within LM seed 0.'])
 (p/'verification.json').write_text(json.dumps(validation,indent=2)+'\n');print(json.dumps(validation,indent=2))


if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);verify(parser.parse_args().output)
