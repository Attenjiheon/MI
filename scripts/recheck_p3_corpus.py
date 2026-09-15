"""Re-run the independent corpus audit without overwriting immutable evidence."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import corpus.audit as audit

def write_report(path, result):
    (ROOT/'experiment_v1/results/p3_cpu_corpus_reaudit.json').write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__':
    audit.write_json=write_report
    audit.audit(ROOT/'data/language_v1')
