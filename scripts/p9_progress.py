"""Read-only progress UI around the frozen P9 training subprocess.

No training imports, RNG calls, checkpoint writes, or changes to the command.
Progress uses last reported committed updates; ETA includes observed validation
and checkpoint time, excludes later audit/evaluation and unobserved startup I/O.
"""
import html
import json
from pathlib import Path
import queue
import re
import subprocess
import threading
import time


def duration(seconds):
    if seconds is None:return '측정 중'
    seconds=max(0,int(seconds));h,r=divmod(seconds,3600);m,s=divmod(r,60)
    return f'{h}시간 {m}분 {s}초' if h else f'{m}분 {s}초'


class Progress:
    def __init__(self,runs,output,names=None,now=None):
        self.runs=[r['name'] for r in runs];self.selected=self.runs if names is None else names
        if not self.selected or not set(self.selected)<=set(self.runs):raise ValueError('Unknown or empty run selection')
        self.steps={n:0 for n in self.runs};self.done=set();self.active=None
        self.started=time.monotonic() if now is None else now
        self.anchor_time=self.started;self.anchor_step=0;self.samples=[];self.current_samples=[]
        for name in self.runs:
            folder=Path(output)/'runs'/name
            # Advisory display only. The unchanged trainer checks identity/hash.
            committed=[int(p.stem.split('_')[1]) for p in folder.glob('update_*.pt') if p.with_suffix('.json').exists()]
            self.steps[name]=min(5000,max(committed,default=0))
            result=folder/'result.json'
            if result.exists():
                saved=json.loads(result.read_text())
                if saved.get('updates')==5000 and saved.get('run',{}).get('name')==name:
                    self.steps[name]=5000;self.done.add(name)
        self.active=next((n for n in self.selected if self.steps[n]<5000),None)
        self.anchor_step=self.steps[self.active] if self.active else 0
        self.last='입력·환경 검증 및 checkpoint 로드 대기'
        self.finished=False;self.failed=False

    def feed(self,line,now=None):
        now=time.monotonic() if now is None else now
        match=re.match(r'^(\S+) (\d+) \{',line)
        if not match or match[1] not in self.selected:return
        name,step=match[1],int(match[2])
        if not 0<step<=5000:return
        if name!=self.active:
            self.active=name;self.anchor_step=step;self.anchor_time=now;self.current_samples=[]
        else:
            delta=step-self.anchor_step;elapsed=now-self.anchor_time
            if delta>0 and elapsed>0:
                # The first sample includes startup; conservative until more samples.
                self.current_samples.append((delta,elapsed));self.samples.append((delta,elapsed))
            self.anchor_step=step;self.anchor_time=now
        self.steps[name]=max(self.steps[name],step)
        if step==5000:self.done.add(name)
        self.last=line.strip()

    def snapshot(self,now=None):
        now=time.monotonic() if now is None else now
        samples=self.current_samples[-8:] or self.samples[-8:]
        observed=sum(n for n,_ in samples)
        rate=sum(t for _,t in samples)/observed if observed>=250 else None
        remaining=sum(5000-self.steps[n] for n in self.selected)
        total_remaining=sum(5000-v for v in self.steps.values())
        current=5000-self.steps[self.active] if self.active else 0
        return dict(total_updates=sum(self.steps.values()),total_budget=5000*len(self.runs),
            completed_runs=len(self.done),run_count=len(self.runs),active=self.active,
            current_update=self.steps[self.active] if self.active else 0,elapsed=now-self.started,
            seconds_per_update=rate,current_eta=current*rate if rate is not None else None,
            selected_eta=remaining*rate if rate is not None else None,
            total_eta=total_remaining*rate if rate is not None else None,last=self.last,
            seconds_since_report=now-self.anchor_time,finished=self.finished,failed=self.failed)

    def render(self):
        s=self.snapshot();pct=100*s['total_updates']/s['total_budget']
        state='실패 — 아래 로그 확인' if self.failed else ('학습 명령 종료' if self.finished else '학습 진행 중')
        speed=f"{s['seconds_per_update']:.3f}초/update" if s['seconds_per_update'] is not None else '250 updates 관측 후 추정'
        return f'''<div style="border:1px solid #aaa;border-radius:10px;padding:16px;line-height:1.8">
<b>P9 TC 학습 · {state}</b><br>
전체 <b>{pct:.1f}%</b> · {s['total_updates']:,}/{s['total_budget']:,} updates · {s['completed_runs']}/{s['run_count']} runs 최종 update 도달<br>
<progress value="{s['total_updates']}" max="{s['total_budget']}" style="width:100%"></progress><br>
현재 <b>{html.escape(s['active'] or '없음')}</b> · {s['current_update']:,}/5,000<br>
이번 실행 경과 {duration(s['elapsed'])} · 속도 {speed}<br>
현재 run 남음 <b>{duration(s['current_eta'])}</b> · 이번 선택 runs 남음 <b>{duration(s['selected_eta'])}</b><br>
전체 24 runs 남음 추정 {duration(s['total_eta'])}<br>
<small>마지막 checkpoint 보고 {int(s['seconds_since_report'])}초 전. 진행률은 저장된 update 기준이며 검증·저장 중에도 경과시간은 갱신됩니다.
ETA는 관측된 학습·validation·저장 시간을 사용한 추정입니다. 향후 입력 로드·결과 hash 검사·독립 수치 검산·의미/인과 평가는 제외됩니다.</small><br>
<code>{html.escape(s['last'])}</code></div>'''


def run_with_progress(command,root,output,names=None):
    from IPython.display import HTML,display
    root,output=Path(root),Path(output)
    contract=json.loads((root/'experiment_v1_4/p9_r1/contract.json').read_text())
    tracker=Progress(contract['runs'],output,names)
    logdir=output/'progress_logs';logdir.mkdir(parents=True,exist_ok=True)
    logpath=logdir/f'training_{time.time_ns()}.log'
    handle=display(HTML(tracker.render()),display_id=True)
    lines=queue.Queue()
    proc=subprocess.Popen(command,cwd=root,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
    def reader():
        try:
            for line in proc.stdout:lines.put(line)
        finally:lines.put(None)
    worker=threading.Thread(target=reader,daemon=True);worker.start()
    tail=[];ended=False;last_paint=0.
    try:
        with logpath.open('w',buffering=1) as log:
            while not ended:
                try:line=lines.get(timeout=1)
                except queue.Empty:line=''
                if line is None:ended=True
                elif line:
                    log.write(line);tracker.feed(line);tail=(tail+[line])[-20:]
                now=time.monotonic()
                if ended or now-last_paint>=1:
                    handle.update(HTML(tracker.render()));last_paint=now
        code=proc.wait();tracker.finished=True;tracker.failed=code!=0
        handle.update(HTML(tracker.render()))
        print('전체 실행 로그:',logpath)
        if code:
            print(''.join(tail));raise subprocess.CalledProcessError(code,command)
    except BaseException:
        if proc.poll() is None:
            proc.terminate()
            try:proc.wait(timeout=10)
            except subprocess.TimeoutExpired:proc.kill();proc.wait()
        raise
    finally:proc.stdout.close();worker.join(timeout=2)
    return tracker.snapshot()
