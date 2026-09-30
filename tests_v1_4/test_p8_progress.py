from scripts.p8_progress import Progress


def test_resume_and_eta(tmp_path):
    p=tmp_path/'runs/a';p.mkdir(parents=True)
    (p/'update_01000.pt').touch();(p/'update_01000.json').write_text('{}')
    (p/'update_01250.pt').touch() # interrupted uncommitted payload is ignored
    t=Progress([{'name':'a'},{'name':'b'}],tmp_path,now=0)
    assert t.snapshot(now=0)['total_updates']==1000
    t.feed("a 1250 {'val_mse': 1}",now=50)
    s=t.snapshot(now=50)
    assert s['current_eta']==750 and s['selected_eta']==1750
    t.feed("a 1250 {'val_mse': 1}",now=55)
    assert t.snapshot(now=55)['total_updates']==1250
    t.feed("a 5000 {'val_mse': 1}",now=800)
    t.feed("b 100 {'train_mse': 1}",now=820)
    assert t.snapshot(now=820)['total_updates']==5100
    t.feed("b 250 {'val_mse': 1}",now=850)
    t.feed("b 500 {'val_mse': 1}",now=900)
    assert t.snapshot(now=900)['seconds_per_update']==.2


def test_selected_scope_and_escaped_logs(tmp_path):
    t=Progress([{'name':'a'},{'name':'b'}],tmp_path,names=['b'],now=0)
    t.feed("b 100 {'train_mse': 1}",now=20)
    assert t.snapshot(now=20)['selected_eta'] is None
    t.feed("b 250 {'val_mse': '<bad>'}",now=50)
    s=t.snapshot(now=50)
    assert s['selected_eta']==950 and s['total_eta']==1950
    assert '&lt;bad&gt;' in t.render()
    t.feed('unrelated 5000 {}',now=60)
    assert t.snapshot(now=60)['total_updates']==250
