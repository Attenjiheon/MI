"""Independent token parser, deliberately not using the generator's executor."""
from .language import VOCAB

TABLES = {5: (0, 0, 0, 1), 6: (0, 1, 1, 1), 7: (0, 1, 1, 0)}


def replay(tokens, partial=False, check_answers=True):
    assert tokens[0] == 1
    s = [0]*4; roles = ['bos']; updates = []; reads = []; serialized = [1]
    i = 1
    for d in range(4):
        assert tokens[i:i+2] == [3, 9+d] and tokens[i+2] in (13,14)
        s[d] = tokens[i+2]-13
        serialized.extend([3, 9+d, s[d]+13]); roles.extend(['init_op','init_dst','init_literal']); i += 3
    bid = 0; group = 0
    while i < len(tokens):
        op = tokens[i]
        if op == 2:
            assert i == len(tokens)-1 and group == 0
            roles.append('eos'); serialized.append(2); i += 1; break
        if op == 8:
            assert 1 <= group <= 3 and 9 <= tokens[i+1] <= 12
            q = tokens[i+1]-9
            roles.extend(['read_op','read_query']); serialized.extend([8, q+9])
            if i+2 == len(tokens) and partial:
                return dict(answer=s[q], state=s.copy(), reads=reads, updates=updates, roles=roles, serialized=serialized)
            assert tokens[i+2] in (13,14)
            if check_answers: assert tokens[i+2] == 13+s[q]
            roles.append('read_answer'); serialized.append(13+s[q])
            reads.append(dict(block_id=bid, query=q, query_index=i+1, answer_index=i+2, answer=s[q], state=s.copy()))
            bid += 1; group = 0; i += 3
        else:
            assert op in (3,4,5,6,7) and 9 <= tokens[i+1] <= 12
            d = tokens[i+1]-9; before = s.copy(); start = i
            roles.extend(['update_op','update_dst']); serialized.extend([op, d+9])
            if op == 4:
                s[d] = (1,0)[s[d]]; i += 2
            else:
                x = tokens[i+2]; serialized.append(x)
                if op == 3:
                    assert x in (13,14); s[d] = x-13; roles.append('update_literal')
                else:
                    assert 9 <= x <= 12 and x != d+9
                    s[d] = TABLES[op][2*s[d]+s[x-9]]; roles.append('update_src')
                i += 3
            group += 1
            assert group <= 3
            updates.append(dict(start=start, end=i-1, before=before, after=s.copy(), block_id=bid))
    assert not partial and tokens[-1] == 2 and i == len(tokens)
    return dict(reads=reads, updates=updates, roles=roles, serialized=serialized)


def validate(e):
    t = e['token_ids']; parsed = replay(t)
    assert parsed['serialized'] == t and parsed['roles'] == e['token_roles']
    assert len(parsed['updates']) == len(e['update_events'])
    assert len(parsed['reads']) == len(e['read_events'])
    for a,b in zip(parsed['updates'], e['update_events']):
        assert (a['start'],a['end'],a['before'],a['after'],a['block_id']) == (
            b['start_token_index'],b['end_token_index'],b['state_before'],b['state_after'],b['block_id'])
    for a,b in zip(parsed['reads'], e['read_events']):
        assert (a['query_index'],a['answer_index'],a['answer'],a['state'],a['block_id']) == (
            b['query_token_index'],b['answer_token_index'],b['answer'],b['state_at_read'],b['block_id'])
        assert b['query_token_index']+1 == b['answer_token_index']
        q = a['query']; prior = [u for u in e['update_events'] if u['end_token_index'] < a['query_index']]
        relevant = [u for u in prior if u['dst'] == VOCAB[9+q]]
        latest = relevant[-1] if relevant else None
        assert b['previous_value_or_null'] == (latest['dst_before'] if latest else None)
        assert b['last_update_id_for_query_var_or_null'] == (latest['update_id'] if latest else None)
        end = latest['end_token_index'] if latest else 3+3*q
        assert b['tokens_since_last_update'] == a['query_index']-end
        assert b['updates_since_last_update'] == sum(u['end_token_index']>end for u in prior)
        sets = [u for u in relevant if u['op']=='SET']
        set_end = sets[-1]['end_token_index'] if sets else 3+3*q
        history = [r for r in parsed['reads'] if r['query']==q and r['query_index']<a['query_index']]
        assert b['reads_of_query_var_since_last_update'] == sum(r['query_index']>end for r in history)
        assert b['reads_of_query_var_since_latest_set'] == sum(r['query_index']>set_end for r in history)
    assert e['target_read_ids'] == [r['read_id'] for r in e['read_events'] if r['answer_is_target']]
    return parsed
