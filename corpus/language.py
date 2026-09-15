"""Language v1.0 sampler and metadata interpreter (no model dependencies)."""
import hashlib
import itertools
import numpy as np

VOCAB = ['PAD', 'BOS', 'EOS', 'SET', 'NOT', 'AND', 'OR', 'XOR', 'READ',
         'A', 'B', 'C', 'D', 'B0', 'B1']
OPS = ['SET', 'NOT', 'AND', 'OR', 'XOR']
PATTERNS = [('XOR', 'AND', 'OR'), ('OR', 'XOR', 'AND')]
MASTER = 20260909


def seed(purpose, index=0):
    return int.from_bytes(hashlib.sha256(
        f'{MASTER}|language-v1.0|{purpose}|{index}'.encode('ascii')).digest()[:8], 'big')


def digest(tokens):
    return hashlib.sha256(','.join(map(str, tokens)).encode('ascii')).hexdigest()


def execute(state, command):
    op, d, x = command
    s = state.copy()
    if op == 'SET': s[d] = x
    elif op == 'NOT': s[d] = 1 - state[d]
    elif op == 'AND': s[d] = state[d] & state[x]
    elif op == 'OR': s[d] = state[d] | state[x]
    elif op == 'XOR': s[d] = state[d] ^ state[x]
    else: raise ValueError(op)
    return s


def sample(rng, length=False, pattern=None):
    initial = rng.integers(0, 2, 4).tolist()
    blocks = []
    for _ in range(int(rng.integers(33 if length else 8, 49 if length else 25))):
        u = float(rng.random())
        commands = []
        for _ in range(1 if u < .80 else 2 if u < .95 else 3):
            v = float(rng.random())
            op = OPS[0 if v < .25 else 1 if v < .5 else 2 if v < 2/3 else 3 if v < 5/6 else 4]
            d = int(rng.integers(4))
            x = (int(rng.integers(2)) if op == 'SET' else None if op == 'NOT'
                 else [i for i in range(4) if i != d][int(rng.integers(3))])
            commands.append((op, d, x))
        q = d if rng.random() < .5 else [i for i in range(4) if i != d][int(rng.integers(3))]
        blocks.append((commands, q))
    injected = None
    if pattern is not None:
        injected = int(rng.integers(len(blocks)))
        d = int(rng.integers(4))
        blocks[injected] = ([(op, d, [i for i in range(4) if i != d][int(rng.integers(3))])
                             for op in PATTERNS[pattern]], d)
    return initial, blocks, injected


def pattern_ok(blocks, injected=None, pattern=None):
    for i, (commands, _) in enumerate(blocks):
        p = tuple(c[0] for c in commands)
        if i == injected:
            if p != PATTERNS[pattern]: return False
        elif p in PATTERNS: return False
    return True


def render(initial, program, split, rng_seed, sensitivity=False, injected=None, pattern=None):
    tokens, roles = [1], ['bos']
    def append(ids, names):
        start = len(tokens)
        tokens.extend(ids); roles.extend(names)
        return start, len(tokens)-1
    state = initial.copy()
    nodes = [{'node_id': i, 'parent_ids': [], 'depth': 0, 'initialization_var': VOCAB[9+i]}
             for i in range(4)]
    current = list(range(4))
    last = [None]*4; ends = []; previous = [None]*4
    counts = [0]*4; since_update = [0]*4; since_set = [0]*4; logical_since_set = [0]*4
    for d in range(4):
        _, end = append([3, 9+d, 13+state[d]], ['init_op', 'init_dst', 'init_literal'])
        ends.append(end)
    updates, reads, blocks = [], [], []
    for bid, (commands, q) in enumerate(program):
        block_start = state.copy(); block_token = len(tokens); first_update = len(updates)
        for op, d, x in commands:
            before = state.copy(); uid = len(updates)
            ids = [VOCAB.index(op), 9+d] + ([] if op == 'NOT' else [13+x if op == 'SET' else 9+x])
            names = ['update_op', 'update_dst'] + ([] if op == 'NOT' else ['update_literal' if op == 'SET' else 'update_src'])
            start, end = append(ids, names)
            state = execute(before, (op, d, x))
            parents = [] if op == 'SET' else [current[d]] if op == 'NOT' else [current[d], current[x]]
            depth = 0 if not parents else 1+max(nodes[n]['depth'] for n in parents)
            node = len(nodes)
            nodes.append({'node_id': node, 'parent_ids': parents, 'depth': depth, 'update_id': uid})
            current[d] = node; last[d] = uid; ends[d] = end; previous[d] = before[d]
            counts[d] += 1; since_update[d] = 0
            if op == 'SET': since_set[d] = 0; logical_since_set[d] = 0
            else: logical_since_set[d] += 1
            binary = op in OPS[2:]
            updates.append(dict(update_id=uid, block_id=bid, op=op, dst=VOCAB[9+d],
                src_or_null=VOCAB[9+x] if binary else None, literal_or_null=x if op=='SET' else None,
                start_token_index=start, end_token_index=end, state_before=before, state_after=state.copy(),
                dst_before=before[d], src_before_or_null=before[x] if binary else None, dst_after=state[d],
                input_truth_pattern_or_null=f'{before[d]}{before[x]}' if binary else None,
                dependency_node_id=node, structural_depth=depth))
        _, query = append([8, 9+q], ['read_op', 'read_query'])
        _, answer = append([13+state[q]], ['read_answer'])
        sens = None
        if sensitivity:
            sens = []
            for v in range(4):
                altered = block_start.copy(); altered[v] ^= 1
                for command in commands: altered = execute(altered, command)
                sens.append(int(altered[q] != state[q]))
        reads.append(dict(read_id=len(reads), block_id=bid, query_var=VOCAB[9+q],
            query_token_index=query, answer_token_index=answer, answer=state[q], state_at_read=state.copy(),
            previous_value_or_null=previous[q], last_update_id_for_query_var_or_null=last[q],
            updates_since_last_update=len(updates) if last[q] is None else len(updates)-1-last[q],
            tokens_since_last_update=query-ends[q], reads_of_query_var_since_last_update=since_update[q],
            reads_of_query_var_since_latest_set=since_set[q], same_as_last_dst=q==commands[-1][1],
            dependency_node_id=current[q], structural_depth=nodes[current[q]]['depth'],
            local_sensitivity=sens, answer_is_target=False, query_update_count=counts[q],
            logical_updates_since_latest_set=logical_since_set[q],
            previous_value_changed=previous[q] is not None and previous[q]!=state[q]))
        since_update[q] += 1; since_set[q] += 1
        blocks.append(dict(block_id=bid, start_token_index=block_token, end_token_index=answer,
            update_ids=list(range(first_update, len(updates))), read_id=len(reads)-1, state_at_start=block_start))
    append([2], ['eos'])
    h = digest(tokens)
    return dict(schema_version='language-v1.0', sequence_id=f'{split}:{h}', split=split,
        rng_seed=rng_seed, token_ids=tokens, token_roles=roles, initial_state=initial.copy(),
        blocks=blocks, update_events=updates, read_events=reads, dependency_nodes=nodes,
        canonical_hash=h, target_read_ids=[], holdout_pattern_id=pattern,
        holdout_block_id=injected)


def select_target(example, rid):
    example['target_read_ids'] = [rid]
    example['read_events'][rid]['answer_is_target'] = True


def all_commands():
    return [('SET', d, b) for d in range(4) for b in range(2)] + [('NOT', d, None) for d in range(4)] + [
        (op, d, r) for op in OPS[2:] for d in range(4) for r in range(4) if d != r]
