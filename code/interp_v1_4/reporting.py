"""Reporting-only v1.4 revision 1; no selection or gate threshold changes."""
from collections import defaultdict
import numpy as np
from .runtime import seed

BOOTSTRAP_DRAWS = 1000


def interval(draws):
    return [float(x) for x in np.percentile(draws, [2.5, 97.5])]


def pair_intervals(rows):
    """Resample independent origins within each cell, keeping both members together."""
    grouped = defaultdict(dict)
    for row in rows:
        members = grouped[row['cell_id']].setdefault(row['pair_id'], {})
        if row['member'] in members:
            raise ValueError('Duplicate pair member')
        members[row['member']] = row
    draws_by_cell = {}
    cell_ci = {'first': {}, 'repeat': {}}
    for cell, pairs in sorted(grouped.items()):
        if any(set(p) != {'first', 'repeat'} for p in pairs.values()):
            raise ValueError('Incomplete pair')
        values = np.array([[p[m][metric] for m in ('first', 'repeat')
                            for metric in ('correct', 'answer_ce')]
                           for _, p in sorted(pairs.items())], dtype=float)
        rng = np.random.Generator(np.random.PCG64(seed('behavior-bootstrap-r1', cell)))
        indices = rng.integers(0, len(values), size=(BOOTSTRAP_DRAWS, len(values)))
        draws = values[indices].mean(axis=1)
        draws_by_cell[cell] = draws
        for i, member in enumerate(('first', 'repeat')):
            cell_ci[member][cell] = dict(accuracy_ci95=interval(draws[:, 2*i]),
                                        answer_ce_ci95=interval(draws[:, 2*i+1]))
    macro = np.stack(list(draws_by_cell.values())).mean(axis=0)
    return dict(cells=cell_ci, macro={m: dict(accuracy_ci95=interval(macro[:, 2*i]),
                answer_ce_ci95=interval(macro[:, 2*i+1])) for i, m in enumerate(('first', 'repeat'))},
                paired_macro_gap=dict(accuracy_ci95=interval(macro[:, 0]-macro[:, 2]),
                                      answer_ce_ci95=interval(macro[:, 1]-macro[:, 3])),
                method='within-cell origin cluster percentile bootstrap; paired members shared',
                draws=BOOTSTRAP_DRAWS, valid_draws=BOOTSTRAP_DRAWS,
                seed_namespace='20260920|experiment-spec-v1.4|behavior-bootstrap-r1|<cell_id>',
                cell_seeds={cell: seed('behavior-bootstrap-r1', cell) for cell in grouped})


def event_context(example, event):
    uid = event['last_update_id_for_query_var_or_null']
    update = example['update_events'][uid] if uid is not None else None
    q = event['query_var']
    last_set = example['initial_state']['ABCD'.index(q)]
    for u in example['update_events']:
        if u['end_token_index'] >= event['query_token_index']:
            break
        if u['op'] == 'SET' and u['dst'] == q:
            last_set = u['literal_or_null']
    previous = [r for r in example['read_events'] if r['query_token_index'] < event['query_token_index']]
    distance = event['updates_since_last_update']
    truth = update['input_truth_pattern_or_null'] if update else None
    return dict(query_var=q, operator=update['op'] if update else 'initialization',
                blocks=len(example['blocks']), distance='0' if distance == 0 else '1-2' if distance <= 2 else '3+',
                truth_table=update['op']+':'+truth if truth is not None else None,
                last_set_prediction=last_set,
                copy_prediction=previous[-1]['answer'] if previous else 0)


def extended_summary(rows):
    domains = dict(operator=['initialization', 'SET', 'NOT', 'AND', 'OR', 'XOR'],
                   query_var=list('ABCD'), answer=[0, 1], blocks=list(range(8, 25)),
                   distance=['0', '1-2', '3+'],
                   truth_table=[op+':'+bits for op in ('AND', 'OR', 'XOR') for bits in ('00', '01', '10', '11')])
    strata = {}
    for key, domain in domains.items():
        groups = defaultdict(list)
        for row in rows:
            groups[str(row[key])].append(row)
        cells = {}
        for value in domain:
            group = groups[str(value)]
            cells[str(value)] = dict(count=len(group), accuracy=sum(r['correct'] for r in group)/len(group) if group else None,
                                    answer_ce=sum(r['answer_ce'] for r in group)/len(group) if group else None)
        observed = [c for c in cells.values() if c['count']]
        strata[key] = dict(cells=cells, coverage=len(observed)/len(domain),
                           macro_accuracy=sum(c['accuracy'] for c in observed)/len(observed) if observed else None,
                           macro_answer_ce=sum(c['answer_ce'] for c in observed)/len(observed) if observed else None)
    majority = int(sum(r['answer'] for r in rows) > len(rows)/2)
    return dict(strata=strata, baselines=dict(
        count=len(rows), majority_prediction=majority,
        split_majority_accuracy=sum(r['answer'] == majority for r in rows)/len(rows),
        independent_bernoulli_expected_accuracy=0.5,
        latest_set_or_initialization_accuracy=sum(r['answer'] == r['last_set_prediction'] for r in rows)/len(rows),
        latest_read_copy_accuracy=sum(r['answer'] == r['copy_prediction'] for r in rows)/len(rows)))
