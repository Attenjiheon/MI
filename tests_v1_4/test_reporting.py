import copy
import math
import numpy as np
import pytest
import torch
from corpus.language import render
from interp_v1_4.behavior import evaluate_records, pair_summary
from interp_v1_4.reporting import event_context, pair_intervals


def test_baselines_use_only_prior_prefix_including_nontarget_reads():
    e = render([1, 0, 0, 0], [([('NOT', 0, None)], 0), ([('SET', 1, 1)], 1),
               ([('NOT', 0, None)], 0), ([('SET', 0, 0)], 0)], 'fixture', 1)
    assert event_context(e, e['read_events'][0])['copy_prediction'] == 0
    context = event_context(e, e['read_events'][2])
    assert context['last_set_prediction'] == 1  # later SET must not leak
    assert context['copy_prediction'] == 1  # most recent READ is B, not A
    assert event_context(e, e['read_events'][3])['last_set_prediction'] == 0


def test_uniform_logits_full_token_ce_target_filter_and_empty_strata():
    class Uniform(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))
        def forward(self, ids, mask):
            return torch.zeros((*ids.shape, 15), device=ids.device) + self.anchor
    e = render([1, 0, 0, 0], [([('NOT', 0, None)], 0), ([('SET', 1, 1)], 1)], 'fixture', 1)
    e['target_read_ids'] = [1]
    result = evaluate_records(Uniform(), [e], target_only=True, microbatch=1)
    assert result['answer_count'] == 1
    assert result['prediction_tokens'] == len(e['token_ids'])-1
    assert result['all_token_ce'] == pytest.approx(math.log(15))
    assert result['answer_ce'] == pytest.approx(math.log(15))
    assert result['bit_mass'] == pytest.approx(2/15)
    assert result['accuracy'] == 0
    assert result['strata']['operator']['cells']['XOR']['accuracy'] is None
    assert result['strata']['operator']['coverage'] == pytest.approx(1/6)
    assert result['baselines']['latest_read_copy_accuracy'] == 0
    assert result['baselines']['latest_set_or_initialization_accuracy'] == 1


def pair_rows():
    return [dict(pair_id=str(i), sequence_id=m+str(i), member=m, cell_id='fixture',
                 operator='NOT', depth_bin='1', answer=i%2, correct=i%2,
                 answer_ce=float(i%2), binary_correct=i%2, bit_mass=1.0)
            for i in range(8) for m in ('first', 'repeat')]


def test_bootstrap_is_paired_deterministic_and_preserves_training_rng():
    rows = pair_rows()
    state = torch.get_rng_state().clone()
    np_state = copy.deepcopy(np.random.get_state())
    result = pair_summary(rows)
    assert result['uncertainty'] == pair_summary(list(reversed(rows)))['uncertainty']
    assert result['uncertainty']['paired_macro_gap']['accuracy_ci95'] == [0, 0]
    assert result['uncertainty']['valid_draws'] == 1000
    lo, hi = result['first']['cell']['cells']['fixture']['accuracy_ci95']
    assert lo < 0.5 < hi
    assert torch.equal(state, torch.get_rng_state())
    assert np.array_equal(np_state[1], np.random.get_state()[1])
    assert result['first']['cell']['macro_accuracy'] == 0.5
    with pytest.raises(ValueError, match='Duplicate'):
        pair_intervals(rows + [rows[0]])
