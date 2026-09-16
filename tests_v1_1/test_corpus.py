import unittest
from corpus.language import render, execute, select_target
from corpus.replay import validate, replay
from corpus.generate import self_test


class LanguageTests(unittest.TestCase):
    def test_exhaustive_and_reproducibility(self):
        self.assertEqual(self_test()['exhaustive_transitions'],768)

    def test_metadata_resets_and_indirect_dependency(self):
        program=[([('NOT',0,None)],1),([('XOR',0,1)],0),
                 ([('SET',0,1),('AND',0,2)],0),([('NOT',3,None)],0)]
        e=render([0,1,0,0],program,'test',0,True)
        validate(e)
        r=e['read_events']
        self.assertIsNone(r[0]['previous_value_or_null'])
        self.assertEqual(r[0]['updates_since_last_update'],1)
        self.assertEqual([x['structural_depth'] for x in r],[0,2,1,1])
        self.assertEqual(r[2]['reads_of_query_var_since_latest_set'],0)
        self.assertEqual(r[3]['reads_of_query_var_since_latest_set'],1)
        self.assertEqual(r[3]['updates_since_last_update'],1)
        self.assertEqual(r[0]['state_at_read'],[1,1,0,0])

    def test_zero_sensitivity_distinct_from_uncomputed(self):
        p=[([('AND',0,1)],0)]
        self.assertEqual(render([0,0,0,0],p,'t',0,True)['read_events'][0]['local_sensitivity'],[0,0,0,0])
        self.assertIsNone(render([0,0,0,0],p,'t',0,False)['read_events'][0]['local_sensitivity'])

    def test_bad_answer_rejected(self):
        e=render([0]*4,[([('NOT',0,None)],0)],'t',0)
        e['token_ids'][-2]=13
        with self.assertRaises(AssertionError): replay(e['token_ids'])

    def test_counterfactual_prior_answers_and_suffix(self):
        e=render([0]*4,[([('NOT',1,None)],1),([('NOT',0,None)],0)],'t',0)
        prefix=e['token_ids'][:e['read_events'][1]['answer_token_index']]
        altered=prefix.copy(); altered[3]=14
        self.assertEqual(replay(prefix,partial=True)['answer'],1)
        self.assertEqual(replay(altered,partial=True)['answer'],0)
        leaked=prefix.copy(); leaked[6]=14
        with self.assertRaises(AssertionError): replay(leaked,partial=True)


if __name__=='__main__': unittest.main()
