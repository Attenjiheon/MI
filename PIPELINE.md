# Experiment Pipeline

## 1. Data with exactly known states

The data consists of programs that update the variables A/B/C/D with SET, NOT, AND, OR, and XOR, and that query them with READ. The answer to a READ is the execution result, B0 or B1. The LM receives only token IDs and a padding mask. Metadata such as the current and previous states and the relations between operations is used only for evaluation.

Shards are generated in advance on a CPU, and their answers are checked by an independent parser and replay. All three LM seeds consume the same shards in the same order, and each shard is used once. Sequences and READ prefixes are checked for overlap across splits. Pairs derived from the same origin are kept in the same split.

In `READ A B1`, the analysis position is **right after A is processed**. The logits at this position predict the next token, B1. Therefore, the activation that is used has not yet seen the answer token. For the interpretation analyses, the sequences are split first, and then READ positions are sampled: 50,000 for train, 10,000 for validation, and 20,000 for test.

**Code:** [language execution](code/corpus/language.py), [independent replay](code/corpus/replay.py), [pair generation](code/corpus/v1_4.py), [corpus construction](code/scripts/generate_v1_4_corpus.py).

## 2. Freezing the LMs after checking their behavior

The model is a 12-block Transformer with residual width 256, MLP width 1,024, and 4 heads. It has a vocabulary of 15 tokens, a context length of 768, RoPE, dropout 0, and untied embedding and unembedding matrices. It has 9,485,312 parameters in total.

Each seed has a budget of about 64M prediction tokens. Training uses an effective batch size of 64, AdamW with a learning rate of 3e-4, and FP32. The CE weight is 4 for READ answer tokens and 1 for all other non-padding tokens. The loss is normalized by the sum of the weights. Answer positions are identified by their token pattern.

The checkpoint is chosen from 10 milestones. The selection criterion is the 42-cell macro answer CE on first members in the select split. If several checkpoints are within 1e-4 nats of the global minimum, the one with the lowest general answer CE is chosen, and remaining ties go to the earlier update. A separate gate split is not used to reselect the checkpoint. To pass the gate, a model needs at least 99% general READ accuracy. It also needs at least 95% on each designated diagnostic, on first and repeat queries, and on each group by operation, depth, and answer, as well as full coverage. Seeds 1 and 2 were run after seed 0 passed, and all three seeds passed.

The test set was evaluated only after all training and selection decisions were frozen. After that, the LMs were not changed because of SAE or TC performance.

**Code:** [model and hooks](code/interp_v1_4/model.py), [loss and optimizer](code/interp_v1_4/training.py), [selection and training](code/interp_v1_4/cli.py), [behavioral gate](code/interp_v1_4/behavior.py), [frozen test](code/interp_v1_4/frozen_test.py).

## 3. Comparing representations at the same position

Three representations are collected at the READ position from each frozen LM.

| Representation | Definition | Used for |
|---|---|---|
| `h_l` | Residual output of the block, before the final LayerNorm | SAE input and reconstruction, and residual patching |
| `u_l` | Normalized input to the block's MLP | TC input |
| `m_l` | Output of the block's MLP | TC target and MLP-output patching |

Full linear probes are fit on all 12 layers as a diagnostic. The sparse analysis is done at blocks 0, 3, 7, and 11. Each layer has its own dictionary and preprocessing, but all layers use the same sequences and READ positions.

**Code:** [extraction and joining](code/interp_v1_4/p5.py), [probe fitting and selection](code/interp_v1_4/p5_probe.py).

## 4. Training the SAEs and TCs

The SAE learns `h→h`, and the TC learns `u→m`. The mean vector and a scalar scale are computed from the train activations and then fixed. Each dictionary has width 512 and applies TopK after a ReLU. Both k=4 and k=16 are evaluated. State labels are not used in either training loss, and the TC target is the original MLP output.

Each run performs 5,000 updates with 512 positions per batch. Validation MSE is computed every 250 updates, and the checkpoint with the lowest value is selected. If there is a tie, the earlier update is chosen. Early stopping and reinitialization of dead features are not used.

- Main experiment: 3 LMs × 4 layers × 2 tools × 2 values of k = **48 runs**.
- Initialization repeats: sparse seed 1 on LM seed 0, 4 × 2 × 2 = **16 runs**.
- All required READ runs: **64 runs**, 320,000 updates, and 163.84M position draws.

**Code:** [dictionary operations](code/interp_v1_4/dictionary.py), [SAE training](code/interp_v1_4/p6.py), [TC training](code/interp_v1_4/p8.py), [initialization repeats](code/interp_v1_4/p9.py).

## 5. Semantics, fidelity, and behavioral substitution

The main label is the current value at the READ. Features are ranked on the train split. Then linear probes that use a single feature or up to 4 features are fit on train, selected on validation, and evaluated. The lambda, the number of features, and the binary threshold are all fixed before the test set is used.

The comparison covers the full probe, the 256 original coordinates, 512 random directions, and the 512 SAE or TC latents. The number of candidates differs between these representations. To separate out this effect, a control that draws 128 candidates from each representation with a fixed RNG is also run. Two cases are reported separately: the subset where the current value differs from the previous value, and transfer from queries on A/B/C to queries on D.

Fidelity is measured with NMSE, R², explained variance, L0, and the fraction of dead features on train. The SAE reconstructs h, while the TC predicts m. Because the targets differ, their scores are not ranked against each other as if they measured the same task. For behavioral substitution, only one fixed READ at one layer in each sequence is replaced with the approximated activation. The resulting changes in CE and accuracy are then compared as paired differences.

**Code:** [SAE evaluation](code/interp_v1_4/p7.py), [TC evaluation](code/interp_v1_4/p8_evaluation.py), [metrics, matching, and bootstrap](code/interp_v1_4/p7_metrics.py).

## 6. Counterfactual interventions and controls

Each pair consists of an origin program and a donor program that differ in a single SET bit, and both are replayed independently. The READ answers before the target READ are kept the same. Two kinds of pairs are evaluated: changed pairs, where the target answer differs, and unchanged pairs, where it stays the same. Memory and composition conditions are distinguished, and interventions are run in both directions.

The decoder difference of the selected features is added to the original activation, so that the approximation error is preserved. For the SAE, this is applied to h. For the TC, it is applied to m, and the TC's residual skip is kept. The frozen controls are reported alongside the main results: identity/null, full donor, approximate substitution, coordinate and random directions, and matched random. The full donor is not assumed to be a guaranteed upper bound.

The main metrics are the change in logit margin on changed pairs, the answer flip rate by argmax over 15 tokens, and the rate of induced errors on unchanged pairs. Matched random and selected features are compared on **the same subset of origins for which matching succeeded**. The coverage that is lost because of unmatched origins is also recorded.

**Code:** [SAE patching](code/interp_v1_4/p7_patching.py), [TC patching](code/interp_v1_4/p8_patching.py), [TC causal evaluation](code/interp_v1_4/p8_causal.py).

## 7. Repeats, uncertainty, and aggregation

Uncertainty is estimated with a 1,000-draw percentile 95% cluster bootstrap. The resampling unit is the sequence ID for semantic metrics and the origin ID for causal metrics. Pair directions and feature candidates are not counted as independent samples. Differences between layers are paired comparisons that use the same resamples. Invalid draws, such as draws in which a class is missing, and NA values are kept in the results.

Repeats with a different sparse seed are done only for LM seed 0. A latent with the same ID in two dictionaries is not treated as the same feature. The mean and range over the 3 LMs are reported separately from the differences caused by sparse initialization. The 12 dictionaries at Update block 3, with their semantic and transfer analyses, are kept separately as exploratory results. The following were not performed: dictionaries at additional layers, length evaluation on GPU, m→m SAEs, and causal interventions at Update positions.

**Code:** [initialization comparison](code/interp_v1_4/p9_comparison.py), [Update](code/interp_v1_4/p10.py), [original final aggregation](code/interp_v1_4/p11.py). The hashes of the actual runs and selections are in the [dictionary registry](results/dictionary_registry.csv.gz). The original completion decision is in [completion.json](provenance/completion.json).

**Correction:** In the frozen P11 `semantic_metrics.csv`, the READ random rows list 256 candidates, but 512 candidates were actually fit. This correction follows the recorded check of the 64 original fits in the [correction evidence](provenance/manuscript_evidence.json). The original CSV is preserved without changes to its bytes, scores, or selections. This document uses the corrected number of candidates.
