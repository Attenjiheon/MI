# P2 debug fixture

`sequences.json` contains 192 independent debug programs generated on local CPU with
NumPy PCG64 seed 20260910 and the existing language sampler/render function. Every
program was checked with the independent replay validator before writing. The total
is 21,755 prediction tokens. These programs, checkpoints, probes and normalization
statistics are excluded from the main experiment.

The GPU receives this fixed fixture in the bundle; it does not generate programs.
Smoke uses 64-sequence effective batches and both dictionaries at k=4/16. The small
pool and 100-update run are debug checks, not changes to the production 50k pool or
5,000-update budget.
