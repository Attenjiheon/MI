"""Plot the fixed-budget validation history without changing selection."""
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parents[1] / 'experiment_v1_2/results'
with (root / 'p3_learning_curve.csv').open() as stream:
    rows = [{k: float(v) for k, v in row.items()} for row in csv.DictReader(stream)]
x = [r['tokens'] / 1e6 for r in rows]
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), layout='constrained')
axes[0].plot(x, [r['current_ce'] for r in rows], label='Current checkpoint', color='#2563eb')
axes[0].plot(x, [r['best_ce'] for r in rows], label='Best so far', color='#0f172a', linestyle='--', linewidth=1)
axes[0].set(ylabel='General READ answer CE (nats)', title='Selection: minimum validation CE')
for key, label in [('general_accuracy', 'General'), ('other_variable', 'Other variable'), ('repeated_update', 'Repeated update'), ('first_read_after_set', 'First READ after SET')]:
    axes[1].plot(x, [100*r[key] for r in rows], label=label, linewidth=1.5)
axes[1].axhline(99, color='black', linestyle='--', linewidth=.8)
axes[1].axhline(95, color='grey', linestyle=':', linewidth=.8)
axes[1].set(ylabel='Full-vocabulary accuracy (%)', ylim=(15, 103), title='Gate: general 99%; each diagnostic 95%')
for ax in axes:
    ax.set(xlabel='Consumed prediction tokens (millions)', xlim=(0, 16.3))
    ax.grid(alpha=.15)
    ax.legend(fontsize=8)
    for milestone in (1, 3, 8, 16):
        ax.axvline(milestone, color='grey', linewidth=.5, alpha=.3)
fig.suptitle('v1.2 · fresh seed 0 · 4 blocks · 16M budget · behavioral gate FAILED')
out = root / 'figures'; out.mkdir(exist_ok=True)
fig.savefig(out / 'p3_learning_curve.png', dpi=180)
fig.savefig(out / 'p3_learning_curve.svg')
svg = out / 'p3_learning_curve.svg'
svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines()) + '\n')
