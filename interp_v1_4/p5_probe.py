"""P5 float64 probes, explicit selection traces and cluster-aware reporting."""
from __future__ import annotations
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logsumexp
from scipy.stats import rankdata
from .probe import objective


def support(y, groups, classes):
    return [dict(class_id=c, positions=int(np.sum(y == c)),
                 sequences=int(len(np.unique(groups[y == c])))) for c in range(classes)]


def probabilities(fit, x):
    x = (np.asarray(x, dtype=np.float64)[:, fit['columns']] - fit['mean']) / fit['std']
    theta = np.asarray(fit['coefficients']); p = x.shape[1]
    if fit['classes'] == 2:
        return expit(x @ theta[:p] + theta[p])
    coef = theta.reshape(fit['classes'], p + 1)
    logits = x @ coef[:, :p].T + coef[:, p]
    return np.exp(logits - logsumexp(logits, axis=1, keepdims=True))


def from_confusion(cm):
    counts = cm.sum(1); observed = counts > 0
    recall = np.divide(cm.diagonal(), counts, out=np.zeros(len(cm)), where=observed)
    denom = cm.sum(0) + counts
    f1 = np.divide(2 * cm.diagonal(), denom, out=np.zeros(len(cm)), where=denom > 0)
    return dict(balanced_accuracy=float(recall.mean()) if observed.all() else None,
                macro_f1=float(f1.mean()) if observed.all() else None,
                observed_class_balanced_accuracy=float(recall[observed].mean()) if observed.any() else None)


def metrics(y, prob, classes, threshold, groups=None, bootstrap_seed=None, draws=1000):
    y = np.asarray(y, dtype=int)
    pred = (prob >= threshold).astype(int) if classes == 2 else prob.argmax(1)
    cm = np.bincount(y * classes + pred, minlength=classes ** 2).reshape(classes, classes)
    result = dict(**from_confusion(cm), confusion_matrix=cm.tolist(), positions=len(y),
                  class_counts=cm.sum(1).tolist(), binary_auroc=None)
    if classes == 2 and all(cm.sum(1) > 0):
        rank = rankdata(prob); pos = y == 1; n1 = int(pos.sum()); n0 = len(y) - n1
        result['binary_auroc'] = float((rank[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
    if groups is not None and bootstrap_seed is not None and len(y):
        _, inverse = np.unique(groups, return_inverse=True); n = int(inverse.max() + 1)
        group_cm = np.zeros((n, classes * classes), dtype=np.int64)
        np.add.at(group_cm, (inverse, y * classes + pred), 1)
        rng = np.random.Generator(np.random.PCG64(bootstrap_seed))
        values = {k: [] for k in ('balanced_accuracy', 'macro_f1', 'binary_auroc')}
        # Weighted tied-score AUROC: sequence multiplicities preserve every within-cluster row.
        if classes == 2:
            order = np.argsort(prob, kind='stable'); sy = y[order]; sg = inverse[order]
            starts = np.r_[0, np.flatnonzero(np.diff(prob[order])) + 1]
        for _ in range(draws):
            weights = np.bincount(rng.integers(n, size=n), minlength=n)
            current = from_confusion((weights @ group_cm).reshape(classes, classes))
            for key in ('balanced_accuracy', 'macro_f1'):
                if current[key] is not None: values[key].append(current[key])
            if classes == 2:
                w = weights[sg]; positive = np.add.reduceat(w * sy, starts)
                negative = np.add.reduceat(w * (1 - sy), starts)
                if positive.sum() and negative.sum():
                    auc = np.sum(positive * (np.cumsum(negative) - negative / 2)) / (positive.sum() * negative.sum())
                    values['binary_auroc'].append(float(auc))
        result['cluster_bootstrap'] = dict(seed=bootstrap_seed, requested=draws, clusters=n,
            metrics={k: dict(valid=len(v), ci95=np.percentile(v, [2.5, 97.5]).tolist() if v else None)
                     for k, v in values.items()})
    return result


def ranking(x, y, candidates):
    candidates = np.asarray(candidates, dtype=int)
    x = np.asarray(x[:, candidates], dtype=np.float64)
    keep = x.std(0) >= 1e-8; candidates = candidates[keep]; x = x[:, keep]
    classes = np.unique(y); mu = x.mean(0)
    between = sum(np.sum(y == c) * (x[y == c].mean(0) - mu) ** 2 for c in classes)
    within = sum(((x[y == c] - x[y == c].mean(0)) ** 2).sum(0) for c in classes)
    scores = np.divide(between / (len(classes) - 1), within / (len(y) - len(classes)),
                       out=np.full(len(candidates), np.inf), where=within > 0)
    return candidates[np.lexsort((candidates, -scores))]


def fit(train, y, groups, val, vy, vgroups, classes=2, candidates=None, prefixes=False):
    y, vy = np.asarray(y, int), np.asarray(vy, int)
    checks = dict(train=support(y, groups, classes), val=support(vy, vgroups, classes))
    if any(s['positions'] < 32 or s['sequences'] < 16 for rows in checks.values() for s in rows):
        return dict(status='NA', reason='class support below 32 positions or 16 sequences', support=checks)
    train, val = np.asarray(train, np.float64), np.asarray(val, np.float64)
    pool = np.arange(train.shape[1]) if candidates is None else np.asarray(candidates)
    ranked = ranking(train, y, pool) if prefixes else pool
    sets = [ranked[:m] for m in range(1, min(4, len(ranked)) + 1)] if prefixes else [pool]
    trace, fitted = [], []
    for columns in sets:
        columns = columns[train[:, columns].std(0) >= 1e-8]
        if not len(columns): continue
        mu, std = train[:, columns].mean(0), train[:, columns].std(0)
        x = (train[:, columns] - mu) / std; vx = (val[:, columns] - mu) / std
        for lam in (.01, .1, 1., 10.):
            theta = np.zeros((len(columns) + 1) * (1 if classes == 2 else classes))
            attempts = []
            for limit in (2000, 10000):
                opt = minimize(objective, theta, args=(x, y, classes, lam), jac=True,
                    method='L-BFGS-B', options=dict(maxiter=limit, maxls=50, ftol=1e-12, gtol=1e-7))
                attempts.append(dict(maxiter=limit, iterations=int(opt.nit), success=bool(opt.success), message=str(opt.message)))
                if opt.success: break
            record = dict(columns=columns.tolist(), lam=lam, attempts=attempts)
            trace.append(record)
            if not opt.success: continue
            model = dict(classes=classes, columns=columns.tolist(), mean=mu.tolist(), std=std.tolist(),
                coefficients=opt.x.tolist(), lam=lam, removed_constant_columns=int(len(pool)-np.sum(train[:, pool].std(0)>=1e-8)))
            prob = probabilities(model, val); ce = float(objective(opt.x, vx, vy, classes, 0)[0])
            choices = []
            for t in (np.arange(1, 20)/20 if classes == 2 else [None]):
                ba = metrics(vy, prob, classes, t)['balanced_accuracy']
                key = (-ba, ce, -lam, abs(t-.5) if t is not None else 0, t or 0)
                choices.append((key, dict(threshold=None if t is None else float(t), validation_balanced_accuracy=ba, validation_ce=ce)))
            _, selected = min(choices, key=lambda z: z[0]); model.update(selected)
            record.update(selected); fitted.append(model)
    if not fitted:
        return dict(status='failed', reason='no converged nonconstant candidate', support=checks, trace=trace)
    # Do not silently drop a nonconverged grid cell from a predeclared selection.
    if any(not r['attempts'][-1]['success'] for r in trace):
        return dict(status='failed', reason='optimizer grid incomplete', support=checks, trace=trace)
    def key(f):
        return (-f['validation_balanced_accuracy'], len(f['columns']) if prefixes else 0,
                f['validation_ce'], -f['lam'], abs((f['threshold'] or .5)-.5), f['threshold'] or 0)
    selected = {'up_to_four': min(fitted, key=key),
                'single': min([f for f in fitted if len(f['columns']) == 1], key=key)} if prefixes else {'full': min(fitted, key=key)}
    return dict(status='passed', support=checks, ranking=ranked.tolist() if prefixes else None,
                candidate_count=len(pool), trace=trace, selected=selected)
