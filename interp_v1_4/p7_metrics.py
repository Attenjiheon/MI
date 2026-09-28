"""P7 numeric summaries and validation-only random matching."""
import numpy as np
from .p5 import derived


def fidelity(x, prediction, z, train_mean):
    x, prediction = np.asarray(x, np.float64), np.asarray(prediction, np.float64)
    error = x-prediction
    sse = float(np.square(error).sum())
    denominator = float(np.square(x-train_mean).sum())
    variance = float(np.square(x-x.mean(0)).sum())
    evden = float(x.var(0).sum())
    l0 = (z > 0).sum(1)
    rates = (z > 0).mean(0)
    return dict(positions=len(x),mse=sse/x.size,nmse=sse/denominator if denominator else None,
        r2=1-sse/variance if variance else None,
        ev=1-float(error.var(0).sum())/evden if evden else None,
        l0=dict(mean=float(l0.mean()),median=float(np.median(l0)),quantiles=np.percentile(l0,[0,5,25,75,95,100]).tolist()),
        activation_rates=rates.tolist(),inactive_count=int((rates==0).sum()),inactive_fraction=float((rates==0).mean()))


def bins(hnorm, znorm):
    """Zero patches have their own bin; duplicate percentile edges merge."""
    hnorm,znorm=np.asarray(hnorm),np.asarray(znorm)
    positive=hnorm>0
    return dict(h_edges=np.unique(np.percentile(hnorm[positive],[20,40,60,80])).tolist() if positive.any() else [],
                z_edges=np.unique(np.percentile(znorm[positive],[20,40,60,80])).tolist() if positive.any() else [],
                fitted_pairs=int(len(hnorm)),zero_count=int((~positive).sum()))


def bin_id(h,z,rule):
    if h==0:return (-1,-1)
    return (int(np.searchsorted(rule['h_edges'],h,side='right')),
            int(np.searchsorted(rule['z_edges'],z,side='right')))


def random_sets(width, features, seed, count=200):
    pool=np.setdiff1d(np.arange(width),features)
    if len(pool)<len(features):raise ValueError('Insufficient random candidates')
    rng=np.random.Generator(np.random.PCG64(seed));seen=set();out=[]
    for _ in range(count):
        item=tuple(sorted(rng.choice(pool,len(features),replace=False).tolist()))
        if item not in seen:out.append(list(item));seen.add(item)
    return out


def matched_indices(selected_h, selected_z, h, z, rule, limit=20):
    target=bin_id(selected_h,selected_z,rule)
    return [i for i,(a,b) in enumerate(zip(h,z)) if bin_id(a,b,rule)==target][:limit]


def paired_summary(rows, fields, seed, draws=1000):
    """Average directions/candidates within origin before a cluster bootstrap."""
    clusters={}
    for r in rows:
        clusters.setdefault(r['origin'],[]).append(r)
    result=dict(rows=len(rows),origins=len(clusters),metrics={})
    if not rows:return result
    ids=sorted(clusters);rng=np.random.Generator(np.random.PCG64(seed))
    # Shared indices for every metric and layer using the same seed and origin order.
    # Fixed condition quotas: resample within memory/composition, then macro average.
    conditions={o:clusters[o][0].get('condition','all') for o in ids}
    strata=[np.asarray([i for i,o in enumerate(ids) if conditions[o]==c]) for c in sorted(set(conditions.values()))]
    indices=[s[rng.integers(len(s),size=(draws,len(s)))] for s in strata]
    for field in fields:
        vals=np.asarray([np.mean([r[field] for r in clusters[o]]) for o in ids],float)
        estimates=np.mean([vals[i].mean(1) for i in indices],axis=0)
        mean=float(np.mean([vals[s].mean() for s in strata]))
        result['metrics'][field]=dict(mean=mean,ci95=np.percentile(estimates,[2.5,97.5]).tolist(),requested=draws,valid=draws,seed=seed)
    return result
