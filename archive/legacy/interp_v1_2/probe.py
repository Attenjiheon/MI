"""CPU float64 supervised probes with train/validation-only selection."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logsumexp


def balanced_accuracy(y,p,classes):
    if any(not np.any(y==c) for c in range(classes)): return float('nan')
    return float(np.mean([np.mean(p[y==c]==c) for c in range(classes)]))


def support(y,groups,classes):
    return all(np.sum(y==c)>=32 and len(np.unique(groups[y==c]))>=16 for c in range(classes))


def objective(theta,x,y,classes,lam):
    p=x.shape[1]
    if classes==2:
        w,b=theta[:p],theta[p]; logits=x@w+b
        ce=np.mean(np.logaddexp(0,logits)-y*logits)
        residual=(expit(logits)-y)/len(y)
        grad=np.r_[x.T@residual+lam*w,residual.sum()]
    else:
        coeff=theta.reshape(classes,p+1); w,b=coeff[:,:p],coeff[:,p]
        logits=x@w.T+b; logp=logits-logsumexp(logits,axis=1,keepdims=True)
        ce=-logp[np.arange(len(y)),y].mean()
        residual=np.exp(logp); residual[np.arange(len(y)),y]-=1; residual/=len(y)
        grad=np.c_[residual.T@x+lam*w,residual.sum(0)].ravel()
    return ce+lam/2*np.sum(w*w),grad


def probability(fit,x):
    x=(np.asarray(x,dtype=np.float64)[:,fit['columns']]-fit['mean'])/fit['std']
    p=x.shape[1]; theta=fit['coefficients']
    if fit['classes']==2: return expit(x@theta[:p]+theta[p])
    coeff=theta.reshape(fit['classes'],p+1); logits=x@coeff[:,:p].T+coeff[:,p]
    return np.exp(logits-logsumexp(logits,axis=1,keepdims=True))


def fit_probe(train,y,groups,val,vy,vgroups,classes=2,columns=None):
    y=np.asarray(y,dtype=int); vy=np.asarray(vy,dtype=int)
    if not support(y,np.asarray(groups),classes) or not support(vy,np.asarray(vgroups),classes):
        return dict(status='NA',reason='train/validation class support below 32 positions or 16 sequences')
    train=np.asarray(train,dtype=np.float64); val=np.asarray(val,dtype=np.float64)
    columns=np.arange(train.shape[1]) if columns is None else np.asarray(columns)
    columns=columns[train[:,columns].std(0)>=1e-8]
    if len(columns)==0: return dict(status='NA',reason='all features constant')
    mu=train[:,columns].mean(0); std=train[:,columns].std(0)
    x=(train[:,columns]-mu)/std; vx=(val[:,columns]-mu)/std
    candidates=[]; failures=[]
    for lam in (.01,.1,1.,10.):
        theta=np.zeros((x.shape[1]+1)*(1 if classes==2 else classes))
        for iterations in (2000,10000):
            result=minimize(objective,theta,args=(x,y,classes,lam),jac=True,method='L-BFGS-B',
                            options=dict(maxiter=iterations,maxls=50,ftol=1e-12,gtol=1e-7))
            if result.success: break
        if not result.success:
            failures.append(dict(lam=lam,message=str(result.message))); continue
        fit=dict(status='passed',classes=classes,columns=columns,mean=mu,std=std,
                 coefficients=result.x,lam=lam,iterations=int(result.nit))
        probs=probability(fit,val)
        val_ce=objective(result.x,vx,vy,classes,0)[0]
        for threshold in (np.arange(1,20)/20 if classes==2 else [None]):
            pred=(probs>=threshold).astype(int) if classes==2 else probs.argmax(1)
            ba=balanced_accuracy(vy,pred,classes)
            key=(-ba,val_ce,-lam,abs(threshold-.5) if threshold else 0,threshold or 0)
            candidates.append((key,{**fit,'threshold':threshold,'validation_balanced_accuracy':ba,'validation_ce':val_ce}))
    if not candidates: return dict(status='failed',reason='optimizer did not converge',failures=failures)
    best=min(candidates,key=lambda a:a[0])[1]; best['optimizer_failures']=failures
    return best


def anova_ranking(x,y):
    x=np.asarray(x,dtype=np.float64); y=np.asarray(y); classes=np.unique(y)
    between=sum(np.sum(y==c)*(x[y==c].mean(0)-x.mean(0))**2 for c in classes)
    within=sum(((x[y==c]-x[y==c].mean(0))**2).sum(0) for c in classes)
    with np.errstate(divide='ignore',invalid='ignore'):
        scores=between/(len(classes)-1)/(within/(len(y)-len(classes)))
    ids=np.flatnonzero(x.std(0)>=1e-8)
    return ids[np.lexsort((ids,-scores[ids]))]


def select_prefix(train,y,groups,val,vy,vgroups,classes=2,candidates=None):
    candidates=np.arange(train.shape[1]) if candidates is None else np.asarray(candidates)
    ranked=candidates[anova_ranking(train[:,candidates],y)]
    fits=[fit_probe(train,y,groups,val,vy,vgroups,classes,ranked[:m]) for m in range(1,min(4,len(ranked))+1)]
    good=[f for f in fits if f['status']=='passed']
    if not good: return dict(status='NA',reason='no fitting prefix',fits=fits)
    # §7.2: feature-count tie precedes the §7.1 within-count rules.
    best=min(good,key=lambda f:(-f['validation_balanced_accuracy'],len(f['columns']),f['validation_ce'],-f['lam'],abs((f['threshold'] or .5)-.5),f['threshold'] or 0))
    return dict(single=fits[0],up_to_four=best,ranking=ranked)
