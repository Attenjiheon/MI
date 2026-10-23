"""Single-position interventions recompute downstream without a KV cache."""
from contextlib import contextmanager
import torch


@contextmanager
def capture(model):
    previous=model.capture
    model.capture={}
    try: yield model.capture
    finally: model.capture=previous


@contextmanager
def patch(model, hook, positions, values):
    previous=model.interventions
    def replace(x):
        y=x.clone()
        for row,pos in enumerate(positions): y[row,pos]=values[row]
        return y
    model.interventions={**previous,hook:replace}
    try: yield
    finally: model.interventions=previous


def sparse_patch(original,zo,zc,dictionary,features,output_scale):
    return original+output_scale*((zc[...,features]-zo[...,features]) @ dictionary.decoder[:,features].T)


def coordinate_patch(original,donor,features):
    result=original.clone(); result[...,features]=donor[...,features]; return result


def direction_patch(original,donor,q):
    return original+(donor-original)@q@torch.linalg.pinv(q.T@q)@q.T
