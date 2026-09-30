"""TC-only intervention primitives: encode u, replace m, preserve residual skip."""
import torch
from .dictionary import normalize, denormalize
from .patching import sparse_patch


def predict_output(dictionary, u, stats):
    return denormalize(dictionary(normalize(u, stats['u']))[0], stats['m'])


def feature_patch(dictionary, original_m, original_u, donor_u, features, stats):
    z_o=dictionary.encode(normalize(original_u,stats['u']))
    z_c=dictionary.encode(normalize(donor_u,stats['u']))
    return sparse_patch(original_m,z_o,z_c,dictionary,features,stats['m']['scale'])
