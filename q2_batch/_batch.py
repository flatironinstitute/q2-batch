import argparse
from biom import load_table
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import pickle
import pystan
import os
import xarray as xr


def _batch_func(counts : np.array, replicates : np.array,
                batches : np.array,
                model : str, mc_samples : int) -> dict:

    replicate_encoder = LabelEncoder()
    replicate_encoder.fit(replicates)
    replicate_ids = replicate_encoder.transform(replicates)

    # identify reference replicates - these will be the
    # first sample for each replicate group
    ref_ids, lookup = np.zeros(len(counts)), {}
    for i, c in enumerate(replicate_ids):
        if c not in lookup:
            lookup[c] = i
    for i, c in enumerate(replicate_ids):
        ref_ids[i] = lookup[c]

    batch_encoder = LabelEncoder()
    batch_encoder.fit(batches)
    batch_ids = batch_encoder.transform(batches)

    # Actual stan modeling
    code = open(model, 'r').read()
    sm = pystan.StanModel(model_code=code)

    dat = {
        'N' : len(counts),
        'R' : max(replicate_ids),
        'B' : max(batch_ids),
        'depth' : np.log(counts.sum(axis=1).values),
        'y' : counts[:, microbe_idx].astype(np.int64),
        'batch_ids' : list(batch_ids),
        'ref_ids' : list(ref_ids)
    }
    fit = sm.sampling(data=dat, iter=mc_samples, chains=4)
    res =  fit.extract(permuted=True)
    return res



# Save  model
# TODO: should we save the model or just the posterior samples?
# pickle.dump({'dat': dat, 'res': res,
#              'microbes' : list(table.ids(axis='observation')),
#              'samples' : list(table.ids(axis='sample'))},
#              open(output_file, 'wb'))
