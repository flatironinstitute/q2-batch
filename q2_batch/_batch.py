import argparse
from biom import load_table
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import pickle
import os
from skbio.stats.composition import ilr_inv
import matplotlib.pyplot as plt
import pickle
from cmdstanpy import CmdStanModel
import tempfile
import json


def _batch_func(counts : np.array, replicates : np.array,
                batches : np.array, depth : int,
                mc_samples : int) -> dict:
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
    code = os.path.join(os.path.dirname(__file__),
                        'assets/batch_pln_single.stan')
    batch_ids = batch_ids.astype(np.int64) + 1
    ref_ids = ref_ids.astype(np.int64) + 1
    sm = CmdStanModel(stan_file=code)
    dat = {
        'N' : len(counts),
        'R' : int(max(replicate_ids) + 1),
        'B' : int(max(batch_ids) + 1),
        'depth' : list(map(int, np.log(depth))),
        'y' : list(map(int, counts.astype(np.int64))),
        'ref_ids' : list(map(int, ref_ids )),
        'batch_ids' : list(map(int, batch_ids))
    }
    with tempfile.TemporaryDirectory() as temp_dir_name:
        data_path = os.path.join(temp_dir_name, 'data.json')
        with open(data_path, 'w') as f:
            json.dump(dat, f)
        fit = sm.sample(data=data_path, iter_sampling=mc_samples, chains=4)
        res =  fit.extract(permuted=True)
        return res

    fit = sm.sampling(data=dat, iter=mc_samples, chains=4)
    res =  fit.extract(permuted=True)
    return res


def _simulate(n=100, d=10, depth=50):
    """ Simulate batch effects

    Parameters
    ----------
    n : int
       Number of samples
    d : int
       Number of microbes
    depth : int
       Sequencing depth

    Returns
    -------
    table : pd.DataFrame
        Simulated counts
    md : pd.DataFrame
        Simulated metadata
    """
    noise = 0.1
    batch = np.random.randn(d - 1) * 10
    diff = np.random.randn(d - 1)
    ref = np.random.randn(d - 1)
    table = np.zeros((n, d))
    batch_md = np.zeros(n)
    diff_md = np.zeros(n)
    rep_md = np.zeros(n)
    for i in range(n // 2):
        delta = np.random.randn(d - 1) * noise
        N = np.random.poisson(depth)
        if i % 2 == 0:
            r1 = ref + delta
            r2 = r1 + batch
            p1 = np.random.dirichlet(ilr_inv(r1))
            p2 = np.random.dirichlet(ilr_inv(r2))
            batch_md[i] = 0
            batch_md[(n // 2) + i] = 1
            diff_md[i] = 0
            diff_md[(n // 2) + i] = 0
            rep_md[i] = i
            rep_md[(n // 2) + i] = i
            table[i] = np.random.multinomial(N, p1)
            table[(n // 2) + i] = np.random.multinomial(N, p2)
        elif i % 2 == 1:
            r1 = ref + delta + diff
            r2 = r1 + batch + diff
            p1 = np.random.dirichlet(ilr_inv(r1))
            p2 = np.random.dirichlet(ilr_inv(r2))
            batch_md[i] = 0
            batch_md[(n // 2) + i] = 1
            diff_md[i] = 1
            diff_md[(n // 2) + i] = 1
            rep_md[i] = i
            rep_md[(n // 2) + i] = i
            table[i] = np.random.multinomial(N, p1)
            table[(n // 2) + i] = np.random.multinomial(N, p2)
    oids = [f'o{x}' for x in range(d)]
    sids = [f's{x}' for x in range(n)]
    table = pd.DataFrame(table, index=sids, columns=oids)
    md = pd.DataFrame({'batch': batch_md, 'diff': diff_md,
                       'reps': rep_md},
                      index=sids)
    return table, md
