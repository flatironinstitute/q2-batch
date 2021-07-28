import biom
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os
from skbio.stats.composition import ilr_inv
import matplotlib.pyplot as plt
import pickle
from cmdstanpy import CmdStanModel
import tempfile
import dask
import xarray as xr
import arviz as az
import json


def merge_inferences(inf_list, log_likelihood, posterior_predictive,
                     coords, concatenation_name='features'):
    group_list = []
    group_list.append(dask.persist(*[x.posterior for x in inf_list]))
    group_list.append(dask.persist(*[x.sample_stats for x in inf_list]))
    if log_likelihood is not None:
        group_list.append(dask.persist(*[x.log_likelihood for x in inf_list]))
    if posterior_predictive is not None:
        group_list.append(
            dask.persist(*[x.posterior_predictive for x in inf_list])
        )

    group_list = dask.compute(*group_list)
    po_ds = xr.concat(group_list[0], concatenation_name)
    ss_ds = xr.concat(group_list[1], concatenation_name)
    group_dict = {"posterior": po_ds, "sample_stats": ss_ds}

    if log_likelihood is not None:
        ll_ds = xr.concat(group_list[2], concatenation_name)
        group_dict["log_likelihood"] = ll_ds
    if posterior_predictive is not None:
        pp_ds = xr.concat(group_list[3], concatenation_name)
        group_dict["posterior_predictive"] = pp_ds

    all_group_inferences = []
    for group in group_dict:
        # Set concatenation dim coords
        group_ds = group_dict[group].assign_coords(
            {concatenation_name: coords[concatenation_name]}
        )

        group_inf = az.InferenceData(**{group: group_ds})  # hacky
        all_group_inferences.append(group_inf)

    return az.concat(*all_group_inferences)


def _batch_func(counts : np.array, replicates : np.array,
                batches : np.array, depth : int,
                mc_samples : int=1000, chains : int=4,
                sigma_scale : float=1,
                reference_loc : float=-5,
                reference_scale : float=5) -> dict:

    replicate_encoder = LabelEncoder()
    replicate_encoder.fit(replicates)
    replicate_ids = replicate_encoder.transform(replicates)
    # identify reference replicates - these will be the
    # first sample for each replicate group
    ref_ids, lookup = np.zeros(len(replicate_ids)), {}
    for i, c in enumerate(replicate_ids):
        if c not in lookup:
            lookup[c] = i
    for i, c in enumerate(replicate_ids):
        ref_ids[i] = lookup[c]
    batch_encoder = LabelEncoder()
    batch_encoder.fit(batches)
    batch_ids = batch_encoder.transform(batches)

    batch_ids = batch_ids.astype(np.int64) + 1
    ref_ids = ref_ids.astype(np.int64) + 1
    code = os.path.join(os.path.dirname(__file__),
                        'assets/batch_pln_single.stan')
    sm = CmdStanModel(stan_file=code)
    dat = {
        'N' : counts.shape[0],
        'R' : int(max(ref_ids) + 1),
        'B' : int(max(batch_ids) + 1),
        'depth' : list(np.log(depth)),
        'y' : list(map(int, counts.astype(np.int64))),
        'ref_ids' : list(map(int, ref_ids)),
        'batch_ids' : list(map(int, batch_ids)),
        # 'mu_scale' : mu_scale,
        'sigma_scale' : sigma_scale,
        'disp_scale' : 1,
        'reference_loc' : reference_loc,
        'reference_scale' : reference_scale
    }
    with tempfile.TemporaryDirectory() as temp_dir_name:
        data_path = os.path.join(temp_dir_name, 'data.json')
        with open(data_path, 'w') as f:
            json.dump(dat, f)
        # Obtain an initial guess with MLE
        # guess = sm.optimize(data=data_path, inits=0)
        # see https://mattocci27.github.io/assets/poilog.html
        # for recommended parameters for poisson log normal
        fit = sm.sample(data=data_path, iter_sampling=mc_samples,
                        # inits=guess.optimized_params_dict,
                        chains=chains, iter_warmup=1000,
                        adapt_delta = 0.9, max_treedepth = 20)
        fit.diagnose()
        inf = az.from_cmdstanpy(fit,
                                posterior_predictive='y_predict',
                                log_likelihood='log_lhood',
        )
        return inf



def _simulate(n=100, d=10, depth=50):
    """ Simulate batch effects from Multinomial distribution

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
    md = pd.DataFrame({'batch': batch_md.astype(np.int64).astype(np.str),
                       'diff': diff_md.astype(np.int64).astype(np.str),
                       'reps': rep_md.astype(np.int64).astype(np.str)},
                      index=sids)
    md.index.name = 'sampleid'
    return table, md
