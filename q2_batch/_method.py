import argparse
from biom import load_table
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import pickle
import pystan
import dask
from q2_batch._batch import _batch_func
import xarray as xr


def estimate(table : pd.DataFrame,
             replicate_column : qiime2.CategoricalMetadataColumn,
             batch_column : qiime2.CategoricalMetadataColumn,
             mc_samples : int) -> xr.DataArray:
    table = load_table(args.biom)
    metadata = pd.read_table(args.metadata)
    # TODO: need to speed this up with dask
    mu = np.zeros(table.shape[0], mc_samples)
    sigma = np.zeros(table.shape[0], mc_samples)
    for i, o in table.ids(axis='observation'):
        res = _batch_func(counts, replicates, batches,
                          model, mc_samples)
        mu[i], sigma[i] = res['mu'], res['sigma']
    samples = xr.DataSet(
        data_vars = {'batch_mean' : mu,
                     'batch_std' : sigma},
        coords = {
            'features' : table.ids(axis='observation'),
            'monte_carlo_samples' : np.arange(mc_samples)
        }
    )
    return samples
