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
import qiime2


# slow estimator
def estimate(counts : pd.DataFrame,
             replicates : qiime2.CategoricalMetadataColumn,
             batches : qiime2.CategoricalMetadataColumn,
             monte_carlo_samples : int = 100,
             cores : int = 1) -> xr.Dataset:
    # match everything up
    replicates = replicates.to_series()
    batches = batches.to_series()
    idx = list(set(counts.index) & set(replicates.index) & set(batches.index))
    counts, replicates, batches = [x.loc[idx] for x in
                                   (counts, replicates, batches)]
    replicates, batches = replicates.values, batches.values
    depth = counts.sum(axis=1)
    pfunc = lambda x: _batch_func(np.array(x.values), replicates, batches,
                                  depth, monte_carlo_samples)
    if cores > 1:
        try:
            import dask.dataframe as dd
            dcounts = dd.from_pandas(counts.T, npartitions=cores)
            res = dcounts.apply(pfunc, axis=1)
            resdf = res.compute(scheduler='processes')
            data_df = list(resdf.values)
        except:
            data_df = list(counts.T.apply(pfunc, axis=1).values)
    else:
        data_df = list(counts.T.apply(pfunc, axis=1).values)
    samples = xr.concat([df.to_xarray() for df in data_df], dim="features")
    samples = samples.assign_coords(coords={
            'features' : counts.columns,
            'monte_carlo_samples' : np.arange(monte_carlo_samples)
    })
    return samples


# Parallel estimation of batch effects
def parallel_estimate(counts : pd.DataFrame,
                      batches : qiime2.CategoricalMetadataColumn,
                      replicates : qiime2.CategoricalMetadataColumn,
                      monte_carlo_samples : int,
                      cores=4,
                      processes=4,
                      memory='16 GB',
                      walltime='01:00:00'):
    from dask_jobqueue import SLURMCluster
    from distributed import Client
    import dask.dataframe as dd

    cluster = SLURMCluster(cores=cores,
                           processes=processes,
                           memory="16GB",
                           project="batch",
                           walltime="01:00:00",
                           env_extra="export TBB_CXX_TYPE=gcc",
                           queue="normal")
    cluster.scale(jobs=processes)
    client = Client(cluster)
    # match everything up
    replicates = replicates.to_series()
    batches = batches.to_series()
    idx = list(set(counts.index) & set(replicates.index) & set(batches.index))
    counts, replicates, batches = [x.loc[idx] for x in
                                   (counts, replicates, batches)]
    replicates, batches = replicates.values, batches.values
    depth = counts.sum(axis=1)
    pfunc = lambda x: _batch_func(np.array(x.values), replicates, batches,
                                  depth, monte_carlo_samples)

    dcounts = dd.from_pandas(counts.T, npartitions=cores)
    res = dcounts.apply(pfunc, axis=1)
    resdf = res.compute(scheduler='processes')
    data_df = list(resdf.values)

    samples = xr.concat([df.to_xarray() for df in data_df], dim="features")
    samples = samples.assign_coords(coords={
            'features' : counts.columns,
            'monte_carlo_samples' : np.arange(monte_carlo_samples)
    })
    return samples
