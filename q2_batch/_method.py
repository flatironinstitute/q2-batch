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

    replicates = replicates.to_series().values
    batches = batches.to_series().values
    # TODO: need to speed this up with either joblib or something
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
                      replicate_column : qiime2.CategoricalMetadataColumn,
                      batch_column : qiime2.CategoricalMetadataColumn,
                      monte_carlo_samples : int,
                      cores=16,
                      memory='16 GB',
                      processes=4):
    from dask_jobqueue import SLURMCluster
