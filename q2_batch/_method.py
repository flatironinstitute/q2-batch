import argparse
from biom import load_table
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import pickle
import dask
from q2_batch._batch import _batch_func, merge_inferences
import arviz as az
import qiime2


# slow estimator
def estimate(counts : pd.DataFrame,
             replicates : qiime2.CategoricalMetadataColumn,
             batches : qiime2.CategoricalMetadataColumn,
             monte_carlo_samples : int = 100,
             cores : int = 1) -> az.InferenceData:
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

    inf_list = list(resdf[0])
    coords={'features' : counts.columns,
            'monte_carlo_samples' : np.arange(args.monte_carlo_samples)}

    samples = merge_inferences(inf_list, 'y_predict', 'log_lhood', coords)

    return samples
