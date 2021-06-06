import qiime2
import argparse
from dask_jobqueue import SLURMCluster
from dask.distributed import Client
import dask
import dask.dataframe as dd
import dask.array as da
from biom import load_table
import pandas as pd
import numpy as np
import xarray as xr
from q2_batch._batch import _batch_func, merge_inferences
import time
import logging
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--biom-table', help='Biom table of counts.', required=True)
    parser.add_argument(
        '--metadata-file', help='Sample metadata file.', required=True)
    parser.add_argument(
        '--batches', help='Column specifying batches.', required=True)
    parser.add_argument(
        '--replicates', help='Column specifying replicates.', required=True)
    parser.add_argument(
        '--feature-id', help='Feature to analyze.', required=True)
    parser.add_argument(
        '--mu-scale', help='Scale of differentials.',
        type=float, required=False, default=10)
    parser.add_argument(
        '--reference-loc', help='Center of control log proportions.',
        type=float, required=False, default=None)
    parser.add_argument(
        '--reference-scale', help='Scale of control log proportions.',
        type=float, required=False, default=10)
    parser.add_argument(
        '--monte-carlo-samples', help='Number of monte carlo samples.',
        type=int, required=False, default=1000)
    parser.add_argument(
        '--output-tensor', help='Output tensor.', type=str, required=True)

    args = parser.parse_args()

    table = load_table(args.biom_table)
    counts = pd.DataFrame(np.array(table.matrix_data.todense()).T,
                          index=table.ids(),
                          columns=table.ids(axis='observation'))
    metadata = pd.read_table(args.metadata_file, index_col=0)
    replicates = metadata[args.replicates]
    batches = metadata[args.batches]
    # match everything up
    idx = list(set(counts.index) & set(replicates.index) & set(batches.index))
    counts, replicates, batches = [x.loc[idx] for x in
                                   (counts, replicates, batches)]
    replicates, batches = replicates.values, batches.values
    depth = counts.sum(axis=1)

    if args.reference_loc is None:
        # Dirichilet-like prior
        reference_loc = np.log(1 / counts.shape[1])
    else:
        reference_loc = args.reference_loc

    x = counts.loc[args.feature_id]
    samples = _batch_func(x, replicates, batches,
                          depth, args.monte_carlo_samples,
                          chains=args.chains,
                          mu_scale=args.mu_scale,
                          reference_loc=reference_loc,
                          reference_scale=args.reference_scale)

    samples.to_netcdf(args.output_tensor)
