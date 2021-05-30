import argparse
from dask.distributed import Client, LocalCluster
from biom import load_table
import pandas as pd
from q2_batch._method import _poisson_log_normal_estimate
from gneiss.util import match
import time
import logging
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)


if __name__ == "__main__":
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
        '--monte-carlo-samples', help='Number of monte carlo samples.',
        type=int, required=False, default=1000)
    parser.add_argument(
        '--chains', help='Number of monte carlo chains.', type=int,
        required=False, default=1)
    parser.add_argument(
        '--cores', help='Number of cores per process.', type=int,
        required=False, default=1)
    parser.add_argument(
        '--chunksize', help='Number species to analyze within a process.',
        type=int, required=False, default=100)
    parser.add_argument(
        '--output-tensor', help='Output tensor.', type=str, required=True)
    args = parser.parse_args()
    print(args)
    # See issue
    # https://github.com/dask/distributed/issues/2520#issuecomment-470817810
    dask_args = {'n_workers': args.cores, 'threads_per_worker': 1}
    cluster = LocalCluster(**dask_args)
    cluster.scale(dask_args['n_workers'])
    client = Client(cluster)
    print(client)
    table = load_table(args.biom_table)
    metadata = pd.read_table(args.metadata_file, index_col=0)
    table, metadata = match(table, metadata)
    replicates = metadata[args.replicates]
    batches = metadata[args.batches]

    samples = _poisson_log_normal_estimate(
        table,
        replicates,
        batches,
        chunksize=args.chunksize,
        mu_scale=1,
        sigma_scale=1,
        disp_scale=1,
        reference_loc=-5,
        reference_scale=3,
        num_iter=args.monte_carlo_samples,
        chains=args.chains
    )

    samples.to_netcdf(args.output_tensor)
