import argparse
from dask_jobqueue import SLURMCluster
from dask.distributed import Client
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
        '--cores', help='Number of cores per process.', type=int, required=False, default=1)
    parser.add_argument(
        '--chunksize', help='Number of features per process.', type=int,
        required=False, default=10)
    parser.add_argument(
        '--chains', help='Number of MCMC chains.', type=int, required=False, default=5)
    parser.add_argument(
        '--processes', help='Number of processes per node.', type=int, required=False, default=1)
    parser.add_argument(
        '--nodes', help='Number of nodes.', type=int, required=False, default=1)
    parser.add_argument(
        '--memory', help='Memory allocation size.', type=str, required=False, default='16GB')
    parser.add_argument(
        '--walltime', help='Walltime.', type=str, required=False, default='01:00:00')
    parser.add_argument(
        '--interface', help='Interface for communication', type=str,
        required=False, default='eth0')
    parser.add_argument(
        '--job-extra', help='Comma delimited list of extra job arguments.',
        type=str, required=False, default='--constraint=rome')
    parser.add_argument(
        '--queue', help='Queue to submit job to.', type=str, required=True)
    parser.add_argument(
        '--local-directory', help='Scratch directory to deposit dask logs.',
        type=str, required=False, default='/scratch')
    parser.add_argument(
        '--output-tensor', help='Output tensor.', type=str, required=True)

    args = parser.parse_args()
    dask.config.set({'admin.tick.limit': '1h'})
    dask.config.set({"distributed.comm.timeouts.tcp": "300s"})
    cluster = SLURMCluster(cores=args.cores,
                           processes=args.processes,
                           memory=args.memory,
                           walltime=args.walltime,
                           interface=args.interface,
                           nanny=True,
                           death_timeout='600s',
                           local_directory=args.local_directory,
                           shebang='#!/usr/bin/env bash',
                           env_extra=["export TBB_CXX_TYPE=gcc"],
                           job_extra=args.job_extra.split(','),
                           queue=args.queue)
    print(args)
    print(cluster.job_script())
    cluster.scale(jobs=args.nodes)
    client = Client(cluster)
    print(client)
    client.wait_for_workers(args.nodes)
    time.sleep(60)

    print(client.get_versions(check=True))

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

    pfunc = lambda x: _batch_func(x, replicates, batches,
                                  depth, args.monte_carlo_samples,
                                  chains=args.chains,
                                  mu_scale=args.mu_scale,
                                  reference_loc=reference_loc,
                                  reference_scale=args.reference_scale)

    dcounts = da.from_array(counts.values.T, chunks=(args.chunksize, -1))

    res = []
    for d in range(dcounts.shape[0]):
        r = dask.delayed(pfunc)(dcounts[d])
        res.append(r)
    print('Res length', len(res))
    futures = dask.persist(*res)
    resdf = dask.compute(futures)
    print('Posteriors have been computed')
    inf_list = list(resdf[0])
    coords={'features' : counts.columns,
            'monte_carlo_samples' : np.arange(args.monte_carlo_samples)}
    samples = merge_inferences(inf_list, 'y_predict', 'log_lhood', coords)
    samples.to_netcdf(args.output_tensor)

