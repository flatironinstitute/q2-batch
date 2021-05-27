import argparse
from dask_jobqueue import SLURMCluster
from dask.distributed import Client
from biom import load_table
import pandas as pd
from q2_batch._method import _poisson_log_normal_estimate
import time
import logging
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

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
    '--cores', help='Number of cores per process.', type=int,
    required=False, default=1)
parser.add_argument(
    '--processes', help='Number of processes per node.', type=int,
    required=False, default=1)
parser.add_argument(
    '--nodes', help='Number of nodes.', type=int,
    required=False, default=1)
parser.add_argument(
    '--memory', help='Memory allocation size.', type=str,
    required=False, default='16GB')
parser.add_argument(
    '--walltime', help='Walltime.', type=str, required=False,
    default='01:00:00')
parser.add_argument(
    '--interface', help='Interface for communication', type=str,
    required=False, default='eth0')
parser.add_argument(
    '--queue', help='Queue to submit job to.', type=str, required=True)
parser.add_argument(
    '--output-tensor', help='Output tensor.', type=str, required=True)


args = parser.parse_args()

print(args)
cluster = SLURMCluster(cores=args.cores,
                       processes=args.processes,
                       memory=args.memory,
                       walltime=args.walltime,
                       interface=args.interface,
                       nanny=True,
                       death_timeout='300s',
                       local_directory=args.local_directory,
                       shebang='#!/usr/bin/env bash',
                       env_extra=["export TBB_CXX_TYPE=gcc"],
                       queue=args.queue)
print(cluster.job_script())
cluster.scale(jobs=args.nodes)
client = Client(cluster)
print(client)
client.wait_for_workers(args.nodes)
time.sleep(60)
print(cluster.scheduler.workers)

table = load_table(args.biom_table)
metadata = pd.read_table(args.metadata_file, index_col=0)
replicates = metadata[args.replicates]
batches = metadata[args.batches]

samples = _poisson_log_normal_estimate(
    table,
    replicates,
    batches,
    cores=args.cores,
    num_iter=args.monte_carlo_samples,
    mu_scale=1,
    sigma_scale=1,
    disp_scale=1,
    reference_loc=-5,
    reference_scale=3,
    num_iter=args.monte_carlo_samples,
    chains=args.chains
)

samples.to_netcdf(args.output_tensor)
