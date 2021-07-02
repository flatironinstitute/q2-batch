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
import arviz as az
from q2_batch._batch import _batch_func, merge_inferences
import time
import logging
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--inference-files',
        metavar='N', type=str, nargs='+',
        help='List of inference files.', required=True)
    parser.add_argument(
        '--monte-carlo-samples', help='Number of monte carlo samples.',
        type=int, required=False, default=1000)
    parser.add_argument(
        '--only-sigma', help='Only focus on sigma.',
        default=False, dest='only_sigma',
        action='store_true')
    parser.add_argument(
        '--output-inference', help='Merged inference file.', type=str, required=True)
    args = parser.parse_args()
    print('Only store sigma', args.only_sigma)
    # A little redundant, but necessary for getting ids
    names = [x.split('.nc')[0].split('/')[-1] for x in args.inference_files]
    inf_list = [az.from_netcdf(x) for x in args.inference_files]
    if args.only_sigma:
        sigmas = [inf['posterior']['sigma'] for inf in inf_list]
        samples = xr.concat(sigmas, 'features')
        samples = samples.assign_coords({'features' : names})
        samples.to_netcdf(args.output_inference)
    else:
        coords={'features' : names,
                'monte_carlo_samples' : np.arange(args.monte_carlo_samples)}
        samples = merge_inferences(inf_list, 'y_predict', 'log_lhood', coords)
        samples.to_netcdf(args.output_inference)
