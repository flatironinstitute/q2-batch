import qiime2
import argparse
from biom import load_table
import pandas as pd
import numpy as np
import xarray as xr
from q2_batch._batch import _batch_func, merge_inferences
import time
import logging
import subprocess, os
import tempfile
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
        '--local-directory',
        help=('Scratch directory to deposit logs '
              'and intermediate files.'),
        type=str, required=False, default='/scratch')
    parser.add_argument(
        '--output-inference', help='Output inference tensor.',
        type=str, required=True)

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
    # Launch disbatch
    ## First create a temporary file with all of the tasks
    tf = tempfile.NamedTemporaryFile()
    task_name = tf.name
    with open(task_name, 'w') as f:
        for feature_id in counts.columns:
            cmd_ = (
                'batch_estimate_single.py '
                f'--biom-table {args.biom_table} '
                f'--metadata-file {args.metadata.file} '
                f'--batches {args.batches} '
                f'--replicates {args.replicates} '
                f'--feature-id {feature_id} '
                f'--mu-scale {args.mu_scale} '
                f'--reference-loc {args.reference_loc} '
                f'--reference-scale {args.reference_scale} '
                f'--monte-carlo-samples {args.monte_carlo_samples} '
                f'--output-tensor {args.local_directory}/{feature_id.nc}'
                # slurm logs
                f' &> {args.local_directory}/{feature_id}.log\n'
            )
            f.write(cmd_)

    ## Run disBatch with the SLURM environmental parameters
    cmd = f'disBatch {task_name}'
    slurm_env = os.environ.copy()
    proc = subprocess.Popen(cmd, env=slurm_env)
    proc.wait()

    ## Aggregate results
    inference_files = [f'{args.local_directory}/{feature_id.nc}'
                       for feature_id in counts.columns]
    inf_list = [az.from_netcdf(x) for x in inference_files]
    coords={'features' : counts.columns,
            'monte_carlo_samples' : np.arange(args.monte_carlo_samples)}
    samples = merge_inferences(inf_list, 'y_predict', 'log_lhood', coords)
    samples.to_netcdf(args.output_inference)
