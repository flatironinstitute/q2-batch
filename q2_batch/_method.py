import biom
import pandas as pd
from q2_batch._batch import PoissonLogNormalBatch
from dask.distributed import Client, LocalCluster
from gneiss.util import match
import xarray as xr
import qiime2


def _poisson_log_normal_estimate(table,
                                 replicates,
                                 batches,
                                 chunksize=50,
                                 **sampler_args):
    metadata = pd.DataFrame({'batch': batches, 'reps': replicates})
    table, metadata = match(table, metadata)
    pln = PoissonLogNormalBatch(
        table=table,
        replicate_column="reps",
        batch_column="batch",
        metadata=metadata,
        **sampler_args)
    pln.compile_model()
    pln.fit_model(chunksize=chunksize, convert_to_inference=True)
    samples = pln.to_inference_object()
    return samples


def estimate(counts: biom.Table,
             replicates: qiime2.CategoricalMetadataColumn,
             batches: qiime2.CategoricalMetadataColumn,
             cores: int = 1) -> xr.Dataset:
    replicates = replicates.to_series()
    batches = batches.to_series()
    # Build me a cluster!
    dask_args = {'n_workers': cores, 'threads_per_worker': 1}
    cluster = LocalCluster(**dask_args)
    cluster.scale(dask_args['n_workers'])
    client = Client(cluster)
    samples = _poisson_log_normal_estimate(
        counts, replicates, batches)
    return samples
