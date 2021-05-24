from q2_batch._batch import _batch_func, _simulate, PoissonLogNormalBatch
import biom

table, metadata = _simulate(n=100, d=100, depth=100)
table = biom.Table(table.values.T,
                   table.columns, table.index)
pln = PoissonLogNormalBatch(
    table=table,
    replicate_column="reps",
    batch_column="batch",
    metadata=metadata,
    num_warmup=1000,
    mu_scale=1,
    reference_scale=5,
    chains=1,
    seed=42)
pln.compile_model()
# pln.fit_model(dask_args={'n_workers': 1, 'threads_per_worker': 1})
pln.fit_model(dask_args=dict(
    cores=4,
    processes=4,
    memory='16GB',
    walltime='10:00:00',
    interface='ib0',
    nanny=True,
    death_timeout='300s',
    local_directory='/scratch',
    shebang='#!/usr/bin/env bash',
    env_extra=["export TBB_CXX_TYPE=gcc"],
    queue='ccb'
)
)
pln.to_inference_object()
