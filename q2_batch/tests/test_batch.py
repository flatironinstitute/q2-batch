import unittest
from q2_batch._batch import _batch_func, _simulate, PoissonLogNormalBatch
from dask.distributed import Client, LocalCluster
import biom

class TestBatch(unittest.TestCase):

    def setUp(self):
        self.table, self.metadata = _simulate(n=100, d=10, depth=50)

    def test_batch(self):
        res = _batch_func(self.table.values[:, 0],
                          replicates=self.metadata['reps'].values,
                          batches=self.metadata['batch'].values,
                          depth=self.table.sum(axis=1),
                          mc_samples=2000)
        self.assertEqual(res.shape, (2000 * 4, 3))


class TestPoissonLogNormalBatch(TestBatch):
    def test_batch(self):
        table = biom.Table(self.table.values.T,
                           self.table.columns, self.table.index)
        pln = PoissonLogNormalBatch(
            table=table,
            replicate_column="reps",
            batch_column="batch",
            metadata=self.metadata,
            num_warmup=1000,
            mu_scale=1,
            reference_scale=5,
            chains=1,
            seed=42)
        pln.compile_model()
        dask_args={'n_workers': 1, 'threads_per_worker': 1}
        cluster = LocalCluster(**dask_args)
        cluster.scale(dask_args['n_workers'])
        client = Client(cluster)
        pln.fit_model()
        inf = pln.to_inference_object()
        self.assertEqual(inf['posterior']['mu'].shape, (10, 1, 1000))


if __name__ == '__main__':
    unittest.main()
