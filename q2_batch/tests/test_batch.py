import unittest
from q2_batch._batch import _simulate, PoissonLogNormalBatch
from dask.distributed import Client, LocalCluster
import biom


class TestPoissonLogNormalBatch(unittest.TestCase):

    def setUp(self):
        self.table, self.metadata = _simulate(n=100, d=10, depth=50)

    def test_batch(self):
        dask_args={'n_workers': 1, 'threads_per_worker': 1}
        cluster = LocalCluster(**dask_args)
        cluster.scale(dask_args['n_workers'])
        client = Client(cluster)
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
        pln.fit_model()
        inf = pln.to_inference_object()
        self.assertEqual(inf['posterior']['mu'].shape, (10, 1, 1000))


if __name__ == '__main__':
    unittest.main()
