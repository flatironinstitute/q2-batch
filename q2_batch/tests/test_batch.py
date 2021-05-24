import unittest
from q2_batch._batch import _batch_func, _simulate, PoissonLogNormalBatch


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
        pln = PoissonLogNormalBatch(
            table=biom_table,
            replicate_column="reps",
            batch_column="batch",
            metadata=metadata,
            num_warmup=1000,
            mu_scale=1,
            reference_scale=5,
            chains=1,
            seed=42)
        pln.compile_model()
        pln.fit_model(jobs=4)


if __name__ == '__main__':
    unittest.main()
