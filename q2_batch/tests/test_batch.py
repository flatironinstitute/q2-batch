import unittest
from q2_batch._batch import _simulate, _batch_func
import biom


class TestPoissonLogNormalBatch(unittest.TestCase):

    def setUp(self):
        self.table, self.metadata = _simulate(n=100, d=10, depth=50)

    def test_batch(self):
        res = _batch_func(self.table.values[:, 0],
                          replicates=self.metadata['reps'].values,
                          batches=self.metadata['batch'].values,
                          depth=self.table.sum(axis=1),
                          mc_samples=100)
        self.assertEqual(res.posterior['sigma'].values.shape, (4, 100))


if __name__ == '__main__':
    unittest.main()
