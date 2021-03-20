import unittest
from q2_batch._method import estimate
from q2_batch._batch import _simulate
import qiime2
import xarray as xr


class TestBatchEstimation(unittest.TestCase):

    def setUp(self):
        self.table, self.metadata = _simulate(n=50, d=4, depth=30)

    def test_batch(self):
        res = estimate(
            self.table,
            replicates=qiime2.CategoricalMetadataColumn(self.metadata['reps']),
            batches=qiime2.CategoricalMetadataColumn(self.metadata['batch']),
            monte_carlo_samples=100,
            cores=1
        )
        self.assertTrue(res is not None)
        self.assertTrue(isinstance(res, xr.Dataset))

    def test_batch_dask(self):
        res = estimate(
            self.table,
            replicates=qiime2.CategoricalMetadataColumn(self.metadata['reps']),
            batches=qiime2.CategoricalMetadataColumn(self.metadata['batch']),
            monte_carlo_samples=100,
            cores=4
        )
        self.assertTrue(res is not None)
        self.assertTrue(isinstance(res, xr.Dataset))


if __name__ == '__main__':
    unittest.main()
