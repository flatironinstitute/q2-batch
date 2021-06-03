import unittest
from q2_batch._batch import _batch_func, _simulate


class TestBatch(unittest.TestCase):

    def setUp(self):
        self.table, self.metadata = _simulate(n=100, d=10, depth=50)

    def test_batch(self):
        print(self.table.values[:, 0])
        print(self.metadata)
        res = _batch_func(self.table.values[:, 0],
                          replicates=self.metadata['reps'].values,
                          batches=self.metadata['batch'].values,
                          depth=self.table.sum(axis=1),
                          mc_samples=2000)
        self.assertEqual(res.shape, (2000 * 4, 3))


if __name__ == '__main__':
    unittest.main()
