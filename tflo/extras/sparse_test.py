import tensorflow as tf

from tflo.extras.sparse import LinearOperatorSparseMatrix
from tflo.utils import get_random_st


class LinearOperatorSparseMatrixTest(tf.test.TestCase):
    def test_matmul(self, seed=0):
        rng = tf.random.Generator.from_seed(seed)
        # batch_shape = (5,)
        batch_shape = ()
        range_dim = 7
        domain_dim = 3
        n_rhs = 11
        st = get_random_st(rng, (*batch_shape, range_dim, domain_dim))
        x = rng.normal((domain_dim, n_rhs))
        lo = LinearOperatorSparseMatrix(st)

        expected = tf.sparse.sparse_dense_matmul(st, x)
        actual = lo.matmul(x)
        self.assertAllClose(actual, expected)


if __name__ == "__main__":
    tf.test.main()
