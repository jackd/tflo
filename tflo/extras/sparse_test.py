import tensorflow as tf

from tflo import test_utils
from tflo.extras.sparse import LinearOperatorSparseMatrix
from tflo.utils import get_random_st


class LinearOperatorSparseMatrixTest(
    tf.test.TestCase,
    test_utils.LinearOperatorTest,
    test_utils.MatrixTest,
):
    def _get_operator(self, rng: tf.random.Generator):
        batch_shape = ()
        range_dim = 7
        domain_dim = 3
        st = get_random_st(rng, (*batch_shape, range_dim, domain_dim))
        return LinearOperatorSparseMatrix(st)

    def test_matmul(self, seed=0, n_rhs=3, atol=1e-3, rtol=1e-3):
        super().test_matmul(seed=seed, n_rhs=n_rhs, rtol=rtol, atol=atol)


if __name__ == "__main__":
    tf.test.main()
