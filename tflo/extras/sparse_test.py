import tensorflow as tf

from tflo.extras import test_utils
from tflo.extras.sparse import LinearOperatorSparseMatrix
from tflo.utils import get_random_st


class LinearOperatorSparseMatrixTest(tf.test.TestCase, test_utils.LinearOperatorTest):
    def _get_operator(self, rng: tf.random.Generator):
        batch_shape = ()
        range_dim = 7
        domain_dim = 3
        st = get_random_st(rng, (*batch_shape, range_dim, domain_dim))
        return LinearOperatorSparseMatrix(st)


if __name__ == "__main__":
    tf.test.main()
