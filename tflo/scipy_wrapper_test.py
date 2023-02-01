import scipy.sparse.linalg as la
import tensorflow as tf

from tflo.extras import LinearOperatorSparseMatrix
from tflo.scipy_wrapper import ScipyWrapper


class ScipyWrapperTest(tf.test.TestCase):
    def test_scipy_wrapper(self, n: int = 5):
        d = tf.range(1, n + 1, dtype=tf.float32)
        D = tf.sparse.eye(n).with_values(d)
        D = LinearOperatorSparseMatrix(D)
        x = tf.ones((n,))
        sol, info = la.cg(ScipyWrapper(D), x.numpy())
        del info
        self.assertAllClose(sol, 1 / d)


if __name__ == "__main__":
    tf.test.main()
