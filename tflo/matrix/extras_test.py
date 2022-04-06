import tensorflow as tf

from tflo import extras, utils
from tflo.matrix.test_utils import MatrixTest


class CGSolverMatrixTest(tf.test.TestCase, MatrixTest):
    def _get_operator(self, rng: tf.random.Generator):
        n = 5
        A = rng.uniform((n, n))
        A = 0.5 * (A + tf.transpose(A))
        A = A + tf.eye(n)
        A = tf.linalg.LinearOperatorFullMatrix(
            A, is_positive_definite=True, is_self_adjoint=True
        )
        return extras.LinearOperatorCGSolver(A)


class VStackedMatrixTest(tf.test.TestCase, MatrixTest):
    def _get_operator(self, rng: tf.random.Generator):
        stacked = rng.normal((7, 11, 3))
        split = tf.split(stacked, [2, 5, 4], axis=-2)
        ops = [tf.linalg.LinearOperatorFullMatrix(s) for s in split]
        return extras.LinearOperatorVStacked(ops)


class HStackedMatrixTest(tf.test.TestCase, MatrixTest):
    def _get_operator(self, rng: tf.random.Generator):
        stacked = rng.normal((5, 7, 3, 11))
        split = tf.split(stacked, [2, 5, 4], axis=-1)
        return extras.LinearOperatorHStacked(
            [tf.linalg.LinearOperatorFullMatrix(s) for s in split]
        )


class SparseMatrixTest(tf.test.TestCase, MatrixTest):
    def _get_operator(self, rng: tf.random.Generator):
        st = utils.get_random_st(rng, (11, 13))
        return extras.LinearOperatorSparseMatrix(st)


class StaticPowerSeriesMatrixTest(tf.test.TestCase, MatrixTest):
    def _get_operator(self, rng: tf.random.Generator):
        a = rng.normal((2, 5, 5))
        op = tf.linalg.LinearOperatorFullMatrix(a)
        coeffs = [float(x) for x in rng.normal((3,)).numpy()]
        return extras.LinearOperatorStaticPowerSeries(op, coeffs)


if __name__ == "__main__":
    tf.test.main()
