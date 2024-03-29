import tensorflow as tf

from tflo import test_utils
from tflo.extras.series import LinearOperatorStaticPowerSeries


class LinearOperatorStaticPowerSeriesTest(
    tf.test.TestCase,
    test_utils.NonSingularLinearOperatorTest,
    test_utils.MatrixTest,
):
    def _get_operator(self, rng: tf.random.Generator):
        n = 5
        coeffs = [float(i) for i in rng.normal((3,)).numpy()]
        A = rng.normal((n, n))
        A_op = tf.linalg.LinearOperatorFullMatrix(A)
        op = LinearOperatorStaticPowerSeries(A_op, coeffs, is_non_singular=True)
        return op

    def test_matvec(self, seed=0, atol=1e-3, rtol=1e-3):
        super().test_matvec(seed=seed, rtol=rtol, atol=atol)

    def test_matmul(self, seed=0, n_rhs=3, atol=1e-3, rtol=1e-3):
        super().test_matmul(seed=seed, n_rhs=n_rhs, rtol=rtol, atol=atol)


if __name__ == "__main__":
    tf.test.main()
