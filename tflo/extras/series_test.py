import tensorflow as tf

from tflo.extras.series import LinearOperatorStaticPowerSeries


class LinearOperatorStaticPowerSeriesTest(tf.test.TestCase):
    def test_matmul(self, seed=0):
        rng = tf.random.Generator.from_seed(seed)
        n = 5
        n_rhs = 2
        coeffs = rng.normal((3,)).numpy()
        coeffs[3:] = 0
        A = rng.normal((n, n))
        x = rng.normal((n, n_rhs))

        A_op = tf.linalg.LinearOperatorFullMatrix(A)
        op = LinearOperatorStaticPowerSeries(A_op, coeffs)

        actual = op.matmul(x)
        Ax = A @ x
        AAx = A @ Ax
        expected = coeffs[0] * x + coeffs[1] * Ax + coeffs[2] * AAx
        self.assertAllClose(actual, expected)

    def test_matvec(self, seed=0):
        rng = tf.random.Generator.from_seed(seed)
        n = 5
        coeffs = rng.normal((3,)).numpy()
        coeffs[3:] = 0
        A = rng.normal((n, n))
        x = rng.normal((n,))

        A_op = tf.linalg.LinearOperatorFullMatrix(A)
        op = LinearOperatorStaticPowerSeries(A_op, coeffs)

        actual = op.matvec(x)
        Ax = tf.linalg.matvec(A, x)
        AAx = tf.linalg.matvec(A, Ax)
        expected = coeffs[0] * x + coeffs[1] * Ax + coeffs[2] * AAx
        self.assertAllClose(actual, expected)


if __name__ == "__main__":
    tf.test.main()
