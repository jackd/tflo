import tensorflow as tf

from tflo import test_utils
from tflo.extras.prog import LinearOperatorProg


class LinearOperatorProgTest(
    tf.test.TestCase,
    test_utils.LinearOperatorTest,
    test_utils.MatrixTest,
):
    def _get_operator(self, rng: tf.random.Generator):
        n = 5
        A = rng.normal((n, n))
        A_op = tf.linalg.LinearOperatorFullMatrix(A)
        op = LinearOperatorProg(A_op)
        return op

    def test_matmul(self, seed=0, n_rhs=3, atol=1e-3, rtol=1e-3):
        super().test_matmul(seed=seed, n_rhs=n_rhs, rtol=rtol, atol=atol)


if __name__ == "__main__":
    tf.test.main()
