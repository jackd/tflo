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


if __name__ == "__main__":
    tf.test.main()
