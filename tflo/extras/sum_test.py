import tensorflow as tf

from tflo import test_utils
from tflo.extras.sum import LinearOperatorSum


class LinearOperatorSumTest(
    tf.test.TestCase,
    test_utils.LinearOperatorTest,
    test_utils.MatrixTest,
):
    def _get_operator(self, rng: tf.random.Generator):
        shape = (5, 7)
        num_terms = 3
        terms = [
            tf.linalg.LinearOperatorFullMatrix(rng.normal(shape))
            for _ in range(num_terms)
        ]
        op = LinearOperatorSum(terms)
        return op


if __name__ == "__main__":
    tf.test.main()
