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

    def test_to_dense(self, seed=0, atol=1e-3, rtol=1e-3):
        super().test_to_dense(seed=seed, atol=atol, rtol=rtol)

    def test_matmul(self, seed=0, n_rhs=3, atol=2e-3, rtol=1e-3):
        super().test_matmul(seed=seed, n_rhs=n_rhs, atol=atol, rtol=rtol)


if __name__ == "__main__":
    tf.test.main()
