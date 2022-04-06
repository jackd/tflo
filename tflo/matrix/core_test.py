import tensorflow as tf

from tflo.matrix.test_utils import MatrixTest


class FullMatrixTest(tf.test.TestCase, MatrixTest):
    def _get_operator(self, rng: tf.random.Generator):
        batch_shape = (5,)
        range_dim = 3
        domain_dim = 7
        A = rng.normal((*batch_shape, range_dim, domain_dim))
        return tf.linalg.LinearOperatorFullMatrix(A)


class CompositionMatrixTest(tf.test.TestCase, MatrixTest):
    def _get_operator(self, rng: tf.random.Generator):
        batch_shape = (5,)
        range_dim = 3
        domain_dim = 7
        inner_dims = 2, 11
        leading_dims = (range_dim, *inner_dims)
        trailing_dims = (*inner_dims, domain_dim)
        tensors = [
            rng.normal((*batch_shape, l, t))
            for l, t in zip(leading_dims, trailing_dims)
        ]
        ops = [tf.linalg.LinearOperatorFullMatrix(t) for t in tensors]
        return tf.linalg.LinearOperatorComposition(ops)


if __name__ == "__main__":
    # tf.test.main()
    CompositionMatrixTest().test_to_operator()
