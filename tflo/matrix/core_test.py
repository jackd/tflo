import tensorflow as tf

from tflo import test_utils


class AdjointMatrixTest(tf.test.TestCase, test_utils.MatrixTest):
    def _get_operator(self, rng: tf.random.Generator):
        batch_shape = (5,)
        range_dim = 3
        domain_dim = 7
        A = rng.normal((*batch_shape, range_dim, domain_dim))
        A = tf.linalg.LinearOperatorFullMatrix(A)
        A = tf.linalg.LinearOperatorAdjoint(A)
        return A


class CompositionMatrixTest(tf.test.TestCase, test_utils.MatrixTest):
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


class DiagMatrixTest(tf.test.TestCase, test_utils.MatrixTest):
    def _get_operator(self, rng: tf.random.Generator):
        batch_shape = (5,)
        num_rows = 3
        diag = rng.normal((*batch_shape, num_rows))
        return tf.linalg.LinearOperatorDiag(diag)


class FullMatrixTest(tf.test.TestCase, test_utils.MatrixTest):
    def _get_operator(self, rng: tf.random.Generator):
        batch_shape = (5,)
        range_dim = 3
        domain_dim = 7
        A = rng.normal((*batch_shape, range_dim, domain_dim))
        return tf.linalg.LinearOperatorFullMatrix(A)


class ScaledIdentityMatrixTest(tf.test.TestCase, test_utils.MatrixTest):
    def _get_operator(self, rng: tf.random.Generator):
        batch_shape = (5,)
        num_rows = 3
        multiplier = rng.normal(batch_shape)
        return tf.linalg.LinearOperatorScaledIdentity(num_rows, multiplier)


if __name__ == "__main__":
    tf.test.main()
