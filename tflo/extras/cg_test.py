import tensorflow as tf

from tflo import test_utils
from tflo.extras import cg


class LinearOperatorCGSolverTest(
    tf.test.TestCase,
    test_utils.LinearOperatorTest,
    test_utils.MatrixTest,
):
    def _get_operator(self, rng: tf.random.Generator):
        n = 5
        A = rng.uniform((n, n))
        A = 0.5 * (A + tf.transpose(A))
        A = A + n * tf.eye(n)
        A = tf.linalg.LinearOperatorFullMatrix(
            A, is_positive_definite=True, is_self_adjoint=True
        )
        return cg.LinearOperatorCGSolver(A, tol=1e-8, atol=1e-8)

    def test_matmul(self, seed=0, n_rhs=3, atol=1e-3, rtol=1e-3):
        super().test_matmul(seed=seed, n_rhs=n_rhs, rtol=rtol, atol=atol)

    def test_matvec(self, seed=0, atol=1e-3, rtol=1e-3):
        super().test_matvec(seed=seed, rtol=rtol, atol=atol)

    def test_adjoint(self, seed=0, atol=1e-3, rtol=1e-3):
        super().test_adjoint(seed=seed, rtol=rtol, atol=atol)


if __name__ == "__main__":
    tf.test.main()
