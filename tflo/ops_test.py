import tensorflow as tf

from tflo.ops import multi_conjugate_gradient, single_conjugate_gradient, CGState


def stacked_conjugate_gradient(operator, rhs, **kwargs):
    rhs = tf.unstack(rhs, axis=-1)
    sols = [single_conjugate_gradient(operator, r, **kwargs) for r in rhs]
    i, x, r, p, gamma, converged = (tf.stack(t, axis=-1) for t in zip(*sols))
    return CGState(i, x, r, p, gamma, converged)


class OpsTest(tf.test.TestCase):
    def test_single_conjugate_gradient_consistent(self, seed=0):
        rng = tf.random.Generator.from_seed(seed)
        b = (5, 3)
        n = 7
        A = rng.uniform((*b, n, n))
        A = (A + tf.transpose(A, perm=(0, 1, 3, 2))) / 2 + n * tf.eye(n)
        A = tf.linalg.LinearOperatorFullMatrix(
            A, is_self_adjoint=True, is_positive_definite=True
        )
        rhs = rng.normal((*b, n))
        tol = 1e-5
        actual = single_conjugate_gradient(A, rhs, tol=tol, atol=0)
        expected = tf.linalg.experimental.conjugate_gradient(A, rhs, tol=tol)
        self.assertAllEqual(actual.i, expected.i)
        self.assertAllClose(actual.x, expected.x)
        self.assertAllClose(actual.r, expected.r)
        self.assertAllClose(actual.p, expected.p)
        self.assertAllClose(actual.gamma, expected.gamma)

    def test_multi_conjugate_gradient(self, seed=0):
        rng = tf.random.Generator.from_seed(seed)
        b = (5, 3)
        n = 7
        n_rhs = 2
        A = rng.uniform((*b, n, n))
        A = A = (A + tf.transpose(A, perm=(0, 1, 3, 2))) / 2 + n * tf.eye(n)
        A = tf.linalg.LinearOperatorFullMatrix(
            A, is_self_adjoint=True, is_positive_definite=True
        )
        kwargs = dict(max_iter=100, tol=1e-5, atol=1e-5)

        rhs = rng.normal((*b, n, n_rhs))
        actual = multi_conjugate_gradient(A, rhs, **kwargs)
        expected = stacked_conjugate_gradient(A, rhs, **kwargs)
        self.assertAllEqual(actual[0], expected[0])
        self.assertAllClose(actual[1], expected[1], atol=1e-5)  # solution

        n_rhs = 20
        rhs = rng.normal((*b, n, n_rhs))
        actual = multi_conjugate_gradient(A, rhs, **kwargs)
        expected = stacked_conjugate_gradient(A, rhs, **kwargs)
        self.assertAllEqual(actual.i, expected.i)
        self.assertAllClose(actual.x, expected.x, atol=1e-4)
        self.assertAllClose(actual.r, expected.r, atol=1e-4)
        self.assertAllClose(actual.p, expected.p, atol=1e-4)
        self.assertAllClose(actual.gamma, expected.gamma, atol=1e-4)
        self.assertAllClose(actual.converged, expected.converged)


if __name__ == "__main__":
    tf.test.main()
