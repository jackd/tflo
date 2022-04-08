import tensorflow as tf

from tflo.ops import multi_conjugate_gradient


def stacked_conjugate_gradient(operator, rhs, **kwargs):
    rhs = tf.unstack(rhs, axis=-1)
    sols = [tf.linalg.experimental.conjugate_gradient(operator, r) for r in rhs]
    i, x, r, p, gamma = zip(*sols)
    i = tf.reduce_max(i)
    x, r, p, gamma = (tf.stack(t, axis=-1) for t in (x, r, p, gamma))
    return i, x, r, p, gamma


class OpsTest(tf.test.TestCase):
    def test_multi_conjugate_gradient(self, seed=0):
        rng = tf.random.Generator.from_seed(seed)
        b = (5, 3)
        n = 7
        n_rhs = 1
        A = rng.uniform((*b, n, n))
        A = (A + A) / 2 + n * tf.eye(n)
        A = tf.linalg.LinearOperatorFullMatrix(
            A, is_self_adjoint=True, is_positive_definite=True
        )

        rhs = rng.normal((*b, n, n_rhs))
        actual = multi_conjugate_gradient(A, rhs)
        expected = stacked_conjugate_gradient(A, rhs)
        # self.assertEqual(actual[0], expected[0])
        self.assertAllClose(actual[1], expected[1], rtol=1e-4)  # solution

        n_rhs = 2
        rhs = rng.normal((*b, n, n_rhs))
        actual = multi_conjugate_gradient(A, rhs)
        expected = stacked_conjugate_gradient(A, rhs)
        # self.assertEqual(actual[0], expected[0])  # iterations
        self.assertAllClose(actual[1], expected[1], rtol=1e-2)  # solution


if __name__ == "__main__":
    tf.test.main()
