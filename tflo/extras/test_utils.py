import abc

import tensorflow as tf


def solvevec(matrix, rhs, adjoint=False):
    sol = tf.linalg.solve(matrix, tf.expand_dims(rhs, -1), adjoint=adjoint)
    return tf.squeeze(sol, axis=-1)


class LinearOperatorTest(abc.ABC):
    """
    Mixin class for testing `tf.linalg.LinearOperator`s.

    Example usage
    ```python
    class LinearOperatorCustomTest(tf.test.TestCase, LinearOperatorTest):
        def _get_operator(self, rng: tf.random.Generator):
            foo = rng.normal(())
            return LinearOperatorCustom(foo)
    ```
    """

    @abc.abstractmethod
    def _get_operator(self, rng: tf.random.Generator) -> tf.linalg.LinearOperator:
        pass

    def _get_vec(
        self,
        rng: tf.random.Generator,
        operator: tf.linalg.LinearOperator,
        adjoint: bool = False,
    ):
        return rng.normal(
            (
                *operator.batch_shape,
                operator.range_dimension if adjoint else operator.domain_dimension,
            )
        )

    def _get_mat(
        self,
        rng: tf.random.Generator,
        operator: tf.linalg.LinearOperator,
        n_rhs: int = 2,
        adjoint: bool = False,
        adjoint_arg: bool = False,
    ):
        dim = operator.range_dimension if adjoint else operator.domain_dimension
        trailing = (dim, n_rhs)
        if adjoint_arg:
            trailing = trailing[-1::-1]
        return rng.normal((*operator.batch_shape, *trailing))

    def test_to_dense(self, seed=0):
        """Ensure `op.to_dense()` does the same as `op @ tf.eye(...)`."""
        op = self._get_operator(tf.random.Generator.from_seed(seed))
        actual = op.to_dense()
        expected = op.matmul(
            tf.eye(op.domain_dimension_tensor(), batch_shape=op.batch_shape_tensor())
        )
        self.assertAllClose(actual, expected)

    def test_adjoint(self, seed=0):
        op = self._get_operator(tf.random.Generator.from_seed(seed))
        actual = op.adjoint().to_dense()
        expected = tf.linalg.adjoint(op.to_dense())
        self.assertAllClose(actual, expected)

    def test_matvec(self, seed=0, atol=1e-6, rtol=1e-6):
        rng = tf.random.Generator.from_seed(seed)
        op = self._get_operator(rng)
        dense = op.to_dense()

        for adjoint in (False, True):
            rhs = self._get_vec(rng, op, adjoint=adjoint)
            actual = op.matvec(rhs, adjoint=adjoint)
            expected = tf.linalg.matvec(dense, rhs, adjoint_a=adjoint)
            self.assertAllClose(
                actual,
                expected,
                atol=atol,
                rtol=rtol,
                msg=f"matvec consistent with adjoint={adjoint}",
            )

    def test_matmul(self, seed=0, n_rhs=3, atol=1e-6, rtol=1e-6):
        rng = tf.random.Generator.from_seed(seed)
        op = self._get_operator(rng)
        dense = op.to_dense()

        for adjoint in (False, True):
            for adjoint_arg in (False, True):
                rhs = self._get_mat(
                    rng, op, n_rhs=n_rhs, adjoint=adjoint, adjoint_arg=adjoint_arg
                )
                actual = op.matmul(rhs, adjoint=adjoint, adjoint_arg=adjoint_arg)
                expected = tf.linalg.matmul(
                    dense, rhs, adjoint_a=adjoint, adjoint_b=adjoint_arg
                )
            self.assertAllClose(
                actual,
                expected,
                atol=atol,
                rtol=rtol,
                msg=f"matmul consistent with adjoint={adjoint}, adjoint_arg={adjoint_arg}",
            )

    def test_shape(self, seed=0):
        op = self._get_operator(tf.random.Generator.from_seed(seed))

        # def assert_consistent(static, dynamic, msg=None):
        #     self.assertEqual(static, dynamic, msg=msg)

        # for i, (static, dynamic) in enumerate(zip(op.shape, op.shape_tensor())):
        #     assert_consistent(static, dynamic, msg=f"dimension {i} consistent")
        # assert_consistent(
        #     op.range_dimension, op.range_dimension_tensor(), "range_dim consistent"
        # )
        # assert_consistent(
        #     op.domain_dimension, op.domain_dimension_tensor(), "domain_dim consistent"
        # )
        # assert_consistent(
        #     op.tensor_rank, op.tensor_rank_tensor(), "tensor_rank consistent"
        # )

        self.assertAllEqual(op.shape, op.shape_tensor())
        self.assertEqual(op.range_dimension, op.range_dimension_tensor())
        self.assertEqual(op.domain_dimension, op.domain_dimension_tensor())
        self.assertEqual(op.tensor_rank, op.tensor_rank_tensor())


class SquareLinearOperatorTest(LinearOperatorTest):
    def test_is_square(self, seed=0):
        op = self._get_operator(tf.random.Generator.from_seed(0))
        assert op.is_square

    def test_determinant(self, seed=0):
        op = self._get_operator(tf.random.Generator.from_seed(seed))
        actual = op.determinant()
        expected = tf.linalg.det(op.to_dense())
        self.assertAllClose(actual, expected)

    def test_log_abs_det(self, seed=0):
        op = self._get_operator(tf.random.Generator.from_seed(seed))
        dense = op.to_dense()
        actual = op.log_abs_determinant()
        expected = tf.math.log(tf.abs(tf.linalg.det(dense)))
        self.assertAllClose(actual, expected)


class NonSingularLinearOperatorTest(SquareLinearOperatorTest):
    def test_non_singular(self, seed=0):
        op = self._get_operator(tf.random.Generator.from_seed(seed))
        assert op.is_non_singular

    def test_inverse(self, seed=0):
        op = self._get_operator(tf.random.Generator.from_seed(seed))
        actual = op.inverse()
        if not tf.is_tensor(actual):
            actual = actual.to_dense()
        expected = tf.linalg.inv(op.to_dense())
        self.assertAllClose(actual, expected)

    def test_solve(self, seed=0, n_rhs=2):
        rng = tf.random.Generator.from_seed(seed)
        op = self._get_operator(rng)
        dense = op.to_dense()
        for adjoint in (False, True):
            for adjoint_arg in (False, True):
                rhs = self._get_mat(
                    rng, op, n_rhs=n_rhs, adjoint=adjoint, adjoint_arg=adjoint_arg
                )
                actual = op.solve(
                    rhs,
                    adjoint=adjoint,
                    adjoint_arg=adjoint_arg,
                )
                expected = tf.linalg.solve(
                    dense,
                    tf.linalg.adjoint(rhs) if adjoint_arg else rhs,
                    adjoint=adjoint,
                )
                self.assertAllClose(
                    actual,
                    expected,
                    rtol=1e-5,
                    msg="solve consistent with "
                    f"adjoint={adjoint}, adjoint_arg={adjoint_arg}",
                )

    def test_solvevec(self, seed=0):
        rng = tf.random.Generator.from_seed(seed)
        op = self._get_operator(rng)
        vec = self._get_vec(rng, op)  # square, so no need to get adjoint vec
        dense = op.to_dense()
        for adjoint in (False, True):
            actual = op.solvevec(vec, adjoint=adjoint)
            expected = solvevec(dense, vec, adjoint=adjoint)
            self.assertAllClose(
                actual,
                expected,
                rtol=1e-5,
                msg=f"solvevec consistent with adjoint={adjoint}",
            )


class PositiveDefiniteLinearOperatorTest(NonSingularLinearOperatorTest):
    def test_positive_definite(self, seed=0):
        op = self._get_operator(tf.random.Generator.from_seed(seed))
        assert op.is_positive_definite
