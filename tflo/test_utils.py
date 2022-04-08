import abc

import tensorflow as tf

from tflo.matrix import core


def solvevec(matrix, rhs, adjoint=False):
    sol = tf.linalg.solve(matrix, tf.expand_dims(rhs, -1), adjoint=adjoint)
    return tf.squeeze(sol, axis=-1)


def _get_vec(
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
            rhs = _get_vec(rng, op, adjoint=adjoint)
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
                rhs = _get_mat(
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

    def test_dimension_properties_are_dimensions(self, seed=0):
        # LinearOperatorComposition does checks that require dimensions to be Dimensions
        # raises an error if the returned value is an int or None.
        op = self._get_operator(tf.random.Generator.from_seed(seed))
        assert isinstance(op.range_dimension, tf.compat.v1.Dimension)
        assert isinstance(op.domain_dimension, tf.compat.v1.Dimension)


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
                rhs = _get_mat(
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
        vec = _get_vec(rng, op)  # square, so no need to get adjoint vec
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


class MatrixTest(abc.ABC):
    @abc.abstractmethod
    def _get_operator(self, rng: tf.random.Generator) -> tf.linalg.LinearOperator:
        pass

    def test_keras_input(self, seed=0):
        rng = tf.random.Generator.from_seed(seed)
        op = self._get_operator(rng)
        x = _get_mat(rng, op)

        matrix = core.from_operator(op)

        mat_inp = tf.keras.Input(type_spec=tf.type_spec_from_value(matrix))
        x_inp = tf.keras.Input(type_spec=tf.type_spec_from_value(x))

        model = tf.keras.Model((mat_inp, x_inp), mat_inp.matmul(x_inp))
        model_out = model((matrix, x))
        simple_out = matrix.matmul(x)

        self.assertAllClose(model_out, simple_out)

    def test_to_operator(self, seed=0):
        rng = tf.random.Generator.from_seed(seed)
        op = self._get_operator(rng)
        matrix = core.from_operator(op)
        op2 = matrix.to_operator()

        def assert_operators_equal(actual, expected):
            assert type(actual) == type(expected)
            flat_ac = tf.nest.flatten(actual.parameters)
            flat_ex = tf.nest.flatten(expected.parameters)
            for ac, ex in zip(flat_ac, flat_ex):
                if isinstance(ac, tf.linalg.LinearOperator):
                    assert_operators_equal(ac, ex)
                elif isinstance(ac, tf.SparseTensor):
                    assert isinstance(ex, tf.SparseTensor)
                    self.assertAllEqual(ac.indices, ex.indices)
                    self.assertAllEqual(ac.values, ex.values)
                    self.assertAllEqual(ac.dense_shape, ex.dense_shape)
                else:
                    self.assertAllEqual(ac, ex)

        assert_operators_equal(op, op2)

    def test_spec_parameters(self, seed=0):
        rng = tf.random.Generator.from_seed(seed)
        op = self._get_operator(rng)
        matrix = core.from_operator(op)
        self.assertAllEqual(op.shape, matrix.shape)
        self.assertEqual(op.dtype, matrix.dtype)
        spec = tf.type_spec_from_value(matrix)
        self.assertAllEqual(op.shape, spec.shape)
        self.assertEqual(op.dtype, spec.dtype)
