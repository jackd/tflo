import abc

import tensorflow as tf

from tflo.matrix import core


class MatrixTest(abc.ABC):
    @abc.abstractmethod
    def _get_operator(self, rng: tf.random.Generator):
        pass

    def _get_vec(self, rng: tf.random.Generator, operator: tf.linalg.LinearOperator):
        return rng.normal((*operator.batch_shape, operator.domain_dimension))

    def _get_mat(
        self,
        rng: tf.random.Generator,
        operator: tf.linalg.LinearOperator,
        n_rhs: int = 2,
    ):
        return rng.normal((*operator.batch_shape, operator.domain_dimension, n_rhs))

    def test_matvec(self, seed=0):
        rng = tf.random.Generator.from_seed(seed)
        op = self._get_operator(rng)
        matrix = core.from_operator(op)

        rhs = self._get_vec(rng, op)
        actual = matrix.matvec(rhs)
        expected = op.matvec(rhs)
        self.assertAllClose(actual, expected)

    def test_matmul(self, seed=0):
        rng = tf.random.Generator.from_seed(seed)
        op = self._get_operator(rng)
        matrix = core.from_operator(op)

        rhs = self._get_mat(rng, op)
        actual = matrix.matmul(rhs)
        expected = op.matmul(rhs)
        self.assertAllClose(actual, expected)

    def test_keras_input(self, seed=0):
        rng = tf.random.Generator.from_seed(seed)
        op = self._get_operator(rng)
        x = self._get_mat(rng, op)

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
