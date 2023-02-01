import tensorflow as tf


class LinearOperatorTiled(tf.linalg.LinearOperator):
    def __init__(
        self,
        leading_dims: tf.Tensor,
        operator: tf.linalg.LinearOperator,
        name="LinearOperatorTiled",
    ):
        self._leading_dims = leading_dims
        self._operator = operator
        super().__init__(
            dtype=operator.dtype,
            is_non_singular=operator.is_non_singular,
            is_self_adjoint=operator.is_self_adjoint,
            is_positive_definite=operator.is_positive_definite,
            is_square=operator.is_square,
            name=name,
            parameters=dict(
                leading_dims,
                operator=operator,
            ),
        )

    def _shape_tensor(self):
        return tf.concat((self._leading_dims, self._operator._shape_tensor()))

    def _batch_shape_tensor(self):
        return tf.concat((self._leading_dims, self._operator._batch_shape_tensor()))

    def _domain_dimension_tensor(self, shape=None):
        return self._operator._domain_dimension_tensor(shape)

    def _range_dimension_tensor(self, shape=None):
        return self._operator._range_dimension_tensor(shape)

    def _shape(self):
        return tf.TensorShape(
            (None,) * self._leading_dims.shape[0] + self._operator.shape
        )

    def _flatten(self, x):
        shape = tf.concat(
            (
                tf.reduce_prod(self._leading_dims, keepdims=True),
                tf.shape(x, self._leading_dims.dtype)[self._leading_dims.shape.ndims :],
            )
        )
        return tf.reshape(x, shape)

    def _reshape(self, x):
        shape = tf.concat(
            (self._leading_dims, tf.shape(x, self._leading_dims.dtype)[1:])
        )
        return tf.reshape(x, shape)

    def _matvec(self, x, adjoint=False):
        x = self._flatten(x)
        out = tf.map_fn(lambda x: self._operator._matvec(x, adjoint=adjoint))
        return self._reshape(out)

    def _matmul(self, x, adjoint=False, adjoint_arg=False):
        x = self._flatten(x)
        out = tf.map_fn(
            lambda x: self._operator._matvec(
                x, adjoint=adjoint, adjoint_arg=adjoint_arg
            )
        )
        return self._reshape(out)
