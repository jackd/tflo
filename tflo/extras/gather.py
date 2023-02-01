import tensorflow as tf

Dimension = tf.compat.v1.Dimension


class LinearOperatorGather(tf.linalg.LinearOperator):
    def __init__(
        self,
        indices: tf.Tensor,
        num_columns: int,
        dtype: tf.DType = tf.float32,
        is_non_singular=None,
        is_self_adjoint=None,
        is_positive_definite=None,
        is_square=None,
        name="LinearOperatorGather",
    ):
        self._indices = tf.convert_to_tensor(indices)
        self._indices.shape.assert_has_rank(1)
        self._num_columns = num_columns
        parameters = dict(
            indices=indices,
            num_columns=num_columns,
            dtype=dtype,
            is_non_singular=is_non_singular,
            is_self_adjoint=is_self_adjoint,
            is_positive_definite=is_positive_definite,
            is_square=is_square,
            name=name,
        )
        super().__init__(
            dtype=dtype,
            is_non_singular=is_non_singular,
            is_self_adjoint=is_self_adjoint,
            is_positive_definite=is_positive_definite,
            is_square=is_square,
            name=name,
            parameters=parameters,
        )

    def _shape(self):
        return tf.TensorShape((self.range_dimension, self.domain_dimension))

    @property
    def range_dimension(self) -> Dimension:
        return Dimension(self._indices.shape[0])

    @property
    def domain_dimension(self) -> Dimension:
        return Dimension(self._num_columns)

    def _shape_tensor(self):
        return tf.stack(
            (self._range_dimension_tensor(), self._domain_dimension_tensor())
        )

    def _range_dimension_tensor(self, shape=None):
        del shape
        return tf.size(self._indices)

    def _domain_dimension_tensor(self, shape=None):
        del shape
        return tf.convert_to_tensor(self._num_columns, tf.int32)

    def _adjoint(self):
        return LinearOperatorScatter(
            self._indices,
            self._num_columns,
            is_square=self.is_square,
            is_non_singular=self.is_non_singular,
            is_positive_definite=self.is_positive_definite,
        )

    def _matmul(self, x, adjoint: bool = False, adjoint_arg: bool = False):
        if adjoint:
            return self._adjoint()._matmul(x, adjoint_arg=adjoint_arg)
        if adjoint_arg:
            x = tf.linalg.adjoint(x)
        return tf.gather(x, self._indices, axis=-2)

    def _matvec(self, x, adjoint: bool = False):
        if adjoint:
            return self._adjoint()._matvec(x)
        return tf.gather(x, self._indices, axis=-1)

    def _to_dense(self):
        return tf.one_hot(self._indices, self._num_columns, dtype=self.dtype)

        # n = tf.size(self._indices, out_type=self._indices.dtype)
        # r = tf.range(n)
        # indices_2d = tf.stack((r, self._indices), axis=1)
        # return tf.scatter_nd(
        #     indices_2d,
        #     tf.ones((n,), dtype=self.dtype),
        #     tf.cast(self._shape_tensor(), tf.int64),
        # )

    @property
    def _composite_tensor_fields(self):
        return ("indices",)


class LinearOperatorScatter(tf.linalg.LinearOperator):
    def __init__(
        self,
        indices: tf.Tensor,
        num_rows: int,
        dtype: tf.DType = tf.float32,
        is_non_singular=None,
        is_self_adjoint=None,
        is_positive_definite=None,
        is_square=None,
        name="LinearOperatorGather",
    ):
        self._indices = tf.convert_to_tensor(indices)
        self._indices.shape.assert_has_rank(1)
        self._num_rows = num_rows
        parameters = dict(
            indices=indices,
            num_rows=num_rows,
            dtype=dtype,
            is_non_singular=is_non_singular,
            is_self_adjoint=is_self_adjoint,
            is_positive_definite=is_positive_definite,
            is_square=is_square,
            name=name,
        )
        super().__init__(
            dtype=dtype,
            is_non_singular=is_non_singular,
            is_self_adjoint=is_self_adjoint,
            is_positive_definite=is_positive_definite,
            is_square=is_square,
            name=name,
            parameters=parameters,
        )

    def _shape(self):
        return tf.TensorShape((self.range_dimension, self.domain_dimension))

    @property
    def range_dimension(self) -> Dimension:
        return Dimension(self._num_rows)

    @property
    def domain_dimension(self) -> Dimension:
        return Dimension(self._indices.shape[0])

    def _shape_tensor(self):
        return tf.stack(
            (self._range_dimension_tensor(), self._domain_dimension_tensor())
        )

    def _range_dimension_tensor(self, shape=None):
        del shape
        return tf.convert_to_tensor(self._num_rows, tf.int32)

    def _domain_dimension_tensor(self, shape=None):
        del shape
        return tf.size(self._indices)

    def _adjoint(self):
        return LinearOperatorGather(
            self._indices,
            self._num_rows,
            is_square=self.is_square,
            is_non_singular=self.is_non_singular,
            is_positive_definite=self.is_positive_definite,
        )

    def _matmul(self, x, adjoint: bool = False, adjoint_arg: bool = False):
        x.shape.assert_has_rank(2)
        if adjoint:
            return self._adjoint()._matmul(x, adjoint_arg=adjoint_arg)
        if adjoint_arg:
            x = tf.linalg.adjoint(x)
        shape = tf.stack((self._range_dimension_tensor(), tf.shape(x)[1]))
        return tf.scatter_nd(
            tf.expand_dims(self._indices, 1), x, tf.cast(shape, tf.int64)
        )

    def _matvec(self, x, adjoint: bool = False):
        if adjoint:
            return self._adjoint()._matvec(x)
        shape = tf.expand_dims(self._range_dimension_tensor(), 0)
        return tf.scatter_nd(
            tf.expand_dims(self._indices, 1),
            x,
            tf.cast(shape, tf.int64),
        )

    def _to_dense(self):
        n = tf.size(self._indices, out_type=self._indices.dtype)
        r = tf.range(n)
        indices_2d = tf.stack((self._indices, r), axis=1)
        return tf.scatter_nd(
            indices_2d,
            tf.ones((n,), dtype=self.dtype),
            tf.cast(self._shape_tensor(), tf.int64),
        )

    @property
    def _composite_tensor_fields(self):
        return ("indices",)
