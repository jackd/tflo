import typing as tp

import tensorflow as tf

from tflo import utils


class LinearOperatorHStacked(tf.linalg.LinearOperator):
    """
    LinearOperator representing horizontally-stacked LinearOperators.

    Horizontal stacking is concatenating along the domain (last) dimension as per
    numpy's `np.hstack`.
    """

    def __init__(
        self,
        operators: tp.Iterable[tf.linalg.LinearOperator],
        is_non_singular: tp.Optional[str] = None,
        is_self_adjoint: tp.Optional[str] = None,
        is_positive_definite: tp.Optional[str] = None,
        is_square: tp.Optional[str] = None,
        name: str = "LinearOperatorHStacked",
    ):
        # don't use tuple - issues with Matrix.to_operator
        self._operators = list(operators)
        assert len(self._operators) > 0
        dtype = self._operators[0].dtype
        for op in self._operators[1:]:
            assert op.dtype == dtype

        super().__init__(
            dtype=dtype,
            is_non_singular=is_non_singular,
            is_self_adjoint=is_self_adjoint,
            is_positive_definite=is_positive_definite,
            is_square=is_square,
            name=name,
            parameters=dict(
                operators=self._operators,
            ),
        )

    def _shape(self):
        return utils.hstacked_shape([op.shape for op in self._operators])

    @property
    def domain_dimension(self):
        dim = utils.concatenated_dimension([op.shape[-1] for op in self._operators])
        return tf.compat.v1.Dimension(dim)

    @property
    def range_dimension(self):
        dim = utils.merged_dimension([op.shape[-2] for op in self._operators])
        return tf.compat.v1.Dimension(dim)

    def _adjoint(self):
        return LinearOperatorVStacked([op.adjoint() for op in self._operators])

    def _matmul(self, x, adjoint: bool = False, adjoint_arg: bool = False):
        if adjoint:
            return self._adjoint().matmul(x, adjoint_arg=adjoint_arg)
        xs = tf.split(
            x,
            [op.domain_dimension_tensor() for op in self._operators],
            axis=-1 if adjoint_arg else -2,
        )
        return tf.add_n(
            op.matmul(xi, adjoint_arg=adjoint_arg)
            for op, xi in zip(self._operators, xs)
        )

    def _matvec(self, x, adjoint: bool = False):
        if adjoint:
            return self._adjoint().matmul(x)
        xs = tf.split(
            x, [op.domain_dimension_tensor() for op in self._operators], axis=-1
        )
        return tf.add_n(op.matvec(xi) for op, xi in zip(self._operators, xs))

    def _to_dense(self):
        return tf.concat([op.to_dense() for op in self._operators], axis=-1)

    @property
    def _composite_tensor_fields(self):
        return ("operators",)


class LinearOperatorVStacked(tf.linalg.LinearOperator):
    """
    LinearOperator representing vertically-stacked LinearOperators.

    Horizontal stacking is concatenating along the range (second-last) dimension as per
    numpy's `np.vstack`.
    """

    def __init__(
        self,
        operators: tp.Iterable[tf.linalg.LinearOperator],
        is_non_singular: tp.Optional[str] = None,
        is_self_adjoint: tp.Optional[str] = None,
        is_positive_definite: tp.Optional[str] = None,
        is_square: tp.Optional[str] = None,
        name: str = "LinearOperatorVStacked",
    ):
        # don't use tuple - issues with Matrix.to_operator
        self._operators = list(operators)
        assert len(self._operators) > 0
        dtype = self._operators[0].dtype
        for op in self._operators[1:]:
            assert op.dtype == dtype

        super().__init__(
            dtype=dtype,
            is_non_singular=is_non_singular,
            is_self_adjoint=is_self_adjoint,
            is_positive_definite=is_positive_definite,
            is_square=is_square,
            name=name,
            parameters=dict(
                operators=self._operators,
            ),
        )

    def _shape(self):
        return utils.vstacked_shape([op.shape for op in self._operators])

    @property
    def domain_dimension(self):
        dim = utils.merged_dimension([op.shape[-1] for op in self._operators])
        return tf.compat.v1.Dimension(dim)

    @property
    def range_dimension(self):
        dim = utils.concatenated_dimension([op.shape[-2] for op in self._operators])
        return tf.compat.v1.Dimension(dim)

    def _adjoint(self):
        return LinearOperatorHStacked(tuple(op.adjoint() for op in self._operators))

    def _matmul(self, x, adjoint: bool = False, adjoint_arg: bool = False):
        if adjoint:
            return self._adjoint().matmul(x, adjoint_arg=adjoint_arg)
        return tf.concat(
            [op.matmul(x, adjoint_arg=adjoint_arg) for op in self._operators], axis=-2
        )

    def _matvec(self, x, adjoint: bool = False):
        if adjoint:
            return self._adjoint().matvec(x)
        return tf.concat([op.matvec(x) for op in self._operators], axis=-1)

    def _to_dense(self):
        return tf.concat([op.to_dense() for op in self._operators], axis=-2)

    @property
    def _composite_tensor_fields(self):
        return ("operators",)
