import typing as tp

import tensorflow as tf


class LinearOperatorSum(tf.linalg.LinearOperator):
    """LinearOperator representing sum of LinearOperators."""

    def __init__(
        self,
        operators: tp.Iterable[tf.linalg.LinearOperator],
        is_non_singular: tp.Optional[str] = None,
        is_self_adjoint: tp.Optional[str] = None,
        is_positive_definite: tp.Optional[str] = None,
        is_square: tp.Optional[str] = None,
        name: str = "LinearOperatorSum",
    ):
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

    def _shape(self) -> tf.TensorShape:
        shape = self._operators[0].shape
        for op in self._operators[1:]:
            shape = tf.broadcast_static_shape(shape, op.shape)
        return shape

    def _shape_tensor(self):
        shape = self._operators[0]._shape_tensor()
        for op in self._operators[1:]:
            shape = tf.broadcast_dynamic_shape(shape, op._shape_tensor())
        return shape

    def _adjoint(self):
        return LinearOperatorSum([op.adjoint() for op in self._operators])

    def _matmul(self, x, adjoint: bool = False, adjoint_arg: bool = False):
        return tf.add_n(
            [
                op.matmul(x, adjoint=adjoint, adjoint_arg=adjoint_arg)
                for op in self._operators
            ]
        )

    def _matvec(self, x, adjoint: bool = False):
        return tf.add_n([op.matvec(x, adjoint=adjoint) for op in self._operators])

    def _to_dense(self):
        return tf.add_n([op.to_dense() for op in self._operators])

    @property
    def _composite_tensor_fields(self):
        return ("operators",)
