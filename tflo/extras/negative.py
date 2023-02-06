import typing as tp

import tensorflow as tf


class LinearOperatorNegative(tf.linalg.LinearOperator):
    """LinearOperator representing `-operator`."""

    def __init__(
        self,
        operator: tf.linalg.LinearOperator,
        name: str = "LinearOperatorNegative",
        *,
        is_non_singular: tp.Optional[bool] = None,
        is_positive_definite: tp.Optional[bool] = None,
        is_square: tp.Optional[bool] = None,
        is_self_adjoint: tp.Optional[bool] = None,
    ):
        assert is_non_singular is None or is_non_singular == operator.is_non_singular
        assert (
            is_positive_definite is None
            or is_positive_definite != operator.is_positive_definite
        )
        assert is_square is None or is_square == operator.is_square
        assert is_self_adjoint is None or is_self_adjoint == operator.is_self_adjoint

        assert isinstance(operator, tf.linalg.LinearOperator)
        self._operator = operator
        super().__init__(
            dtype=operator.dtype,
            is_non_singular=operator.is_non_singular,
            is_self_adjoint=operator.is_self_adjoint,
            is_positive_definite=None
            if operator.is_positive_definite is None
            else not operator.is_positive_definite,
            is_square=operator.is_square,
            name=name,
            parameters=dict(operator=operator),
        )

    @property
    def operator(self):
        return self._operator

    def _shape(self):
        return self._operator.shape

    def _shape_tensor(self):
        return self._operator._shape_tensor()

    def _batch_shape_tensor(self):
        return self._operator._batch_shape_tensor()

    def _tensor_rank_tensor(self, shape=None):
        return self._operator._tensor_rank_tensor(shape=shape)

    def _domain_dimension_tensor(self, shape=None):
        return self._operator._domain_dimension_tensor(shape=shape)

    def _range_dimension_tensor(self, shape=None):
        return self._operator._range_dimension_tensor(shape=shape)

    def _assert_non_singular(self):
        return self._operator._assert_non_singular()

    def _max_condition_number_to_be_non_singular(self):
        return self._operator._max_condition_number_to_be_non_singular()

    def _assert_positive_definite(self):
        return self._operator._assert_positive_definite()

    def _assert_self_adjoint(self):
        return self._operator._assert_self_adjoint()

    def _matmul(self, x, adjoint: bool = False, adjoint_arg: bool = False):
        return -self._operator._matmul(x, adjoint=adjoint, adjoint_arg=adjoint_arg)

    def _matvec(self, x, adjoint: bool = False):
        return -self._operator._matvec(x, adjoint=adjoint)

    def _determinant(self):
        return self._operator._determinant() * (-1) ** self._domain_dimension_tensor()

    def _log_abs_determinant(self):
        return self._operator._log_abs_determinant()

    def _solve(self, rhs, adjoint: bool = False, adjoint_arg: bool = False):
        return -self._operator._solve(rhs, adjoint=adjoint, adjoint_arg=adjoint_arg)

    def _solvevec(self, rhs, adjoint: bool = False):
        return -self._operator._solvevec(rhs, adjoint=adjoint)

    def _to_dense(self):
        return -self._operator._to_dense()

    def _diag_part(self):
        return -self._operator._diag_part()

    def _trace(self):
        return -self._operator._trace()

    def _add_to_tensor(self, x):
        return -self._operator._add_to_tensor(-x)

    def _eigvals(self):
        return -self._operator._eigvals()

    def _cond(self):
        return self._operator._cond()

    @property
    def _composite_tensor_fields(self):
        return ("operator",)
