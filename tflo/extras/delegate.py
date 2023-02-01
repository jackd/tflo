import typing as tp

import tensorflow as tf


class LinearOperatorDelegate(tf.linalg.LinearOperator):
    """
    LinearOperator class that delegates implementations.

    Intended as a base class for other implementations with minimal changes. See
    `tflo.extras.prog` for an example.
    """

    def __init__(
        self,
        operator: tf.linalg.LinearOperator,
        name: str,
        parameters: tp.Optional[tp.Mapping] = None,
    ):
        parameters = dict(parameters) if parameters else {}
        parameters["operator"] = operator
        assert isinstance(operator, tf.linalg.LinearOperator)
        self._operator = operator
        super().__init__(
            dtype=operator.dtype,
            is_non_singular=operator.is_non_singular,
            is_self_adjoint=operator.is_self_adjoint,
            is_positive_definite=operator.is_positive_definite,
            is_square=operator.is_square,
            name=name,
            parameters=parameters,
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
        return self._operator._matmul(x, adjoint=adjoint, adjoint_arg=adjoint_arg)

    def _matvec(self, x, adjoint: bool = False):
        return self._operator._matvec(x, adjoint=adjoint)

    def _determinant(self):
        return self._operator._determinant()

    def _log_abs_determinant(self):
        return self._operator._log_abs_determinant()

    def _solve(self, rhs, adjoint: bool = False, adjoint_arg: bool = False):
        return self._operator._solve(rhs, adjoint=adjoint, adjoint_arg=adjoint_arg)

    def _solvevec(self, rhs, adjoint: bool = False):
        return self._operator._solvevec(rhs, adjoint=adjoint)

    def _to_dense(self):
        return self._operator._to_dense()

    def _diag_part(self):
        return self._operator._diag_part()

    def _trace(self):
        return self._operator._trace()

    def _add_to_tensor(self, x):
        return self._operator._add_to_tensor(x)

    def _eigvals(self):
        return self._operator._eigvals()

    def _cond(self):
        return self._operator._cond()

    @property
    def _composite_tensor_fields(self):
        return ("operator",)
