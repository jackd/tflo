import typing as tp

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as la
import tensorflow as tf

from tflo.extras.sparse import LinearOperatorSparseMatrix


class ScipyWrapper(la.LinearOperator):
    def __init__(self, operator: tf.linalg.LinearOperator):
        assert isinstance(operator, tf.linalg.LinearOperator), type(operator)
        self._operator = operator

    @property
    def dtype(self):
        return np.dtype(self._operator.dtype.as_numpy_dtype)

    @property
    def shape(self):
        return tuple(self._operator.shape)

    def _adjoint(self):
        return ScipyWrapper(self._operator.adjoint())

    def _matvec(self, x: np.ndarray):
        return self._operator.matvec(tf.convert_to_tensor(x)).numpy()

    def _matmul(self, x: np.ndarray):
        return self._operator.matmul(tf.convert_to_tensor(x)).numpy()


def to_scipy(
    operator: tf.linalg.LinearOperator,
) -> tp.Union[la.LinearOperator, sp.spmatrix]:
    if isinstance(operator, LinearOperatorSparseMatrix):
        st: tf.SparseTensor = operator.matrix
        return sp.coo_matrix(
            (st.values.numpy(), st.indices.numpy().T), shape=tuple(st.shape)
        )
    return ScipyWrapper(operator)
