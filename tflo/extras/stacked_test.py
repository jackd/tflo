import abc
import typing as tp

import tensorflow as tf

from tflo.extras import stacked
from tflo.utils import hstack, vstack


def get_tensors(
    rng: tf.random.Generator, var_axis: int, var_dims=(3, 11, 13), base_shape=(2, 7)
):
    assert var_axis < 0
    var_axis = var_axis + len(base_shape) + 1
    shape = list(base_shape)
    shape.insert(var_axis, sum(var_dims))
    stacked = rng.normal(shape)
    return tf.split(stacked, var_dims, axis=var_axis)


class StackedTest(abc.ABC, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def get_op_and_dense(
        self, rng: tf.random.Generator
    ) -> tp.Tuple[tf.linalg.LinearOperator, tf.Tensor]:
        pass

    def get_op(self, rng: tf.random.Generator) -> tf.linalg.LinearOperator:
        return self.get_op_and_dense(rng)[0]

    def test_shape(self, seed=0):
        rng = tf.random.Generator.from_seed(seed)
        op = self.get_op(rng)
        dense = op.to_dense()
        self.assertAllEqual(op.shape, dense.shape)

    def test_to_dense(self, seed=0):
        rng = tf.random.Generator.from_seed(seed)
        op = self.get_op(rng)
        actual = op.to_dense()
        expected = op.matmul(
            tf.eye(int(op.domain_dimension), batch_shape=op.batch_shape)
        )
        self.assertAllClose(actual, expected)

    def test_matmul(self, seed=0):
        rng = tf.random.Generator.from_seed(seed)
        op, dense = self.get_op_and_dense(rng)
        n_rhs = 11
        rhs = rng.normal((*op.batch_shape, op.domain_dimension, n_rhs))

        actual = op.matmul(rhs)
        expected = dense @ rhs
        self.assertAllClose(actual, expected)

        adjointed = op.matmul(tf.linalg.adjoint(rhs), adjoint_arg=True)
        self.assertAllClose(actual, adjointed)

    def test_matvec(self, seed=0):
        rng = tf.random.Generator.from_seed(seed)
        op, dense = self.get_op_and_dense(rng)
        rhs = rng.normal((*op.batch_shape, op.domain_dimension))

        actual = op.matvec(rhs)
        expected = tf.linalg.matvec(dense, rhs)
        self.assertAllClose(actual, expected)

    def test_to_dense(self):
        op, dense = self.get_op_and_dense(tf.random.Generator.from_seed(0))
        self.assertAllEqual(op.to_dense(), dense)


class VStackedTest(tf.test.TestCase, StackedTest):
    def get_op_and_dense(self, rng):
        tensors = get_tensors(rng, -2)
        op = stacked.LinearOperatorVStacked(
            [tf.linalg.LinearOperatorFullMatrix(t) for t in tensors]
        )
        return op, vstack(tensors)


class HStackedTest(tf.test.TestCase, StackedTest):
    def get_op_and_dense(self, rng):
        tensors = get_tensors(rng, -1)
        op = stacked.LinearOperatorHStacked(
            [tf.linalg.LinearOperatorFullMatrix(t) for t in tensors]
        )
        return op, hstack(tensors)


if __name__ == "__main__":
    tf.test.main()
