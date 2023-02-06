import typing as tp

import tensorflow as tf

import tflo.matrix.dispatch  # pylint: disable=unused-import
from tflo.matrix import core, extras

FullMatrix = core.FullMatrix


class DispatchTest(tf.test.TestCase):
    def _get_operator(self, rng: tf.random.Generator, shape: tp.Sequence[int]):
        return FullMatrix(rng.normal(shape=shape))

    def test_matmul(self, seed: int = 0):
        rng = tf.random.Generator.from_seed(seed)

        x = rng.normal((5, 7))
        y = rng.normal((7, 3))

        expected = x @ y
        actual = FullMatrix(x) @ FullMatrix(y)
        self.assertIsInstance(actual, core.CompositionMatrix)
        self.assertAllClose(actual.to_dense(), expected)

    def test_add(self, seed: int = 0):
        rng = tf.random.Generator.from_seed(seed)
        shape = (5, 7)
        x = rng.normal(shape)
        y = rng.normal(shape)

        expected = x + y
        actual = FullMatrix(x) + FullMatrix(y)
        self.assertIsInstance(actual, extras.SumMatrix)
        self.assertAllClose(actual.to_dense(), expected)

    def test_sub(self, seed: int = 0):
        rng = tf.random.Generator.from_seed(seed)
        shape = (5, 7)
        x = rng.normal(shape)
        y = rng.normal(shape)

        expected = x - y
        actual = FullMatrix(x) - FullMatrix(y)
        self.assertIsInstance(actual, extras.SumMatrix)
        self.assertAllClose(actual.to_dense(), expected)

    def test_neg(self, seed: int = 0):
        rng = tf.random.Generator.from_seed(seed)
        shape = (5, 7)
        x = rng.normal(shape)

        expected = -x
        actual = -FullMatrix(x)

        self.assertIsInstance(actual, extras.NegativeMatrix)
        self.assertAllClose(actual.to_dense(), expected)


if __name__ == "__main__":
    tf.config.experimental.enable_op_determinism()
    tf.test.main()
