import tensorflow as tf

from tflo import test_utils
from tflo.extras import stacked


def get_tensors(
    rng: tf.random.Generator, var_axis: int, var_dims=(3, 5, 7), base_shape=(2, 11)
):
    assert var_axis < 0
    var_axis = var_axis + len(base_shape) + 1
    shape = list(base_shape)
    shape.insert(var_axis, sum(var_dims))
    stacked = rng.normal(shape)
    return tf.split(stacked, var_dims, axis=var_axis)


class VStackedTest(
    tf.test.TestCase,
    test_utils.LinearOperatorTest,
    test_utils.MatrixTest,
):
    def _get_operator(self, rng):
        tensors = get_tensors(rng, -2)
        return stacked.LinearOperatorVStacked(
            [tf.linalg.LinearOperatorFullMatrix(t) for t in tensors]
        )

    def test_matmul(self, seed=0, n_rhs=3, atol=1e-3, rtol=1e-3):
        super().test_matmul(seed=seed, n_rhs=n_rhs, rtol=rtol, atol=atol)


class HStackedTest(
    tf.test.TestCase,
    test_utils.LinearOperatorTest,
    test_utils.MatrixTest,
):
    def _get_operator(self, rng):
        tensors = get_tensors(rng, -1)
        return stacked.LinearOperatorHStacked(
            [tf.linalg.LinearOperatorFullMatrix(t) for t in tensors]
        )

    def test_matmul(self, seed=0, n_rhs=3, atol=1e-3, rtol=1e-3):
        super().test_matmul(seed=seed, n_rhs=n_rhs, rtol=rtol, atol=atol)


if __name__ == "__main__":
    tf.test.main()
    # HStackedTest().test_matmul()
    # VStackedTest().test_matmul()
