import tensorflow as tf

from tflo import test_utils
from tflo.extras import gather


class LinearOperatorGatherTest(
    tf.test.TestCase,
    test_utils.LinearOperatorTest,
    test_utils.MatrixTest,
):
    def _get_operator(self, rng: tf.random.Generator):
        num_rows = 5
        num_columns = 10

        indices = rng.uniform((num_rows,), maxval=num_columns, dtype=tf.int64)
        return gather.LinearOperatorGather(indices, num_columns)


class LinearOperatorScatterTest(tf.test.TestCase, test_utils.LinearOperatorTest):
    def _get_operator(self, rng: tf.random.Generator):
        num_rows = 10
        num_columns = 5

        indices = rng.uniform((num_columns,), maxval=num_rows, dtype=tf.int64)
        return gather.LinearOperatorScatter(indices, num_rows)


if __name__ == "__main__":
    tf.test.main()
