# Tensorflow LinearOperators

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

- [tflo.extras](./tflo/extras): custom [tf.linalg.LinearOperator](https://www.tensorflow.org/api_docs/python/tf/linalg/LinearOperator?hl=en) implementations; and
- [tflo.matrix](./tflo/matrix): experimental keras-compatible wrappers.

See tests for example usage.

## Pre-commit

This package uses [pre-commit](https://pre-commit.com/) to ensure commits meet minimum criteria. To Install, use

```bash
pip install pre-commit
pre-commit install
```

This will ensure git hooks are run before each commit. While it is not advised to do so, you can skip these hooks with

```bash
git commit --no-verify -m "commit message"
```
