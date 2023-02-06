from tflo.extras.cg import LinearOperatorCGSolver
from tflo.extras.exponential import LinearOperatorExponential
from tflo.extras.gather import LinearOperatorGather, LinearOperatorScatter
from tflo.extras.mapped import LinearOperatorMapped
from tflo.extras.prog import LinearOperatorProg
from tflo.extras.series import LinearOperatorStaticPowerSeries
from tflo.extras.sparse import LinearOperatorSparseMatrix
from tflo.extras.stacked import LinearOperatorHStacked, LinearOperatorVStacked
from tflo.extras.sum import LinearOperatorSum

__all__ = [
    "LinearOperatorCGSolver",
    "LinearOperatorExponential",
    "LinearOperatorGather",
    "LinearOperatorScatter",
    "LinearOperatorMapped",
    "LinearOperatorProg",
    "LinearOperatorSum",
    "LinearOperatorSparseMatrix",
    "LinearOperatorStaticPowerSeries",
    "LinearOperatorHStacked",
    "LinearOperatorVStacked",
]

from tensorflow.python.ops.linalg.linear_operator import make_composite_tensor

loc = locals()
for k in __all__:
    if k.startswith("LinearOperator"):
        make_composite_tensor(loc[k])
del make_composite_tensor, k, loc
