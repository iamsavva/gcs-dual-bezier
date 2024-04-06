import typing as T  # pylint: disable=unused-import

import numpy as np
import numpy.typing as npt

from pydrake.solvers import (  # pylint: disable=import-error, no-name-in-module, unused-import
    MathematicalProgram,
    MathematicalProgramResult,
    Solve,
    MosekSolver,
    MosekSolverDetails,
    SnoptSolver,
    IpoptSolver,
    SolverOptions,
    CommonSolverOption,
)
from pydrake.geometry.optimization import (  # pylint: disable=import-error, no-name-in-module
    GraphOfConvexSets,
    GraphOfConvexSetsOptions,
    HPolyhedron,
    Point,
    ConvexSet,
    Hyperrectangle,
)
import numbers
import pydot

from pydrake.symbolic import (  # pylint: disable=import-error, no-name-in-module, unused-import
    Polynomial,
    Variable,
    Variables,
    Expression,
)
from pydrake.math import ( # pylint: disable=import-error, no-name-in-module, unused-import
    ge,
    eq,
    le,
)  

import plotly.graph_objects as go  # pylint: disable=import-error
from plotly.express.colors import sample_colorscale  # pylint: disable=import-error
import plotly.graph_objs as go
from plotly.subplots import make_subplots

from program_options import FREE_POLY, PSD_POLY, CONVEX_POLY, ProgramOptions

from util import (
    timeit,
    diditwork,
    INFO,
    YAY,
    WARN,
    ERROR,
    ChebyshevCenter,
)  # pylint: disable=import-error, no-name-in-module, unused-import


def get_product_constraints(constraints: T.List[Expression]) -> T.List[Expression]:
    """
    given a list of constraints, returns a list of all constraints g_i g_j >= 0, i != j.
    """
    product_constraints = []
    for i, con_i in enumerate(constraints):
        # for j in range(i + 1, len(constraints)):
        for j in range(0, len(constraints)):
            product_constraints.append(con_i * constraints[j])
    return product_constraints

def get_B_matrix(A:npt.NDArray, b:npt.NDArray):
    return np.hstack((b.reshape((len(b), 1)), -A))

def make_linear_set_inequalities(vars:npt.NDArray, convex_set: ConvexSet, set_type):
    if set_type in (HPolyhedron, Hyperrectangle):
        if set_type == HPolyhedron:
            hpoly = convex_set
        else:
            hpoly = convex_set.MakeHPolyhedron()
        A, b = hpoly.A(), hpoly.b()
        # NOTE: B matrix is only defined for hpolyhedrons!!
        linear_set_inequalities = [b[i] - A[i].dot(vars) for i in range(len(b))]
    else:
        raise Exception("set inqualities not implemented for Point or Ellipsoids")
    return linear_set_inequalities



def define_quadratic_polynomial(
    prog: MathematicalProgram,
    x: npt.NDArray,
    pot_type: str,
    specific_J_matrix: npt.NDArray = None,
) -> T.Tuple[npt.NDArray, Expression]:
    # specific potential provided
    state_dim = len(x)

    if specific_J_matrix is not None:
        assert specific_J_matrix.shape == (state_dim + 1, state_dim + 1)
        J_matrix = specific_J_matrix
    else:
        # potential is a free polynomial
        # TODO: change order to match Russ's notation

        if pot_type == FREE_POLY:
            J_matrix = prog.NewSymmetricContinuousVariables(state_dim + 1)

        elif pot_type == PSD_POLY:
            J_matrix = prog.NewSymmetricContinuousVariables(state_dim + 1)
            prog.AddPositiveSemidefiniteConstraint(J_matrix)

        elif pot_type == CONVEX_POLY:
            # i don't actually care about the whole thing being PSD;
            # only about the x component being convex
            J_matrix = prog.NewSymmetricContinuousVariables(state_dim + 1)
            prog.AddPositiveSemidefiniteConstraint(J_matrix[1:, 1:])

            # NOTE: Russ uses different notation for PSD matrix. the following won't work
            # NOTE: i am 1 x^T; x X. Russ is X x; x^T 1
            # self.potential, self.J_matrix = prog.NewSosPolynomial( self.vars, 2 )
        else:
            raise NotImplementedError("potential type not supported")

    x_and_1 = np.hstack(([1], x))
    potential = x_and_1.dot(J_matrix).dot(x_and_1)

    return J_matrix, potential


def define_sos_constraint_over_polyhedron(
    prog: MathematicalProgram,
    left_vars: npt.NDArray,
    right_vars: npt.NDArray,
    function: Expression,
    B_left: npt.NDArray,
    B_right: npt.NDArray,
    options: ProgramOptions
):

    s_procedure = Expression(0)

    x_and_1 = np.hstack(([1], left_vars))
    y_and_1 = np.hstack(([1], right_vars))

    Bl = B_left
    Br = B_right

    # deg 0
    lambda_0 = prog.NewContinuousVariables(1)[0]
    prog.AddLinearConstraint(lambda_0 >= 0)
    s_procedure += lambda_0

    # deg 1
    deg_1_cons_l = Bl.dot(x_and_1)
    deg_1_cons_r = Br.dot(y_and_1)
    deg1 = options.s_procedure_multiplier_degree_for_linear_inequalities
    if deg1 == 0:
        lambda_1_left = prog.NewContinuousVariables(len(deg_1_cons_l))
        lambda_1_right = prog.NewContinuousVariables(len(deg_1_cons_r))
        prog.AddLinearConstraint(ge(lambda_1_left, 0))
        prog.AddLinearConstraint(ge(lambda_1_right, 0))
    else:
        lambda_1_left = [
            prog.NewSosPolynomial(left_vars, deg1)[0].ToExpression()
            for _ in range(len(deg_1_cons_l))
        ]
        lambda_1_right = [
            prog.NewSosPolynomial(right_vars, deg1)[0].ToExpression()
            for _ in range(len(deg_1_cons_r))
        ]
        
    s_procedure += deg_1_cons_l.dot(lambda_1_left) + deg_1_cons_r.dot(lambda_1_right)

    # deg 2
    if options.s_procedure_use_quadratic_multilpiers:
        lambda_2_left = prog.NewSymmetricContinuousVariables(len(deg_1_cons_l))
        lambda_2_right = prog.NewSymmetricContinuousVariables(len(deg_1_cons_r))
        prog.AddLinearConstraint(ge(lambda_2_left, 0))
        prog.AddLinearConstraint(ge(lambda_2_right, 0))
        s_procedure += np.sum((Bl.dot(np.outer(x_and_1, x_and_1)).dot(Bl.T)) * lambda_2_left)
        s_procedure += np.sum((Br.dot(np.outer(y_and_1, y_and_1)).dot(Br.T)) * lambda_2_right)

        if options.s_procedure_quadratic_multiply_left_and_right:
            lambda_2_left_right = prog.NewContinuousVariables(len(deg_1_cons_r), len(deg_1_cons_r))
            prog.AddLinearConstraint(ge(lambda_2_left_right, 0))
            s_procedure += np.sum((Bl.dot(np.outer(x_and_1, y_and_1)).dot(Br.T)) * lambda_2_left_right)
        else:
            lambda_2_left_right = []
    else:
        lambda_2_left = []
        lambda_2_right = []
        lambda_2_left_right = []

        
    expr = function - s_procedure
    prog.AddSosConstraint(expr)

    return (
        lambda_0,
        lambda_1_left,
        lambda_1_right,
        lambda_2_left,
        lambda_2_right,
        lambda_2_left_right,
    )