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
    make_moment_matrix,
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


def make_potential(indet_list: npt.NDArray, pot_type:str, poly_deg:int, prog: MathematicalProgram) -> T.Tuple[Expression, npt.NDArray]:
    state_dim = len(indet_list)
    vars_from_indet = Variables(indet_list)
    # quadratic polynomial. special case due to convex functions and special quadratic implementations
    if poly_deg == 0:
        a = np.zeros(state_dim)
        b = prog.NewContinuousVariables(1)[0]
        J_matrix = make_moment_matrix(b, a, np.zeros((state_dim, state_dim)))
        potential = Expression(b)
        if pot_type == PSD_POLY:
            prog.AddLinearConstraint(b >= 0)
            
    elif poly_deg == 1:
        a = prog.NewContinuousVariables(state_dim)
        b = prog.NewContinuousVariables(1)[0]
        J_matrix = make_moment_matrix(b, a, np.zeros((state_dim, state_dim)))
        potential = 2 * a.dot(indet_list) + b
    elif poly_deg == 2:
        J_matrix, potential = define_quadratic_polynomial(prog, indet_list, pot_type)
    else:
        J_matrix = None
        # free polynomial
        if pot_type == FREE_POLY:
            potential = prog.NewFreePolynomial(
                vars_from_indet, poly_deg
            ).ToExpression()
        # PSD polynomial
        elif pot_type == PSD_POLY:
            assert (
                poly_deg % 2 == 0
            ), "can't make a PSD potential of uneven degree"
            # potential is PSD polynomial
            potential = prog.NewSosPolynomial(vars_from_indet, poly_deg)[0].ToExpression()
        else:
            raise NotImplementedError("potential type not supported")
    return potential, J_matrix



def define_sos_constraint_over_polyhedron_multivar(
    prog: MathematicalProgram,
    list_of_vars: T.List[npt.NDArray],
    B_matrices: T.List[npt.NDArray],
    function: Expression,
    options: ProgramOptions,
    ellipsoid_matrices: T.List[npt.NDArray] = None,
) -> None:
    if ellipsoid_matrices is None:
        ellipsoid_matrices = [None]*len(list_of_vars)
    s_procedure = Expression(0)
    # deg 0
    lambda_0 = prog.NewContinuousVariables(1)[0]
    prog.AddLinearConstraint(lambda_0 >= 0)
    s_procedure += lambda_0

    all_variables = Variables(np.array(list_of_vars).flatten())

    # deg 1
    deg1 = options.s_procedure_multiplier_degree_for_linear_inequalities
    for i, vars_i in enumerate(list_of_vars):
        x_and_1 = np.hstack(([1], vars_i))
        B_i = B_matrices[i]
        if B_i is not None:
            deg_1_cons = B_i.dot(x_and_1)

            if deg1 == 0:
                lambda_1 = prog.NewContinuousVariables(len(deg_1_cons))
                prog.AddLinearConstraint(ge(lambda_1, 0))
            else:
                lambda_1 = [
                    prog.NewSosPolynomial(all_variables, deg1)[0].ToExpression()
                    for _ in range(len(deg_1_cons))
                ]
            s_procedure += deg_1_cons.dot(lambda_1)

            # deg 2
            if options.s_procedure_use_quadratic_multilpiers:
                lambda_2 = prog.NewSymmetricContinuousVariables(len(deg_1_cons))
                prog.AddLinearConstraint(ge(lambda_2, 0))
                s_procedure += np.sum((B_i.dot(np.outer(x_and_1, x_and_1)).dot(B_i.T)) * lambda_2)
                
                if options.s_procedure_quadratic_multiply_left_and_right:
                    for j in range(i+1, len(list_of_vars)):
                        B_j = B_matrices[j]
                        if B_j is not None:
                            y_and_1 = np.hstack(([1], list_of_vars[j]))
                            lambda_2_left_right = prog.NewContinuousVariables(len(deg_1_cons), B_j.shape[0])
                            prog.AddLinearConstraint(ge(lambda_2_left_right, 0))
                            s_procedure += np.sum((B_i.dot(np.outer(x_and_1, y_and_1)).dot(B_j.T)) * lambda_2_left_right)

        E_i = ellipsoid_matrices[i]
        if E_i is not None:
            el_mat = ellipsoid_matrices[i]
            lambda_el = prog.NewContinuousVariables(1)[0]
            prog.AddLinearConstraint(lambda_el >= 0)
            s_procedure += lambda_el * np.sum( el_mat * (np.outer(x_and_1, x_and_1)) )
            
        
    expr = function - s_procedure
    prog.AddSosConstraint(expr)



## storign and extracting lists into yaml files
import yaml
def store_it(num, trajectory):
    # Store the list into a YAML file
    with open("a_few_trajectories/traj" + str(num) + ".yaml", "w") as yaml_file:
        yaml.dump(trajectory, yaml_file)

def extract(file_name:str):
    # Extract the list back from the YAML file
    with open(file_name, "r") as yaml_file:
        extracted_list = yaml.load(yaml_file, Loader=yaml.FullLoader)
    return extracted_list