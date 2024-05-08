import typing as T  # pylint: disable=unused-import

import numpy as np
import numpy.typing as npt

from pydrake.solvers import (  # pylint: disable=import-error, no-name-in-module, unused-import
    MathematicalProgram,
    MathematicalProgramResult,
    Solve,
    MosekSolver,
    GurobiSolver,
    MosekSolverDetails,
    SnoptSolver,
    OsqpSolver,
    ClarabelSolver,
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
from pydrake.math import (  # pylint: disable=import-error, no-name-in-module, unused-import
    ge,
    eq,
    le,
)
import cProfile

import plotly.graph_objects as go  # pylint: disable=import-error
from plotly.express.colors import sample_colorscale  # pylint: disable=import-error
import plotly.graph_objs as go
from plotly.subplots import make_subplots

from collections import deque
from queue import PriorityQueue

from program_options import FREE_POLY, PSD_POLY, CONVEX_POLY, ProgramOptions

from util import (
    timeit,
    diditwork,
    INFO,
    YAY,
    WARN,
    ERROR,
    ChebyshevCenter,
    get_kth_control_point
)  # pylint: disable=import-error, no-name-in-module, unused-import

from gcs_util import get_edge_name, make_quadratic_cost_function_matrices, plot_a_gcs

from graph_dual import PolynomialDualGCS, DualEdge, DualVertex
from graph_dual_goal_conditioned import GoalConditionedDualEdge, GoalConditionedDualVertex, GoalConditionedPolynomialDualGCS
from plot_utils import plot_bezier



# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
# ---------------------------------------------------------------------

def get_path_cost(
    graph: PolynomialDualGCS,
    vertex_path: T.List[DualVertex],
    bezier_path: T.List[T.List[npt.NDArray]],
    add_edge_and_vertex_violations:bool = True,
    add_terminal_heuristic:bool = True,
    terminal_state:npt.NDArray = None
) -> float:
    """
    Note: this is cost of the full path with the terminal cost
    TODO: should i do the add G term heiuristic?
    """
    cost = 0.0
    for index, bezier_curve in enumerate(bezier_path):
        edge = graph.edges[get_edge_name(vertex_path[index].name, vertex_path[index + 1].name)]
        for i in range(len(bezier_curve) - 1):
            if graph.options.policy_use_l2_norm_cost:
                # use l2 norm
                cost += np.linalg.norm(bezier_curve[i]-bezier_curve[i + 1])
            elif graph.options.policy_use_quadratic_cost:
                delta = bezier_curve[i]-bezier_curve[i + 1]
                cost += np.linalg.norm(delta.dot(delta))
            else:
                # use regular cost instead
                if terminal_state is not None:
                    cost += edge.cost_function(bezier_curve[i], bezier_curve[i + 1], terminal_state)
                else:
                    cost += edge.cost_function(bezier_curve[i], bezier_curve[i + 1])

        if add_edge_and_vertex_violations:
            violations = graph.value_function_solution.GetSolution(edge.bidirectional_edge_violation) + graph.value_function_solution.GetSolution(edge.right.total_flow_in_violation)
            if isinstance(violations, Expression):
                violations = violations.Evaluate()
            cost += violations

        if add_terminal_heuristic and (index == len(bezier_path) - 1):
            if terminal_state is not None:
                assert isinstance(graph, GoalConditionedPolynomialDualGCS), "passed terminal state but not a Goal Conditioned policy"
                cost_to_go_at_last_point = vertex_path[-1].cost_at_point(bezier_curve[-1], terminal_state, graph.value_function_solution)
                if vertex_path[-1].vertex_is_target:
                    assert np.allclose(bezier_curve[-1], terminal_state), "last vertex in curve is terminal, but last point in curve is not a terminal point"
                    assert np.allclose(cost_to_go_at_last_point, 0.0), "cost-to-go should be 0"
                else:
                    cost += cost_to_go_at_last_point
            else:
                cost += vertex_path[-1].cost_at_point(bezier_curve[-1], graph.value_function_solution)
    return cost


def solve_parallelized_convex_restriction(
    graph: PolynomialDualGCS,
    vertex_paths: T.List[T.List[DualVertex]],
    state_now: npt.NDArray,
    options: ProgramOptions = None,
    verbose_failure=False,
    terminal_state:npt.NDArray = None,
    one_last_solve:bool = False,
    verbose_solve_success = True
) -> T.List[T.Tuple[T.List[DualVertex], T.List[T.List[npt.NDArray]]]]:

    """
    solve a convex restriction over a vertex path
    return cost of the vertex_path
    and return a list of bezier curves
    where bezier curve is a list of numpy arrays (vectors).
    """
    
    if options is None:
        options = graph.options

    # construct an optimization problem
    prog = MathematicalProgram()
    all_bezier_curves = []
    timer = timeit()

    def add_ge_lin_con(B:npt.NDArray, x:npt.NDArray):
        prog.AddLinearConstraint(-B[:, 1:], 
                                 -np.infty*np.ones( B.shape[0]),
                                 B[:, 0],
                                 x)

    for vertex_path in vertex_paths:

        # initial state
        last_x = None

        # previous direction of motion -- for bezier curve continuity
        bezier_curves = []

        # for every vertex:
        for i, vertex in enumerate(vertex_path):
            x = prog.NewContinuousVariables(vertex_path[0].state_dim)
            add_ge_lin_con(vertex.B, x)

            if i == 0:
                prog.AddLinearConstraint( eq(x, state_now))
            elif i == len(vertex_path) - 1:
                
                # if using terminal heuristic cost:
                if not options.policy_use_zero_heuristic:
                    if terminal_state is not None:
                        # goal conditioned case
                        potential = graph.value_function_solution.GetSolution(vertex.potential)
                        assert isinstance(vertex, GoalConditionedDualVertex)
                        if vertex.vertex_is_target:
                            # if a terminal vertex -- just add terminal constraint
                            prog.AddLinearConstraint( eq(x, terminal_state)) 
                        else:
                            # if not the terminal vertex -- plug in for the potential
                            # TODO: reimplement to speed up
                            def f_potential(x):
                                sub_x = potential.Substitute(
                                    {vertex.x[i]: x[i] for i in range(vertex.state_dim)}
                                )
                                sub_xt = sub_x.Substitute(
                                    {vertex.xt[i]: terminal_state[i] for i in range(vertex.state_dim)}
                                )
                                return sub_xt
                            prog.AddQuadraticCost(f_potential(x))
                    else:
                        potential = graph.value_function_solution.GetSolution(vertex.potential)
                        f_potential = lambda x: potential.Substitute(
                            {vertex.x[i]: x[i] for i in range(vertex.state_dim)}
                        )
                        prog.AddQuadraticCost(f_potential(x))
                bezier_curve = [last_x, x]
                bezier_curves.append(bezier_curve)
            else:
                bezier_curve = [last_x, x]
                bezier_curves.append(bezier_curve)
                edge = graph.edges[get_edge_name(vertex_path[i - 1].name, vertex.name)]

                # add the cost
                if graph.options.policy_use_l2_norm_cost:
                    add_l2_norm(prog, last_x, x)
                elif graph.options.policy_use_quadratic_cost:
                    add_quadratic_cost(prog, last_x, x)
                else:
                    if terminal_state is not None:
                        prog.AddQuadraticCost(edge.cost_function(last_x, x, terminal_state))
                    else:
                        prog.AddQuadraticCost(edge.cost_function(last_x, x))

                # # store the bezier curve
                # bezier_curves.append(bezier_curve)

            last_x = x

        all_bezier_curves.append(bezier_curves)

    timer.dt("just building", print_stuff=options.verbose_solve_times)
    
    # TODO: kinda nasty. how about instead i pass a solver constructor
    if options.policy_solver is None:
        solution = Solve(prog)
    else:
        if options.policy_solver == MosekSolver:
            mosek_solver = MosekSolver()
            solver_options = SolverOptions()
            # set the solver tolerance gaps
            if not one_last_solve:
                solver_options.SetOption(
                    MosekSolver.id(),
                    "MSK_DPAR_INTPNT_CO_TOL_REL_GAP",
                    options.policy_MSK_DPAR_INTPNT_CO_TOL_REL_GAP,
                )
                solver_options.SetOption(
                    MosekSolver.id(),
                    "MSK_DPAR_INTPNT_CO_TOL_PFEAS",
                    options.policy_MSK_DPAR_INTPNT_CO_TOL_PFEAS,
                )
                solver_options.SetOption(
                    MosekSolver.id(),
                    "MSK_DPAR_INTPNT_CO_TOL_DFEAS",
                    options.policy_MSK_DPAR_INTPNT_CO_TOL_DFEAS,
                )
            if options.policy_use_robust_mosek_params:
                solver_options.SetOption(MosekSolver.id(), "MSK_DPAR_INTPNT_CO_TOL_REL_GAP", 1e-3)
                solver_options.SetOption(MosekSolver.id(), "MSK_IPAR_INTPNT_SOLVE_FORM", 1)

            # solve the program
            solution = mosek_solver.Solve(prog, solver_options=solver_options)
        else:
            solution = options.policy_solver().Solve(prog)

    timer.dt("just solving", print_stuff=options.verbose_solve_times)

    final_result = []
    if solution.is_success():
        for i, v_path in enumerate(vertex_paths):
            bezier_curves = all_bezier_curves[i]
            bezier_solutions = [[solution.GetSolution(control_point) for control_point in bezier_curve] for bezier_curve in bezier_curves]
            full_tuple = (v_path, bezier_solutions)
            final_result.append(full_tuple)
        return final_result
    else:
        WARN("failed to solve",verbose = verbose_solve_success)
        if verbose_failure:
            diditwork(solution)
        return None
    

def solve_convex_restriction(
    graph: PolynomialDualGCS,
    vertex_path: T.List[DualVertex],
    state_now: npt.NDArray,
    options: ProgramOptions = None,
    verbose_failure:bool =False,
    terminal_state:npt.NDArray = None,
    one_last_solve = False
) -> T.List[T.List[npt.NDArray]]:
    result = solve_parallelized_convex_restriction(graph, [vertex_path], state_now, options, verbose_failure, terminal_state, one_last_solve, verbose_solve_success = False)
    if result is None:
        return None
    else:
        return result[0][1]
    

# ---

def add_l2_norm(prog: MathematicalProgram, x:npt.NDArray, y:npt.NDArray):
    n = len(x)
    A = np.hstack( (np.eye(n), -np.eye(n)) )
    b = np.zeros(n)
    prog.AddL2NormCostUsingConicConstraint(A, b, np.append(x,y))

def add_quadratic_cost(prog: MathematicalProgram, x:npt.NDArray, y:npt.NDArray):
    n = len(x)
    Q = 2 * np.vstack((np.hstack( (np.eye(n), -np.eye(n)) ), np.hstack( (-np.eye(n), np.eye(n)) ) ))
    b = np.zeros(2*n)
    c = 0
    prog.AddQuadraticCost(Q, b, c, np.append(x,y), True)



def get_optimal_path(
    graph: PolynomialDualGCS,
    vertex: DualVertex,
    state: npt.NDArray,
    options: ProgramOptions = None,
    terminal_state:npt.NDArray = None
) -> T.Tuple[float, float, T.List[T.List[npt.NDArray]], T.List[DualVertex]]:
    """
    return (time, cost, bezier path, vertex path) of the optimal solution.
    """
    if options is None:
        options = graph.options

    if terminal_state is not None:
        gcs, vertices, pseudo_terminal_vertex = graph.export_a_gcs(terminal_state)
    else:
        gcs, vertices, pseudo_terminal_vertex = graph.export_a_gcs()
    
    # set initial vertex constraint
    start_vertex = vertices[vertex.name]
    first_point = start_vertex.x()
    cons = eq(first_point, state)
    for con in cons:
        start_vertex.AddConstraint( con )

    # set terminal vertex constraint:
    if terminal_state is not None:
        assert isinstance(graph, GoalConditionedPolynomialDualGCS), "passed terminal state but not a Goal Conditioned policy"
        terminal_vertex = vertices[graph.terminal_vertex.name]
        last_point = terminal_vertex.x()
        assert len(last_point) == graph.terminal_vertex.state_dim
        cons = eq(last_point, terminal_state)
        for con in cons:
            terminal_vertex.AddConstraint( con )

    gcs_options = GraphOfConvexSetsOptions()
    gcs_options.convex_relaxation = options.gcs_policy_use_convex_relaxation
    gcs_options.max_rounding_trials = options.gcs_policy_max_rounding_trials
    gcs_options.preprocessing = options.gcs_policy_use_preprocessing
    gcs_options.max_rounded_paths = options.gcs_policy_max_rounded_paths
    if options.gcs_policy_solver is not None:
        gcs_options.solver = options.gcs_policy_solver()

    # solve
    timer = timeit()
    result = gcs.SolveShortestPath(
        start_vertex, pseudo_terminal_vertex, gcs_options
    )  # type: MathematicalProgramResult
    dt = timer.dt("just SolveShortestPath solve time", print_stuff=options.verbose_solve_times)

    assert result.is_success()
    cost = result.get_optimal_cost()
    if options.verbose_solve_times:
        diditwork(result)

    edge_path = gcs.GetSolutionPath(start_vertex, pseudo_terminal_vertex, result)
    vertex_name_path = []
    value_path = []
    for e in edge_path[:-1]:
        vertex_name_path.append(e.u().name())
        full_curve = [result.GetSolution(e.u().x()), result.GetSolution(e.v().x())]
        value_path.append(full_curve)
    vertex_name_path.append(edge_path[-1].u().name())
    if terminal_state is not None:
        assert np.allclose(value_path[-1][-1], terminal_state), "terminal state isn't 0, why?"
        
    return dt, cost, value_path, [graph.vertices[name] for name in vertex_name_path]

