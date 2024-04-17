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
# from polynomial_dual_gcs_utils import (
#     define_quadratic_polynomial,
#     # define_sos_constraint_over_polyhedron,
# )

from bezier_dual import PolynomialDualGCS, DualEdge, DualVertex
from bezier_dual_goal_conditioned import GoalConditionedDualEdge, GoalConditionedDualVertex, GoalConditionedPolynomialDualGCS
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
    state_last: npt.NDArray = None,
    options: ProgramOptions = None,
    verbose_failure=False,
    terminal_state:npt.NDArray = None
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
    for vertex_path in vertex_paths:

        # initial state
        last_x = prog.NewContinuousVariables(vertex_path[0].state_dim)
        prog.AddLinearConstraint(eq(last_x, state_now))
        # previous direction of motion -- for bezier curve continuity
        last_delta = None
        if state_last is not None:
            last_delta = last_x - state_last

        bezier_curves = []
        # for every vertex:
        for i, vertex in enumerate(vertex_path):
            # it's the last vertex -- don't add a bezier curve; add terminal cost instead
            if i == len(vertex_path) - 1:
                # if using terminal heuristic cost:
                if not options.policy_use_zero_heuristic:
                    potential = graph.value_function_solution.GetSolution(vertex.potential)
                    if terminal_state is not None:
                        assert isinstance(vertex, GoalConditionedDualVertex)
                        if vertex.vertex_is_target:
                            prog.AddLinearConstraint( eq(last_x, terminal_state)) 
                        else:
                            def f_potential(x):
                                sub_x = potential.Substitute(
                                    {vertex.x[i]: x[i] for i in range(vertex.state_dim)}
                                )
                                sub_xt = sub_x.Substitute(
                                    {vertex.xt[i]: terminal_state[i] for i in range(vertex.state_dim)}
                                )
                                return sub_xt
                            prog.AddQuadraticCost(f_potential(last_x))
                    else:
                        f_potential = lambda x: potential.Substitute(
                            {vertex.x[i]: x[i] for i in range(vertex.state_dim)}
                        )
                        prog.AddQuadraticCost(f_potential(last_x))

                # assert that next control point is feasible -- for bezier curve continuity
                if not vertex.vertex_is_target:
                    prog.AddLinearConstraint(ge(vertex.B.dot(np.hstack(([1], last_x + last_delta))), 0))

            else:
                bezier_curve = [last_x]
                edge = graph.edges[get_edge_name(vertex.name, vertex_path[i + 1].name)]
                for j in range(1, options.num_control_points):
                    # add a new knot point
                    x_j = prog.NewContinuousVariables(vertex.state_dim)
                    bezier_curve.append(x_j)
                    # knot point inside a set
                    if j == options.num_control_points - 1:
                        # inside the intersection
                        prog.AddLinearConstraint(ge(edge.B_intersection.dot(np.hstack(([1], x_j))), 0))
                    else:
                        # inside the vertex
                        prog.AddLinearConstraint(ge(vertex.B.dot(np.hstack(([1], x_j))), 0))

                    # quadratic cost with previous point
                    if terminal_state is not None:
                        prog.AddQuadraticCost(edge.cost_function(last_x, x_j, terminal_state))
                    else:
                        prog.AddQuadraticCost(edge.cost_function(last_x, x_j))

                    # if the point is the first knot point in this set -- add the bezier continuity constraint
                    if j == 1 and last_delta is not None:
                        prog.AddLinearConstraint(eq(x_j - last_x, last_delta))

                    # we just added the last point, store last_delta
                    if j == options.num_control_points - 1:
                        last_delta = x_j - last_x
                    last_x = x_j
                # store the bezier curve
                bezier_curves.append(bezier_curve)

            # on all but the initial vertex:
            if i > 0:
                # add flow violation penalty.
                # only do so so far as we are using heuristic values
                if options.policy_add_violation_penalties and not options.policy_use_zero_heuristic:
                    edge = graph.edges[get_edge_name(vertex_path[i-1].name, vertex.name)]
                    vcost = graph.value_function_solution.GetSolution(edge.bidirectional_edge_violation)
                    vcost += graph.value_function_solution.GetSolution(vertex.total_flow_in_violation)
                    prog.AddLinearCost(vcost)

            #     NOTE: add the G term. 
            #     NOTE: this will make the problem non-convex
            #     TODO: is this is a hack or genuinly useful.
                if options.policy_add_G_term:
                    G_matrix = graph.value_function_solution.GetSolution(vertex.G_matrix)
                    def eval_G(x):
                        x_and_1 = np.hstack(([1], x))
                        return x_and_1.dot(G_matrix).dot(x_and_1)
                    prog.AddCost(eval_G(last_delta))
        all_bezier_curves.append(bezier_curves)

    timer.dt("just building", print_stuff=options.verbose_solve_times)
    # TODO: kinda nasty. how about instead i pass a solver constructor
    if options.policy_solver is None:
        solution = Solve(prog)
    else:
        solution = options.policy_solver().Solve(prog)
    timer.dt("just solving", print_stuff=options.verbose_solve_times)

    final_result = []
    if solution.is_success():
        # optimization_cost = solution.get_optimal_cost()
        for i, v_path in enumerate(vertex_paths):
            bezier_curves = all_bezier_curves[i]
            # add_edge_and_vertex_violations = options.policy_add_violation_penalties and not options.policy_use_zero_heuristic
            # add_terminal_heuristic = not options.policy_use_zero_heuristic
            bezier_solutions = [[solution.GetSolution(control_point) for control_point in bezier_curve] for bezier_curve in bezier_curves]
            # cost = get_path_cost(graph, v_path, bezier_solutions, False, True)
            # cost = get_path_cost(graph, v_path, bezier_solutions, add_edge_and_vertex_violations, add_terminal_heuristic)
            full_tuple = (v_path, bezier_solutions)
            final_result.append(full_tuple)
            # TODO: get the best one too
            # YAY([v.name for v in v_path], cost)
        # INFO("--------")
        return final_result
    else:
        WARN("failed to solve")
        if verbose_failure:
            diditwork(solution)
        return []
    



def solve_convex_restriction(
    graph: PolynomialDualGCS,
    vertex_path: T.List[DualVertex],
    state_now: npt.NDArray,
    state_last: npt.NDArray = None,
    options: ProgramOptions = None,
    verbose_failure:bool =False,
    terminal_state:npt.NDArray = None
) -> T.Tuple[float, T.List[T.List[npt.NDArray]]]:
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
    # initial state
    last_x = prog.NewContinuousVariables(vertex_path[0].state_dim)
    prog.AddLinearConstraint(eq(last_x, state_now))
    # previous direction of motion -- for bezier curve continuity
    last_delta = None
    if state_last is not None:
        last_delta = last_x - state_last

    bezier_curves = []
    # for every vertex:
    for i, vertex in enumerate(vertex_path):
        # it's the last vertex -- don't add a bezier curve; add terminal cost instead
        if i == len(vertex_path) - 1:
            # if using terminal heuristic cost:
            if not options.policy_use_zero_heuristic:
                potential = graph.value_function_solution.GetSolution(vertex.potential)
                if terminal_state is not None:
                    assert isinstance(vertex, GoalConditionedDualVertex)
                    if vertex.vertex_is_target:
                        prog.AddLinearConstraint( eq(last_x, terminal_state)) 
                    else:
                        def f_potential(x):
                            sub_x = potential.Substitute(
                                {vertex.x[i]: x[i] for i in range(vertex.state_dim)}
                            )
                            sub_xt = sub_x.Substitute(
                                {vertex.xt[i]: terminal_state[i] for i in range(vertex.state_dim)}
                            )
                            return sub_xt
                        prog.AddCost(f_potential(last_x))
                else:
                    f_potential = lambda x: potential.Substitute(
                        {vertex.x[i]: x[i] for i in range(vertex.state_dim)}
                    )
                    prog.AddCost(f_potential(last_x))

            # assert that next control point is feasible -- for bezier curve continuity
            if not vertex.vertex_is_target:
                prog.AddLinearConstraint(ge(vertex.B.dot(np.hstack(([1], last_x + last_delta))), 0))

        else:
            bezier_curve = [last_x]
            edge = graph.edges[get_edge_name(vertex.name, vertex_path[i + 1].name)]
            for j in range(1, options.num_control_points):
                # add a new knot point
                x_j = prog.NewContinuousVariables(vertex.state_dim)
                bezier_curve.append(x_j)
                # knot point inside a set
                if j == options.num_control_points - 1:
                    # inside the intersection
                    prog.AddLinearConstraint(ge(edge.B_intersection.dot(np.hstack(([1], x_j))), 0))
                else:
                    # inside the vertex
                    prog.AddLinearConstraint(ge(vertex.B.dot(np.hstack(([1], x_j))), 0))

                # quadratic cost with previous point
                if terminal_state is not None:
                    prog.AddQuadraticCost(edge.cost_function(last_x, x_j, terminal_state))
                else:
                    prog.AddQuadraticCost(edge.cost_function(last_x, x_j))

                # if the point is the first knot point in this set -- add the bezier continuity constraint
                if j == 1 and last_delta is not None:
                    prog.AddLinearConstraint(eq(x_j - last_x, last_delta))

                # we just added the last point, store last_delta
                if j == options.num_control_points - 1:
                    last_delta = x_j - last_x
                last_x = x_j
            # store the bezier curve
            bezier_curves.append(bezier_curve)

        # on all but the initial vertex:
        if i > 0:
            # add flow violation penalty.
            # only do so so far as we are using heuristic values
            if options.policy_add_violation_penalties and not options.policy_use_zero_heuristic:
                edge = graph.edges[get_edge_name(vertex_path[i-1].name, vertex.name)]
                vcost = graph.value_function_solution.GetSolution(edge.bidirectional_edge_violation)
                vcost += graph.value_function_solution.GetSolution(vertex.total_flow_in_violation)
                prog.AddLinearCost(vcost)

            # NOTE: add the G term. 
            # NOTE: this will make the problem non-convex
            # TODO: is this is a hack or genuinly useful.
            if options.policy_add_G_term:
                G_matrix = graph.value_function_solution.GetSolution(vertex.G_matrix)
                def eval_G(x):
                    x_and_1 = np.hstack(([1], x))
                    return x_and_1.dot(G_matrix).dot(x_and_1)
                prog.AddCost(eval_G(last_delta))

    if options.policy_solver is None:
        solution = Solve(prog)
    else:
        solution = options.policy_solver().Solve(prog)

    if solution.is_success():
        optimization_cost = solution.get_optimal_cost()
        # bezier_solutions = [solution.GetSolution(bezier_curve) for bezier_curve in bezier_curves]
        bezier_solutions = [[solution.GetSolution(control_point) for control_point in bezier_curve] for bezier_curve in bezier_curves]
        return optimization_cost, bezier_solutions
    else:
        if verbose_failure:
            diditwork(solution)
        return np.inf, []


# ---


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

    k = options.num_control_points
    if terminal_state is not None:
        gcs, vertices, pseudo_terminal_vertex = graph.export_a_gcs(terminal_state)
    else:
        gcs, vertices, pseudo_terminal_vertex = graph.export_a_gcs()
    
    # set initial vertex constraint
    start_vertex = vertices[vertex.name]
    first_point = get_kth_control_point(start_vertex.x(), 0, k)
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

    # solve
    timer = timeit()
    result = gcs.SolveShortestPath(
        start_vertex, pseudo_terminal_vertex, gcs_options
    )  # type: MathematicalProgramResult
    dt = timer.dt("just SolveShortestPath solve time", print_stuff=options.verbose_solve_times)
    assert result.is_success()
    cost = result.get_optimal_cost()

    edge_path = gcs.GetSolutionPath(start_vertex, pseudo_terminal_vertex, result)
    vertex_name_path = []
    value_path = []
    for e in edge_path[:-1]:
        vertex_name_path.append(e.u().name())
        full_curve = result.GetSolution(e.u().x())
        split_up_curve = full_curve.reshape( (k, vertex.state_dim) )
        value_path.append([x for x in split_up_curve])
    vertex_name_path.append(edge_path[-1].u().name())
    if terminal_state is not None:
        assert np.allclose(value_path[-1][-1], terminal_state), "terminal state isn't 0, why?"
        
    return dt, cost, value_path, [graph.vertices[name] for name in vertex_name_path]

