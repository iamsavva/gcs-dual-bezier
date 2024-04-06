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
from polynomial_dual_gcs_utils import (
    define_quadratic_polynomial,
    define_sos_constraint_over_polyhedron,
)

from bezier_dual import QUADRATIC_COST, PolynomialDualGCS, DualEdge, DualVertex
from plot_utils import plot_bezier


def get_all_n_step_paths(
    graph: PolynomialDualGCS, start_lookahead: int, start_vertex: DualVertex
) -> T.List[T.List[DualVertex]]:
    """
    find every n-step path from the current vertex.
    """
    paths = []  # type: T.List[T.List[DualVertex]]
    vertex_expand_que = deque([(start_vertex, [start_vertex], start_lookahead)])
    while len(vertex_expand_que) > 0:
        vertex, path, lookahead = vertex_expand_que.pop()  # type: DualVertex
        if lookahead == 0:
            paths.append(path)
        else:
            if vertex.vertex_is_target:
                paths.append(path)
            else:
                for edge_name in vertex.edges_out:
                    right_vertex = graph.edges[edge_name].right
                    vertex_expand_que.append((right_vertex, path + [right_vertex], lookahead - 1))
    return paths


# implement policy without revisits
def get_all_n_step_paths_no_revisits(
    graph: PolynomialDualGCS,
    start_lookahead: int,
    start_vertex: DualVertex,
    already_visited=T.List[DualVertex],
) -> T.List[T.List[DualVertex]]:
    """
    find every n-step path without revisits
    there isn't actually a way to incorporate that on the policy level. must add as constraint.
    there is a heuristic
    """
    paths = []  # type: T.List[T.List[DualVertex]]
    vertex_expand_que = deque([(start_vertex, [start_vertex], start_lookahead)])
    while len(vertex_expand_que) > 0:
        vertex, path, lookahead = vertex_expand_que.pop()
        # ran out of lookahead -- stop
        if lookahead == 0:
            paths.append(path)
        else:

            if vertex.vertex_is_target:
                paths.append(path)
            else:
                for edge_name in vertex.edges_out:
                    right_vertex = graph.edges[edge_name].right
                    # don't do revisits
                    if right_vertex not in path and right_vertex not in already_visited:
                        vertex_expand_que.append(
                            (right_vertex, path + [right_vertex], lookahead - 1)
                        )
    return paths


def get_all_n_step_vertices_and_edges(
    graph: PolynomialDualGCS, start_lookahead: int, start_vertex: DualVertex
) -> T.Tuple[T.List[DualVertex], T.List[DualEdge]]:
    """
    find every vertex and edge in a n step radius
    NOTE: this will prove useful when writing GCS controller
    """
    vertices = [start_vertex]  # type: T.List[DualVertex]
    edges = []  # type: T.List[DualEdge]
    vertex_expand_que = deque([(start_vertex, start_lookahead)])
    while len(vertex_expand_que) > 0:
        vertex, lookahead = vertex_expand_que.pop()  # type: DualVertex
        if lookahead > 0:
            if vertex.vertex_is_target:
                pass
            else:
                for edge_name in vertex.edges_out:
                    edge = graph.edges[edge_name]
                    right_vertex = edge.right
                    edges.append(edge)
                    if right_vertex not in vertices:
                        vertices.append(right_vertex)
                    if lookahead > 1:
                        vertex_expand_que.append((right_vertex, lookahead - 1))
    return vertices, edges



# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
# ---------------------------------------------------------------------


def solve_convex_restriction(
    graph: PolynomialDualGCS,
    vertex_path: T.List[DualVertex],
    state_now: npt.NDArray,
    state_last: npt.NDArray = None,
    options: ProgramOptions = None,
    verbose_failure=False,
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

    if options.solve_with_snopt:
        solver = SnoptSolver()
        solution = solver.Solve(prog)
    elif options.solve_with_mosek:
        solver = MosekSolver()
        solution = solver.Solve(prog)
    elif options.solve_with_gurobi:
        solver = GurobiSolver()
        solution = solver.Solve(prog)
    elif options.solve_with_osqp:
        solver = OsqpSolver()
        solution = solver.Solve(prog)
    elif options.solve_with_clarabel:
        solver = ClarabelSolver()
        solution = solver.Solve(prog)
    else:
        solution = Solve(prog)
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
    gcs: PolynomialDualGCS,
    vertex: DualVertex,
    state: npt.NDArray,
    options: ProgramOptions = None
) -> T.Tuple[float, float, T.List[T.List[npt.NDArray]], T.List[DualVertex]]:
    """
    return (time, cost, bezier path, vertex path) of the optimal solution.
    """
    if options is None:
        options = gcs.options

    k = options.num_control_points
    regular_gcs, vertices, terminal_vertex = gcs.export_a_gcs()
    start_vertex = vertices[vertex.name]
    first_point = get_kth_control_point(start_vertex.x(), 0, k)
    cons = eq(first_point, state)
    for con in cons:
        start_vertex.AddConstraint( con )

    gcs_options = GraphOfConvexSetsOptions()
    gcs_options.convex_relaxation = options.gcs_policy_use_convex_relaxation
    gcs_options.max_rounding_trials = options.gcs_policy_max_rounding_trials
    gcs_options.preprocessing = options.gcs_policy_use_preprocessing

    # solve
    timer = timeit()
    result = regular_gcs.SolveShortestPath(
        start_vertex, terminal_vertex, gcs_options
    )  # type: MathematicalProgramResult
    dt = timer.dt()
    assert result.is_success()
    cost = result.get_optimal_cost()

    edge_path = regular_gcs.GetSolutionPath(start_vertex, terminal_vertex, result)
    vertex_name_path = [vertex.name]
    value_path = []
    for e in edge_path[:-1]:
        vertex_name_path.append(e.u().name())
        full_curve = result.GetSolution(e.u().x())
        split_up_curve = full_curve.reshape( (k, vertex.state_dim) )
        value_path.append([x for x in split_up_curve])
        
    return dt, cost, value_path, [gcs.vertices[name] for name in vertex_name_path[:-1]]


def get_k_step_optimal_path(
    gcs: PolynomialDualGCS,
    vertex: DualVertex,
    state: npt.NDArray,
    last_state: npt.NDArray = None,
    options: ProgramOptions = None,
    already_visited: T.List[DualVertex] = [],
) -> T.Tuple[float, T.List[T.List[npt.NDArray]], T.List[DualVertex]]:
    """ 
    do not use this to compute optimal trajectories
    """
    if options is None:
        options = gcs.options
    # get all possible n-step paths from current vertex -- with or without revisits
    vertex_paths = get_all_n_step_paths_no_revisits(
        gcs, options.policy_lookahead, vertex, already_visited
    )
    # for every path -- solve convex restriction
    best_cost, best_path, best_vertex_path = np.inf, None, None
    for vertex_path in vertex_paths:
        cost, bezier_curves = solve_convex_restriction(gcs, vertex_path, state, last_state, options)
        if options.policy_verbose_choices:
            print([v.name for v in vertex_path], cost)
        # maintain the best path / choice
        if cost < best_cost:
            best_cost, best_path, best_vertex_path = cost, bezier_curves, vertex_path
    if options.policy_verbose_choices:
        print("----")
    return best_cost, best_path, best_vertex_path


def get_path_cost(
    graph: PolynomialDualGCS,
    vertex_path: T.List[DualVertex],
    bezier_path: T.List[T.List[npt.NDArray]],
    add_edge_and_vertex_violations:bool = True,
    add_terminal_heuristic:bool = True,
) -> float:
    """
    Note: this is cost of the full path with the terminal cost
    TODO: should i do the add G term heiuristic?
    """
    cost = 0.0
    for index, bezier_curve in enumerate(bezier_path):
        edge = graph.edges[get_edge_name(vertex_path[index].name, vertex_path[index + 1].name)]
        for i in range(len(bezier_curve) - 1):
            cost += edge.cost_function(bezier_curve[i], bezier_curve[i + 1])
        if add_edge_and_vertex_violations:
            violations = graph.value_function_solution.GetSolution(edge.bidirectional_edge_violation) + graph.value_function_solution.GetSolution(edge.right.total_flow_in_violation)
            if isinstance(violations, Expression):
                violations = violations.Evaluate()
            cost += violations
        if add_terminal_heuristic and (index == len(bezier_path) - 1):
            cost += vertex_path[-1].cost_at_point(bezier_curve[-1], graph.value_function_solution)
    return cost

class Node:
    def __init__(self, vertex_now: DualVertex, state_now:npt.NDArray, state_last:npt.NDArray, bezier_path_so_far:T.List[T.List[npt.NDArray]], vertex_path_so_far:T.List[DualVertex]):
        self.vertex_now = vertex_now
        self.state_now = state_now
        self.state_last = state_last
        self.bezier_path_so_far = bezier_path_so_far
        self.vertex_path_so_far = vertex_path_so_far

    def extend(self, next_bezier_curve: T.List[npt.NDArray], next_vertex: DualVertex) -> "Node":
        return Node(next_vertex, 
                    next_bezier_curve[-1], 
                    next_bezier_curve[-2], 
                    self.bezier_path_so_far + [next_bezier_curve], 
                    self.vertex_path_so_far + [next_vertex]
                    )

def lookahead_policy(
    gcs: PolynomialDualGCS,
    vertex: DualVertex,
    initial_state: npt.NDArray,
    initial_previous_state: npt.NDArray = None,
    options: ProgramOptions = None,
) -> T.List[T.List[npt.NDArray]]:
    """
    K-step lookahead rollout policy.
    Returns a list of bezier curves. Each bezier curve is a list of control points (numpy arrays).
    """
    if options is None:
        options = gcs.options
    options.vertify_options_validity()
    vertex_now, state_now, state_last = vertex, initial_state, initial_previous_state

    full_path = []  # type: T.List[T.List[npt.NDarray]]
    vertex_path_so_far = [vertex_now]  # type: T.List[DualVertex]

    while not vertex_now.vertex_is_target:
        # use a k-step lookahead to obtain optimal k-step lookahead path
        _, bezier_path, vertex_path = get_k_step_optimal_path(
            gcs,
            vertex_now,
            state_now,
            state_last,
            options,
            already_visited=vertex_path_so_far,
        )
        if bezier_path is None:
            WARN("k-step optimal path couldn't find a solution")
            return None
        # take just the first action from that path, then repeat
        first_segment = bezier_path[0]
        full_path.append(first_segment)
        vertex_now, state_now, state_last = (
            vertex_path[1],
            first_segment[-1],
            first_segment[-2],
        )
        vertex_path_so_far.append(vertex_now)

    if options.verbose_restriction_improvement:
        cost_before = get_path_cost(gcs, vertex_path_so_far, full_path, False, True)

    # solve a convex restriction on the vertex sequence
    if options.postprocess_by_solving_restriction_on_mode_sequence:
        _, full_path = solve_convex_restriction(gcs, vertex_path_so_far, initial_state, initial_previous_state, options)
        # verbose
        if options.verbose_restriction_improvement:
            cost_after = get_path_cost(gcs, vertex_path_so_far, full_path, False, True)
            INFO(
                "path cost improved from",
                np.round(cost_before, 1),
                "to",
                np.round(cost_after, 1),
                "; original is",
                np.round((cost_before / cost_after - 1) * 100, 1),
                "% worse",
            )

    return full_path



def lookahead_with_backtracking_policy(
    gcs: PolynomialDualGCS,
    vertex: DualVertex,
    initial_state: npt.NDArray,
    initial_previous_state: npt.NDArray = None,
    options: ProgramOptions = None,
) -> T.List[T.List[npt.NDArray]]:
    """
    K-step lookahead rollout policy.
    If you reach a point from which no action is available --
    -- backtrack to the last state when some action was available.
    Returns a list of bezier curves. Each bezier curve is a list of control points (numpy arrays).
    """
    if options is None:
        options = gcs.options
    options.vertify_options_validity()

    # cost, current state, last state, current vertex, state path so far, vertex path so far
    decision_options = [ PriorityQueue() ]
    decision_options[0].put( (0, Node(vertex, initial_state, initial_previous_state, [], [vertex])) )
    num_times_solved_convex_restriction = 0


    decision_index = 0
    found_target = False
    target_node = None

    while not found_target:
        if decision_index == -1:
            return None
        if decision_options[decision_index].empty():
            decision_index -= 1
        else:
            node = decision_options[decision_index].get()[1] # type: Node
            if node.vertex_now.vertex_is_target:
                found_target = True
                target_node = node
                break

            if len(decision_options) == decision_index + 1:
                decision_options.append( PriorityQueue() )

            vertex_paths = get_all_n_step_paths_no_revisits(
                gcs, options.policy_lookahead, node.vertex_now, node.vertex_path_so_far
            )
            # for every path -- solve convex restriction, add next states
            for vertex_path in vertex_paths:
                cost, bezier_curves = solve_convex_restriction(gcs, vertex_path, node.state_now, node.state_last, options)
                num_times_solved_convex_restriction += 1
                if np.isfinite(cost):
                    next_node = node.extend(bezier_curves[0], vertex_path[1])
                    decision_options[decision_index + 1].put( (cost, next_node ))
            decision_index += 1

    if options.policy_verbose_number_of_restrictions_solves:
        INFO("solved the convex restriction", num_times_solved_convex_restriction, "of times")

    if found_target:
        if options.verbose_restriction_improvement:
            cost_before = get_path_cost(gcs, target_node.vertex_path_so_far, target_node.bezier_path_so_far, False, True)

        full_path = target_node.bezier_path_so_far
        # solve a convex restriction on the vertex sequence
        if options.postprocess_by_solving_restriction_on_mode_sequence:
            _, full_path = solve_convex_restriction(gcs, target_node.vertex_path_so_far, initial_state, initial_previous_state, options)
            # verbose
            if options.verbose_restriction_improvement:
                cost_after = get_path_cost(gcs, target_node.vertex_path_so_far, full_path, False, True)
                INFO(
                    "path cost improved from",
                    np.round(cost_before, 1),
                    "to",
                    np.round(cost_after, 1),
                    "; original is",
                    np.round((cost_before / cost_after - 1) * 100, 1),
                    "% worse",
                )
        return full_path
        
    else:
        WARN("no path from start vertex to target!")
        return None
    

def cheap_a_star_policy(
    gcs: PolynomialDualGCS,
    vertex: DualVertex,
    initial_state: npt.NDArray,
    initial_previous_state: npt.NDArray = None,
    options: ProgramOptions = None,
) -> T.List[T.List[npt.NDArray]]:
    """
    K-step lookahead rollout policy.
    If you reach a point from which no action is available --
    -- backtrack to the last state when some action was available.
    Returns a list of bezier curves. Each bezier curve is a list of control points (numpy arrays).
    """
    if options is None:
        options = gcs.options
    options.vertify_options_validity()

    # cost, current state, last state, current vertex, state path so far, vertex path so far
    que = PriorityQueue()
    que.put( (0.0, Node(vertex, initial_state, initial_previous_state, [], [vertex]) ) )
    num_times_solved_convex_restriction = 0


    found_target = False
    target_node = None # type: Node

    while not found_target:
        
        node = que.get()[1] # type: Node
        if node.vertex_now.vertex_is_target:
            found_target = True
            target_node = node
            break
        else:
            vertex_paths = get_all_n_step_paths_no_revisits(
                gcs, options.policy_lookahead, node.vertex_now, node.vertex_path_so_far
            )
            # for every path -- solve convex restriction, add next states
            for vertex_path in vertex_paths:
                cost, bezier_curves = solve_convex_restriction(gcs, vertex_path, node.state_now, node.state_last, options)
                num_times_solved_convex_restriction += 1
                # check if solution exists
                if np.isfinite(cost):
                    next_node = node.extend(bezier_curves[0], vertex_path[1])
                    # evaluate the cost
                    add_edge_and_vertex_violations = options.policy_add_violation_penalties and not options.policy_use_zero_heuristic
                    add_terminal_heuristic = not options.policy_use_zero_heuristic
                    cost_of_path = get_path_cost(gcs, next_node.vertex_path_so_far, next_node.bezier_path_so_far, add_edge_and_vertex_violations, add_terminal_heuristic)
                    que.put( (cost_of_path, next_node) )

    if options.policy_verbose_number_of_restrictions_solves:
        INFO("solved the convex restriction", num_times_solved_convex_restriction, "of times")


    if found_target:
        if options.verbose_restriction_improvement:
            cost_before = get_path_cost(gcs, target_node.vertex_path_so_far, target_node.bezier_path_so_far, False, True)

        full_path = target_node.bezier_path_so_far
        # solve a convex restriction on the vertex sequence
        if options.postprocess_by_solving_restriction_on_mode_sequence:
            _, full_path = solve_convex_restriction(gcs, target_node.vertex_path_so_far, initial_state, initial_previous_state, options)
            # verbose
            if options.verbose_restriction_improvement:
                cost_after = get_path_cost(gcs, target_node.vertex_path_so_far, full_path, False, True)
                INFO(
                    "path cost improved from",
                    np.round(cost_before, 1),
                    "to",
                    np.round(cost_after, 1),
                    "; original is",
                    np.round((cost_before / cost_after - 1) * 100, 1),
                    "% worse",
                )
        return full_path
        
    else:
        WARN("no path from start vertex to target!")
        return None
           

def plot_optimal_and_rollout(
    fig: go.Figure,
    gcs: PolynomialDualGCS,
    lookahead: int,
    vertex: DualVertex,
    state: npt.NDArray,
    last_state: npt.NDArray = None,
    plot_optimal:bool=True,
    optimal_lookahead:int=10,
    rollout_color:str="red",
    optimal_color:str="blue",
    plot_control_points:bool=True,
    linewidth:int=3,
    verbose_time_comparison_to_optimal:bool = False
) -> T.Tuple[bool, float]:
    """
    rollout the policy from the initial condition, plot it out on a given figure`
    return whether the problem solved successfully + how long it took to solve for the tajectory.
    """
    options = gcs.options
    gcs.options.policy_lookahead = lookahead
    options.policy_lookahead = lookahead
    options.vertify_options_validity()

    timer = timeit()
    if options.use_lookahead_policy:
        rollout_path = lookahead_policy(gcs, vertex, state, last_state, options)
    elif options.use_lookahead_with_backtracking_policy:
        rollout_path = lookahead_with_backtracking_policy(gcs, vertex, state, last_state, options)
    elif options.use_cheap_a_star_policy:
        rollout_path = cheap_a_star_policy(gcs, vertex, state, last_state, options)
    dt = timer.dt(print_stuff = False)

    if rollout_path is None:
        return False, dt
    
    if plot_control_points:
        plot_bezier(fig, rollout_path, rollout_color, rollout_color, name="rollout", linewidth=linewidth)
    else:
        plot_bezier(fig, rollout_path, rollout_color, None, name="rollout",linewidth=linewidth)

    if plot_optimal:
        options.policy_lookahead = optimal_lookahead
        optimal_dt, _, optimal_path, _ = get_optimal_path(gcs, vertex, state, options)
        if verbose_time_comparison_to_optimal:
            INFO("policy", dt)
            INFO("GCS   ", optimal_dt)
        if plot_control_points:
            plot_bezier(fig, optimal_path, optimal_color, optimal_color, name="optimal",linewidth=linewidth)
        else:
            plot_bezier(fig, optimal_path, optimal_color, None, name="optimal",linewidth=linewidth)
    return True, dt
