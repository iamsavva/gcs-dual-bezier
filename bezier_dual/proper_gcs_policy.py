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
from solve_restriction import solve_convex_restriction, get_optimal_path, get_path_cost, solve_parallelized_convex_restriction


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
) -> T.Tuple[T.List[T.List[npt.NDArray]], T.List[DualVertex]]:
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
            return None, None
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

    return full_path, vertex_path_so_far



def lookahead_with_backtracking_policy(
    gcs: PolynomialDualGCS,
    vertex: DualVertex,
    initial_state: npt.NDArray,
    initial_previous_state: npt.NDArray = None,
    options: ProgramOptions = None,
) -> T.Tuple[T.List[T.List[npt.NDArray]], T.List[DualVertex]]:
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
            return None, None
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
        INFO("solved the convex restriction", num_times_solved_convex_restriction, "times")

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
        return full_path, target_node.vertex_path_so_far
        
    else:
        WARN("no path from start vertex to target!")
        return None, None



def cheap_a_star_policy(
    gcs: PolynomialDualGCS,
    vertex: DualVertex,
    initial_state: npt.NDArray,
    initial_previous_state: npt.NDArray = None,
    options: ProgramOptions = None,
) -> T.Tuple[T.List[T.List[npt.NDArray]], T.List[DualVertex]]:
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
            # print(len(vertex_paths))
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
                else:
                    WARN("failed to solve")

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
        return full_path, target_node.vertex_path_so_far
        
    else:
        WARN("no path from start vertex to target!")
        return None, None
    

def cheap_a_star_policy_parallelized(
    gcs: PolynomialDualGCS,
    vertex: DualVertex,
    initial_state: npt.NDArray,
    initial_previous_state: npt.NDArray = None,
    options: ProgramOptions = None,
) -> T.Tuple[T.List[T.List[npt.NDArray]], T.List[DualVertex]]:
    """
    K-step lookahead rollout policy.
    If you reach a point from which no action is available --
    -- backtrack to the last state when some action was available.
    Returns a list of bezier curves. Each bezier curve is a list of control points (numpy arrays).
    """
    timer = timeit()
    if options is None:
        options = gcs.options
    options.vertify_options_validity()

    # cost, current state, last state, current vertex, state path so far, vertex path so far
    que = PriorityQueue()
    que.put( (0.0, Node(vertex, initial_state, initial_previous_state, [], [vertex]) ) )
    num_times_solved_convex_restriction = 0


    found_target = False
    target_node = None # type: Node

    timer.dt("start up")
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
            # print(len(vertex_paths))
            timer = timeit()
            solutions = solve_parallelized_convex_restriction(gcs, vertex_paths, node.state_now, node.state_last, options)
            timer.dt("solving")
            num_times_solved_convex_restriction += 1
            for (vertex_path, bezier_curves) in solutions:
                next_node = node.extend(bezier_curves[0], vertex_path[1])
                # evaluate the cost
                add_edge_and_vertex_violations = options.policy_add_violation_penalties and not options.policy_use_zero_heuristic
                add_terminal_heuristic = not options.policy_use_zero_heuristic
                cost_of_path = get_path_cost(gcs, next_node.vertex_path_so_far, next_node.bezier_path_so_far, add_edge_and_vertex_violations, add_terminal_heuristic)
                que.put( (cost_of_path, next_node) )
            timer.dt("quing")

    if options.policy_verbose_number_of_restrictions_solves:
        INFO("solved the convex restriction", num_times_solved_convex_restriction, "of times")


    if found_target:
        timer = timeit()
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
        timer.dt("one last solve")
        return full_path, target_node.vertex_path_so_far
        
    else:
        WARN("no path from start vertex to target!")
        return None, None
           

def obtain_rollout(
    gcs: PolynomialDualGCS,
    lookahead: int,
    vertex: DualVertex,
    state: npt.NDArray,
    last_state: npt.NDArray = None
) -> T.Tuple[T.List[T.List[npt.NDArray]], T.List[DualVertex], float]:
    options = gcs.options
    gcs.options.policy_lookahead = lookahead
    options.policy_lookahead = lookahead
    gcs.options.vertify_options_validity()

    timer = timeit()
    if options.use_lookahead_policy:
        rollout_path, v_path = lookahead_policy(gcs, vertex, state, last_state, options)
    elif options.use_lookahead_with_backtracking_policy:
        rollout_path, v_path = lookahead_with_backtracking_policy(gcs, vertex, state, last_state, options)
    elif options.use_cheap_a_star_policy:
        rollout_path, v_path = cheap_a_star_policy(gcs, vertex, state, last_state, options)
    elif options.use_cheap_a_star_policy_parallelized:
        rollout_path, v_path = cheap_a_star_policy_parallelized(gcs, vertex, state, last_state, options)
        
    dt = timer.dt(print_stuff = False)
    return rollout_path, v_path, dt


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
    verbose_comparison_to_optimal:bool = False
) -> T.Tuple[bool, float]:
    """
    rollout the policy from the initial condition, plot it out on a given figure`
    return whether the problem solved successfully + how long it took to solve for the tajectory.
    """
    # TODO: drop this, this is repeated
    options = gcs.options
    gcs.options.policy_lookahead = lookahead
    options.policy_lookahead = lookahead
    options.vertify_options_validity()

    rollout_path, rollout_v_path, dt = obtain_rollout(gcs, lookahead, vertex, state, last_state)

    if rollout_path is None:
        return False, dt
    
    if plot_control_points:
        plot_bezier(fig, rollout_path, rollout_color, rollout_color, name="rollout", linewidth=linewidth)
    else:
        plot_bezier(fig, rollout_path, rollout_color, None, name="rollout",linewidth=linewidth)

    if plot_optimal:
        options.policy_lookahead = optimal_lookahead
        optimal_dt, _, optimal_path, optimal_v_path = get_optimal_path(gcs, vertex, state, options)
        if verbose_comparison_to_optimal:
            rollout_cost = get_path_cost(gcs, rollout_v_path, rollout_path, False, True)
            optimal_cost = get_path_cost(gcs, optimal_v_path, optimal_path, False, True)
            INFO("policy.  time", np.round(dt,3), "cost", np.round(rollout_cost, 3))
            INFO("optimal. time", np.round(optimal_dt,3), "cost", np.round(optimal_cost, 3))
            INFO("----")

        if plot_control_points:
            plot_bezier(fig, optimal_path, optimal_color, optimal_color, name="optimal",linewidth=linewidth)
        else:
            plot_bezier(fig, optimal_path, optimal_color, None, name="optimal",linewidth=linewidth)
    return True, dt
