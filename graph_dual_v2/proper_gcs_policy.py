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

# from gcs_util import get_edge_name, make_quadratic_cost_function_matrices, plot_a_gcs

from gcs_dual import PolynomialDualGCS, DualEdge, DualVertex
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


def get_k_step_optimal_path(
    graph: PolynomialDualGCS,
    vertex: DualVertex,
    state: npt.NDArray,
    options: ProgramOptions = None,
    already_visited: T.List[DualVertex] = [],
    target_state: npt.NDArray = None,
) -> T.Tuple[float, T.List[T.List[npt.NDArray]], T.List[DualVertex]]:
    """ 
    do not use this to compute optimal trajectories
    """
    if options is None:
        options = graph.options

    if target_state is None:
        assert vertex.target_set_type is Point, "target set not passed when target set not a point"
        target_state = vertex.target_convex_set.x()

    # get all possible n-step paths from current vertex -- with or without revisits
    if options.allow_vertex_revisits:
        vertex_paths = get_all_n_step_paths(graph, options.policy_lookahead, vertex)
    else:
        vertex_paths = get_all_n_step_paths_no_revisits(
            graph, options.policy_lookahead, vertex, already_visited
        )

    # for every path -- solve convex restriction, add next states
    timer = timeit()
    solutions = solve_parallelized_convex_restriction(graph, vertex_paths, state, options, target_state=target_state, one_last_solve=False)
    timer.dt("solving", print_stuff = options.verbose_solve_times)
    
    best_cost, best_path, best_vertex_path = np.inf, None, None
    if solutions is None:
        return best_cost, best_path, best_vertex_path
    for (vertex_path, bezier_curves) in solutions:
        cost = get_path_cost(graph, vertex_path, bezier_curves, False, True, target_state=target_state)
        if cost < best_cost:
            best_cost, best_path, best_vertex_path = cost, bezier_curves, vertex_path
    timer.dt("finding best", print_stuff = options.verbose_solve_times)
            
    return best_cost, best_path, best_vertex_path


def postprocess_the_path(graph:PolynomialDualGCS, 
                          vertex_path_so_far: T.List[DualVertex], 
                          full_path: T.List[T.List[npt.NDArray]], 
                          initial_state:npt.NDArray, 
                          options:ProgramOptions = None, 
                          target_state:npt.NDArray = None):
    if options is None:
        options = graph.options
    if options.verbose_restriction_improvement:
        cost_before = get_path_cost(graph, vertex_path_so_far, full_path, False, True, target_state=target_state)
    timer = timeit()
    # solve a convex restriction on the vertex sequence
    if options.postprocess_by_solving_restriction_on_mode_sequence:
        # print("postprocessing")
        full_path = solve_convex_restriction(graph, vertex_path_so_far, initial_state, options, target_state=target_state, one_last_solve = True)
        # verbose
        if options.verbose_restriction_improvement:
            cost_after = get_path_cost(graph, vertex_path_so_far, full_path, False, True, target_state=target_state)
            INFO(
                "path cost improved from",
                np.round(cost_before, 1),
                "to",
                np.round(cost_after, 1),
                "; original is",
                np.round((cost_before / cost_after - 1) * 100, 1),
                "% worse",
            )
    timer.dt("one last solve", print_stuff = options.verbose_solve_times)
    return full_path

class Node:
    def __init__(self, vertex_now: DualVertex, state_now:npt.NDArray, bezier_path_so_far:T.List[T.List[npt.NDArray]], vertex_path_so_far:T.List[DualVertex]):
        self.vertex_now = vertex_now
        self.state_now = state_now
        self.bezier_path_so_far = bezier_path_so_far
        self.vertex_path_so_far = vertex_path_so_far

    def extend(self, next_bezier_curve: T.List[npt.NDArray], next_vertex: DualVertex) -> "Node":
        next_point = next_bezier_curve[-1]
        if not next_vertex.convex_set.PointInSet(next_point):
            # point not in set, need to project due to bad numerics
            next_point = next_vertex.convex_set.Projection(next_point)[1].flatten()

        return Node(next_vertex, 
                    next_point,
                    self.bezier_path_so_far + [next_bezier_curve], 
                    self.vertex_path_so_far + [next_vertex]
                    )
    
def get_lookahead_cost(
    graph: PolynomialDualGCS,
    lookahead:int,
    vertex: DualVertex,
    initial_state: npt.NDArray,
    options: ProgramOptions = None,
    target_state: npt.NDArray = None,
) -> float:
    """
    K-step lookahead rollout policy.
    Returns a list of bezier curves. Each bezier curve is a list of control points (numpy arrays).
    """
    if options is None:
        options = graph.options
    options.policy_lookahead = lookahead
    graph.options.policy_lookahead = lookahead
    options.vertify_options_validity()
    if target_state is None:
        assert vertex.target_set_type is Point, "target set not passed when target set not a point"
        target_state = vertex.target_convex_set.x()

    # use a k-step lookahead to obtain optimal k-step lookahead path
    return get_k_step_optimal_path(
        graph,
        vertex,
        initial_state,
        options,
        already_visited=[vertex],
        target_state = target_state,
    )[0]

def lookahead_policy(
    graph: PolynomialDualGCS,
    vertex: DualVertex,
    initial_state: npt.NDArray,
    options: ProgramOptions = None,
    target_state: npt.NDArray = None,
) -> T.Tuple[T.List[T.List[npt.NDArray]], T.List[DualVertex]]:
    """
    K-step lookahead rollout policy.
    Returns a list of bezier curves. Each bezier curve is a list of control points (numpy arrays).
    """
    # INFO("running lookahead")
    if options is None:
        options = graph.options
    options.vertify_options_validity()
    vertex_now, state_now = vertex, initial_state

    if target_state is None:
        assert vertex.target_set_type is Point, "target set not passed when target set not a point"
        target_state = vertex.target_convex_set.x()

    full_trajectory = []  # type: T.List[T.List[npt.NDarray]]
    vertex_path_so_far = [vertex_now]  # type: T.List[DualVertex]

    num_iterations = 0

    while not vertex_now.vertex_is_target:
        # print([v.name for v in vertex_path_so_far])
        # print(full_trajectory)
        # use a k-step lookahead to obtain optimal k-step lookahead path
        _, traj, vertex_path = get_k_step_optimal_path(
            graph,
            vertex_now,
            state_now,
            options,
            already_visited=vertex_path_so_far,
            target_state = target_state,
        )
        if traj is None:
            WARN("k-step optimal path couldn't find a solution", initial_state)
            ERROR(vertex_now.name, state_now, vertex_now.convex_set.PointInSet(state_now))
            return None, None
        # take just the first action from that path, then repeat
        vertex_now = vertex_path[1]
        state_now = vertex_now.convex_set.Projection(traj[1])[1].flatten()
        vertex_path_so_far.append(vertex_now)
        full_trajectory.append(state_now)

        num_iterations += 1
        if num_iterations > options.forward_iteration_limit:
            WARN("exceeded number of fowrard iterations")
            return full_trajectory, vertex_path_so_far

    full_trajectory = postprocess_the_path(graph, vertex_path_so_far, full_trajectory, initial_state, options, target_state)

    return full_trajectory, vertex_path_so_far


def lookahead_with_backtracking_policy(
    graph: PolynomialDualGCS,
    vertex: DualVertex,
    initial_state: npt.NDArray,
    options: ProgramOptions = None,
    target_state: npt.NDArray = None,
) -> T.Tuple[T.List[T.List[npt.NDArray]], T.List[DualVertex]]:
    """
    K-step lookahead rollout policy.
    If you reach a point from which no action is available --
    -- backtrack to the last state when some action was available.
    Returns a list of bezier curves. Each bezier curve is a list of control points (numpy arrays).
    """
    INFO("running lookahead backtracking")
    if options is None:
        options = graph.options
    options.vertify_options_validity()

    if target_state is None:
        assert vertex.target_set_type is Point, "target set not passed when target set not a point"
        target_state = vertex.target_convex_set.x()

    # cost, current state, last state, current vertex, state path so far, vertex path so far
    decision_options = [ PriorityQueue() ]
    decision_options[0].put( (0, Node(vertex, initial_state, [], [vertex])) )
    num_times_solved_convex_restriction = 0


    decision_index = 0
    found_target = False
    target_node = None
    number_of_iterations = 0

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
                
            if options.allow_vertex_revisits:
                vertex_paths = get_all_n_step_paths(
                    graph, options.policy_lookahead, node.vertex_now
                )
            else:
                vertex_paths = get_all_n_step_paths_no_revisits(
                    graph, options.policy_lookahead, node.vertex_now, node.vertex_path_so_far
                )
            # print("vertex paths at vertex ", node.vertex_now.name)
            # for v_path in vertex_paths:
            #     print([v.name for v in v_path])

            # for every path -- solve convex restriction, add next states
            number_of_iterations += 1
            if number_of_iterations >= options.forward_iteration_limit:
                WARN("exceeded number of forward iterations")
                decision_index = -1
                return None, None
            
            for vertex_path in vertex_paths:
                bezier_curves = solve_convex_restriction(graph, vertex_path, node.state_now, options, target_state=target_state, one_last_solve=False)
                # print(node.state_now, node.vertex_now.convex_set.PointInSet(node.state_now))
                # print([v.name for v in vertex_path])
                # print(bezier_curves)
                num_times_solved_convex_restriction += 1
                if bezier_curves is not None:
                    add_edge_and_vertex_violations = False
                    add_target_heuristic = not options.policy_use_zero_heuristic
                    next_node = node.extend(bezier_curves[0], vertex_path[1])
                    cost_of_path = get_path_cost(graph, next_node.vertex_path_so_far, next_node.bezier_path_so_far, add_edge_and_vertex_violations, add_target_heuristic, target_state=target_state)
                    try:
                        decision_options[decision_index + 1].put( (cost_of_path, next_node ))
                    except:
                        print(cost_of_path, next_node)
            # print("------ next")
            decision_index += 1
            # TODO: make parallelized problem to be always feasible

            # for every path -- solve convex restriction, add next states
            # print(len(vertex_paths))
            # timer = timeit()
            # solutions = solve_parallelized_convex_restriction(graph, vertex_paths, node.state_now, options, target_state=target_state, one_last_solve=False)
            # timer.dt("solving", print_stuff = options.verbose_solve_times)
            # num_times_solved_convex_restriction += 1
            # for (vertex_path, bezier_curves) in solutions:
            #     next_node = node.extend(bezier_curves[0], vertex_path[1])
            #     # evaluate the cost
            #     add_edge_and_vertex_violations = options.policy_add_violation_penalties and not options.policy_use_zero_heuristic
            #     add_target_heuristic = not options.policy_use_zero_heuristic
            #     cost_of_path = get_path_cost(graph, next_node.vertex_path_so_far, next_node.bezier_path_so_far, add_edge_and_vertex_violations, add_target_heuristic, target_state=target_state)
            #     decision_options[decision_index + 1].put( (cost_of_path, next_node ))
            # timer.dt("quing", print_stuff = options.verbose_solve_times)

    if options.policy_verbose_number_of_restrictions_solves:
        INFO("solved the convex restriction", num_times_solved_convex_restriction, "times")

    if found_target:
        full_path = postprocess_the_path(graph, target_node.vertex_path_so_far, target_node.bezier_path_so_far, initial_state, options, target_state)
        return full_path, target_node.vertex_path_so_far
        
    else:
        WARN("no path from start vertex to target!")
        return None, None



def cheap_a_star_policy(
    graph: PolynomialDualGCS,
    vertex: DualVertex,
    initial_state: npt.NDArray,
    options: ProgramOptions = None,
    target_state: npt.NDArray = None,
) -> T.Tuple[T.List[T.List[npt.NDArray]], T.List[DualVertex]]:
    """
    K-step lookahead rollout policy.
    If you reach a point from which no action is available --
    -- backtrack to the last state when some action was available.
    Returns a list of bezier curves. Each bezier curve is a list of control points (numpy arrays).
    """
    if options is None:
        options = graph.options
    options.vertify_options_validity()

    if target_state is None:
        assert vertex.target_set_type is Point, "target set not passed when target set not a point"
        target_state = vertex.target_convex_set.x()

    # cost, current state, last state, current vertex, state path so far, vertex path so far
    que = PriorityQueue()
    que.put( (0.0, Node(vertex, initial_state, [], [vertex]) ) )
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
            if options.allow_vertex_revisits:
                vertex_paths = get_all_n_step_paths(
                    graph, options.policy_lookahead, node.vertex_now
                )
            else:
                vertex_paths = get_all_n_step_paths_no_revisits(
                    graph, options.policy_lookahead, node.vertex_now, node.vertex_path_so_far
                )
            # for every path -- solve convex restriction, add next states
            # print(len(vertex_paths))
            for vertex_path in vertex_paths:
                bezier_curves = solve_convex_restriction(graph, vertex_path, node.state_now, options, target_state=target_state, one_last_solve=False)
                num_times_solved_convex_restriction += 1
                # check if solution exists
                if bezier_curves is not None:
                    next_node = node.extend(bezier_curves[0], vertex_path[1])
                    # evaluate the cost
                    add_edge_and_vertex_violations = False
                    add_target_heuristic = not options.policy_use_zero_heuristic
                    cost_of_path = get_path_cost(graph, next_node.vertex_path_so_far, next_node.bezier_path_so_far, add_edge_and_vertex_violations, add_target_heuristic, target_state=target_state)
                    que.put( (cost_of_path, next_node) )
                else:
                    WARN("failed to solve")

    if options.policy_verbose_number_of_restrictions_solves:
        INFO("solved the convex restriction", num_times_solved_convex_restriction, "of times")


    if found_target:
        full_path = postprocess_the_path(graph, target_node.vertex_path_so_far, target_node.bezier_path_so_far, initial_state, options, target_state)
        return full_path, target_node.vertex_path_so_far
        
    else:
        WARN("no path from start vertex to target!")
        return None, None
    

def cheap_a_star_policy_parallelized(
    graph: PolynomialDualGCS,
    vertex: DualVertex,
    initial_state: npt.NDArray,
    options: ProgramOptions = None,
    target_state: npt.NDArray = None,
) -> T.Tuple[T.List[T.List[npt.NDArray]], T.List[DualVertex]]:
    """
    K-step lookahead rollout policy.
    If you reach a point from which no action is available --
    -- backtrack to the last state when some action was available.
    Returns a list of bezier curves. Each bezier curve is a list of control points (numpy arrays).
    """
    timer = timeit()
    if options is None:
        options = graph.options
    options.vertify_options_validity()

    # cost, current state, last state, current vertex, state path so far, vertex path so far
    que = PriorityQueue()
    que.put( (0.0, Node(vertex, initial_state, [], [vertex]) ) )
    num_times_solved_convex_restriction = 0

    if target_state is None:
        assert vertex.target_set_type is Point, "target set not passed when target set not a point"
        target_state = vertex.target_convex_set.x()


    found_target = False
    target_node = None # type: Node

    timer.dt("start up", print_stuff = options.verbose_solve_times)
    while not found_target:
        
        node = que.get()[1] # type: Node
        if node.vertex_now.vertex_is_target:
            found_target = True
            target_node = node
            break
        else:
            if options.allow_vertex_revisits:
                vertex_paths = get_all_n_step_paths(
                    graph, options.policy_lookahead, node.vertex_now
                )
            else:
                vertex_paths = get_all_n_step_paths_no_revisits(
                    graph, options.policy_lookahead, node.vertex_now, node.vertex_path_so_far
                )
            # for every path -- solve convex restriction, add next states
            # print(len(vertex_paths))
            timer = timeit()
            solutions = solve_parallelized_convex_restriction(graph, vertex_paths, node.state_now, options, target_state=target_state, one_last_solve=False)
            timer.dt("solving", print_stuff = options.verbose_solve_times)
            num_times_solved_convex_restriction += 1
            for (vertex_path, bezier_curves) in solutions:
                next_node = node.extend(bezier_curves[0], vertex_path[1])
                # evaluate the cost
                add_edge_and_vertex_violations = False
                add_target_heuristic = not options.policy_use_zero_heuristic
                cost_of_path = get_path_cost(graph, next_node.vertex_path_so_far, next_node.bezier_path_so_far, add_edge_and_vertex_violations, add_target_heuristic, target_state=target_state)
                que.put( (cost_of_path, next_node) )
            timer.dt("quing", print_stuff = options.verbose_solve_times)

    if options.policy_verbose_number_of_restrictions_solves:
        INFO("solved the convex restriction", num_times_solved_convex_restriction, "of times")


    if found_target:
        full_path = postprocess_the_path(graph, target_node.vertex_path_so_far, target_node.bezier_path_so_far, initial_state, options, target_state)
        return full_path, target_node.vertex_path_so_far
        
    else:
        WARN("no path from start vertex to target!")
        return None, None
           

def obtain_rollout(
    graph: PolynomialDualGCS,
    lookahead: int,
    vertex: DualVertex,
    state: npt.NDArray,
    target_state: npt.NDArray = None,
) -> T.Tuple[T.List[T.List[npt.NDArray]], T.List[DualVertex], float]:
    options = graph.options
    graph.options.policy_lookahead = lookahead
    options.policy_lookahead = lookahead
    graph.options.vertify_options_validity()

    if target_state is None:
        assert vertex.target_set_type is Point, "target set not passed when target set not a point"
        target_state = vertex.target_convex_set.x()
    

    timer = timeit()
    if options.use_lookahead_policy:
        rollout_path, v_path = lookahead_policy(graph, vertex, state, options, target_state)
    elif options.use_lookahead_with_backtracking_policy:
        rollout_path, v_path = lookahead_with_backtracking_policy(graph, vertex, state, options, target_state)
    elif options.use_cheap_a_star_policy:
        rollout_path, v_path = cheap_a_star_policy(graph, vertex, state, options, target_state)
    elif options.use_cheap_a_star_policy_parallelized:
        rollout_path, v_path = cheap_a_star_policy_parallelized(graph, vertex, state, options, target_state)
    else:
        raise Exception("not selected policy")
        
    dt = timer.dt(print_stuff = False)
    return rollout_path, v_path, dt


def get_statistics(
    graph: PolynomialDualGCS,
    lookahead: int,
    vertex: DualVertex,
    state: npt.NDArray,
    get_optimal: True,
    verbose_comparison_to_optimal:bool = False,
    target_state: npt.NDArray = None) -> T.Tuple[bool, float, float, float, float]:
    options = graph.options
    graph.options.policy_lookahead = lookahead
    options.policy_lookahead = lookahead
    options.vertify_options_validity()

    rollout_path, rollout_v_path, rollout_dt = obtain_rollout(graph, lookahead, vertex, state, target_state)
    rollout_cost = get_path_cost(graph, rollout_v_path, rollout_path, False, True, target_state)
    if rollout_path is None:
        WARN("faile to solve for ", state)
        return False, None, None, None, None
    if verbose_comparison_to_optimal:
            INFO("policy.  time", np.round(rollout_dt,3), "cost", np.round(rollout_cost, 3))
    
    if get_optimal:
        optimal_dt, _, optimal_path, optimal_v_path = get_optimal_path(graph, vertex, state, options, target_state)
        optimal_cost = get_path_cost(graph, optimal_v_path, optimal_path, False, True, target_state)

        if verbose_comparison_to_optimal:
            INFO("optimal.  time", np.round(optimal_dt,3), "cost", np.round(optimal_cost, 3))
        
        return True, rollout_cost, rollout_dt, optimal_cost, optimal_dt
    else:
        return True, rollout_cost, rollout_dt, None, None


def plot_optimal_and_rollout(
    fig: go.Figure,
    graph: PolynomialDualGCS,
    lookahead: int,
    vertex: DualVertex,
    state: npt.NDArray,
    rollout_color:str="red",
    optimal_color:str="blue",
    plot_control_points:bool=True,
    plot_start_point:bool = True,
    linewidth:int=3,
    marker_size:int=3,
    verbose_comparison:bool = False,
    target_state: npt.NDArray = None,
    optimal_name = "optimal",
    rollout_name = "rollout"
) -> T.Tuple[bool, float]:
    """
    rollout the policy from the initial condition, plot it out on a given figure`
    return whether the problem solved successfully + how long it took to solve for the tajectory.
    """

    success, rollout_time = plot_rollout(fig,graph, lookahead, vertex, state, rollout_color, plot_control_points, plot_start_point, linewidth, marker_size, verbose_comparison,target_state, rollout_name, True, False)

    plot_optimal(fig, graph, vertex, state, optimal_color, plot_control_points, plot_start_point, linewidth, marker_size, verbose_comparison, target_state, optimal_name, False, False)
    return success, rollout_time
    


def plot_optimal(
    fig: go.Figure,
    graph: PolynomialDualGCS,
    vertex: DualVertex,
    state: npt.NDArray,
    optimal_color:str="blue",
    plot_control_points:bool=True,
    plot_start_point:bool = True,
    linewidth:int=3,
    marker_size:int=3,
    verbose_comparison:bool = False,
    target_state: npt.NDArray = None,
    optimal_name = "optimal",
    dotted=False,
    plot_start_target_only=False
) -> T.Tuple[bool, float]:
    """
    rollout the policy from the initial condition, plot it out on a given figure`
    return whether the problem solved successfully + how long it took to solve for the tajectory.
    """
    # TODO: drop this, this is repeated
    options = graph.options
    options.vertify_options_validity()

    if target_state is None:
        assert vertex.target_set_type is Point, "target set not passed into plot rollout when target set not a point"
        target_state = vertex.target_convex_set.x()

    optimal_dt, _, optimal_path, optimal_v_path = get_optimal_path(graph, vertex, state, options, target_state)
    if verbose_comparison:
        optimal_cost = get_path_cost(graph, optimal_v_path, optimal_path, False, True, target_state)
        INFO("optimal. time", np.round(optimal_dt,3), "cost", np.round(optimal_cost, 3))
        INFO("----")

    if plot_control_points:
        plot_bezier(fig, optimal_path, optimal_color, optimal_color, name=optimal_name,linewidth=linewidth, marker_size=marker_size, plot_start_point=plot_start_point, dotted=dotted, plot_start_target_only=plot_start_target_only)
    else:
        plot_bezier(fig, optimal_path, optimal_color, None, name=optimal_name,linewidth=linewidth, marker_size=marker_size, plot_start_point=plot_start_point, dotted=dotted, plot_start_target_only=plot_start_target_only)
    return True, optimal_dt

def plot_rollout(
    fig: go.Figure,
    graph: PolynomialDualGCS,
    lookahead: int,
    vertex: DualVertex,
    state: npt.NDArray,
    rollout_color:str="red",
    plot_control_points:bool=True,
    plot_start_point:bool = True,
    linewidth:int=3,
    marker_size:int=3,
    verbose_comparison:bool = False,
    target_state: npt.NDArray = None,
    rollout_name = "rollout",
    dotted=False,
    plot_start_target_only=False
) -> T.Tuple[bool, float]:
    
    options = graph.options
    graph.options.policy_lookahead = lookahead
    options.policy_lookahead = lookahead
    options.vertify_options_validity()

    if target_state is None:
        assert vertex.target_set_type is Point, "target set not passed into plot rollout when target set not a point"
        target_state = vertex.target_convex_set.x()
    
    rollout_path, rollout_v_path, dt = obtain_rollout(graph, lookahead, vertex, state, target_state)

    if rollout_path is None:
        return False, dt

    if verbose_comparison:
        rollout_cost = get_path_cost(graph, rollout_v_path, rollout_path, False, True, target_state)
        INFO("policy.  time", np.round(dt,3), "cost", np.round(rollout_cost, 3))

    if plot_control_points:
        plot_bezier(fig, rollout_path, rollout_color, rollout_color, name=rollout_name, linewidth=linewidth, marker_size=marker_size, plot_start_point=plot_start_point, dotted=dotted, plot_start_target_only=plot_start_target_only)
    else:
        plot_bezier(fig, rollout_path, rollout_color, None, name=rollout_name, linewidth=linewidth, marker_size=marker_size, plot_start_point=plot_start_point, dotted=dotted, plot_start_target_only=plot_start_target_only)
    return True, dt