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
from solve_restriction import solve_convex_restriction, get_optimal_path, solve_parallelized_convex_restriction, RestrictionSolution


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


def get_k_step_optimal_paths(
    graph: PolynomialDualGCS,
    node: RestrictionSolution,
    target_state: npt.NDArray = None,
) -> T.Tuple[PriorityQueue, float]:
    """ 
    do not use this to compute optimal trajectories
    """
    options = graph.options
    if options.allow_vertex_revisits:
        vertex_paths = get_all_n_step_paths(
            graph, options.policy_lookahead, node.vertex_now()
        )
    else:
        vertex_paths = get_all_n_step_paths_no_revisits(
            graph, options.policy_lookahead, node.vertex_now(), node.vertex_path
        )
    
    # for every path -- solve convex restriction, add next states
    decision_options = PriorityQueue()
    
    INFO("options", verbose=options.policy_verbose_choices)
    solve_times = [0.0]*len(vertex_paths)
    for i, vertex_path in enumerate(vertex_paths):
        # r_sols, solver_time = solve_parallelized_convex_restriction(graph, [vertex_path], node.point_now(), target_state=target_state, one_last_solve=False)
        # r_sol = None if r_sols is None else r_sols[0]
        r_sol, solver_time = solve_convex_restriction(graph, vertex_path, node.point_now(), target_state=target_state, one_last_solve=False)
        solve_times[i] = solver_time
        if r_sol is not None:
            add_target_heuristic = True
            if r_sol.length() == 1:
                next_node = r_sol
            else:
                next_node = node.extend(r_sol.trajectory[1], r_sol.edge_variable_trajectory[0], r_sol.vertex_path[1]) # type: RestrictionSolution
            cost_of_que_node = r_sol.get_cost(graph, False, add_target_heuristic, target_state)
            INFO(r_sol.vertex_names(), np.round(cost_of_que_node, 3), verbose=options.policy_verbose_choices)
            decision_options.put( (cost_of_que_node+np.random.uniform(0,1e-9), next_node ))

    
    if options.use_parallelized_solve_time_reporting:
        num_parallel_solves = np.ceil(len(vertex_paths)/options.num_simulated_cores)
        total_solver_time = np.max(solve_times)*num_parallel_solves
    else:
        total_solver_time = np.sum(solve_times)

    INFO("---", verbose=options.policy_verbose_choices)
    return decision_options, total_solver_time


def postprocess_the_path(graph:PolynomialDualGCS, 
                          restriction: RestrictionSolution,
                          initial_state:npt.NDArray, 
                          options:ProgramOptions = None, 
                          target_state:npt.NDArray = None) -> T.Tuple[RestrictionSolution, float]:
    if options is None:
        options = graph.options
    if options.verbose_restriction_improvement:
        cost_before = restriction.get_cost(graph, False, True, target_state=target_state)
    timer = timeit()
    solver_time = 0.0
    # solve a convex restriction on the vertex sequence
    if options.postprocess_by_solving_restriction_on_mode_sequence:
        # print("postprocessing")
        new_restriction, solver_time = solve_convex_restriction(graph, restriction.vertex_path, initial_state, options, target_state=target_state, one_last_solve = True)
        # verbose
        if options.verbose_restriction_improvement:
            cost_after = new_restriction.get_cost(graph, False, True, target_state=target_state)
            INFO(
                "path cost improved from",
                np.round(cost_before, 1),
                "to",
                np.round(cost_after, 1),
                "; original is",
                np.round((cost_before / cost_after - 1) * 100, 1),
                "% worse",
            )
    else:
        new_restriction = restriction
    timer.dt("one last solve", print_stuff = options.verbose_solve_times)
    return new_restriction, solver_time


def get_lookahead_cost(
    graph: PolynomialDualGCS,
    lookahead:int,
    vertex: DualVertex,
    initial_state: npt.NDArray,
    target_state: npt.NDArray = None,
) -> float:
    """
    K-step lookahead rollout policy.
    Returns a list of bezier curves. Each bezier curve is a list of control points (numpy arrays).
    """
    graph.options.policy_lookahead = lookahead
    options = graph.options
    options.vertify_options_validity()

    if target_state is None:
        if options.dont_do_goal_conditioning:
            target_state = np.zeros(vertex.state_dim)
        else:
            assert vertex.target_set_type is Point, "target set not passed when target set not a point"
            target_state = vertex.target_convex_set.x()

    node = RestrictionSolution([vertex], [initial_state])
    que = get_k_step_optimal_paths(graph, node)[0]
    return que.get()[0]


def lookahead_with_backtracking_policy(
    graph: PolynomialDualGCS,
    vertex: DualVertex,
    initial_state: npt.NDArray,
    target_state: npt.NDArray = None,
) -> RestrictionSolution:
    """
    K-step lookahead rollout policy.
    If you reach a point from which no action is available --
    -- backtrack to the last state when some action was available.
    Returns a list of bezier curves. Each bezier curve is a list of control points (numpy arrays).
    """
    # TODO: add a parallelized options
    options = graph.options
    options.vertify_options_validity()
    INFO("running lookahead backtracking", verbose = options.policy_verbose_choices)

    if target_state is None:
        if options.dont_do_goal_conditioning:
            target_state = np.zeros(vertex.state_dim)
        else:
            assert vertex.target_set_type is Point, "target set not passed when target set not a point"
            target_state = vertex.target_convex_set.x()

    # cost, current state, last state, current vertex, state path so far, vertex path so far
    decision_options = [ PriorityQueue() ]
    decision_options[0].put( (0, RestrictionSolution([vertex], [initial_state], [])) )

    decision_index = 0
    found_target = False
    target_node = None
    number_of_iterations = 0

    total_solver_time = 0.0

    while not found_target:
        if decision_index == -1:
            return None, None
        if decision_options[decision_index].empty():
            decision_index -= 1
            INFO("backtracking", verbose=options.policy_verbose_choices)
        else:
            node = decision_options[decision_index].get()[1] # type: RestrictionSolution

            if node.vertex_now().vertex_is_target:
                found_target = True
                target_node = node
                break

            # heuristic: don't ever consider a point you've already been in
            stop = False
            for point in node.trajectory[:-1]:
                if np.allclose(node.point_now(), point, atol=1e-3):
                    stop = True
                    break
            if stop:
                continue

            # add another priority que if needed (i.e., first time we are at horizon H)
            if len(decision_options) == decision_index + 1:
                decision_options.append( PriorityQueue() )

            # get all k step optimal paths
            que, solve_time = get_k_step_optimal_paths(graph, node, target_state)
            decision_options[decision_index + 1] = que            
            total_solver_time += solve_time

            # stop if the number of iterations is too high
            number_of_iterations += 1
            if number_of_iterations >= options.forward_iteration_limit:
                WARN("exceeded number of forward iterations")
                return None, total_solver_time
        
            decision_index += 1


    if found_target:
        final_solution, solver_time = postprocess_the_path(graph, target_node, initial_state, options, target_state)
        total_solver_time += solver_time
        return final_solution, total_solver_time
        
    else:
        WARN("did not find path from start vertex to target!")
        return None, total_solver_time


def cheap_a_star_policy(
    graph: PolynomialDualGCS,
    vertex: DualVertex,
    initial_state: npt.NDArray,
    target_state: npt.NDArray = None,
) -> T.Tuple[T.List[T.List[npt.NDArray]], T.List[DualVertex]]:
    """
    K-step lookahead rollout policy.
    If you reach a point from which no action is available --
    -- backtrack to the last state when some action was available.
    Returns a list of bezier curves. Each bezier curve is a list of control points (numpy arrays).
    """
    raise NotImplementedError()
    options = graph.options
    options.vertify_options_validity()

    if target_state is None:
        if options.dont_do_goal_conditioning:
            target_state = np.zeros(vertex.state_dim)
        else:
            assert vertex.target_set_type is Point, "target set not passed when target set not a point"
            target_state = vertex.target_convex_set.x()

    # cost, current state, last state, current vertex, state path so far, vertex path so far
    que = PriorityQueue()
    que.put( (0.0, RestrictionSolution([vertex], [initial_state], []) ) )
    num_times_solved_convex_restriction = 0


    found_target = False
    target_node = None # type: RestrictionSolution

    while not found_target:
        
        node = que.get()[1] # type: RestrictionSolution
        if node.vertex_now().vertex_is_target:
            found_target = True
            target_node = node
            break
        else:
            if options.allow_vertex_revisits:
                vertex_paths = get_all_n_step_paths(
                    graph, options.policy_lookahead, node.vertex_now()
                )
            else:
                vertex_paths = get_all_n_step_paths_no_revisits(
                    graph, options.policy_lookahead, node.vertex_now(), node.vertex_path_so_far
                )
            # for every path -- solve convex restriction, add next states
            # print(len(vertex_paths))
            for vertex_path in vertex_paths:
                bezier_curves = solve_convex_restriction(graph, vertex_path, node.point_now(), options, target_state=target_state, one_last_solve=False)
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
               

def obtain_rollout(
    graph: PolynomialDualGCS,
    lookahead: int,
    vertex: DualVertex,
    state: npt.NDArray,
    target_state: npt.NDArray = None,
) -> T.Tuple[RestrictionSolution, float]:
    graph.options.policy_lookahead = lookahead
    graph.options.vertify_options_validity()
    options = graph.options
    
    if target_state is None:
        if options.dont_do_goal_conditioning:
            target_state = np.zeros(vertex.state_dim)
        else:
            assert vertex.target_set_type is Point, "target set not passed when target set not a point"
            target_state = vertex.target_convex_set.x()
    
    timer = timeit()
    if options.use_lookahead_with_backtracking_policy:
        restriction, solve_time = lookahead_with_backtracking_policy(graph, vertex, state, target_state)
    elif options.use_cheap_a_star_policy:
        restriction, solve_time = cheap_a_star_policy(graph, vertex, state, target_state)
    else:
        raise Exception("not selected policy")
        
    dt = timer.dt(print_stuff = False)
    return restriction, solve_time


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
    raise NotImplementedError()

    rollout_path, rollout_v_path, rollout_dt = obtain_rollout(graph, lookahead, vertex, state, target_state)
    rollout_cost = get_path_cost(graph, rollout_v_path, rollout_path, False, True, target_state)
    if rollout_path is None:
        WARN("faile to solve for ", state)
        return False, None, None, None, None
    if verbose_comparison_to_optimal:
            INFO("policy.  time", np.round(rollout_dt,3), "cost", np.round(rollout_cost, 3))
    
    if get_optimal:
        raise NotImplementedError()
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
    raise NotImplementedError()
    # TODO: drop this, this is repeated
    options = graph.options
    options.vertify_options_validity()

    if target_state is None:
        if options.dont_do_goal_conditioning:
            target_state = np.zeros(vertex.state_dim)
        else:
            assert vertex.target_set_type is Point, "target set not passed when target set not a point"
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
        if options.dont_do_goal_conditioning:
            target_state = np.zeros(vertex.state_dim)
        else:
            assert vertex.target_set_type is Point, "target set not passed when target set not a point"
            target_state = vertex.target_convex_set.x()
    
    restriction, dt = obtain_rollout(graph, lookahead, vertex, state, target_state)

    rollout_path, rollout_v_path = restriction.vertex_path, restriction.trajectory

    if rollout_path is None:
        return False, dt

    if verbose_comparison:
        rollout_cost = restriction.get_cost(graph, False, True, target_state)
        INFO("policy.  time", np.round(dt,3), "cost", np.round(rollout_cost, 3))

    if plot_control_points:
        plot_bezier(fig, rollout_path, rollout_color, rollout_color, name=rollout_name, linewidth=linewidth, marker_size=marker_size, plot_start_point=plot_start_point, dotted=dotted, plot_start_target_only=plot_start_target_only)
    else:
        plot_bezier(fig, rollout_path, rollout_color, None, name=rollout_name, linewidth=linewidth, marker_size=marker_size, plot_start_point=plot_start_point, dotted=dotted, plot_start_target_only=plot_start_target_only)
    return True, dt