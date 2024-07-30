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
import copy

import plotly.graph_objects as go  # pylint: disable=import-error
import plotly.graph_objs as go

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
from solve_restriction import solve_convex_restriction, solve_parallelized_convex_restriction, RestrictionSolution


def precompute_k_step_feasible_paths_from_every_vertex(graph: PolynomialDualGCS, lookaheads: T.List[int]):
    assert graph.table_of_feasible_paths is None
    assert graph.value_function_solution is not None
    graph.table_of_feasible_paths = dict()
    for lookahead in lookaheads:
        lookahead_dict = dict()
        for v_name, vertex in graph.vertices.items():
            vertex_paths = graph.get_all_n_step_paths(lookahead, vertex)
            feasible_vertex_paths = []
            for vertex_path in vertex_paths:
                solution, _ = solve_convex_restriction(graph, vertex_path, None, False, None, True, None)
                if solution is not None:
                    feasible_vertex_paths.append([v.name for v in vertex_path])
            lookahead_dict[v_name] = feasible_vertex_paths
        graph.table_of_feasible_paths[lookahead] = lookahead_dict




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
        previous_vertex = None
        if node.length() >= 2:
            previous_vertex = node.vertex_path[-2]
        vertex_paths = graph.get_all_n_step_paths( options.policy_lookahead, node.vertex_now(), previous_vertex)
    else:
        vertex_paths = graph.get_all_n_step_paths_no_revisits( options.policy_lookahead, node.vertex_now(), node.vertex_path)
    
    # for every path -- solve convex restriction, add next states
    decision_options = PriorityQueue()

    pre_solve_time = 0.0
    # node = node.make_copy()
    if options.policy_rollout_reoptimize_path_so_far_before_K_step and node.length() > 1:
        node.reoptimize(graph, False, target_state, False)
        WARN("this probably does not work")
        # node, pre_solve_time = solve_convex_restriction(graph, 
        #                                                 node.vertex_path, 
        #                                                 node.point_initial(), 
        #                                                 target_state=target_state, 
        #                                                 one_last_solve=False)
    
    INFO("options", len(vertex_paths), verbose=options.policy_verbose_choices)
    solve_times = [0.0]*len(vertex_paths)
    for i, vertex_path in enumerate(vertex_paths):
        if options.policy_rollout_reoptimize_path_so_far_and_K_step:
            index = len(node.vertex_path)
            r_sol, solver_time = solve_convex_restriction(graph, 
                                                          node.vertex_path[:-1] + vertex_path, 
                                                          node.point_initial(), 
                                                          target_state=target_state, 
                                                          one_last_solve=False)
        else:
            index = 1
            r_sol, solver_time = solve_convex_restriction(graph, 
                                                          vertex_path, 
                                                          node.point_now(), 
                                                          target_state=target_state, 
                                                          one_last_solve=False)
        solve_times[i] = solver_time
        if r_sol is not None:
            add_target_heuristic = True
            if r_sol.length() == index:
                next_node = r_sol
            else:
                next_node = node.extend(r_sol.trajectory[index], r_sol.edge_variable_trajectory[index-1], r_sol.vertex_path[index]) # type: RestrictionSolution
            cost_of_que_node = r_sol.get_cost(graph, False, add_target_heuristic, target_state)
            INFO(r_sol.vertex_names(), np.round(cost_of_que_node, 3), verbose=options.policy_verbose_choices)
            decision_options.put( (cost_of_que_node+np.random.uniform(0,1e-9), next_node ))
        else:
            WARN([v.name for v in vertex_path], "failed", verbose=options.policy_verbose_choices)

    if options.use_parallelized_solve_time_reporting:
        num_parallel_solves = np.ceil(len(vertex_paths)/options.num_simulated_cores)
        total_solver_time = np.max(solve_times)*num_parallel_solves
    else:
        total_solver_time = np.sum(solve_times)

    total_solver_time += pre_solve_time

    INFO("---", verbose=options.policy_verbose_choices)
    return decision_options, total_solver_time



# helper functions
def make_a_list_of_shortcuts(numbers:T.List[int], K:int, index:int=0):
    assert 0 <= index and index < len(numbers)
    res = []
    for i in range(0,K+1):
        if numbers[index] - i >= 1:
            if index == len(numbers)-1:        
                res.append( numbers[:index] + [numbers[index] - i] )
            else:
                res = res + make_a_list_of_shortcuts( numbers[:index] + [numbers[index] - i] + numbers[index+1:], K, index+1 )
        else:
            break
    return res
def get_repeats(solution: RestrictionSolution):
    vertex_names = solution.vertex_names()
    vertices = []
    repeats = []
    i = 0
    while i < len(vertex_names):
        vertices.append(solution.vertex_path[i])
        r = 1
        while i+1 < len(vertex_names) and vertex_names[i+1] == vertex_names[i]:
            r += 1
            i += 1
        repeats.append(r)
        i+=1
    return vertices, repeats
def repeats_to_vertex_names(vertices, repeats):
    res = []
    for i in range(len(repeats)):
        res += [vertices[i]] * repeats[i]
    return res

def postprocess_the_path(graph:PolynomialDualGCS, 
                          restriction: RestrictionSolution,
                          initial_state:npt.NDArray, 
                          target_state:npt.NDArray = None,
                          ) -> T.Tuple[RestrictionSolution, float]:
    options = graph.options
    cost_before = restriction.get_cost(graph, False, True, target_state=target_state)
    timer = timeit()
    total_solver_time = 0.0
    # solve a convex restriction on the vertex sequence

    if options.postprocess_via_shortcutting:
        INFO("using shortcut posptprocessing", verbose = options.verbose_restriction_improvement)
        unique_vertices, repeats = get_repeats(restriction)
        shortcut_repeats = make_a_list_of_shortcuts(repeats, options.max_num_shortcut_steps)
        solve_times = [0.0]*len(shortcut_repeats)
        que = PriorityQueue()
        for i, shortcut_repeat in enumerate(shortcut_repeats):
            vertex_path = repeats_to_vertex_names(unique_vertices, shortcut_repeat)
            new_restriction, solver_time = solve_convex_restriction(graph, vertex_path, initial_state, verbose_failure=False, target_state=target_state, one_last_solve = True)
            solve_times[i] = solver_time
            if new_restriction is not None:
                restriction_cost = new_restriction.get_cost(graph, False, True, target_state=target_state)
                que.put((restriction_cost+np.random.uniform(0,1e-9), new_restriction))
        best_cost, best_restriction = que.get()

        if options.use_parallelized_solve_time_reporting:
            num_parallel_solves = np.ceil(len(solve_times)/options.num_simulated_cores)
            total_solver_time = np.max(solve_times)*num_parallel_solves
            INFO(
                "shortcut posptprocessing, num_parallel_solves",
                num_parallel_solves,
                verbose = options.verbose_restriction_improvement
            )
            INFO(np.round(solve_times, 3), verbose = options.verbose_restriction_improvement)
        else:
            total_solver_time = np.sum(solve_times)
        INFO("shortcut posptprocessing time", total_solver_time, verbose = options.verbose_restriction_improvement)

    elif options.postprocess_by_solving_restriction_on_mode_sequence:
        INFO("using restriction post-processing", verbose = options.verbose_restriction_improvement)
        best_restriction, total_solver_time = solve_convex_restriction(graph, restriction.vertex_path, initial_state, verbose_failure=False, target_state=target_state, one_last_solve = True)
        best_cost = best_restriction.get_cost(graph, False, True, target_state=target_state)
        INFO("shortcut posptprocessing time", total_solver_time, verbose = options.verbose_restriction_improvement)
    else:
        best_restriction = restriction
        best_cost = cost_before
        
    INFO(
        "path cost improved from",
        np.round(cost_before, 2),
        "to",
        np.round(best_cost, 2),
        "; original is",
        np.round((cost_before / best_cost - 1) * 100, 1),
        "% worse",
        verbose = options.verbose_restriction_improvement
    )
    timer.dt("solve times", print_stuff = options.verbose_solve_times)
    return best_restriction, total_solver_time


def double_integrator_postprocessing(graph:PolynomialDualGCS, 
                                    restriction: RestrictionSolution,
                                    initial_state:npt.NDArray, 
                                    target_state:npt.NDArray = None
                                    )-> T.Tuple[RestrictionSolution, T.List[float], float]:
    options = graph.options
    unique_vertices, repeats = get_repeats(restriction)
    INFO("using double integrator post-processing", verbose = options.verbose_restriction_improvement)
    
    delta_t = options.delta_t
    ratio = options.double_integrator_post_processing_ratio

    schedules = []
    num = len(unique_vertices)-1
    for i in range(2**num ):
        pick_or_not = bin(i)[2:]
        if len(pick_or_not) < num:
            pick_or_not = "0"*(num - len(pick_or_not)) + pick_or_not

        delta_t_schedule = [delta_t] * (restriction.length()-1)
        for index, pick in enumerate(pick_or_not):
            if pick == "1":
                delta_t_schedule[sum(repeats[:index])] = delta_t * ratio
        schedules.append(np.array(delta_t_schedule))
    
    solve_times = [0.0]*len(schedules)
    que = PriorityQueue()
    for i, schedule in enumerate(schedules):
        new_restriction, solver_time = solve_convex_restriction(graph, 
                                                                restriction.vertex_path, 
                                                                initial_state, 
                                                                verbose_failure=False, 
                                                                target_state=target_state, 
                                                                one_last_solve = True,
                                                                double_itnegrator_delta_t_list=schedule)
        solve_times[i] = solver_time
        if new_restriction is not None:
            restriction_cost = new_restriction.get_cost(graph, False, True, target_state, schedule)
            que.put((restriction_cost+np.random.uniform(0,1e-9), (new_restriction, schedule)))

    best_cost, (best_restriction, best_schedule) = que.get()

    if options.use_parallelized_solve_time_reporting:
        num_parallel_solves = np.ceil(len(solve_times)/options.num_simulated_cores)
        total_solver_time = np.max(solve_times)*num_parallel_solves
        INFO(
            "double inegrator postprocessing, num_parallel_solves",
            num_parallel_solves,
            verbose = options.verbose_restriction_improvement
        )
        INFO(np.round(solve_times, 3), verbose = options.verbose_restriction_improvement)
    else:
        total_solver_time = np.sum(solve_times)

    INFO(
        "double integrator improvement",
        np.round(best_cost, 2),
        verbose = options.verbose_restriction_improvement
    )
    INFO(
        "double inegrator postprocessing time",
        total_solver_time,
        verbose = options.verbose_restriction_improvement
    )

    return best_restriction, best_schedule, total_solver_time





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
            INFO("at", node.vertex_names(), verbose = options.policy_verbose_choices)

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
        final_solution, solver_time = postprocess_the_path(graph, target_node, initial_state, target_state)
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
    options = graph.options
    options.vertify_options_validity()
    INFO("running cheap A*", verbose = options.policy_verbose_choices)

    if target_state is None:
        if options.dont_do_goal_conditioning:
            target_state = np.zeros(vertex.state_dim)
        else:
            assert vertex.target_set_type is Point, "target set not passed when target set not a point"
            target_state = vertex.target_convex_set.x()

    # cost, current state, last state, current vertex, state path so far, vertex path so far
    que = PriorityQueue()
    que.put( (0.0, RestrictionSolution([vertex], [initial_state], []) ) )

    found_target = False
    target_node = None # type: RestrictionSolution
    total_solver_time = 0.0
    number_of_iterations = 0

    while not found_target:
        if que.empty():
            WARN("que is empty")
            break
        
        node = que.get()[1] # type: RestrictionSolution
        INFO("at", node.vertex_names(), verbose = options.policy_verbose_choices)

        # stop if the number of iterations is too high
        number_of_iterations += 1
        if number_of_iterations >= options.forward_iteration_limit:
            WARN("exceeded number of forward iterations")
            return None, total_solver_time

        if node.vertex_now().vertex_is_target:
            YAY("found target", verbose = options.policy_verbose_choices)
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
            WARN("skipped a point due to", verbose = options.policy_verbose_choices)
            continue # skip this one, we've already been in this particular point


        # get all k step optimal paths
        next_decision_que, solve_time = get_k_step_optimal_paths(graph, node, target_state)
        total_solver_time += solve_time
        while not next_decision_que.empty():
            next_cost, next_node = next_decision_que.get()
            # TODO: fix this cost; need to extend
            if options.policy_rollout_reoptimize_path_so_far_and_K_step:
                que.put( (next_cost+np.random.uniform(0,1e-9), next_node) )
            else:
                que.put( (next_cost+np.random.uniform(0,1e-9) + node.get_cost(graph, False, False, target_state), next_node) )

        
    if found_target:
        final_solution, solver_time = postprocess_the_path(graph, target_node, initial_state, target_state)
        total_solver_time += solver_time
        return final_solution, total_solver_time
        
    else:
        WARN("did not find path from start vertex to target!")
        return None, total_solver_time
               

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