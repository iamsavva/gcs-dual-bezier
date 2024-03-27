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
from pydrake.math import (
    ge,
    eq,
    le,
)  # pylint: disable=import-error, no-name-in-module, unused-import

import plotly.graph_objects as go  # pylint: disable=import-error
from plotly.express.colors import sample_colorscale  # pylint: disable=import-error
import plotly.graph_objs as go
from plotly.subplots import make_subplots

from collections import deque

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

from gcs_util import get_edge_name, make_quadratic_cost_function_matrices
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
    find every n-step path
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
                    vertex_expand_que.append(
                        (right_vertex, path + [right_vertex], lookahead - 1)
                    )
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


# ---------------------------------------------------------------


def solve_convex_restriction(
    graph: PolynomialDualGCS,
    path: T.List[DualVertex],
    state_now: npt.NDArray,
    state_last: npt.NDArray = None,
    options: ProgramOptions = None,
) -> T.Tuple[float, T.List[T.List[npt.NDArray]]]:
    """
    solve a convex restriction over a vertex path
    return cost of the path
    and return a list of bezier curves
    where bezier curve is a list of numpy arrays (vectors).
    """
    if options is None:
        options = graph.options
    # need to construct an optimization problem
    prog = MathematicalProgram()
    last_x = prog.NewContinuousVariables(path[0].state_dim)
    prog.AddLinearConstraint(eq(last_x, state_now))
    last_delta = None
    if state_last is not None:
        last_delta = last_x - state_last

    bezier_curves = []
    # for every but the last vertex:
    for i, vertex in enumerate(path):
        # it's the last vertex -- don't add a bezier curve; add terminal cost
        if i == len(path) - 1:
            potential = graph.value_function_solution.GetSolution(vertex.potential)
            f_potential = lambda x: potential.Substitute(
                {vertex.x[i]: x[i] for i in range(vertex.state_dim)}
            )
            prog.AddCost(f_potential(last_x))

            # NOTE: UNCOMMENT THIS FOR SMOOTH TRAJECTORIES
            # assert that next point is feasible
            if not vertex.vertex_is_target:
                prog.AddLinearConstraint(
                    ge(vertex.B.dot(np.hstack(([1], last_x + last_delta))), 0)
                )

            if options.policy_add_G_term:
                G_matrix = graph.value_function_solution.GetSolution(vertex.G_matrix)

                def eval_G(x):
                    x_and_1 = np.hstack(([1], x))
                    return x_and_1.dot(G_matrix).dot(x_and_1)

                prog.AddCost(eval_G(last_delta))

            if options.policy_add_total_flow_in_violation_penalty:
                prog.AddLinearCost(
                    graph.value_function_solution.GetSolution(
                        vertex.total_flow_in_violation
                    )
                )

        else:
            bezier_curve = [last_x]
            edge = graph.edges[get_edge_name(vertex.name, path[i + 1].name)]
            for j in range(1, options.num_control_points):
                # add a new knot point
                x_j = prog.NewContinuousVariables(vertex.state_dim)
                bezier_curve.append(x_j)
                # knot point inside a set
                if j == options.num_control_points - 1:
                    # inside the intersection
                    prog.AddLinearConstraint(
                        ge(edge.B_intersection.dot(np.hstack(([1], x_j))), 0)
                    )
                else:
                    # inside the vertex
                    prog.AddLinearConstraint(ge(vertex.B.dot(np.hstack(([1], x_j))), 0))
                # quadratic cost with previous point
                prog.AddQuadraticCost(edge.cost_function(last_x, x_j))

                # NOTE: UNCOMMENT THIS FOR SMOOTH TRAJECTORIES
                # if the point is the first knot point in this set -- add the bezier continuity constraint
                if j == 1 and last_delta is not None:
                    prog.AddLinearConstraint(eq(x_j - last_x, last_delta))

                # we just added the last point
                if j == options.num_control_points - 1:
                    last_delta = x_j - last_x
                last_x = x_j
            bezier_curves.append(bezier_curve)

            # if i == len(path)-2:
            #     prog.AddCost(graph.value_function_solution.GetSolution(edge.bidirectional_edge_violation))

    solution = Solve(prog)
    if solution.is_success():
        return solution.get_optimal_cost(), [
            solution.GetSolution(bezier_curve) for bezier_curve in bezier_curves
        ]
    else:
        return np.inf, []


# ---


def get_k_step_optimal_path(
    gcs: PolynomialDualGCS,
    vertex: DualVertex,
    state: npt.NDArray,
    last_state: npt.NDArray = None,
    options: ProgramOptions = None,
    already_visited: T.List[DualVertex] = [],
) -> T.Tuple[float, T.List[T.List[npt.NDArray]], T.List[DualVertex]]:
    actual_cost = vertex.cost_at_point(state, gcs.value_function_solution)
    if options is None:
        options = gcs.options
    if options.policy_no_vertex_revisits:
        vertex_paths = get_all_n_step_paths_no_revisits(
            gcs, options.policy_lookahead, vertex, already_visited
        )
    else:
        vertex_paths = get_all_n_step_paths(gcs, options.policy_lookahead, vertex)
    best_cost, best_path, best_vertex_path = np.inf, None, None
    for vertex_path in vertex_paths:
        cost, bezier_curves = solve_convex_restriction(
            gcs, vertex_path, state, last_state, options
        )
        if options.policy_verbose_choices:
            print([v.name for v in vertex_path], cost)
        if options.policy_min_cost:
            if cost < best_cost:
                best_cost, best_path, best_vertex_path = (
                    cost,
                    bezier_curves,
                    vertex_path,
                )
        else:
            if np.abs(cost - actual_cost) < np.abs(best_cost - actual_cost):
                best_cost, best_path, best_vertex_path = (
                    cost,
                    bezier_curves,
                    vertex_path,
                )
    if options.policy_verbose_choices:
        print("----")
    return best_cost, best_path, best_vertex_path


# def get_path_cost(vertex_path: T.List[DualVertex], bezier_path:T.List[T.List[npt.NDArray]]):


def rollout_the_policy(
    gcs: PolynomialDualGCS,
    vertex: DualVertex,
    state: npt.NDArray,
    last_state: npt.NDArray = None,
    options: ProgramOptions = None,
) -> T.List[T.List[npt.NDArray]]:
    if options is None:
        options = gcs.options
    vertex_now, state_now, state_last = vertex, state, last_state

    full_path = []
    vertex_path_so_far = [vertex_now]

    while not vertex_now.vertex_is_target:
        _, bezier_path, vertex_path = get_k_step_optimal_path(
            gcs,
            vertex_now,
            state_now,
            state_last,
            options,
            already_visited=vertex_path_so_far,
        )
        first_segment = bezier_path[0]
        full_path.append(first_segment)
        vertex_now, state_now, state_last = (
            vertex_path[1],
            first_segment[-1],
            first_segment[-2],
        )
        vertex_path_so_far.append(vertex_now)

    # solve a convex restriction on the mode sequence
    if options.postprocess_by_solving_restrction_on_mode_sequence:
        _, full_path = solve_convex_restriction(
            gcs, vertex_path_so_far, state, last_state, options
        )

    return full_path


def plot_optimal_and_rollout(
    fig: go.Figure,
    gcs: PolynomialDualGCS,
    lookahead: int,
    vertex: DualVertex,
    state: npt.NDArray,
    last_state: npt.NDArray = None,
    plot_optimal=True,
    optimal_lookahead=10,
    rollout_color="red",
) -> None:
    options = gcs.options
    options.policy_lookahead = lookahead
    rollout_path = rollout_the_policy(gcs, vertex, state, last_state, options)
    plot_bezier(fig, rollout_path, rollout_color, rollout_color, name="rollout")

    # options.policy_add_G_term=True
    # rollout_path_with_G = rollout_the_policy(gcs, vertex, state, last_state, options)
    # plot_bezier(fig, rollout_path_with_G, "purple", "purple", name="rollout with G")

    if plot_optimal:
        options.policy_lookahead = optimal_lookahead
        _, optimal_path, _ = get_k_step_optimal_path(
            gcs, vertex, state, last_state, options
        )
        plot_bezier(fig, optimal_path, "blue", "blue", name="optimal")
