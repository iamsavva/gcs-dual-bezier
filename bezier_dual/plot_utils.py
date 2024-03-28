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

from scipy.special import comb

# ------------------------------------------------------------------
# bezier plot utils


def get_bezier_point_combination(i, N, t):
    val = comb(N, i) * t**i * (1.0 - t) ** (N - i)
    return val


tt = np.linspace(0, 1, 100)


def MakeBezier(t: npt.NDArray, X: npt.NDArray):
    """
    Evaluates a Bezier curve for the points in X.

    Args:
        t (npt.NDArray): npt.ND: number (or list of numbers) in [0,1] where to evaluate the Bezier curve
        X (npt.NDArray): list of control points

    Returns:
        npt.NDArray: set of 2D points along the Bezier curve
    """
    N, d = np.shape(X)  # num points, dim of each point
    assert d == 2, "control points need to be dimension 2"
    xx = np.zeros((len(t), d))
    # evaluate bezier per row
    for i in range(N):
        xx += np.outer(get_bezier_point_combination(i, N - 1, t), X[i])
    return xx


def plot_bezier(
    fig: go.Figure,
    bezier_curves: T.List[T.List[npt.NDArray]],
    bezier_color="purple",
    control_point_color="red",
    name=None,
    linewidth=3,
):
    showlegend = False
    line_name = ""
    if name is not None:
        line_name = name
        showlegend = True

    full_path = None
    for curve_index, bezier_curve in enumerate(bezier_curves):
        X = np.array(bezier_curve)
        tt = np.linspace(0, 1, 100)
        xx = MakeBezier(tt, X)
        if full_path is None:
            full_path = xx
        else:
            full_path = np.vstack((full_path, xx))

        # plot contorl points
        if control_point_color is not None:
            fig.add_trace(
                go.Scatter(
                    x=X[:, 0],
                    y=X[:, 1],
                    marker_color=control_point_color,
                    marker_symbol="circle",
                    mode="markers",
                    showlegend=False,
                )
            )

        if curve_index == 0:
            fig.add_trace(
                go.Scatter(
                    x=[xx[0, 0]],
                    y=[xx[0, 1]],
                    marker_color=bezier_color,
                    mode="markers",
                    showlegend=False,
                )
            )

    # plot bezier curves
    fig.add_trace(
        go.Scatter(
            x=full_path[:, 0],
            y=full_path[:, 1],
            marker_color=bezier_color,
            line=dict(width=linewidth),
            mode="lines",
            name=line_name,
            showlegend=showlegend,
        )
    )


## ------------------------------------------------------
# extracting policy


def solve_m_step_horizon_from_layers(
    gcs: PolynomialDualGCS,
    layers: T.List[T.List[DualVertex]],
    m: int,
    start_vertex: DualVertex,
    layer_index: int,
    point: npt.NDArray,
):
    new_gcs = GraphOfConvexSets()
    init_vertex = new_gcs.AddVertex(Point(point), start_vertex.name)
    new_layers = []
    new_layers.append([init_vertex])

    # for every layer
    last_index = min(len(layers) - 1, layer_index + m)
    for n in range(layer_index + 1, last_index + 1):
        layer = []
        for og_graph_v in layers[n]:
            new_v = new_gcs.AddVertex(og_graph_v.convex_set, og_graph_v.name)
            layer.append(new_v)

            # connect with edges to previous layer
            for left_v in new_layers[-1]:
                edge = gcs.edges[get_edge_name(left_v.name(), new_v.name())]

                cost_function = gcs.get_policy_edge_cost(edge, (n == last_index))

                new_edge = new_gcs.AddEdge(
                    left_v, new_v, get_edge_name(left_v.name(), new_v.name())
                )
                new_edge.AddCost(cost_function(new_edge.xu(), new_edge.xv()))

        new_layers.append(layer)

    target_v = new_gcs.AddVertex(Hyperrectangle([0], [0]), "target")
    for left_v in new_layers[-1]:
        new_gcs.AddEdge(left_v, target_v, get_edge_name(left_v.name(), target_v.name()))

    gcs_options = GraphOfConvexSetsOptions()
    gcs_options.convex_relaxation = True
    gcs_options.max_rounded_paths = 30
    gcs_options.max_rounding_trials = 30
    gcs_options.solver = MosekSolver()

    # solve
    result = new_gcs.SolveShortestPath(
        init_vertex, target_v, gcs_options
    )  # type: MathematicalProgramResult
    assert result.is_success()
    cost = result.get_optimal_cost()

    edge_path = new_gcs.GetSolutionPath(init_vertex, target_v, result)
    vertex_name_path = [init_vertex.name()]
    value_path = [point]
    for e in edge_path:
        vertex_name_path.append(e.v().name())
        value_path.append(result.GetSolution(e.v().x()))
    return cost, vertex_name_path, value_path


def get_next_action_gcs(
    gcs: PolynomialDualGCS,
    layers: T.List[T.List[DualVertex]],
    m: int,
    vertex: DualVertex,
    point: npt.NDArray,
    layer_index: int,
):
    cost, vertex_name_path, value_path = solve_m_step_horizon_from_layers(
        gcs, layers, m, vertex, layer_index, point
    )
    # print(cost, vertex_name_path[1], value_path[1])
    return gcs.vertices[vertex_name_path[1]], value_path[1]


def rollout_m_step_policy(
    gcs: PolynomialDualGCS,
    layers: T.List[T.List[DualVertex]],
    m: int,
    vertex: DualVertex,
    point: npt.NDArray,
    layer_index: int,
) -> T.Tuple[float, T.List[DualVertex], T.List[npt.NDArray]]:
    if layer_index < len(layers) - 1:
        if gcs.options.policy_use_gcs:
            next_vertex, next_point = get_next_action_gcs(
                gcs, layers, m, vertex, point, layer_index
            )
        else:
            next_vertex, next_point = gcs.solve_m_step_policy(layers, m, vertex, point, layer_index)
        cost, vertex_trajectory, trajectory = rollout_m_step_policy(
            gcs, layers, m, next_vertex, next_point, layer_index + 1
        )
        return (
            QUADRATIC_COST(point, next_point) + cost,
            [next_vertex] + vertex_trajectory,
            [next_point] + trajectory,
        )
    else:
        return 0.0, [], []


def plot_policy_rollout(
    gcs: PolynomialDualGCS,
    layers: T.List[T.List[DualVertex]],
    m: int,
    vertex: DualVertex,
    layer_index: int,
    fig: go.Figure,
    point: npt.NDArray,
):
    _, vertex_trajectory, trajectory = rollout_m_step_policy(
        gcs, layers, m, vertex, point, layer_index
    )
    vertex_trajectory = [vertex] + vertex_trajectory
    trajectory = [point] + trajectory

    x, y, z = [], [], []
    n = len(vertex_trajectory)
    for i in range(n):
        point = trajectory[i]
        x.append(point[0])
        y.append(point[1])
        z.append(vertex_trajectory[i].cost_at_point(point, gcs.value_function_solution))
    fig.add_traces(
        go.Scatter3d(
            x=x,
            y=y,
            z=np.array(z) - gcs.options.zero_offset,
            mode="lines",
            line=dict(width=5, color="black"),
        )
    )


########################################################################
# plotting trajectories


def plot_a_layered_graph_1d(layers: T.List[T.List[DualVertex]]):
    fig = go.Figure()

    def add_trace(x_min, x_max, y):
        xs = [x_min, x_max]
        ys = [y, y]
        fig.add_trace(go.Scatter(x=xs, y=ys, line=dict(color="black")))

    y = len(layers)
    for n, layer in enumerate(layers):
        for v in layer:
            add_trace(v.convex_set.lb()[0], v.convex_set.ub()[0], y)
        y -= 1

    fig.update_layout(height=800, width=800, title_text="Graph view")
    fig.update_layout(showlegend=False)
    fig.update_layout(
        yaxis=dict(scaleanchor="x"),  # set y-axis to have the same scaling as x-axis
        yaxis2=dict(
            scaleanchor="x", overlaying="y", side="right"
        ),  # set y-axis2 to have the same scaling as x-axis
    )
    return fig


def plot_a_2d_graph(vertices=T.List[DualVertex]):
    # TODO: assumes that vertices are
    fig = go.Figure()

    def add_trace(lb, ub):
        xs = [lb[0], lb[0], ub[0], ub[0], lb[0]]
        ys = [lb[1], ub[1], ub[1], lb[1], lb[1]]
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                line=dict(color="black"),
                fillcolor="grey",
                fill="tozeroy",
                showlegend=False,
            )
        )

    for v in vertices:
        add_trace(v.convex_set.lb(), v.convex_set.ub())
        center = (v.convex_set.lb() + v.convex_set.ub()) / 2
        fig.add_trace(
            go.Scatter(
                x=[center[0]],
                y=[center[1]],
                mode="text",
                text=[v.name],
                showlegend=False,
            )
        )

    fig.update_layout(height=800, width=800, title_text="Graph view")
    fig.update_layout(
        yaxis=dict(scaleanchor="x"),  # set y-axis to have the same scaling as x-axis
        yaxis2=dict(
            scaleanchor="x", overlaying="y", side="right"
        ),  # set y-axis2 to have the same scaling as x-axis
    )

    return fig


def plot_policy_rollout_1d(
    gcs: PolynomialDualGCS,
    layers: T.List[T.List[DualVertex]],
    m: int,
    vertex: DualVertex,
    layer_index: int,
    fig: go.Figure,
    point: npt.NDArray,
):
    _, vertex_trajectory, trajectory = rollout_m_step_policy(
        gcs, layers, m, vertex, point, layer_index
    )
    vertex_trajectory = [vertex] + vertex_trajectory
    trajectory = [point] + trajectory

    x, y = [], []
    n = len(vertex_trajectory)
    # print("trajectory: ", trajectory)
    for i in range(n):
        point = trajectory[i]
        x.append(point[0])
        y.append(n - i)

    fig.add_trace(go.Scatter(x=x, y=y, line=dict(color="blue"), showlegend=True))


def plot_policy_rollout_2d(
    gcs: PolynomialDualGCS,
    layers: T.List[T.List[DualVertex]],
    m: int,
    vertex: DualVertex,
    layer_index: int,
    fig: go.Figure,
    point: npt.NDArray,
    linewidth=3,
):
    _, vertex_trajectory, trajectory = rollout_m_step_policy(
        gcs, layers, m, vertex, point, layer_index
    )
    vertex_trajectory = [vertex] + vertex_trajectory
    trajectory = [point] + trajectory

    x, y = [], []
    n = len(vertex_trajectory)
    # print("trajectory: ", trajectory)
    for i in range(n):
        point = trajectory[i]
        x.append(point[0])
        y.append(point[1])

    fig.add_trace(go.Scatter(x=x, y=y, line=dict(color="blue", width=linewidth), showlegend=True))
