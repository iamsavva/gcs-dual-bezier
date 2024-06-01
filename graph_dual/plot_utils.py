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
    VPolytope,
    Hyperellipsoid
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
)  # pylint: disable=import-error, no-name-in-module, unused-import

from gcs_util import get_edge_name, make_quadratic_cost_function_matrices

from graph_dual import QUADRATIC_COST, PolynomialDualGCS, DualEdge, DualVertex

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
    marker_size = 3,
    plot_start_point = True
):
    showlegend = False
    line_name = ""
    if name is not None:
        line_name = name
        showlegend = True

    full_path = None
    first = True
    for curve_index, bezier_curve in enumerate(bezier_curves):
        X = np.array(bezier_curve)
        tt = np.linspace(0, 1, 100)
        xx = MakeBezier(tt, X)
        if full_path is None:
            full_path = xx
        else:
            full_path = np.vstack((full_path, xx))

        # plot bezier curves
        fig.add_trace(
            go.Scatter(
                x=full_path[:, 0],
                y=full_path[:, 1],
                marker_color=bezier_color,
                line=dict(width=linewidth),
                mode="lines",
                name=line_name,
                showlegend=first and line_name is not "",
            )
        )
        first = False
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
                    # marker_color=control_point_color,
                    marker_symbol="circle",
                    mode="markers",
                    showlegend=False,
                    marker=dict(
                        color='white',        # Set the fill color of the markers
                        size=marker_size,             # Set the size of the markers
                        line=dict(
                            color=control_point_color,     # Set the outline color of the markers
                            width=3          # Set the width of the marker outline
                        )
                    ),
                )
            )
        if plot_start_point:
            if curve_index == 0:
                fig.add_trace(
                    go.Scatter(
                        x=[xx[0, 0]],
                        y=[xx[0, 1]],
                        # marker_color=bezier_color,
                        mode="markers",
                        marker=dict(
                        color='white',        # Set the fill color of the markers
                        size=marker_size,             # Set the size of the markers
                        line=dict(
                            color=control_point_color,     # Set the outline color of the markers
                            width=3          # Set the width of the marker outline
                        )
                    ),
                        showlegend=False,
                    )
                )

def get_clockwise_vertices(vpoly:VPolytope):
    vertices = list(vpoly.vertices().T)
    c = np.mean(vpoly.vertices(),axis=1) # compute center
    vertices.sort(key = lambda p: np.arctan2( (p-c)[1], (p-c)[0] ) )
    return np.array(vertices).T

def get_ellipse(mu, sigma):
    eigenvalues, eigenvectors = np.linalg.eig(sigma)
    mu = np.array(mu).reshape((2,1))
    theta = np.linspace(0, 2*np.pi, 1000)
    ellipse = (np.sqrt(eigenvalues[None,:]) * eigenvectors) @ [np.sin(theta), np.cos(theta)] + mu
    xs=ellipse[0,:]
    ys=ellipse[1,:]
    return xs,ys

def plot_a_2d_graph(vertices:T.List[DualVertex], width = 800, fill_color = "mintcream"):
    fig = go.Figure()

    def add_trace(convex_set:ConvexSet):
        if isinstance(convex_set, Hyperrectangle):
            lb,ub = convex_set.lb(), convex_set.ub()
            xs = [lb[0], lb[0], ub[0], ub[0], lb[0]]
            ys = [lb[1], ub[1], ub[1], lb[1], lb[1]]
        elif isinstance(convex_set, HPolyhedron):
            vertices = get_clockwise_vertices(VPolytope(convex_set))
            xs = np.append(vertices[0,:], vertices[0,0])
            ys = np.append(vertices[1,:], vertices[1,0])
        elif isinstance(convex_set, Hyperellipsoid):
            mu = convex_set.center()
            sigma = np.linalg.inv(convex_set.A().T.dot(convex_set.A()))
            xs,ys = get_ellipse(mu,sigma)
        elif isinstance(convex_set, Point):
            xs = [convex_set.x()[0]]
            ys = [convex_set.x()[1]]
        else:
            raise Exception("what the bloody hell is that set")

        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                line=dict(color="black"),
                fillcolor=fill_color,
                fill="tozeroy",
                showlegend=False,
            )
        )

    for v in vertices:
        add_trace(v.convex_set)
        if isinstance(v.convex_set, Hyperrectangle):
            center = (v.convex_set.lb() + v.convex_set.ub()) / 2
        elif isinstance(v.convex_set, HPolyhedron):
            vertices = get_clockwise_vertices(VPolytope(v.convex_set))
            center = np.mean(vertices, axis=1)
        elif isinstance(v.convex_set, Hyperellipsoid):
            center = v.convex_set.center()

        fig.add_trace(
            go.Scatter(
                x=[center[0]],
                y=[center[1]+0.5],
                mode="text",
                text=[v.name],
                showlegend=False,
            )
        )

    fig.update_layout(height=width, width=width, title_text="Graph view")
    fig.update_layout(
        yaxis=dict(scaleanchor="x"),  # set y-axis to have the same scaling as x-axis
        yaxis2=dict(
            scaleanchor="x", overlaying="y", side="right"
        ),  # set y-axis2 to have the same scaling as x-axis
    )

    return fig
