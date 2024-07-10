from plot_utils import get_clockwise_vertices, get_ellipse
from pydrake.all import VPolytope
from gcs_dual import DualVertex, PolynomialDualGCS, DualEdge
import numpy as np
from program_options import ProgramOptions, PSD_POLY, FREE_POLY, CONVEX_POLY
from pydrake.all import MosekSolver, Point, Hyperellipsoid, Hyperrectangle, HPolyhedron, ConvexSet
from proper_gcs_policy import obtain_rollout, plot_rollout, RestrictionSolution
import numpy.typing as npt
import plotly.graph_objects as go  # pylint: disable=import-error
from plotly.express.colors import sample_colorscale  # pylint: disable=import-error
import plotly.graph_objs as go
from plotly.subplots import make_subplots

import typing as T
import plotly
import plotly.graph_objs as go
from IPython.display import display, HTML
plotly.offline.init_notebook_mode()
display(HTML(
    '<script type="text/javascript" async src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/MathJax.js?config=TeX-MML-AM_SVG"></script>'
))

import logging
logging.getLogger("drake").setLevel(logging.WARNING)
np.set_printoptions(suppress=True) 

from testing_footstep_helpers import SteppingStone, Terrain, footstep2position

from pydrake.all import MathematicalProgram, Solve, MathematicalProgramResult
from util import add_set_membership, concatenate_polyhedra

from pydrake.math import (  # pylint: disable=import-error, no-name-in-module, unused-import
    ge,
    eq,
    le,
)

def plot_a_2d_graph(vertex_sets:T.List[ConvexSet], cheap_sets:T.List[int], height=800, width = 600, make_terminal_black=True, fill_color = "mintcream", font_size = 10, override_points_size=False, plot_points = False, point_size=10, s_offset=0.35, t_offset=0.35, plot_all_names=False):
    fig = go.Figure()

    def add_trace(convex_set:ConvexSet, v_name):
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
            if override_points_size or (v_name == "t" and make_terminal_black):
                sigma = np.linalg.inv(np.eye(2)*point_size*point_size)
            else:
                sigma = np.linalg.inv(convex_set.A().T.dot(convex_set.A()))
            xs,ys = get_ellipse(mu,sigma)
        elif isinstance(convex_set, Point):
            xs = [convex_set.x()[0]]
            ys = [convex_set.x()[1]]
        else:
            raise Exception("what the bloody hell is that set")

        if isinstance(convex_set, Point):
            mu = convex_set.x()
            sigma = np.linalg.inv(np.eye(2)*point_size*point_size)
            xs,ys = get_ellipse(mu,sigma)
            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=ys,
                    line=dict(color="black", width=3),
                    fillcolor="black",
                    fill="tozeroy",
                    mode="lines",
                    showlegend=False,
                )
            )
        else:
            if int(v_name) in cheap_sets:
                fillcolor = "#EEFFEF"
            else:
                fillcolor="#FFD6D7"
            
            if v_name == "t" and make_terminal_black:
                fillcolor = "black"
            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=ys,
                    line=dict(color="black", width=3),
                    fillcolor=fillcolor,
                    fill="tozeroy",
                    mode="lines",
                    showlegend=False,
                )
            )

    for index, convex_set in enumerate(vertex_sets):
        if isinstance(convex_set, Hyperrectangle):
            center = (convex_set.lb() + convex_set.ub()) / 2
        elif isinstance(convex_set, HPolyhedron):
            vertices = get_clockwise_vertices(VPolytope(convex_set))
            center = np.mean(vertices, axis=1)
        elif isinstance(convex_set, Hyperellipsoid):
            center = convex_set.center()
            
        add_trace(convex_set, str(index))

        if plot_all_names:
            delta= 0.0
            fig.add_trace(
                go.Scatter(
                    x=[center[0]],
                    y=[center[1]+delta],
                    mode="text",
                    textfont=dict(size=font_size),
                    text=["<i>"+str(index)+"</i>"],
                    showlegend=False,
                )
            )

    fig.update_layout(height=height, width=width)
    fig.update_layout(
        yaxis=dict(scaleanchor="x"),  # set y-axis to have the same scaling as x-axis
        yaxis2=dict(
            scaleanchor="x", overlaying="y", side="right"
        ),  # set y-axis2 to have the same scaling as x-axis
    )
    return fig

def update_layout(fig, tickfont=25):

    fig.update_layout(
        # paper_bgcolor='white',  # Background color of the paper (outside the plot area)
        plot_bgcolor='white'    # Background color of the plot area
    )

    fig.update_layout(
        xaxis=dict(
                gridcolor='lightgrey',
                linecolor='lightgrey',
                zeroline=False,
                dtick=1,
                tickfont=dict(size=tickfont),
                ),
        yaxis=dict(
                gridcolor='lightgrey',
                linecolor='lightgrey',
                zeroline=False,
                dtick=1,
                tickfont=dict(size=tickfont),
                )
    )
    fig.update_layout(
        xaxis=dict(
            showline=True,       # Show x-axis line
            linecolor='black',   # Set x-axis line color to black
            mirror=True          # Mirror the axis line
        ),
        yaxis=dict(
            showline=True,       # Show y-axis line
            linecolor='black',   # Set y-axis line color to black
            mirror=True          # Mirror the axis line
        )
    )

def plot_restriction(fig:go.Figure, restriction:RestrictionSolution, plot_arrows=True):
    # for q in restriction.trajectory:
    fig.add_trace(
            go.Scatter(
                x=[restriction.trajectory[0][0],restriction.trajectory[-1][0]],
                y=[restriction.trajectory[0][1],restriction.trajectory[-1][1]],
                mode="markers",
                
                marker=dict(
                    symbol="cross",
                    size=20,             # Set the size of the markers
                    color="black",
                    line=dict(
                        color="black",     # Set the outline color of the markers
                        width=0          # Set the width of the marker outline
                    )
                ),
                showlegend=False,
            )
        )
    fig.add_trace(
            go.Scatter(
                x=np.array(restriction.trajectory)[:,0],
                y=np.array(restriction.trajectory)[:,1],
                line=dict(color="blue", width=7),
                mode="lines+markers",
                marker=dict(
                    color="white",        # Set the fill color of the markers
                    size=9,             # Set the size of the markers
                    line=dict(
                        color="blue",     # Set the outline color of the markers
                        width=2          # Set the width of the marker outline
                    )
                ),
                showlegend=False,
            )
        )
    
    if plot_arrows:
        for i, u in enumerate(restriction.edge_variable_trajectory):
            arrow_start = restriction.trajectory[i][:2]
            arrow_end = arrow_start + u 
            if np.linalg.norm(u) > 1e-2:
                fig.add_trace(
                        go.Scatter(
                            x=[arrow_start[0],arrow_end[0]],
                            y=[arrow_start[1],arrow_end[1]],
                            marker_color="orange",
                            line=dict(width=4),
                            mode="lines+markers",
                            showlegend=False,
                            marker=dict(
                                symbol='arrow',
                                size=12,
                                angleref="previous",
                            ),
                        )
                    )