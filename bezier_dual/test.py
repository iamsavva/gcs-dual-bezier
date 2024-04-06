import typing as T  # pylint: disable=unused-import

import numpy as np
import numpy.typing as npt

from pydrake.solvers import (  # pylint: disable=import-error, no-name-in-module, unused-import
    MathematicalProgram,
    MathematicalProgramResult,
    Solve,
    MosekSolver,
    MosekSolverDetails,
    SolverOptions,
    CommonSolverOption,
    IpoptSolver,
    SnoptSolver,
    GurobiSolver
)
from pydrake.geometry.optimization import (  # pylint: disable=import-error, no-name-in-module
    HPolyhedron,
    Point,
    ConvexSet,
    Hyperrectangle,
    VPolytope
)

from pydrake.symbolic import ( # pylint: disable=import-error, no-name-in-module, unused-import
    Polynomial,
    Variable,
    Variables,
    Expression,
)  

from util import (
    timeit,
    diditwork,
    INFO,
    YAY,
    WARN,
    ERROR,
)  # pylint: disable=import-error, no-name-in-module, unused-import

import plotly.graph_objects as go  # pylint: disable=import-error
from plotly.express.colors import sample_colorscale  # pylint: disable=import-error
import plotly.graph_objs as go
from plotly.subplots import make_subplots

from bezier_dual import PolynomialDualGCS, QUADRATIC_COST, DualVertex, DualEdge, QUADRATIC_COST_AUGMENTED
from program_options import FREE_POLY, PSD_POLY, CONVEX_POLY, ProgramOptions

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

from collections import deque

from proper_gcs_policy import solve_convex_restriction, get_all_n_step_vertices_and_edges, get_all_n_step_paths, plot_optimal_and_rollout
from plot_utils import plot_bezier, plot_a_2d_graph

from random_graph_generator import random_uniform_graph_generator




options = ProgramOptions()
options.pot_type = CONVEX_POLY
# options.pot_type = FREE_POLY
options.num_control_points = 4

delta = 0.0001
x_star = np.array([10,0])
QUADRATIC_COST_AUGMENTED = lambda x,y: np.sum([(x[i]-y[i])**2 for i in range(len(x)) ]) + delta * ( np.sum([(x[i]-x_star[0])**2 + (y[i]-x_star[1])**2 for i in range(len(x)) ]) )
QUADRATIC_COST = lambda x,y: np.sum([(x[i]-y[i])**2 for i in range(len(x)) ])
cost_function = QUADRATIC_COST_AUGMENTED
# cost_function = QUADRATIC_COST

options.MSK_DPAR_INTPNT_CO_TOL_DFEAS=1e-9
options.MSK_DPAR_INTPNT_CO_TOL_PFEAS=1e-9
options.MSK_DPAR_INTPNT_CO_TOL_REL_GAP=1e-9

options.policy_add_G_term = False

options.max_flow_through_edge = 1

options.policy_verbose_choices=False
options.policy_add_violation_penalties = True
options.max_flow_through_edge = 1


# options.use_lookahead_policy = True
# options.use_lookahead_with_backtracking_policy=True
options.use_cheap_a_star_policy=True

gcs, layers, start_vertex = random_uniform_graph_generator(options=options, cost_function=cost_function, use_bidirecitonal_edges=True, num_layers=20, random_seed=0, x_min=0, x_max=40, min_region=2, max_region=4, min_blank=0.1, max_blank=2, min_goal_blank=10, max_goal_blank=12, goal_num=1)

gcs.options.MSK_DPAR_INTPNT_CO_TOL_DFEAS=5e-9
gcs.options.MSK_DPAR_INTPNT_CO_TOL_PFEAS=5e-9
gcs.options.MSK_DPAR_INTPNT_CO_TOL_REL_GAP=5e-9
gcs.solve_policy()




fig = plot_a_2d_graph(gcs.vertices.values())

total_dt = 0.0
num_successes = 0
for i in range(0, 41, 2):
    gcs.options.postprocess_by_solving_restriction_on_mode_sequence=True
    gcs.options.policy_add_G_term = False

    gcs.options.use_lookahead_policy=True
    gcs.options.use_lookahead_with_backtracking_policy=False
    gcs.options.use_cheap_a_star_policy = False

    gcs.options.policy_verbose_number_of_restrictions_solves=True

    gcs.options.policy_add_violation_penalties = False

    # gcs.options.policy_use_zero_heuristic = False
    # success, dt = plot_optimal_and_rollout(fig, gcs, 1, start_vertex, np.array([i,40]), None, plot_optimal=False, rollout_color="blue", plot_control_points=False, linewidth=4)


    gcs.options.use_lookahead_policy=True
    gcs.options.use_lookahead_with_backtracking_policy=False
    gcs.options.use_cheap_a_star_policy = False
    gcs.options.solve_with_snopt=False
    gcs.options.solve_with_gurobi = True
    gcs.options.policy_use_zero_heuristic = False
    gcs.options.policy_add_violation_penalties = False

    gcs.options.gcs_policy_use_convex_relaxation = True
    gcs.options.gcs_policy_max_rounding_trials = 0
    # success, dt = plot_optimal_and_rollout(fig, gcs, 2, start_vertex, np.array([i,40]), None, plot_optimal=True, rollout_color="red", plot_control_points=False, linewidth=4, verbose_time_comparison_to_optimal=True)

    success, dt = plot_optimal_and_rollout(fig, gcs, 2, start_vertex, np.array([i,40]), None, plot_optimal=True, rollout_color="red", plot_control_points=False, linewidth=4, verbose_time_comparison_to_optimal=True)
    break

    # gcs.options.use_lookahead_policy=True
    # gcs.options.use_lookahead_with_backtracking_policy=False
    # gcs.options.use_cheap_a_star_policy = False
    # gcs.options.solve_with_snopt=False
    # gcs.options.solve_with_gurobi = True
    # gcs.options.policy_use_zero_heuristic = False
    # gcs.options.policy_add_violation_penalties = False
    # success, dt = plot_optimal_and_rollout(fig, gcs, 3, start_vertex, np.array([i,40]), None, plot_optimal=False, rollout_color="green", plot_control_points=False, linewidth=4)

    # gcs.options.solve_with_snopt=False
    # gcs.options.solve_with_gurobi = True
    # gcs.options.policy_use_zero_heuristic = False
    # gcs.options.policy_add_violation_penalties = True
    # success, dt = plot_optimal_and_rollout(fig, gcs, 1, start_vertex, np.array([i,40]), None, plot_optimal=False, rollout_color="blue", plot_control_points=False, linewidth=4)

    # gcs.options.solve_with_snopt = False
    # gcs.options.solve_with_gurobi = True
    # gcs.options.policy_use_zero_heuristic = True
    # gcs.options.policy_add_violation_penalties = False
    # success, dt = plot_optimal_and_rollout(fig, gcs, 1, start_vertex, np.array([i,40]), None, plot_optimal=False, rollout_color="blue", plot_control_points=False, linewidth=4)

mean_dt = total_dt / num_successes
YAY("average solve time is", mean_dt)
fig.show()

