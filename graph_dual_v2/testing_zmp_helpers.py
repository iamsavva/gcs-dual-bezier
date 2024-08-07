import typing as T  # pylint: disable=unused-import

import numpy as np
import numpy.typing as npt

from pydrake.geometry.optimization import (  # pylint: disable=import-error, no-name-in-module
    HPolyhedron,
    Point,
    ConvexSet,
    Hyperrectangle,
    VPolytope,
    Hyperellipsoid,
    LoadIrisRegionsYamlFile,
)



from pydrake.symbolic import ( # pylint: disable=import-error, no-name-in-module, unused-import
    Polynomial,
    Variable,
    Variables,
    Expression,
)  

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
    GurobiSolver,
    OsqpSolver,
    ClarabelSolver,
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
# from plotly.express.colors import sample_colorscale  # pylint: disable=import-error
# import plotly.graph_objs as go
# from plotly.subplots import make_subplots

from gcs_dual import DualEdge, DualVertex, PolynomialDualGCS
from program_options import FREE_POLY, PSD_POLY, CONVEX_POLY, ProgramOptions

import plotly # pylint: disable=import-error, no-name-in-module, unused-import
# import plotly.graph_objs as go # pylint: disable=import-error, no-name-in-module, unused-import
from IPython.display import display, HTML

# plotly.offline.init_notebook_mode()
# display(HTML(
#     '<script type="text/javascript" async src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/MathJax.js?config=TeX-MML-AM_SVG"></script>'
# ))

import logging
# logging.getLogger("drake").setLevel(logging.WARNING)
# np.set_printoptions(suppress=True) 
from proper_gcs_policy import obtain_rollout, RestrictionSolution, precompute_k_step_feasible_paths_from_every_vertex, double_integrator_postprocessing
from plot_utils import plot_bezier, plot_a_2d_graph
from util import ChebyshevCenter

# from tqdm import tqdm

import yaml

from arm_visualization import visualize_arm_at_state, visualize_arm_at_samples_from_set, save_the_visualization, arm_components_loader, ArmComponents, create_arm, visualize_arm_at_state, make_path_paramed_traj_from_list_of_bezier_curves, Simulator
# from pydrake.math import RigidTransform, RollPitchYaw

# from pydrake.all import RandomGenerator, PathParameterizedTrajectory

from util import concatenate_polyhedra

from util_moments import (
    extract_moments_from_vector_of_spectrahedron_prog_variables, 
    make_moment_matrix, 
    get_moment_matrix_for_a_measure_over_set,
    make_product_of_indepent_moment_matrices,
    verify_necessary_conditions_for_moments_supported_on_set,
)

np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})


import matplotlib.pyplot as plt
import numpy as np
from IPython.display import HTML, display
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
from matplotlib.patches import Rectangle, Arrow, FancyArrow


def check_that_foot_forces_exist(restriction: RestrictionSolution, z, m, g, w, h, mu):
    failed = False
    for i in range(restriction.length()-1):
        restriction.trajectory = np.array(restriction.trajectory)
        restriction.edge_variable_trajectory = np.array(restriction.edge_variable_trajectory)
        x_com = restriction.trajectory[i][0:2]
        left_foot = restriction.trajectory[i][4:6]
        right_foot = restriction.trajectory[i][6:8]
        u = restriction.edge_variable_trajectory[i][0:2]
        cop = x_com - (z/g) * u
        v_name = restriction.vertex_path[i].name
        corners = []
        if v_name in ("Ld_Rd_1", "Ld_Rd_2", "Ld_Ru_1", "Ld_Ru_2", "target"):
            corners.append( left_foot + [w,h] )
            corners.append( left_foot + [-w,h] )
            corners.append( left_foot + [w,-h] )
            corners.append( left_foot + [-w,-h] )
        if v_name in ("Ld_Rd_1", "Ld_Rd_2", "Lu_Rd_1", "Lu_Rd_2", "target"):
            corners.append( right_foot + [w,h] )
            corners.append( right_foot + [-w,h] )
            corners.append( right_foot + [w,-h] )
            corners.append( right_foot + [-w,-h] )
        prog = MathematicalProgram()
        forces = []
        constants = []
        for _ in corners:
            f = prog.NewContinuousVariables(3)
            c = prog.NewContinuousVariables(1)[0]
            prog.AddLinearConstraint(f[2] >= 0) # non-negative normal force
            prog.AddLinearConstraint(c >= 0) # coefficients non-negative
            prog.AddLorentzConeConstraint(mu*f[2], f[0]**2+f[1]**2) # forces in the friction cone
            forces.append(f)
            constants.append(c)

        prog.AddLinearConstraint( np.sum([f[2] for f in forces]) == m*g ) # total normal force
        prog.AddLinearConstraint( np.sum([f[1] for f in forces]) == u[1] ) # total control in y
        prog.AddLinearConstraint( np.sum([f[0] for f in forces]) == u[0] ) # total control in x
        prog.AddLinearConstraint( np.sum([c for c in constants]) == 1 ) # inside support
        prog.AddLinearConstraint( np.sum([constants[i]*corners[i][0] for i in range(len(corners))]) == cop[0] )
        prog.AddLinearConstraint( np.sum([constants[i]*corners[i][1] for i in range(len(corners))]) == cop[1] )
        solver = MosekSolver()
        solver_options = SolverOptions()
        solver_options.SetOption(
            MosekSolver.id(),
            "MSK_DPAR_INTPNT_CO_TOL_REL_GAP",
            1e-4
        )
        solver_options.SetOption(
            MosekSolver.id(),
            "MSK_DPAR_INTPNT_CO_TOL_PFEAS",
            1e-4
        )
        solver_options.SetOption(
            MosekSolver.id(),
            "MSK_DPAR_INTPNT_CO_TOL_DFEAS",
            1e-4
        )
        solution = solver.Solve(prog, solver_options=solver_options)
        
        if not solution.is_success():
            failed = True
            WARN("failed at timestep ", i)
            WARN(restriction.trajectory[i])
            break
    if not failed:
        YAY("forces are good")

def current_cop(xl,u,z,g):
    return xl[0:2] - (z/g)* u[0:2]


class TempData:
    def __init__(self, restriction: RestrictionSolution, dt:float, z, g) -> None:
        restriction.trajectory = np.array(restriction.trajectory)
        restriction.edge_variable_trajectory = np.array(restriction.edge_variable_trajectory)
        self.com_positions = np.array(restriction.trajectory[:, 0:2])
        self.com_velocities = np.array(restriction.trajectory[:, 2:4])
        self.accelerations = np.array(restriction.edge_variable_trajectory[:, 0:2])
        self.cop_positions = [
            current_cop(restriction.trajectory[i], restriction.edge_variable_trajectory[i], z,g) for i in range(restriction.length()-1)
            ]
        # self.cop_positions.append(next_cop(None, restriction.edge_variable_trajectory[-1], restriction.trajectory[-1]))
        self.cop_positions.append(self.com_positions[-1])
        self.cop_positions = np.array(self.cop_positions)
        self.left_foot_positions = np.array(restriction.trajectory[:, 4:6])
        self.right_foot_positions = np.array(restriction.trajectory[:, 6:8])
        self.vertex_path = np.array(restriction.vertex_names())
        self.dt = dt
        self.time_traj = np.array(list(range(0, len(self.right_foot_positions)))) * dt

    def interpolate(self, num_points:int):
        if num_points > 2:
            small_time_traj = np.hstack([np.linspace(self.time_traj[i], self.time_traj[i+1], num_points)[:-1] for i in range(len(self.time_traj)-1)] + [self.time_traj[-1]])
            small_pos_traj = np.zeros((len(small_time_traj), 2))
            small_vel_traj = np.zeros((len(small_time_traj), 2))
            small_acc_traj = np.zeros((len(small_time_traj)-1, 2))
            small_cop_traj = np.zeros((len(small_time_traj), 2))

            small_left_positions = np.zeros((len(small_time_traj), 2))
            small_right_positions = np.zeros((len(small_time_traj), 2))

            for i in range(len(self.time_traj)-1):
                for j in range(num_points-1):
                    small_acc_traj[i*(num_points-1) + j] = self.accelerations[i]

            small_pos_traj[0] = self.com_positions[0]
            small_vel_traj[0] = self.com_velocities[0]
            small_vertex_path = []

            for i in range(len(small_pos_traj)-1):
                small_delta_t = small_time_traj[i+1] - small_time_traj[i]
                small_pos_traj[i+1] = small_pos_traj[i] + small_vel_traj[i] * small_delta_t +  small_acc_traj[i] * small_delta_t ** 2 / 2
                small_vel_traj[i+1] = small_vel_traj[i] + small_acc_traj[i] * small_delta_t

                small_vertex_path.append(self.vertex_path[i//(num_points-1)])

                # small_cop_traj[i] = small_pos_traj[i] - (z/g) * small_acc_traj[i]
                small_cop_traj[i] = self.cop_positions[i//(num_points-1)]
                c1 = (num_points -1 - i % (num_points-1))/(num_points-1)
                c2 = 1-c1
                small_left_positions[i] = self.left_foot_positions[i // (num_points-1)] * c1 + self.left_foot_positions[i // (num_points-1)+1] * c2
                small_right_positions[i] = self.right_foot_positions[i // (num_points-1)] * c1 + self.right_foot_positions[i // (num_points-1)+1] * c2
                # small_cop_traj[i] =  self.cop_positions[i // (num_points-1)] * c1 + self.cop_positions[i // (num_points-1)+1] * c2


            small_cop_traj[-1] = self.cop_positions[-1]
            small_left_positions[-1] = self.left_foot_positions[-1]
            small_right_positions[-1] = self.right_foot_positions[-1]
            small_vertex_path.append(self.vertex_path[-1])
            small_vertex_path = np.array(small_vertex_path)

            assert np.allclose(self.com_positions[-1], small_pos_traj[-1], atol=1e-3)

            self.com_positions = small_pos_traj
            self.com_velocities = small_vel_traj
            self.accelerations = small_acc_traj
            self.cop_positions = small_cop_traj
            self.left_foot_positions = small_left_positions
            self.right_foot_positions = small_right_positions
            self.vertex_path = small_vertex_path
            self.dt = self.dt/num_points
            self.time_traj = small_time_traj


# helper function that plots a rectangle with given center, width, and height
def plot_rectangle(center, width, height, ax=None, **kwargs):
    # make black the default edgecolor
    if not "edgecolor" in kwargs:
        kwargs["edgecolor"] = "black"

    # make transparent the default facecolor
    if not "facecolor" in kwargs:
        kwargs["facecolor"] = "none"

    # get current plot axis if one is not given
    if ax is None:
        ax = plt.gca()

    # get corners
    c2c = np.array([width, height])
    bottom_left = center - c2c
    top_right = center + c2c

    # plot rectangle
    rect = Rectangle(bottom_left, 2*width, 2*height, **kwargs)
    ax.add_patch(rect)

    # scatter fake corners to update plot limits (bad looking but compact)
    ax.scatter(*bottom_left, s=0)
    ax.scatter(*top_right, s=0)

    # make axis scaling equal
    ax.set_aspect("equal")

    return rect

def animate_footstep_plan(
    width, 
    height, 
    dt, 
    restriction: RestrictionSolution, 
    z,
    g,
    xlim=[-1,1], 
    ylim=[-1,1], 
    scale_time=1,
    num_interpolation_points = 5,
    velocity_scale = 0.1,
    bbox_to_anchor = (-0.2,-0.1,0,0),
):
    # initialize figure for animation
    fig, ax = plt.subplots()

    # initial position of the feet
    com_position = ax.scatter(restriction.trajectory[0][0], restriction.trajectory[0][1], color="black", zorder=5, label="CoM")
    cop = current_cop(restriction.trajectory[0], restriction.edge_variable_trajectory[0], z,g)
    zmp_position = ax.scatter(cop[0], cop[1], color="red", zorder=6, label="ZMP")

    velocity = FancyArrow(x=restriction.trajectory[0][0], 
                     y=restriction.trajectory[0][1], 
                     dx=velocity_scale*restriction.trajectory[0][2], 
                     dy=velocity_scale*restriction.trajectory[0][3], 
                     width=0.01,
                     zorder = 4,
                     color="grey")
    ax.add_patch(velocity)


    # initial step limits
    left_limits = plot_rectangle(
        restriction.trajectory[0][4:6],  # center
        width,  # width
        height,  # eight
        ax=ax,
        edgecolor="blue",
        zorder=3,
        label="Left foot",
    )
    right_limits = plot_rectangle(
        restriction.trajectory[0][6:8],  # center
        width,  # width
        height,  # eight
        ax=ax,
        edgecolor="orange",
        zorder=3,
        label="Right foot",
    )

    ax.scatter(restriction.trajectory[-1][0], restriction.trajectory[-1][1], marker="x", color="magenta", zorder=1, label="target")
    # initial step limits
    # target_limits_left = plot_rectangle(
    #     restriction.trajectory[-1][4:6],  # center
    #     width,  # width
    #     height,  # eight
    #     ax=ax,
    #     edgecolor="magenta",
    #     zorder=1,
    # )
    # target_limits_right = plot_rectangle(
    #     restriction.trajectory[-1][6:8],  # center
    #     width,  # width
    #     height,  # eight
    #     ax=ax,
    #     edgecolor="magenta",
    #     zorder=1,
    # )

    # misc settings
    plt.close()
                
    
    temp_data = TempData(restriction, dt, z, g)
    temp_data.interpolate(num_interpolation_points)

    def animate(i):
        # scatter feet
        com_position.set_offsets(temp_data.com_positions[i])

        zmp_position.set_offsets(temp_data.cop_positions[i])

        velocity.set_data(x = temp_data.com_positions[i][0],
                          y = temp_data.com_positions[i][1],
                          dx = velocity_scale*temp_data.com_velocities[i][0],
                          dy = velocity_scale*temp_data.com_velocities[i][1])


        # limits of reachable set for each foot
        c2c = np.array([width, height])
        left_limits.set_xy(temp_data.left_foot_positions[i] - c2c)
        right_limits.set_xy(temp_data.right_foot_positions[i] - c2c)
        if temp_data.vertex_path[i] in ("target", "Ld_Rd_2", "Ld_Rd_1", "Ld_Ru_2", "Ld_Ru_1"):
            left_limits.set_facecolor("blue")
        else:
            left_limits.set_facecolor("none")
        if temp_data.vertex_path[i] in ("target", "Ld_Rd_2", "Ld_Rd_1", "Lu_Rd_1", "Lu_Rd_2"):
            right_limits.set_facecolor("orange")
        else:
            right_limits.set_facecolor("none")
    
    
    ax.legend(loc="upper left", bbox_to_anchor=bbox_to_anchor, ncol=5)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    fig.subplots_adjust(left=0.1, right=0.9, top=0.95, bottom=0.15)
    

    # create ad display animation
    n_steps = len(temp_data.time_traj)
    ani = FuncAnimation(fig, animate, frames=n_steps, interval=temp_data.dt*1000*scale_time)
    display(HTML(ani.to_jshtml()))
    # if filename is not None:
    #     # ani.save(filename + '.mp4', writer='ffmpeg', fps=30)
    #     # FFwriter = animation.FFMpegWriter(fps=10)
    #     Writer = animation.writers['ffmpeg']
    #     writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
    #     ani.save('temp.mp4', writer = Writer)
