import typing as T  # pylint: disable=unused-import

import numpy as np
import numpy.typing as npt

from pydrake.geometry.optimization import (  # pylint: disable=import-error, no-name-in-module
    HPolyhedron,
    Point,
    ConvexSet,
    Hyperrectangle,
    VPolytope,
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
from plotly.express.colors import sample_colorscale  # pylint: disable=import-error
import plotly.graph_objs as go
from plotly.subplots import make_subplots

from bezier_dual_goal_conditioned import GoalConditionedPolynomialDualGCS
from program_options import FREE_POLY, PSD_POLY, CONVEX_POLY, ProgramOptions

import plotly
import plotly.graph_objs as go
from IPython.display import display, HTML

plotly.offline.init_notebook_mode()
display(HTML(
    '<script type="text/javascript" async src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/MathJax.js?config=TeX-MML-AM_SVG"></script>'
))

import logging
# logging.getLogger("drake").setLevel(logging.WARNING)
# np.set_printoptions(suppress=True) 

from random_graph_generator import random_uniform_graph_generator
from tqdm import tqdm

import yaml
from pydrake.all import RandomGenerator, PathParameterizedTrajectory # pylint: disable=import-error, no-name-in-module

from proper_gcs_policy import get_all_n_step_vertices_and_edges, get_all_n_step_paths, plot_optimal_and_rollout, obtain_rollout, get_optimal_path, get_path_cost
from plot_utils import plot_bezier, plot_a_2d_graph
from util import ChebyshevCenter, make_moment_matrix

from arm_visualization import visualize_arm_at_state, visualize_arm_at_samples_from_set, save_the_visualization, arm_components_loader, ArmComponents, create_arm, visualize_arm_at_state, make_path_paramed_traj_from_list_of_bezier_curves, Simulator


def reverse_trajectory(lbc: T.List[T.List[npt.NDArray]]):
    return [lbc[i][::-1] for i in range(len(lbc)-1, -1, -1)]

def make_a_rollout_sequence(graph_dict: T.Dict[str, GoalConditionedPolynomialDualGCS],
                            regions_names_list: T.List[str], 
                            make_optimal:bool, 
                            use_rohan_scenario:bool = False,
                            use_cheap:bool = True, 
                            num_timesteps:int = 1000,
                            random_seed: int = 1
    ) -> T.Tuple[T.List[float], T.List[float], T.List[PathParameterizedTrajectory]]:
    generator = RandomGenerator(random_seed)

    # Create arm
    arm_components = arm_components_loader(use_rohan_scenario, use_cheap, False)
    simulator = Simulator(arm_components.diagram)
    simulator.set_target_realtime_rate(1.0)

    index_now = 0
    terminal_state_now = None
    source_state_now = None

    trajectory_distances = []
    solve_times = []
    trajectories = []

    # generate all trajectories
    for index_now in tqdm(range(0, len(regions_names_list)-1)):
        if index_now % 2 == 0:
            source_name = regions_names_list[index_now+1]
            terminal_name = regions_names_list[index_now]
            must_reverse_now = True
        else:
            source_name = regions_names_list[index_now]
            terminal_name = regions_names_list[index_now+1]
            must_reverse_now = False
        graph = graph_dict[terminal_name]
        source_vertex = graph.vertices[source_name]
        terminal_vertex = graph.vertices[terminal_name]

        if index_now == 0:
            assert terminal_state_now is None
            terminal_state_now = terminal_vertex.convex_set.UniformSample(generator)

        if terminal_state_now is None:
            terminal_state_now = terminal_vertex.convex_set.UniformSample(generator)
        if source_state_now is None:
            source_state_now = source_vertex.convex_set.UniformSample(generator)
        
        if make_optimal:
            solve_time, _, rollout, v_rollout = get_optimal_path(graph, source_vertex, source_state_now, terminal_state=terminal_state_now)
        else:
            rollout, v_rollout, solve_time = obtain_rollout(graph, 1, source_vertex, source_state_now, None, terminal_state=terminal_state_now)

        rollout_cost = get_path_cost(graph, v_rollout, rollout, False, False, terminal_state=terminal_state_now)

        # reverse the trajectory if going back 
        if must_reverse_now:
            rollout = reverse_trajectory(rollout)
        retimed_traj = make_path_paramed_traj_from_list_of_bezier_curves(rollout, arm_components.plant, num_timesteps)
    
        
        trajectory_distances.append(rollout_cost)
        solve_times.append(solve_time)
        trajectories.append(retimed_traj)

        if must_reverse_now:
            terminal_state_now = None
        else:
            source_state_now = None

    return trajectory_distances, solve_times, trajectories


def visualize_trajectories( solve_times:T.List[float], 
                            trajectories:T.List[PathParameterizedTrajectory],
                            use_rohan_scenario:bool = False,
                            use_cheap:bool = True, 
                            num_timesteps:int = 1000)->ArmComponents:
    # Create arm
    arm_components = arm_components_loader(use_rohan_scenario, use_cheap, True)
    simulator = Simulator(arm_components.diagram)
    simulator.set_target_realtime_rate(1.0)
    context = simulator.get_mutable_context()
    plant_context = arm_components.plant.GetMyContextFromRoot(context)
    arm_components.meshcat_visualizer.StartRecording()
    time_now = 0.0
    total_solve_times = 0.0
    total_trajectory_time = 0.0

    # generate all trajectories
    for i, trajectory in enumerate(trajectories):
        solve_time = solve_times[i]

        

        # wait to simulate the optimization being solved
        stay_point = trajectory.value(0).flatten()
        wait_times = np.linspace(time_now, time_now+solve_time, num=200, endpoint=False)
        for t in wait_times:
            arm_components.plant.SetPositions(plant_context, stay_point)
            arm_components.plant.SetVelocities(plant_context, np.zeros(7)) 
            simulator.AdvanceTo(t)
        time_now += solve_time
        total_solve_times += solve_time

        
        q_numeric = np.empty((num_timesteps, arm_components.num_joints))
        q_dot_numeric = np.empty((num_timesteps, arm_components.num_joints))
        sample_times_s = np.linspace(
            trajectory.start_time(), trajectory.end_time(), num=num_timesteps, endpoint=True
        )
        for i, t in enumerate(sample_times_s):
            q_numeric[i] = trajectory.value(t).flatten()
            q_dot_numeric[i] = trajectory.EvalDerivative(t, derivative_order=1).flatten()

        sample_times_s += time_now
        
        for q, q_dot, t in zip(
            q_numeric,
            q_dot_numeric,
            sample_times_s
        ):
            arm_components.plant.SetPositions(plant_context, q)
            arm_components.plant.SetVelocities(plant_context, q_dot) 
            simulator.AdvanceTo(t)

        time_now += trajectory.end_time()
        total_trajectory_time += trajectory.end_time()

    YAY("------")
    INFO("whole duration", time_now)
    INFO("total solve time", total_solve_times)
    INFO("total movement duration", total_trajectory_time)

    # replay trajectories
    arm_components.meshcat_visualizer.StopRecording()
    arm_components.meshcat_visualizer.PublishRecording()

    return arm_components