import numpy as np

from pydrake.all import ( # pylint: disable=import-error, no-name-in-module, unused-import
    AddMultibodyPlantSceneGraph,
    DiagramBuilder,
    InverseDynamicsController,
    LogVectorOutput,
    MeshcatVisualizer,
    MultibodyPlant,
    PiecewisePolynomial,
    StartMeshcat,
    TrajectorySource,
)

from pydrake.all import ( # pylint: disable=import-error, no-name-in-module, unused-import
    BsplineBasis,
    BsplineTrajectory,
    Context,
    Diagram,
    Meshcat,
    MeshcatVisualizer,
    MultibodyPlant,
    Trajectory,
    TrajectorySource,
    VectorLogSink,
    Simulator,
    Parser,
)


import numpy as np


import typing as T
import numpy.typing as npt

from dataclasses import dataclass

def get_parser(plant: MultibodyPlant) -> Parser:
    """Creates a parser for a plant and adds package paths to it."""
    parser = Parser(plant)
    return parser
import os

from pydrake.trajectories import Trajectory, BezierCurve, CompositeTrajectory, PathParameterizedTrajectory # pylint: disable=import-error, no-name-in-module, unused-import
from pydrake.multibody.optimization import Toppra # pylint: disable=import-error, no-name-in-module, unused-import

from manipulation.utils import ConfigureParser

@dataclass
class ArmComponents:
    """
    A dataclass that contains all the robotic arm system components.
    """
    num_joints: int
    diagram: Diagram
    plant: MultibodyPlant
    trajectory_source: TrajectorySource
    meshcat: Meshcat
    meshcat_visualizer: MeshcatVisualizer
    

def create_arm(
    arm_file_path: str = "./iiwa.dmd.yaml",
    num_joints: int = 7,
    time_step: float = 0.0,
    use_meshcat: bool = True,
) -> ArmComponents:
    """Creates a robotic arm system.

    Args:
        arm_file_path (str): The URDF or SDFormat file of the robotic arm.
        num_joints (int): The number of joints of the robotic arm.
        time_step (float, optional): The time step to use for the plant. Defaults to 0.0.

    Returns:
        ArmComponents: The components of the robotic arm system.
    """

    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step)
    parser = get_parser(plant)
    ConfigureParser(parser)
    parser.package_map().AddPackageXml(filename=os.path.abspath("./models/package.xml"))

    # parser.AddModelsFromUrl(
    #     "package://manipulation/schunk_wsg_50_welded_fingers.sdf"
    # )

    # Add arm
    parser.AddModels(arm_file_path)
    try:
        arm = plant.GetModelInstanceByName("iiwa")
    except:
        arm = plant.GetModelInstanceByName("arm")
    plant.Finalize()

    placeholder_trajectory = PiecewisePolynomial(np.zeros((num_joints, 1)))
    trajectory_source = builder.AddSystem(
        TrajectorySource(placeholder_trajectory, output_derivative_order=1)
    )

    # Add Controller
    # controller_plant = MultibodyPlant(time_step)
    # controller_parser = get_parser(controller_plant)
    # controller_parser.AddModels(arm_file_path)
    # controller_plant.Finalize()
    # arm_controller = builder.AddSystem(
    #     InverseDynamicsController(
    #         controller_plant,
    #         kp=[100] * num_joints,
    #         kd=[10] * num_joints,
    #         ki=[1] * num_joints,
    #         has_reference_acceleration=False,
    #     )
    # )
    # arm_controller.set_name("arm_controller")
    # builder.Connect(
    #     plant.get_state_output_port(arm),
    #     arm_controller.get_input_port_estimated_state(),
    # )
    # builder.Connect(
    #     arm_controller.get_output_port_control(), plant.get_actuation_input_port(arm)
    # )
    # builder.Connect(
    #     trajectory_source.get_output_port(),
    #     arm_controller.get_input_port_desired_state(),
    # )

    # Meshcat
    if use_meshcat:
        meshcat = StartMeshcat()
        if num_joints < 3:
            meshcat.Set2dRenderMode()
        meshcat_visualizer = MeshcatVisualizer.AddToBuilder(
            builder, scene_graph, meshcat
        )
    else:
        meshcat = None
        meshcat_visualizer = None

    # state_logger = LogVectorOutput(plant.get_state_output_port(), builder)
    # commanded_torque_logger = LogVectorOutput(
    #     arm_controller.get_output_port_control(), builder
    # )

    diagram = builder.Build()

    return ArmComponents(
        num_joints=num_joints,
        diagram=diagram,
        plant=plant,
        trajectory_source=trajectory_source,
        meshcat=meshcat,
        meshcat_visualizer=meshcat_visualizer,
    )

def scenario_loader(arm_components: ArmComponents = None, 
                    use_rohan_scenario:bool = False,
                    use_meshcat:bool = True
                    ):
    if arm_components is None:
        if use_rohan_scenario:
            arm_components = create_arm(arm_file_path="./models/iiwa14_rohan.dmd.yaml", use_meshcat=use_meshcat)
        else:
            arm_components = create_arm(arm_file_path="./models/iiwa14_david.dmd.yaml", use_meshcat=use_meshcat)
    return arm_components


def reparameterize_with_toppra(
    trajectory: Trajectory,
    plant: MultibodyPlant,
    # velocity_limits: np.ndarray,
    # acceleration_limits: np.ndarray,
    num_grid_points: int = 1000,
) -> PathParameterizedTrajectory:
    toppra = Toppra(
        path=trajectory,
        plant=plant,
        gridpoints=np.linspace(
            trajectory.start_time(), trajectory.end_time(), num_grid_points
        ),
    )
    velocity_limits = np.min(
        [
            np.abs(plant.GetVelocityLowerLimits()),
            np.abs(plant.GetVelocityUpperLimits()),
        ],
        axis=0,
    )
    acceleration_limits=np.min(
        [
            np.abs(plant.GetAccelerationLowerLimits()),
            np.abs(plant.GetAccelerationUpperLimits()),
        ],
        axis=0,
    )
    toppra.AddJointVelocityLimit(-velocity_limits, velocity_limits)
    toppra.AddJointAccelerationLimit(-acceleration_limits, acceleration_limits)
    time_trajectory = toppra.SolvePathParameterization()
    return PathParameterizedTrajectory(trajectory, time_trajectory)

def make_path_paramed_traj_from_list_of_bezier_curves(list_of_list_of_lists: T.List[T.List[npt.NDArray]], plant: MultibodyPlant, num_grid_points = 1000):
    l_l_l = list_of_list_of_lists
    list_of_bez_curves = [BezierCurve(i, i+1, np.array(l_l_l[i]).T ) for i in range(len(l_l_l)) ]
    composite_traj = CompositeTrajectory(list_of_bez_curves)
    return reparameterize_with_toppra(composite_traj, plant, num_grid_points)



def visualize_a_trajectory(solution: T.List[T.List[npt.NDArray]], 
                           arm_components: ArmComponents = None, 
                           num_timesteps = 1000, 
                           use_rohan_scenario:bool = True, 
                           debug_t_start:float=None, debug_t_end:float=None):
    if debug_t_start is None:
        debug_t_start = 100
    if debug_t_end is None:
        debug_t_end = -1
    # Create arm
    arm_components = scenario_loader(arm_components, use_rohan_scenario)

    simulator = Simulator(arm_components.diagram)
    simulator.set_target_realtime_rate(1.0)

    context = simulator.get_mutable_context()
    plant_context = arm_components.plant.GetMyContextFromRoot(context)

    # Sample the trajectory
    traj = make_path_paramed_traj_from_list_of_bezier_curves(solution, arm_components.plant, num_timesteps)

    q_numeric = np.empty((num_timesteps, arm_components.num_joints))
    q_dot_numeric = np.empty((num_timesteps, arm_components.num_joints))
    # q_ddot_numeric = np.empty((num_timesteps, arm_components.num_joints))
    sample_times_s = np.linspace(
        traj.start_time(), traj.end_time(), num=num_timesteps, endpoint=True
    )
    for i, t in enumerate(sample_times_s):
        q_numeric[i] = traj.value(t).flatten()
        q_dot_numeric[i] = traj.EvalDerivative(t, derivative_order=1).flatten()

    arm_components.meshcat_visualizer.StartRecording()
    for q, q_dot, t in zip(
        q_numeric,
        q_dot_numeric,
        sample_times_s
    ):
        if debug_t_start <= t and t <= debug_t_end:
            print(q)
        arm_components.plant.SetPositions(plant_context, q)
        arm_components.plant.SetVelocities(plant_context, q_dot) 
        simulator.AdvanceTo(t)

    arm_components.meshcat_visualizer.StopRecording()
    arm_components.meshcat_visualizer.PublishRecording()


def visualize_arm_at_state(state:npt.NDArray, arm_components: ArmComponents = None, use_rohan_scenario = False):
    # Create arm
    arm_components = scenario_loader(arm_components, use_rohan_scenario)

    simulator = Simulator(arm_components.diagram)
    simulator.set_target_realtime_rate(1.0)

    context = simulator.get_mutable_context()
    plant_context = arm_components.plant.GetMyContextFromRoot(context)

    arm_components.meshcat_visualizer.StartRecording()
    arm_components.plant.SetPositions(plant_context, state)
    simulator.AdvanceTo(0)
    arm_components.plant.SetPositions(plant_context, state)

    arm_components.meshcat_visualizer.StopRecording()
    arm_components.meshcat_visualizer.PublishRecording()


def visualize_arm_at_init(arm_components: ArmComponents = None, use_rohan_scenario = False):
    visualize_arm_at_state(np.zeros(7), arm_components, use_rohan_scenario)