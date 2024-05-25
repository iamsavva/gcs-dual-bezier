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

import os

from pydrake.trajectories import Trajectory, BezierCurve, CompositeTrajectory, PathParameterizedTrajectory # pylint: disable=import-error, no-name-in-module, unused-import
from pydrake.multibody.optimization import Toppra # pylint: disable=import-error, no-name-in-module, unused-import
from pydrake.all import RandomGenerator, ConvexSet # pylint: disable=import-error, no-name-in-module, unused-import

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
    diagram_context: Context
    

def create_arm(
    arm_file_path: str,
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
    parser = Parser(plant)
    ConfigureParser(parser)
    parser.package_map().AddPackageXml(filename=os.path.abspath("./models/package.xml"))

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

    diagram = builder.Build()

    diagram_context = diagram.CreateDefaultContext()
    # plant_context = plant.GetMyContextFromRoot(diagram_context)
    diagram.ForcedPublish(diagram_context)

    return ArmComponents(
        num_joints=num_joints,
        diagram=diagram,
        plant=plant,
        trajectory_source=trajectory_source,
        meshcat=meshcat,
        meshcat_visualizer=meshcat_visualizer,
        diagram_context=diagram_context
    )

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

def arm_components_loader(use_rohan_scenario:bool = False,
                          use_cheap: bool = True,
                          use_meshcat:bool = True
                            ):
    """
    a very dumb loader for the arm components.

    use_rohan_scenario = True -- load left-right bin scenario
    use_rohan_scenario = False -- load shelves and left right bins scenario
    use_cheap = True -- load iiwa7 with box collision geometry
    use_cheap = False -- load iiwa14 with cylinder collision geometry
    """
    if use_rohan_scenario:
        if use_cheap:
            arm_components = create_arm(arm_file_path="./models/iiwa14_rohan_cheap.dmd.yaml", use_meshcat=use_meshcat)
        else:
            arm_components = create_arm(arm_file_path="./models/iiwa14_rohan.dmd.yaml", use_meshcat=use_meshcat)
    else:
        if use_cheap:
            arm_components = create_arm(arm_file_path="./models/iiwa14_david_cheap.dmd.yaml", use_meshcat=use_meshcat)
        else:
            arm_components = create_arm(arm_file_path="./models/iiwa14_david.dmd.yaml", use_meshcat=use_meshcat)
    return arm_components



def visualize_arm_at_state(state:npt.NDArray, use_rohan_scenario:bool = True, use_cheap:bool =True):
    """
    draw the arm at a provided state
    """
    print("YO")
    assert len(state) == 7
    arm_components = arm_components_loader(use_rohan_scenario, use_cheap, True)

    simulator = Simulator(arm_components.diagram)
    simulator.set_target_realtime_rate(1.0)

    context = simulator.get_mutable_context()
    plant_context = arm_components.plant.GetMyContextFromRoot(context)
    plant = arm_components.plant

    arm_components.meshcat_visualizer.StartRecording()
    plant.SetPositions(plant_context, state)
    simulator.AdvanceTo(0)
    plant.SetPositions(plant_context, state)

    arm_components.meshcat_visualizer.StopRecording()
    arm_components.meshcat_visualizer.PublishRecording()


def visualize_arm_at_init(use_rohan_scenario:bool = True, use_cheap:bool = True):
    """
    draw the arm at the 0000000 state 
    """
    visualize_arm_at_state(np.zeros(7), use_rohan_scenario, use_cheap)


def visualize_arm_at_samples_from_set(convex_set: ConvexSet, num_samples:int = 30, dt:float = 0.1, use_rohan_scenario:bool=True, use_cheap:bool = True, recording_name = None):
    """
    draw a bunch of samples from the set

    Args:
        convex_set (ConvexSet): convex set from which to draw samples
        num_samples (int, optional): number of samples to draw
        dt (float, optional): amount of time to wait between visualizing individual samples
        use_rohan_scenario (bool, optional): use Rohan's or David's scenario
        use_cheap: use simple box or cylinder geometry
    """
    # collect a bunch of samples 
    np.random.seed(0)
    sample = None
    generator = RandomGenerator(1)
    samples = []
    for _ in range(0,num_samples):    
        if sample is not None:
            sample = convex_set.UniformSample(generator, sample)
        else:
            sample = convex_set.UniformSample(generator)
        samples.append(sample)

    # star the sim
    arm_components = arm_components_loader(use_rohan_scenario, use_cheap, True)
    simulator = Simulator(arm_components.diagram)
    simulator.set_target_realtime_rate(1.0)
    context = simulator.get_mutable_context()
    plant_context = arm_components.plant.GetMyContextFromRoot(context)

    arm_components.meshcat_visualizer.StartRecording()
    # draw a new sample every dt
    t = 0
    N = 20
    for sample in samples:
        # draw the sample 20 times in the span of dt -- so that the arm doesn't move at all
        for _ in range(N):
            t += dt/N
            arm_components.plant.SetPositions(plant_context, sample)
            arm_components.plant.SetVelocities(plant_context, np.zeros(7))
            simulator.AdvanceTo(t)

    arm_components.meshcat_visualizer.StopRecording()
    arm_components.meshcat_visualizer.PublishRecording()

    if recording_name is not None:
        save_the_visualization(arm_components, recording_name)

def save_the_visualization(arm_components: ArmComponents, recording_name:str):
    """
    save the visualization into an html file -- for future use
    """
    link = arm_components.meshcat.StaticHtml()
    with open(recording_name + ".html", "w+") as f:
        f.write(link)

