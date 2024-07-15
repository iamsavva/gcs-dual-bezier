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

from gcs_dual import DualEdge, DualVertex, PolynomialDualGCS
from program_options import FREE_POLY, PSD_POLY, CONVEX_POLY, ProgramOptions

import plotly
import plotly.graph_objs as go
from IPython.display import display, HTML


import logging
# logging.getLogger("drake").setLevel(logging.WARNING)
# np.set_printoptions(suppress=True) 
from proper_gcs_policy import obtain_rollout
from plot_utils import plot_bezier, plot_a_2d_graph
from util import ChebyshevCenter

from tqdm import tqdm

import yaml

from arm_visualization import visualize_arm_at_state, visualize_arm_at_samples_from_set, save_the_visualization, arm_components_loader, ArmComponents, create_arm, visualize_arm_at_state, make_path_paramed_traj_from_list_of_bezier_curves, Simulator
from pydrake.math import RigidTransform, RollPitchYaw

from pydrake.all import RandomGenerator, PathParameterizedTrajectory

from iiwa_paper_data_collection import make_a_rollout_sequence, visualize_trajectories
from util import concatenate_polyhedra


def produce_a_policy(convex_set_dict: T.Dict[str, HPolyhedron], delta_t:float, terminal_set_name:str, start_regions_names:T.List[str], delta_surr:int, target_box_x:float=0.1, target_box_x_dot:float=0.05,):
    
    # info
    INFO("number of sets:", len(convex_set_dict.keys()))

    arm_components = arm_components_loader(use_meshcat=False)
    vel_lb = arm_components.plant.GetVelocityLowerLimits()
    vel_ub = arm_components.plant.GetVelocityUpperLimits()
    vel_limits = Hyperrectangle(vel_lb, vel_ub)
    acc_lb = arm_components.plant.GetAccelerationLowerLimits()
    acc_ub = arm_components.plant.GetAccelerationUpperLimits()
    acc_limits = Hyperrectangle(acc_lb, acc_ub)

    def cost_function_surrogate(xl,u,xr,xt):
        return 1 + delta_surr*((xl-xt).dot(xl-xt) + (xr-xt).dot(xr-xt))

    def cost_function(xl,u,xr,xt):
        return 1 

    edge_name_pairs = []
    set_names = list(convex_set_dict.keys())
    for i, name1 in enumerate(set_names):
        set1 = convex_set_dict[name1]
        edge_name_pairs.append((name1, name1))
        for j in range(i+1, len(set_names)):
            name2 = set_names[j]
            set2 = convex_set_dict[name2]
            intersection = set1.Intersection(set2)
            not_empty, _, r = ChebyshevCenter(intersection)
            if not_empty:
                assert not_empty == (not intersection.IsEmpty())
                assert r > 1e-5
                edge_name_pairs.append((name1, name2))
    INFO("number of edges:", len(edge_name_pairs), "bidirectional edges")


    options = ProgramOptions()
    options.potential_poly_deg = 2
    options.pot_type = PSD_POLY

    options.allow_vertex_revisits = True
    options.forward_iteration_limit = 10000
    options.use_lookahead_policy = False
    options.use_lookahead_with_backtracking_policy = True
    options.s_procedure_multiplier_degree_for_linear_inequalities = 0
    options.s_procedure_take_product_of_linear_constraints = False
    options.postprocess_by_solving_restriction_on_mode_sequence=True

    options.value_synthesis_use_robust_mosek_parameters = True
    options.value_synthesis_MSK_DPAR_INTPNT_CO_TOL_REL_GAP = 1e-9
    options.value_synthesis_MSK_DPAR_INTPNT_CO_TOL_PFEAS = 1e-9
    options.value_synthesis_MSK_DPAR_INTPNT_CO_TOL_DFEAS = 1e-9

    options.right_point_inside_intersection = True
    options.policy_use_target_condition_only=True
    options.use_add_sos_constraint = False

    target_small_box = np.hstack((target_box_x*np.ones(7), target_box_x_dot*np.ones(7)))
    terminating_condition = Hyperrectangle(-target_small_box, target_small_box)
    source_condition_vel = Hyperrectangle(-target_box_x_dot*np.ones(7), target_box_x_dot*np.ones(7))

    # create vertices:
    target_box_x_dot
    graph = PolynomialDualGCS(options, 
                              concatenate_polyhedra((convex_set_dict[terminal_set_name], source_condition_vel)), 
                              target_policy_terminating_condition=terminating_condition )

    INFO("adding vertices")
    # TODO: consider making a duplicate here?
    for vertex_name in tqdm(convex_set_dict.keys()):
        convex_set = convex_set_dict[vertex_name]
        hpoly = concatenate_polyhedra((convex_set, vel_limits))
        graph.AddVertex(vertex_name, hpoly)


    INFO("adding edges")
    def groebner_evaluator1(xl,u,xr,xt):
        return xr[0:7] - (xl[0:7] + xl[7:14]*delta_t + u*delta_t**2/2)
    def groebner_evaluator2(xl,u,xr,xt):
        return xr[7:14] - (xl[7:14] + u*delta_t)
    for (left_name, right_name) in tqdm(edge_name_pairs):
        # need to ensure that no edges go into terminal vertices
        temp_edges = [] # type: T.List[DualEdge]
        if left_name == terminal_set_name:
            temp_edges.append(graph.AddEdge(graph.vertices[right_name], graph.vertices[left_name], cost_function, cost_function_surrogate))
        elif right_name == terminal_set_name:
            temp_edges.append(graph.AddEdge(graph.vertices[left_name], graph.vertices[right_name], cost_function, cost_function_surrogate))
        else:
            temp_edges.append(graph.AddEdge(graph.vertices[left_name], graph.vertices[right_name], cost_function, cost_function_surrogate))
            if left_name != right_name:
                temp_edges.append(graph.AddEdge(graph.vertices[right_name], graph.vertices[left_name], cost_function, cost_function_surrogate))
        for edge in temp_edges:
            edge.u = graph.prog.NewIndeterminates(7, "u_"+edge.name)
            edge.u_bounding_set = acc_limits
            for i in range(7):
                edge.groebner_basis_substitutions[edge.right.x[i]] = edge.left.x[i] + edge.left.x[7+i]*delta_t + edge.u[i]*delta_t**2/2
                edge.groebner_basis_substitutions[edge.right.x[7+i]] = edge.left.x[7+i] + edge.u[i]*delta_t
            edge.groebner_basis_equality_evaluators = [groebner_evaluator1, groebner_evaluator2]

            
    INFO("Pushing up cost-to-gos")
    for vertex_name in tqdm(start_regions_names):
        graph.MaxCostOverVertex(graph.vertices[vertex_name])


    timer = timeit()
    graph.SolvePolicy()
    timer.dt("policy solve time")

    return graph


regions_file_string = "./iris_regions/bigger_instance/bi5.yaml"

# load regions
use_rohan_scenario = False
convex_set_dict = LoadIrisRegionsYamlFile(regions_file_string) # type: T.Dict[str, HPolyhedron]

print(convex_set_dict.keys())

# terminal set, source sets
terminal_set_name_1 = "LB"
terminal_set_name_2 = "RB"
terminal_set_name_3 = "FB"

all_start_sets = ["LS_B", "LS_M", "LS_T", "RS_B","RS_M","RS_T", "LS_B2", "LS_M2", "LS_T2","RS_B2","RS_M2","RS_T2",]


delta_surr = 0.01
graph_dict3 = dict() # type: T.Dict[str, GoalConditionedPolynomialDualGCS]
graph_dict3[terminal_set_name_1] = produce_a_policy(convex_set_dict, 0.5, terminal_set_name_1, all_start_sets, delta_surr, 0.1, 0.025)
# graph_dict3[terminal_set_name_2] = produce_a_policy(convex_set_dict, terminal_set_name_2, all_start_sets, delta, just_build_graph)
# graph_dict3[terminal_set_name_3] = produce_a_policy(convex_set_dict, terminal_set_name_3, all_start_sets, delta, just_build_graph)