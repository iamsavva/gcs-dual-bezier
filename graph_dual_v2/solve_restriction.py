import typing as T  # pylint: disable=unused-import

import numpy as np
import numpy.typing as npt

from pydrake.solvers import (  # pylint: disable=import-error, no-name-in-module, unused-import
    MathematicalProgram,
    MathematicalProgramResult,
    Solve,
    MosekSolver,
    GurobiSolver,
    MosekSolverDetails,
    SnoptSolver,
    OsqpSolver,
    ClarabelSolver,
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
from pydrake.math import (  # pylint: disable=import-error, no-name-in-module, unused-import
    ge,
    eq,
    le,
)
import cProfile

import plotly.graph_objects as go  # pylint: disable=import-error
from plotly.express.colors import sample_colorscale  # pylint: disable=import-error
import plotly.graph_objs as go
from plotly.subplots import make_subplots

from collections import deque
from queue import PriorityQueue

from program_options import FREE_POLY, PSD_POLY, CONVEX_POLY, ProgramOptions

from util import (
    timeit,
    diditwork,
    INFO,
    YAY,
    WARN,
    ERROR,
    latex
    # ChebyshevCenter,
    # get_kth_control_point
)  # pylint: disable=import-error, no-name-in-module, unused-import

from gcs_util import get_edge_name, make_quadratic_cost_function_matrices, plot_a_gcs

from gcs_dual import PolynomialDualGCS, DualEdge, DualVertex
# from plot_utils import plot_bezier

from util import add_set_membership, recenter_convex_set

# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
# ---------------------------------------------------------------------

class RestrictionSolution:
    def __init__(self, vertex_path:T.List[DualVertex], trajectory:T.List[npt.NDArray], edge_variable_trajectory:T.List[npt.NDArray] = None):
        self.vertex_path = vertex_path
        self.trajectory = trajectory
        self.edge_variable_trajectory = edge_variable_trajectory 
        if self.edge_variable_trajectory is None:
            self.edge_variable_trajectory = [None] * (len(vertex_path)-1)
        assert len(self.edge_variable_trajectory) == len(vertex_path)-1

    def vertex_now(self) -> DualVertex:
        return self.vertex_path[-1]
    
    def point_now(self) -> npt.NDArray:
        return self.trajectory[-1]
    
    def edge_var_now(self) -> npt.NDArray:
        return self.edge_variable_trajectory[-1] if len(self.edge_variable_trajectory) > 0 else None

    def extend(self, next_point: npt.NDArray, next_edge_var: npt.NDArray, next_vertex: DualVertex) -> "RestrictionSolution":
        if not next_vertex.convex_set.PointInSet(next_point):
            # point not in set, need to project due to bad numerics
            next_point = next_vertex.convex_set.Projection(next_point)[1].flatten()

        return RestrictionSolution(
                    self.vertex_path + [next_vertex],
                    self.trajectory + [next_point], 
                    self.edge_variable_trajectory + [next_edge_var]
                    )

    def get_cost(self, graph:PolynomialDualGCS, use_surrogate: bool, add_target_heuristic:bool = True, target_state:npt.NDArray = None) -> float:
        """
        Note: this is cost of the full path with the target cost
        """
        if target_state is None:
            assert isinstance(graph.target_convex_set, Point), "target set not passed when target set not a point"
            target_state = graph.target_convex_set.x()

        cost = 0.0
        for index in range(len(self.trajectory)-1):
            edge = graph.edges[get_edge_name(self.vertex_path[index].name, self.vertex_path[index + 1].name)]
            x_v = self.trajectory[index]
            x_v_1 = self.trajectory[index+1]
            u = self.edge_variable_trajectory[index]
            if use_surrogate:
                cost += edge.cost_function_surrogate(x_v, u, x_v_1, target_state)
            else:
                cost += edge.cost_function(x_v, u, x_v_1, target_state)

        if add_target_heuristic:
            cost_to_go_at_last_point = self.vertex_path[-1].get_cost_to_go_at_point(self.trajectory[-1], target_state, True)
            cost += cost_to_go_at_last_point
        return cost


# def get_path_cost(
#     graph: PolynomialDualGCS,
#     vertex_path: T.List[DualVertex],
#     vertex_trajectory: T.List[npt.NDArray], # TODO: here -- have vertex / edge variables?
#     use_surrogate: bool,
#     add_target_heuristic:bool = True,
#     target_state:npt.NDArray = None
# ) -> float:
#     """
#     Note: this is cost of the full path with the target cost
#     """
#     if target_state is None:
#         assert isinstance(graph.target_convex_set, Point), "target set not passed when target set not a point"
#         target_state = graph.target_convex_set.x()

#     cost = 0.0
#     for index in range(len(vertex_trajectory)-1):
#         edge = graph.edges[get_edge_name(vertex_path[index].name, vertex_path[index + 1].name)]
#         x_v = vertex_trajectory[index]
#         x_v_1 = vertex_trajectory[index+1]
#         if use_surrogate:
#             cost += edge.cost_function_surrogate(x_v, None, x_v_1, target_state)
#         else:
#             cost += edge.cost_function(x_v, None, x_v_1, target_state)

#     if add_target_heuristic:
#         cost_to_go_at_last_point = vertex_path[-1].get_cost_to_go_at_point(vertex_trajectory[-1], target_state, True)
#         cost += cost_to_go_at_last_point
#     return cost



def solve_parallelized_convex_restriction(
    graph: PolynomialDualGCS,
    vertex_paths: T.List[T.List[DualVertex]],
    state_now: npt.NDArray,
    options: ProgramOptions = None,
    verbose_failure=True,
    target_state:npt.NDArray = None,
    one_last_solve:bool = False,
    verbose_solve_success = True
) -> T.List[RestrictionSolution]:
    """
    solve a convex restriction over a vertex path
    return cost of the vertex_path
    and return a list of bezier curves
    where bezier curve is a list of numpy arrays (vectors).
    """
    
    if options is None:
        options = graph.options
    if target_state is None:
        assert isinstance(graph.target_convex_set, Point), "target set not passed when target set not a point"
        target_state = graph.target_convex_set.x()

    # construct an optimization problem
    prog = MathematicalProgram()
    vertex_trajectories = []
    edge_variable_trajectories = []
    timer = timeit()

    for vertex_path in vertex_paths:
        # previous direction of motion -- for bezier curve continuity
        vertex_trajectory = []
        edge_variable_trajectory = []

        # for every vertex:
        for i, vertex in enumerate(vertex_path):
            x = prog.NewContinuousVariables(vertex.state_dim, "x"+str(i))
            vertex_trajectory.append(x)
            add_set_membership(prog, vertex.convex_set, x, True)

            if i == 0:
                # print(i, "lin con",  eq(x, state_now))
                prog.AddLinearConstraint( eq(x, state_now))
            else:
                edge = graph.edges[get_edge_name(vertex_path[i - 1].name, vertex.name)]

                u = None if edge.u is None else prog.NewContinuousVariables(len(edge.u))
                edge_variable_trajectory.append(u)
                if edge.u_bounding_set is not None:
                    add_set_membership(prog, edge.u_bounding_set, u, True)

                cost = edge.cost_function(vertex_trajectory[i-1], u, x, target_state)
                prog.AddCost(cost)

                for evaluator in edge.linear_inequality_evaluators:
                    prog.AddLinearConstraint(ge(evaluator(vertex_trajectory[i-1], u, x, target_state), 0))
                for evaluator in edge.equality_evaluators:
                    prog.AddLinearConstraint(eq(evaluator(vertex_trajectory[i-1], u, x, target_state), 0))

                # TODO: in practice this should be put as lorentz cone, not quadratic
                for evaluator in edge.quadratic_inequality_evaluators:
                    prog.AddConstraint(ge(evaluator(vertex_trajectory[i-1], u, x, target_state), 0))

                # groebner bases related stuff
                for evaluator in edge.groebner_basis_equality_evaluators:
                    prog.AddLinearConstraint(eq(evaluator(vertex_trajectory[i-1], u, x, target_state), 0))


            if i == len(vertex_path) - 1:  
                # if using target heuristic cost:
                if not options.policy_use_zero_heuristic:
                    cost = vertex.get_cost_to_go_at_point(x, target_state)
                    prog.AddCost(cost)
                if vertex.vertex_is_target:
                    if vertex.target_policy_terminating_condition is None:
                        prog.AddLinearConstraint( eq(x, target_state)) 
                    else:
                        terminating_condition = recenter_convex_set(vertex.target_policy_terminating_condition, target_state)
                        add_set_membership(prog, terminating_condition, x, True)

        vertex_trajectories.append(vertex_trajectory)
        edge_variable_trajectories.append(edge_variable_trajectory)

    timer.dt("just building", print_stuff=options.verbose_solve_times)
    
    if options.policy_solver is None:
        solution = Solve(prog)
    else:
        if options.policy_solver == MosekSolver:
            mosek_solver = MosekSolver()
            solver_options = SolverOptions()
            # set the solver tolerance gaps
            if not one_last_solve:
                solver_options.SetOption(
                    MosekSolver.id(),
                    "MSK_DPAR_INTPNT_CO_TOL_REL_GAP",
                    options.policy_MSK_DPAR_INTPNT_CO_TOL_REL_GAP,
                )
                solver_options.SetOption(
                    MosekSolver.id(),
                    "MSK_DPAR_INTPNT_CO_TOL_PFEAS",
                    options.policy_MSK_DPAR_INTPNT_CO_TOL_PFEAS,
                )
                solver_options.SetOption(
                    MosekSolver.id(),
                    "MSK_DPAR_INTPNT_CO_TOL_DFEAS",
                    options.policy_MSK_DPAR_INTPNT_CO_TOL_DFEAS,
                )
                solver_options.SetOption(
                    MosekSolver.id(),
                    "MSK_DPAR_INTPNT_TOL_INFEAS",
                    options.MSK_DPAR_INTPNT_TOL_INFEAS,
                )

            solver_options.SetOption(MosekSolver.id(), 
                                    "MSK_IPAR_PRESOLVE_USE", 
                                    options.MSK_IPAR_PRESOLVE_USE)
            
            solver_options.SetOption(MosekSolver.id(), 
                                        "MSK_IPAR_INTPNT_SOLVE_FORM", 
                                        options.MSK_IPAR_INTPNT_SOLVE_FORM)
                
            if options.policy_use_robust_mosek_params:
                solver_options.SetOption(MosekSolver.id(), "MSK_DPAR_INTPNT_CO_TOL_REL_GAP", 1e-3)
                solver_options.SetOption(MosekSolver.id(), "MSK_IPAR_INTPNT_SOLVE_FORM", 1)

            # latex(prog)

            # solve the program
            solution = mosek_solver.Solve(prog, solver_options=solver_options)
        else:
            solution = options.policy_solver().Solve(prog)

    timer.dt("just solving", print_stuff=options.verbose_solve_times)

    final_result = []
    if solution.is_success():
        for i, v_path in enumerate(vertex_paths):
            vertex_trajectory = vertex_trajectories[i]
            edge_variable_trajectory = edge_variable_trajectories[i]
            traj_solution = [solution.GetSolution(x) for x in vertex_trajectory]
            edge_traj_solution = [None if u is None else solution.GetSolution(u) for u in edge_variable_trajectory]
            final_result.append(RestrictionSolution(v_path, traj_solution, edge_traj_solution))
        return final_result
    else:
        WARN("failed to solve", verbose = verbose_solve_success)
        if verbose_failure:
            diditwork(solution)
        return None
    

def solve_convex_restriction(
    graph: PolynomialDualGCS,
    vertex_path: T.List[DualVertex],
    state_now: npt.NDArray,
    options: ProgramOptions = None,
    verbose_failure:bool =False,
    target_state:npt.NDArray = None,
    one_last_solve = False
) -> RestrictionSolution:
    result = solve_parallelized_convex_restriction(graph, [vertex_path], state_now, options, verbose_failure, target_state, one_last_solve, verbose_solve_success = False)
    if result is None:
        return None
    else:
        return result[0]
    

# ---

def get_optimal_path(
    graph: PolynomialDualGCS,
    vertex: DualVertex,
    state: npt.NDArray,
    options: ProgramOptions = None,
    target_state:npt.NDArray = None
) -> T.Tuple[float, float, T.List[T.List[npt.NDArray]], T.List[DualVertex]]:
    """
    return (time, cost, bezier path, vertex path) of the optimal solution.
    """
    raise NotImplementedError("needs a rewrite for walks / paths / new constraints")
    if options is None:
        options = graph.options

    if target_state is not None:
        gcs, vertices, pseudo_target_vertex = graph.export_a_gcs(target_state)
    else:
        gcs, vertices, pseudo_target_vertex = graph.export_a_gcs()
    
    # set initial vertex constraint
    start_vertex = vertices[vertex.name]
    first_point = start_vertex.x()
    cons = eq(first_point, state)
    for con in cons:
        start_vertex.AddConstraint( con )

    # set target vertex constraint:
    if target_state is not None:
        assert isinstance(graph, GoalConditionedPolynomialDualGCS), "passed target state but not a Goal Conditioned policy"
        target_vertex = vertices[graph.target_vertex.name]
        last_point = target_vertex.x()
        assert len(last_point) == graph.target_vertex.state_dim
        cons = eq(last_point, target_state)
        for con in cons:
            target_vertex.AddConstraint( con )

    gcs_options = GraphOfConvexSetsOptions()
    gcs_options.convex_relaxation = options.gcs_policy_use_convex_relaxation
    gcs_options.max_rounding_trials = options.gcs_policy_max_rounding_trials
    gcs_options.preprocessing = options.gcs_policy_use_preprocessing
    gcs_options.max_rounded_paths = options.gcs_policy_max_rounded_paths
    if options.gcs_policy_solver is not None:
        gcs_options.solver = options.gcs_policy_solver()

    # solve
    timer = timeit()
    result = gcs.SolveShortestPath(
        start_vertex, pseudo_target_vertex, gcs_options
    )  # type: MathematicalProgramResult
    dt = timer.dt("just SolveShortestPath solve time", print_stuff=options.verbose_solve_times)

    assert result.is_success()
    cost = result.get_optimal_cost()
    if options.verbose_solve_times:
        diditwork(result)

    edge_path = gcs.GetSolutionPath(start_vertex, pseudo_target_vertex, result)
    vertex_name_path = []
    value_path = []
    for e in edge_path[:-1]:
        vertex_name_path.append(e.u().name())
        full_curve = [result.GetSolution(e.u().x()), result.GetSolution(e.v().x())]
        value_path.append(full_curve)
    vertex_name_path.append(edge_path[-1].u().name())
    if target_state is not None:
        assert np.allclose(value_path[-1][-1], target_state), "target state isn't 0, why?"
        
    return dt, cost, value_path, [graph.vertices[name] for name in vertex_name_path]

