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
    
    def point_initial(self) -> npt.NDArray:
        return self.trajectory[0]
    
    def edge_var_now(self) -> npt.NDArray:
        return self.edge_variable_trajectory[-1] if len(self.edge_variable_trajectory) > 0 else None
    
    def length(self):
        return len(self.trajectory)

    def vertex_names(self):
        return np.array([v.name for v in self.vertex_path])
    
    def make_copy(self):
        return RestrictionSolution(self.vertex_path, self.trajectory, self.edge_variable_trajectory)

    def extend(self, next_point: npt.NDArray, next_edge_var: npt.NDArray, next_vertex: DualVertex) -> "RestrictionSolution":
        if not next_vertex.convex_set.PointInSet(next_point):
            # point not in set, need to project due to bad numerics
            next_point = next_vertex.convex_set.Projection(next_point)[1].flatten()

        return RestrictionSolution(
                    self.vertex_path + [next_vertex],
                    self.trajectory + [next_point], 
                    self.edge_variable_trajectory + [next_edge_var]
                    )
    
    def reoptimize(self, graph:PolynomialDualGCS, verbose_failure:bool, target_state:npt.NDArray, one_last_solve:bool):
        new_sol, solver_time = solve_convex_restriction(graph, self.vertex_path, self.point_initial(), verbose_failure, target_state, one_last_solve)
        assert new_sol is not None, WARN("resolving on the same restriction failed. something is fishy")
            
        self.trajectory = new_sol.trajectory
        self.edge_variable_trajectory = new_sol.edge_variable_trajectory
        return solver_time

    def get_cost(self, graph:PolynomialDualGCS, 
                 use_surrogate: bool, 
                 add_target_heuristic:bool = True, 
                 target_state:npt.NDArray = None,
                 double_integrator_delta_t_list: T.List[float] = None
                 ) -> float:
        """
        Note: this is cost of the full path with the target cost
        """
        if target_state is None:
            if graph.options.dont_do_goal_conditioning:
                target_state = np.zeros(graph.target_convex_set.ambient_dimension())
            else:
                assert isinstance(graph.target_convex_set, Point), "target set not passed when target set not a point"
                target_state = graph.target_convex_set.x()

        cost = 0.0
        for index in range(len(self.trajectory)-1):
            edge = graph.edges[get_edge_name(self.vertex_path[index].name, self.vertex_path[index + 1].name)]
            x_v = self.trajectory[index]
            x_v_1 = self.trajectory[index+1]
            u = self.edge_variable_trajectory[index]
            if use_surrogate:
                if double_integrator_delta_t_list is not None:
                    delta_t = double_integrator_delta_t_list[index]
                    cost += edge.cost_function_surrogate(x_v, u, x_v_1, target_state, delta_t)
                else:
                    cost += edge.cost_function_surrogate(x_v, u, x_v_1, target_state)
            else:
                if double_integrator_delta_t_list is not None:
                    delta_t = double_integrator_delta_t_list[index]
                    cost += edge.cost_function(x_v, u, x_v_1, target_state, delta_t)
                else:
                    cost += edge.cost_function(x_v, u, x_v_1, target_state)                

        if add_target_heuristic:
            cost_to_go_at_last_point = self.vertex_path[-1].get_cost_to_go_at_point(self.trajectory[-1], target_state, graph.options.check_cost_to_go_at_point)
            cost += cost_to_go_at_last_point
        return cost




def solve_parallelized_convex_restriction(
    graph: PolynomialDualGCS,
    vertex_paths: T.List[T.List[DualVertex]],
    state_now: npt.NDArray,
    verbose_failure=True,
    target_state:npt.NDArray = None,
    one_last_solve:bool = False,
    verbose_solve_success = True,
    double_integrator_delta_t_lists: T.List[T.List[float]] = None,
) -> T.Tuple[T.List[RestrictionSolution], float]:
    """
    solve a convex restriction over a vertex path
    return cost of the vertex_path
    and return a list of bezier curves
    where bezier curve is a list of numpy arrays (vectors).
    """
    options = graph.options
    # TODO: remove double_integrator_delta_t_lists bullshit

    # construct an optimization problem
    prog = MathematicalProgram()
    vertex_trajectories = []
    edge_variable_trajectories = []
    timer = timeit()

    if target_state is None:
        if options.dont_do_goal_conditioning:
            target_state = np.zeros(graph.target_convex_set.ambient_dimension())
        else:
            if isinstance(graph.target_convex_set, Point):
                # assert , "target set not passed when target set not a point"
                target_state = graph.target_convex_set.x()
            else:
                target_state = prog.NewContinuousVariables(graph.target_convex_set.ambient_dimension())

    for v_path_index, vertex_path in enumerate(vertex_paths):
        # previous direction of motion -- for bezier curve continuity
        vertex_trajectory = []
        edge_variable_trajectory = []

        # for every vertex:
        for i, vertex in enumerate(vertex_path):
            x = prog.NewContinuousVariables(vertex.state_dim, "x"+str(i))
            vertex_trajectory.append(x)
            
            if not vertex.vertex_is_target:
                add_set_membership(prog, vertex.convex_set, x, True)
            else:
                assert i == len(vertex_path) - 1, "something is fishy"

                if options.dont_do_goal_conditioning:
                    add_set_membership(prog, vertex.convex_set, x, True)
                else:
                    # new implementation
                    if options.relax_target_condition_during_rollout and not one_last_solve:
                        # during rollout: relax target condition.
                        # at the end when solving restriction -- don't.
                        assert vertex.relaxed_target_condition_for_policy is not None
                        terminating_condition = recenter_convex_set(vertex.relaxed_target_condition_for_policy, target_state)
                        add_set_membership(prog, terminating_condition, x, True)
                    else:
                        # prog.AddLinearEqualityConstraint(x, target_state)
                        prog.AddLinearConstraint(eq(x, target_state))


            if i == 0:
                # NOTE: if state_now is None, we have a free initial state problem
                if state_now is not None:
                    prog.AddLinearEqualityConstraint(x, state_now)
            else:
                edge = graph.edges[get_edge_name(vertex_path[i - 1].name, vertex.name)]

                if options.add_right_point_inside_intersection_constraint and edge.left.state_dim == edge.right.state_dim:
                    if edge.left.name != edge.right.name:
                        add_set_membership(prog, edge.left.convex_set, x, True)

                u = None if edge.u is None else prog.NewContinuousVariables(len(edge.u))
                edge_variable_trajectory.append(u)
                if edge.u_bounding_set is not None:
                    add_set_membership(prog, edge.u_bounding_set, u, True)

                
                if double_integrator_delta_t_lists is None:
                    cost = edge.cost_function(vertex_trajectory[i-1], u, x, target_state)
                else:
                    delta_t = double_integrator_delta_t_lists[v_path_index][i-1]
                    cost = edge.cost_function(vertex_trajectory[i-1], u, x, target_state, delta_t)
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
                    if double_integrator_delta_t_lists is None:
                        prog.AddLinearConstraint(eq(evaluator(vertex_trajectory[i-1], u, x, target_state), 0))
                    else:
                        delta_t = double_integrator_delta_t_lists[v_path_index][i-1]
                        prog.AddLinearConstraint(eq(evaluator(vertex_trajectory[i-1], u, x, target_state, delta_t), 0))

            # add heuristic cost on the last point
            if i == len(vertex_path) - 1:  
                if not options.policy_use_zero_heuristic:
                    cost = vertex.get_cost_to_go_at_point(x, target_state, options.check_cost_to_go_at_point)
                    prog.AddCost(cost)

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
            
            if options.MSK_IPAR_OPTIMIZER is not None:
                solver_options.SetOption(MosekSolver.id(), 
                                            "MSK_IPAR_OPTIMIZER", 
                                            options.MSK_IPAR_OPTIMIZER)
                
                
            # solve the program
            solution = mosek_solver.Solve(prog, solver_options=solver_options)
        else:
            solution = options.policy_solver().Solve(prog)

    timer.dt("just solving", print_stuff=options.verbose_solve_times)

    if options.policy_solver == MosekSolver or solution.get_solver_id().name() == "Mosek":
        solver_solve_time = solution.get_solver_details().optimizer_time
    elif options.policy_solver == ClarabelSolver or solution.get_solver_id().name() == "Clarabel":
        solver_solve_time = solution.get_solver_details().solve_time
    elif options.policy_solver == GurobiSolver or solution.get_solver_id().name() == "Gurobi":
        solver_solve_time = solution.get_solver_details().optimizer_time
    elif options.policy_solver == OsqpSolver or solution.get_solver_id().name() == "OSQP":
        solver_solve_time = solution.get_solver_details().solve_time
    else:
        WARN("don't know how to get solver time for the solver", solution.get_solver_id().name())
        raise NotImplementedError()

    final_result = []
    if solution.is_success():
        for i, v_path in enumerate(vertex_paths):
            vertex_trajectory = vertex_trajectories[i]
            edge_variable_trajectory = edge_variable_trajectories[i]
            traj_solution = [solution.GetSolution(x) for x in vertex_trajectory]
            edge_traj_solution = [None if u is None else solution.GetSolution(u) for u in edge_variable_trajectory]
            final_result.append(RestrictionSolution(v_path, traj_solution, edge_traj_solution))
        return final_result, solver_solve_time
    else:
        WARN("failed to solve", verbose = verbose_solve_success)
        if verbose_failure:
            diditwork(solution)
        return None, solver_solve_time
    

def solve_convex_restriction(
    graph: PolynomialDualGCS,
    vertex_path: T.List[DualVertex],
    state_now: npt.NDArray,
    verbose_failure:bool =False,
    target_state:npt.NDArray = None,
    one_last_solve = False,
    double_itnegrator_delta_t_list = None,
) -> T.Tuple[RestrictionSolution, float]:
    double_integrator_delta_t_lists = None if double_itnegrator_delta_t_list is None else [double_itnegrator_delta_t_list]
    result, dt = solve_parallelized_convex_restriction(graph, [vertex_path], state_now, verbose_failure, target_state, one_last_solve, False, double_integrator_delta_t_lists)
    if result is None:
        return None, dt
    else:
        return result[0], dt
    

# ---


def solve_double_integrator_convex_restriction(
    options: ProgramOptions,
    convex_set_path: T.List[ConvexSet],
    vertex_name_path: T.List[str],
    vel_bounds: ConvexSet,
    acc_bounds: ConvexSet,
    start_state: npt.NDArray,
    target_state:npt.NDArray,
    dt: T.List[float],
    cost_function: T.Callable,
    traj_reversed:bool,
    past_solution = None,
) -> T.Tuple[T.List[RestrictionSolution], float]:
    """
    solve a convex restriction over a vertex path
    return cost of the vertex_path
    and return a list of bezier curves
    where bezier curve is a list of numpy arrays (vectors).
    """
    # construct an optimization problem
    prog = MathematicalProgram()

    # previous direction of motion -- for bezier curve continuity
    x_traj = []
    v_traj = []
    a_traj = []

    # ERROR([convex_set.ambient_dimension() for convex_set in convex_set_path])

    # for every vertex:
    for i, convex_set in enumerate(convex_set_path):
        x = prog.NewContinuousVariables(convex_set.ambient_dimension(), "x"+str(i))
        v = prog.NewContinuousVariables(vel_bounds.ambient_dimension(), "v"+str(i))
        x_traj.append(x)
        v_traj.append(v)
        if past_solution is not None:
            prog.AddLinearConstraint(eq(x, past_solution[0][i]))
            prog.AddLinearConstraint(eq(v, past_solution[1][i]))

        if i == 0:
            prog.AddLinearEqualityConstraint(x, start_state)
            prog.AddLinearEqualityConstraint(v, np.zeros(vel_bounds.ambient_dimension()))
        elif i == len(convex_set_path)-1:
            prog.AddLinearEqualityConstraint(x, target_state)
            prog.AddLinearEqualityConstraint(v, np.zeros(vel_bounds.ambient_dimension()))
            # add_set_membership(prog, Hyperrectangle(-1e-5*np.ones(7), 1e-5*np.ones(7)), v, True)
        else:
            add_set_membership(prog, convex_set, x, True)
            add_set_membership(prog, vel_bounds, v, True)

        if not traj_reversed and i > 0:
            if options.add_right_point_inside_intersection_constraint and vertex_name_path[i-1] != vertex_name_path[i]:
            #     YAY(convex_set_path[i-1].IntersectsWith(convex_set_path[i]))
            #     WARN(convex_set_path[i-1].Intersection(convex_set_path[i]).PointInSet(past_solution[0][i]))
                add_set_membership(prog, convex_set_path[i-1], x, True)
        elif traj_reversed and i+1 <= len(convex_set_path)-1:
            if options.add_right_point_inside_intersection_constraint and vertex_name_path[i] != vertex_name_path[i+1]:
            #     YAY(convex_set_path[i-1].IntersectsWith(convex_set_path[i]))
            #     WARN(convex_set_path[i-1].Intersection(convex_set_path[i]).PointInSet(past_solution[0][i]))
                add_set_membership(prog, convex_set_path[i+1], x, True)



        if i > 0:
            a = prog.NewContinuousVariables(acc_bounds.ambient_dimension(), "a"+str(i))
            a_traj.append(a)
            if past_solution is not None:
                prog.AddLinearConstraint(eq(a, past_solution[2][i-1]))
                
            add_set_membership(prog, acc_bounds, a, True)

            # if options.add_right_point_inside_intersection_constraint and vertex_name_path[i-1] != vertex_name_path[i]:
            #     YAY(convex_set_path[i-1].IntersectsWith(convex_set_path[i]))
            #     WARN(convex_set_path[i-1].Intersection(convex_set_path[i]).PointInSet(past_solution[0][i]))

            #     add_set_membership(prog, convex_set_path[i-1], x, True)

            cost = cost_function(np.hstack((x_traj[i-1],v_traj[i-1])), a, np.hstack((x_traj[i],v_traj[i])), target_state, dt[i-1])
            prog.AddCost(cost)
            
            prog.AddLinearConstraint(eq(x_traj[i], x_traj[i-1] + v_traj[i-1]*dt[i-1] + a_traj[i-1] * dt[i-1]**2 / 2 ))
            prog.AddLinearConstraint(eq(v_traj[i], v_traj[i-1] + a_traj[i-1]*dt[i-1]))


    if options.policy_solver is None:
        solution = Solve(prog)
    else:
        if options.policy_solver == MosekSolver:
            mosek_solver = MosekSolver()
            solver_options = SolverOptions()
            # set the solver tolerance gaps
            
            solver_options.SetOption(
                MosekSolver.id(),
                "MSK_DPAR_INTPNT_TOL_INFEAS",
                options.MSK_DPAR_INTPNT_TOL_INFEAS,
            )
            # solver_options.SetOption(
            #     MosekSolver.id(),
            #     "MSK_DPAR_INTPNT_CO_TOL_REL_GAP",
            #     options.policy_MSK_DPAR_INTPNT_CO_TOL_REL_GAP,
            # )
            # solver_options.SetOption(
            #     MosekSolver.id(),
            #     "MSK_DPAR_INTPNT_CO_TOL_PFEAS",
            #     options.policy_MSK_DPAR_INTPNT_CO_TOL_PFEAS,
            # )
            # solver_options.SetOption(
            #     MosekSolver.id(),
            #     "MSK_DPAR_INTPNT_CO_TOL_DFEAS",
            #     options.policy_MSK_DPAR_INTPNT_CO_TOL_DFEAS,
            # )

            solver_options.SetOption(MosekSolver.id(), 
                                    "MSK_IPAR_PRESOLVE_USE", 
                                    options.MSK_IPAR_PRESOLVE_USE)
            
            solver_options.SetOption(MosekSolver.id(), 
                                        "MSK_IPAR_INTPNT_SOLVE_FORM", 
                                        options.MSK_IPAR_INTPNT_SOLVE_FORM)
            
            if options.MSK_IPAR_OPTIMIZER is not None:
                solver_options.SetOption(MosekSolver.id(), 
                                            "MSK_IPAR_OPTIMIZER", 
                                            options.MSK_IPAR_OPTIMIZER)
                
            # solve the program
            solution = mosek_solver.Solve(prog, solver_options=solver_options)
        else:
            solution = options.policy_solver().Solve(prog)

    if options.policy_solver == MosekSolver or solution.get_solver_id().name() == "Mosek":
        solver_solve_time = solution.get_solver_details().optimizer_time
    elif options.policy_solver == ClarabelSolver or solution.get_solver_id().name() == "Clarabel":
        solver_solve_time = solution.get_solver_details().solve_time
    elif options.policy_solver == GurobiSolver or solution.get_solver_id().name() == "Gurobi":
        solver_solve_time = solution.get_solver_details().optimizer_time
    elif options.policy_solver == OsqpSolver or solution.get_solver_id().name() == "OSQP":
        solver_solve_time = solution.get_solver_details().solve_time
    else:
        WARN("don't know how to get solver time for the solver", solution.get_solver_id().name())
        raise NotImplementedError()
    

    if solution.is_success():
        # YAY("problem solved")
        x_traj_sol = np.array([solution.GetSolution(x) for x in x_traj])
        v_traj_sol = np.array([solution.GetSolution(v) for v in v_traj])
        a_traj_sol = np.array([solution.GetSolution(a) for a in a_traj])
        return x_traj_sol, v_traj_sol, a_traj_sol, solution.get_optimal_cost(), solver_solve_time
    else:
        # diditwork(solution)
        # from IPython.display import Markdown, display
        # for con in solution.GetInfeasibleConstraints(prog):
        #     display(Markdown(con.ToLatex()))
        return None, None, None, np.inf, solver_solve_time