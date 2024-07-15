import typing as T  # pylint: disable=unused-import

import numpy as np
import numpy.typing as npt
from scipy.linalg import block_diag

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
    L2NormCost,
    Binding,
)
from pydrake.geometry.optimization import (  # pylint: disable=import-error, no-name-in-module
    GraphOfConvexSets,
    GraphOfConvexSetsOptions,
    HPolyhedron,
    Point,
    ConvexSet,
    Hyperrectangle,
    Hyperellipsoid,
)
from pydrake.all import MakeSemidefiniteRelaxation # pylint: disable=import-error, no-name-in-module
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
import plotly.graph_objs as go # pylint: disable=import-error
from plotly.subplots import make_subplots # pylint: disable=import-error

from program_options import FREE_POLY, PSD_POLY, CONVEX_POLY, ProgramOptions

from util import (
    timeit,
    diditwork,
    INFO,
    YAY,
    WARN,
    ERROR,
    ChebyshevCenter,
    make_polyhedral_set_for_bezier_curve,
    get_kth_control_point,
    add_set_membership,
)  # pylint: disable=import-error, no-name-in-module, unused-import

from gcs_util import get_edge_name, make_quadratic_cost_function_matrices
from polynomial_dual_gcs_utils import (
    define_quadratic_polynomial,
    get_product_constraints,
    make_linear_set_inequalities, 
    get_B_matrix,
    # define_sos_constraint_over_polyhedron_multivar,
    define_sos_constraint_over_polyhedron_multivar_new,
    make_potential,
    get_set_membership_inequalities,
    get_set_intersection_inequalities,
    # define_general_nonnegativity_constraint_on_a_set
)
# from graph_dual import DualVertex, DualEdge, PolynomialDualGCS

from util_moments import (
    extract_moments_from_vector_of_spectrahedron_prog_variables, 
    make_moment_matrix, 
    get_moment_matrix_for_a_measure_over_set,
    make_product_of_indepent_moment_matrices,
    verify_necessary_conditions_for_moments_supported_on_set,
)

from tqdm import tqdm
# delta = 0.001
# QUADRATIC_COST = lambda xl, u, xr, , z: np.sum([(x[i] - y[i]) ** 2 for i in range(len(x))])
# QUADRATIC_COST_AUGMENTED = lambda x, y, z: np.sum(
#     [(x[i] - y[i]) ** 2 for i in range(len(x))]
# ) + delta * 0.5* (np.sum([ (x[i]-z[i]) ** 2 + (y[i]-z[i]) ** 2 for i in range(len(x))]))


class DualVertex:
    def __init__(
        self,
        name: str,
        prog: MathematicalProgram,
        convex_set: ConvexSet,
        target_convex_set: ConvexSet,
        xt: npt.NDArray,
        options: ProgramOptions,
        vertex_is_start: bool = False,
        vertex_is_target: bool = False,
        target_policy_terminating_condition: ConvexSet = None,
        target_cost_matrix: npt.NDArray = None,
    ):
        self.name = name
        self.options = options
        self.vertex_is_start = vertex_is_start
        self.vertex_is_target = vertex_is_target

        # NOTE: all these variables will need to be set in define variables and define potentials
        self.potential = None # type: Expression
        self.J_matrix = None # type: npt.NDArray
        self.J_matrix_solution = None # type: npt.NDArray
        # self.use_target_constraint = None # type: bool

        self.convex_set = convex_set
        self.set_type = type(convex_set)
        self.state_dim = convex_set.ambient_dimension()
        if self.set_type not in (HPolyhedron, Hyperrectangle, Hyperellipsoid, Point):
            raise Exception("bad state set")
        
        self.target_convex_set = target_convex_set
        self.target_policy_terminating_condition = target_policy_terminating_condition
        self.target_set_type = type(target_convex_set)        
        # TODO: fix that, shouldn't need this technically
        assert convex_set.ambient_dimension() == target_convex_set.ambient_dimension(), "convex set and target set must have same ambient dimension"
        if self.target_set_type not in (HPolyhedron, Hyperrectangle, Hyperellipsoid, Point):
            raise Exception("bad target state set")
        
        self.xt = xt # target variables
        self.define_variables(prog)
        self.define_set_inequalities()
        self.define_potential(prog, target_cost_matrix)

        self.edges_in = []  # type: T.List[str]
        self.edges_out = []  # type: T.List[str]

    def get_hpoly(self) -> HPolyhedron:
        assert self.set_type in (HPolyhedron, Hyperrectangle), "can't get hpoly for set"
        if self.set_type == HPolyhedron:
            return self.convex_set
        if self.set_type == Hyperrectangle:
            return self.convex_set.MakeHPolyhedron()
        
    def get_target_hpoly(self) -> HPolyhedron:
        assert self.target_set_type in (HPolyhedron, Hyperrectangle), "can't get hpoly for set"
        if self.target_set_type == HPolyhedron:
            return self.target_convex_set
        if self.target_set_type == Hyperrectangle:
            return self.target_convex_set.MakeHPolyhedron()

    def add_edge_in(self, name: str):
        assert name not in self.edges_in
        self.edges_in.append(name)

    def add_edge_out(self, name: str):
        assert not self.vertex_is_target, "adding an edge to a target vertex"
        assert name not in self.edges_out
        self.edges_out.append(name)

    def define_variables(self, prog: MathematicalProgram):
        """
        Defining indeterminates for x and flow-in violation polynomial, if necesary
        """
        self.x = prog.NewIndeterminates(self.state_dim, "x_" + self.name)
        self.vars = Variables(np.hstack((self.x, self.xt)))
        if self.options.allow_vertex_revisits or self.vertex_is_target:
            self.total_flow_in_violation = Expression(0)
            self.total_flow_in_violation_mat = np.zeros((self.state_dim+1,self.state_dim+1))
        else:
            if self.options.flow_violation_polynomial_degree not in (0,2):
                raise Exception("bad vilation polynomial degree " + str(self.options.flow_violation_polynomial_degree))
            self.total_flow_in_violation, self.total_flow_in_violation_mat = make_potential(self.xt, PSD_POLY, self.options.flow_violation_polynomial_degree, prog)
        

    def define_set_inequalities(self):
        self.vertex_set_linear_inequalities, self.vertex_set_quadratic_inequalities = get_set_membership_inequalities(self.x, self.convex_set)        
        self.target_set_linear_inequalities, self.target_set_quadratic_inequalities = get_set_membership_inequalities(self.xt, self.target_convex_set)

    def define_potential(self, prog: MathematicalProgram, target_cost_matrix:npt.NDArray):
        if not self.vertex_is_target:
            assert target_cost_matrix is None, "passed a target cost matrix not a non-target vertex"
            assert self.target_policy_terminating_condition is None, "passed target box width why"

        if self.vertex_is_target:
            if target_cost_matrix is not None:
                assert target_cost_matrix.shape == (2*self.state_dim+1, 2*self.state_dim+1), "bad shape for forced J matrix"
                self.J_matrix = target_cost_matrix
                x_and_xt_and_1 = np.hstack(([1], self.x, self.xt))
                self.potential = np.sum( self.J_matrix * np.outer(x_and_xt_and_1, x_and_xt_and_1))
            else:
                self.J_matrix = np.zeros(((2*self.state_dim+1, 2*self.state_dim+1)))
                x_and_xt_and_1 = np.hstack(([1], self.x, self.xt))
                self.potential = np.sum( self.J_matrix * np.outer(x_and_xt_and_1, x_and_xt_and_1))
        else:
            if self.target_set_type is Point:
                _, J_mat_vars = make_potential(self.x, self.options.pot_type, self.options.potential_poly_deg, prog)
                self.J_matrix = block_diag(J_mat_vars, np.zeros((self.state_dim,self.state_dim)))
                x_and_xt_and_1 = np.hstack(([1], self.x, self.xt))
                self.potential = np.sum( self.J_matrix * np.outer(x_and_xt_and_1, x_and_xt_and_1))
            else:
                x_and_xt = np.hstack((self.x, self.xt))
                self.potential, self.J_matrix = make_potential(x_and_xt, self.options.pot_type, self.options.potential_poly_deg, prog)

        assert self.J_matrix.shape == (2*self.state_dim+1, 2*self.state_dim+1)
    

    def get_cost_to_go_at_point(self, x: npt.NDArray, xt:npt.NDArray = None, point_must_be_in_set:bool=True):
        """
        Evaluate potential at a particular point.
        Return expression if solution not passed, returns a float value if solution is passed.
        """
        if xt is None and self.target_set_type is Point:
            xt = self.target_convex_set.x()
        assert xt is not None, "did not pass xt to get the cost-to-go, when xt is non-unique"
        assert len(x) == self.state_dim
        assert len(xt) == self.state_dim

        if point_must_be_in_set:
            prog = MathematicalProgram()
            x_var = prog.NewContinuousVariables(self.state_dim)
            xt_var = prog.NewContinuousVariables(self.state_dim)
            add_set_membership(prog, self.convex_set, x_var, True)
            add_set_membership(prog, self.target_convex_set, xt_var, True)
            solution = Solve(prog)
            assert solution.is_success(), "getting cost-to-go for a point that's not in the set"

        assert self.J_matrix_solution is not None, "cost-to-go lower bounds have not been set yet"

        x_and_xt_and_1 = np.hstack(([1], x, xt))
        return np.sum( self.J_matrix_solution * np.outer(x_and_xt_and_1, x_and_xt_and_1))
    
    def push_down_on_flow_violation(self, prog:MathematicalProgram, target_moment_matrix:npt.NDArray):
        # add the cost on violations
        prog.AddLinearCost(np.sum(target_moment_matrix * self.total_flow_in_violation_mat))


    def push_up_on_potentials(self, prog:MathematicalProgram, vertex_moments:npt.NDArray, target_moments:npt.NDArray) -> Expression:
        # assert verify_necessary_conditions_for_moments_supported_on_set(vertex_moments, self.convex_set), "moment matrix does not satisfy necessary SDP conditions for being supported on vertexsets"
        # assert verify_necessary_conditions_for_moments_supported_on_set(target_moments, self.target_convex_set), "targetmoment matrix does not satisfy necessary SDP conditions for being supported on target set"

        moment_matrix = make_product_of_indepent_moment_matrices(vertex_moments,target_moments )
        prog.AddLinearCost(-np.sum(self.J_matrix * moment_matrix))



class DualEdge:
    def __init__(
        self,
        name: str,
        v_left: DualVertex,
        v_right: DualVertex,
        cost_function: T.Callable,
        cost_function_surrogate: T.Callable,
        options: ProgramOptions,
        bidirectional_edge_violation=Expression(0),
    ):
        self.name = name
        self.left = v_left
        self.right = v_right

        self.cost_function = cost_function
        self.cost_function_surrogate = cost_function_surrogate
        self.options = options

        self.bidirectional_edge_violation = bidirectional_edge_violation

        # TODO: need to be implemented
        self.linear_inequality_evaluators = []
        self.quadratic_inequality_evaluators = []
        self.equality_evaluators = []

        self.u = None
        self.u_bounding_set = None
        self.groebner_basis_substitutions = dict()
        self.groebner_basis_equality_evaluators = []

    def make_constraints(self, prog:MathematicalProgram):
        linear_inequality_constraints = []
        quadratic_inequality_constraints = []
        equality_constraints = []

        xl, xr, xt = self.left.x, self.right.x, self.left.xt
        if self.left.name == self.right.name:
            self.temp_right_indet = prog.NewIndeterminates(len(self.right.x))
            xr = self.temp_right_indet

        for evaluator in self.linear_inequality_evaluators:
            linear_inequality_constraints.append(evaluator(xl,self.u,xr,xt))
        for evaluator in self.quadratic_inequality_evaluators:
            quadratic_inequality_constraints.append(evaluator(xl,self.u,xr,xt))
        for evaluator in self.equality_evaluators:
            equality_constraints.append(evaluator(xl,self.u,xr,xt))

        linear_inequality_constraints = np.array(linear_inequality_constraints).flatten()
        quadratic_inequality_constraints = np.array(quadratic_inequality_constraints).flatten()
        equality_constraints = np.array(equality_constraints).flatten()
        return linear_inequality_constraints, quadratic_inequality_constraints, equality_constraints

    def define_edge_polynomials_and_sos_constraints(self, prog: MathematicalProgram):
        """
        define edge appropriate SOS constraints
        """

        # -----------------------------------------------
        # NOTE: the easiest thing to do would be to store functions, then evaluate functions, then subsititute them

        unique_variables = []
        all_linear_inequalities = []
        all_quadratic_inequalities = []
        substitutions = dict()

        edge_linear_inequality_constraints, edge_quadratic_inequality_constraints, edge_equality_constraints = self.make_constraints(prog)

        # what's happening?
        # i am trying to produce a list of unique variables that are necessary
        # and a list of substitutions
        
        right_vars = self.right.x
        if self.left.name == self.right.name:
            right_vars = self.temp_right_indet

        # handle left vertex
        if self.left.set_type is Point:
            # left vertex is a point -- substitute indeterminates with values
            for i, xl_i in enumerate(self.left.x):
                substitutions[xl_i] = self.left.convex_set.x()[i]
        else:
            # left vertex is full dimensional, add constraints and variables
            unique_variables.append(self.left.x)
            if len(self.left.vertex_set_linear_inequalities) > 0:
                all_linear_inequalities.append(self.left.vertex_set_linear_inequalities)
            if len(self.left.vertex_set_quadratic_inequalities) > 0:
                all_quadratic_inequalities.append(self.left.vertex_set_quadratic_inequalities)
        
        # handle edge variables
        if self.u is not None:
            # if there are edge variables -- do stuff
            unique_variables.append(self.u)
            if self.u_bounding_set is not None:
                u_linear_ineq, u_quad_ineq = get_set_membership_inequalities(self.u, self.u_bounding_set)
                if len(u_linear_ineq) > 0:
                    all_linear_inequalities.append(u_linear_ineq)
                if len(u_quad_ineq) > 0:
                    all_quadratic_inequalities.append(u_quad_ineq)

        if len(edge_linear_inequality_constraints) > 0:
            all_linear_inequalities.append(edge_linear_inequality_constraints)
        if len(edge_quadratic_inequality_constraints) > 0:
            all_quadratic_inequalities.append(edge_quadratic_inequality_constraints)

        # handle right vertex
        # can't have right vertex be a point and dynamics constraint, dunno how to substitute.
        # need an equality constraint instead, x_right will be subsituted
        if self.right.set_type is Point and len(self.groebner_basis_equality_evaluators) > 0:
            assert False, "can't have right vertex be a point AND have dynamics constraints; put dynamics as an equality constraint instead."

        if self.right.set_type is Point:
            # right vertex is a point -- substitutions
            for i, xr_i in enumerate(right_vars):
                substitutions[xr_i] = self.right.convex_set.x()[i]
        else:
            # full dimensional set
            if len(self.groebner_basis_substitutions) > 0:
                # we have (possibly partial) dynamics constraints -- add them
                right_vertex_variables = []
                for i, xr_i in enumerate(right_vars):
                    # if it's in subsitutions -- add to substition dictionary, else add to unique vars
                    if self.right.x[i] in self.groebner_basis_substitutions: 
                        assert xr_i not in substitutions
                        substitutions[xr_i] = self.groebner_basis_substitutions[self.right.x[i]]
                    else:
                        right_vertex_variables.append(xr_i)
                if len(right_vertex_variables) > 0:
                    unique_variables.append(np.array(right_vertex_variables))
            else:
                unique_variables.append(right_vars)

            if self.left.name == self.right.name:
                rv_linear_inequalities, rv_quadratic_inequalities = get_set_membership_inequalities(right_vars, self.right.convex_set)
            else:
                if self.options.right_point_inside_intersection:
                    rv_linear_inequalities, rv_quadratic_inequalities = get_set_intersection_inequalities(right_vars, self.left.convex_set, self.right.convex_set)
                else:
                    rv_linear_inequalities = self.right.vertex_set_linear_inequalities
                    rv_quadratic_inequalities = self.right.vertex_set_quadratic_inequalities
            if len(rv_linear_inequalities) > 0:
                all_linear_inequalities.append(rv_linear_inequalities)
            if len(rv_quadratic_inequalities) > 0:
                all_quadratic_inequalities.append(rv_quadratic_inequalities)

        # handle target vertex
        if self.right.target_set_type is Point:
            # right vertex is a point -- substitutions
            for i, xt_i in enumerate(self.left.xt):
                substitutions[xt_i] = self.left.target_convex_set.x()[i]
        elif self.right.vertex_is_target:
            # right vertex is target: put subsitutions on target variables too
            for i, xt_i in enumerate(self.right.xt):
                if right_vars[i] in substitutions:
                    substitutions[xt_i] = substitutions[right_vars[i]]
                else:
                    substitutions[xt_i] = right_vars[i]
        else:
            unique_variables.append(self.right.xt)
            if len(self.right.target_set_linear_inequalities) > 0:
                all_linear_inequalities.append(self.right.target_set_linear_inequalities)
            if len(self.right.target_set_quadratic_inequalities) > 0:
                all_quadratic_inequalities.append(self.right.target_set_quadratic_inequalities)


        xt_vars_subed = np.array([substitutions[x] if x in substitutions else x for x in self.left.xt ])
        xr_vars_subed = np.array([substitutions[x] if x in substitutions else x for x in right_vars ])
        edge_cost = self.cost_function_surrogate(self.left.x, self.u, xr_vars_subed, xt_vars_subed)
        left_potential = self.left.potential
        x_and_xt_and_1 = np.hstack(([1], xr_vars_subed, xt_vars_subed))
        right_potential = np.sum(self.right.J_matrix * np.outer(x_and_xt_and_1, x_and_xt_and_1))
        expr = (
            edge_cost
            + right_potential
            - left_potential
            + self.bidirectional_edge_violation
            + self.right.total_flow_in_violation
        )

        # edge_cost = self.cost_function_surrogate(self.left.x, self.u, right_vars, self.left.xt)
        # left_potential = self.left.potential
        # if self.left.name == self.right.name:
        #     x_and_xt_and_1 = np.hstack(([1], right_vars, self.right.xt))
        #     right_potential = np.sum(self.right.J_matrix * np.outer(x_and_xt_and_1, x_and_xt_and_1))
        # else:
        #     right_potential = self.right.potential
        # expr = (
        #     edge_cost
        #     + right_potential
        #     - left_potential
        #     + self.bidirectional_edge_violation
        #     + self.right.total_flow_in_violation
        # )


        print("num_unique_vars", len(np.hstack(unique_variables).flatten()))
        define_sos_constraint_over_polyhedron_multivar_new(
            prog,
            Variables(np.hstack(unique_variables).flatten()),
            all_linear_inequalities,
            all_quadratic_inequalities,
            edge_equality_constraints,
            substitutions,
            expr, 
            self.options,
        )


class PolynomialDualGCS:
    def __init__(self, options: ProgramOptions, target_convex_set:ConvexSet, target_cost_matrix:npt.NDArray = None, target_policy_terminating_condition:ConvexSet=None):
        # variables creates for policy synthesis
        self.vertices = dict()  # type: T.Dict[str, DualVertex]
        self.edges = dict()  # type: T.Dict[str, DualEdge]
        self.prog = MathematicalProgram()  # type: MathematicalProgram
        self.value_function_solution = None  # type: MathematicalProgramResult
        self.options = options
        self.push_up_vertices = [] # type: T.List[T.Tuple[str, npt.NDArray, npt.NDArray]]

        self.bidir_flow_violation_matrices = []

        self.target_convex_set = target_convex_set
        self.target_moment_matrix = get_moment_matrix_for_a_measure_over_set(target_convex_set)
        self.target_state_dim = self.target_convex_set.ambient_dimension()
        self.xt = self.prog.NewIndeterminates(self.target_state_dim)
        if target_cost_matrix is None:
            target_cost_matrix = np.zeros((2*self.target_state_dim+1, 2*self.target_state_dim+1))
        vt = DualVertex(
            "target",
            self.prog,
            self.target_convex_set,
            self.target_convex_set,
            self.xt,
            options=options,
            vertex_is_target=True,
            target_policy_terminating_condition=target_policy_terminating_condition,
            target_cost_matrix=target_cost_matrix,
        )
        self.vertices["target"] = vt


    def AddVertex(
        self,
        name: str,
        convex_set: ConvexSet,
        options:ProgramOptions=None,
        vertex_is_start: bool = False,
    )->DualVertex:
        """
        Options will default to graph initialized options if not specified
        """
        assert name not in self.vertices
        if options is None:
            options = self.options
        # add vertex to policy graph
        v = DualVertex(
            name,
            self.prog,
            convex_set,
            self.target_convex_set,
            self.xt,
            options=options,
            vertex_is_start=vertex_is_start,
        )
        self.vertices[name] = v
        return v
    
    def MaxCostOverVertex(self, vertex:DualVertex):
        vertex_moments = get_moment_matrix_for_a_measure_over_set(vertex.convex_set)
        self.push_up_vertices.append((vertex.name, vertex_moments, self.target_moment_matrix))

    def PushUpOnPotentialsAtVertex(self, vertex:DualVertex, vertex_moments:npt.NDArray, target_vertex_moments:npt.NDArray):
        self.push_up_vertices.append((vertex.name, vertex_moments, target_vertex_moments))

    def BuildTheProgram(self):
        INFO("pushing up")
        for (v_name, v_moments, vt_moments) in tqdm(self.push_up_vertices):
            self.vertices[v_name].push_up_on_potentials(self.prog, v_moments, vt_moments)

            if not self.options.allow_vertex_revisits:
                for v in self.vertices.values():
                    v.push_down_on_flow_violation(self.prog, vt_moments)

                # add penalty cost on edge penelties
                for mat in self.bidir_flow_violation_matrices:
                    self.prog.AddLinearCost( np.sum(mat * vt_moments))

        INFO("adding edge polynomial constraints")
        for edge in tqdm(self.edges.values()):
            edge.define_edge_polynomials_and_sos_constraints(self.prog)

    def AddBidirectionalEdges(
        self,
        v_left: DualVertex,
        v_right: DualVertex,
        cost_function: T.Callable,
        cost_function_surrogate: T.Callable,
        options: ProgramOptions = None,
    ) -> T.Tuple[DualEdge, DualEdge]:
        """
        adding two edges
        """
        if options is None:
            options = self.options

        if not self.options.allow_vertex_revisits:
            bidirectional_edge_violation, bidirectional_edge_violation_mat = make_potential(self.xt, PSD_POLY, self.options.flow_violation_polynomial_degree, self.prog)
            self.bidir_flow_violation_matrices.append(bidirectional_edge_violation_mat)
        else:
            bidirectional_edge_violation = Expression(0)
        v_lr = self.AddEdge(v_left, v_right, cost_function, cost_function_surrogate, options, bidirectional_edge_violation)
        v_rl = self.AddEdge(v_right, v_left, cost_function, cost_function_surrogate, options, bidirectional_edge_violation)
        return v_lr, v_rl

    def AddEdge(
        self,
        v_left: DualVertex,
        v_right: DualVertex,
        cost_function: T.Callable,
        cost_function_surrogate: T.Callable,
        options: ProgramOptions = None,
        bidirectional_edge_violation=Expression(0),
    ) -> DualEdge:
        """
        Options will default to graph initialized options if not specified
        """
        if options is None:
            options = self.options
        edge_name = get_edge_name(v_left.name, v_right.name)
        assert edge_name not in self.edges
        e = DualEdge(
            edge_name,
            v_left,
            v_right,
            cost_function,
            cost_function_surrogate,
            options=options,
            bidirectional_edge_violation=bidirectional_edge_violation,
        )
        self.edges[edge_name] = e
        v_left.add_edge_out(edge_name)
        v_right.add_edge_in(edge_name)
        return e
    
    def SolvePolicy(self) -> MathematicalProgramResult:
        """
        Synthesize a policy over the graph.
        Policy is stored in the solution: you'd need to extract it per vertex.
        """
        self.BuildTheProgram()

        timer = timeit()
        mosek_solver = MosekSolver()
        solver_options = SolverOptions()

        # set the solver tolerance gaps
        solver_options.SetOption(
            MosekSolver.id(),
            "MSK_DPAR_INTPNT_CO_TOL_REL_GAP",
            self.options.value_synthesis_MSK_DPAR_INTPNT_CO_TOL_REL_GAP,
        )
        solver_options.SetOption(
            MosekSolver.id(),
            "MSK_DPAR_INTPNT_CO_TOL_PFEAS",
            self.options.value_synthesis_MSK_DPAR_INTPNT_CO_TOL_PFEAS,
        )
        solver_options.SetOption(
            MosekSolver.id(),
            "MSK_DPAR_INTPNT_CO_TOL_DFEAS",
            self.options.value_synthesis_MSK_DPAR_INTPNT_CO_TOL_DFEAS,
        )

        if self.options.value_synthesis_use_robust_mosek_parameters:
            solver_options.SetOption(MosekSolver.id(), "MSK_DPAR_INTPNT_CO_TOL_REL_GAP", 1e-3)
            solver_options.SetOption(MosekSolver.id(), "MSK_IPAR_INTPNT_SOLVE_FORM", 1)

        # solve the program
        self.value_function_solution = mosek_solver.Solve(self.prog, solver_options=solver_options)
        timer.dt("Solve")
        diditwork(self.value_function_solution)
        for v in self.vertices.values():
            v.J_matrix_solution = self.value_function_solution.GetSolution(v.J_matrix).reshape(v.J_matrix.shape)
            if not np.issubdtype(v.J_matrix_solution.dtype, np.number):
                vec_eval = np.vectorize(lambda x: x.Evaluate())
                v.J_matrix_solution = vec_eval(v.J_matrix_solution)

        return self.value_function_solution
    
    
    def export_a_gcs(self, target_state:npt.NDArray) -> T.Tuple[GraphOfConvexSets, T.Dict[str, GraphOfConvexSets.Vertex], GraphOfConvexSets.Vertex]:
        raise NotImplementedError("need to rewrite. in particular, troubles expected when walks are allowed.")
        gcs = GraphOfConvexSets()
        target_vertex = gcs.AddVertex(Hyperrectangle([],[]), "the_target_vertex")

        gcs_vertices = dict()
        state_dim = 0

        # create all the vertices
        for v in self.vertices.values():
            state_dim = v.state_dim
            gcs_v = gcs.AddVertex(v.convex_set, v.name)
            if v.vertex_is_target:
                gcs.AddEdge(gcs_v, target_vertex, name = get_edge_name(v.name, target_vertex.name()))
            gcs_vertices[v.name] = gcs_v

        A = np.hstack( (np.eye(state_dim),-np.eye(state_dim)) )
        b = np.zeros(state_dim)
        cost = L2NormCost(A, b)

        # TODO: fix this, need to add edge constraints and stuff

        for e in self.edges.values():
            left_gcs_v = gcs_vertices[ e.left.name ]
            right_gcs_v = gcs_vertices[ e.right.name ]
            gcs_e = gcs.AddEdge(left_gcs_v, right_gcs_v, e.name)
            # add cost
            left_point, right_point = gcs_e.xu(), gcs_e.xv()
            # if self.options.policy_use_l2_norm_cost:
            #     gcs_e.AddCost(Binding[L2NormCost](cost, np.hstack((left_point, right_point))))
            # elif self.options.policy_use_quadratic_cost:
            #     gcs_e.AddCost((left_point-right_point).dot(left_point-right_point))
            # else:
            #     gcs_e.AddCost(e.cost_function(left_point, right_point, target_state))

        return gcs, gcs_vertices, target_vertex