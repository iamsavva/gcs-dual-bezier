import typing as T  # pylint: disable=unused-import

import numpy as np
import numpy.typing as npt

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
)
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
import plotly.graph_objs as go
from plotly.subplots import make_subplots

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
    make_moment_matrix
)  # pylint: disable=import-error, no-name-in-module, unused-import

from gcs_util import get_edge_name, make_quadratic_cost_function_matrices
from polynomial_dual_gcs_utils import (
    define_quadratic_polynomial,
    get_product_constraints,
    make_linear_set_inequalities, 
    get_B_matrix,
    define_sos_constraint_over_polyhedron_multivar,
    make_potential
    # define_general_nonnegativity_constraint_on_a_set
)
from graph_dual import DualVertex, DualEdge, PolynomialDualGCS

delta = 0.001
QUADRATIC_COST_GC = lambda x, y, z: np.sum([(x[i] - y[i]) ** 2 for i in range(len(x))])
QUADRATIC_COST_AUGMENTED_GC = lambda x, y, z: np.sum(
    [(x[i] - y[i]) ** 2 for i in range(len(x))]
) + delta * 0.5* (np.sum([ (x[i]-z[i]) ** 2 + (y[i]-z[i]) ** 2 for i in range(len(x))]))


class GoalConditionedDualVertex(DualVertex):
    def __init__(
        self,
        name: str,
        prog: MathematicalProgram,
        convex_set: ConvexSet,
        terminal_convex_set: ConvexSet,
        xt: npt.NDArray,
        options: ProgramOptions,
        vertex_is_start: bool = False,
        vertex_is_target: bool = False,
    ):
        self.name = name
        self.options = options
        self.vertex_is_start = vertex_is_start
        self.vertex_is_target = vertex_is_target
        self.potential = Expression(0)
        self.J_matrix = None

        # TODO: handle ellipsoids exactly
        # TODO: handle points exactly
        self.convex_set = convex_set
        if isinstance(convex_set, HPolyhedron):
            self.set_type = HPolyhedron
            self.state_dim = convex_set.A().shape[1]
        elif isinstance(convex_set, Hyperrectangle):
            self.set_type = Hyperrectangle
            self.state_dim = convex_set.lb().shape[0]
        else:
            self.set_type = None
            self.state_dim = None
            raise Exception("bad state set")
        
        self.terminal_convex_set = terminal_convex_set
        if isinstance(terminal_convex_set, HPolyhedron):
            self.terminal_set_type = HPolyhedron
        elif isinstance(terminal_convex_set, Hyperrectangle):
            self.terminal_set_type = Hyperrectangle
        else:
            self.terminal_set_type = None
            raise Exception("bad terminal state set")
        
        self.xt = xt
        self.define_variables(prog)
        self.define_set_inequalities()
        self.define_potential(prog)

        self.edges_in = []  # type: T.List[str]
        self.edges_out = []  # type: T.List[str]

    def set_J_matrix(self, J_matrix: npt.NDArray):
        self.J_matrix = J_matrix
        assert False, "need to write a check to ensure that provided J matrix matches the potential"


    def get_hpoly(self) -> HPolyhedron:
        # TODO: BAD
        assert self.set_type in (HPolyhedron, Hyperrectangle), "can't get hpoly for set"
        if self.set_type == HPolyhedron:
            return self.convex_set
        if self.set_type == Hyperrectangle:
            return self.convex_set.MakeHPolyhedron()
        
    def get_terminal_hpoly(self) -> HPolyhedron:
        # TODO: OK?
        assert self.terminal_set_type in (HPolyhedron, Hyperrectangle), "can't get hpoly for set"
        if self.terminal_set_type == HPolyhedron:
            return self.terminal_convex_set
        if self.terminal_set_type == Hyperrectangle:
            return self.terminal_convex_set.MakeHPolyhedron()

    def add_edge_in(self, name: str):
        assert name not in self.edges_in
        self.edges_in.append(name)

    def add_edge_out(self, name: str):
        assert not self.vertex_is_target, "adding an edge to a target vertex"
        assert name not in self.edges_out
        self.edges_out.append(name)

    def define_variables(self, prog: MathematicalProgram):
        self.x = prog.NewIndeterminates(self.state_dim, "x_" + self.name)
        # self.xt = prog.NewIndeterminates(self.state_dim, "xt_" + self.name)
        self.vars = Variables(np.hstack((self.x, self.xt)))

        if self.options.flow_violation_polynomial_degree == 0 or self.vertex_is_target:
            self.total_flow_in_violation, self.total_flow_in_violation_mat = make_potential(self.xt, PSD_POLY, 0, prog)
            if self.vertex_is_target:
                prog.AddLinearConstraint(self.total_flow_in_violation == 0)
        elif self.options.flow_violation_polynomial_degree == 2:
            self.total_flow_in_violation, self.total_flow_in_violation_mat = make_potential(self.xt, PSD_POLY, 2, prog)
        else:
            raise Exception("bad vilation polynomial degree " + str(self.options.flow_violation_polynomial_degree))

    def define_set_inequalities(self):
        """
        Assuming that sets are polyhedrons
        """
        # self.linear_set_inequalities = make_linear_set_inequalities(self.x, self.convex_set, self.set_type)
        # self.quadratic_set_inequalities = get_product_constraints(self.linear_set_inequalities)
        # TODO: drop the polyhedral assumption
        # in principle, expectations should be taken differently
        hpoly = self.get_hpoly()
        # inequalities of the form b[i] - a.T x = g_i(x) >= 0
        A, b = hpoly.A(), hpoly.b()
        self.B = np.hstack((b.reshape((len(b), 1)), -A))

        # terminal
        hpoly = self.get_terminal_hpoly()
        # inequalities of the form b[i] - a.T x = g_i(x) >= 0
        A, b = hpoly.A(), hpoly.b()
        self.B_terminal = np.hstack((b.reshape((len(b), 1)), -A))

        self.E = None # TODO: fix me
        

    def define_potential(self, prog: MathematicalProgram):
        # provided a specific potential to use
        if self.vertex_is_target:
            self.potential = Expression(0)
        else:
            x_and_xt = np.hstack((self.x, self.xt))
            self.potential, self.J_matrix = make_potential(x_and_xt, self.options.pot_type, self.options.potential_poly_deg, prog)

    def cost_at_point(self, x: npt.NDArray, xt:npt.NDArray, solution: MathematicalProgramResult):
        """
        Evaluate potential at a particular point.
        Return expression if solution not passed, returns a float value if solution is passed.
        """
        assert len(x) == self.state_dim
        potential = solution.GetSolution(self.potential)
        sub_x = potential.Substitute({self.x[i]: x[i] for i in range(self.state_dim)})
        sub_xt = sub_x.Substitute({self.xt[i]: xt[i] for i in range(self.state_dim)}).Evaluate()
        return sub_xt


    def cost_of_moment_measure(self, moment_matrix:npt.NDArray) -> Expression:
        assert self.J_matrix is not None
        assert moment_matrix.shape == (2*self.state_dim+1, 2*self.state_dim+1)
        # assert that moment matrix satisfies necessary conditions to be supported on set
        e1 = np.zeros(2*self.state_dim + 1)
        e1[0] = 1
        
        hpoly, hpoly_t = self.get_hpoly(), self.get_terminal_hpoly()
        As, bs = hpoly.A(), hpoly.b()
        At, bt = hpoly_t.A(), hpoly_t.b()

        B_row1 = np.hstack( (bs.reshape((len(bs), 1)), -As, np.zeros((As.shape[0], At.shape[1])) ) )
        B_row2 = np.hstack( (bt.reshape((len(bt), 1)), np.zeros((At.shape[0], As.shape[1])), -At ) )
        B = np.vstack((B_row1, B_row2))

        eps = 1e-6
        assert np.all(B.dot(moment_matrix).dot(e1) >= -eps), "moment matrix not supported on set"
        assert np.all(B.dot(moment_matrix).dot(B.T) >= -eps), "moment matrix not supported on set"
        return np.sum(self.J_matrix * moment_matrix)
    
    def push_down_on_flow_violation(self, prog:MathematicalProgram, terminal_moment_matrix:npt.NDArray):
        # add the cost on violations
        prog.AddLinearCost(np.sum(terminal_moment_matrix * self.total_flow_in_violation_mat))



class GoalConditionedDualEdge(DualEdge):
    def __init__(
        self,
        name: str,
        v_left: GoalConditionedDualVertex,
        v_right: GoalConditionedDualVertex,
        prog: MathematicalProgram,
        cost_function: T.Callable,
        options: ProgramOptions,
        bidirectional_edge_violation=Expression(0),
    ):
        self.name = name
        self.left = v_left
        self.right = v_right

        self.cost_function = cost_function
        self.options = options

        self.bidirectional_edge_violation = bidirectional_edge_violation

        self.define_edge_polynomials_and_sos_constraints(prog)

    def define_edge_polynomials_and_sos_constraints(self, prog: MathematicalProgram):
        """
        define edge appropriate SOS constraints
        """
        opt = self.options

        # -----------------------------------------------
        # -----------------------------------------------
        # define n-2 intermediate potentials (at least 1 intermediary point in the set)
        xt = self.left.xt
        Bt = self.left.B_terminal


        # -----------------------------------------------
        # -----------------------------------------------
        # make n+1 SOS constraints

        # -----------------------------------------------
        # J_{v} to J_{vw,1}
        x_left, x_right = self.left.x, self.right.x
        B_left, B_right = self.left.B, self.right.B
        E_left, E_right, E_t = self.left.E, self.right.E, None
        edge_cost = self.cost_function(x_left, x_right, xt)
        left_potential = self.left.potential
        right_potential = self.right.potential

        # NOTE: adding bidriectional edge violation just to the last constraint
        if self.right.vertex_is_target:
            edge_cost = self.cost_function(x_left, xt, xt)
            expr = (
                edge_cost
                - left_potential
                + self.bidirectional_edge_violation
                + self.right.total_flow_in_violation
            )
            define_sos_constraint_over_polyhedron_multivar(prog, [x_left, xt], [B_left, Bt], expr, opt, [E_left, E_right])
        else:
            expr = (
                edge_cost
                + right_potential
                - left_potential
                + self.bidirectional_edge_violation
                + self.right.total_flow_in_violation
            )
            define_sos_constraint_over_polyhedron_multivar(prog, [x_left, x_right, xt], [B_left, B_right, Bt], expr, opt, [E_left, E_right, E_t])


class GoalConditionedPolynomialDualGCS(PolynomialDualGCS):
    def __init__(self, options: ProgramOptions, terminal_convex_set:ConvexSet):
        # variables creates for policy synthesis
        self.vertices = dict()  # type: T.Dict[str, GoalConditionedDualVertex]
        self.edges = dict()  # type: T.Dict[str, GoalConditionedDualEdge]
        self.prog = MathematicalProgram()  # type: MathematicalProgram
        self.value_function_solution = None  # type: MathematicalProgramResult
        self.options = options
        self.terminal_convex_set = terminal_convex_set
        t_hpoly = self.get_terminal_hpoly()
        self.xt = self.prog.NewIndeterminates(t_hpoly.A().shape[1])
        terminal_ellipsoid = t_hpoly.MaximumVolumeInscribedEllipsoid()
        self.terminal_mu = terminal_ellipsoid.center()
        self.terminal_sigma = np.linalg.inv(terminal_ellipsoid.A().T.dot(terminal_ellipsoid.A()))
        self.terminal_vertex_set = False
        self.terminal_vertex = self.AddTargetVertex("terminal")
        self.pushing_up = False
        self.bidir_flow_violation_matrices = [] # type: T.List[npt.NDArray]

    def get_terminal_hpoly(self) -> HPolyhedron:
        if isinstance(self.terminal_convex_set, HPolyhedron):
            return self.terminal_convex_set
        elif isinstance(self.terminal_convex_set, Hyperrectangle):
            return self.terminal_convex_set.MakeHPolyhedron()
        else:
            raise Exception("terminal set must be polyhedron for now")

    def AddVertex(
        self,
        name: str,
        convex_set: ConvexSet,
        options:ProgramOptions=None,
        vertex_is_start: bool = False,
    )->GoalConditionedDualVertex:
        """
        Options will default to graph initialized options if not specified
        """
        assert name not in self.vertices
        if options is None:
            options = self.options
        if self.pushing_up:
            raise Exception("adding a vertex after pushing up, that's bad")
        # add vertex to policy graph
        v = GoalConditionedDualVertex(
            name,
            self.prog,
            convex_set,
            self.terminal_convex_set,
            self.xt,
            options=options,
            vertex_is_start=vertex_is_start,
        )
        self.vertices[name] = v
        return v
    
    def MaxCostOverVertex(self, vertex:GoalConditionedDualVertex):
        hpoly = vertex.get_hpoly()
        ellipsoid = hpoly.MaximumVolumeInscribedEllipsoid()
        mu = ellipsoid.center()
        sigma = np.linalg.inv(ellipsoid.A().T.dot(ellipsoid.A()))
        assert np.all(np.linalg.eigvals(sigma) >= 1e-4)
        m1_s = mu
        m2_s = np.outer(mu, mu) + sigma

        m1_t = self.terminal_mu
        m2_t = np.outer(m1_t, m1_t) + self.terminal_sigma

        row1 = np.hstack((1, m1_s, m1_t))
        row2 = np.hstack(( m1_s.reshape((len(m1_s), 1)), m2_s, np.outer(m1_s, m1_t) ))
        row3 = np.hstack( (m1_t.reshape((len(m1_t), 1)), np.outer(m1_t, m1_s), m2_t ))
        moment_matrix = np.vstack((row1, row2, row3))

        cost = -vertex.cost_of_moment_measure(moment_matrix)
        self.prog.AddLinearCost(cost)

        terminal_moment_matrix = make_moment_matrix(1, m1_t, m2_t)
        if not self.pushing_up:
            self.pushing_up = True

        # add penalty cost on vertex penelties
        for v in self.vertices.values():
           v.push_down_on_flow_violation(self.prog, terminal_moment_matrix)

        # add penalty cost on edge penelties
        for mat in self.bidir_flow_violation_matrices:
            self.prog.AddLinearCost( np.sum(mat * terminal_moment_matrix))


    def AddTargetVertex(
        self,
        name: str,
        options: ProgramOptions = None,
    ) -> GoalConditionedDualVertex:
        """
        Options will default to graph initialized options if not specified

        Target vertices are vertices with fixed potentials functions.

        HPolyhedron with quadratics, or
        Point and 0 potential.
        """
        if self.terminal_vertex_set:
            raise Exception("terminal vertex being set the second time! goal conditioned policy cna have only one.")
        else:
            self.terminal_vertex_set = True
        assert name not in self.vertices
        if options is None:
            options = self.options

        v = GoalConditionedDualVertex(
            name,
            self.prog,
            self.terminal_convex_set,
            self.terminal_convex_set,
            self.xt,
            options=options,
            vertex_is_target=True,
        )
        # building proper GCS
        self.vertices[name] = v
        return v

    def AddBidirectionalEdge(
        self,
        v_left: GoalConditionedDualVertex,
        v_right: GoalConditionedDualVertex,
        cost_function: T.Callable,
        options: ProgramOptions = None,
    ) -> None:
        """
        adding two edges
        """
        if self.pushing_up:
            raise Exception("adding bidir edges after pushing up, bad")
        
        if options is None:
            options = self.options

        bidirectional_edge_violation, bidirectional_edge_violation_mat = make_potential(self.xt, PSD_POLY, self.options.flow_violation_polynomial_degree, self.prog)
        self.bidir_flow_violation_matrices.append(bidirectional_edge_violation_mat)

        self.AddEdge(v_left, v_right, cost_function, options, bidirectional_edge_violation)
        self.AddEdge(v_right, v_left, cost_function, options, bidirectional_edge_violation)

    def AddEdge(
        self,
        v_left: GoalConditionedDualVertex,
        v_right: GoalConditionedDualVertex,
        cost_function: T.Callable,
        options: ProgramOptions = None,
        bidirectional_edge_violation=Expression(0),
    ) -> GoalConditionedDualEdge:
        """
        Options will default to graph initialized options if not specified
        """
        if self.pushing_up:
            raise Exception("adding bidir edges after pushing up, bad")
        if options is None:
            options = self.options
        edge_name = get_edge_name(v_left.name, v_right.name)
        e = GoalConditionedDualEdge(
            edge_name,
            v_left,
            v_right,
            self.prog,
            cost_function,
            options=options,
            bidirectional_edge_violation=bidirectional_edge_violation,
        )
        self.edges[edge_name] = e
        v_left.add_edge_out(edge_name)
        v_right.add_edge_in(edge_name)
        return e
    
    def export_a_gcs(self, terminal_state:npt.NDArray) -> T.Tuple[GraphOfConvexSets, T.Dict[str, GraphOfConvexSets.Vertex], GraphOfConvexSets.Vertex]:
        gcs = GraphOfConvexSets()
        terminal_vertex = gcs.AddVertex(Hyperrectangle([],[]), "the_terminal_vertex")

        gcs_vertices = dict()
        state_dim = 0

        # create all the vertices
        for v in self.vertices.values():
            state_dim = v.state_dim
            gcs_v = gcs.AddVertex(v.convex_set, v.name)
            if v.vertex_is_target:
                gcs.AddEdge(gcs_v, terminal_vertex, name = get_edge_name(v.name, terminal_vertex.name()))
            gcs_vertices[v.name] = gcs_v

        A = np.hstack( (np.eye(state_dim),-np.eye(state_dim)) )
        b = np.zeros(state_dim)
        cost = L2NormCost(A, b)

        for e in self.edges.values():
            left_gcs_v = gcs_vertices[ e.left.name ]
            right_gcs_v = gcs_vertices[ e.right.name ]
            gcs_e = gcs.AddEdge(left_gcs_v, right_gcs_v, e.name)
            # add cost
            left_point, right_point = gcs_e.xu(), gcs_e.xv()
            if self.options.policy_use_l2_norm_cost:
                gcs_e.AddCost(Binding[L2NormCost](cost, np.hstack((left_point, right_point))))
            elif self.options.policy_use_quadratic_cost:
                gcs_e.AddCost((left_point-right_point).dot(left_point-right_point))
            else:
                gcs_e.AddCost(e.cost_function(left_point, right_point, terminal_state))

        return gcs, gcs_vertices, terminal_vertex