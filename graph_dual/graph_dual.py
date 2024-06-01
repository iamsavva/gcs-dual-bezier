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
    Hyperellipsoid
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
    # define_sos_constraint_over_polyhedron,
    make_potential,
    define_sos_constraint_over_polyhedron_multivar
    # define_general_nonnegativity_constraint_on_a_set
)

delta = 0.01
QUADRATIC_COST = lambda x, y: np.sum([(x[i] - y[i]) ** 2 for i in range(len(x))])
QUADRATIC_COST_AUGMENTED = lambda x, y: np.sum(
    [(x[i] - y[i]) ** 2 for i in range(len(x))]
) + delta * (np.sum([x[i] ** 2 + y[i] ** 2 for i in range(len(x))]))

class DualVertex:
    def __init__(
        self,
        name: str,
        prog: MathematicalProgram,
        convex_set: ConvexSet,
        options: ProgramOptions,
        specific_potential: T.Callable = None, 
        vertex_is_start: bool = False,
        vertex_is_target: bool = False,
    ):
        self.name = name
        self.options = options
        self.vertex_is_start = vertex_is_start
        self.vertex_is_target = vertex_is_target
        self.potential = Expression(0)
        self.J_matrix = None

        # TODO: handle points exactly
        
        self.convex_set = convex_set
        if isinstance(convex_set, HPolyhedron):
            self.set_type = HPolyhedron
            self.state_dim = convex_set.A().shape[1]
        elif isinstance(convex_set, Hyperrectangle):
            self.set_type = Hyperrectangle
            self.state_dim = convex_set.lb().shape[0]
        elif isinstance(convex_set, Hyperellipsoid):
            self.set_type = Hyperellipsoid
            self.state_dim = len(convex_set.center())
        else:
            self.set_type = None
            self.state_dim = None
            raise Exception("bad state set")

        self.define_variables(prog)
        self.define_set_inequalities()
        self.define_potential(prog, specific_potential)

        self.edges_in = []  # type: T.List[str]
        self.edges_out = []  # type: T.List[str]

    def set_J_matrix(self, J_matrix: npt.NDArray):
        self.J_matrix = J_matrix


    def get_hpoly(self) -> HPolyhedron:
        assert self.set_type in (HPolyhedron, Hyperrectangle), "can't get hpoly for set"
        if self.set_type == HPolyhedron:
            return self.convex_set
        if self.set_type == Hyperrectangle:
            return self.convex_set.MakeHPolyhedron()
        raise Exception("shouldn't be calling this on other sets!")

    def add_edge_in(self, name: str):
        assert name not in self.edges_in
        self.edges_in.append(name)

    def add_edge_out(self, name: str):
        assert not self.vertex_is_target, "adding an edge to a target vertex"
        assert name not in self.edges_out
        self.edges_out.append(name)

    def define_variables(self, prog: MathematicalProgram):
        self.x = prog.NewIndeterminates(self.state_dim, "x_" + self.name)
        self.vars = Variables(self.x)
        self.total_flow_in_violation = prog.NewContinuousVariables(1)[0]
        # prog.AddLinearConstraint(self.total_flow_in_violation == 0)
        # if self.name == "p0":
        #     prog.AddLinearConstraint(self.total_flow_in_violation == 3)
    
        if not self.vertex_is_target:
            prog.AddLinearConstraint(self.total_flow_in_violation >= 0)
            prog.AddLinearCost(self.total_flow_in_violation * self.options.max_flow_through_edge)
        else:
            prog.AddLinearConstraint(self.total_flow_in_violation == 0)

    def define_set_inequalities(self):
        """
        Assuming that sets are polyhedrons
        """
        if self.set_type in (Hyperrectangle, HPolyhedron):
            hpoly = self.get_hpoly()
            # inequalities of the form b[i] - a.T x = g_i(x) >= 0
            A, b = hpoly.A(), hpoly.b()
            self.B = np.hstack((b.reshape((len(b), 1)), -A))
            self.E = None
        elif isinstance(self.convex_set, Hyperellipsoid):
            self.B = None
            mu = self.convex_set.center()
            A = self.convex_set.A()
            c11 = 1 - mu.T.dot(A.T).dot(A).dot(mu)
            c12 = mu.T.dot(A.T).dot(A)
            c21 = A.T.dot(A).dot(mu).reshape((len(mu), 1))
            c22 = -A.T.dot(A)
            self.E = np.vstack((np.hstack((c11,c12)), np.hstack((c21,c22))))
        else:
            raise Exception("set type not supported")

        

    def define_potential(self, prog: MathematicalProgram, specific_potential: T.Callable = None):
        # provided a specific potential to use
        if specific_potential is not None:
            self.potential = specific_potential(self.x)
        else:
            self.potential, self.J_matrix = make_potential(self.x, self.options.pot_type, self.options.potential_poly_deg, prog)

    def cost_at_point(self, x: npt.NDArray, solution: MathematicalProgramResult = None):
        """
        Evaluate potential at a particular point.
        Return expression if solution not passed, returns a float value if solution is passed.
        """
        assert len(x) == self.state_dim
        # assert self.convex_set.PointInSet(x, 1e-5)
        if solution is None:
            return self.potential.Substitute({self.x[i]: x[i] for i in range(self.state_dim)})
        else:
            potential = solution.GetSolution(self.potential)
            return potential.Substitute({self.x[i]: x[i] for i in range(self.state_dim)}).Evaluate()

    def cost_of_uniform_integral_over_box(self, lb:npt.NDArray, ub:npt.NDArray, solution: MathematicalProgramResult = None):
        """
        Return expression if solution not passed, returns a float value if solution is passed.
        """
        assert len(lb) == len(ub) == self.state_dim
        # compute by integrating each monomial term
        monomial_to_coef_map = Polynomial(self.potential).monomial_to_coefficient_map()
        expectation = Expression(0)
        for monomial in monomial_to_coef_map.keys():
            coef = monomial_to_coef_map[monomial]
            poly = Polynomial(monomial)
            for i in range(self.state_dim):
                x_min, x_max, x_val = lb[i], ub[i], self.x[i]
                integral_of_poly = poly.Integrate(x_val)
                poly = (
                    integral_of_poly.EvaluatePartial({x_val: x_max})
                    - integral_of_poly.EvaluatePartial({x_val: x_min})
                ) / (x_max - x_min)
            expectation += coef * poly.ToExpression()

        if solution is None:
            return expectation
        else:
            return solution.GetSolution(expectation)

    def cost_of_small_uniform_box_around_point(
        self, point: npt.NDArray, solution: MathematicalProgramResult = None, eps:float=0.001
    ):
        """
        Return expression if solution not passed, returns a float value if solution is passed.
        """
        assert len(point) == self.state_dim
        return self.cost_of_uniform_integral_over_box(point - eps, point + eps, solution)
    
    def cost_of_moment_measure(self, moment_matrix:npt.NDArray) -> Expression:
        assert self.J_matrix is not None
        assert moment_matrix.shape == (self.state_dim+1, self.state_dim+1)
        # assert that moment matrix satisfies necessary conditions to be supported on set
        e1 = np.zeros(self.state_dim + 1)
        e1[0] = 1
        assert np.all(self.B.dot(moment_matrix).dot(e1) >= 0), "moment matrix not supported on set"
        assert np.all(self.B.dot(moment_matrix).dot(self.B.T) >= 0), "moment matrix not supported on set"
        return np.sum(self.J_matrix * moment_matrix)



class DualEdge:
    def __init__(
        self,
        name: str,
        v_left: DualVertex,
        v_right: DualVertex,
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

        # TODO: fix this for ellipsoids
        x_left, x_right = self.left.x,  self.right.x
        B_left, B_right = self.left.B, self.right.B
        E_left, E_right = self.left.E, self.right.E
        edge_cost = self.cost_function(x_left, x_right)
        left_potential = self.left.potential
        right_potential = self.right.potential

        expr = (
            edge_cost
            + right_potential
            - left_potential
            + self.bidirectional_edge_violation
            + self.right.total_flow_in_violation
        )

        define_sos_constraint_over_polyhedron_multivar(prog, [x_left, x_right], [B_left, B_right], expr, opt, [E_left, E_right])


class PolynomialDualGCS:
    def __init__(self, options: ProgramOptions) -> None:
        # variables creates for policy synthesis
        self.vertices = dict()  # type: T.Dict[str, DualVertex]
        self.edges = dict()  # type: T.Dict[str, DualEdge]
        self.prog = MathematicalProgram()  # type: MathematicalProgram
        self.value_function_solution = None  # type: MathematicalProgramResult

        self.options = options

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
            options=options,
            vertex_is_start=vertex_is_start,
        )
        self.vertices[name] = v
        return v

    def MaxCostOverABox(self, vertex: DualVertex, lb: npt.NDArray, ub: npt.NDArray)->None:
        cost = -vertex.cost_of_uniform_integral_over_box(lb, ub)
        self.prog.AddLinearCost(cost)

    def MaxCostAtAPoint(self, vertex: DualVertex, point:npt.NDArray, scaling=1)->None:
        cost = -vertex.cost_at_point(point)
        self.prog.AddLinearCost(cost * scaling)

    def MaxCostAtSmallIntegralAroundPoint(self, vertex: DualVertex, point, scaling=1, eps=0.001)->None:
        cost = -vertex.cost_of_small_uniform_box_around_point(point, eps=eps)
        self.prog.AddLinearCost(cost * scaling)

    def MaxCostOverMoments(self, vertex:DualVertex, moment_matrix:npt.NDArray) -> None:
        cost = -vertex.cost_of_moment_measure(moment_matrix)
        self.prog.AddLinearCost(cost)
        # TODO: ADD cost on violations here??

    def MaxCostOverVertex(self, vertex:DualVertex):
        hpoly = vertex.get_hpoly()
        ellipsoid = hpoly.MaximumVolumeInscribedEllipsoid()
        mu = ellipsoid.center()
        sigma = np.linalg.inv(ellipsoid.A().T.dot(ellipsoid.A()))
        assert np.all(np.linalg.eigvals(sigma) >= 1e-4)
        m1 = mu
        m2 = np.outer(mu, mu) + sigma

        moment_matrix = make_moment_matrix(1, m1, m2)
        
        self.MaxCostOverMoments(vertex, moment_matrix)
        

    def AddTargetVertexWithQuadraticTerminalCost(
        self,
        name: str,
        convex_set: HPolyhedron,
        Q_terminal: npt.NDArray,
        x_terminal: npt.NDArray,
        options: ProgramOptions = None,
    ) -> DualVertex:
        """
        Options will default to graph initialized options if not specified

        Target vertices are vertices with fixed potentials functions.

        HPolyhedron with quadratics, or
        Point and 0 potential.
        """
        assert name not in self.vertices
        if options is None:
            options = self.options

        assert Q_terminal.shape == (len(x_terminal), len(x_terminal))
        assert x_terminal.shape == (len(x_terminal),)
        J_11 = np.array([[x_terminal.dot(Q_terminal).dot(x_terminal)]])
        J_12 = -Q_terminal.dot(x_terminal).reshape((1, len(x_terminal)))
        J_21 = J_12.T
        J_22 = Q_terminal
        specific_J_matrix = np.vstack((np.hstack((J_11, J_12)), np.hstack((J_21, J_22))))
        def specific_potential(x):
            x_and_1 = np.hstack(([1], x))
            return x_and_1.dot(specific_J_matrix).dot(x_and_1)
        
        v = DualVertex(
            name,
            self.prog,
            convex_set,
            options=options,
            specific_potential=specific_potential,
            vertex_is_target=True,
        )
        v.set_J_matrix(specific_J_matrix)
        # building proper GCS
        self.vertices[name] = v
        return v

    def AddBidirectionalEdge(
        self,
        v_left: DualVertex,
        v_right: DualVertex,
        cost_function: T.Callable,
        options: ProgramOptions = None,
    ) -> None:
        """
        adding two edges
        """
        if options is None:
            options = self.options
        bidirectional_edge_violation = self.prog.NewContinuousVariables(1)[0]
        
        self.prog.AddLinearConstraint(
            bidirectional_edge_violation == 0
        )  
        # self.prog.AddLinearConstraint(
        #     bidirectional_edge_violation >= 0
        # )  
        # self.prog.AddLinearCost(bidirectional_edge_violation * self.options.max_flow_through_edge)
        
        self.AddEdge(v_left, v_right, cost_function, options, bidirectional_edge_violation)
        self.AddEdge(v_right, v_left, cost_function, options, bidirectional_edge_violation)

    def AddEdge(
        self,
        v_left: DualVertex,
        v_right: DualVertex,
        cost_function: T.Callable,
        options: ProgramOptions = None,
        bidirectional_edge_violation=Expression(0),
    ) -> DualEdge:
        """
        Options will default to graph initialized options if not specified
        """
        if options is None:
            options = self.options
        edge_name = get_edge_name(v_left.name, v_right.name)
        e = DualEdge(
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

    def solve_policy(self) -> MathematicalProgramResult:
        """
        Synthesize a policy over the graph.
        Policy is stored in the solution: you'd need to extract it per vertex.
        """
        timer = timeit()
        mosek_solver = MosekSolver()
        solver_options = SolverOptions()

        # set the solver tolerance gaps
        solver_options.SetOption(
            MosekSolver.id(),
            "MSK_DPAR_INTPNT_CO_TOL_REL_GAP",
            self.options.MSK_DPAR_INTPNT_CO_TOL_REL_GAP,
        )
        solver_options.SetOption(
            MosekSolver.id(),
            "MSK_DPAR_INTPNT_CO_TOL_PFEAS",
            self.options.MSK_DPAR_INTPNT_CO_TOL_PFEAS,
        )
        solver_options.SetOption(
            MosekSolver.id(),
            "MSK_DPAR_INTPNT_CO_TOL_DFEAS",
            self.options.MSK_DPAR_INTPNT_CO_TOL_DFEAS,
        )

        if self.options.use_robust_mosek_parameters:
            solver_options.SetOption(MosekSolver.id(), "MSK_DPAR_INTPNT_CO_TOL_REL_GAP", 1e-3)
            solver_options.SetOption(MosekSolver.id(), "MSK_IPAR_INTPNT_SOLVE_FORM", 1)

        # solve the program
        self.value_function_solution = mosek_solver.Solve(self.prog, solver_options=solver_options)
        timer.dt("Solve")
        diditwork(self.value_function_solution)

        return self.value_function_solution
    
    def export_a_gcs(self) -> T.Tuple[GraphOfConvexSets, T.Dict[str, GraphOfConvexSets.Vertex], GraphOfConvexSets.Vertex]:
        gcs = GraphOfConvexSets()
        terminal_vertex = gcs.AddVertex(Hyperrectangle([],[]), "the_terminal_vertex")

        gcs_vertices = dict()
        state_dim = 0

        for v in self.vertices.values():
            state_dim = v.state_dim
            # add all vertices
            gcs_v = gcs.AddVertex(v.convex_set, v.name)
            if v.vertex_is_target:
                # NOTE: we add the terminal cost here
                gcs_v.AddCost( v.cost_at_point( gcs_v.x() ) )
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
                gcs_e.AddCost(e.cost_function(left_point, right_point))

        return gcs, gcs_vertices, terminal_vertex