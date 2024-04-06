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
    get_kth_control_point
)  # pylint: disable=import-error, no-name-in-module, unused-import

from gcs_util import get_edge_name, make_quadratic_cost_function_matrices
from polynomial_dual_gcs_utils import (
    define_quadratic_polynomial,
    get_product_constraints,
    make_linear_set_inequalities, 
    get_B_matrix,
    define_general_nonnegativity_constraint_on_a_set
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
        self.potential_poly_deg = 2  # type: int
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

        self.define_variables(prog)
        self.define_set_inequalities()
        self.define_potential(prog, specific_potential)

        self.edges_in = []  # type: T.List[str]
        self.edges_out = []  # type: T.List[str]

        hpoly = self.get_hpoly()
        cheb_success, _, r = ChebyshevCenter(hpoly)
        assert cheb_success and (r > 1e-5), "vertex set is low dimensional"

    def set_J_matrix(self, J_matrix: npt.NDArray):
        self.J_matrix = J_matrix
        assert False, "need to write a check to ensure that provided J matrix matches the potential"


    def get_hpoly(self) -> HPolyhedron:
        assert self.set_type in (HPolyhedron, Hyperrectangle), "can't get hpoly for set"
        if self.set_type == HPolyhedron:
            return self.convex_set
        if self.set_type == Hyperrectangle:
            return self.convex_set.MakeHPolyhedron()

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
    
        if not self.vertex_is_target:
            prog.AddLinearConstraint(self.total_flow_in_violation >= 0)
            prog.AddLinearCost(self.total_flow_in_violation * self.options.max_flow_through_edge)
        else:
            prog.AddLinearConstraint(self.total_flow_in_violation == 0)

    def define_set_inequalities(self):
        """
        Assuming that sets are polyhedrons
        """
        self.linear_set_inequalities = make_linear_set_inequalities(self.x, self.convex_set, self.set_type)
        self.quadratic_set_inequalities = get_product_constraints(self.linear_set_inequalities)
        

    def define_potential(self, prog: MathematicalProgram, specific_potential: T.Callable = None):
        # provided a specific potential to use
        if specific_potential is not None:
            self.potential = specific_potential(self.x)
        else:
            # quadratic polynomial. special case due to convex functions and special quadratic implementations
            if self.options.potential_poly_deg == 2:
                self.J_matrix, self.potential = define_quadratic_polynomial(
                    prog, self.x, self.options.pot_type,
                )
                self.potential = self.potential
            else:
                # free polynomial
                if self.options.pot_type == FREE_POLY:
                    self.potential = prog.NewFreePolynomial(
                        self.vars, self.potential_poly_deg
                    ).ToExpression()
                # PSD polynomial
                elif self.options.pot_type == PSD_POLY:
                    assert (
                        self.potential_poly_deg % 2 == 0
                    ), "can't make a PSD potential of uneven degree"
                    # potential is PSD polynomial
                    self.potential = prog.NewSosPolynomial(
                        self.vars, self.potential_poly_deg
                    )[0].ToExpression()
                else:
                    raise NotImplementedError("potential type not supported")

        # define G -- the bezier curve continuity vector
        # TODO: i don't like my current implementation of G factors
        if self.vertex_is_target or self.vertex_is_start:
            self.G_matrix = np.zeros((self.state_dim + 1, self.state_dim + 1))
            self.eval_G = lambda x: Expression(0)
        else:
            self.G_matrix = prog.NewSymmetricContinuousVariables(self.state_dim + 1)

            def eval_G(x):
                x_and_1 = np.hstack(([1], x))
                return x_and_1.dot(self.G_matrix).dot(x_and_1)

            self.eval_G = eval_G
            # TODO: check if this matters or not
            if not self.options.use_G_term_in_value_synthesis:
                self.eval_G = lambda x: Expression(0) 

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

        self.intersection_set = self.intersect_left_right_sets()
        self.bidirectional_edge_violation = bidirectional_edge_violation

        self.define_edge_polynomials_and_sos_constraints(prog)

    def intersect_left_right_sets(self) -> HPolyhedron:
        left_hpoly = self.left.get_hpoly()
        right_hpoly = self.right.get_hpoly()
        intersection_hpoly = left_hpoly.Intersection(right_hpoly)
        intersection_hpoly = intersection_hpoly.ReduceInequalities()
        # TODO: forcing hpoly again

        cheb_success, _, r = ChebyshevCenter(intersection_hpoly)
        assert cheb_success and (
            r > 1e-5
        ), "intersection of vertex sets connected by edge is low dimensional"
        return intersection_hpoly
        # A, b = intersection_hpoly.A(), intersection_hpoly.b()
        # B = np.hstack((b.reshape((len(b), 1)), -A))
        # return B

    def define_edge_polynomials_and_sos_constraints(self, prog: MathematicalProgram):
        """
        define edge appropriate SOS constraints
        """

        self.x_vectors = []
        self.potentials = []

        # -----------------------------------------------
        # -----------------------------------------------
        # define n-2 intermediate potentials (at least 1 intermediary point in the set)

        for k in range(self.options.num_control_points - 2):
            x = prog.NewIndeterminates(self.left.state_dim)
            potential = define_quadratic_polynomial(prog, x, self.options.pot_type)[1]
            self.x_vectors.append(x)
            self.potentials.append(potential)

        # -----------------------------------------------
        # -----------------------------------------------
        # make n+1 SOS constraints

        # -----------------------------------------------
        # J_{v} to J_{vw,1}
        x_left, x_right = self.left.x, self.x_vectors[0]
        left_linear_ineq = make_linear_set_inequalities(x_left, self.left.convex_set, self.left.set_type)
        right_linear_ineq = make_linear_set_inequalities(x_right, self.left.convex_set, self.left.set_type)
        linear_ineq = left_linear_ineq + right_linear_ineq
        if self.options.s_procedure_use_quadratic_multilpiers:
            if self.options.s_procedure_quadratic_multiply_left_and_right:
                quadratic_ineq = get_product_constraints(linear_ineq)
            else:
                left_quadratic_ineq = get_product_constraints(left_linear_ineq)
                right_quadratic_ineq = get_product_constraints(right_linear_ineq)
                quadratic_ineq = left_quadratic_ineq + right_quadratic_ineq
        else:
            quadratic_ineq = []

        edge_cost = self.cost_function(x_left, x_right)
        G_of_v = self.left.eval_G(x_right - x_left)
        left_potential = self.left.potential
        right_potential = self.potentials[0]
        # expr = edge_cost + right_potential - left_potential - G_of_v+ self.bidirectional_edge_violation/(self.options.num_control_points-1)
        expr = edge_cost + right_potential - left_potential - G_of_v
        define_general_nonnegativity_constraint_on_a_set(prog, expr, linear_ineq, quadratic_ineq, x_left, x_right, self.options)

        # -------------------------------------------------
        # J_{vw, k} to J_{vw,k+1}
        for k in range(self.options.num_control_points - 3):
            x_left, x_right = self.x_vectors[k], self.x_vectors[k + 1]
            # TODO: repeated code, should in principle be put into a function
            left_linear_ineq = make_linear_set_inequalities(x_left, self.left.convex_set, self.left.set_type)
            right_linear_ineq = make_linear_set_inequalities(x_right, self.left.convex_set, self.left.set_type)
            linear_ineq = left_linear_ineq + right_linear_ineq
            if self.options.s_procedure_use_quadratic_multilpiers:
                if self.options.s_procedure_quadratic_multiply_left_and_right:
                    quadratic_ineq = get_product_constraints(linear_ineq)
                else:
                    left_quadratic_ineq = get_product_constraints(left_linear_ineq)
                    right_quadratic_ineq = get_product_constraints(right_linear_ineq)
                    quadratic_ineq = left_quadratic_ineq + right_quadratic_ineq
            else:
                quadratic_ineq = []

            edge_cost = self.cost_function(x_left, x_right)
            left_potential = self.potentials[k]
            right_potential = self.potentials[k + 1]
            # expr = edge_cost + right_potential - left_potential+ self.bidirectional_edge_violation/(self.options.num_control_points-1)
            expr = edge_cost + right_potential - left_potential
            define_general_nonnegativity_constraint_on_a_set(prog, expr, linear_ineq, quadratic_ineq, x_left, x_right, self.options)

        # -------------------------------------------------
        # J_{vw, n} to J_{w}
        n = self.options.num_control_points - 3
        x_left, x_right = self.x_vectors[n], self.right.x
        
        left_linear_ineq = make_linear_set_inequalities(x_left, self.left.convex_set, self.left.set_type)
        right_linear_ineq = make_linear_set_inequalities(x_right, self.intersection_set, HPolyhedron)
        linear_ineq = left_linear_ineq + right_linear_ineq
        if self.options.s_procedure_use_quadratic_multilpiers:
            if self.options.s_procedure_quadratic_multiply_left_and_right:
                quadratic_ineq = get_product_constraints(linear_ineq)
            else:
                left_quadratic_ineq = get_product_constraints(left_linear_ineq)
                right_quadratic_ineq = get_product_constraints(right_linear_ineq)
                quadratic_ineq = left_quadratic_ineq + right_quadratic_ineq
        else:
            quadratic_ineq = []

        edge_cost = self.cost_function(x_left, x_right)
        G_of_v = self.left.eval_G(x_right - x_left)
        left_potential = self.potentials[n]
        right_potential = self.right.potential
        # NOTE: adding bidriectional edge violation just to the last constraint
        # expr = edge_cost + right_potential + G_of_v - left_potential + self.bidirectional_edge_violation/(self.options.num_control_points-1)
        expr = (
            edge_cost
            + right_potential
            + G_of_v
            - left_potential
            + self.bidirectional_edge_violation
            + self.right.total_flow_in_violation
        )
        define_general_nonnegativity_constraint_on_a_set(prog, expr, linear_ineq, quadratic_ineq, x_left, x_right, self.options)


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
            bidirectional_edge_violation >= 0
        )  # TODO: shouldn't be necessary?
        self.prog.AddLinearCost(bidirectional_edge_violation * self.options.max_flow_through_edge)
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
        k = self.options.num_control_points

        gcs_vertices = dict()

        for v in self.vertices.values():
            # add all vertices
            if v.vertex_is_target:
                gcs_v = gcs.AddVertex(v.get_hpoly(), v.name)
                gcs_v.AddCost( v.cost_at_point( gcs_v.x() ) )
                gcs.AddEdge(gcs_v, terminal_vertex, name = get_edge_name(v.name, terminal_vertex.name()))
            else:    
                convex_set = make_polyhedral_set_for_bezier_curve(v.get_hpoly(), k)
                gcs_v = gcs.AddVertex(convex_set, v.name)
            gcs_vertices[v.name] = gcs_v

        for e in self.edges.values():
            left_gcs_v = gcs_vertices[ e.left.name ]
            right_gcs_v = gcs_vertices[ e.right.name ]
            gcs_e = gcs.AddEdge(left_gcs_v, right_gcs_v, e.name)
            # add cost
            cost = Expression(0)
            cost_function = e.cost_function
            for i in range(k-1):
                left_point = get_kth_control_point(gcs_e.xu(), i, k)
                right_point = get_kth_control_point(gcs_e.xu(), i+1, k)
                cost += cost_function(left_point, right_point)
            gcs_e.AddCost(cost)

            # add bezier curve continuity constraint
            last_point = get_kth_control_point(gcs_e.xu(), k-1, k)
            if e.right.vertex_is_target:
                first_point = gcs_e.xv()
            else:
                first_point = get_kth_control_point(gcs_e.xv(), 0, k)
            cons = eq( last_point, first_point )
            for con in cons:
                gcs_e.AddConstraint(con)
            # add bezier curve smoothness constraint
            if not e.right.vertex_is_target:
                lastlast_point = get_kth_control_point(gcs_e.xu(), k-2, k)
                second_point = get_kth_control_point(gcs_e.xv(), 1, k)
                cons = eq( last_point-lastlast_point, second_point-first_point )
                for con in cons:
                    gcs_e.AddConstraint(con)
        return gcs, gcs_vertices, terminal_vertex


    def get_policy_cost_for_region_plot(self, vertex_name: str):
        vertex = self.vertices[vertex_name]
        assert vertex.set_type == Hyperrectangle, "vertex not a Hyperrectangle, can't make a plot"
        assert vertex.state_dim == 2, "can't plot generate plot for non-2d vertex"

        box = vertex.convex_set
        eps = 0
        N = 100

        X = np.linspace(box.lb()[0] - eps, box.ub()[0] + eps, N)
        Y = np.linspace(box.lb()[1] - eps, box.ub()[1] + eps, N)
        X, Y = np.meshgrid(X, Y)

        def eval_func(x, y):
            expression = vertex.cost_at_point(np.array([x, y]), self.value_function_solution)
            return expression

        evaluator = np.vectorize(eval_func)

        return X, Y, evaluator(X, Y)

    def make_plots(self, fig=None, cmax=30, offset=0):
        if fig is None:
            fig = go.Figure()

        for v_name in self.vertices.keys():
            X, Y, Z = self.get_policy_cost_for_region_plot(v_name)
            # Create filled 3D contours
            fig.add_trace(
                go.Surface(
                    x=X,
                    y=Y,
                    z=Z - offset,
                    surfacecolor=Z - offset,
                    name=v_name,
                    cmin=0,
                    cmax=cmax,
                )
            )

        # Update layout
        fig.update_layout(scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"))

        fig.update_layout(height=800, width=800, title_text="Value functions ")

        return fig

    def make_1d_plot_potential(self, fig: go.Figure, v: DualVertex):
        assert type(v.convex_set) == Hyperrectangle
        assert len(v.convex_set.lb()) == 1
        x_lin = np.linspace(v.convex_set.lb()[0], v.convex_set.ub()[0], 100, endpoint=True)

        def evaluate_0(y):
            potential = self.value_function_solution.GetSolution(v.potential)
            f_potential = lambda x: potential.Substitute({v.x[i]: x[i] for i in range(v.state_dim)})
            return f_potential(np.array([y])).Evaluate()

        offset = self.options.zero_offset + self.options.policy_gcs_edge_cost_offset
        # fig.update_layout(title=r"$\text{Cost-to-go comparison over set }\;X_w$")
        fig.add_trace(
            go.Scatter(
                x=x_lin,
                y=np.vectorize(evaluate_0)(x_lin) - offset,
                mode="lines",
                name=r"$J_v(x)$",
                line=dict(color="blue"),
            )
        )
