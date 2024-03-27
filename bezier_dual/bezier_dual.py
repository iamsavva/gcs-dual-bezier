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
from pydrake.math import ge, eq,le # pylint: disable=import-error, no-name-in-module, unused-import

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
)  # pylint: disable=import-error, no-name-in-module, unused-import

from gcs_util import get_edge_name, make_quadratic_cost_function_matrices
from polynomial_dual_gcs_utils import define_quadratic_polynomial, define_sos_constraint_over_polyhedron

delta = 0.01
QUADRATIC_COST = lambda x,y: np.sum([(x[i]-y[i])**2 for i in range(len(x)) ])
QUADRATIC_COST_AUGMENTED = lambda x,y: np.sum([(x[i]-y[i])**2 for i in range(len(x)) ]) + delta * ( np.sum([x[i]**2+y[i]**2 for i in range(len(x)) ]) )

class DualVertex:
    def __init__(
        self,
        name: str,
        prog: MathematicalProgram,
        convex_set: ConvexSet,
        options: ProgramOptions,
        specific_J_matrix: npt.NDArray = None,
        vertex_is_start: bool = False,
        vertex_is_target: bool = False
    ):
        self.name = name
        self.potential_poly_deg = 2 # type: int
        self.options = options
        self.vertex_is_start = vertex_is_start
        self.vertex_is_target = vertex_is_target
        self.potential = Expression(0)
        self.J_matrix = np.zeros((1,1))

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
        self.define_potential(prog, specific_J_matrix)

        self.edges_in = [] # type: T.List[str]
        self.edges_out = [] # type: T.List[str]
        
        hpoly = self.get_hpoly()
        cheb_success, _, r = ChebyshevCenter(hpoly)
        assert cheb_success and (r > 1e-5), "vertex set is low dimensional"
    
    def get_hpoly(self):
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
        prog.AddLinearConstraint(self.total_flow_in_violation >= 0)
        prog.AddLinearCost(self.total_flow_in_violation * self.options.max_flow_through_edge)

    def define_set_inequalities(self):
        """
        Assuming that sets are polyhedrons
        """
        # TODO: drop the polyhedral assumption
        # in principle, expectations should be taken differently
        hpoly = self.get_hpoly()

        # inequalities of the form b[i] - a.T x = g_i(x) >= 0
        A, b = hpoly.A(), hpoly.b()
        self.B = np.hstack( (b.reshape((len(b),1)), -A) )

    def define_potential(
        self,
        prog: MathematicalProgram,
        specific_J_matrix: npt.NDArray = None
    ):
        self.J_matrix, self.potential = define_quadratic_polynomial(prog, self.x, self.options.pot_type, specific_J_matrix)
        
        # define G -- the bezier curve continuity vector
        if self.vertex_is_target or self.vertex_is_start:
            self.G_matrix = np.zeros((self.state_dim+1, self.state_dim+1))
            self.eval_G = lambda x: Expression(0)
        else:
            self.G_matrix = prog.NewSymmetricContinuousVariables( self.state_dim+1 )
            def eval_G(x):
                x_and_1 = np.hstack(([1], x))
                return x_and_1.dot(self.G_matrix).dot(x_and_1)
            self.eval_G = eval_G
            # self.eval_G = lambda x: Expression(0)


    def evaluate_partial_potential_at_point(self, x: npt.NDArray):
        """
        Use this to evaluate polynomial at a point.
        Needed when parameters are still optimizaiton variables
        """
        assert len(x) == self.state_dim
        assert self.convex_set.PointInSet(x, 1e-5)  # evaluate only on set
        return self.potential.EvaluatePartial(
            {self.x[i]: x[i] for i in range(self.state_dim)}
        )

    def cost_at_point(self, x: npt.NDArray, solution: MathematicalProgramResult = None):

        assert len(x) == self.state_dim
        assert self.convex_set.PointInSet(x, 1e-5)
        if solution is None:
            return self.evaluate_partial_potential_at_point(x)
        else:
            potential = solution.GetSolution(self.potential)
            return potential.Evaluate({self.x[i]: x[i] for i in range(self.state_dim)})

    def cost_of_uniform_integral_over_box(
        self, lb, ub, solution: MathematicalProgramResult = None
    ):
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
                poly = (integral_of_poly.EvaluatePartial({x_val: x_max}) - integral_of_poly.EvaluatePartial({x_val: x_min})) / (x_max-x_min)
            expectation += coef * poly.ToExpression()

        if solution is None:
            return expectation
        else:
            return solution.GetSolution(expectation)

    def cost_of_small_uniform_box_around_point(
        self, point: npt.NDArray, solution: MathematicalProgramResult = None, eps=0.001
    ):
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
        bidirectional_edge_violation = Expression(0)
    ):
        self.name = name
        self.left = v_left
        self.right = v_right

        self.cost_function = cost_function
        self.options = options

        self.B_intersection = self.intersect_left_right_sets()
        self.bidirectional_edge_violation = bidirectional_edge_violation

        self.define_edge_polynomials_and_sos_constraints(prog)

    def intersect_left_right_sets(self):
        left_hpoly = self.left.get_hpoly()
        right_hpoly = self.right.get_hpoly()
        intersection_hpoly = left_hpoly.Intersection(right_hpoly)
        intersection_hpoly = intersection_hpoly.ReduceInequalities()

        cheb_success, _, r = ChebyshevCenter(intersection_hpoly)
        assert cheb_success and (r > 1e-5), "intersection of vertex sets connected by edge is low dimensional"
        A, b = intersection_hpoly.A(), intersection_hpoly.b()
        B = np.hstack( (b.reshape((len(b),1)), -A) )
        return B
        

    def define_edge_polynomials_and_sos_constraints(self, prog: MathematicalProgram):
        """
        define edge appropriate SOS constraints
        """

        self.x_vectors = [] 
        self.J_matrices = []
        self.potentials = []
        
        # -----------------------------------------------
        # -----------------------------------------------
        # define n-2 intermediate potentials (at least 1 intermediary point in the set)

        for k in range(self.options.num_control_points-2):
            x = prog.NewIndeterminates(self.left.state_dim)
            J_matrix, potential = define_quadratic_polynomial(prog, x, self.options.pot_type)
            self.x_vectors.append(x)
            self.J_matrices.append(J_matrix)
            self.potentials.append(potential)


        # -----------------------------------------------
        # -----------------------------------------------
        # make n+1 SOS constraints


        # -----------------------------------------------
        # J_{v} to J_{vw,1}
        x_left, x_right = self.left.x, self.x_vectors[0]
        B_left, B_right = self.left.B, self.left.B
        edge_cost = self.cost_function(x_left, x_right)
        G_of_v = self.left.eval_G(x_right - x_left)
        left_potential = self.left.potential
        right_potential = self.potentials[0]

        # expr = edge_cost + right_potential - left_potential - G_of_v+ self.bidirectional_edge_violation/(self.options.num_control_points-1)
        expr = edge_cost + right_potential - left_potential - G_of_v
        define_sos_constraint_over_polyhedron(prog, x_left, x_right, expr, B_left, B_right)

        # -------------------------------------------------
        # J_{vw, k} to J_{vw,k+1}
        for k in range(self.options.num_control_points-3):
            x_left, x_right = self.x_vectors[k], self.x_vectors[k+1]
            B_left, B_right = self.left.B, self.left.B
            edge_cost = self.cost_function(x_left, x_right)
            left_potential = self.potentials[k]
            right_potential = self.potentials[k+1]

            # expr = edge_cost + right_potential - left_potential+ self.bidirectional_edge_violation/(self.options.num_control_points-1)
            expr = edge_cost + right_potential - left_potential
            define_sos_constraint_over_polyhedron(prog, x_left, x_right, expr, B_left, B_right)

        # -------------------------------------------------
        # J_{vw, n} to J_{w}
        n = self.options.num_control_points-3
        x_left, x_right = self.x_vectors[n], self.right.x
        B_left, B_right = self.left.B, self.B_intersection
        edge_cost = self.cost_function(x_left, x_right)
        G_of_v = self.left.eval_G(x_right - x_left)
        left_potential = self.potentials[n]
        right_potential = self.right.potential

        # NOTE: adding bidriectional edge violation just to the last constraint
        # expr = edge_cost + right_potential + G_of_v - left_potential + self.bidirectional_edge_violation/(self.options.num_control_points-1)
        expr = edge_cost + right_potential + G_of_v - left_potential + self.bidirectional_edge_violation + self.right.total_flow_in_violation
        define_sos_constraint_over_polyhedron(prog, x_left, x_right, expr, B_left, B_right)




class PolynomialDualGCS:
    def __init__(self, options:ProgramOptions) -> None:
        # variables creates for policy synthesis
        self.vertices = dict()  # type: T.Dict[str, DualVertex]
        self.edges = dict()  # type: T.Dict[str, DualEdge]
        self.prog = MathematicalProgram()  # type: MathematicalProgram
        self.value_function_solution = None # type: MathematicalProgramResult

        self.options = options

    def AddVertex(
        self,
        name: str,
        convex_set: ConvexSet,
        options = None,
        vertex_is_start:bool = False
    ):
        """
        Options will default to graph initialized options if not specified
        """
        assert name not in self.vertices
        if options is None:
            options = self.options
        # add vertex to policy graph
        v = DualVertex(name, self.prog, convex_set, options=options, vertex_is_start=vertex_is_start)
        self.vertices[name] = v
        return v

    def MaxCostOverABox(self, vertex: DualVertex, lb: npt.NDArray, ub: npt.NDArray):
        cost = -vertex.cost_of_uniform_integral_over_box(lb, ub)
        self.prog.AddLinearCost(cost)

    def MaxCostAtAPoint(self, vertex: DualVertex, point, scaling = 1):
        cost = -vertex.cost_at_point(point)
        self.prog.AddLinearCost(cost*scaling)

    def MaxCostAtSmallIntegralAroundPoint(self, vertex: DualVertex, point, scaling = 1, eps=0.001):
        cost = -vertex.cost_of_small_uniform_box_around_point(point, eps=eps)
        self.prog.AddLinearCost(cost*scaling)

    def AddTargetVertexWithQuadraticTerminalCost(
        self,
        name: str,
        convex_set: HPolyhedron,
        Q_terminal:npt.NDArray,
        x_terminal:npt.NDArray,
        options: ProgramOptions = None
    ):
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
        J_12 = - Q_terminal.dot(x_terminal).reshape((1, len(x_terminal)))
        J_21 = J_12.T
        J_22 = Q_terminal

        specific_J_matrix = np.vstack( (np.hstack( (J_11, J_12) ), np.hstack( (J_21, J_22) ) ) )


        v = DualVertex(
            name,
            self.prog,
            convex_set,
            options = options,
            specific_J_matrix=specific_J_matrix,
            vertex_is_target=True
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
    ):
        """
        adding two edges
        """
        if options is None:
            options = self.options
        bidirectional_edge_violation = self.prog.NewContinuousVariables(1)[0]
        self.prog.AddLinearConstraint(bidirectional_edge_violation >= 0) # TODO: shouldn't be necessary?
        self.prog.AddLinearCost(bidirectional_edge_violation * self.options.max_flow_through_edge)
        self.AddEdge(v_left, v_right, cost_function, options, bidirectional_edge_violation)
        self.AddEdge(v_right, v_left, cost_function, options, bidirectional_edge_violation)

        

    def AddEdge(
        self,
        v_left: DualVertex,
        v_right: DualVertex,
        cost_function: T.Callable,
        options: ProgramOptions = None,
        bidirectional_edge_violation = Expression(0),
    ):
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
            bidirectional_edge_violation = bidirectional_edge_violation,
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
            self.options.MSK_DPAR_INTPNT_CO_TOL_REL_GAP
        )
        solver_options.SetOption(
            MosekSolver.id(),
            "MSK_DPAR_INTPNT_CO_TOL_PFEAS",
            self.options.MSK_DPAR_INTPNT_CO_TOL_PFEAS
        )
        solver_options.SetOption(
            MosekSolver.id(),
            "MSK_DPAR_INTPNT_CO_TOL_DFEAS",
            self.options.MSK_DPAR_INTPNT_CO_TOL_DFEAS
        )

        if self.options.use_robust_mosek_parameters:
            solver_options.SetOption(MosekSolver.id(), "MSK_DPAR_INTPNT_CO_TOL_REL_GAP", 1e-3)
            solver_options.SetOption(MosekSolver.id(), "MSK_IPAR_INTPNT_SOLVE_FORM", 1)

        # solve the program
        self.value_function_solution = mosek_solver.Solve(self.prog, solver_options=solver_options)
        timer.dt("Solve")
        diditwork(self.value_function_solution)

        return self.value_function_solution

    def get_policy_cost_for_region_plot(self, vertex_name:str):
        vertex = self.vertices[vertex_name]
        assert vertex.set_type == Hyperrectangle, "vertex not a Hyperrectangle, can't make a plot"
        assert vertex.state_dim == 2, "can't plot generate plot for non-2d vertex"

        box = vertex.convex_set
        eps = 0
        N= 100

        X = np.linspace(box.lb()[0]-eps, box.ub()[0]+eps, N)
        Y = np.linspace(box.lb()[1]-eps, box.ub()[1]+eps, N)
        X, Y = np.meshgrid(X, Y)

        def eval_func(x,y):
            expression = vertex.cost_at_point( np.array([x,y]), self.value_function_solution )
            return expression

        evaluator = np.vectorize( eval_func )

        return X, Y, evaluator(X, Y)
    

    def make_plots(self, fig=None, cmax=30, offset=0):
        if fig is None:
            fig = go.Figure()

        for v_name in self.vertices.keys():
            X, Y, Z = self.get_policy_cost_for_region_plot(v_name)
            # Create filled 3D contours
            fig.add_trace(go.Surface(x=X, y=Y, z=Z-offset, surfacecolor=Z-offset, name = v_name, cmin=0,cmax=cmax))

        # Update layout
        fig.update_layout(scene=dict(
                            xaxis_title='X',
                            yaxis_title='Y',
                            zaxis_title='Z'))

        fig.update_layout(height=800, width=800, title_text="Value functions ")

        return fig
    

    def make_1d_plot_potential(self, fig:go.Figure, v: DualVertex):
        assert type(v.convex_set) == Hyperrectangle
        assert len(v.convex_set.lb()) == 1
        x_lin = np.linspace(v.convex_set.lb()[0], v.convex_set.ub()[0], 100, endpoint=True)

    
        def evaluate_0(y):
            potential = self.value_function_solution.GetSolution(v.potential)
            f_potential = lambda x: potential.Substitute({v.x[i]: x[i] for i in range(v.state_dim)})
            return f_potential(np.array([y])).Evaluate()

        offset = self.options.zero_offset + self.options.policy_gcs_edge_cost_offset
        # fig.update_layout(title=r"$\text{Cost-to-go comparison over set }\;X_w$")
        fig.add_trace(go.Scatter(x=x_lin, y=np.vectorize(evaluate_0)(x_lin)-offset, mode='lines', name=r"$J_v(x)$", line=dict(color="blue") ))
