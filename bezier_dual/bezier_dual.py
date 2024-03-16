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

QUADRATIC_COST = lambda x,y: np.sum([(x[i]-y[i])**2 for i in range(len(x)) ])

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
                poly = integral_of_poly.EvaluatePartial(
                    {x_val: x_max}
                ) - integral_of_poly.EvaluatePartial({x_val: x_min})
            expectation += coef * poly.ToExpression()

        if solution is None:
            return expectation
        else:
            return solution.GetSolution(expectation)

    def cost_of_small_uniform_box_around_point(
        self, point: npt.NDArray, solution: MathematicalProgramResult = None, eps=0.001
    ):
        assert len(point) == self.state_dim
        return self.cost_of_uniform_integral_over_box(
            point - eps, point + eps, solution
        )


class DualEdge:
    def __init__(
        self,
        name: str,
        v_left: DualVertex,
        v_right: DualVertex,
        prog: MathematicalProgram,
        cost_function: T.Callable,
        options: ProgramOptions
    ):
        self.name = name
        self.left = v_left
        self.right = v_right

        self.cost_function = cost_function
        self.options = options

        self.B_intersection = self.intersect_left_right_sets()

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

        expr = edge_cost + right_potential - left_potential - G_of_v
        # print(edge_cost.Expand())
        # print(right_potential.Expand())
        # print(left_potential.Expand())
        # print(G_of_v.Expand())
        define_sos_constraint_over_polyhedron(prog, x_left, x_right, expr, B_left, B_right)

        # -------------------------------------------------
        # J_{vw, k} to J_{vw,k+1}
        for k in range(self.options.num_control_points-3):
            x_left, x_right = self.x_vectors[k], self.x_vectors[k+1]
            B_left, B_right = self.left.B, self.left.B
            edge_cost = self.cost_function(x_left, x_right)
            left_potential = self.potentials[k]
            right_potential = self.potentials[k+1]

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

        expr = edge_cost + right_potential + G_of_v - left_potential
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

    def AddTargetVertex(
        self,
        name: str,
        convex_set: HPolyhedron,
        specific_J_matrix: npt.NDArray,
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

    def AddEdge(
        self,
        v_left: DualVertex,
        v_right: DualVertex,
        cost_function: T.Callable,
        options: ProgramOptions = None
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



    def make_1d_plot_edge_1_step(self, fig:go.Figure, edge: DualEdge, x_point:npt.NDArray, plot_just_1=False, force_dont_add_noise=False):
        v = edge.right
        assert type(v.convex_set) == Hyperrectangle
        assert len(v.convex_set.lb()) == 1
        x_lin = np.linspace(v.convex_set.lb()[0], v.convex_set.ub()[0], 100, endpoint=True)

    
        def evaluate_0(y):
            func = self.get_policy_edge_cost(edge, add_potential=True, force_dont_add_s_proecdure=True,force_dont_add_noise=force_dont_add_noise)
            return func(x_point, np.array([y])).Evaluate()
        
        def evaluate_1(y):
            func = self.get_policy_edge_cost(edge, add_potential=True, force_dont_add_s_proecdure=False, force_dont_add_noise=force_dont_add_noise)
            return func(x_point, np.array([y])).Evaluate()
        
        def plot_min(x, y, fig, color, name):
            # Find the index of the minimum y value
            min_y_index = np.argmin(y)
            # Get the corresponding x value
            min_x_value = x[min_y_index]
            min_y_value = y[min_y_index]
            fig.add_trace(go.Scatter(x=[min_x_value], y=[min_y_value], mode='markers', line=dict(color=color), showlegend=False, name = name))

        offset = self.options.zero_offset + self.options.policy_gcs_edge_cost_offset
        fig.update_layout(title=r"$\text{Cost-to-go comparison over set }\;X_w$")
        fig.add_trace(go.Scatter(x=x_lin, y=np.vectorize(evaluate_0)(x_lin)-offset, mode='lines', name=r"$c(x,x')+J_w(x')$", line=dict(color="red") ))
        plot_min(x_lin, np.vectorize(evaluate_0)(x_lin)-offset, fig, "red", r"$\min c(x,x')+J_w(x')$")

        if not plot_just_1:
            fig.add_trace(go.Scatter(x=x_lin, y=np.vectorize(evaluate_1)(x_lin)-offset, mode='lines', name = r"$\text{noise}=%d,\; c(x,x')+J_w(x')-\text{LM}$" %self.options.noise_magnitude, line=dict(color="blue") ))
            plot_min(x_lin, np.vectorize(evaluate_1)(x_lin)-offset, fig, "blue", r"$\text{noise}=%d,\; \min c(x,x')+J_w(x')-\text{LM}$" %self.options.noise_magnitude)



    def make_1d_plot_edge_1_step_just_gamma(self, fig:go.Figure, edge: DualEdge, x_point:npt.NDArray, offset=0, gamma=0, color="blue"):
        v = edge.right
        assert type(v.convex_set) == Hyperrectangle
        assert len(v.convex_set.lb()) == 1
        x_lin = np.linspace(v.convex_set.lb()[0], v.convex_set.ub()[0], 100, endpoint=True)

        potential = self.value_function_solution.GetSolution(v.potential)
        f_potential = lambda x: potential.Substitute({v.x[i]: x[i] for i in range(v.state_dim)})
        
        q = self.value_function_solution.GetSolution(edge.lambda_1_right)
        Q = self.value_function_solution.GetSolution(edge.lambda_2_right)
        B = v.B

        def evaluate_0(y):
            return (QUADRATIC_COST(x_point,np.array([y]) ) + f_potential(np.array([y]))).Evaluate()
            
        def evaluate_1(y):
            Y = np.outer( np.array([1,y]), np.array([1,y]) )
            return (QUADRATIC_COST(x_point,np.array([y])) + f_potential(np.array([y])) - np.sum( B.dot(Y).dot(B.T) * Q)).Evaluate()
        
        def evaluate_2(y):
            Y = np.outer( np.array([1,y]), np.array([1,y]) )
            return (QUADRATIC_COST(x_point,np.array([y])) + f_potential(np.array([y])) - np.sum( B.dot(Y).dot(B.T) * Q) - B.dot(np.array([1,y])).dot(q)).Evaluate()
        
        def evaluate_3(y):
            Y = np.outer( np.array([1,y]), np.array([1,y]) )
            return (QUADRATIC_COST(x_point,np.array([y])) - np.sum( B.dot(Y).dot(B.T) * Q) - B.dot(np.array([1,y])).dot(q))
        
        def evaluate_4(y):
            Y = np.outer( np.array([1,y]), np.array([1,y]) )
            return ( - np.sum( B.dot(Y).dot(B.T) * Q) - B.dot(np.array([1,y])).dot(q))
        
        def plot_min(x, y, fig, color,name):
            # Find the index of the minimum y value
            min_y_index = np.argmin(y)
            # Get the corresponding x value
            min_x_value = x[min_y_index]
            min_y_value = y[min_y_index]
            fig.add_trace(go.Scatter(x=[min_x_value], y=[min_y_value], mode='markers', line=dict(color=color), showlegend=True, name=name))

        # fig.update_layout(title=r"$\text{Cost-to-go comparison over set }\;X_w$")
        # fig.add_trace(go.Scatter(x=x_lin, y=np.vectorize(evaluate_0)(x_lin)-offset, mode='lines', name=r"$c(x,x')+J_w(x')$", line=dict(color="red") ))
        fig.add_trace(go.Scatter(x=x_lin, y=np.vectorize(evaluate_1)(x_lin)-offset, mode='lines', name = r"$\gamma=%d,\; c(x,x')+J_w(x')-\text{LM}$" %gamma, line=dict(color=color) ))
        plot_min(x_lin, np.vectorize(evaluate_1)(x_lin)-offset, fig, color=color, name=r"$\gamma=%d,\; \min c(x,x')+J_w(x')-\text{LM}$" %gamma)
        # fig.add_trace(go.Scatter(x=x_lin, y=np.vectorize(evaluate_2)(x_lin)-offset, mode='lines', name="c+J-BB-B", line=dict(color="green") ))
        # fig.add_trace(go.Scatter(x=x_lin, y=np.vectorize(evaluate_3)(x_lin)-offset, mode='lines', name="c-BB-B", line=dict(color="purple") ))
        # fig.add_trace(go.Scatter(x=x_lin, y=np.vectorize(evaluate_4)(x_lin)-offset, mode='lines', name="-BB-B", line=dict(color="magenta") ))

        # plot_min(x_lin, np.vectorize(evaluate_0)(x_lin)-offset, fig, "red")
        
    def make_2d_plot_edge_1_step(self, fig:go.Figure, edge: DualEdge, x_point:npt.NDArray, offset=0):
        v = edge.right
        assert type(v.convex_set) == Hyperrectangle
        assert len(v.convex_set.lb()) == 2
        x_lin = np.linspace(v.convex_set.lb()[0], v.convex_set.ub()[0], 100, endpoint=True)
        y_lin = np.linspace(v.convex_set.lb()[1], v.convex_set.ub()[1], 100, endpoint=True)
        X, Y = np.meshgrid(x_lin, y_lin)

        potential = self.value_function_solution.GetSolution(v.potential)
        f_potential = lambda x: potential.Substitute({v.x[i]: x[i] for i in range(v.state_dim)})
        
        q = self.value_function_solution.GetSolution(edge.lambda_1_right)
        Q = self.value_function_solution.GetSolution(edge.lambda_2_right)
        B = v.B
        

        def evaluate_0(x,y):
            return (QUADRATIC_COST(x_point, np.array([x,y]) ) + f_potential(np.array([x,y]))).Evaluate()
            
        def evaluate_1(x,y):
            Y = np.outer( np.array([1,x, y]), np.array([1,x, y]) )
            return (QUADRATIC_COST(x_point,np.array([x, y])) + f_potential(np.array([x, y])) - np.sum( B.dot(Y).dot(B.T) * Q)).Evaluate()
        
        def evaluate_2(x, y):
            Y = np.outer( np.array([1,x,y]), np.array([1,x,y]) )
            return (QUADRATIC_COST(x_point,np.array([x, y])) + f_potential(np.array([x, y])) - np.sum( B.dot(Y).dot(B.T) * Q) - B.dot(np.array([1, x, y])).dot(q)).Evaluate()
        
        def evaluate_3(x,y):
            Y = np.outer( np.array([1,x,y]), np.array([1,x,y]) )
            return (QUADRATIC_COST(x_point,np.array([x,y])) - np.sum( B.dot(Y).dot(B.T) * Q) - B.dot(np.array([1,x,y])).dot(q))
        
        def evaluate_4(x,y):
            Y = np.outer( np.array([1,x,y]), np.array([1,x,y]) )
            return ( - np.sum( B.dot(Y).dot(B.T) * Q) - B.dot(np.array([1,x,y])).dot(q))
        
        def plot_min(x, y, z, fig, color):
            min_z_index = np.unravel_index(np.argmin(z, axis=None), z.shape)

            # Get the corresponding x and y values
            min_x_value = x[min_z_index]
            min_y_value = y[min_z_index]
            min_z_value = z[min_z_index]

            fig.add_trace(go.Scatter3d(x=[min_x_value], y=[min_y_value], z=[min_z_value],
                           mode='markers', marker=dict(size=5, color=color), showlegend=False))
        
        # Create filled 3D contours
        Z0 = np.vectorize(evaluate_0)(X, Y)-offset
        Z1 = np.vectorize(evaluate_1)(X, Y)-offset
        Z2 = np.vectorize(evaluate_2)(X, Y)-offset
        Z3 = np.vectorize(evaluate_3)(X, Y)-offset
        Z4 = np.vectorize(evaluate_4)(X, Y)-offset
        fig.add_trace(go.Surface(x=X, y=Y, z=Z0, colorscale=[[0, "red"], [1, "red"]], name = "c+J"))
        fig.add_trace(go.Surface(x=X, y=Y, z=Z1, colorscale=[[0, "blue"], [1, "blue"]], name = "c+J-BB"))
        fig.add_trace(go.Surface(x=X, y=Y, z=Z2, colorscale=[[0, "green"], [1, "green"]], name = "c+J-BB-B"))
        fig.add_trace(go.Surface(x=X, y=Y, z=Z3, colorscale=[[0, "purple"], [1, "purple"]], name = "c-BB-B"))
        fig.add_trace(go.Surface(x=X, y=Y, z=Z4, colorscale=[[0, "magenta"], [1, "magenta"]], name = "-BB-B"))

        plot_min(X,Y,Z0, fig, "red")
        plot_min(X,Y,Z1, fig, "blue")
        plot_min(X,Y,Z2, fig, "green")
        plot_min(X,Y,Z3, fig, "purple")
        plot_min(X,Y,Z4, fig, "magenta")

    def get_policy_edge_cost(self, edge:DualEdge, add_potential:bool, force_dont_add_s_proecdure:bool=False, force_dont_add_noise=False):
        def cost_function(x,y):
            cost = QUADRATIC_COST(x,y)

            s_procedure_terms = Expression(0)
            if self.options.policy_subtract_full_s_procedure or self.options.policy_subtract_right_vertex_s_procedure:
                # subtract right-vertex relaxation terms
                Qr = self.value_function_solution.GetSolution(edge.lambda_2_right)
                qr = self.value_function_solution.GetSolution(edge.lambda_1_right)
                Br = edge.right.B
                Y = np.outer( np.hstack(([1], y)), np.hstack(([1], y)) )
                s_procedure_terms -= np.sum( Br.dot(Y).dot(Br.T) * Qr) 
                s_procedure_terms -= Br.dot( np.hstack(([1], y)) ).dot(qr)

                if edge.force_deterministic:
                    pass
                elif self.options.solve_ot_relaxed_stochastic_transitions or self.options.solve_ot_deterministic_transitions_inflated:
                    s_procedure_terms -= self.value_function_solution.GetSolution(edge.delta)
                elif self.options.solve_ot_stochastic_transitions:
                    Sigma = edge.noise_mat
                    s_procedure_terms -= np.sum( Br.dot(Sigma).dot(Br.T) * Qr) 
                elif self.options.solve_robustified_set_membership:
                    Sigma = edge.noise_mat
                    if edge.left.name != "s":
                        Ql = self.value_function_solution.GetSolution(edge.lambda_2_left)
                        Bl = edge.left.B
                        s_procedure_terms -= np.sum( Bl.dot(Sigma).dot(Bl.T) * Ql) 
                    s_procedure_terms -= np.sum( Br.dot(Sigma).dot(Br.T) * Qr) 

            if self.options.policy_subtract_full_s_procedure:
                QLR = self.value_function_solution.GetSolution(edge.lambda_2_left_right)
                QL = self.value_function_solution.GetSolution(edge.lambda_2_left)
                qL = self.value_function_solution.GetSolution(edge.lambda_1_left)
                BL = edge.right.B
                BR = edge.left.B
                YL = np.outer( np.hstack(([1], x)), np.hstack(([1], x)) )
                YLR = np.outer( np.hstack(([1], x)), np.hstack(([1], y)) )

                s_procedure_terms -= np.sum( BL.dot(YL).dot(BL.T) * QL)
                s_procedure_terms -= BL.dot( np.hstack(([1], x)) ).dot(qL)
                s_procedure_terms -= np.sum( BL.dot(YLR).dot(BR.T) * QLR)

            if force_dont_add_s_proecdure:
                s_procedure_terms = Expression(0)

            cost += s_procedure_terms

            if add_potential:
                potential = self.value_function_solution.GetSolution(edge.right.potential)
                f_potential = lambda x: potential.Substitute({edge.right.x[i]: x[i] for i in range(edge.right.state_dim)})
                cost += f_potential(y)
                
                if self.options.solve_ot_stochastic_transitions:
                    Jr = self.value_function_solution.GetSolution(edge.right.J_matrix)
                    Sigma = edge.noise_mat
                    if not force_dont_add_noise:
                        cost += np.sum( Jr * Sigma) 
            # this is done because GCS doesn't like negative costs
            cost += self.options.policy_gcs_edge_cost_offset
            return cost
        return cost_function

    def solve_restriction(self, vertices: T.List[DualVertex], x_0:npt.NDArray) -> T.Tuple[float, npt.NDArray, DualVertex, npt.NDArray]:
        prog = MathematicalProgram()
        assert np.all(vertices[0].B.dot(np.hstack(([1], x_0))) >= 0-1e-3, )
        x_n = x_0
        x_traj = []
        for v_index in range(1, len(vertices)):
            v = vertices[v_index]
            edge = self.edges[get_edge_name(vertices[v_index-1].name, v.name)]

            cost_function = self.get_policy_edge_cost(edge, (v_index == len(vertices)-1) )
            
            x_n1 = prog.NewContinuousVariables(v.state_dim)
            x_traj.append(x_n1)
            prog.AddLinearConstraint( ge(v.B.dot(np.hstack(([1], x_n1))), 0 ) )
            prog.AddCost( cost_function(x_n, x_n1) )
            x_n = x_n1

        solution = Solve(prog)
        # INFO(solution.get_solver_id().name())
        assert solution.is_success()
        x_traj_solution = np.array( [solution.GetSolution(xn) for xn in x_traj] )
        cost = solution.get_optimal_cost()
        x_next = x_traj_solution[0]
        return cost, x_next, vertices[1], x_traj_solution

    def solve_m_step_policy(self, layers:T.List[T.List[DualVertex]], m:int, start_vertex:DualVertex, x_0:npt.NDArray, layer_index:int):
        first_index = layer_index+1
        last_index = min(len(layers), layer_index+m+1)
        relevant_layers = layers[first_index:last_index]
        upper = [len(layer) for layer in relevant_layers]

        def get_vertex_sequence(index_sequence):
            return [start_vertex] + [relevant_layers[i][int(index_sequence[i])] for i in range(len(index_sequence)) ]
        
        def get_next_index_sequence(vec:npt.NDArray, upper:npt.NDArray):
            for i in range(len(vec)-1, -1, -1):
                vec[i] = vec[i]+ 1 
                if vec[i] < upper[i]-1e-5:
                    break
                vec[i] = 0
            if np.allclose(vec, np.zeros(len(vec))):
                return False, vec
            return True, vec
        
        true_cost = start_vertex.cost_at_point(x_0, self.value_function_solution)
        # print(true_cost)
        best_cost, best_action, best_vertex = np.inf, None, None
        index_sequence = np.zeros(len(relevant_layers))
        go_on = True
        while go_on:
            vertex_sequence = get_vertex_sequence(index_sequence)
            # print(vertex_sequence)
            new_cost, new_action, new_vertex, _ = self.solve_restriction(vertex_sequence, x_0)
            if np.abs(new_cost-true_cost) < np.abs(best_cost-true_cost): # TODO: make this into an option
                best_cost, best_action, best_vertex = new_cost, new_action, new_vertex
            go_on, index_sequence = get_next_index_sequence(index_sequence, upper)

        # print(best_cost, best_vertex.name, best_action)
        return best_vertex, best_action
