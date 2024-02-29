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
from pydrake.math import ge, eq,le

import plotly.graph_objects as go  # pylint: disable=import-error
from plotly.express.colors import sample_colorscale  # pylint: disable=import-error
import plotly.graph_objs as go
from plotly.subplots import make_subplots

from program_options import FREE_POLY, PSD_POLY

from util import (
    timeit,
    diditwork,
    INFO,
    YAY,
    WARN,
    ERROR,
)  # pylint: disable=import-error, no-name-in-module, unused-import
from gcs_util import get_edge_name, make_quadratic_cost_function_matrices

QUADRATIC_COST = lambda x,y: np.sum([(x[i]-y[i])**2 for i in range(len(x)) ])

class DualVertex:
    def __init__(
        self,
        name: str,
        prog: MathematicalProgram,
        convex_set: ConvexSet,
        specific_potential: T.Callable = None,
        pot_type = PSD_POLY
    ):
        self.name = name
        self.potential_poly_deg = 2
        self.pot_type = pot_type

        # Ax <= b
        self.convex_set = convex_set  # TODO: handle point exactly
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

        self.edges_in = []
        self.edges_out = []

    def add_edge_in(self, name: str):
        assert name not in self.edges_in
        self.edges_in.append(name)

    def add_edge_out(self, name: str):
        assert name not in self.edges_out
        self.edges_out.append(name)

    def define_variables(self, prog: MathematicalProgram):
        self.x = prog.NewIndeterminates(self.state_dim, "x_" + self.name)
        self.vars = Variables(self.x)

    def define_set_inequalities(self):
        # redefine
        if self.set_type == HPolyhedron:
            hpoly = self.convex_set
        if self.set_type == Hyperrectangle:
            hpoly = self.convex_set.MakeHPolyhedron()

        # inequalities of the form b[i] - a.T x = g_i(x) >= 0
        A, b = hpoly.A(), hpoly.b()
        self.B = np.hstack( (b.reshape((len(b),1)), -A) )

    def define_potential(
        self,
        prog: MathematicalProgram,
        specific_potential: T.Callable = None,
    ):
        # specific potential here is a function
        if specific_potential is not None:
            self.potential = Polynomial(specific_potential(self.x))
        else:
            # potential is a free polynomial
            if self.pot_type == FREE_POLY:
                self.potential = prog.NewFreePolynomial( self.vars, 2)
            elif self.pot_type == PSD_POLY:
                # potential is PSD polynomial
                self.potential, _ = prog.NewSosPolynomial( self.vars, 2 ) # i.e.: convex
            else:
                raise NotImplementedError("potential type not supported")

    # @staticmethod
    # def get_product_constraints(constraints):
    #     product_constraints = []
    #     for i, con_i in enumerate(constraints):
    #         for j in range(i + 1, len(constraints)):
    #             product_constraints.append(con_i * constraints[j])
    #     return product_constraints

    def evaluate_partial_potential_at_point(self, x: npt.NDArray):
        # needed when polynomial parameters are still optimizaiton variables
        assert len(x) == self.state_dim
        assert self.convex_set.PointInSet(x, 1e-5)  # evaluate only on set
        return self.potential.EvaluatePartial(
            {self.x[i]: x[i] for i in range(self.state_dim)}
        )

    def cost_at_point(self, x: npt.NDArray, solution: MathematicalProgramResult = None):
        assert len(x) == self.state_dim
        assert self.convex_set.PointInSet(x, 1e-5)
        if solution is None:
            return self.evaluate_partial_potential_at_point(x).ToExpression()
        else:
            potential = solution.GetSolution(self.potential)
            return potential.Evaluate({self.x[i]: x[i] for i in range(self.state_dim)})

    def cost_of_uniform_integral_over_box(
        self, lb, ub, solution: MathematicalProgramResult = None
    ):
        assert len(lb) == len(ub) == self.state_dim

        # compute by integrating each monomial term
        monomial_to_coef_map = self.potential.monomial_to_coefficient_map()
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
        add_noise=False,
        gamma = 1,
    ):
        self.name = name
        self.left = v_left
        self.right = v_right
        self.max_constraint_degree = 2

        self.cost_function = cost_function
        self.add_noise = add_noise
        self.gamma = gamma

        self.define_sos_constaint(prog)

    def define_sos_constaint(self, prog: MathematicalProgram):
        # -------------------------------------------------
        # get a bunch of variables
        x, y = self.left.x, self.right.x

        xy_vars = Variables(np.hstack((x, y)))

        # -------------------------------------------------
        # define cost
        edge_cost = self.cost_function(x, y)


        # -------------------------------------------------
        # set membership through the S procedure

        s_procedure = Expression(0)

        # x \in X
        # y \in Y
        x_and_1 = np.hstack(([1], x))
        y_and_1 = np.hstack(([1], y))

        Bl = self.left.B
        Br = self.right.B

        # deg 0 
        lambda_0 = prog.NewContinuousVariables(1)[0]
        prog.AddLinearConstraint(lambda_0 >= 0)
        s_procedure += lambda_0

        # deg 1
        deg_1_cons_l = Bl.dot(x_and_1)
        deg_1_cons_r = Br.dot(y_and_1)
        lambda_1_left = prog.NewContinuousVariables(len(deg_1_cons_l))
        lambda_1_right = prog.NewContinuousVariables(len(deg_1_cons_r))
        prog.AddLinearConstraint(ge(lambda_1_left, 0))
        prog.AddLinearConstraint(ge(lambda_1_right, 0))
        s_procedure += deg_1_cons_l.dot(lambda_1_left) + deg_1_cons_r.dot(lambda_1_right)

        # deg 2
        lambda_2_left = prog.NewSymmetricContinuousVariables(len(deg_1_cons_l))
        lambda_2_right = prog.NewSymmetricContinuousVariables(len(deg_1_cons_r))
        lambda_2_left_right = prog.NewContinuousVariables(len(deg_1_cons_r), len(deg_1_cons_r) )

        self.lambda_1_left = lambda_1_left
        self.lambda_1_right = lambda_1_right
        self.lambda_2_left = lambda_2_left
        self.lambda_2_right = lambda_2_right

        prog.AddLinearConstraint(ge(lambda_2_left, 0))
        prog.AddLinearConstraint(ge(lambda_2_right, 0))
        prog.AddLinearConstraint(ge(lambda_2_left_right, 0))


        s_procedure += np.sum( ( Bl.dot( np.outer(x_and_1, x_and_1) ).dot(Bl.T)) * lambda_2_left )

        s_procedure += np.sum( ( Br.dot( np.outer(y_and_1, y_and_1) ).dot(Br.T) ) * lambda_2_right )

        s_procedure += np.sum( ( Bl.dot( np.outer(x_and_1, y_and_1) ).dot(Br.T) ) * lambda_2_left_right )

        # TODO: add inflation cosntaints here
        if self.add_noise:
            eye = np.eye(len(x_and_1))
            eye[0,0] = 0
            prog.AddLinearConstraint( self.gamma + np.sum( (Br.dot(eye).dot(Br.T)) * lambda_2_right ) <= 0)

        # -------------------------------------------------
        # obtain right and left potentials
        right_potential = self.right.potential.ToExpression()
        left_potential = self.left.potential.ToExpression()

        # -------------------------------------------------
        # form the entire expression
        expr = edge_cost + right_potential - left_potential - s_procedure
        prog.AddSosConstraint(expr)


class PolynomialDualGCS:
    def __init__(self, solver_for_spp = None) -> None:
        # variables creates for policy synthesis
        self.vertices = dict()  # type: T.Dict[str, Vertex]
        self.edges = dict()  # type: T.Dict[str, Edge]
        self.prog = MathematicalProgram()  # type: MathematicalProgram

        self.value_function_solution = None

        # variables for GCS ground truth solves
        self.gcs_vertices = dict()  # type: T.Dict[str, GraphOfConvexSets.Vertex]
        self.gcs_edges = dict()  # type: T.Dict[str, GraphOfConvexSets.Edge]
        self.gcs = GraphOfConvexSets()  # type: GraphOfConvexSets
        # i'm adding an arbitary target vertex that terminates any process
        self.gcs_vertices["target"] = self.gcs.AddVertex(Point([0]), "target")
        self.solver_for_spp = solver_for_spp 

    def AddVertex(
        self,
        name: str,
        convex_set: HPolyhedron,
        pot_type=PSD_POLY
    ):
        """
        Options will default to graph initialized options if not specified
        """
        assert name not in self.vertices
        # add vertex to policy graph
        v = DualVertex(name, self.prog, convex_set, pot_type=pot_type)
        self.vertices[name] = v
        # add vertex to GCS graph
        self.gcs_vertices[name] = self.gcs.AddVertex(convex_set, name)
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
        specific_potential: T.Callable,
    ):
        """
        Options will default to graph initialized options if not specified

        Target vertices are vertices with fixed potentials functions.

        HPolyhedron with quadratics, or
        Point and 0 potential.
        """
        assert name not in self.vertices
        v = DualVertex(
            name,
            self.prog,
            convex_set,
            specific_potential=specific_potential,
        )
        # building proper GCS
        self.vertices[name] = v


        # in the GCS graph, add a fake edge to the fake target vetex
        gcs_v = self.gcs.AddVertex(convex_set, name)
        self.gcs_vertices[name] = gcs_v
        self.gcs.AddEdge(gcs_v, self.gcs_vertices["target"])
        return v

    def AddEdge(
        self,
        v_left: DualVertex,
        v_right: DualVertex,
        cost_function: T.Callable,
        add_noise=False,
        gamma = 1,
    ):
        """
        Options will default to graph initialized options if not specified
        """
        edge_name = get_edge_name(v_left.name, v_right.name)
        e = DualEdge(
            edge_name,
            v_left,
            v_right,
            self.prog,
            cost_function,
            add_noise=add_noise,
            gamma=gamma
        )
        self.edges[edge_name] = e
        v_left.add_edge_out(edge_name)
        v_right.add_edge_in(edge_name)

        # building proper GCS
        gcs_edge = self.gcs.AddEdge(
            self.gcs_vertices[v_left.name], self.gcs_vertices[v_right.name], edge_name
        )
        self.gcs_edges[edge_name] = gcs_edge
        gcs_edge.AddCost(cost_function(gcs_edge.xu(), gcs_edge.xv()))
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
            1e-8,
        )
        solver_options.SetOption(
            MosekSolver.id(),
            "MSK_DPAR_INTPNT_CO_TOL_PFEAS",
            1e-8
        )
        solver_options.SetOption(
            MosekSolver.id(),
            "MSK_DPAR_INTPNT_CO_TOL_DFEAS",
            1e-8
        )

        # solve the program
        self.value_function_solution = mosek_solver.Solve(self.prog, solver_options=solver_options)
        timer.dt("Solve")
        diditwork(self.value_function_solution)

        return self.value_function_solution

    def solve_for_true_shortest_path(
        self, vertex_name: str, point: npt.NDArray
    ) -> T.Tuple[float, T.List[str], T.List[npt.NDArray]]:
        """
        Solve for an optimal GCS path from point inside vertex_name vertex.
        Pass the options vector with specifications of convex relaxation / rounding / etc.

        Returns the cost and a mode sequence (vertex name path).
        """
        assert vertex_name in self.vertices
        assert self.vertices[vertex_name].convex_set.PointInSet(
            point, 1e-5
        )  # evaluate only on set

        start_vertex = self.gcs.AddVertex(Point(point), "start")
        target_vertex = self.gcs_vertices["target"]

        # add edges from start point to neighbours
        for edge_name in self.vertices[vertex_name].edges_out:
            cost_function = self.edges[edge_name].cost_function
            right_vertex = self.gcs_vertices[self.edges[edge_name].right.name]
            gcs_edge = self.gcs.AddEdge(
                start_vertex,
                right_vertex,
                get_edge_name(start_vertex.name(), right_vertex.name())
            )
            gcs_edge.AddCost(cost_function(gcs_edge.xu(), gcs_edge.xv()))

        gcs_options = GraphOfConvexSetsOptions()
        # gcs_options.convex_relaxation = False
        gcs_options.convex_relaxation = True
        gcs_options.max_rounded_paths=30
        gcs_options.max_rounding_trials=30        
        gcs_options.solver = MosekSolver()

        # solve
        result = self.gcs.SolveShortestPath(
            start_vertex, target_vertex, gcs_options
        )  # type: MathematicalProgramResult
        assert result.is_success()
        cost = result.get_optimal_cost()

        edge_path = self.gcs.GetSolutionPath(start_vertex, target_vertex, result)
        vertex_name_path = [vertex_name]
        value_path = [point]
        for e in edge_path:
            vertex_name_path.append(e.v().name())
            value_path.append(result.GetSolution(e.v().x()))

        self.gcs.RemoveVertex(start_vertex)
        
        return cost, vertex_name_path, value_path

    def get_policy_cost_for_region_plot(self, vertex_name:str):
        vertex = self.vertices[vertex_name]
        assert vertex.set_type == Hyperrectangle, "vertex not a Hyperrectangle, can't make a plot"

        box = vertex.convex_set
        eps = 0
        N= 100

        X = np.linspace(box.lb()[0]-eps, box.ub()[0]+eps, N)
        Y = np.linspace(box.lb()[1]-eps, box.ub()[1]+eps, N)
        X, Y = np.meshgrid(X, Y)

        # eval_func = lambda x,y: vertex.cost_at_point( np.array([x,y]), self.value_function_solution )
        def eval_func(x,y):
            expression = vertex.cost_at_point( np.array([x,y]), self.value_function_solution )
            return expression

        evaluator = np.vectorize( eval_func )

        return X, Y, evaluator(X, Y)
    

    def make_plots(self, fig=None, cmax=30):
        if fig is None:
            fig = go.Figure()

        for v_name in self.vertices.keys():
            X, Y, Z = self.get_policy_cost_for_region_plot(v_name)
            # Create filled 3D contours
            fig.add_trace(go.Surface(x=X, y=Y, z=Z, surfacecolor=Z, name = v_name, cmin=0,cmax=cmax))

        # Update layout
        fig.update_layout(scene=dict(
                            xaxis_title='X',
                            yaxis_title='Y',
                            zaxis_title='Z'))

        fig.update_layout(height=800, width=800, title_text="Value functions ")

        return fig
    

    def make_1d_plot_edge_1_step(self, fig:go.Figure, edge: DualEdge, x_point:npt.NDArray, offset=0):
        v = edge.right
        assert type(v.convex_set) == Hyperrectangle
        assert len(v.convex_set.lb()) == 1
        x_lin = np.linspace(v.convex_set.lb()[0], v.convex_set.ub()[0], 100, endpoint=True)

        potential = self.value_function_solution.GetSolution(v.potential).ToExpression()
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
        
        def plot_min(x, y, fig, color):
            # Find the index of the minimum y value
            min_y_index = np.argmin(y)
            # Get the corresponding x value
            min_x_value = x[min_y_index]
            min_y_value = y[min_y_index]
            fig.add_trace(go.Scatter(x=[min_x_value], y=[min_y_value], mode='markers', line=dict(color=color), showlegend=False))

        fig.add_trace(go.Scatter(x=x_lin, y=np.vectorize(evaluate_0)(x_lin)-offset, mode='lines', name="c+J", line=dict(color="red") ))
        fig.add_trace(go.Scatter(x=x_lin, y=np.vectorize(evaluate_1)(x_lin)-offset, mode='lines', name="c+J-BB", line=dict(color="blue") ))
        fig.add_trace(go.Scatter(x=x_lin, y=np.vectorize(evaluate_2)(x_lin)-offset, mode='lines', name="c+J-BB-B", line=dict(color="green") ))
        fig.add_trace(go.Scatter(x=x_lin, y=np.vectorize(evaluate_3)(x_lin)-offset, mode='lines', name="c-BB-B", line=dict(color="purple") ))
        fig.add_trace(go.Scatter(x=x_lin, y=np.vectorize(evaluate_4)(x_lin)-offset, mode='lines', name="-BB-B", line=dict(color="magenta") ))

        plot_min(x_lin, np.vectorize(evaluate_0)(x_lin)-offset, fig, "red")
        plot_min(x_lin, np.vectorize(evaluate_1)(x_lin)-offset, fig, "blue")
        plot_min(x_lin, np.vectorize(evaluate_2)(x_lin)-offset, fig, "green")
        plot_min(x_lin, np.vectorize(evaluate_3)(x_lin)-offset, fig, "purple")
        plot_min(x_lin, np.vectorize(evaluate_4)(x_lin)-offset, fig, "magenta")
        


    def make_2d_plot_edge_1_step(self, fig:go.Figure, edge: DualEdge, x_point:npt.NDArray, offset=0):
        v = edge.right
        assert type(v.convex_set) == Hyperrectangle
        assert len(v.convex_set.lb()) == 2
        x_lin = np.linspace(v.convex_set.lb()[0], v.convex_set.ub()[0], 100, endpoint=True)
        y_lin = np.linspace(v.convex_set.lb()[1], v.convex_set.ub()[1], 100, endpoint=True)
        X, Y = np.meshgrid(x_lin, y_lin)

        potential = self.value_function_solution.GetSolution(v.potential).ToExpression()
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

    def solve_restriction(self, vertices: T.List[DualVertex], x_0:npt.NDArray) -> T.Tuple[float, npt.NDArray, DualVertex, npt.NDArray]:
        prog = MathematicalProgram()
        assert np.all(vertices[0].B.dot(np.hstack(([1], x_0))) >= 0)
        x_n = x_0
        x_traj = []
        for v_index in range(1, len(vertices)):
            v = vertices[v_index]
            edge = self.edges[get_edge_name(vertices[v_index-1].name, v.name)]
            def cost_function(x,y):
                Q = self.value_function_solution.GetSolution(edge.lambda_2_right)
                q = self.value_function_solution.GetSolution(edge.lambda_1_right)
                B = v.B
                Y = np.outer( np.hstack(([1], y)), np.hstack(([1], y)) )
                cost = QUADRATIC_COST(x,y) 
                cost -= np.sum( B.dot(Y).dot(B.T) * Q) 
                cost -= B.dot( np.hstack(([1], y)) ).dot(q)
                if v_index == len(vertices)-1:
                    potential = self.value_function_solution.GetSolution(v.potential).ToExpression()
                    f_potential = lambda x: potential.Substitute({v.x[i]: x[i] for i in range(v.state_dim)})
                    cost += f_potential(y)
                cost += 1000
                return cost
            
            x_n1 = prog.NewContinuousVariables(v.state_dim)
            x_traj.append(x_n1)
            prog.AddLinearConstraint( ge(v.B.dot(np.hstack(([1], x_n1))), 0 ) )
            prog.AddCost( cost_function(x_n, x_n1) )
            x_n = x_n1

        solution = Solve(prog)
        INFO(solution.get_solver_id().name())
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
        
        best_cost, best_action, best_vertex = np.inf, None, None
        index_sequence = np.zeros(len(relevant_layers))
        go_on = True
        while go_on:
            vertex_sequence = get_vertex_sequence(index_sequence)
            # print(vertex_sequence)
            new_cost, new_action, new_vertex, _ = self.solve_restriction(vertex_sequence, x_0)
            if new_cost < best_cost:
                best_cost, best_action, best_vertex = new_cost, new_action, new_vertex
            go_on, index_sequence = get_next_index_sequence(index_sequence, upper)

        print(best_cost, best_vertex.name, best_action)
        return best_vertex, best_action


## ------------------------------------------------------
# extracting policy 

def solve_m_step_horizon_from_layers(gcs:PolynomialDualGCS, layers:T.List[T.List[DualVertex]], m:int, start_vertex:DualVertex, layer_index:int, point:npt.NDArray):
    new_gcs = GraphOfConvexSets()
    init_vertex = new_gcs.AddVertex(Point(point), start_vertex.name)
    new_layers = []
    new_layers.append([init_vertex])

    # for every layer
    last_index = min(len(layers)-1, layer_index+m)
    for n in range(layer_index+1, last_index+1):
        layer = []
        for og_graph_v in layers[n]:
            new_v = new_gcs.AddVertex(og_graph_v.convex_set, og_graph_v.name)
            layer.append(new_v)

            # connect with edges to previous layer
            for left_v in new_layers[-1]:
                edge = gcs.edges[ get_edge_name(left_v.name(), new_v.name()) ]
                def cost_function(x,y):
                    Q = gcs.value_function_solution.GetSolution(edge.lambda_2_right)
                    q = gcs.value_function_solution.GetSolution(edge.lambda_1_right)
                    B = og_graph_v.B
                    Y = np.outer( np.hstack(([1], y)), np.hstack(([1], y)) )
                    cost = QUADRATIC_COST(x,y) 
                    cost -= np.sum( B.dot(Y).dot(B.T) * Q) 
                    cost -= B.dot( np.hstack(([1], y)) ).dot(q)
                    if n == last_index:
                        potential = gcs.value_function_solution.GetSolution(og_graph_v.potential).ToExpression()
                        f_potential = lambda x: potential.Substitute({og_graph_v.x[i]: x[i] for i in range(og_graph_v.state_dim)})
                        cost += f_potential(y)
                    cost += 1000
                    return cost
                
                new_edge = new_gcs.AddEdge(left_v, new_v, get_edge_name(left_v.name(), new_v.name()) )
                new_edge.AddCost( cost_function(new_edge.xu(), new_edge.xv()))

        new_layers.append(layer)

    target_v = new_gcs.AddVertex(Hyperrectangle([0],[0]), "target" )
    for left_v in new_layers[-1]:
        new_gcs.AddEdge(left_v, target_v, get_edge_name(left_v.name(), target_v.name()) )


    gcs_options = GraphOfConvexSetsOptions()
    gcs_options.convex_relaxation = True
    gcs_options.max_rounded_paths = 30
    gcs_options.max_rounding_trials = 30        
    gcs_options.solver = MosekSolver()

    # solve
    result = new_gcs.SolveShortestPath(
        init_vertex, target_v, gcs_options
    )  # type: MathematicalProgramResult
    assert result.is_success()
    cost = result.get_optimal_cost()

    edge_path = new_gcs.GetSolutionPath(init_vertex, target_v, result)
    vertex_name_path = [init_vertex.name()]
    value_path = [point]
    for e in edge_path:
        vertex_name_path.append(e.v().name())
        value_path.append(result.GetSolution(e.v().x()))
    return cost, vertex_name_path, value_path


def get_next_action_by_solving_restrictions(gcs:PolynomialDualGCS, layers:T.List[T.List[DualVertex]], m:int, vertex:DualVertex, point:npt.NDArray, layer_index:int):
    return gcs.solve_m_step_policy(layers, m, vertex, point, layer_index)

def get_next_action_gcs(gcs:PolynomialDualGCS, layers:T.List[T.List[DualVertex]], m:int, vertex:DualVertex, point:npt.NDArray, layer_index:int):
    cost, vertex_name_path, value_path = solve_m_step_horizon_from_layers(gcs, layers, m, vertex, layer_index, point)
    print(cost, vertex_name_path[1], value_path[1])
    return gcs.vertices[vertex_name_path[1]], value_path[1]


def rollout_m_step_policy(gcs:PolynomialDualGCS, layers:T.List[T.List[DualVertex]], m:int, vertex:DualVertex, point:npt.NDArray, layer_index:int, use_gcs=True) -> T.Tuple[float, T.List[DualVertex], T.List[npt.NDArray]]:
    if layer_index < len(layers)-1:
        if use_gcs:
            next_vertex, next_point = get_next_action_gcs(gcs, layers, m, vertex, point, layer_index)
        else:
            next_vertex, next_point = get_next_action_by_solving_restrictions(gcs, layers, m, vertex, point, layer_index)
        cost, vertex_trajectory, trajectory = rollout_m_step_policy(gcs, layers, m, next_vertex, next_point, layer_index+1, use_gcs=use_gcs)
        return QUADRATIC_COST(point, next_point) + cost, [next_vertex] + vertex_trajectory, [next_point] + trajectory
    else:
        return 0.0, [], []


def plot_policy_rollout(gcs:PolynomialDualGCS, layers:T.List[T.List[DualVertex]], m:int, vertex:DualVertex, layer_index:int, fig:go.Figure, point:npt.NDArray, use_gcs=True):
    
    _, vertex_trajectory, trajectory = rollout_m_step_policy(gcs, layers, m, vertex, point, layer_index, use_gcs=use_gcs)
    vertex_trajectory = [vertex] + vertex_trajectory
    trajectory = [point] + trajectory

    x,y,z = [],[],[]
    n = len(vertex_trajectory)
    print("trajectory: ", trajectory)
    for i in range(n):
        point = trajectory[i]
        x.append(point[0]) 
        y.append(point[1])
        z.append(vertex_trajectory[i].cost_at_point(point, gcs.value_function_solution))
    fig.add_traces(go.Scatter3d(x=x, y=y, z=z, mode='lines', line=dict(width=5, color='black')))
    
