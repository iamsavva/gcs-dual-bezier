import typing as T
import numpy.typing as npt

import numpy as np
import scipy as sp

from pydrake.math import eq, le, ge

from pydrake.solvers import (  # pylint: disable=import-error, no-name-in-module, unused-import
    MathematicalProgram,
    MathematicalProgramResult,
    Solve,
    MosekSolver,
    MosekSolverDetails,
    SolverOptions,
    CommonSolverOption,
)
from pydrake.geometry.optimization import (  # pylint: disable=import-error, no-name-in-module
    HPolyhedron,
    Point,
    ConvexSet,
    Hyperrectangle,
    VPolytope,
)
from pydrake.symbolic import (
    Polynomial,
    Variable,
    Variables,
    Expression,
)  # pylint: disable=import-error, no-name-in-module, unused-import


import plotly.graph_objs as go


class PrimalVertex:
    def __init__(self, name: str, dim: int):
        self.bounds = dict()
        self.dim = dim
        self.name = name
        self.edges_in = []
        self.edges_out = []

    def add_edge_in(self, edge):
        self.edges_in.append(edge)

    def add_edge_out(self, edge):
        self.edges_out.append(edge)

    def add_measure_conservation_constraints_just_01(
        self, prog: MathematicalProgram, start_measure=None, target_measure=None
    ):
        if len(self.edges_out) > 0:
            measure_out = sum([e.get_left_measure()[0, :] for e in self.edges_out])
        else:
            assert target_measure is not None
            measure_out = target_measure[0, :]

        if len(self.edges_in) > 0:
            measure_in = sum([e.get_right_measure()[0, :] for e in self.edges_in])
        else:
            assert start_measure is not None
            measure_in = start_measure[0, :]

        prog.AddLinearConstraint(eq(measure_in, measure_out))

    def add_measure_conservation_constraints(
        self,
        prog: MathematicalProgram,
        start_measure=None,
        target_measure=None,
        Q: npt.NDArray = np.zeros((3, 3)),
    ):
        if len(self.edges_out) > 0:
            measure_out = sum([e.get_left_measure() for e in self.edges_out])
        else:
            assert target_measure is not None
            measure_out = target_measure

        if len(self.edges_in) > 0:
            measure_in = sum([e.get_right_measure() for e in self.edges_in])
        else:
            assert start_measure is not None
            measure_in = start_measure

        flow_in = measure_in[0, 0]
        prog.AddLinearConstraint(eq(measure_in + flow_in * Q, measure_out))

    def get_measure(self, solution: MathematicalProgramResult):
        left_m = sum([e.get_right_measure(solution) for e in self.edges_in])
        right_m = sum([e.get_left_measure(solution) for e in self.edges_out])
        if len(self.edges_in) == 0:
            return np.round(right_m, 3)
        elif len(self.edges_out) == 0:
            return np.round(left_m, 3)
        else:
            return np.round(left_m, 3), np.round(right_m, 3)


class PrimalBoxVertex(PrimalVertex):
    def __init__(self, name: str, box: Hyperrectangle):
        dim = len(box.Center())
        super(PrimalBoxVertex, self).__init__(name, dim)
        self.box = box

    def get_nonnegative_constraints_matrix(self):
        hpoly = self.box.MakeHPolyhedron()
        A, b = hpoly.A(), hpoly.b()
        B = np.hstack((b.reshape((len(b), 1)), -A))
        return B

    def get_singleton_measure(self, x: npt.NDArray = None):
        if x is not None:
            assert np.all(self.box.lb() <= x)
            assert np.all(self.box.ub() >= x)
        else:
            x = self.box.Center()
        y = np.hstack(([1], x))
        return np.outer(y, y)

    def get_uniform_measure(self, box: Hyperrectangle):
        lb, ub = box.lb(), box.ub()
        n = len(lb)
        prog = MathematicalProgram()
        vars = prog.NewContinuousVariables(n)

        def compute_moments(moment):
            return compute_box_moment(lb, ub, vars, moment)

        vecotrized_moment_compute = np.vectorize(compute_moments)

        m0 = vecotrized_moment_compute(np.ones(1))
        m1 = vecotrized_moment_compute(vars)
        m2 = vecotrized_moment_compute(np.outer(vars, vars))
        return np.vstack(
            (np.hstack((m0, m1)), np.hstack((m1.reshape((len(m1), 1)), m2)))
        )
        # return m0[0], m1, m2


def compute_box_moment(lb: npt.NDArray, ub: npt.NDArray, vars: npt.NDArray, moment):
    state_dim = len(vars)
    p = 1 / np.prod(ub - lb)
    if not isinstance(moment, Expression):
        moment = Expression(moment)
    poly = Polynomial(moment)
    for i in range(state_dim):
        x_min, x_max, x_val = lb[i], ub[i], vars[i]
        integral_of_poly = poly.Integrate(x_val)
        poly = integral_of_poly.EvaluatePartial(
            {x_val: x_max}
        ) - integral_of_poly.EvaluatePartial({x_val: x_min})
    poly = poly * p
    return float(poly.Evaluate(dict()))


class PrimalEdge:
    def __init__(
        self,
        v_left: PrimalVertex,
        v_right: PrimalVertex,
        prog: MathematicalProgram,
        add_noise: bool = False,
        gamma: int = 0,
        multiplier=1,
    ):
        self.multiplier = 1
        self.left = v_left.name
        self.right = v_right.name
        self.dim = v_left.dim
        assert v_left.dim == v_right.dim

        self.add_noise = add_noise
        self.gamma = gamma
        if self.add_noise:
            self.eps_left = prog.NewContinuousVariables(1)[0]
            self.eps_right = prog.NewContinuousVariables(1)[0]
            prog.AddLinearConstraint(self.eps_left >= 0)
            prog.AddLinearConstraint(self.eps_right >= 0)

        self.define_psd_distribution_matrix(prog)
        self.add_quadratic_cost(prog)
        self.add_constraints(prog, v_left, v_right)
        v_left.add_edge_out(self)
        v_right.add_edge_in(self)

    def define_psd_distribution_matrix(self, prog: MathematicalProgram):
        dim = 1 + 2 * self.dim
        self.edge_measure = prog.NewSymmetricContinuousVariables(rows=dim)
        prog.AddPositiveSemidefiniteConstraint(self.edge_measure)
        # 1 x    y     u     v
        # x x^2  xy    ux    vx
        # y xy   y^2   uy    vy
        # u xu  uy     u^2   uv
        # v xv  vy     uv    v^2

    def get_left_measure(self, solution: MathematicalProgramResult = None):
        m = self.edge_measure[: 1 + self.dim, : 1 + self.dim]
        if solution is None:
            return m
        else:
            return solution.GetSolution(m)

    def get_right_measure(self, solution: MathematicalProgramResult = None):
        ls = 1 + self.dim
        m = np.vstack(
            (
                np.hstack((self.edge_measure[0, 0], self.edge_measure[0, ls:])),
                np.hstack((self.edge_measure[ls:, 0:1], self.edge_measure[ls:, ls:])),
            )
        )
        if solution is None:
            return m
        else:
            return solution.GetSolution(m)

    def get_left_right_measure(self, solution: MathematicalProgramResult = None):
        ls = 1 + self.dim
        m = np.vstack(
            (
                np.hstack((self.edge_measure[0, 0], self.edge_measure[0, ls:])),
                np.hstack((self.edge_measure[1:ls, 0:1], self.edge_measure[1:ls, ls:])),
            )
        )
        if solution is None:
            return m
        else:
            return solution.GetSolution(m)

    def add_quadratic_cost(self, prog: MathematicalProgram):
        dim = self.dim
        eye = np.eye(dim)
        C = np.vstack((np.hstack((eye, -eye)), np.hstack((-eye, eye))))
        C = np.vstack((np.zeros(1 + 2 * dim), np.hstack((np.zeros((2 * dim, 1)), C))))
        assert C.shape == (1 + 2 * dim, 1 + 2 * dim)
        prog.AddLinearCost(np.sum(C * self.edge_measure))
        # prog.AddLinearCost(np.trace(C @ self.edge_measure))
        if self.add_noise:
            prog.AddLinearCost(-self.gamma * (self.eps_right + self.eps_left))
            # prog.AddLinearCost(-self.gamma * (self.eps_right) )

    def add_constraints(
        self, prog: MathematicalProgram, left: PrimalVertex, right: PrimalVertex
    ):
        Bl = left.get_nonnegative_constraints_matrix()
        Br = right.get_nonnegative_constraints_matrix()

        if self.add_noise:
            noise_mat = self.eps_right * np.eye(1 + self.dim)
            noise_mat[0, 0] = 0
            prog.AddLinearConstraint(
                self.eps_right <= self.edge_measure[0, 0] * self.multiplier
            )

            noise_mat_left = self.eps_left * np.eye(1 + self.dim)
            noise_mat_left[0, 0] = 0
            prog.AddLinearConstraint(
                self.eps_left <= self.edge_measure[0, 0] * self.multiplier
            )
        else:
            noise_mat = np.zeros((1 + self.dim, 1 + self.dim))
            noise_mat_left = np.zeros((1 + self.dim, 1 + self.dim))

        Ml = self.get_left_measure() + noise_mat_left
        Mr = self.get_right_measure() + noise_mat
        Mlr = self.get_left_right_measure()

        e_1 = np.zeros((1 + self.dim, 1))
        e_1[0] = 1

        prog.AddLinearConstraint(ge(Bl @ Ml @ e_1, 0))
        prog.AddLinearConstraint(ge(Bl @ Ml @ Bl.T, 0))

        prog.AddLinearConstraint(ge(Br @ Mr @ e_1, 0))
        prog.AddLinearConstraint(ge(Br @ Mr @ Br.T, 0))

        prog.AddLinearConstraint(ge(Bl @ Mlr @ Br.T, 0))


def plot_surface(
    fig: go.Figure,
    box: Hyperrectangle,
    m: npt.NDArray,
    name: str,
    scaling: float = 1,
    cmax=10,
):
    # Our 2-dimensional distribution will be over variables X and Y
    eps = 0.0
    N = 100
    X = np.linspace(box.lb()[0] - eps, box.ub()[0] + eps, N)
    Y = np.linspace(box.lb()[1] - eps, box.ub()[1] + eps, N)
    X, Y = np.meshgrid(X, Y)

    m0 = m[0, 0]
    print(m0, name)
    # Mean vector and covariance matrix
    mu = m[0, 1:] / m0
    print(mu)

    Sigma = m[1:, 1:] / m0 - np.outer(mu, mu)
    # if (not np.allclose(m[0,0],0)) and np.allclose(Sigma,0, atol=1e-4):
    #     Sigma = np.eye(2)*0.001

    Sigma += np.eye(2) * 0.001

    print(Sigma)

    # Pack X and Y into a single 3-dimensional array
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X
    pos[:, :, 1] = Y

    def multivariate_gaussian(pos, mu, Sigma):
        """Return the multivariate Gaussian distribution on array pos."""

        n = mu.shape[0]
        Sigma_det = np.linalg.det(Sigma)
        try:
            Sigma_inv = np.linalg.inv(Sigma)
        except:
            return np.zeros((len(pos), len(pos)))

        N = np.sqrt((2 * np.pi) ** n * Sigma_det)
        # This einsum call calculates (x-mu)T.Sigma-1.(x-mu) in a vectorized
        # way across all the input variables.
        fac = np.einsum("...k,kl,...l->...", pos - mu, Sigma_inv, pos - mu)

        return np.exp(-fac / 2) / N

    # The distribution on the variables X, Y packed into pos.
    Z = multivariate_gaussian(pos, mu, Sigma) * m0 * scaling

    # Create filled 3D contours
    fig.add_trace(
        go.Surface(x=X, y=Y, z=Z, surfacecolor=Z, name=name, cmin=0, cmax=cmax)
    )

    # Update layout
    fig.update_layout(scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"))
