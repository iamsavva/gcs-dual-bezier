import pydot
import numpy as np

from pydrake.geometry.optimization import (
    GraphOfConvexSets,
    GraphOfConvexSetsOptions,
    Point,
)
from pydrake.solvers import (
    CommonSolverOption,
    MathematicalProgram,
    MosekSolver,
    SolverOptions,
)
from pydrake.all import le

from gcs.rounding import MipPathExtraction

def polytopeDimension(A, b, tol=1e-4):
    
    assert A.shape[0] == b.size
    
    m, n = A.shape
    eq = []

    while True:
        ineq = [i for i in range(m) if i not in eq]
        A_ineq = A[ineq]
        b_ineq = b[ineq]

        prog = MathematicalProgram()
        x = prog.NewContinuousVariables(n)
        r = prog.NewContinuousVariables(1)[0]
        
        if len(eq) > 0:
            prog.AddLinearEqualityConstraint(A_eq.dot(x), b_eq)
        if len(ineq) > 0:
            c = prog.AddLinearConstraint(le(A_ineq.dot(x) + r * np.ones(len(ineq)), b_ineq))
        prog.AddBoundingBoxConstraint(0, 1, r)
        
        prog.AddLinearCost(-r)

        solver = MosekSolver()
        result = solver.Solve(prog)

        if not result.is_success():
            return -1
        
        if result.GetSolution(r) > tol:
            eq_rank = 0 if len(eq) == 0 else np.linalg.matrix_rank(A_eq)
            return n - eq_rank
        
        c_opt = np.abs(result.GetDualSolution(c)) 
        eq += [ineq[i] for i, ci in enumerate(c_opt) if ci > tol]
        A_eq = A[eq]
        b_eq = b[eq]

class BaseGCS:
    def __init__(self, gcs:GraphOfConvexSets, options:GraphOfConvexSetsOptions, source:GraphOfConvexSets.Vertex, target:GraphOfConvexSets.Vertex):
        self.rounding_fn = []
        self.rounding_kwargs = {}

        self.gcs = gcs
        self.options = options
        self.source = source
        self.target = target

    def setRoundingStrategy(self, rounding_fn, **kwargs):
        self.rounding_kwargs = kwargs
        if callable(rounding_fn):
            self.rounding_fn = [rounding_fn]
        elif isinstance(rounding_fn, list):
            assert len(rounding_fn) > 0
            for fn in rounding_fn:
                assert callable(fn)
            self.rounding_fn = rounding_fn
        else:
            raise ValueError("Rounding strategy must either be "
                             "a function or list of functions.")

    def solveGCS(self, rounding, preprocessing, verbose):

        results_dict = {}
        self.options.convex_relaxation = rounding
        self.options.preprocessing = preprocessing
        self.options.max_rounded_paths = 0

        result = self.gcs.SolveShortestPath(self.source, self.target, self.options)

        if rounding:
            results_dict["relaxation_result"] = result
            try:
                results_dict["relaxation_solver_time"] = result.get_solver_details().optimizer_time
            except:
                results_dict["relaxation_solver_time"] = result.get_solver_details().solve_time
            results_dict["relaxation_cost"] = result.get_optimal_cost()
        else:
            results_dict["mip_result"] = result
            results_dict["mip_solver_time"] = result.get_solver_details().optimizer_time
            results_dict["mip_cost"] = result.get_optimal_cost()

        if not result.is_success():
            print("First solve failed")
            return None, None, results_dict

        if verbose:
            try:
                print("Solution\t",
                    "Success:", result.get_solution_result(),
                    "Cost:", result.get_optimal_cost(),
                    "Solver time:", result.get_solver_details().optimizer_time)
            except:
                print("Solution\t",
                    "Success:", result.get_solution_result(),
                    "Cost:", result.get_optimal_cost(),
                    "Solver time:", result.get_solver_details().solve_time)


        # Solve with hard edge choices
        if rounding and len(self.rounding_fn) > 0:
            # Extract path
            active_edges = []
            found_path = False
            for fn in self.rounding_fn:
                rounded_edges = fn(self.gcs, result, self.source, self.target,
                                   **self.rounding_kwargs)
                if rounded_edges is None:
                    print(fn.__name__, "could not find a path.")
                    active_edges.append(rounded_edges)
                else:
                    found_path = True
                    active_edges.extend(rounded_edges)
            results_dict["rounded_paths"] = active_edges
            if not found_path:
                print("All rounding strategies failed to find a path.")
                return None, None, results_dict

            self.options.preprocessing = False
            rounded_results = []
            best_cost = np.inf
            best_path = None
            best_result = None
            max_rounded_solver_time = 0.0
            total_rounded_solver_time = 0.0
            for path_edges in active_edges:
                if path_edges is None:
                    rounded_results.append(None)
                    continue
                for edge in self.gcs.Edges():
                    if edge in path_edges:
                        edge.AddPhiConstraint(True)
                    else:
                        edge.AddPhiConstraint(False)
                rounded_results.append(self.gcs.SolveShortestPath(
                    self.source, self.target, self.options))
                try:
                    solve_time = rounded_results[-1].get_solver_details().optimizer_time
                except:
                    solve_time = rounded_results[-1].get_solver_details().solve_time
                max_rounded_solver_time = np.maximum(solve_time, max_rounded_solver_time)
                total_rounded_solver_time += solve_time
                if (rounded_results[-1].is_success()
                    and rounded_results[-1].get_optimal_cost() < best_cost):
                    best_cost = rounded_results[-1].get_optimal_cost()
                    best_path = path_edges
                    best_result = rounded_results[-1]

            results_dict["best_path"] = best_path
            results_dict["best_result"] = best_result
            results_dict["rounded_results"] = rounded_results
            results_dict["max_rounded_solver_time"] =  max_rounded_solver_time
            results_dict["total_rounded_solver_time"] = total_rounded_solver_time
            results_dict["rounded_cost"] = best_result.get_optimal_cost()

            if verbose:
                print("Rounded Solutions:")
                for r in rounded_results:
                    if r is None:
                        print("\t\tNo path to solve")
                        continue
                    try:
                        print("\t\t",
                            "Success:", r.get_solution_result(),
                            "Cost:", r.get_optimal_cost(),
                            "Solver time:", r.get_solver_details().optimizer_time)
                    except:
                        print("\t\t",
                        "Success:", r.get_solution_result(),
                        "Cost:", r.get_optimal_cost(),
                        "Solver time:", r.get_solver_details().solve_time)

            if best_path is None:
                print("Second solve failed on all paths.")
                return best_path, best_result, results_dict
        elif rounding:
            self.options.max_rounded_paths = 10

            rounded_result = self.gcs.SolveShortestPath(self.source, self.target, self.options)
            best_path = MipPathExtraction(self.gcs, rounded_result, self.source, self.target)[0]
            best_result = rounded_result
            results_dict["best_path"] = best_path
            results_dict["best_result"] = best_result
            results_dict["rounded_results"] = [rounded_result]
            results_dict["rounded_cost"] = best_result.get_optimal_cost()
        else:
            best_path = MipPathExtraction(self.gcs, result, self.source, self.target)[0]
            best_result = result
            results_dict["best_path"] = best_path
            results_dict["best_result"] = best_result
            results_dict["mip_path"] = best_path

        if verbose:
            for edge in best_path:
                print("Added", edge.name(), "to path.")

        return best_path, best_result, results_dict
