
from pydrake.solvers import (  # pylint: disable=import-error, no-name-in-module, unused-import
    MathematicalProgram,
    GurobiSolver
)
prog  = MathematicalProgram()
sovler = GurobiSolver()
sovler.Solve(prog)