import typing as T

import numpy as np
import numpy.typing as npt

from util import timeit, INFO, YAY, ERROR, WARN  # pylint: disable=unused-import

from pydrake.geometry.optimization import (  # pylint: disable=import-error, no-name-in-module
    HPolyhedron,
)

FREE_POLY = "free_poly"
PSD_POLY = "psd_poly"
CONVEX_POLY = "convex_poly"


class ProgramOptions:
    def __init__(self):
        # -----------------------------------------------------------------------------------
        # general settings pertaining to any optimization program
        # -----------------------------------------------------------------------------------
        # multi step lookahead policy

        self.pot_type = FREE_POLY
        self.zero_offset = 50

        self.solve_ot = False
        self.solve_ot_stochastic_transitions = False
        self.solve_ot_relaxed_stochastic_transitions = False
        self.solve_ot_deterministic_transitions_inflated = False
        self.solve_robustified_set_membership = False

        self.gamma = 1
        self.noise_magnitude = 1

        self.policy_subtract_full_s_procedure = False
        self.policy_subtract_right_vertex_s_procedure = False

        self.policy_lookahead = 1
        self.policy_use_gcs = True

        self.policy_gcs_edge_cost_offset = 1e4

        self.MSK_DPAR_INTPNT_CO_TOL_REL_GAP = 1e-8
        self.MSK_DPAR_INTPNT_CO_TOL_PFEAS = 1e-8
        self.MSK_DPAR_INTPNT_CO_TOL_DFEAS = 1e-8
