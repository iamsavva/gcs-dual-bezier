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

        # potential type
        self.pot_type = FREE_POLY
        self.zero_offset = 50

        self.max_flow_through_edge = 1

        self.use_robust_mosek_parameters = True

        self.policy_lookahead = 1
        self.policy_use_gcs = True

        self.policy_gcs_edge_cost_offset = 0

        self.MSK_DPAR_INTPNT_CO_TOL_REL_GAP = 1e-8
        self.MSK_DPAR_INTPNT_CO_TOL_PFEAS = 1e-8
        self.MSK_DPAR_INTPNT_CO_TOL_DFEAS = 1e-8

        # ---------------------------------------------
        # parameters specific to bezier
        self.num_control_points = 3
        self.policy_add_G_term = False
        # self.policy_no_vertex_revisits = True # depricated
        # self.policy_min_cost = True # depricated

        self.policy_add_total_flow_in_violation_penalty = False

        self.policy_verbose_choices = False

        self.postprocess_by_solving_restrction_on_mode_sequence = True
        self.verbose_restriction_improvement = False

    def test_options(self):
        assert (
            self.num_control_points >= 3
        ), "need at least 1 control point in the interior, 3 total"
        assert self.policy_lookahead >= 1, "lookahead must be positive"
        assert self.pot_type in (
            FREE_POLY,
            PSD_POLY,
            CONVEX_POLY,
        ), "undefined potential"
