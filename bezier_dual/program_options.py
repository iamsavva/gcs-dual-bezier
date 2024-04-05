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


        # this should be added automatically if i use corresponding constraints on dual
        self.policy_add_violation_penalties = False # this doesn't seem to help
        

        self.policy_verbose_choices = False
        self.policy_verbose_number_of_restrictions_solves = False

        self.postprocess_by_solving_restriction_on_mode_sequence = True
        self.verbose_restriction_improvement = False

        # ---

        self.use_lookahead_policy = False
        self.use_lookahead_with_backtracking_policy = False
        self.use_cheap_a_star_policy = False

        self.policy_use_zero_heuristic = False

        # solve with default otherwise
        self.solve_with_gurobi = False
        self.solve_with_mosek = False
        self.solve_with_snopt = False
        self.solve_with_clarabel = False
        self.solve_with_osqp = False


        # ----------------------------------
        self.gcs_policy_use_convex_relaxation = False
        self.gcs_policy_max_rounding_trials = 30
        self.gcs_policy_use_preprocessing = True

    def vertify_options_validity(self):
        assert (
            self.num_control_points >= 3
        ), "need at least 1 control point in the interior, 3 total"
        assert self.policy_lookahead >= 1, "lookahead must be positive"
        assert self.pot_type in (
            FREE_POLY,
            PSD_POLY,
            CONVEX_POLY,
        ), "undefined potentia type"
        policy_options = np.array([self.use_lookahead_policy, self.use_lookahead_with_backtracking_policy, self.use_cheap_a_star_policy])
        assert not np.sum(policy_options) < 1, "must select policy lookahead option"
        assert not np.sum(policy_options) > 1, "selected multiple policy lookahead options"

        solver_options = np.array([self.solve_with_gurobi, self.solve_with_mosek, self.solve_with_snopt, self.solve_with_clarabel, self.solve_with_osqp])
        assert np.sum(solver_options) <= 1, "more than 1 solver option selected"


