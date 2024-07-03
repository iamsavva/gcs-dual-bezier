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

        # potential type
        self.potential_poly_deg = 2
        self.pot_type = FREE_POLY

        self.max_flow_through_edge = 1
        self.policy_lookahead = 1

        # ----------------------------------
        # MOSEK solver related
        self.use_robust_mosek_parameters = True
        self.MSK_DPAR_INTPNT_CO_TOL_REL_GAP = 1e-8
        self.MSK_DPAR_INTPNT_CO_TOL_PFEAS = 1e-8
        self.MSK_DPAR_INTPNT_CO_TOL_DFEAS = 1e-8


        # ----------------------------------
        # S procedure
        self.s_procedure_multiplier_degree_for_linear_inequalities = 0
        self.s_procedure_use_quadratic_multilpiers = True
        self.s_procedure_quadratic_multiply_left_and_right = True

        # ---------------------------------------------
        # parameters specific to bezier curves
        self.postprocess_by_solving_restriction_on_mode_sequence = True
        self.verbose_restriction_improvement = False

        # ----------------------------------
        # specify the policy
        self.use_lookahead_policy = False
        self.use_lookahead_with_backtracking_policy = False
        self.use_cheap_a_star_policy = False
        self.use_cheap_a_star_policy_parallelized = False
        self.policy_use_zero_heuristic = False # if you wanna use Dijkstra instead

        # ----------------------------------
        # solver selection for the lookaheads in the policy
        self.policy_solver = None
        self.gcs_policy_solver = None
        self.policy_use_robust_mosek_params = False
        self.policy_MSK_DPAR_INTPNT_CO_TOL_REL_GAP = 1e-6
        self.policy_MSK_DPAR_INTPNT_CO_TOL_PFEAS = 1e-6
        self.policy_MSK_DPAR_INTPNT_CO_TOL_DFEAS = 1e-6

        self.policy_verbose_choices = False
        self.policy_verbose_number_of_restrictions_solves = False


        # ----------------------------------
        # GCS policy related settings.
        # as of now, only used for computing optimal solutions.
        self.gcs_policy_use_convex_relaxation = True
        self.gcs_policy_max_rounding_trials = 100
        self.gcs_policy_max_rounded_paths = 100
        self.gcs_policy_use_preprocessing = True

        self.verbose_solve_times = False

        self.flow_violation_polynomial_degree = 0

        self.policy_use_l2_norm_cost = False
        self.policy_use_quadratic_cost = False

        self.backtracking_iteration_limit = 10000

        self.dont_use_flow_violations = False

        self.allow_vertex_revisits = False

        # ----------------------------------
        # these should probably be depricated
        # ----------------------------------


        

    def vertify_options_validity(self):
        assert self.policy_lookahead >= 1, "lookahead must be positive"
        assert self.pot_type in (
            FREE_POLY,
            PSD_POLY,
            CONVEX_POLY,
        ), "undefined potentia type"
        policy_options = np.array([self.use_lookahead_policy, self.use_lookahead_with_backtracking_policy, self.use_cheap_a_star_policy, self.use_cheap_a_star_policy_parallelized])
        assert not np.sum(policy_options) < 1, "must select policy lookahead option"
        assert not np.sum(policy_options) > 1, "selected multiple policy lookahead options"

        assert not (self.policy_use_l2_norm_cost and self.policy_use_quadratic_cost)



