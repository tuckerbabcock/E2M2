import copy

import numpy as np

import openmdao.api as om

from openmdao.core.driver import Driver

from .calibration import calibrate
from .utils import \
    estimate_lagrange_multipliers2, \
    get_active_constraints2, \
    constraint_violation2, \
    l1_merit_function2, \
    optimality2

CITATIONS = """
@article{alexandrov1998trust,
  title={A trust-region framework for managing the use of approximation models in optimization},
  author={Alexandrov, Natalia M and Dennis Jr, John E and Lewis, Robert Michael and Torczon, Virginia},
  journal={Structural optimization},
  volume={15},
  number={1},
  pages={16--23},
  year={1998},
  publisher={Springer}
}

@article{alexandrov2001approximation,
  title={Approximation and model management in aerodynamic optimization with variable-fidelity models},
  author={Alexandrov, Natalia M and Lewis, Robert Michael and Gumbert, Clyde R and Green, Lawrence L and Newman, Perry A},
  journal={Journal of Aircraft},
  volume={38},
  number={6},
  pages={1093--1101},
  year={2001}
}

"""


class TRMMDriver(Driver):
    def __init__(self, **kwargs):
        """
        Initialize the E2M2Driver.
        """
        super().__init__(**kwargs)

        # What we support
        self.supports['optimization'] = True
        self.supports['inequality_constraints'] = True
        self.supports['equality_constraints'] = True
        self.supports['two_sided_constraints'] = True
        self.supports['linear_constraints'] = True
        self.supports['simultaneous_derivatives'] = True

        # What we don't support
        self.supports['multiple_objectives'] = False
        self.supports['active_set'] = False
        self.supports['integer_design_vars'] = False
        self.supports['distributed_design_vars'] = False
        self.supports._read_only = True

        # The OpenMDAO problem that defines the low-fidelity problem
        self.low_fidelity_problem = None

        self.delta = self.options['delta0']
        self.penalty_param = 1.0

        self.lf_obj_name = None
        self.lf_con_names = []

        self.cite = CITATIONS

    def _declare_options(self):
        """
        Declare options before kwargs are processed in the init method.
        """
        self.options.declare('response_map',
                             default=dict(),
                             desc='')
        self.options.declare('opt_tol',
                             default=1e-6,
                             desc='The high-fidelity optimality tolerance used to determine convergence')
        self.options.declare('feas_tol',
                             default=1e-6,
                             desc='The high-fidelity feasibility tolerance used to determine convergence')
        self.options.declare('delta0',
                             default=0.01,
                             desc='')
        self.options.declare('delta_star',
                             default=10,
                             desc='')
        self.options.declare('r1',
                             default=0.1,
                             desc='')
        self.options.declare('r2',
                             default=0.75,
                             desc='')
        self.options.declare('c1',
                             default=0.5,
                             desc='')
        self.options.declare('c2',
                             default=2.0,
                             desc='')
        self.options.declare('max_iter',
                             default=15,
                             lower=0,
                             desc='Maximum number of iterations.')
        self.options.declare('disp',
                             default=True,
                             types=bool,
                             desc='Set to False to prevent printing of convergence messages')

    def _setup_driver(self, problem):
        """
        Prepare the driver for execution.

        This is the final thing to run during setup.

        Parameters
        ----------
        problem : <Problem>
            Pointer to the containing problem.
        """
        print(f"TRMMDriver::_setup_driver()!")
        super()._setup_driver(problem)

        if self.low_fidelity_problem is None:
            raise RuntimeError('Low fidelity problem is not set!')

    def run(self):
        """
        Optimize the problem using TRMM optimizer.

        Returns
        -------
        bool
            Failure flag; True if failed to converge, False is successful.
        """
        opt_tol = self.options['opt_tol']
        feas_tol = self.options['feas_tol']

        hf_prob = self._problem()
        lf_prob = self.low_fidelity_problem
        lf_model = self.low_fidelity_problem.model

        # Calculate initial merit function
        hf_prob.run_model()
        old_merit_function = l1_merit_function2(
            self, self.penalty_param, feas_tol)

        for k in range(1, self.options['max_iter']+1):

            # Update LF reference point
            unscaled_hf_dv_vals = self.get_design_var_values(
                driver_scaling=False)
            for dv, dv_val in unscaled_hf_dv_vals.items():
                lf_prob.set_val(f"{dv}_k", dv_val)
                lf_prob.set_val(f"delta_{dv}", 0.0)

            # Update trust radius
            lf_prob.set_val('delta', self.delta)

            lf_prob.run_model()

            # Calibrate LF model at current design point
            calibrate(lf_prob,
                      hf_prob,
                      self.options['response_map'],
                      self._designvars)

            # Optimize calibrated LF model
            lf_prob.driver.opt_settings['Print file'] = f"{lf_prob._name}_{k}.out"
            lf_prob.driver.opt_settings['hist_file'] = f"{lf_prob._name}_{k}.db"
            lf_prob.run_driver(case_prefix=f'sub_opt_{k}')

            # Evaluated predicted merit function at new point
            lf_merit_function = l1_merit_function2(
                lf_prob.driver, self.penalty_param, feas_tol)

            # Evaluate HF merit function at new point
            lf_con_vals = lf_prob.driver.get_constraint_values()
            hf_dvs = self.get_design_var_values()
            for dv in hf_dvs:
                self.set_design_var(dv, lf_con_vals[f'new_design.{dv}'])

            hf_prob.run_model()
            merit_function = l1_merit_function2(
                self, self.penalty_param, feas_tol)

            lf_prob.model.list_inputs()
            lf_prob.model.list_outputs()
            hf_prob.model.list_inputs()
            hf_prob.model.list_outputs()

            # Update trust radius based on actual and predicted behavior
            ared = old_merit_function - merit_function
            pred = old_merit_function - lf_merit_function
            r = ared / pred
            self._update_trust_radius(lf_prob['step_norm'], r)

            lf_cons = lf_prob.driver._cons
            lf_dvs = lf_prob.driver._designvars
            lf_dv_vals = lf_prob.driver.get_design_var_values()
            active_lf_cons = get_active_constraints2(
                lf_cons, lf_con_vals, lf_dvs, lf_dv_vals, feas_tol)
            print(f"active lf cons: {active_lf_cons}")

            # Estimate LF lagrange multipliers
            obj_vals = self.get_objective_values()
            for obj in obj_vals:
                hf_obj_name = obj
                break
            # lf_obj_name = self.options['response_map'][hf_obj_name]
            hf_obj_name = "f"
            lf_obj_name = "f_hat"
            lf_totals = lf_prob.compute_totals([lf_obj_name, *active_lf_cons],
                                               [*lf_dvs.keys()],
                                               driver_scaling=True)
            print(f"lf_totals: {lf_totals}")
            lf_duals = estimate_lagrange_multipliers2(lf_obj_name,
                                                      active_lf_cons,
                                                      lf_dvs,
                                                      lf_totals)

            print(f"lf_duals: {lf_duals}")
            hf_duals = copy.deepcopy(lf_duals)

            con_vals = self.get_constraint_values()
            con_violation = constraint_violation2(self, self._cons, con_vals, feas_tol)

            hf_duals.pop("trust_radius_constraint.trust_radius_constraint", None)
            for con in self._cons:
                hf_duals[con] = hf_duals.pop(f"{con}_hat")
                hf_duals.pop(f"{con}_slack_1", None)
                hf_duals.pop(f"{con}_slack_2", None)
            for dv in self._designvars:
                dv_dual = hf_duals.pop(f"new_design.{dv}", None)
                if dv_dual is not None:
                    hf_duals[dv] = dv_dual
            print(f"hf_duals: {hf_duals}")

            # hf_multipliers['power_out'] = hf_multipliers.pop('power_out_con')
            # hf_multipliers.pop('slack_power_out_1', None)
            # hf_multipliers.pop('slack_power_out_2', None)

            if len(hf_duals) > 0:
                self.penalty_param = 2.0 * abs(max(hf_duals.values(), key=abs))

            if len(con_violation) > 0:
                max_constraint_violation = abs(max(con_violation.values(), key=abs))
            else:
                max_constraint_violation = 0.0

            hf_cons = self._cons
            hf_con_vals = self.get_constraint_values()
            hf_dvs = self._designvars
            hf_dv_vals = self.get_design_var_values()
            active_hf_cons = get_active_constraints2(
                hf_cons, hf_con_vals, hf_dvs, hf_dv_vals, feas_tol)

            print(f"active hf cons: {active_hf_cons}")
            print(f"hf inputs:")
            hf_prob.model.list_inputs()
            hf_totals = hf_prob.compute_totals([hf_obj_name, *active_hf_cons],
                                               [*hf_dvs.keys()],
                                               driver_scaling=True)
            print(f"hf_totals: {hf_totals}")

            optim = optimality2(hf_obj_name, active_hf_cons, hf_dvs, hf_duals, hf_totals)
            print(f"{80*'#'}")
            print(
                f"{k}: Merit function value: {merit_function}, optimality: {optim}, max constraint violation: {max_constraint_violation}")
            print(f"{80*'#'}")


            if optim < opt_tol and max_constraint_violation < feas_tol:
                break

            # # x.append(dict())
            # # for key in x[k-1].keys():
            # #     x[k][key] = copy.deepcopy(lf_prob[key])
            # #     print(f"Setting hf value {key} to {x[k][key]}")
            # #     hf_prob[key] = x[k][key]

            # for dv in lf_dvs.keys():
            #     hf_prob[dv][:] = lf_prob[dv]


    def _update_trust_radius(self, step_norm, r):
        print(f"old delta: {self.delta}")
        if r < self.options['r1']:
            self.delta = self.options['c1'] * np.sqrt(step_norm)
        elif r > self.options['r2']:
            self.delta = min(
                self.options['c2']*self.delta, self.options['delta_star'])
        print(f"new delta: {self.delta}")
