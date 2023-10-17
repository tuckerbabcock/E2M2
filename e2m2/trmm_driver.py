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
                             default=0.1,
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
                             default=20,
                             lower=0,
                             desc='Maximum number of iterations.')
        self.options.declare('mu_max',
                             default=10,
                             lower=1,
                             desc='Maximum penalty parameter for solving elastic sub-problem.')
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

        obj_vals = self.get_objective_values()
        for obj in obj_vals:
            hf_obj_name = obj
            break
        lf_prob["obj_scaler"] = self._objs[hf_obj_name]['total_scaler'] or 1.0
        lf_prob["obj_adder"] = self._objs[hf_obj_name]['total_adder'] or 0.0

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
            lf_prob.driver.options['hist_file'] = f"{lf_prob._name}_{k}.db"
            lf_prob.run_driver(case_prefix=f'sub_opt_{k}')

            # lf_prob['delta_x'] -= 0.1
            # lf_prob.run_model()

            # Evaluated predicted merit function at new point
            lf_merit_function = l1_merit_function2(
                lf_prob.driver, self.penalty_param, feas_tol)

            # Evaluate HF merit function at new point
            lf_con_vals = lf_prob.driver.get_constraint_values()
            lf_dv_vals = lf_prob.driver.get_design_var_values()
            hf_dvs = self._designvars
            hf_dv_vals = self.get_design_var_values()
            for dv in hf_dvs:
                # self.set_design_var(dv, lf_con_vals[f'new_design.{dv}'])
                scaler = hf_dvs[dv]['total_scaler'] or 1.0
                new_dv = hf_dv_vals[dv] + lf_dv_vals[f"delta_{dv}"] * scaler
                self.set_design_var(dv, new_dv)

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
            # print([lf_obj_name, *active_lf_cons])
            # print([*hf_dvs.keys()])
            lf_totals = lf_prob.compute_totals([lf_obj_name, *active_lf_cons],
                                               [*lf_dvs.keys()],
                                               driver_scaling=False)
            print(f"lf_totals: {lf_totals}")
            lf_duals = estimate_lagrange_multipliers2(lf_obj_name,
                                                      active_lf_cons,
                                                      lf_dvs,
                                                      lf_totals)

            print(f"lf_duals: {lf_duals}")
            hf_duals = copy.deepcopy(lf_duals)

            con_vals = self.get_constraint_values()
            con_violation = constraint_violation2(self, self._cons, con_vals, feas_tol)

            hf_duals.pop("trust_radius_con", None)
            for metadata in self._cons.values():
                con = metadata['name']
                eq_con_name = f"elastic_{con}_con"
                ineq_lb_con_name = f"elastic_{con}_con_lb"
                ineq_ub_con_name = f"elastic_{con}_con_ub"
                if eq_con_name in hf_duals:
                    hf_duals[con] = hf_duals.pop(eq_con_name)
                elif ineq_lb_con_name in hf_duals:
                    hf_duals[con] = hf_duals.pop(ineq_lb_con_name)
                elif ineq_ub_con_name in hf_duals:
                    hf_duals[con] = hf_duals.pop(ineq_ub_con_name)

                hf_duals.pop(f"{con}_slack_1", None)
                hf_duals.pop(f"{con}_slack_2", None)
                hf_duals.pop(f"{con}_lb_slack", None)
                hf_duals.pop(f"{con}_ub_slack", None)

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
            # print(f"hf inputs:")
            # hf_prob.model.list_inputs()
            hf_totals = hf_prob.compute_totals([hf_obj_name, *active_hf_cons],
                                               [*hf_dvs.keys()],
                                               driver_scaling=False)
            print(f"hf_totals: {hf_totals}")

            # hf_duals = estimate_lagrange_multipliers2(hf_obj_name,
            #                                           active_hf_cons,
            #                                           hf_dvs,
            #                                           hf_totals)
            # print(f"hf_duals: {hf_duals}")

            optim = optimality2(hf_obj_name, active_hf_cons, hf_dvs, hf_duals, hf_totals)
            print(f"{80*'#'}")
            print(
                f"{k}: Merit function value: {merit_function}, optimality: {optim}, max constraint violation: {max_constraint_violation}")
            print(f"{80*'#'}")

            if optim < opt_tol and max_constraint_violation < feas_tol:
                break

            # update_penalty(lofi_prob, constraint_targets, constraints0, constraintsk, feas_tol, mu_max=10)

            self._update_penalty(k, self._cons, con_violation)
            # # x.append(dict())
            # # for key in x[k-1].keys():
            # #     x[k][key] = copy.deepcopy(lf_prob[key])
            # #     print(f"Setting hf value {key} to {x[k][key]}")
            # #     hf_prob[key] = x[k][key]

            # for dv in lf_dvs.keys():
            #     hf_prob[dv][:] = lf_prob[dv]


    def _update_trust_radius(self, step_norm, r):
        # print(f"old delta: {self.delta}")
        if r < self.options['r1']:
            self.delta = self.options['c1'] * np.sqrt(step_norm)
        elif r > self.options['r2']:
            self.delta = min(
                self.options['c2']*self.delta, self.options['delta_star'])
        # print(f"new delta: {self.delta}")

    def _update_penalty(self, k, cons, con_violation):
        print(f"cons: {cons}")
        print(f"con violation: {con_violation}")
        if len(con_violation) == 0:
            return
        
        feas_tol = self.options['feas_tol']

        lf_prob = self.low_fidelity_problem
        current_mu = lf_prob['mu']

        lf_prob['mu'] = 1.0
        lf_prob["obj_scaler"] = 0.0
        lf_prob["obj_adder"] = 0.0

        lf_prob.driver.opt_settings['Print file'] = f"{lf_prob._name}_feasibility_opt_{k}.out"
        lf_prob.driver.options['hist_file'] = f"{lf_prob._name}_feasibility_opt_{k}.db"
        lf_prob.run_driver(case_prefix=f'feasibility_opt_{k}')

        lf_prob.model.list_inputs()
        lf_prob.model.list_outputs()
        # lf_prob["obj_scaler"] = self._objs[hf_obj_name]['total_scaler']
        # lf_prob["obj_adder"] = self._objs[hf_obj_name]['total_adder']

        con_violation_inf = {}
        for con in con_violation:
            con_name = cons[con]['name']
            lf_con_name = self.options['response_map'][con_name][1]
            scaler = cons[con]['total_scaler'] or 1.0
            adder = cons[con]['total_adder'] or 0.0
            con_val = (lf_prob[lf_con_name] + adder) * scaler
            print(f"{con} val: {con_val}")
            if cons[con]['equals'] is not None:
                con_target = cons[con]["equals"]
                print(f"{con} target: {con_target}, value: {con_val}")
                if not np.isclose(con_val, con_target, atol=feas_tol, rtol=feas_tol):
                    print(f"violates equality constraint!")
                    con_violation_inf[con] = con_val - con_target
                else:
                    con_violation_inf[con] = 0.0
            else:
                con_ub = cons[con].get("upper", np.inf)
                con_lb = cons[con].get("lower", -np.inf)
                print(f"{con} lower bound: {con_lb}, upper bound: {con_ub}, value: {con_val}")
                if con_val > con_ub:
                    if not np.isclose(con_val, con_ub, atol=feas_tol, rtol=feas_tol):
                        print(f"violates upper bound!")
                        con_violation_inf[con] = con_val - con_ub
                elif con_val < con_lb:
                    if not np.isclose(con_val, con_lb, atol=feas_tol, rtol=feas_tol):
                        print(f"violates lower bound!")
                        con_violation_inf[con] = con_val - con_lb
                else:
                    con_violation_inf[con] = 0.0

        print(f"con_violation_inf: {con_violation_inf}")

        max_con_violation_k = abs(max(con_violation.values(), key=abs))
        max_con_violation_inf = abs(max(con_violation_inf.values(), key=abs))

        # if max_con_violation_k < feas_tol and 0.75*current_mu > feas_tol:
        #     # lofi_prob[f'mu_{constraint}'] = 0.75*old_mu[constraint]
        #     lf_prob['mu'] = 0.75 * current_mu
        if  max_con_violation_inf < 0.99*max_con_violation_k:
            # lofi_prob[f'mu_{constraint}'] = np.minimum(1.5*old_mu[constraint], mu_max)
            lf_prob['mu'] = np.minimum(1.5*current_mu, self.options['mu_max'])
        else:
            lf_prob['mu'] = current_mu

        print(f"new mu: {lf_prob['mu']}")

        # # # compute m_k(p_inf)
        # # lofi_prob['opt_eff'] = 0.0
        # # lofi_prob.run_driver()
        # # lofi_prob['opt_eff'] = 1.0

        # for constraint in constraints.keys():
        #     constraint_violation_0 = np.abs(constraints0[constraint] - constraints[constraint]) / constraints[constraint]
        #     constraint_violation_k = np.abs(constraintsk[constraint] - constraints[constraint]) / constraints[constraint]
        #     print(f"constraint violation k: {constraint_violation_k}")
        #     feasible = True
        #     if constraint_violation_k < epsilon_feas:
        #         feasible = feasible and True
        #     else:
        #         feasible = feasible and False
        
        # print(f"feasible : {feasible}")

        # if feasible:
        #     for constraint in constraints.keys():
        #         print(f"old mu ({constraint}): {old_mu[constraint]}")
        #         print(f"new mu ({constraint}): {lofi_prob[f'mu_{constraint}']}")
        #     return
        
        # for constraint in constraints.keys():
        #     old_mu[constraint] = np.copy(lofi_prob[f'mu_{constraint}'])
        #     lofi_prob[f'mu_{constraint}'] = 1.0
        
        # # compute m_k(p_inf)
        # lofi_prob['opt_eff'] = 0.0
        # lofi_prob.run_driver()
        # lofi_prob['opt_eff'] = 1.0

        # for constraint in constraints.keys():
        #     constraint_violation_0 = np.abs(constraints0[constraint] - constraints[constraint]) / constraints[constraint]
        #     constraint_violation_k = np.abs(constraintsk[constraint] - constraints[constraint]) / constraints[constraint]
        #     constraint_violation_inf = np.abs(lofi_prob[constraint] - constraints[constraint]) / constraints[constraint]
        #     print(f"constraint violation 0: {constraint_violation_0}")
        #     print(f"constraint violation k: {constraint_violation_k}")
        #     print(f"constraint violation inf: {constraint_violation_inf}")
        #     if constraint_violation_k < epsilon_feas and 0.75*old_mu[constraint] > epsilon_feas:
        #         lofi_prob[f'mu_{constraint}'] = 0.75*old_mu[constraint]
        #     elif constraint_violation_k > epsilon_feas and 0.995*(constraint_violation_0 - constraint_violation_inf) > (constraint_violation_0 - constraint_violation_k):
        #         lofi_prob[f'mu_{constraint}'] = np.minimum(1.5*old_mu[constraint], mu_max)
        #     else:
        #         lofi_prob[f'mu_{constraint}'] = old_mu[constraint]

        # for constraint in constraints.keys():
        #     print(f"old mu ({constraint}): {old_mu[constraint]}")
        #     print(f"new mu ({constraint}): {lofi_prob[f'mu_{constraint}']}")