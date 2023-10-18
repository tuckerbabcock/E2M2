import copy

import numpy as np

import openmdao.api as om

from openmdao.core.driver import Driver

from .calibration import AdditiveCalibration, calibrate
from .error_est import ErrorEstimate, update_error_ests
from .new_design import NewDesign
from .utils import \
    estimate_lagrange_multipliers2, \
    get_active_constraints2, \
    constraint_violation2, \
    l1_merit_function2, \
    optimality2

CITATIONS = """

"""


class E2M2Driver(Driver):
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

        self.penalty_param = 1.0

        self.lf_obj_name = None
        self.hf_obj_name = None

        self._actually_setup = False

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
        self.options.declare('tau',
                             default=1e-1,
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
        print(f"E2M2Driver::_setup_driver()!")
        super()._setup_driver(problem)

        if self.low_fidelity_problem is None:
            raise RuntimeError('Low fidelity problem is not set!')

        if not self._actually_setup:
            self._setup_lf_prob()
            self._actually_setup = True

    def _setup_lf_prob(self):
        actual_lf_model = self.low_fidelity_problem.model

        lf_model = om.Group()
        lf_model.add_subsystem("new_design",
                               NewDesign(design_vars=self._designvars),
                               promotes=['*'])

        lf_model.add_subsystem("lf_model",
                               actual_lf_model,
                               promotes=['*'])

        print(len(self._designvars.items()))

        for dv, meta in self._designvars.items():
            print(dv, meta)
            lf_model.add_design_var(f"delta_{dv}")

            scaler = meta['total_scaler'] or 1.0
            adder = meta['total_adder'] or 0.0
            dv_lb = meta['lower'] / scaler - adder
            dv_ub = meta['upper'] / scaler - adder

            print(f"{dv} lb: {dv_lb}, ub: {dv_ub}")
            lf_model.add_constraint(dv,
                                    lower=dv_lb,
                                    upper=dv_ub,
                                    scaler=scaler,
                                    adder=adder,
                                    # ref=meta['ref'] or 1.0,
                                    # ref0=meta['ref0'] or 0.0,
                                    linear=True)
            lf_model.set_input_defaults(f'delta_{dv}', val=0)

        print(self._responses)
        response_map = self.options['response_map']
        slacks = []
        for response, meta in self._responses.items():
            response_name = meta['name']
            if response_name not in response_map:
                response_map[response_name] = (
                    response_name, f"{response_name}_hat")

            calibrated_response_name = response_map[response_name][1]

            lf_model.add_subsystem(f"{response_name}_cal",
                                   AdditiveCalibration(inputs=self._designvars,
                                                       order=1),
                                   promotes_inputs=['*'],
                                   promotes_outputs=[('gamma', f'{response_name}_bias')])

            lf_model.add_subsystem(f"{response_name}_hat",
                                   om.ExecComp(
                                       f"{calibrated_response_name} = {response_name} + {response_name}_bias"),
                                   promotes=['*'])
            
            lf_model.add_subsystem(f"{response_name}_error_est",
                                   ErrorEstimate(dvs=self._designvars),
                                   promotes_inputs=['*'],
                                   promotes_outputs=[('error_est', f'{response_name}_error_est')])
            lf_model.add_subsystem(f"{response_name}_error_con",
                                   om.ExecComp(f"{response_name}_error_con = {response_name}_error_est - tau_{response_name}"),
                                   promotes=['*'])
            lf_model.add_constraint(f"{response_name}_error_con", upper=0.0)

            if meta['type'] == 'con':
                scaler = meta['total_scaler'] or 1.0
                adder = meta['total_adder'] or 0.0
                if meta['equals'] is not None:
                    con_target = meta['equals'] / scaler - adder
                    lf_model.add_subsystem(f"elastic_{response_name}_con",
                                           om.ExecComp(
                                               f"elastic_{response_name}_con = {calibrated_response_name} \
                                                              + {con_target} * ({response_name}_slack_1 - {response_name}_slack_2)"),
                                           promotes=['*'])
                    lf_model.add_design_var(
                        f"{response_name}_slack_1", lower=0)
                    lf_model.add_design_var(
                        f"{response_name}_slack_2", lower=0)
                    lf_model.add_constraint(f"elastic_{response_name}_con",
                                            equals=con_target,
                                            scaler=scaler,
                                            adder=adder)
                    slacks.append(f"{response_name}_slack_1")
                    slacks.append(f"{response_name}_slack_2")
                else:
                    if not np.isclose(meta['lower'], -1e30):
                        con_lb = meta['lower'] / scaler - adder
                        lf_model.add_subsystem(f"elastic_{response_name}_con_lb",
                                               om.ExecComp(
                                                   f"elastic_{response_name}_con_lb = {calibrated_response_name} \
                                                                + {con_lb} * {response_name}_lb_slack"),
                                               promotes=['*'])
                        lf_model.add_design_var(
                            f"{response_name}_lb_slack", lower=0)
                        lf_model.add_constraint(f"elastic_{response_name}_con_lb",
                                                lower=con_lb,
                                                scaler=scaler,
                                                adder=adder)
                        slacks.append(f"{response_name}_lb_slack")

                    if not np.isclose(meta['upper'], 1e30):
                        con_ub = meta['upper'] / scaler - adder
                        lf_model.add_subsystem(f"elastic_{response_name}_con_ub",
                                               om.ExecComp(
                                                   f"elastic_{response_name}_con_ub = {calibrated_response_name} \
                                                                - {con_ub} * {response_name}_ub_slack"),
                                               promotes=['*'])
                        lf_model.add_design_var(
                            f"{response_name}_ub_slack", lower=0)
                        lf_model.add_constraint(f"elastic_{response_name}_con_ub",
                                                upper=con_ub,
                                                scaler=scaler,
                                                adder=adder)
                        slacks.append(f"{response_name}_ub_slack")

            if meta['type'] == 'obj':
                self.hf_obj_name = response
                self.lf_obj_name = calibrated_response_name

        obj_string = f"obj = obj_scaler * ({self.lf_obj_name} + obj_adder)"
        if len(slacks) > 0:
            obj_string += " + mu * ("
            for i, slack in enumerate(slacks):
                if i != (len(slacks) - 1):
                    obj_string += f"{slack} + "
                else:
                    obj_string += f"{slack})"

        print(f"slack string: {obj_string}")
        lf_model.add_subsystem("objective",
                               om.ExecComp(obj_string,
                                           obj={
                                               'val': 1.0,
                                               'units': self._responses[self.hf_obj_name]['units']
                                           },
                                           obj_scaler={
                                               'val': self._responses[self.hf_obj_name]['total_scaler'] or 1.0
                                           },
                                           obj_adder={
                                               'val': self._responses[self.hf_obj_name]['total_adder'] or 0.0
                                           }
                                           ),
                               promotes=['*'])
        lf_model.add_objective('obj', ref=1, ref0=0)

        self.low_fidelity_problem.model = lf_model
        self.low_fidelity_problem.setup()

    def run(self):
        """
        Optimize the problem using E2M2 optimizer.

        Returns
        -------
        bool
            Failure flag; True if failed to converge, False is successful.
        """
        opt_tol = self.options['opt_tol']
        feas_tol = self.options['feas_tol']

        hf_prob = self._problem()
        lf_prob = self.low_fidelity_problem

        lf_prob["obj_scaler"] = self._objs[self.hf_obj_name]['total_scaler'] or 1.0
        lf_prob["obj_adder"] = self._objs[self.hf_obj_name]['total_adder'] or 0.0

        # Calculate initial merit function
        hf_prob.run_model()
        old_merit_function = l1_merit_function2(
            self, self.penalty_param, feas_tol)

        con_vals = self.get_constraint_values()
        old_con_violation = constraint_violation2(
            self, self._cons, con_vals, feas_tol)

        for k in range(1, self.options['max_iter']+1):

            # Update LF reference point
            unscaled_hf_dv_vals = self.get_design_var_values(
                driver_scaling=False)
            for dv, dv_val in unscaled_hf_dv_vals.items():
                lf_prob.set_val(f"{dv}_k", dv_val)
                lf_prob.set_val(f"delta_{dv}", 0.0)

            # Update error estimate bounds
            for meta in self._responses.values():
                response_name = meta['name']
                tau = self.options['tau']
                # lf_prob.set_val(f'tau_{response_name}', min(tau, tau*hf_prob[response_name]))
                lf_prob.set_val(f'tau_{response_name}', tau)

            lf_prob.run_model()

            calibrate(lf_prob,
                      hf_prob,
                      self.options['response_map'],
                      self._designvars)

            # Optimize calibrated LF model
            lf_prob.driver.opt_settings['Print file'] = f"{lf_prob._name}_{k}.out"
            lf_prob.driver.options['hist_file'] = f"{lf_prob._name}_{k}.db"
            lf_prob.run_driver(case_prefix=f'sub_opt_{k}')

            # Evaluated predicted merit function at new point
            lf_merit_function = l1_merit_function2(
                lf_prob.driver, self.penalty_param, feas_tol)
            print(f"lf merit: {lf_merit_function}")

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

            lf_prob.model.list_inputs()
            lf_prob.model.list_outputs()
            hf_prob.model.list_inputs()
            hf_prob.model.list_outputs()

            # Evaluate merit function and feasibility at new design point
            merit_function = l1_merit_function2(
                self, self.penalty_param, feas_tol)
            print(f"hf merit: {merit_function}")
            con_vals = self.get_constraint_values()
            con_violation = constraint_violation2(
                self, self._cons, con_vals, feas_tol)

            # Update trust radius based on actual and predicted behavior
            ared = old_merit_function - merit_function
            pred = old_merit_function - lf_merit_function
            print(f"ared: {ared}")
            print(f"pred: {pred}")
            r = ared / pred
            print(f"r: {r}")

            if len(con_violation) > 0:
                feasibility_improvement = max([np.abs(old_violation) - np.abs(violation)
                                              for old_violation, violation in zip(old_con_violation.values(), con_violation.values())])
            else:
                feasibility_improvement = 0.0
            print(f"feasibility_improvement: {feasibility_improvement}")

            lf_cons = lf_prob.driver._cons
            lf_dvs = lf_prob.driver._designvars
            active_lf_cons = get_active_constraints2(
                lf_cons, lf_con_vals, lf_dvs, lf_dv_vals, feas_tol)
            print(f"active lf cons: {active_lf_cons}")

            f_error_est = lf_prob['f_error_est']
            true_f_error = lf_prob['f_hat'] - hf_prob['f']
            print(f"Estimated f error: {f_error_est}")
            print(f"True f error: {true_f_error}")

            # TODO: maybe manually apply scaling here based on HF constraint scalers
            lf_totals = lf_prob.compute_totals([*self.options['response_map'][self._objs[self.hf_obj_name]['name']], *active_lf_cons],
                                               # lf_totals = lf_prob.compute_totals(["obj", *active_lf_cons],
                                               [*lf_dvs.keys()],
                                               driver_scaling=False)
            print(f"lf_totals: {lf_totals}")
            lf_duals = estimate_lagrange_multipliers2(self.lf_obj_name,
                                                      # lf_duals = estimate_lagrange_multipliers2("obj",
                                                      active_lf_cons,
                                                      lf_dvs,
                                                      lf_totals)

            print(f"lf_duals: {lf_duals}")
            hf_duals = copy.deepcopy(lf_duals)

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

            if len(hf_duals) > 0:
                self.penalty_param = 2.0 * abs(max(hf_duals.values(), key=abs))

            if len(con_violation) > 0:
                max_constraint_violation = abs(
                    max(con_violation.values(), key=abs))
            else:
                max_constraint_violation = 0.0

            hf_cons = self._cons
            hf_con_vals = self.get_constraint_values()
            hf_dvs = self._designvars
            hf_dv_vals = self.get_design_var_values()
            active_hf_cons = get_active_constraints2(
                hf_cons, hf_con_vals, hf_dvs, hf_dv_vals, feas_tol)

            print(f"active hf cons: {active_hf_cons}")
            hf_totals = hf_prob.compute_totals([self.hf_obj_name, *active_hf_cons],
                                               [*hf_dvs.keys()],
                                               driver_scaling=False)
            print(f"hf_totals: {hf_totals}")

            update_error_ests(self._designvars,
                              self._responses,
                              self.options['response_map'],
                              lf_prob,
                              lf_totals,
                              hf_totals)
                    #   include_error_est=True,
                    #   direct_hessian_diff=True,
                    #   sr1_hessian_diff=True)

            optim = optimality2(self.hf_obj_name, active_hf_cons,
                                hf_dvs, hf_duals, hf_totals)
            print(f"{80*'#'}")
            print(
                f"{k}: Merit function value: {merit_function}, optimality: {optim}, max constraint violation: {max_constraint_violation}")
            print(f"{80*'#'}")

            if optim < opt_tol and max_constraint_violation < feas_tol:
                break

            self._update_penalty(k, self._cons, con_violation)
            old_merit_function = merit_function
            old_con_violation = con_violation

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
                print(
                    f"{con} lower bound: {con_lb}, upper bound: {con_ub}, value: {con_val}")
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
        if max_con_violation_inf < 0.99*max_con_violation_k:
            # lofi_prob[f'mu_{constraint}'] = np.minimum(1.5*old_mu[constraint], mu_max)
            lf_prob['mu'] = np.minimum(1.5*current_mu, self.options['mu_max'])
        else:
            lf_prob['mu'] = current_mu

        print(f"new mu: {lf_prob['mu']}")