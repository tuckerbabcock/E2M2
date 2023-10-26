import copy

import numpy as np

import openmdao.api as om

from openmdao.core.driver import Driver

from .trust_region import TrustRegion
from .calibration import AdditiveCalibration, calibrate
from .new_design import NewDesign
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

        if not self._actually_setup:
            self._setup_lf_prob(problem)
            self._actually_setup = True

        # obj_vals = self.get_objective_values()
        # for obj in obj_vals:
        #     print(f"hf_obj_name: {obj}")
        #     self.hf_obj_name = obj
        #     break

    def _setup_lf_prob(self, problem):
        actual_lf_model = self.low_fidelity_problem.model

        lf_model = om.Group()

        model_metadata = problem.model.get_io_metadata()
        print(model_metadata)
        print(f"designvars: {self._designvars}")
        lf_model.add_subsystem("new_design",
                               NewDesign(metadata=model_metadata,
                                         design_vars=self._designvars),
                               promotes=['*'])

        lf_model.add_subsystem("lf_model",
                               actual_lf_model,
                               promotes=['*'])

        print(len(self._designvars.items()))

        for dv, meta in self._designvars.items():
            print(dv, meta)
            dv_name = meta['name']
            lf_model.add_design_var(f"delta_{dv_name}")

            scaler = meta['total_scaler'] or 1.0
            adder = meta['total_adder'] or 0.0
            dv_lb = meta['lower'] / scaler - adder
            dv_ub = meta['upper'] / scaler - adder

            print(f"{dv} lb: {dv_lb}, ub: {dv_ub}")
            lf_model.add_constraint(dv_name,
                                    lower=dv_lb,
                                    upper=dv_ub,
                                    scaler=scaler,
                                    adder=adder,
                                    # ref=meta['ref'] or 1.0,
                                    # ref0=meta['ref0'] or 0.0,
                                    linear=True)
            # lf_model.set_input_defaults(f'delta_{dv}', val=np.array([0]))

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
                                   AdditiveCalibration(metadata=model_metadata,
                                                       inputs=self._designvars,
                                                       order=1),
                                   promotes_inputs=['*'],
                                   promotes_outputs=[('gamma', f'{response_name}_bias')])

            lf_model.add_subsystem(f"{response_name}_hat",
                                   om.ExecComp(
                                       f"{calibrated_response_name} = {response_name} + {response_name}_bias"),
                                   promotes=['*'])

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

        lf_model.add_subsystem("trust_region",
                               TrustRegion(
                                   metadata=model_metadata,
                                   #    dvs=[f"delta_{dv}" for dv in self._designvars.keys()]),
                                   dvs=self._designvars),
                               promotes_inputs=[
                                   f"delta_{dv}" for dv in self._designvars.keys()],
                               promotes_outputs=['step_norm'])

        lf_model.add_subsystem("trust_radius_con",
                               om.ExecComp(
                                   "trust_radius_con = step_norm - delta**2"),
                               promotes=['*'])
        lf_model.add_constraint('trust_radius_con', upper=0.0)

        self.low_fidelity_problem.model = lf_model
        self.low_fidelity_problem.setup()

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

            # Update trust radius
            lf_prob.set_val('delta', self.delta)

            lf_prob.run_model()

            # Calibrate LF model at current design point
            calibrate(lf_prob,
                      hf_prob,
                      self.options['response_map'],
                      #   self.get_design_var_values())
                      self._designvars)

            # Optimize calibrated LF model
            lf_prob.driver.opt_settings['Print file'] = f"{lf_prob._name}_{k}.out"
            # lf_prob.driver.options['hist_file'] = f"{lf_prob._name}_{k}.db"
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
            self._update_trust_radius(lf_prob['step_norm'], r)

            if len(con_violation) > 0:
                feasibility_improvement = max([np.abs(old_violation) - np.abs(violation)
                                              for old_violation, violation in zip(old_con_violation.values(), con_violation.values())])
            else:
                feasibility_improvement = 0
            print(f"feasibility_improvement: {feasibility_improvement}")

            lf_cons = lf_prob.driver._cons
            lf_dvs = lf_prob.driver._designvars
            active_lf_cons = get_active_constraints2(
                lf_cons, lf_con_vals, lf_dvs, lf_dv_vals, feas_tol)
            print(f"active lf cons: {active_lf_cons}")

            # Estimate LF lagrange multipliers
            # obj_vals = self.get_objective_values()
            # for obj in obj_vals:
            #     hf_obj_name = obj
            #     break

            # TODO: maybe manually apply scaling here based on HF constraint scalers
            lf_totals = lf_prob.compute_totals([self.lf_obj_name, *active_lf_cons],
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
                self.penalty_param = 2.0 * \
                    abs(max(max(duals, key=abs)
                        for duals in hf_duals.values()))

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
            # hf_totals = hf_prob.compute_totals([self.hf_obj_name, *active_hf_cons],
            hf_totals = hf_prob.compute_totals([*self._responses.keys()],
                                               [*hf_dvs.keys()],
                                               driver_scaling=False)
            print(f"hf_totals: {hf_totals}")

            optim = optimality2(self._responses, self.hf_obj_name, active_hf_cons,
                                hf_dvs, hf_duals, hf_totals)
            print(f"{80*'#'}")
            print(
                f"{k}: Merit function value: {merit_function}, optimality: {optim}, max constraint violation: {max_constraint_violation}")
            print(f"{80*'#'}")

            if optim < opt_tol and max_constraint_violation < feas_tol:
                self.k = k
                break
            self.k = k

            # If the merit function increases AND there is no reduction in infeasibility, reject the step
            if ared <= 0 and r <= 0 and feasibility_improvement <= 0:
                hf_dvs = self._designvars
                hf_dv_vals = self.get_design_var_values()
                for dv in hf_dvs:
                    scaler = hf_dvs[dv]['total_scaler'] or 1.0
                    new_dv = hf_dv_vals[dv] - \
                        lf_dv_vals[f"delta_{dv}"] * scaler
                    self.set_design_var(dv, new_dv)

                print(f"Rejected step!")
                # print(f"{80*'#'}")
                # print(
                #     f"{k}: Merit function value: {old_merit_function}, optimality: {optim}, max constraint violation: {max_constraint_violation} (r)")
                # print(f"{80*'#'}")
                continue

            self._update_penalty(k, self._cons, con_violation)
            old_merit_function = merit_function
            old_con_violation = con_violation

    def _update_trust_radius(self, step_norm, r):
        print(f"old delta: {self.delta}")
        if r < self.options['r1']:
            # self.delta = self.options['c1'] * np.sqrt(step_norm)
            self.delta = self.options['c1'] * self.delta
        elif r > self.options['r2']:
            self.delta = min(
                self.options['c2']*self.delta, self.options['delta_star'])
        print(f"new delta: {self.delta}")

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
        # lf_prob.driver.options['hist_file'] = f"{lf_prob._name}_feasibility_opt_{k}.db"
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
