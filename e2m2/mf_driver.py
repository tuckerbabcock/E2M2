import copy

import numpy as np

from openmdao.core.driver import Driver


class MFDriver(Driver):
    """
    Base class for multi-fidelity optimizers
    """

    def __init__(self, **kwargs):
        """
        Initialize the MFDriver base class.
        """
        super().__init__(**kwargs)

        # The OpenMDAO problem that defines the low-fidelity problem
        self.low_fidelity_problem = None

        # The penalty parameter used when evaluating the merit function
        self.penalty_param = 1.0

    def _setup_driver(self, problem):
        """
        Prepare the driver for execution.

        This is the final thing to run during setup.

        Parameters
        ----------
        problem : <Problem>
            Pointer to the containing problem.
        """
        super()._setup_driver(problem)

    # def _estimate_lagrange_multipliers(self, obj, active_cons, dvs, totals):
    #     pass

    def _update_lf_design_point(self):
        unscaled_hf_dv_vals = self.get_design_var_values(driver_scaling=False)
        for dv, dv_val in unscaled_hf_dv_vals.items():
            self.low_fidelity_problem.set_val(f"{dv}_k", dv_val)
            self.low_fidelity_problem.set_val(f"delta_{dv}", 0.0)

    def _update_hf_design_point(self):
        lf_dv_vals = self.low_fidelity_problem.driver.get_design_var_values()
        hf_dvs = self._designvars
        hf_dv_vals = self.get_design_var_values()
        for dv in hf_dvs:
            scaler = hf_dvs[dv]['total_scaler'] or 1.0
            new_dv = hf_dv_vals[dv] + lf_dv_vals[f"delta_{dv}"] * scaler
            self.set_design_var(dv, new_dv)

    def _calibrate(self, dvs, responses, response_map, lf_prob, lf_totals, hf_prob, hf_totals):
        for response, meta in responses.items():
            response_name = meta['name']
            raw_lf_response_name = response_map[response_name][0]

            cal = getattr(lf_prob.model, f"{response_name}_cal")
            cal.options["f_lofi_x0"] = copy.deepcopy(lf_prob[raw_lf_response_name])
            cal.options["f_hifi_x0"] = copy.deepcopy(hf_prob[response_name])

            cal.options["g_lofi_x0"] = {
                dv: copy.deepcopy(lf_totals[raw_lf_response_name, f"delta_{dv}"]) for dv in dvs.keys()
            }
            cal.options["g_hifi_x0"] = {
                dv: copy.deepcopy(hf_totals[response, dv]) for dv in dvs.keys()
            }

            # print(f"{response}:\n")
            # print(cal.options["g_lofi_x0"])
            # print(cal.options["g_hifi_x0"])


    def _clean_hf_duals(self, hf_duals):
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
        # print(f"hf_duals: {hf_duals}")



    def _update_penalty(self, k, cons, con_violation):
        # print(f"cons: {cons}")
        # print(f"con violation: {con_violation}")
        if len(con_violation) == 0:
            return
        print(80*"*")
        print(f"update penalty!")
        print(80*"*")
        feas_tol = self.options['feas_tol']

        lf_prob = self.low_fidelity_problem
        current_mu = np.copy(lf_prob['mu'])

        lf_prob['mu'] = 1.0
        lf_prob["obj_scaler"] = 0.0
        lf_prob["obj_adder"] = 0.0

        lf_prob.driver.opt_settings['Print file'] = f"{lf_prob._name}_feasibility_opt_{k}.out"
        # lf_prob.driver.options['hist_file'] = f"{lf_prob._name}_feasibility_opt_{k}.db"
        lf_prob.run_driver(case_prefix=f'feasibility_opt_{k}')

        # lf_prob.model.list_inputs()
        # lf_prob.model.list_outputs()
        lf_prob["obj_scaler"] = self._objs[self.hf_obj_name]['total_scaler'] or 1.0
        lf_prob["obj_adder"] = self._objs[self.hf_obj_name]['total_adder'] or 0.0

        con_violation_inf = {}
        for con in con_violation:
            con_name = cons[con]['name']
            lf_con_name = self.options['response_map'][con_name][1]
            scaler = cons[con]['total_scaler'] or 1.0
            adder = cons[con]['total_adder'] or 0.0
            con_val = (lf_prob[lf_con_name] + adder) * scaler
            # print(f"{con} val: {con_val}")
            if cons[con]['equals'] is not None:
                con_target = cons[con]["equals"]
                # print(f"{con} target: {con_target}, value: {con_val}")
                if not np.isclose(con_val, con_target, atol=feas_tol, rtol=feas_tol):
                    # print(f"violates equality constraint!")
                    con_violation_inf[con] = con_val - con_target
                else:
                    con_violation_inf[con] = 0.0
            else:
                con_ub = cons[con].get("upper", np.inf)
                con_lb = cons[con].get("lower", -np.inf)
                # print(
                    # f"{con} lower bound: {con_lb}, upper bound: {con_ub}, value: {con_val}")
                if con_val > con_ub:
                    if not np.isclose(con_val, con_ub, atol=feas_tol, rtol=feas_tol):
                        # print(f"violates upper bound!")
                        con_violation_inf[con] = con_val - con_ub
                    else:
                        con_violation_inf[con] = 0.0
                elif con_val < con_lb:
                    if not np.isclose(con_val, con_lb, atol=feas_tol, rtol=feas_tol):
                        # print(f"violates lower bound!")
                        con_violation_inf[con] = con_val - con_lb
                    else:
                        con_violation_inf[con] = 0.0
                else:
                    con_violation_inf[con] = 0.0

        # print(f"con_violation_inf: {con_violation_inf}")

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

        # print(f"new mu: {lf_prob['mu']}")