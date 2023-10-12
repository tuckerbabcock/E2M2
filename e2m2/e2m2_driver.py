import numpy as np

import openmdao.api as om

from openmdao.core.driver import Driver

from .calibration import AdditiveCalibration, calibrate
from .error_est import ErrorEstimate
from .utils import estimate_lagrange_multipliers, get_active_constraints, constraint_violation, l1_merit_function, optimality

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
        # self.options.declare('dv_map',
        #                      default=None,
        #                      desc='')
        self.options.declare('opt_tol',
                             default=1e-6,
                             desc='The high-fidelity optimality tolerance used to determine convergence')
        self.options.declare('feas_tol',
                             default=1e-6,
                             desc='The high-fidelity feasibility tolerance used to determine convergence')
        self.options.declare('tau_abs',
                             default=1e-3,
                             desc='')
        self.options.declare('tau_rel',
                             default=1e-6,
                             desc='')
        self.options.declare('max_iter',
                             default=200,
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
        print(f"E2M2Driver::_setup_driver()!")
        super()._setup_driver(problem)

        if self.low_fidelity_problem is None:
            raise RuntimeError('Low fidelity problem is not set!')

        lf_model = self.low_fidelity_problem.model

        print(len(self._designvars.items()))
        for dv, dv_metadata in self._designvars.items():
            # print(dv, dv_metadata)
            lf_model.add_design_var(dv,
                                    lower=dv_metadata['lower'],
                                    upper=dv_metadata['upper'],
                                    ref=dv_metadata['ref'],
                                    ref0=dv_metadata['ref0'],
                                    scaler=dv_metadata['scaler'],
                                    adder=dv_metadata['adder'])

        print(self._responses)
        response_map = self.options['response_map']
        for response_metadata in self._responses.values():
            response_name = response_metadata['name']
            if response_name not in response_map:
                response_map[response_name] = (response_name, f"calibrated_{response_name}")
            
            calibrated_response_name = response_map[response_name][1]


            lf_model.add_subsystem(f"{response_name}_cal",
                                   AdditiveCalibration(inputs=self._designvars,
                                                       order=1),
                                   promotes_inputs=['*'],
                                   promotes_outputs=[('gamma', f'{response_name}_bias')])

            lf_model.add_subsystem(f"{response_name}_update",
                                   om.ExecComp(
                                       f"{calibrated_response_name} = {response_name} + {response_name}_bias"),
                                   promotes=['*'])
            if response_metadata['type'] == 'con':
                if response_metadata['equals'] is not None:
                    lf_model.add_subsystem(f"elastic_{response_name}_constraint",
                                   om.ExecComp(
                                       f"elastic_{response_name}_constraint = {calibrated_response_name} \
                                                              + {response_metadata['equals']} * ({response_name}_slack_1 - {response_name}_slack_2)"),
                                   promotes=['*'])
                    lf_model.add_constraint(f"elastic_{response_name}_constraint",
                                            lower=response_metadata['lower'],
                                            upper=response_metadata['upper'],
                                            ref=response_metadata['ref'],
                                            ref0=response_metadata['ref0'],
                                            scaler=response_metadata['scaler'],
                                            adder=response_metadata['adder'])
                    self.lf_con_names.append(f"elastic_{response_name}_constraint")
                else:
                    if response_metadata['lower'] is not None:
                        lf_model.add_subsystem(f"elastic_{response_name}_constraint_lb",
                                    om.ExecComp(
                                        f"elastic_{response_name}_constraint_lb = {calibrated_response_name} \
                                                                + {np.sign(response_metadata['lower']) * response_metadata['lower']} * {response_name}_slack_lb "),
                                    promotes=['*'])
                        lf_model.add_constraint(f"elastic_{response_name}_constraint_lb",
                                                lower=response_metadata['lower'],
                                                upper=response_metadata['upper'],
                                                ref=response_metadata['ref'],
                                                ref0=response_metadata['ref0'],
                                                scaler=response_metadata['scaler'],
                                                adder=response_metadata['adder'])
                        self.lf_con_names.append(f"elastic_{response_name}_constraint_lb")

                    if response_metadata['upper'] is not None:
                        lf_model.add_subsystem(f"elastic_{response_name}_constraint_ub",
                                    om.ExecComp(
                                        f"elastic_{response_name}_constraint_ub = {calibrated_response_name} \
                                                                + {np.sign(response_metadata['upper']) * response_metadata['upper']} * {response_name}_slack_ub"),
                                    promotes=['*'])
                        lf_model.add_constraint(f"elastic_{response_name}_constraint_ub",
                                                lower=response_metadata['lower'],
                                                upper=response_metadata['upper'],
                                                ref=response_metadata['ref'],
                                                ref0=response_metadata['ref0'],
                                                scaler=response_metadata['scaler'],
                                                adder=response_metadata['adder'])
                        self.lf_con_names.append(f"elastic_{response_name}_constraint_ub")

            lf_model.add_subsystem(f"{response_name}_error_est",
                                   ErrorEstimate(inputs=self._designvars,
                                                 order=1),
                                   promotes_inputs=['*'],
                                   promotes_outputs=[('error_est', f'{response_name}_abs_error_est'),
                                                     ('relative_error_est', f'{response_name}_rel_error_est')])

            lf_model.add_subsystem(f"{response_name}_error_con",
                                   om.ExecComp(
                                       f"{response_name}_abs_error_con = {response_name}_abs_error_est / tau_{response_name}_abs"),
                                   promotes=['*'])
            lf_model.add_subsystem(f"{response_name}_rel_error_con",
                                   om.ExecComp(
                                       f"{response_name}_rel_error_con = {response_name}_rel_error_est / tau_{response_name}_rel"),
                                   promotes=['*'])

            lf_model.add_constraint(
                f"{response_name}_abs_error_con", upper=1.0, ref=1.0, ref0=0.0)
            lf_model.add_constraint(
                f"{response_name}_rel_error_con", upper=1.0, ref=1.0, ref0=0.0)

            if response_metadata['type'] == 'obj':
                lf_model.add_objective(calibrated_response_name,
                                       ref=response_metadata['ref'],
                                       ref0=response_metadata['ref0'],
                                       scaler=response_metadata['scaler'],
                                       adder=response_metadata['adder'])
                self.lf_obj_name = calibrated_response_name

        self.low_fidelity_problem.setup()
        self.low_fidelity_problem.final_setup()

    def run(self):
        """
        Optimize the problem using E2M2 optimizer.

        Returns
        -------
        bool
            Failure flag; True if failed to converge, False is successful.
        """
        pass

        hf_prob = self._problem()
        lf_prob = self.low_fidelity_problem
        lf_model = self.low_fidelity_problem.model

        hf_prob.run_model()

        for k in range(1, self.options['max_iter']):

            lf_prob.run_model()

            calibrate(lf_prob,
                      hf_prob,
                      self.options['response_map'],
                      self._designvars.keys(),
                      include_error_est=True,
                      direct_hessian_diff=True,
                      sr1_hessian_diff=True)

            # lf_prob.driver.opt_settings['Print file'] = f"{opt_name}_{cal_order}_{k}.out"
            # lf_prob.driver.hist_file = f"{opt_name}_{cal_order}_{k}.db"

            # lf_prob.run_driver(case_prefix=f'sub_opt{k}')
            lf_prob.run_driver()

            lf_dvs = lf_model.get_design_vars(recurse=True, get_sizes=False, use_prom_ivc=False)
            lf_responses = lf_model.get_responses(recurse=True, get_sizes=False, use_prom_ivc=False)

            lf_cons = {key: value for key, value in lf_responses.items() if key in self.lf_con_names}
            active_lf_constraints = get_active_constraints(lf_prob,
                                                           lf_cons,
                                                           lf_dvs,
                                                           self.options['feas_tol'])

            lf_totals = lf_prob.compute_totals([self.lf_obj_name, *self.lf_con_names],
                                               [*lf_dvs.keys()],
                                               driver_scaling=True)
            
            est_multipliers = estimate_lagrange_multipliers(lf_prob,
                                                            self.lf_obj_name,
                                                            active_lf_constraints,
                                                            lf_totals,
                                                            lf_dvs,
                                                            # unscaled=True)
                                                            unscaled=False)
            # x.append(dict())
            # for key in x[k-1].keys():
            #     x[k][key] = copy.deepcopy(lf_prob[key])
            #     print(f"Setting hf value {key} to {x[k][key]}")
            #     hf_prob[key] = x[k][key]
            
            for dv in lf_dvs.keys():
                hf_prob[dv][:] = lf_prob[dv]

            hf_prob.run_model()

            # hf_multipliers = copy.deepcopy(est_multipliers)
            # hf_multipliers['power_out'] = hf_multipliers.pop('power_out_con')
            # hf_multipliers.pop('slack_power_out_1', None)
            # hf_multipliers.pop('slack_power_out_2', None)

            # new_penalty_parameter = 2.0 * \
            #     abs(max(hf_multipliers.values(), key=abs))
            # if k == 1:
            #     penalty_parameter = new_penalty_parameter

            # penalty_parameter = max(penalty_parameter, new_penalty_parameter)
            # merit_function = l1_merit_function(
            #     hf_prob, objective, constraints, penalty_parameter, feas_tol, True)
            # # max_constraint_error = np.maximum(np.absolute(const_error.values()))

            # _, const_error = constraint_violation(hf_prob, constraints, feas_tol)

            # if len(const_error) > 0:
            #     max_constraint_error = abs(max(const_error.values(), key=abs))
            # else:
            #     max_constraint_error = 0.0

            # active_constraints = get_active_constraints(
            #     hf_prob, constraints, des_vars, feas_tol)

            # hf_totals = hf_prob.compute_totals(
            #     [objective, *constraints.keys()], [*des_vars.keys()], driver_scaling=True)
            # print(f"hf_totals: {hf_totals}")
            # print(f"hf_multipliers: {hf_multipliers}")
            # print(f"hf active_constraints: {active_constraints}")
            # # optim = optimality(hf_totals, objective, active_constraints, des_vars, hf_multipliers, maximize=True)
            # optim = optimality(hf_totals, objective,
            #                 active_constraints, des_vars, hf_multipliers)
            # print(f"{80*'#'}")
            # print(
            #     f"Hifi merit function value: {merit_function}, optimality: {optim}, max constraint violation: {max_constraint_error}")
            # print(f"{80*'#'}")

            # subopt_tol = lf_prob.driver.opt_settings['Major optimality tolerance']
            # subfeas_tol = lf_prob.driver.opt_settings['Major feasibility tolerance']
            # lf_prob.driver.opt_settings['Major optimality tolerance'] = 1e-4
            # lf_prob.driver.opt_settings['Major feasibility tolerance'] = 1e-4
            # update_penalty(lf_prob, constraint_targets,
            #             constraints0, constraintsk, feas_tol, mu_max=10)
            # lf_prob.driver.opt_settings['Major optimality tolerance'] = subopt_tol
            # lf_prob.driver.opt_settings['Major feasibility tolerance'] = subfeas_tol
