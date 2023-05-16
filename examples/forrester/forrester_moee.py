import copy

import numpy as np

import openmdao.api as om

from e2m2 import AdditiveCalibration, ErrorEstimate, calibrate
from e2m2 import estimate_lagrange_multipliers, \
                 get_active_constraints, \
                 constraint_violation, \
                 l1_merit_function, \
                 optimality

from model import ForresterLoFiBad, ForresterLoFiGood, ForresterHiFi

def optimize(lofi_prob,
             hifi_prob,
             x0,
             des_vars,
             objective,
             constraints,
             lofi_objective,
             lofi_constraints,
             tau_abs,
             tau_rel,
             max_iter,
             feas_tol=1e-6,
             opt_tol=1e-6):

    x = [x0]

    f_hi = np.zeros(max_iter+1)
    f_hat = np.zeros(max_iter+1)

    taus_abs = [tau_abs for _ in range(max_iter)]
    taus_abs[0] = {tau:1e16 for tau in tau_abs}
    taus_rel = [tau_rel for _ in range(max_iter)]
    taus_rel[0] = {tau:1e16 for tau in tau_rel}


    for key, value in x0.items():
        hifi_prob[key] = value
    hifi_prob.run_model()

    f_hi[0] = copy.deepcopy(hifi_prob['f'])

    for key, value in taus_abs[0].items():
        lofi_prob[key] = value
    for key, value in taus_rel[0].items():
        lofi_prob[key] = value

    for k in range(1, max_iter):

        for key, value in x[k-1].items():
            lofi_prob[key] = value
            hifi_prob[key] = value

        lofi_prob.run_model()
        # lofi_prob.check_totals()

        calibrate(lofi_prob,
                  hifi_prob,
                  ['f'],
                  x[k-1],
                  include_error_est=True,
                  direct_hessian_diff=True,
                  sr1_hessian_diff=True)

        # lofi_prob.model.list_inputs()
        # lofi_prob.model.list_outputs()

        lofi_prob.driver.opt_settings['Print file'] = f"{opt_name}_{cal_order}_{k}.out"
        lofi_prob.driver.hist_file = f"{opt_name}_{cal_order}_{k}.db"

        # lofi_prob.run_driver(case_prefix=f'sub_opt{k}')
        lofi_prob.run_driver()

        # multipliers = lofi_prob.driver.pyopt_solution.lambdaStar
        # active_constraints = get_active_constraints(lofi_prob.driver.pyopt_solution)
        # active_lofi_constraints = get_active_constraints(lofi_prob, lofi_constraints, lofi_des_vars, feas_tol)
        active_lofi_constraints = get_active_constraints(lofi_prob, lofi_constraints, des_vars, feas_tol)
        # print(f"active constraints: {active_lofi_constraints}")
        # lofi_totals = lofi_prob.compute_totals([lofi_objective, *lofi_constraints.keys()], [*lofi_des_vars.keys()], driver_scaling=True)
        lofi_totals = lofi_prob.compute_totals([lofi_objective, *lofi_constraints.keys()], [*des_vars.keys()], driver_scaling=True)
        est_multipliers = estimate_lagrange_multipliers(lofi_prob,
                                                        lofi_objective,
                                                        active_lofi_constraints,
                                                        lofi_totals,
                                                        # lofi_des_vars,
                                                        des_vars,
                                                        # unscaled=True)
                                                        unscaled=False)

        # est_multipliers = unscale_lagrange_multipliers(lofi_prob,
        #                                                lofi_objective,
        #                                                active_lofi_constraints,
        #                                                copy.deepcopy(scaled_multipliers))

        # print(f"active constraints: {active_lofi_constraints}")
        # print(f"active constraints2: {active_constraints2}")
        # print(f"Estimated Lagrange Multipliers: {est_multipliers}")
        # print(f"Lagrange Multipliers: {multipliers}")

        f_error_est = lofi_prob['f_error_est']
        f_rel_error_est = lofi_prob['f_rel_error_est']
        
        if f_error_est >= lofi_prob['tau_f_abs'] or \
           f_rel_error_est >= lofi_prob['tau_f_rel']:
            const_active = True
        else:
            const_active = False

        x.append(dict())
        for key in x[k-1].keys():
            x[k][key] = copy.deepcopy(lofi_prob[key])
            # print(f"Setting hifi value {key} to {x[k][key]}")
            hifi_prob[key] = x[k][key]

        # print(f"design history:\n{x}")

        hifi_prob.run_model()

        f_hat[k] = copy.deepcopy(lofi_prob['f'])
        f_hi[k] = copy.deepcopy(hifi_prob['f'])

        true_f_error = np.abs(f_hat[k] - f_hi[k])
        true_f_rel_error = true_f_error / f_hi[k]
        # print(f"Hifi f: {f_hi[k]}")
        # print(f"cal f: {f_hat[k]}")
        # print(f"Estimated f error: {f_error_est}")
        # print(f"True f error: {true_f_error}")
        # print(f"Estimated f relative error: {f_rel_error_est}")
        # print(f"True f relative error: {true_f_rel_error}")

        # lofi_prob.model.list_outputs()
        # hifi_prob.model.list_outputs()

        if true_f_error < tau_abs['tau_f_abs'] and \
           true_f_rel_error < tau_rel['tau_f_rel']:
            error_acceptable = True
        else:
            error_acceptable = False

        hifi_multipliers = copy.deepcopy(est_multipliers)
        # hifi_multipliers['power_out'] = hifi_multipliers.pop('power_out_con')
        # hifi_multipliers.pop('slack_power_out_1', None)
        # hifi_multipliers.pop('slack_power_out_2', None)

        # new_penalty_parameter = 2.0 * abs(max(hifi_multipliers.values(), key=abs))
        # if k == 1:
        #     penalty_parameter = new_penalty_parameter

        # penalty_parameter = max(penalty_parameter, new_penalty_parameter)
        penalty_parameter = 1.0
        merit_function = l1_merit_function(hifi_prob, objective, constraints, penalty_parameter, feas_tol, True)
        # max_constraint_error = np.maximum(np.absolute(const_error.values()))

        _, const_error = constraint_violation(hifi_prob, constraints, feas_tol)

        if len(const_error) > 0:
            max_constraint_error = abs(max(const_error.values(), key=abs))
        else:
            max_constraint_error = 0.0

        active_constraints = get_active_constraints(hifi_prob, constraints, des_vars, feas_tol)

        lofi_totals = lofi_prob.compute_totals([objective, *constraints.keys()], [*des_vars.keys()], driver_scaling=True)
        # print(f"lofi_totals: {lofi_totals}")

        hifi_totals = hifi_prob.compute_totals([objective, *constraints.keys()], [*des_vars.keys()], driver_scaling=True)

        ### Need to convert between lofi multipliers and hifi multipliers (different keys for efficiency and power out)
        ### Need to change power out constraint such that the multiplier is consistent with the hifi power out
        ### Efficiency and current should be okay, same with design variables
        ### Updates to paper need to include least squares lagrange multipliers estimates
        ### Still need to address step acceptance/rejection

        # print(f"hifi_totals: {hifi_totals}")
        # print(f"hifi_multipliers: {hifi_multipliers}")
        # print(f"hifi active_constraints: {active_constraints}")
        optim = optimality(hifi_totals, objective, active_constraints, des_vars, hifi_multipliers)
        print(f"{80*'#'}")
        print(f"Hifi merit function value: {merit_function}, optimality: {optim}, max constraint violation: {max_constraint_error}")
        print(f"{80*'#'}")

        # lofi_prob.model.list_outputs()
        # hifi_prob.model.list_outputs()

        if optim < opt_tol and max_constraint_error < feas_tol:
            # print(f"Break! k = {k}")
            break

        # if not const_active and error_acceptable and k != 1:
            # print(f"Break! k = {k}")
            # break

        # update tolerances
        for key, value in taus_abs[k].items():
            lofi_prob[key] = value
        for key, value in taus_rel[k].items():
            lofi_prob[key] = value
    
    return f_hi, f_hat, x


if __name__ == "__main__":
    cal_order = 1

    # good = True
    good = False
    if good:
        opt_name = "forrester_moee_good"
    else:
        opt_name = "forrester_moee_bad"


    lofi_prob = om.Problem(name=f"{opt_name}_lofi")

    lofi_prob.model.add_subsystem("f_cal",
                                  AdditiveCalibration(inputs=['x'], order=cal_order),
                                  promotes_inputs=["*"],
                                  promotes_outputs=[("gamma", "f_lofi_bias")])

    if good:
        lofi_prob.model.add_subsystem('lofi',
                                    ForresterLoFiGood(),
                                    promotes_inputs=['*'],
                                    promotes_outputs=['*'])
    else:
        lofi_prob.model.add_subsystem('lofi',
                                    ForresterLoFiBad(),
                                    promotes_inputs=['*'],
                                    promotes_outputs=['*'])

    lofi_prob.model.add_subsystem("f_error_est",
                                  ErrorEstimate(inputs=['x'], order=cal_order),
                                  promotes_inputs=["*"],
                                  promotes_outputs=[("error_est", "f_error_est"), ("relative_error_est", "f_rel_error_est")])

    lofi_prob.model.add_subsystem("f_error_con",
                                  om.ExecComp("f_abs_error_con = f_error_est / tau_f_abs"),
                                  promotes=['*'])
    lofi_prob.model.add_subsystem("f_rel_error_con",
                                  om.ExecComp("f_rel_error_con = f_rel_error_est / tau_f_rel"),
                                  promotes=['*'])

    lofi_prob.model.add_constraint("f_abs_error_con", upper=1.0, ref=1.0, ref0=0.0)
    lofi_prob.model.add_constraint("f_rel_error_con", upper=1.0, ref=1.0, ref0=0.0)

    # Set up optimization variables
    lofi_prob.driver = om.pyOptSparseDriver()

    lofi_prob.driver.options['optimizer'] = 'SNOPT'
    lofi_prob.driver.opt_settings['Major optimality tolerance'] = 1e-6
    lofi_prob.driver.opt_settings['Verify level'] = -1

    lofi_prob.model.add_design_var('x', lower=0.0, upper=1.0)
    lofi_prob.model.add_objective('f')

    lofi_prob.setup()

    hifi_prob = om.Problem(name=f"{opt_name}_hifi")
    hifi_prob.model.add_subsystem('hifi',
                                  ForresterHiFi(),
                                  promotes_inputs=['*'],
                                  promotes_outputs=['*'])
    hifi_prob.setup()

    ###############################################################################
    # Initial conditions
    ###############################################################################
    x0 = {
        'x': np.array([0.55]),
    }

    tau_abs = {
        "tau_f_abs": 1e-1,
    }

    tau_rel = {
        "tau_f_rel": 1e-1,
    }

    des_vars = {
        "x": {
            "lower": 0.0,
            "upper": 1.0,
        },
    }

    objective = "f"

    constraints = {
    }

    lofi_objective = "f"

    lofi_constraints = {
        "f_abs_error_con": {
            "upper": 1.0
        },
        "f_rel_error_con": {
            "upper": 1.0
        },
    }

    max_iter = 100
    f_hi, f_hat, x = optimize(lofi_prob,
                                hifi_prob,
                                x0,
                                des_vars,
                                objective,
                                constraints,
                                lofi_objective,
                                lofi_constraints,
                                tau_abs,
                                tau_rel,
                                max_iter,
                                feas_tol=1e-6,
                                opt_tol=1e-4)
    print(f"f_hi: {f_hi}")
    print(f"f_hat: {f_hat}")
    print(f"x: {x}")