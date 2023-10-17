import numpy as np

import openmdao.api as om
from e2m2 import TRMMDriver, AdditiveCalibration, TrustRegion

from model import ForresterLoFiGood, ForresterHiFi

if __name__ == "__main__":
    eq_con = False

    opt_name = "forrester_trmm_driver"
    lf_prob = om.Problem(name=f"{opt_name}_lofi")

    lf_prob.model.add_subsystem("new_design",
                                om.ExecComp("x = x_k + delta_x"),
                                promotes=['*'])

    lf_prob.model.add_subsystem('lofi',
                                ForresterLoFiGood(),
                                promotes_inputs=['x'],
                                promotes_outputs=['f'])

    lf_prob.model.add_subsystem("f_cal",
                                AdditiveCalibration(inputs=['x'], order=1),
                                promotes_inputs=["*"],
                                promotes_outputs=[("gamma", "f_bias")])

    lf_prob.model.add_subsystem("f_hat",
                                om.ExecComp("f_hat = f + f_bias"),
                                promotes=['*'])

    lf_prob.model.add_subsystem("c",
                                om.ExecComp('c = 4*x'),
                                promotes=['*'])

    lf_prob.model.add_subsystem("c_cal",
                                AdditiveCalibration(inputs=['x'], order=1),
                                promotes_inputs=["*"],
                                promotes_outputs=[("gamma", "c_bias")])

    lf_prob.model.add_subsystem("c_hat",
                                om.ExecComp("c_hat = c + c_bias"),
                                promotes=['*'])

    if eq_con:
        lf_prob.model.add_subsystem(f"elastic_c_con",
                                    om.ExecComp(
                                        f"elastic_c_con = c_hat + {3} * (c_slack_1 - c_slack_2)"),
                                    promotes=['*'])
    else:
        lf_prob.model.add_subsystem(f"elastic_c_con_ub",
                                    om.ExecComp(
                                        f"elastic_c_con_ub = c_hat + {-3} * c_ub_slack"),
                                    promotes=['*'])

    lf_prob.model.add_subsystem("trust_region",
                                TrustRegion(dvs=['delta_x']),
                                promotes_inputs=['delta_x'],
                                promotes_outputs=['step_norm'])

    lf_prob.model.add_subsystem("trust_radius_con",
                                om.ExecComp(
                                    "trust_radius_con = step_norm - delta**2"),
                                promotes=['*'])
    
    if eq_con:
        lf_prob.model.add_subsystem("objective",
                                    om.ExecComp(f"obj = obj_scaler * (f_hat + obj_adder) + mu * (c_slack_1 + c_slack_2)"),
                                    promotes=['*'])
    else:
        lf_prob.model.add_subsystem("objective",
                                    om.ExecComp(f"obj = obj_scaler * (f_hat + obj_adder) + mu * c_ub_slack"),
                                    promotes=['*'])
    lf_prob.model.add_objective('obj', ref=1, ref0=0)

    lf_prob.driver = om.pyOptSparseDriver(print_results=True)
    lf_prob.driver.options['optimizer'] = 'SNOPT'
    lf_prob.driver.opt_settings['Major optimality tolerance'] = 1e-8
    # lf_prob.driver.opt_settings['Verify level'] = -1

    lf_prob.model.add_design_var('delta_x')


    if eq_con:
        lf_prob.model.add_design_var("c_slack_1", lower=0)
        lf_prob.model.add_design_var("c_slack_2", lower=0)
        lf_prob.model.add_constraint(f"elastic_c_con", equals=3, ref=3, ref0=1)
    else:
        lf_prob.model.add_design_var("c_ub_slack", lower=0)
        lf_prob.model.add_constraint(f"elastic_c_con_ub", upper=3, ref=3, ref0=1)

    lf_prob.model.add_constraint(
        'x', lower=0, upper=1.0, ref=2, ref0=-1, linear=True)  # is actually linear

    lf_prob.model.add_constraint('trust_radius_con', upper=0.0)

    lf_prob.model.set_input_defaults('delta_x', val=0)

    lf_prob.setup()

    hf_prob = om.Problem(name=f"{opt_name}_hifi")
    hf_prob.model.add_subsystem('hifi',
                                ForresterHiFi(),
                                promotes_inputs=['x'],
                                promotes_outputs=['f'])

    hf_prob.model.add_subsystem("c",
                                om.ExecComp('c = 4*x'),
                                promotes=['*'])

    response_map = {
        "f": ("f", "f_hat"),
        "c": ("c", "c_hat")
    }
    hf_prob.driver = TRMMDriver(response_map=response_map)
    hf_prob.driver.low_fidelity_problem = lf_prob
    hf_prob.driver.options["opt_tol"] = 1e-6

    hf_prob.model.add_design_var('x', lower=0.0, upper=1.0, ref=2, ref0=-1)
    hf_prob.model.add_objective('f', ref=14, ref0=-6)

    if eq_con:
        hf_prob.model.add_constraint('c', equals=3, ref=3, ref0=1)
    else:
        hf_prob.model.add_constraint('c', upper=3, ref=3, ref0=1)

    hf_prob.setup()
    hf_prob['x'] = 1.0

    hf_prob.run_driver()

    # lf_prob.model.list_inputs()
    # lf_prob.model.list_outputs()
    # hf_prob.model.list_inputs()
    # hf_prob.model.list_outputs()
