import numpy as np

import openmdao.api as om
from e2m2 import E2M2Driver

from model import ForresterLoFiBad, ForresterLoFiGood, ForresterHiFi

if __name__ == "__main__":
    eq_con = False
    ub_con = True

    opt_name = "constrained_forrester_e2m2_driver"

    lf_prob = om.Problem(name=f"{opt_name}_lofi")

    lf_prob.model.add_subsystem('lofi',
                                ForresterLoFiGood(),
                                promotes_inputs=['x'],
                                promotes_outputs=['f'])

    lf_prob.model.add_subsystem("c",
                                om.ExecComp('c = 4*x'),
                                promotes=['*'])

    lf_prob.driver = om.pyOptSparseDriver()
    lf_prob.driver.options['optimizer'] = 'SNOPT'
    lf_prob.driver.opt_settings['Major optimality tolerance'] = 1e-8
    # lf_prob.driver.opt_settings['Verify level'] = -1

    hf_prob = om.Problem(name=f"{opt_name}_hifi")
    hf_prob.model.add_subsystem('hifi',
                                ForresterHiFi(),
                                promotes_inputs=['x'],
                                promotes_outputs=['f'])
    
    hf_prob.model.add_subsystem("c",
                                om.ExecComp('c = 4*x'),
                                promotes=['*'])

    hf_prob.driver = E2M2Driver()
    hf_prob.driver.low_fidelity_problem = lf_prob
    hf_prob.driver.options["opt_tol"] = 1e-6
    hf_prob.driver.options["max_iter"] = 100
    hf_prob.driver.options["mu_max"] = 100

    hf_prob.model.add_design_var('x', lower=0.0, upper=1.0, ref=1, ref0=-1)
    hf_prob.model.add_objective('f', ref=14, ref0=-6)

    if eq_con:
        hf_prob.model.add_constraint('c', equals=3, ref=3, ref0=1)
    elif ub_con:
        hf_prob.model.add_constraint('c', upper=3, ref=3, ref0=1)
    else:
        hf_prob.model.add_constraint('c', lower=3, ref=3, ref0=1)

    hf_prob.setup()
    hf_prob['x'] = 1.0

    hf_prob.run_driver()

    lf_prob.model.list_inputs()
    lf_prob.model.list_outputs()
    hf_prob.model.list_inputs()
    hf_prob.model.list_outputs()