import numpy as np

import openmdao.api as om
from e2m2 import E2M2Driver

from model import RosenbrockLF, RosenbrockHF

if __name__ == "__main__":

    opt_name = "rosenbrock_e2m2"

    lf_prob = om.Problem(name=f"{opt_name}_lofi")
    lf_prob.model.add_subsystem('lofi',
                                RosenbrockLF(),
                                promotes_inputs=['x1', 'x2'],
                                promotes_outputs=['f'])
    lf_prob.driver = om.pyOptSparseDriver()
    lf_prob.driver.options['optimizer'] = 'SNOPT'
    lf_prob.driver.opt_settings['Major optimality tolerance'] = 1e-8
    # lf_prob.driver.opt_settings['Verify level'] = -1

    hf_prob = om.Problem(name=f"{opt_name}_hifi")
    hf_prob.model.add_subsystem('hifi',
                                RosenbrockHF(),
                                promotes_inputs=['x1', 'x2'],
                                promotes_outputs=['f'])
    
    hf_prob.driver = E2M2Driver()
    hf_prob.driver.low_fidelity_problem = lf_prob
    hf_prob.driver.options["opt_tol"] = 1e-6
    hf_prob.driver.options["max_iter"] = 10000

    # hf_prob.model.add_design_var('x', lower=0.0, upper=1.0)
    # hf_prob.model.add_objective('f')

    # hf_prob.model.add_design_var('x1', lower=-2, upper=2, ref=2, ref0=-2)
    # hf_prob.model.add_design_var('x2', lower=2, upper=2, ref=2, ref0=-2)
    hf_prob.model.add_design_var('x1', lower=-2, upper=2, ref=2, ref0=-2)
    hf_prob.model.add_design_var('x2', lower=-2, upper=2, ref=2, ref0=-2)
    hf_prob.model.add_objective('f', ref=1, ref0=0)

    hf_prob.setup()
    hf_prob['x1'] = 1.01
    hf_prob['x2'] = 1.01

    hf_prob.run_driver()
    # hf_prob.final_setup()

    # lf_prob['delta_x1'] = 1.2
    # lf_prob['delta_x2'] = -1.4
    # lf_prob.run_model()
    # # lf_prob.check_partials()
    # lf_prob.check_totals()

    # lf_prob.model.list_inputs()
    # lf_prob.model.list_outputs(scaling=True)
    # hf_prob.model.list_inputs()
    # hf_prob.model.list_outputs(scaling=True)