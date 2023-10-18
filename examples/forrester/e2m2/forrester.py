import numpy as np

import openmdao.api as om
from e2m2 import E2M2Driver, TRMMDriver

from model import ForresterLoFiBad, ForresterLoFiGood, ForresterHiFi

if __name__ == "__main__":

    opt_name = "forrester_e2m2_driver"

    lf_prob = om.Problem(name=f"{opt_name}_lofi")

    lf_prob.model.add_subsystem('lofi',
                                ForresterLoFiGood(),
                                promotes_inputs=['x'],
                                promotes_outputs=['f'])
    lf_prob.driver = om.pyOptSparseDriver()
    lf_prob.driver.options['optimizer'] = 'SNOPT'
    lf_prob.driver.opt_settings['Major optimality tolerance'] = 1e-8
    # lf_prob.driver.opt_settings['Verify level'] = -1

    hf_prob = om.Problem(name=f"{opt_name}_hifi")
    hf_prob.model.add_subsystem('hifi',
                                ForresterHiFi(),
                                promotes_inputs=['x'],
                                promotes_outputs=['f'])
    
    hf_prob.driver = E2M2Driver()
    # hf_prob.driver = TRMMDriver()
    hf_prob.driver.low_fidelity_problem = lf_prob
    hf_prob.driver.options["opt_tol"] = 1e-6

    # hf_prob.model.add_design_var('x', lower=0.0, upper=1.0)
    # hf_prob.model.add_objective('f')

    hf_prob.model.add_design_var('x', lower=0.0, upper=1.0, ref=1, ref0=0)
    hf_prob.model.add_objective('f', ref=1, ref0=0)

    hf_prob.setup()
    hf_prob['x'] = 1.0

    hf_prob.run_driver()

    lf_prob.model.list_inputs()
    lf_prob.model.list_outputs()
    hf_prob.model.list_inputs()
    hf_prob.model.list_outputs()