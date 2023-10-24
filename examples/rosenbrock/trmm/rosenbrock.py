import numpy as np

import openmdao.api as om
from e2m2 import TRMMDriver

from model import RosenbrockLF, RosenbrockHF

if __name__ == "__main__":

    opt_name = "rosenbrock_trmm"

    lf_prob = om.Problem(name=f"{opt_name}_lofi")
    lf_prob.model.add_subsystem('lofi',
                                RosenbrockLF(),
                                promotes_inputs=['x'],
                                promotes_outputs=['f'])
    lf_prob.driver = om.pyOptSparseDriver()
    lf_prob.driver.options['optimizer'] = 'SNOPT'
    lf_prob.driver.opt_settings['Major optimality tolerance'] = 1e-7
    # lf_prob.driver.opt_settings['Verify level'] = -1

    hf_prob = om.Problem(name=f"{opt_name}_hifi")
    hf_prob.model.add_subsystem('hifi',
                                RosenbrockHF(),
                                promotes_inputs=['x'],
                                promotes_outputs=['f'])

    hf_prob.driver = TRMMDriver()
    hf_prob.driver.low_fidelity_problem = lf_prob
    hf_prob.driver.options["opt_tol"] = 1e-6
    hf_prob.driver.options["max_iter"] = 10000

    hf_prob.model.add_design_var('x', lower=-2, upper=2, ref=2, ref0=-2)
    hf_prob.model.add_objective('f', ref=1, ref0=0)

    hf_prob.setup()
    hf_prob['x'] = np.array([1.01, 1.01])

    hf_prob.run_driver()
