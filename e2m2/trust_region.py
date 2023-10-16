import numpy as np

import openmdao.api as om

class TrustRegion(om.ExplicitComponent):
    def initialize(self):
        self.options.declare("dvs",
                             types=(str, list, dict),
                             desc="Names of design variables.")

    def setup(self):
        dvs = self.options["dvs"]
        if isinstance(dvs, str):
            self.add_input(dvs)
            # self.declare_partials("step_norm", dvs, method='cs')
            self.declare_partials("step_norm", dvs)
        elif isinstance(dvs, list):
            for dv in dvs:
                if isinstance(dv, str):
                    self.add_input(dv)
                    # self.declare_partials("step_norm", dv, method='cs')
                    self.declare_partials("step_norm", dv)
                else:
                    raise RuntimeError(f"dv: {dv} supplied to Calibration is not a string!")
                
        elif isinstance(dvs, dict):
            for dv, dv_opts in dvs.items():
                if not isinstance(dv, str):
                    raise RuntimeError(f"dv: {dv} supplied to Calibration is not a string!")
                val = dv_opts.get("val", 1)
                shape = dv_opts.get("shape", 1)
                units = dv_opts.get("units", None)
                self.add_input(dv, val=val, shape=shape, units=units)
                # self.declare_partials("step_norm", dv, method='cs')
                self.declare_partials("step_norm", dv)

        self.add_output("step_norm", 0.0, desc="The squared 2-norm of the design step")

    def compute(self, inputs, outputs):
        # print(inputs._get_data())
        # print(inputs.values())
        # vals = np.squeeze(np.array([*inputs.values()]))
        # print(vals)
        outputs['step_norm'] = np.inner(inputs._get_data(), inputs._get_data())

    def compute_partials(self, inputs, partials):
        # print(inputs._get_data())
        # print(inputs.values())
        # vals = np.squeeze(np.array([*inputs.values()]))
        # print(vals)

        for input in inputs:
            partials['step_norm', input] = 2 * inputs[input]

if __name__ == "__main__":
    import unittest
    from openmdao.utils.assert_utils import assert_check_partials

    np.random.seed(0)

    class TestTrustRegion(unittest.TestCase):
        def test_trust_region_dv_str(self):
            prob = om.Problem()

            prob.model.add_subsystem("trust_region",
                                     TrustRegion(dvs='delta_x'),
                                     promotes=['*'])
            
            prob.setup()

            prob.run_model()
            self.assertAlmostEqual(prob['step_norm'][0], 1)

            data = prob.check_partials(method='fd', form="central")
            assert_check_partials(data, atol=1.e-6, rtol=1.e-6)

        def test_trust_region_dv_list(self):
            prob = om.Problem()

            dvs = ['delta_x', 'delta_y', 'delta_z']
            prob.model.add_subsystem("trust_region",
                                     TrustRegion(dvs=dvs),
                                     promotes=['*'])
            
            prob.setup()

            prob.run_model()
            self.assertAlmostEqual(prob['step_norm'][0], 3)

            data = prob.check_partials(method='fd', form="central")
            assert_check_partials(data, atol=1.e-6, rtol=1.e-6)

        def test_trust_region_dv_dict(self):
            prob = om.Problem()

            dvs = {
                'delta_x': {
                    'shape': (2, 3),
                },
                'delta_y': {
                    'shape': (2, 2),
                },
                'delta_z': {
                    'shape': (2),
                }
            }
            prob.model.add_subsystem("trust_region",
                                     TrustRegion(dvs=dvs),
                                     promotes=['*'])
            
            prob.setup()

            prob.run_model()
            self.assertAlmostEqual(prob['step_norm'][0], 12)

            data = prob.check_partials(method='fd', form="central")
            assert_check_partials(data, atol=1.e-6, rtol=1.e-6)

    unittest.main()
