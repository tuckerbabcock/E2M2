import numpy as np

import openmdao.api as om

def _forrester_hifi(x):
    return (6 * x - 2) ** 2 * np.sin(12 * x - 4)

def _forrester_lofi(x, A=0.5, B=10, C=-5):
    return A*_forrester_hifi(x) + B*(x - 0.5) + C

class ForresterLoFiGood(om.ExplicitComponent):
    def setup(self):
        
        self.add_input("x")

        self.add_input("f_lofi_bias", val=0.0)
        self.add_input("f_lofi_coefficient", val=1.0)

        self.add_output("uncalibrated_f")
        self.add_output("f")

        self.declare_partials('*', '*', method='cs')

    def compute(self, inputs, outputs):
        x = inputs['x']
        bias = inputs['f_lofi_bias']
        coefficient = inputs['f_lofi_coefficient']

        outputs['uncalibrated_f'] = _forrester_lofi(x, 0.85, 5.0, -2.0)
        outputs['f'] = outputs['uncalibrated_f'] * coefficient + bias

class ForresterLoFiBad(om.ExplicitComponent):
    def setup(self):
        
        self.add_input("x")

        self.add_input("f_lofi_bias", val=0.0)
        self.add_input("f_lofi_coefficient", val=1.0)

        self.add_output("uncalibrated_f")
        self.add_output("f")

        self.declare_partials('*', '*', method='cs')

    def compute(self, inputs, outputs):
        x = inputs['x']
        bias = inputs['f_lofi_bias']
        coefficient = inputs['f_lofi_coefficient']

        outputs['uncalibrated_f'] = _forrester_lofi(x, 0.6, 10.0, -5.0)
        outputs['f'] = outputs['uncalibrated_f'] * coefficient + bias

class ForresterHiFi(om.ExplicitComponent):
    def setup(self):
        self.add_input("x")

        self.add_output("f")

        self.declare_partials('*', '*', method='cs')

    def compute(self, inputs, outputs):
        x = inputs['x']
        outputs['f'] = _forrester_hifi(x)
        # outputs['f'] = _forrester_lofi(x, 1.0, 10.0, -5.0)

if __name__ == "__main__":
    import unittest
    from openmdao.utils.assert_utils import assert_check_partials

    class TestForresterLofiPartials(unittest.TestCase):
        np.random.seed(0)

        prob = om.Problem()

        prob.model.add_subsystem('lofi',
                                 ForresterLoFiBad(),
                                 promotes_inputs=['*'],
                                 promotes_outputs=['*'])

        prob.setup()

        prob['f_lofi_bias'] = np.random.normal()
        prob['f_lofi_coefficient'] = np.random.normal()

        xs = np.linspace(0.0, 1.0, 10)
        for x in xs:
            prob['x'] = x
            data = prob.check_partials(form="central")
            assert_check_partials(data)

    class TestForresterHifiPartials(unittest.TestCase):
        np.random.seed(0)

        prob = om.Problem()

        prob.model.add_subsystem('lofi',
                                 ForresterHiFi(),
                                 promotes_inputs=['*'],
                                 promotes_outputs=['*'])

        prob.setup()

        xs = np.linspace(0.0, 1.0, 10)
        for x in xs:
            prob['x'] = x
            data = prob.check_partials(form="central")
            assert_check_partials(data)
        