import numpy as np

import openmdao.api as om


def _rosenbrock_hf(x):
    return (1 - x[0])**2 + (x[1] - x[0]**2)**2


def _rosenbrock_lf(x):
    # return (1 - x[0])**2 + 100*(x[1] - x[0]**2)**2 + 2 + x[0]*10 - x[1]*20 + 1
    # return x[0] + x[1]
    return 0.0
    # return (1 - x[0])**2 + 100*(x[1] - x[0]**2)**2 + 2  # - 456.3
    # return x[0]**4 + x[1]**2
    # return (1 - x[0])**2 + 100


class RosenbrockLF(om.ExplicitComponent):
    def setup(self):

        self.add_input("x1")
        self.add_input("x2")

        self.add_output("f")

        self.declare_partials('*', '*', method='cs')

    def compute(self, inputs, outputs):
        x1 = inputs['x1']
        x2 = inputs['x2']
        x = np.array([x1, x2])

        outputs['f'] = _rosenbrock_lf(x)


class RosenbrockHF(om.ExplicitComponent):
    def setup(self):
        self.add_input("x1")
        self.add_input("x2")

        self.add_output("f")

        self.declare_partials('*', '*', method='cs')

    def compute(self, inputs, outputs):
        x1 = inputs['x1']
        x2 = inputs['x2']
        x = np.array([x1, x2])
        outputs['f'] = _rosenbrock_hf(x)


if __name__ == "__main__":
    import unittest
    from openmdao.utils.assert_utils import assert_check_partials

    class TestRosenbrockLF(unittest.TestCase):
        np.random.seed(0)

        prob = om.Problem()

        prob.model.add_subsystem('lofi',
                                 RosenbrockLF(),
                                 promotes_inputs=['*'],
                                 promotes_outputs=['*'])

        prob.setup()

        prob['x1'] = np.random.normal()
        prob['x2'] = np.random.normal()
        data = prob.check_partials(form="central")
        assert_check_partials(data)

    class TestRosenbrockHF(unittest.TestCase):
        np.random.seed(0)

        prob = om.Problem()

        prob.model.add_subsystem('hifi',
                                 RosenbrockHF(),
                                 promotes_inputs=['*'],
                                 promotes_outputs=['*'])

        prob.setup()

        prob['x1'] = np.random.normal()
        prob['x2'] = np.random.normal()
        data = prob.check_partials(form="central")
        assert_check_partials(data)

    unittest.main()
