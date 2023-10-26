import copy
import numpy as np

import openmdao.api as om

# from .error_est import hessian, hessian_differences

# Error estimates aren't right when the names are not `uncalibrated_{output}`
# Need to think more critically about that, and then test on forrester problem making sure it's correct


def calibrate(lofi_problem, hifi_problem, outputs, inputs, include_error_est=False, direct_hessian_diff=False, sr1_hessian_diff=False):

    lofi_raw_outputs = []
    lofi_calibrated_outputs = []
    hifi_outputs = []

    if isinstance(outputs, list):
        for output in outputs:
            if isinstance(output, (list, tuple)):
                if isinstance(output[0], (list, tuple)):
                    lofi_raw_outputs.append(output[0][0])
                    lofi_calibrated_outputs.append(output[0][1])
                else:
                    raise RuntimeError(
                        "Need to specify low-fidelity raw and calibrated outputs")
                hifi_outputs.append(output[1])
            else:
                lofi_raw_outputs.append(f"uncalibrated_{output}")
                lofi_calibrated_outputs.append(output)
                hifi_outputs.append(output)
    elif isinstance(outputs, dict):
        for hifi_output, lofi_output in outputs.items():
            if isinstance(lofi_output, (list, tuple)):
                lofi_raw_outputs.append(lofi_output[0])
                lofi_calibrated_outputs.append(lofi_output[1])
            else:
                raise RuntimeError(
                    "Need to specify low-fidelity raw and calibrated outputs")
            hifi_outputs.append(hifi_output)

    # lofi_outputs = [f"uncalibrated_{output}" for output in outputs]
    cal_orders = {}
    for hifi_output, lofi_output in zip(hifi_outputs, lofi_raw_outputs):
        cal = getattr(lofi_problem.model, f"{hifi_output}_cal")
        cal.options["f_lofi_x0"] = copy.deepcopy(lofi_problem[lofi_output])
        cal.options["f_hifi_x0"] = copy.deepcopy(hifi_problem[hifi_output])
        cal_orders[hifi_output] = cal.options["order"]
        # print(f"calibration f_lofi: {cal.options['f_lofi_x0']}")
        # print(f"calibration f_hifi: {cal.options['f_hifi_x0']}")

    if any(order > 0 for order in cal_orders.values()):
        lofi_totals = lofi_problem.compute_totals(
            lofi_raw_outputs, [*inputs.keys()])
        hifi_totals = hifi_problem.compute_totals(
            hifi_outputs, [*inputs.keys()])
        for hifi_output, lofi_output in zip(hifi_outputs, lofi_raw_outputs):
            cal = getattr(lofi_problem.model, f"{hifi_output}_cal")

            if cal_orders[hifi_output] > 0:
                # lofi_output_totals = {}
                # hifi_output_totals = {}
                # for input in inputs.keys():
                #     lofi_output_totals[input] = copy.deepcopy(lofi_totals[lofi_output, input])
                #     hifi_output_totals[input] = copy.deepcopy(hifi_totals[output, input])
                lofi_output_totals = {
                    input: copy.deepcopy(lofi_totals[lofi_output, input]) for input in inputs.keys()
                }
                hifi_output_totals = {
                    input: copy.deepcopy(hifi_totals[hifi_output, input]) for input in inputs.keys()
                }

                cal.options["g_lofi_x0"] = copy.deepcopy(lofi_output_totals)
                cal.options["g_hifi_x0"] = copy.deepcopy(hifi_output_totals)
                cal.options["dvs"] = copy.deepcopy(inputs)


class MultiplicativeCalibration(om.ExplicitComponent):
    """
    MultiplicativeCalibration handles calibrating a low-fidelity model
    (and optionally its gradient) by multiplying the low-fidelity model
    with a scaling coefficient \beta.
    """

    def initialize(self):
        self.options.declare("inputs",
                             types=(str, list, dict),
                             desc="Names of inputs to take gradient with respect to.")

        self.options.declare("f_lofi_x0",
                             default=1.0,
                             desc="Low-fidelity model value at the calibration point")

        self.options.declare("f_hifi_x0",
                             default=1.0,
                             desc="High-fidelity model value at the calibration point")

        self.options.declare("x0",
                             types=dict,
                             default=None,
                             desc="Dictionary of inputs and their values at the calibration point.")

        self.options.declare("g_lofi_x0",
                             default=None,
                             desc="Low-fidelity gradient at the calibration point")

        self.options.declare("g_hifi_x0",
                             default=None,
                             desc="High-fidelity gradient at the calibration point")

        self.options.declare("order",
                             default=0,
                             types=int,
                             desc="Calibration order. 0th order calibrates values, 1st order also calibrates gradients")

    def setup(self):

        order = self.options["order"]
        if order > 0:
            inputs = self.options["inputs"]
            if isinstance(inputs, str):
                self.add_input(inputs)
            elif isinstance(inputs, list):
                for input in inputs:
                    if isinstance(input, str):
                        self.add_input(input)
                    else:
                        raise RuntimeError(
                            f"Input: {input} supplied to Calibration is not a string!")

            elif isinstance(inputs, dict):
                for input, input_opts in inputs.items():
                    if not isinstance(input, str):
                        raise RuntimeError(
                            f"Input: {input} supplied to Calibration is not a string!")
                    val = input_opts.get("val", 1)
                    shape = input_opts.get("shape", 1)
                    units = input_opts.get("units", None)
                    self.add_input(input, val=val, shape=shape, units=units)

        f_lofi_x0 = self.options["f_lofi_x0"]
        f_hifi_x0 = self.options["f_hifi_x0"]
        self.add_output("beta", val=f_hifi_x0/f_lofi_x0,
                        desc="Multiplicative scaling value")

        if order > 0:
            inputs = self.options["inputs"]
            if isinstance(inputs, str):
                self.declare_partials("beta", inputs)

            elif isinstance(inputs, list):
                for input in inputs:
                    if not isinstance(input, str):
                        raise RuntimeError(
                            f"Input: {input} supplied to Calibration is not a string!")
                self.declare_partials("beta", inputs)

            elif isinstance(inputs, dict):
                for key, value in inputs.items():
                    if not isinstance(key, str):
                        raise RuntimeError(
                            f"Input: {input} supplied to Calibration is not a string!")
                self.declare_partials("beta", [*inputs.keys()])

    def compute(self, inputs, outputs):
        order = self.options["order"]

        f_lofi_x0 = self.options["f_lofi_x0"]
        f_hifi_x0 = self.options["f_hifi_x0"]
        beta = f_hifi_x0 / f_lofi_x0

        if order > 0:
            x0 = self.options["x0"]
            g_lofi_x0 = self.options["g_lofi_x0"]
            g_hifi_x0 = self.options["g_hifi_x0"]

            if g_lofi_x0 is None or g_hifi_x0 is None or x0 is None:
                return

            for input in inputs.keys():
                g_beta = (g_hifi_x0[input] - beta *
                          g_lofi_x0[input]) / f_lofi_x0
                beta += g_beta @ (inputs[input] - x0[input])

        outputs["beta"] = beta

    def compute_partials(self, inputs, partials):
        order = self.options["order"]
        if order > 0:
            f_lofi_x0 = self.options["f_lofi_x0"]
            f_hifi_x0 = self.options["f_hifi_x0"]
            beta = f_hifi_x0 / f_lofi_x0

            g_lofi_x0 = self.options["g_lofi_x0"]
            g_hifi_x0 = self.options["g_hifi_x0"]
            if g_lofi_x0 is None or g_hifi_x0 is None:
                return

            for input in inputs.keys():
                g_beta = (g_hifi_x0[input] - beta *
                          g_lofi_x0[input]) / f_lofi_x0

                partials["beta", input] = g_beta


class AdditiveCalibration(om.ExplicitComponent):
    """
    AdditiveCalibration handles calibrating a low-fidelity model
    (and optionally its gradient) by summing the low-fidelity model
    with a coefficient \gamma.

    """

    def initialize(self):
        self.options.declare("metadata",
                             desc="Dictionary of HF model variables and their metadata")

        self.options.declare("inputs",
                             types=(str, list, dict),
                             desc="Names of inputs to take gradient with respect to.")

        self.options.declare("response_metadata",
                             desc="Dictionary of metadata for the calibrated response")

        self.options.declare("f_lofi_x0",
                             default=1.0,
                             desc="Low-fidelity model value at the calibration point")

        self.options.declare("f_hifi_x0",
                             default=1.0,
                             desc="High-fidelity model value at the calibration point")

        self.options.declare("dvs",
                             types=dict,
                             default=None,
                             desc="Dictionary of design variables and their metadata.")

        self.options.declare("g_lofi_x0",
                             default=None,
                             desc="Low-fidelity gradient at the calibration point")

        self.options.declare("g_hifi_x0",
                             default=None,
                             desc="High-fidelity gradient at the calibration point")

        self.options.declare("order",
                             default=0,
                             types=int,
                             desc="Calibration order. 0th order calibrates values, 1st order also calibrates gradients")

    def setup(self):
        metadata = self.options['metadata']

        order = self.options["order"]
        if order > 0:
            inputs = self.options["inputs"]
            if isinstance(inputs, str):
                self.add_input(f"delta_{inputs}")
            elif isinstance(inputs, list):
                for input in inputs:
                    if isinstance(input, str):
                        self.add_input(f"delta_{input}")
                    else:
                        raise RuntimeError(
                            f"Input: {input} supplied to Calibration is not a string!")

            elif isinstance(inputs, dict):
                for input, input_opts in inputs.items():
                    if not isinstance(input, str):
                        raise RuntimeError(
                            f"Input: {input} supplied to Calibration is not a string!")
                    # val = input_opts.get("val", 1)
                    dv_name = input_opts['name']
                    shape = metadata[input_opts['source']]['shape']
                    units = input_opts.get("units", None)
                    self.add_input(f"delta_{dv_name}",
                                   val=0,
                                   shape=shape,
                                   units=units)

        resp_metadata = self.options['repsonse_metadata']
        resp_name = resp_metadata['name'],
        resp_shape = metadata['resp_name']['shape']
        resp_units = resp_metadata['units']
        self.add_output(f"gamma_{resp_name}",
                        val=0,
                        shape=resp_shape,
                        units=resp_units,
                        desc="Additive correction value")

        if order > 0:
            inputs = self.options["inputs"]
            if isinstance(inputs, str):
                self.declare_partials("gamma", f"delta_{inputs}")

            elif isinstance(inputs, list):
                for input in inputs:
                    if not isinstance(input, str):
                        raise RuntimeError(
                            f"Input: {input} supplied to Calibration is not a string!")
                self.declare_partials("gamma", f"delta_{input}")

            elif isinstance(inputs, dict):
                for input, meta in inputs.items():
                    if not isinstance(input, str):
                        raise RuntimeError(
                            f"Input: {input} supplied to Calibration is not a string!")
                    input_name = meta['name']
                    self.declare_partials(
                        f"gamma_{resp_name}", f"delta_{input_name}")

    def compute(self, inputs, outputs):
        resp_metadata = self.options['repsonse_metadata']
        resp_name = resp_metadata['name'],
        order = self.options["order"]

        f_lofi_x0 = self.options["f_lofi_x0"]
        f_hifi_x0 = self.options["f_hifi_x0"]
        gamma = f_hifi_x0 - f_lofi_x0

        if order > 0:
            g_lofi_x0 = self.options["g_lofi_x0"]
            g_hifi_x0 = self.options["g_hifi_x0"]

            if g_lofi_x0 is None or g_hifi_x0 is None:
                return

            for input in inputs.keys():
                dv_name = input[6:]
                g_gamma = g_hifi_x0[dv_name] - g_lofi_x0[dv_name]
                gamma += np.dot(g_gamma, inputs[input])

        outputs[f"gamma_{resp_name}"] = gamma

    def compute_partials(self, inputs, partials):
        resp_metadata = self.options['repsonse_metadata']
        resp_name = resp_metadata['name'],
        order = self.options["order"]
        if order > 0:
            g_lofi_x0 = self.options["g_lofi_x0"]
            g_hifi_x0 = self.options["g_hifi_x0"]
            if g_lofi_x0 is None or g_hifi_x0 is None:
                return

            for input in inputs.keys():
                dv_name = input[6:]
                g_gamma = g_hifi_x0[dv_name] - g_lofi_x0[dv_name]
                partials[f"gamma_{resp_name}", input] = g_gamma


if __name__ == "__main__":
    import unittest
    import copy
    import numpy as np

    from openmdao.utils.assert_utils import assert_check_totals

    class TestMultiplicativeCalibration(unittest.TestCase):
        def test_zero_order_calibration(self):
            n = 3
            x0 = np.ones(n)

            lofi_prob = om.Problem()

            lofi_prob.model.add_subsystem("cal",
                                          MultiplicativeCalibration(inputs="x",
                                                                    f_lofi_x0=1.0,
                                                                    f_hifi_x0=1.0,
                                                                    order=0),
                                          promotes=["*"])

            lofi_prob.model.add_subsystem("f_lofi_x",
                                          om.ExecComp("f_lofi_x = dot(x,x)",
                                                      x={"val": x0}),
                                          promotes=["*"])

            lofi_prob.model.add_subsystem("f_hat",
                                          om.ExecComp(
                                              "f_hat = beta * f_lofi_x"),
                                          promotes=["*"])
            lofi_prob.setup()

            hifi_prob = om.Problem()
            hifi_prob.model.add_subsystem("f_hifi_x",
                                          om.ExecComp("f_hifi_x = dot(x,x)**1.5",
                                                      x={"val": x0}),
                                          promotes=["*"])

            hifi_prob.setup()

            lofi_prob["x"] = x0
            lofi_prob.run_model()

            hifi_prob["x"] = x0
            hifi_prob.run_model()

            lofi_prob.model.cal.options["f_lofi_x0"] = copy.deepcopy(
                lofi_prob["f_lofi_x"])
            lofi_prob.model.cal.options["f_hifi_x0"] = copy.deepcopy(
                hifi_prob["f_hifi_x"])

            delta = 1e-1
            pert = np.array([0.242554, 0.5830354, 0.428559])
            x = x0 + delta * pert

            lofi_prob["x"] = x0
            lofi_prob.run_model()
            self.assertAlmostEqual(lofi_prob["f_lofi_x"][0], 3.0)
            self.assertAlmostEqual(
                lofi_prob["f_hat"][0], hifi_prob["f_hifi_x"][0])

            lofi_prob["x"] = x
            lofi_prob.run_model()
            self.assertAlmostEqual(lofi_prob["f_lofi_x"][0], 3.25665394)
            self.assertAlmostEqual(lofi_prob["f_hat"][0], 5.64069008)

        def test_zero_order_calibration_totals(self):
            n = 3
            x0 = np.ones(n)

            lofi_prob = om.Problem()

            lofi_prob.model.add_subsystem("cal",
                                          MultiplicativeCalibration(inputs="x",
                                                                    f_lofi_x0=1.0,
                                                                    f_hifi_x0=1.0,
                                                                    order=0),
                                          promotes=["*"])

            lofi_prob.model.add_subsystem("f_lofi_x",
                                          om.ExecComp("f_lofi_x = dot(x,x)",
                                                      x={"val": x0}),
                                          promotes=["*"])

            lofi_prob.model.add_subsystem("f_hat",
                                          om.ExecComp(
                                              "f_hat = beta * f_lofi_x"),
                                          promotes=["*"])
            lofi_prob.setup()

            hifi_prob = om.Problem()
            hifi_prob.model.add_subsystem("f_hifi_x",
                                          om.ExecComp("f_hifi_x = dot(x,x)**1.5",
                                                      x={"val": x0}),
                                          promotes=["*"])

            hifi_prob.setup()

            lofi_prob["x"] = x0
            lofi_prob.run_model()

            hifi_prob["x"] = x0
            hifi_prob.run_model()

            lofi_prob.model.cal.options["f_lofi_x0"] = copy.deepcopy(
                lofi_prob["f_lofi_x"])
            lofi_prob.model.cal.options["f_hifi_x0"] = copy.deepcopy(
                hifi_prob["f_hifi_x"])

            lofi_prob.run_model()
            totals_data = lofi_prob.check_totals(of=["f_hat", "f_lofi_x", "beta"], wrt=[
                                                 "x"], method="fd", form="central")
            assert_check_totals(totals_data)

        def test_first_order_calibration(self):
            n = 3
            x0 = np.ones(n)

            lofi_prob = om.Problem()

            inputs = {"x": {"shape": n}}
            lofi_prob.model.add_subsystem("cal",
                                          MultiplicativeCalibration(inputs=inputs,
                                                                    f_lofi_x0=1.0,
                                                                    f_hifi_x0=1.0,
                                                                    order=1),
                                          promotes=["*"])

            lofi_prob.model.add_subsystem("f_lofi_x",
                                          om.ExecComp("f_lofi_x = dot(x,x)",
                                                      x={"val": x0}),
                                          promotes=["*"])

            lofi_prob.model.add_subsystem("f_hat",
                                          om.ExecComp(
                                              "f_hat = beta * f_lofi_x"),
                                          promotes=["*"])

            lofi_prob.setup()

            hifi_prob = om.Problem()
            hifi_prob.model.add_subsystem("f_hifi_x",
                                          om.ExecComp("f_hifi_x = dot(x,x)**1.5",
                                                      x={"val": x0}),
                                          promotes=["*"])

            hifi_prob.setup()

            lofi_prob["x"] = x0
            lofi_prob.run_model()
            lofi_totals = lofi_prob.compute_totals("f_lofi_x", "x")
            lofi_totals = {"x": copy.deepcopy(lofi_totals["f_lofi_x", "x"])}

            hifi_prob["x"] = x0
            hifi_prob.run_model()
            hifi_totals = hifi_prob.compute_totals("f_hifi_x", "x")
            hifi_totals = {"x": copy.deepcopy(hifi_totals["f_hifi_x", "x"])}

            lofi_prob.model.cal.options["f_lofi_x0"] = copy.deepcopy(
                lofi_prob["f_lofi_x"])
            lofi_prob.model.cal.options["f_hifi_x0"] = copy.deepcopy(
                hifi_prob["f_hifi_x"])
            lofi_prob.model.cal.options["g_lofi_x0"] = copy.deepcopy(
                lofi_totals)
            lofi_prob.model.cal.options["g_hifi_x0"] = copy.deepcopy(
                hifi_totals)
            lofi_prob.model.cal.options["x0"] = {"x": x0}

            delta = 1e-1
            pert = np.array([0.242554, 0.5830354, 0.428559])
            x = x0 + delta * pert

            lofi_prob["x"] = x0
            lofi_prob.run_model()
            self.assertAlmostEqual(lofi_prob["f_lofi_x"][0], 3.0)
            self.assertAlmostEqual(
                lofi_prob["f_hat"][0], hifi_prob["f_hifi_x"][0])
            f_hat_totals = lofi_prob.compute_totals("f_hat", "x")["f_hat", "x"]
            np.testing.assert_allclose(f_hat_totals, hifi_totals["x"])

            lofi_prob["x"] = x
            lofi_prob.run_model()
            self.assertAlmostEqual(lofi_prob["f_lofi_x"][0], 3.25665394)
            self.assertAlmostEqual(lofi_prob["f_hat"][0], 5.876498826635378)

        def test_first_order_calibration_totals(self):
            n = 3
            x0 = np.ones(n)

            lofi_prob = om.Problem()

            inputs = {"x": {"shape": n}}
            lofi_prob.model.add_subsystem("cal",
                                          MultiplicativeCalibration(inputs=inputs,
                                                                    f_lofi_x0=1.0,
                                                                    f_hifi_x0=1.0,
                                                                    order=1),
                                          promotes=["*"])

            lofi_prob.model.add_subsystem("f_lofi_x",
                                          om.ExecComp("f_lofi_x = dot(x,x)",
                                                      x={"val": x0}),
                                          promotes=["*"])

            lofi_prob.model.add_subsystem("f_hat",
                                          om.ExecComp(
                                              "f_hat = beta * f_lofi_x"),
                                          promotes=["*"])

            lofi_prob.setup()

            hifi_prob = om.Problem()
            hifi_prob.model.add_subsystem("f_hifi_x",
                                          om.ExecComp("f_hifi_x = dot(x,x)**1.5",
                                                      x={"val": x0}),
                                          promotes=["*"])

            hifi_prob.setup()

            lofi_prob["x"] = x0
            lofi_prob.run_model()
            lofi_totals = lofi_prob.compute_totals("f_lofi_x", "x")
            lofi_totals = {"x": copy.deepcopy(lofi_totals["f_lofi_x", "x"])}

            hifi_prob["x"] = x0
            hifi_prob.run_model()
            hifi_totals = hifi_prob.compute_totals("f_hifi_x", "x")
            hifi_totals = {"x": copy.deepcopy(hifi_totals["f_hifi_x", "x"])}

            lofi_prob.model.cal.options["f_lofi_x0"] = copy.deepcopy(
                lofi_prob["f_lofi_x"])
            lofi_prob.model.cal.options["f_hifi_x0"] = copy.deepcopy(
                hifi_prob["f_hifi_x"])
            lofi_prob.model.cal.options["g_lofi_x0"] = copy.deepcopy(
                lofi_totals)
            lofi_prob.model.cal.options["g_hifi_x0"] = copy.deepcopy(
                hifi_totals)
            lofi_prob.model.cal.options["x0"] = {"x": x0}

            lofi_prob.run_model()
            totals_data = lofi_prob.check_totals(of=["f_hat", "f_lofi_x", "beta"], wrt=[
                                                 "x"], method="fd", form="central")
            assert_check_totals(totals_data)

    class TestAdditiveCalibration(unittest.TestCase):
        def test_zero_order_calibration(self):
            n = 3
            x0 = np.ones(n)

            lofi_prob = om.Problem()

            lofi_prob.model.add_subsystem("cal",
                                          AdditiveCalibration(inputs="x",
                                                              f_lofi_x0=1.0,
                                                              f_hifi_x0=1.0,
                                                              order=0),
                                          promotes=["*"])

            lofi_prob.model.add_subsystem("f_lofi_x",
                                          om.ExecComp("f_lofi_x = dot(x,x)",
                                                      x={"val": x0}),
                                          promotes=["*"])

            lofi_prob.model.add_subsystem("f_hat",
                                          om.ExecComp(
                                              "f_hat = f_lofi_x + gamma"),
                                          promotes=["*"])
            lofi_prob.setup()

            hifi_prob = om.Problem()
            hifi_prob.model.add_subsystem("f_hifi_x",
                                          om.ExecComp("f_hifi_x = dot(x,x)**1.5",
                                                      x={"val": x0}),
                                          promotes=["*"])

            hifi_prob.setup()

            lofi_prob["x"] = x0
            lofi_prob.run_model()

            hifi_prob["x"] = x0
            hifi_prob.run_model()

            lofi_prob.model.cal.options["f_lofi_x0"] = copy.deepcopy(
                lofi_prob["f_lofi_x"])
            lofi_prob.model.cal.options["f_hifi_x0"] = copy.deepcopy(
                hifi_prob["f_hifi_x"])

            delta = 1e-1
            pert = np.array([0.242554, 0.5830354, 0.428559])
            x = x0 + delta * pert

            lofi_prob["x"] = x0
            lofi_prob.run_model()
            self.assertAlmostEqual(lofi_prob["f_lofi_x"][0], 3.0)
            self.assertAlmostEqual(
                lofi_prob["f_hat"][0], hifi_prob["f_hifi_x"][0])

            lofi_prob["x"] = x
            lofi_prob.run_model()
            self.assertAlmostEqual(lofi_prob["f_lofi_x"][0], 3.25665394)
            self.assertAlmostEqual(lofi_prob["f_hat"][0], 5.452806358077133)

        def test_zero_order_calibration_totals(self):
            n = 3
            x0 = np.ones(n)

            lofi_prob = om.Problem()

            lofi_prob.model.add_subsystem("cal",
                                          AdditiveCalibration(inputs="x",
                                                              f_lofi_x0=1.0,
                                                              f_hifi_x0=1.0,
                                                              order=0),
                                          promotes=["*"])

            lofi_prob.model.add_subsystem("f_lofi_x",
                                          om.ExecComp("f_lofi_x = dot(x,x)",
                                                      x={"val": x0}),
                                          promotes=["*"])

            lofi_prob.model.add_subsystem("f_hat",
                                          om.ExecComp(
                                              "f_hat = f_lofi_x + gamma"),
                                          promotes=["*"])
            lofi_prob.setup()

            hifi_prob = om.Problem()
            hifi_prob.model.add_subsystem("f_hifi_x",
                                          om.ExecComp("f_hifi_x = dot(x,x)**1.5",
                                                      x={"val": x0}),
                                          promotes=["*"])

            hifi_prob.setup()

            lofi_prob["x"] = x0
            lofi_prob.run_model()

            hifi_prob["x"] = x0
            hifi_prob.run_model()

            lofi_prob.model.cal.options["f_lofi_x0"] = copy.deepcopy(
                lofi_prob["f_lofi_x"])
            lofi_prob.model.cal.options["f_hifi_x0"] = copy.deepcopy(
                hifi_prob["f_hifi_x"])

            lofi_prob.run_model()
            totals_data = lofi_prob.check_totals(of=["f_hat", "f_lofi_x", "gamma"], wrt=[
                                                 "x"], method="fd", form="central")
            assert_check_totals(totals_data)

        def test_first_order_calibration(self):
            n = 3
            x0 = np.ones(n)

            lofi_prob = om.Problem()

            inputs = {"x": {"shape": n}}
            lofi_prob.model.add_subsystem("cal",
                                          AdditiveCalibration(inputs=inputs,
                                                              f_lofi_x0=1.0,
                                                              f_hifi_x0=1.0,
                                                              order=1),
                                          promotes=["*"])

            lofi_prob.model.add_subsystem("f_lofi_x",
                                          om.ExecComp("f_lofi_x = dot(x,x)",
                                                      x={"val": x0}),
                                          promotes=["*"])

            lofi_prob.model.add_subsystem("f_hat",
                                          om.ExecComp(
                                              "f_hat = f_lofi_x + gamma"),
                                          promotes=["*"])

            lofi_prob.setup()

            hifi_prob = om.Problem()
            hifi_prob.model.add_subsystem("f_hifi_x",
                                          om.ExecComp("f_hifi_x = dot(x,x)**1.5",
                                                      x={"val": x0}),
                                          promotes=["*"])

            hifi_prob.setup()

            lofi_prob["x"] = x0
            lofi_prob.run_model()
            lofi_totals = lofi_prob.compute_totals("f_lofi_x", "x")
            lofi_totals = {"x": copy.deepcopy(lofi_totals["f_lofi_x", "x"])}

            hifi_prob["x"] = x0
            hifi_prob.run_model()
            hifi_totals = hifi_prob.compute_totals("f_hifi_x", "x")
            hifi_totals = {"x": copy.deepcopy(hifi_totals["f_hifi_x", "x"])}

            lofi_prob.model.cal.options["f_lofi_x0"] = copy.deepcopy(
                lofi_prob["f_lofi_x"])
            lofi_prob.model.cal.options["f_hifi_x0"] = copy.deepcopy(
                hifi_prob["f_hifi_x"])
            lofi_prob.model.cal.options["g_lofi_x0"] = copy.deepcopy(
                lofi_totals)
            lofi_prob.model.cal.options["g_hifi_x0"] = copy.deepcopy(
                hifi_totals)
            lofi_prob.model.cal.options["x0"] = {"x": x0}

            delta = 1e-1
            pert = np.array([0.242554, 0.5830354, 0.428559])
            x = x0 + delta * pert

            lofi_prob["x"] = x0
            lofi_prob.run_model()
            self.assertAlmostEqual(lofi_prob["f_lofi_x"][0], 3.0)
            self.assertAlmostEqual(
                lofi_prob["f_hat"][0], hifi_prob["f_hifi_x"][0])
            f_hat_totals = lofi_prob.compute_totals("f_hat", "x")["f_hat", "x"]
            np.testing.assert_allclose(f_hat_totals, hifi_totals["x"])

            lofi_prob["x"] = x
            lofi_prob.run_model()
            self.assertAlmostEqual(lofi_prob["f_lofi_x"][0], 3.25665394)
            self.assertAlmostEqual(lofi_prob["f_hat"][0], 5.853651302786497)

        def test_first_order_calibration_totals(self):
            n = 3
            x0 = np.ones(n)

            lofi_prob = om.Problem()

            inputs = {"x": {"shape": n}}
            lofi_prob.model.add_subsystem("cal",
                                          AdditiveCalibration(inputs=inputs,
                                                              f_lofi_x0=1.0,
                                                              f_hifi_x0=1.0,
                                                              order=1),
                                          promotes=["*"])

            lofi_prob.model.add_subsystem("f_lofi_x",
                                          om.ExecComp("f_lofi_x = dot(x,x)",
                                                      x={"val": x0}),
                                          promotes=["*"])

            lofi_prob.model.add_subsystem("f_hat",
                                          om.ExecComp(
                                              "f_hat = f_lofi_x + gamma"),
                                          promotes=["*"])

            lofi_prob.setup()

            hifi_prob = om.Problem()
            hifi_prob.model.add_subsystem("f_hifi_x",
                                          om.ExecComp("f_hifi_x = dot(x,x)**1.5",
                                                      x={"val": x0}),
                                          promotes=["*"])

            hifi_prob.setup()

            lofi_prob["x"] = x0
            lofi_prob.run_model()
            lofi_totals = lofi_prob.compute_totals("f_lofi_x", "x")
            lofi_totals = {"x": copy.deepcopy(lofi_totals["f_lofi_x", "x"])}

            hifi_prob["x"] = x0
            hifi_prob.run_model()
            hifi_totals = hifi_prob.compute_totals("f_hifi_x", "x")
            hifi_totals = {"x": copy.deepcopy(hifi_totals["f_hifi_x", "x"])}

            lofi_prob.model.cal.options["f_lofi_x0"] = copy.deepcopy(
                lofi_prob["f_lofi_x"])
            lofi_prob.model.cal.options["f_hifi_x0"] = copy.deepcopy(
                hifi_prob["f_hifi_x"])
            lofi_prob.model.cal.options["g_lofi_x0"] = copy.deepcopy(
                lofi_totals)
            lofi_prob.model.cal.options["g_hifi_x0"] = copy.deepcopy(
                hifi_totals)
            lofi_prob.model.cal.options["x0"] = {"x": x0}

            lofi_prob.run_model()
            totals_data = lofi_prob.check_totals(of=["f_hat", "f_lofi_x", "gamma"], wrt=[
                                                 "x"], method="fd", form="central")
            assert_check_totals(totals_data)

    unittest.main()
