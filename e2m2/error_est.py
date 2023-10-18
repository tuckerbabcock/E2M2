import copy

import numpy as np

import openmdao.api as om

from .approximate_hessian import update_approximate_hessian_difference

def update_error_ests(dvs, responses, response_map, lf_prob, lf_totals, hf_totals):
    n = 0
    for dv in dvs.keys():
        n += lf_prob.get_val(dv).size
    
    design_step = np.zeros(n)
    offset = 0
    for dv in dvs.keys():
        step = lf_prob.get_val(f"delta_{dv}")
        design_step[offset:offset + step.size] = step
        offset += step.size

    print()
    print(f"update error ests:")
    print(f"design step: {design_step}")
    print(f"lf_totals: {lf_totals}")
    print(f"hf_totals: {hf_totals}")
    print(f"responses: {responses}")
    print()
    for response, meta in responses.items():
        response_name = meta['name']
        error_est = getattr(lf_prob.model, f"{response_name}_error_est")

        grad_diff = np.zeros(n)
        offset = 0
        for dv in dvs.keys():
            diff = lf_totals[response_map[response_name][0], f"delta_{dv}"] - hf_totals[response, dv]
            print(f"diff: {diff}")
            grad_diff[offset:offset + diff.size] = diff
            offset += diff.size

        print(f"grad_diff: {grad_diff}")

        h_diff_x0 = error_est.options['h_diff_x0']
        error_est.options['h_diff_x0'] = update_approximate_hessian_difference(grad_diff,
                                                                               design_step,
                                                                               h_diff_x0)
        print(f"Hessian:\n{error_est.options['h_diff_x0']}")


class ErrorEstimate(om.ExplicitComponent):
    """
    ErrorEstimate estimates the error between a low-fidelity model and a high-fidelity model with a
    second order Taylor series expansion of the difference between the models
    """
    def initialize(self):
        self.options.declare("dvs",
                             types=(str, list, dict),
                             desc="Names of design variables.")

        self.options.declare("h_diff_x0",
                             default=None,
                             desc="Difference in Hessians at the calibration point. Low-fidelity - high-fidelity.")

    def setup(self):
        dvs = self.options["dvs"]
        if isinstance(dvs, str):
            self.add_input(f"delta_{dvs}")
            # self.declare_partials("error_est", dvs, method='cs')
            self.declare_partials("error_est", f"delta_{dvs}")
            self.declare_partials("gradient_error_est", f"delta_{dvs}")
        elif isinstance(dvs, list):
            for dv in dvs:
                if isinstance(dv, str):
                    self.add_input(f"delta_{dv}")
                    # self.declare_partials("error_est", dv, method='cs')
                    self.declare_partials("error_est", f"delta_{dv}")
                    self.declare_partials("gradient_error_est", f"delta_{dv}")
                else:
                    raise RuntimeError(f"dv: {dv} supplied to Calibration is not a string!")
                
        elif isinstance(dvs, dict):
            for dv, input_opts in dvs.items():
                if not isinstance(dv, str):
                    raise RuntimeError(f"dv: {dv} supplied to Calibration is not a string!")
                val = input_opts.get("val", 1)
                shape = input_opts.get("shape", 1)
                units = input_opts.get("units", None)
                self.add_input(f"delta_{dv}", val=val, shape=shape, units=units)
                # self.declare_partials("error_est", dv, method='cs')
                self.declare_partials("error_est", f"delta_{dv}")
                self.declare_partials("gradient_error_est", f"delta_{dv}")

        self.add_output("error_est", val=0.0, desc="Function error estimate")
        self.add_output("gradient_error_est", val=0.0, desc="Squared norm of the estimated gradient error")

    def compute(self, inputs, outputs):
        h_diff_x0 = self.options["h_diff_x0"]
        if h_diff_x0 is None:
            h_diff_x0 = np.eye(inputs._get_data().size)
        elif isinstance(h_diff_x0, list):
            h_diff_x0 = h_diff_x0[2]

        # print(f"Error est h_diff_x0:\n{h_diff_x0}")
        tmp = h_diff_x0 @ inputs._get_data()
        xTHx = np.dot(inputs._get_data(), tmp)
        outputs["error_est"] = 0.5*xTHx
        outputs["gradient_error_est"] = np.dot(tmp, tmp)

    def compute_partials(self, inputs, partials):
        h_diff_x0 = self.options["h_diff_x0"]
        if h_diff_x0 is None:
            h_diff_x0 = np.eye(inputs._get_data().size)
        elif isinstance(h_diff_x0, list):
            h_diff_x0 = h_diff_x0[2]

        tmp = h_diff_x0 @ inputs._get_data()
        grad_partial = 2 * h_diff_x0 @ tmp

        offset = 0
        for input in inputs.keys():
            input_size = inputs[input].size
            partials["error_est", input] = tmp[offset:offset + input_size]
            partials["gradient_error_est", input] = grad_partial[offset:offset+input_size]
            offset += input_size

if __name__ == "__main__":
    import unittest
    import copy

    from calibration import AdditiveCalibration, calibrate
    from openmdao.utils.assert_utils import assert_check_partials

    np.random.seed(0)

    class TestErrorEstimate(unittest.TestCase):
        def test_error_estimate_with_calibrate_with_exact_hessian(self):
            lofi_prob = om.Problem()

            lofi_prob.model.add_subsystem("uncalibrated_z",
                                          om.ExecComp("uncalibrated_z = x + y"),
                                          promotes=["*"])

            lofi_prob.model.add_subsystem("z_cal",
                                          AdditiveCalibration(inputs=["x", "y"],
                                                              order=1),
                                          promotes_inputs=["*"],
                                          promotes_outputs=[("gamma", "z_bias")])

            lofi_prob.model.add_subsystem("z",
                                          om.ExecComp("z = uncalibrated_z + z_bias"),
                                          promotes=["*"])

            lofi_prob.model.add_subsystem("z_error_est",
                                          ErrorEstimate(dvs=["x", "y"]),
                                          promotes_inputs=["*"],
                                          promotes_outputs=[("error_est", "z_error_est")])
            lofi_prob.setup()

            hifi_prob = om.Problem()
            hifi_prob.model.add_subsystem("z",
                                          om.ExecComp("z = x + y + 0.05*x**0.5 + 0.05*y**0.5"),
                                          promotes=["*"])
            hifi_prob.setup()

            lofi_prob["x"] = 0.1
            lofi_prob["y"] = 0.1
            lofi_prob.run_model()

            hifi_prob["x"] = 0.1
            hifi_prob["y"] = 0.1
            hifi_prob.run_model()

            x0 = {"x": copy.deepcopy(lofi_prob["x"]), "y": copy.deepcopy(lofi_prob["y"])}
            calibrate(lofi_prob,
                      hifi_prob,
                      ["z"],
                      x0,
                      include_error_est=True)

            lofi_prob.model.z_error_est.options["h_diff_x0"] = np.array([[0.0125/0.2**1.5, 0.0],[0.0, 0.0125/0.2**1.5]])

            lofi_prob["x"] = 0.2
            lofi_prob["y"] = 0.2
            lofi_prob.run_model()

            hifi_prob["x"] = 0.2
            hifi_prob["y"] = 0.2
            hifi_prob.run_model()

            # lofi_prob.model.list_inputs(units=True, prom_name=True)
            # lofi_prob.model.list_outputs(residuals=True, units=True, prom_name=True)
            # hifi_prob.model.list_outputs(residuals=True, units=True, prom_name=True)

            # self.assertAlmostEqual(lofi_prob["z_error_est"][0], 0.00279508)
            self.assertAlmostEqual(lofi_prob["z_error_est"][0], 0.001397542486)
            # self.assertAlmostEqual(lofi_prob["z_error_est"][0], 1.9531249999999996e-06)

        def test_error_estimate(self):
            lofi_prob = om.Problem()

            lofi_prob.model.add_subsystem("uncalibrated_z",
                                          om.ExecComp("uncalibrated_z = x + y"),
                                          promotes=["*"])

            lofi_prob.model.add_subsystem("z_cal",
                                          AdditiveCalibration(inputs=["x", "y"],
                                                              order=1),
                                          promotes_inputs=["*"],
                                          promotes_outputs=[("gamma", "z_bias")])

            lofi_prob.model.add_subsystem("z",
                                          om.ExecComp("z = uncalibrated_z + z_bias"),
                                          promotes=["*"])

            lofi_prob.model.add_subsystem("z_error_est",
                                          ErrorEstimate(inputs=["x", "y"],
                                                        order=1),
                                          promotes_inputs=["*"],
                                          promotes_outputs=[("error_est", "z_error_est")])
            lofi_prob.setup()

            hifi_prob = om.Problem()
            hifi_prob.model.add_subsystem("z",
                                          om.ExecComp("z = x + y + 0.05*x**0.5 + 0.05*y**0.5"),
                                          promotes=["*"])
            hifi_prob.setup()

            lofi_prob["x"] = 1.1
            lofi_prob["y"] = 1.1
            lofi_prob.run_model()

            hifi_prob["x"] = 1.1
            hifi_prob["y"] = 1.1
            hifi_prob.run_model()

            x0 = {"x": copy.deepcopy(lofi_prob["x"]), "y": copy.deepcopy(lofi_prob["y"])}
            calibrate(lofi_prob,
                      hifi_prob,
                      ["z"],
                      x0,
                      include_error_est=False)

            lofi_totals = lofi_prob.compute_totals(["z", "uncalibrated_z"], ["x", "y"])
            g_lofi_x0 = {input: copy.deepcopy(lofi_totals["z", input]) for input in x0.keys()}
            h_lofi_x0 = hessian(lofi_prob, "z", x0, g_lofi_x0)
            # h_lofi_x0 = np.zeros([2,2])
            # print(f"h_lofi_x:\n{h_lofi_x0}")

            hifi_totals = hifi_prob.compute_totals("z", ["x", "y"])
            g_hifi_x0 = {input: copy.deepcopy(hifi_totals["z", input]) for input in x0.keys()}
            h_hifi_x0 = hessian(hifi_prob, "z", x0, g_hifi_x0)
            # print(f"h_hifi_x:\n{h_hifi_x0}")

            h_diff_x0 = h_lofi_x0 - h_hifi_x0
            lofi_prob.model.z_error_est.options["x0"] = x0
            lofi_prob.model.z_error_est.options["h_diff_x0"] = h_diff_x0

            lofi_prob["x"] = 1.2
            lofi_prob["y"] = 1.2
            lofi_prob.run_model()

            hifi_prob["x"] = 1.2
            hifi_prob["y"] = 1.2
            hifi_prob.run_model()

            # lofi_prob.model.list_inputs(units=True, prom_name=True)
            # lofi_prob.model.list_outputs(residuals=True, units=True, prom_name=True)
            # hifi_prob.model.list_outputs(residuals=True, units=True, prom_name=True)

            # self.assertAlmostEqual(lofi_prob["z_error_est"][0], 0.0002166961835370615)
            self.assertAlmostEqual(lofi_prob["z_error_est"][0], 0.0001083480071)
            # self.assertAlmostEqual(lofi_prob["z_error_est"][0], 1.1739290632948382e-08)

        def test_error_estimate_with_calibrate(self):
            lofi_prob = om.Problem()

            lofi_prob.model.add_subsystem("z_cal",
                                          AdditiveCalibration(inputs=["x", "y"],
                                                              order=1),
                                          promotes_inputs=["*"],
                                          promotes_outputs=[("gamma", "z_bias")])

            lofi_prob.model.add_subsystem("uncalibrated_z",
                                          om.ExecComp("uncalibrated_z = x + y"),
                                          promotes=["*"])

            lofi_prob.model.add_subsystem("z",
                                          om.ExecComp("z = uncalibrated_z + z_bias"),
                                          promotes=["*"])

            lofi_prob.model.add_subsystem("z_error_est",
                                          ErrorEstimate(inputs=["x", "y"],
                                                        order=1),
                                          promotes_inputs=["*"],
                                          promotes_outputs=[("error_est", "z_error_est")])
            lofi_prob.setup()

            hifi_prob = om.Problem()
            hifi_prob.model.add_subsystem("z",
                                          om.ExecComp("z = x + y + 0.05*x**0.5 + 0.05*y**0.5"),
                                          promotes=["*"])
            hifi_prob.setup()

            lofi_prob["x"] = 1.1
            lofi_prob["y"] = 1.1
            lofi_prob.run_model()

            hifi_prob["x"] = 1.1
            hifi_prob["y"] = 1.1
            hifi_prob.run_model()

            x0 = {"x": copy.deepcopy(lofi_prob["x"]), "y": copy.deepcopy(lofi_prob["y"])}
            calibrate(lofi_prob,
                      hifi_prob,
                      ["z"],
                      x0,
                      include_error_est=True)

            lofi_prob["x"] = 1.2
            lofi_prob["y"] = 1.2
            lofi_prob.run_model()

            hifi_prob["x"] = 1.2
            hifi_prob["y"] = 1.2
            hifi_prob.run_model()

            # lofi_prob.model.list_inputs(units=True, prom_name=True)
            # lofi_prob.model.list_outputs(residuals=True, units=True, prom_name=True)
            # hifi_prob.model.list_outputs(residuals=True, units=True, prom_name=True)

            # self.assertAlmostEqual(lofi_prob["z_error_est"][0], 0.00021669618)
            self.assertAlmostEqual(lofi_prob["z_error_est"][0], 0.0001083480071)
            # self.assertAlmostEqual(lofi_prob["z_error_est"][0], 1.1739290632948382e-08)

    class TestErrorEstimatePartials(unittest.TestCase):
        def test_zero_order_error_estimate_partials(self):
            lofi_prob = om.Problem()

            lofi_prob.model.add_subsystem("z_cal",
                                          AdditiveCalibration(inputs=["x", "y"],
                                                              order=0),
                                          promotes_inputs=["*"],
                                          promotes_outputs=[("gamma", "z_bias")])

            lofi_prob.model.add_subsystem("uncalibrated_z",
                                          om.ExecComp("uncalibrated_z = x + y"),
                                          promotes=["*"])

            lofi_prob.model.add_subsystem("z",
                                          om.ExecComp("z = uncalibrated_z + z_bias"),
                                          promotes=["*"])

            lofi_prob.model.add_subsystem("z_error_est",
                                          ErrorEstimate(inputs=["x", "y"],
                                                        order=0),
                                          promotes_inputs=["*"],
                                          promotes_outputs=[("error_est", "z_error_est")])
            lofi_prob.setup(force_alloc_complex=True)

            hifi_prob = om.Problem()
            hifi_prob.model.add_subsystem("z",
                                          om.ExecComp("z = x + y + 0.05*x**0.5 + 0.05*y**0.5"),
                                          promotes=["*"])
            hifi_prob.setup()

            lofi_prob["x"] = 1.1
            lofi_prob["y"] = 1.1
            lofi_prob.run_model()

            hifi_prob["x"] = 1.1
            hifi_prob["y"] = 1.1
            hifi_prob.run_model()

            x0 = {"x": copy.deepcopy(lofi_prob["x"]), "y": copy.deepcopy(lofi_prob["y"])}
            calibrate(lofi_prob,
                      hifi_prob,
                      ["z"],
                      x0,
                      include_error_est=True)

            lofi_prob["x"] = 1.2
            lofi_prob["y"] = 1.2
            lofi_prob.run_model()

            hifi_prob["x"] = 1.2
            hifi_prob["y"] = 1.2
            hifi_prob.run_model()

            data = lofi_prob.check_partials(method='fd', form="central")
            assert_check_partials(data, atol=1.e-6, rtol=1.e-6)

        def test_error_estimate_partials(self):
            lofi_prob = om.Problem()

            lofi_prob.model.add_subsystem("z_cal",
                                          AdditiveCalibration(inputs=["x", "y"],
                                                              order=1),
                                          promotes_inputs=["*"],
                                          promotes_outputs=[("gamma", "z_bias")])

            lofi_prob.model.add_subsystem("uncalibrated_z",
                                          om.ExecComp("uncalibrated_z = x + y"),
                                          promotes=["*"])

            lofi_prob.model.add_subsystem("z",
                                          om.ExecComp("z = uncalibrated_z + z_bias"),
                                          promotes=["*"])

            lofi_prob.model.add_subsystem("z_error_est",
                                          ErrorEstimate(inputs=["x", "y"],
                                                        order=1),
                                          promotes_inputs=["*"],
                                          promotes_outputs=[("error_est", "z_error_est")])
            lofi_prob.setup(force_alloc_complex=True)

            hifi_prob = om.Problem()
            hifi_prob.model.add_subsystem("z",
                                          om.ExecComp("z = x + y + 0.05*x**0.5 + 0.05*y**0.5"),
                                          promotes=["*"])
            hifi_prob.setup()

            lofi_prob["x"] = 1.1
            lofi_prob["y"] = 1.1
            lofi_prob.run_model()

            hifi_prob["x"] = 1.1
            hifi_prob["y"] = 1.1
            hifi_prob.run_model()

            x0 = {"x": copy.deepcopy(lofi_prob["x"]), "y": copy.deepcopy(lofi_prob["y"])}
            calibrate(lofi_prob,
                      hifi_prob,
                      ["z"],
                      x0,
                      include_error_est=True)

            lofi_prob["x"] = 1.2
            lofi_prob["y"] = 1.2
            lofi_prob.run_model()

            hifi_prob["x"] = 1.2
            hifi_prob["y"] = 1.2
            hifi_prob.run_model()

            data = lofi_prob.check_partials(method='fd', form="central")
            assert_check_partials(data, atol=1.e-6, rtol=1.e-6)

    class TestApproximateHessian(unittest.TestCase):
        def test_approximate_hessian(self):
            lofi_prob = om.Problem()

            lofi_prob.model.add_subsystem("z_cal",
                                          AdditiveCalibration(inputs=["x", "y"],
                                                              order=1),
                                          promotes_inputs=["*"],
                                          promotes_outputs=[("gamma", "z_bias")])

            lofi_prob.model.add_subsystem("uncalibrated_z",
                                          om.ExecComp("uncalibrated_z = x + y"),
                                          promotes=["*"])

            lofi_prob.model.add_subsystem("z",
                                          om.ExecComp("z = uncalibrated_z + z_bias"),
                                          promotes=["*"])

            lofi_prob.model.add_subsystem("z_error_est",
                                          ErrorEstimate(inputs=["x", "y"],
                                                        order=1),
                                          promotes_inputs=["*"],
                                          promotes_outputs=[("error_est", "z_error_est")])
            lofi_prob.setup()

            hifi_prob = om.Problem()
            hifi_prob.model.add_subsystem("z",
                                          om.ExecComp("z = x + y + 0.05*x**0.5 + 0.05*y**0.5"),
                                          promotes=["*"])
            hifi_prob.setup()

            lofi_prob["x"] = 0.1
            lofi_prob["y"] = 0.1
            lofi_prob.run_model()

            hifi_prob["x"] = 0.1
            hifi_prob["y"] = 0.1
            hifi_prob.run_model()

            lofi_prob["x"] = 0.2
            lofi_prob["y"] = 0.2
            lofi_prob.run_model()

            hifi_prob["x"] = 0.2
            hifi_prob["y"] = 0.2
            hifi_prob.run_model()

            x0 = {"x": copy.deepcopy(lofi_prob["x"]), "y": copy.deepcopy(lofi_prob["y"])}

            lofi_totals = lofi_prob.compute_totals(["z", "uncalibrated_z"], ["x", "y"])
            hifi_totals = hifi_prob.compute_totals("z", ["x", "y"])
            hessian_diff = hessian_differences(lofi_prob,
                                               hifi_prob,
                                               ['z'],
                                               ['z'],
                                               x0,
                                               lofi_totals,
                                               hifi_totals)
            exact_hessian_diff = np.array([[0.0125/0.2**1.5, 0.0],[0.0, 0.0125/0.2**1.5]])

            # print(f"Hessian diff:\n{hessian_diff[:,:,0]}")
            # print(f"Exact Hessian diff:\n{exact_hessian_diff}")

            np.testing.assert_allclose(hessian_diff[:, :, 0], exact_hessian_diff, rtol=1e-4)

        def test_approximate_hessian_calibrate(self):
            lofi_prob = om.Problem()

            lofi_prob.model.add_subsystem("z_cal",
                                          AdditiveCalibration(inputs=["x", "y"],
                                                              order=1),
                                          promotes_inputs=["*"],
                                          promotes_outputs=[("gamma", "z_bias")])

            lofi_prob.model.add_subsystem("uncalibrated_z",
                                          om.ExecComp("uncalibrated_z = x + y"),
                                          promotes=["*"])

            lofi_prob.model.add_subsystem("z",
                                          om.ExecComp("z = uncalibrated_z + z_bias"),
                                          promotes=["*"])

            lofi_prob.model.add_subsystem("z_error_est",
                                          ErrorEstimate(inputs=["x", "y"],
                                                        order=1),
                                          promotes_inputs=["*"],
                                          promotes_outputs=[("error_est", "z_error_est")])
            lofi_prob.setup()

            hifi_prob = om.Problem()
            hifi_prob.model.add_subsystem("z",
                                          om.ExecComp("z = x + y + 0.05*x**0.5 + 0.05*y**0.5"),
                                          promotes=["*"])
            hifi_prob.setup()

            lofi_prob["x"] = 0.2
            lofi_prob["y"] = 0.2
            lofi_prob.run_model()

            hifi_prob["x"] = 0.2
            hifi_prob["y"] = 0.2
            hifi_prob.run_model()

            x0 = {"x": copy.deepcopy(lofi_prob["x"]), "y": copy.deepcopy(lofi_prob["y"])}
            calibrate(lofi_prob,
                      hifi_prob,
                      ['z'],
                      x0,
                      include_error_est=True,
                      direct_hessian_diff=True)

            hessian_diff = lofi_prob.model.z_error_est.options['h_diff_x0']
            exact_hessian_diff = np.array([[0.0125/0.2**1.5, 0.0],[0.0, 0.0125/0.2**1.5]])

            # print(f"Hessian diff:\n{hessian_diff}")
            # print(f"Exact Hessian diff:\n{exact_hessian_diff}")

            np.testing.assert_allclose(hessian_diff, exact_hessian_diff, rtol=1e-4)

    class TestSR1ErrorEstimate(unittest.TestCase):
        def test_sr1_error_estimate_with_calibrate(self):
            lofi_prob = om.Problem()

            lofi_prob.model.add_subsystem("z_cal",
                                          AdditiveCalibration(inputs=["x", "y"],
                                                              order=1),
                                          promotes_inputs=["*"],
                                          promotes_outputs=[("gamma", "z_bias")])

            lofi_prob.model.add_subsystem("uncalibrated_z",
                                          om.ExecComp("uncalibrated_z = x + y"),
                                          promotes=["*"])

            lofi_prob.model.add_subsystem("z",
                                          om.ExecComp("z = uncalibrated_z + z_bias"),
                                          promotes=["*"])

            lofi_prob.model.add_subsystem("z_error_est",
                                          ErrorEstimate(inputs=["x", "y"],
                                                        order=1),
                                          promotes_inputs=["*"],
                                          promotes_outputs=[("error_est", "z_error_est")])
            lofi_prob.setup()

            hifi_prob = om.Problem()
            hifi_prob.model.add_subsystem("z",
                                          om.ExecComp("z = x + y + 0.05*x**0.5 + 0.05*y**0.5"),
                                          promotes=["*"])
            hifi_prob.setup()

            lofi_prob["x"] = 1.1
            lofi_prob["y"] = 1.1
            lofi_prob.run_model()

            hifi_prob["x"] = 1.1
            hifi_prob["y"] = 1.1
            hifi_prob.run_model()

            x0 = {"x": copy.deepcopy(lofi_prob["x"]), "y": copy.deepcopy(lofi_prob["y"])}
            calibrate(lofi_prob,
                      hifi_prob,
                      ["z"],
                      x0,
                      include_error_est=True,
                      direct_hessian_diff=True,
                      sr1_hessian_diff=True)

            lofi_prob["x"] = 1.3
            lofi_prob["y"] = 1.3
            lofi_prob.run_model()

            hifi_prob["x"] = 1.3
            hifi_prob["y"] = 1.3
            hifi_prob.run_model()

            x0 = {"x": copy.deepcopy(lofi_prob["x"]), "y": copy.deepcopy(lofi_prob["y"])}
            calibrate(lofi_prob,
                      hifi_prob,
                      ["z"],
                      x0,
                      include_error_est=True,
                      direct_hessian_diff=True,
                      sr1_hessian_diff=True)

            lofi_prob["x"] = 1.15
            lofi_prob["y"] = 1.15
            lofi_prob.run_model()

            hifi_prob["x"] = 1.15
            hifi_prob["y"] = 1.15
            hifi_prob.run_model()

            x0 = {"x": copy.deepcopy(lofi_prob["x"]), "y": copy.deepcopy(lofi_prob["y"])}
            calibrate(lofi_prob,
                      hifi_prob,
                      ["z"],
                      x0,
                      include_error_est=True,
                      direct_hessian_diff=True,
                      sr1_hessian_diff=True)
            
            lofi_prob["x"] = 1.1
            lofi_prob["y"] = 1.1
            lofi_prob.run_model()

            hifi_prob["x"] = 1.1
            hifi_prob["y"] = 1.1
            hifi_prob.run_model()

            x0 = {"x": copy.deepcopy(lofi_prob["x"]), "y": copy.deepcopy(lofi_prob["y"])}
            calibrate(lofi_prob,
                      hifi_prob,
                      ["z"],
                      x0,
                      include_error_est=True,
                      direct_hessian_diff=True,
                      sr1_hessian_diff=True)

            lofi_prob["x"] = 1.2
            lofi_prob["y"] = 1.2
            lofi_prob.run_model()

            hifi_prob["x"] = 1.2
            hifi_prob["y"] = 1.2
            hifi_prob.run_model()
            # lofi_prob.model.list_inputs(units=True, prom_name=True)
            # lofi_prob.model.list_outputs(residuals=True, units=True, prom_name=True)
            # hifi_prob.model.list_outputs(residuals=True, units=True, prom_name=True)

            # self.assertAlmostEqual(lofi_prob["z_error_est"][0], 0.00021669618)
            self.assertAlmostEqual(lofi_prob["z_error_est"][0], 0.0001083480071)
            # self.assertAlmostEqual(lofi_prob["z_error_est"][0], 1.1739290632948382e-08)

    unittest.main()