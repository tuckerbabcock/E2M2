import copy
import numbers

import numpy as np
from scipy.optimize import SR1

import openmdao.api as om

def _hessian_vector_product(problem, output, x0, g0, pert, delta):
    """
    Finite difference Hessian-vector product
    """
    offset = 0
    for input in x0.keys():
        input_size = x0[input].size
        problem[input] = x0[input] + delta * pert[offset:offset+input_size]
        offset += input_size
    
    problem.run_model()
    totals = problem.compute_totals(output, [*x0.keys()])
    output_totals = {input: copy.deepcopy(totals[output, input]) for input in x0.keys()}
    hvp = {input: (output_totals[input] - g0[input])/delta for input in x0.keys()}

    hvp_vec = np.zeros(offset)
    offset = 0
    for input in hvp.keys():
        input_size = hvp[input].size
        hvp_vec[offset:offset + input_size] = hvp[input]
        offset += input_size

    return hvp_vec

def _arnoldi_iterations(mat_vec, g, n, tol):
    """
    Use the Hessian-vector product function to create a reduced Hessian approximation
    """
    m = g.shape[0]
    if n > m:
        n = m

    H = np.zeros([n+1, n])
    Q = np.zeros([m, n+1])

    Q[:, 0] = -g / np.linalg.norm(g, 2)
    for i in range(0, n):
        Q[:, i+1] = mat_vec(Q[:, i])
        for j in range(0, i+1): # Modified Gram-Schmidt
            H[j, i] = np.dot(Q[:, i+1], Q[:, j])
            Q[:, i+1] -= H[j, i] * Q[:, j]
        H[i+1, i] = np.linalg.norm(Q[:, i+1], 2)
        if H[i+1, i] < tol:
            # print("Arnoldi Stopping!")
            return Q, H
        Q[:, i+1] /= H[i+1, i]

    return Q, H

def reduced_hessian(problem, output, x0, g0, delta=1e-7, n=None, tol=1e-14):
    """
    Return an approximate Hessian based on the Arnoldi procedure
    """
    mat_vec = lambda z: _hessian_vector_product(problem, output, x0, g0, z, delta)

    if n is None:
        n = len(x0)

    g0_size = 0
    for input in g0.keys():
        g0_size += g0[input].size

    g0_vec = np.zeros(g0_size)
    offset = 0
    for input in g0.keys():
        input_size = x0[input].size
        g0_vec[offset:offset + input_size] = g0[input]
        offset += input_size
    g0_vec = np.random.normal(size=g0_size)
    # g0_vec = np.array([-1, 0])

    Qnp1, Hn = _arnoldi_iterations(mat_vec, g0_vec, n, tol)
    # print(f"Qnp1:\n{Qnp1}")
    # print(f"Hn:\n{Hn}")
    Qn = Qnp1[:,:-1]

    hessian_approx = Qn @ Qn.T @ Qnp1 @ Hn @ Qn.T
    return 0.5*(hessian_approx + hessian_approx.T)

def hessian(problem, output, x0, g0):
    return reduced_hessian(problem, output, x0, g0)

def _hessian_difference_vector_product(lofi_prob,
                                       hifi_prob,
                                       lofi_outputs,
                                       hifi_outputs,
                                       x0,
                                       lofi_g0s,
                                       hifi_g0s,
                                       pert,
                                       delta):
    """
    Finite differenced Hessian-difference-vector products
    """

    if isinstance(delta, numbers.Number):
        delta = {input: delta for input in x0.keys()}
    
    offset = 0
    for input in x0.keys():
        input_size = x0[input].size
        lofi_prob[input] = x0[input] + delta[input] * pert[offset:offset+input_size]
        hifi_prob[input] = x0[input] + delta[input] * pert[offset:offset+input_size]
        offset += input_size
    
    lofi_prob.run_model()
    lofi_totals = lofi_prob.compute_totals(lofi_outputs, [*x0.keys()])

    hifi_prob.run_model()
    hifi_totals = hifi_prob.compute_totals(hifi_outputs, [*x0.keys()])

    n_outputs = len(x0)
    n = pert.size
    hvp_vec = np.zeros([n, n_outputs])

    for i, (lofi_output, hifi_output) in enumerate(zip(lofi_outputs, hifi_outputs)):

        hvp = {input: (lofi_totals[lofi_output, input] - lofi_g0s[lofi_output, input]) / delta[input] \
                    - (hifi_totals[hifi_output, input] - hifi_g0s[hifi_output, input]) / delta[input] \
               for input in x0.keys()}

        offset = 0
        for input in hvp.keys():
            input_size = hvp[input].size
            hvp_vec[offset:offset + input_size, i] = hvp[input]
            offset += input_size

    return hvp_vec

def approximate_hessian_differences(lofi_prob,
                                    hifi_prob,
                                    lofi_outputs,
                                    hifi_outputs,
                                    x0,
                                    lofi_g0s,
                                    hifi_g0s,
                                    delta=1e-5):
    """
    Return an approximate Hessian difference based on the finite-differencing
    """

    n_outputs = len(x0)
    n = len(x0)
    h_diffs = np.zeros([n, n, n_outputs])

    for i in range(n):
        pert = np.zeros(n)
        pert[i] = 1.0
        hess_diff_rows = _hessian_difference_vector_product(lofi_prob,
                                                            hifi_prob,
                                                            lofi_outputs,
                                                            hifi_outputs,
                                                            x0,
                                                            lofi_g0s,
                                                            hifi_g0s,
                                                            pert,
                                                            delta)

        h_diffs[i, :, :] = hess_diff_rows[:, :]
        pert[i] = 0.0

    return h_diffs

def sr1_hessian_differences(lofi_outputs,
                            hifi_outputs,
                            x0,
                            lofi_g0s,
                            hifi_g0s,
                            old_differences):
    # n = len(x0)
    n_outputs = len(lofi_outputs)
    n = 0
    for input in x0.keys():
        n += x0[input].size

    for i, (lofi_output, hifi_output) in enumerate(zip(lofi_outputs, hifi_outputs)):
        # print(f"i: {i}, lofi_output: {lofi_output}, hifi_output: {hifi_output}")
        # print(f"old_diff[{i}]: {old_differences[i]}")
        grad_diff = {input: lofi_g0s[lofi_output, input] - hifi_g0s[hifi_output, input] \
                     for input in x0.keys()}

        grad_diff_kp1 = np.zeros([n])
        offset = 0
        for input in grad_diff.keys():
            input_size = grad_diff[input].size
            grad_diff_kp1[offset:offset + input_size] = grad_diff[input]
            offset += input_size

        # print(grad_diff_kp1)

        # print(f"x0: {x0}")
        x_kp1 = np.zeros(n)
        offset = 0
        for input in x0.keys():
                input_size = x0[input].size
                x_kp1[offset:offset + input_size] = x0[input]
                offset += input_size
        # print(f"x_kp1: {x_kp1}")

        h_diff = old_differences[i]
        if h_diff is None:
            sr1 = SR1()
            sr1.initialize(n, 'hess')
            old_differences[i] = [sr1, grad_diff_kp1, x_kp1, np.copy(sr1.get_matrix())]
            # print(f"continue!")
            continue

        grad_diff_k = h_diff[1]
        x_k = h_diff[2]
        # print(f"x_k: {x_k}")

        # print(f"x_kp1 - x_k: {x_kp1 - x_k}")
        # print(f"grad_diff_kp1 - grad_diff_k: {grad_diff_kp1 - grad_diff_k}")
        h_diff[0].update(x_kp1 - x_k, grad_diff_kp1 - grad_diff_k)
        h_diff[1] = grad_diff_kp1
        h_diff[2] = x_kp1
        h_diff[3] = np.copy(h_diff[0].get_matrix())

    # print(old_differences)

    return old_differences


def hessian_differences(lofi_prob, hifi_prob, lofi_outputs, hifi_outputs, x0, lofi_g0s, hifi_g0s, old_differences=None, sr1_hessian_diff=False):
    if not sr1_hessian_diff:
        return approximate_hessian_differences(lofi_prob, hifi_prob, lofi_outputs, hifi_outputs, x0, lofi_g0s, hifi_g0s)
    else:
        return sr1_hessian_differences(lofi_outputs, hifi_outputs, x0, lofi_g0s, hifi_g0s, old_differences)

class ErrorEstimate(om.ExplicitComponent):
    """
    ErrorEstimate estimates the *squared* error between a low-fidelity model
    and a high-fidelity model by linearizing the difference between the models
    """
    def initialize(self):
        self.options.declare("inputs",
                             types=(str, list, dict),
                             desc="Names of inputs to take gradient with respect to.")

        self.options.declare("x0",
                             types=dict,
                             default=None,
                             desc="Dictionary of inputs and their values at the calibration point.")

        self.options.declare("f_hifi_x0",
                             default=1.0,
                             desc="High-fidelity model value at the calibration point")

        self.options.declare("g_diff_x0",
                             default=None,
                             desc="Difference in gradients at the calibration point. Low-fidelity - high-fidelity.")

        self.options.declare("h_diff_x0",
                             default=None,
                             desc="Difference in Hessians at the calibration point. Low-fidelity - high-fidelity.")

        self.options.declare("order",
                             default=0,
                             types=int,
                             desc="Calibration order. 0th order calibrates values, 1st order also calibrates gradients")

    def setup(self):
        inputs = self.options["inputs"]
        if isinstance(inputs, str):
            self.add_input(inputs)
            # self.declare_partials("error_est", inputs, method='cs')
            self.declare_partials("error_est", inputs)
            self.declare_partials("relative_error_est", inputs)
        elif isinstance(inputs, list):
            for input in inputs:
                if isinstance(input, str):
                    self.add_input(input)
                    # self.declare_partials("error_est", input, method='cs')
                    self.declare_partials("error_est", input)
                    self.declare_partials("relative_error_est", input)
                else:
                    raise RuntimeError(f"Input: {input} supplied to Calibration is not a string!")
                
        elif isinstance(inputs, dict):
            for input, input_opts in inputs.items():
                if not isinstance(input, str):
                    raise RuntimeError(f"Input: {input} supplied to Calibration is not a string!")
                val = input_opts.get("val", 1)
                shape = input_opts.get("shape", 1)
                units = input_opts.get("units", None)
                self.add_input(input, val=val, shape=shape, units=units)
                # self.declare_partials("error_est", input, method='cs')
                self.declare_partials("error_est", input)
                self.declare_partials("relative_error_est", input)

        self.add_output("error_est", val=0.0, desc="Squared error estimate")
        self.add_output("relative_error_est", val=0.0, desc="Squared relative error estimate")

    def compute(self, inputs, outputs):
        order = self.options["order"]
        x0 = self.options["x0"]
        if x0 is None:
            return

        error_est = 0.0
        if order == 0:
            g_diff_x0 = self.options["g_diff_x0"]
            if g_diff_x0 is None:
                return
            
            x0_size = 0
            for input in inputs.keys():
                x0_size += x0[input].size

            x_diff = np.zeros(x0_size)
            g_diff_x0_vec = np.zeros(x0_size)
            offset = 0
            for input in inputs.keys():
                # # Old version of error est is general linear constraint
                # x_diff = inputs[input] - x0[input]
                # error_est += np.dot(g_diff_x0[input], x_diff)

                ## New version bounds step based on Cauchy-Schwarz inequality
                input_size = x0[input].size
                x_diff[offset:offset + input_size] = inputs[input] - x0[input]
                g_diff_x0_vec[offset:offset + input_size] = g_diff_x0[input]
                offset += input_size

            # error_est = np.linalg.norm(g_diff_x0_vec)**2 * np.linalg.norm(x_diff)**2
            error_est = np.linalg.norm(g_diff_x0_vec) * np.linalg.norm(x_diff)

        elif order == 1:
            h_diff_x0 = self.options["h_diff_x0"]
            if h_diff_x0 is None:
                return
            elif isinstance(h_diff_x0, list):
                h_diff_x0 = h_diff_x0[3]

            x0_size = 0
            for input in inputs.keys():
                x0_size += x0[input].size

            x_diff = np.zeros(x0_size)
            offset = 0
            for input in inputs.keys():
                input_size = x0[input].size
                x_diff[offset:offset + input_size] = inputs[input] - x0[input]
                offset += input_size

            tmp = h_diff_x0 @ x_diff
            xTHx = np.dot(x_diff, tmp)
            # error_est = (0.5*xTHx)**2
            error_est = 0.5*xTHx

        outputs["error_est"] = error_est

        f_hifi_x0 = self.options["f_hifi_x0"]
        # outputs["relative_error_est"] = error_est / f_hifi_x0**2
        outputs["relative_error_est"] = error_est / np.abs(f_hifi_x0)

    def compute_partials(self, inputs, partials):
        order = self.options["order"]
        x0 = self.options["x0"]
        if x0 is None:
            return

        f_hifi_x0 = self.options["f_hifi_x0"]
        if order == 0:
            g_diff_x0 = self.options["g_diff_x0"]
            if g_diff_x0 is None:
                return
            
            ### partials for old error estimate, still works with the abs around g_diff
            # for input in inputs.keys():
            #     # partials["error_est", input] = np.abs(g_diff_x0[input])
            #     partials["error_est", input] = g_diff_x0[input]
            #     partials["relative_error_est", input] = partials["error_est", input] / f_hifi_x0

            x0_size = 0
            for input in inputs.keys():
                x0_size += x0[input].size

            x_diff = np.zeros(x0_size)
            g_diff_x0_vec = np.zeros(x0_size)
            offset = 0
            for input in inputs.keys():
                # x_diff = inputs[input] - x0[input]
                # error_est += np.dot(g_diff_x0[input], x_diff)

                input_size = x0[input].size
                x_diff[offset:offset + input_size] = inputs[input] - x0[input]
                g_diff_x0_vec[offset:offset + input_size] = g_diff_x0[input]
                offset += input_size
            g_norm = np.linalg.norm(g_diff_x0_vec)
            x_norm = np.linalg.norm(x_diff)

            offset = 0
            for input in inputs.keys():
                # partials["error_est", input] = 2 * g_norm**2 * (inputs[input] - x0[input])
                # partials["relative_error_est", input] = partials["error_est", input] / f_hifi_x0**2
                partials["error_est", input] = g_norm * (inputs[input] - x0[input]) / x_norm
                partials["relative_error_est", input] = partials["error_est", input] / np.abs(f_hifi_x0)

        elif order == 1:
            h_diff_x0 = self.options["h_diff_x0"]
            if h_diff_x0 is None:
                return
            elif isinstance(h_diff_x0, list):
                h_diff_x0 = h_diff_x0[3]

            x0_size = 0
            for input in inputs.keys():
                x0_size += x0[input].size

            x_diff = np.zeros(x0_size)
            offset = 0
            for input in inputs.keys():
                input_size = x0[input].size
                x_diff[offset:offset + input_size] = inputs[input] - x0[input]
                offset += input_size

            tmp = h_diff_x0 @ x_diff
            xTHx = np.dot(x_diff, tmp)

            offset = 0
            for input in inputs.keys():
                input_size = x0[input].size
                # partials["error_est", input] = xTHx * tmp[offset:offset + input_size]
                # partials["relative_error_est", input] = partials["error_est", input] / f_hifi_x0**2
                partials["error_est", input] = tmp[offset:offset + input_size]
                partials["relative_error_est", input] = partials["error_est", input] / np.abs(f_hifi_x0)
                offset += input_size

class PolynomialErrorPenalty(om.ExplicitComponent):
    def initialize(self):
        self.options.declare("p",
                             default=2,
                             desc="Penalty polynomial order, defaults to quadratic")
    def setup(self):
        self.add_input("tau", desc="error tolerance")
        self.add_input("error_est", desc="squared error estimate")

        self.add_output("penalty", val=0.0, desc="Polynomial penalty term")

        self.declare_partials("penalty", ["tau", "error_est"])

    def compute(self, inputs, outputs):
        p = self.options['p']
        tau = inputs["tau"]
        error_est = inputs["error_est"]

        tau_squared = tau**2

        if error_est < tau_squared:
            outputs["penalty"] = 0
        else:
            outputs["penalty"] = (error_est - tau_squared)**p

    def compute_partials(self, inputs, partials):
        p = self.options['p']
        tau = inputs["tau"]
        error_est = inputs["error_est"]

        tau_squared = tau**2

        if error_est < tau_squared:
            partials["penalty", 'tau'] = 0
            partials["penalty", 'error_est'] = 0
        else:
            partials["penalty", 'tau'] = -2*p*(error_est - tau_squared)**(p-1)
            partials["penalty", 'error_est'] = p*(error_est - tau_squared)**(p-1)


if __name__ == "__main__":
    import unittest
    import copy

    from calibration import AdditiveCalibration, calibrate
    from openmdao.utils.assert_utils import assert_check_partials

    np.random.seed(0)

    class TestErrorEstimate(unittest.TestCase):
        def test_zero_order_error_estimate(self):
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
                      x0)

            lofi_prob.model.z_error_est.options["x0"] = x0
            lofi_totals = lofi_prob.compute_totals(["z"], ["x", "y"])
            lofi_z_totals = {
                "x": copy.deepcopy(lofi_totals["z", "x"]),
                "y": copy.deepcopy(lofi_totals["z", "y"])
            }

            hifi_totals = hifi_prob.compute_totals(["z"], ["x", "y"])
            hifi_z_totals = {
                "x": copy.deepcopy(hifi_totals["z", "x"]),
                "y": copy.deepcopy(hifi_totals["z", "y"])
            }

            g_diff = {input: lofi_z_totals[input] - hifi_z_totals[input] for input in lofi_z_totals.keys()}
            lofi_prob.model.z_error_est.options["g_diff_x0"] = g_diff

            lofi_prob["x"] = 0.2
            lofi_prob["y"] = 0.2
            lofi_prob.run_model()

            hifi_prob["x"] = 0.2
            hifi_prob["y"] = 0.2
            hifi_prob.run_model()

            # lofi_prob.model.list_inputs(units=True, prom_name=True)
            # lofi_prob.model.list_outputs(residuals=True, units=True, prom_name=True)
            # hifi_prob.model.list_outputs(residuals=True, units=True, prom_name=True)

            # self.assertAlmostEqual(lofi_prob["z_error_est"][0], -0.01581139)
            self.assertAlmostEqual(lofi_prob["z_error_est"][0], 0.01581139)
            # self.assertAlmostEqual(lofi_prob["z_error_est"][0], 0.00025)

        def test_zero_order_error_estimate_with_calibrate(self):
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

            lofi_prob["x"] = 0.2
            lofi_prob["y"] = 0.2
            lofi_prob.run_model()

            hifi_prob["x"] = 0.2
            hifi_prob["y"] = 0.2
            hifi_prob.run_model()

            # lofi_prob.model.list_inputs(units=True, prom_name=True)
            # lofi_prob.model.list_outputs(residuals=True, units=True, prom_name=True)
            # hifi_prob.model.list_outputs(residuals=True, units=True, prom_name=True)

            # self.assertAlmostEqual(lofi_prob["z_error_est"][0], -0.01581139)
            self.assertAlmostEqual(lofi_prob["z_error_est"][0], 0.01581139)
            # self.assertAlmostEqual(lofi_prob["z_error_est"][0], 0.00025)

        def test_first_order_error_estimate_with_calibrate_with_exact_hessian(self):
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

        def test_first_order_error_estimate(self):
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

        def test_first_order_error_estimate_with_calibrate(self):
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

        def test_first_order_error_estimate_partials(self):
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

    class TestErrorPenaltyPartials(unittest.TestCase):
        def test_error_penalty_partials_inside_bound(self):
            prob = om.Problem()

            prob.model.add_subsystem("error_penalty",
                                     PolynomialErrorPenalty(),
                                     promotes_inputs=["*"],
                                     promotes_outputs=["*"])
            prob.setup(force_alloc_complex=True)

            prob['tau'] = 1
            prob['error_est'] = 0.5

            data = prob.check_partials(method='fd', form="central")
            assert_check_partials(data, atol=1.e-6, rtol=1.e-6)

        def test_error_penalty_partials_outside_bound(self):
            prob = om.Problem()

            prob.model.add_subsystem("error_penalty",
                                     PolynomialErrorPenalty(),
                                     promotes_inputs=["*"],
                                     promotes_outputs=["*"])
            prob.setup(force_alloc_complex=True)

            prob['tau'] = 1
            prob['error_est'] = 1.5

            data = prob.check_partials(method='fd', form="central")
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