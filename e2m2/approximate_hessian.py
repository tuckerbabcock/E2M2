import copy
import numbers

import numpy as np
from scipy.optimize import SR1

import openmdao.api as om

def update_approximate_hessian_difference(gradient_difference, design_step, hessian=None):
    # print(f"Hessian: {hessian}")
    n = design_step.size
    if hessian is None:
        hess = SR1()
        hess.initialize(n, 'hess')
        hessian = [hess, np.copy(gradient_difference), np.copy(hess.get_matrix())]

    else:
        gradient_difference_k_1 = hessian[1]
        # print(f"gradient_update: {gradient_difference - gradient_difference_k_1}")
        hessian[0].update(design_step, gradient_difference - gradient_difference_k_1)
        hessian[1] = np.copy(gradient_difference)

        e, v = np.linalg.eig(hessian[0].get_matrix())

        e = np.abs(e)
        e[e < 1e-12] = 0
        e[np.where(e == 0)[0]] = np.min(e[np.nonzero(e)])

        hessian[2] = v @ np.diag(e) @ v.T

    return hessian

# def approximate_hessian_difference(response_map, lf_totals, hf_totals, design_step):
#     """
#     Approximates the Hessian difference between functions using an SR1 quasi-Newton approximant
#     """


#     for i, (lofi_output, hifi_output) in enumerate(zip(lofi_outputs, hifi_outputs)):
#         # print(f"i: {i}, lofi_output: {lofi_output}, hifi_output: {hifi_output}")
#         # print(f"old_diff[{i}]: {old_differences[i]}")
#         grad_diff = {input: lofi_g0s[lofi_output, input] - hifi_g0s[hifi_output, input] \
#                      for input in x0.keys()}

#         grad_diff_kp1 = np.zeros([n])
#         offset = 0
#         for input in grad_diff.keys():
#             input_size = grad_diff[input].size
#             grad_diff_kp1[offset:offset + input_size] = grad_diff[input]
#             offset += input_size

#         # print(grad_diff_kp1)

#         # print(f"x0: {x0}")
#         x_kp1 = np.zeros(n)
#         offset = 0
#         for input in x0.keys():
#                 input_size = x0[input].size
#                 x_kp1[offset:offset + input_size] = x0[input]
#                 offset += input_size
#         # print(f"x_kp1: {x_kp1}")

#         h_diff = old_differences[i]
#         if h_diff is None:
#             sr1 = SR1()
#             sr1.initialize(n, 'hess')
#             old_differences[i] = [sr1, grad_diff_kp1, x_kp1, np.copy(sr1.get_matrix())]
#             # print(f"continue!")
#             continue

#         grad_diff_k = h_diff[1]
#         x_k = h_diff[2]
#         # print(f"x_k: {x_k}")

#         # print(f"x_kp1 - x_k: {x_kp1 - x_k}")
#         # print(f"grad_diff_kp1 - grad_diff_k: {grad_diff_kp1 - grad_diff_k}")
#         h_diff[0].update(x_kp1, grad_diff_kp1 - grad_diff_k)
#         h_diff[1] = grad_diff_kp1
#         h_diff[2] = x_kp1
#         h_diff[3] = np.copy(h_diff[0].get_matrix())


#     # if sr1_hessian_diff:
#     #     lofi_totals = lofi_problem.compute_totals(lofi_raw_outputs, [*inputs.keys()], driver_scaling=True)
#     # else:
#     #     lofi_totals = lofi_problem.compute_totals(lofi_calibrated_outputs, [*inputs.keys()])
#     # hifi_totals = hifi_problem.compute_totals(hifi_outputs, [*inputs.keys()], driver_scaling=True)

#     # # print(f"lofi_totals: {lofi_totals}")
#     # # print(f"hifi_totals: {hifi_totals}")
#     # for lofi_output, hifi_output in zip(lofi_calibrated_outputs, hifi_outputs):
#     #     error_est = getattr(lofi_problem.model, f"{hifi_output}_error_est")
#     #     # error_est.options["x0"] = copy.deepcopy(inputs)
#     #     # error_est.options["f_hifi_x0"] = copy.deepcopy(hifi_problem[hifi_output])
    
#     # lofi_hess_outputs = []
#     # hifi_hess_outputs = []
#     # if sr1_hessian_diff:
#     #     for lofi_output, hifi_output in zip(lofi_raw_outputs, hifi_outputs):
#     #         if cal_orders[hifi_output] > 0:
#     #             lofi_hess_outputs.append(lofi_output)
#     #             hifi_hess_outputs.append(hifi_output)
#     # else:
#     #     for lofi_output, hifi_output in zip(lofi_calibrated_outputs, hifi_outputs):
#     #         if cal_orders[hifi_output] > 0:
#     #             lofi_hess_outputs.append(lofi_output)
#     #             hifi_hess_outputs.append(hifi_output)

#     # if sr1_hessian_diff:
#     #     old_diffs = []
#     #     for i, output in enumerate(hifi_outputs):
#     #         error_est = getattr(lofi_problem.model, f"{output}_error_est")
#     #         old_diffs.append(error_est.options["h_diff_x0"])
#     # else:
#     #     old_diffs = None

#     # # print(lofi_hess_outputs)
#     # if len(lofi_hess_outputs) > 0:
#     #     hessian_diffs = hessian_differences(lofi_problem,
#     #                                         hifi_problem,
#     #                                         lofi_hess_outputs,
#     #                                         hifi_hess_outputs,
#     #                                         # lofi_calibrated_outputs,
#     #                                         # hifi_outputs,
#     #                                         inputs,
#     #                                         lofi_totals,
#     #                                         hifi_totals,
#     #                                         old_diffs,
#     #                                         sr1_hessian_diff)

#     #     for i, output in enumerate(hifi_outputs):
#     #         error_est = getattr(lofi_problem.model, f"{output}_error_est")
#     #         if not sr1_hessian_diff:
#     #             h_diff_x0 = hessian_diffs[:,:, i]

#     #             h_diff_x0 = 0.5*(h_diff_x0+h_diff_x0.T)
#     #             # print(f"Hessian difference:\n{h_diff_x0}")

#     #             # make Hessian diff positive definite
#     #             e, v = np.linalg.eig(h_diff_x0)

#     #             # print(f"Hessian difference eigenvalues: {e}")
#     #             e = np.abs(e)
#     #             min_eigval = np.min(e)
#     #             for i, eigval in enumerate(e):
#     #                 if eigval < 1e-6:
#     #                     e[i] = min_eigval
#     #             h_diff_x0 = v @ np.diag(e) @ v.T

#     #             # print(f"Hessian difference (SPD):\n{h_diff_x0}")

#     #             error_est.options["h_diff_x0"] = h_diff_x0
#     #         else:
#     #             h_diff_x0 = hessian_diffs[i]
#     #             h_diff_mat = h_diff_x0[3]
#     #             h_diff_mat = 0.5*(h_diff_mat+h_diff_mat.T)
#     #             # print(f"Hessian difference:\n{h_diff_mat}")

#     #             # make Hessian diff positive definite
#     #             e, v = np.linalg.eig(h_diff_mat)
#     #             e = e.real
#     #             v = v.real
#     #             # print(f"Hessian difference eigenvalues: {e}")
#     #             e = np.abs(e.real)
#     #             min_eigval = np.min(e)
#     #             for i, eigval in enumerate(e):
#     #                 if eigval < 1e-6:
#     #                     e[i] = min_eigval
#     #             h_diff_mat = v @ np.diag(e) @ v.T
#     #             # print(f"Hessian difference (SPD):\n{h_diff_mat}")
#     #             h_diff_x0[3] = h_diff_mat

#     #             error_est.options["h_diff_x0"] = h_diff_x0

# def _hessian_vector_product(problem, output, x0, g0, pert, delta):
#     """
#     Finite difference Hessian-vector product
#     """
#     offset = 0
#     for input in x0.keys():
#         input_size = x0[input].size
#         problem[input] = x0[input] + delta * pert[offset:offset+input_size]
#         offset += input_size
    
#     problem.run_model()
#     totals = problem.compute_totals(output, [*x0.keys()])
#     output_totals = {input: copy.deepcopy(totals[output, input]) for input in x0.keys()}
#     hvp = {input: (output_totals[input] - g0[input])/delta for input in x0.keys()}

#     hvp_vec = np.zeros(offset)
#     offset = 0
#     for input in hvp.keys():
#         input_size = hvp[input].size
#         hvp_vec[offset:offset + input_size] = hvp[input]
#         offset += input_size

#     return hvp_vec

# def _arnoldi_iterations(mat_vec, g, n, tol):
#     """
#     Use the Hessian-vector product function to create a reduced Hessian approximation
#     """
#     m = g.shape[0]
#     if n > m:
#         n = m

#     H = np.zeros([n+1, n])
#     Q = np.zeros([m, n+1])

#     Q[:, 0] = -g / np.linalg.norm(g, 2)
#     for i in range(0, n):
#         Q[:, i+1] = mat_vec(Q[:, i])
#         for j in range(0, i+1): # Modified Gram-Schmidt
#             H[j, i] = np.dot(Q[:, i+1], Q[:, j])
#             Q[:, i+1] -= H[j, i] * Q[:, j]
#         H[i+1, i] = np.linalg.norm(Q[:, i+1], 2)
#         if H[i+1, i] < tol:
#             # print("Arnoldi Stopping!")
#             return Q, H
#         Q[:, i+1] /= H[i+1, i]

#     return Q, H

# def reduced_hessian(problem, output, x0, g0, delta=1e-7, n=None, tol=1e-14):
#     """
#     Return an approximate Hessian based on the Arnoldi procedure
#     """
#     mat_vec = lambda z: _hessian_vector_product(problem, output, x0, g0, z, delta)

#     if n is None:
#         n = len(x0)

#     g0_size = 0
#     for input in g0.keys():
#         g0_size += g0[input].size

#     g0_vec = np.zeros(g0_size)
#     offset = 0
#     for input in g0.keys():
#         input_size = x0[input].size
#         g0_vec[offset:offset + input_size] = g0[input]
#         offset += input_size
#     g0_vec = np.random.normal(size=g0_size)
#     # g0_vec = np.array([-1, 0])

#     Qnp1, Hn = _arnoldi_iterations(mat_vec, g0_vec, n, tol)
#     # print(f"Qnp1:\n{Qnp1}")
#     # print(f"Hn:\n{Hn}")
#     Qn = Qnp1[:,:-1]

#     hessian_approx = Qn @ Qn.T @ Qnp1 @ Hn @ Qn.T
#     return 0.5*(hessian_approx + hessian_approx.T)

# def hessian(problem, output, x0, g0):
#     return reduced_hessian(problem, output, x0, g0)

# def _hessian_difference_vector_product(lofi_prob,
#                                        hifi_prob,
#                                        lofi_outputs,
#                                        hifi_outputs,
#                                        x0,
#                                        lofi_g0s,
#                                        hifi_g0s,
#                                        pert,
#                                        delta):
#     """
#     Finite differenced Hessian-difference-vector products
#     """

#     if isinstance(delta, numbers.Number):
#         delta = {input: delta for input in x0.keys()}
    
#     offset = 0
#     for input in x0.keys():
#         input_size = x0[input].size
#         lofi_prob[input] = x0[input] + delta[input] * pert[offset:offset+input_size]
#         hifi_prob[input] = x0[input] + delta[input] * pert[offset:offset+input_size]
#         offset += input_size
    
#     lofi_prob.run_model()
#     lofi_totals = lofi_prob.compute_totals(lofi_outputs, [*x0.keys()])

#     hifi_prob.run_model()
#     hifi_totals = hifi_prob.compute_totals(hifi_outputs, [*x0.keys()])

#     n_outputs = len(x0)
#     n = pert.size
#     hvp_vec = np.zeros([n, n_outputs])

#     for i, (lofi_output, hifi_output) in enumerate(zip(lofi_outputs, hifi_outputs)):

#         hvp = {input: (lofi_totals[lofi_output, input] - lofi_g0s[lofi_output, input]) / delta[input] \
#                     - (hifi_totals[hifi_output, input] - hifi_g0s[hifi_output, input]) / delta[input] \
#                for input in x0.keys()}

#         offset = 0
#         for input in hvp.keys():
#             input_size = hvp[input].size
#             hvp_vec[offset:offset + input_size, i] = hvp[input]
#             offset += input_size

#     return hvp_vec

# def approximate_hessian_differences(lofi_prob,
#                                     hifi_prob,
#                                     lofi_outputs,
#                                     hifi_outputs,
#                                     x0,
#                                     lofi_g0s,
#                                     hifi_g0s,
#                                     delta=1e-5):
#     """
#     Return an approximate Hessian difference based on the finite-differencing
#     """

#     n_outputs = len(x0)
#     n = len(x0)
#     h_diffs = np.zeros([n, n, n_outputs])

#     for i in range(n):
#         pert = np.zeros(n)
#         pert[i] = 1.0
#         hess_diff_rows = _hessian_difference_vector_product(lofi_prob,
#                                                             hifi_prob,
#                                                             lofi_outputs,
#                                                             hifi_outputs,
#                                                             x0,
#                                                             lofi_g0s,
#                                                             hifi_g0s,
#                                                             pert,
#                                                             delta)

#         h_diffs[i, :, :] = hess_diff_rows[:, :]
#         pert[i] = 0.0

#     return h_diffs

# def sr1_hessian_differences(lofi_outputs,
#                             hifi_outputs,
#                             x0,
#                             lofi_g0s,
#                             hifi_g0s,
#                             old_differences):
#     # n = len(x0)
#     n_outputs = len(lofi_outputs)
#     n = 0
#     print(f"x0: {x0}")
#     for input in x0.keys():
#         n += x0[input].size

#     for i, (lofi_output, hifi_output) in enumerate(zip(lofi_outputs, hifi_outputs)):
#         # print(f"i: {i}, lofi_output: {lofi_output}, hifi_output: {hifi_output}")
#         # print(f"old_diff[{i}]: {old_differences[i]}")
#         grad_diff = {input: lofi_g0s[lofi_output, input] - hifi_g0s[hifi_output, input] \
#                      for input in x0.keys()}

#         grad_diff_kp1 = np.zeros([n])
#         offset = 0
#         for input in grad_diff.keys():
#             input_size = grad_diff[input].size
#             grad_diff_kp1[offset:offset + input_size] = grad_diff[input]
#             offset += input_size

#         # print(grad_diff_kp1)

#         # print(f"x0: {x0}")
#         x_kp1 = np.zeros(n)
#         offset = 0
#         for input in x0.keys():
#                 input_size = x0[input].size
#                 x_kp1[offset:offset + input_size] = x0[input]
#                 offset += input_size
#         # print(f"x_kp1: {x_kp1}")

#         h_diff = old_differences[i]
#         if h_diff is None:
#             sr1 = SR1()
#             sr1.initialize(n, 'hess')
#             old_differences[i] = [sr1, grad_diff_kp1, x_kp1, np.copy(sr1.get_matrix())]
#             # print(f"continue!")
#             continue

#         grad_diff_k = h_diff[1]
#         x_k = h_diff[2]
#         # print(f"x_k: {x_k}")

#         # print(f"x_kp1 - x_k: {x_kp1 - x_k}")
#         # print(f"grad_diff_kp1 - grad_diff_k: {grad_diff_kp1 - grad_diff_k}")
#         h_diff[0].update(x_kp1, grad_diff_kp1 - grad_diff_k)
#         h_diff[1] = grad_diff_kp1
#         h_diff[2] = x_kp1
#         h_diff[3] = np.copy(h_diff[0].get_matrix())

#     # print(old_differences)

#     return old_differences


# def hessian_differences(lofi_prob, hifi_prob, lofi_outputs, hifi_outputs, x0, lofi_g0s, hifi_g0s, old_differences=None, sr1_hessian_diff=False):
#     if not sr1_hessian_diff:
#         return approximate_hessian_differences(lofi_prob, hifi_prob, lofi_outputs, hifi_outputs, x0, lofi_g0s, hifi_g0s)
#     else:
#         return sr1_hessian_differences(lofi_outputs, hifi_outputs, x0, lofi_g0s, hifi_g0s, old_differences)
