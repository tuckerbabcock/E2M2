import numpy as np

def get_active_constraints(prob, constraints, des_vars, feas_tol=1e-6):
    active_cons = list()
    for constraint in constraints.keys():
        constraint_value = prob[constraint]
        if "equals" in constraints[constraint]:
            active_cons.append(constraint)
        else:
            constraint_upper = constraints[constraint].get("upper", np.inf)
            constraint_lower = constraints[constraint].get("lower", -np.inf)
            if constraint_value > constraint_upper or np.isclose(constraint_value, constraint_upper, atol=feas_tol, rtol=feas_tol):
                active_cons.append(constraint)
            elif constraint_value < constraint_lower or np.isclose(constraint_value, constraint_lower, atol=feas_tol, rtol=feas_tol):
                active_cons.append(constraint)

    for des_var in des_vars.keys():
        des_var_value = prob[des_var]
        des_var_upper = des_vars[des_var].get("upper", np.inf)
        des_var_lower = des_vars[des_var].get("lower", -np.inf)
        if des_var_value > des_var_upper or np.isclose(des_var_value, des_var_upper, atol=feas_tol, rtol=feas_tol):
            active_cons.append(des_var)
        elif des_var_value < des_var_lower or np.isclose(des_var_value, des_var_lower, atol=feas_tol, rtol=feas_tol):
            active_cons.append(des_var)

    return active_cons

def constraint_violation(prob, constraints, feas_tol=1e-6):
    constraint_error = dict()
    scaled_constraint_error = dict()
    for constraint in constraints.keys():
        constraint_value = prob[constraint]
        if "equals" in constraints[constraint]:
            constraint_target = constraints[constraint]["equals"]
            if not np.isclose(constraint_value, constraint_target, atol=feas_tol, rtol=feas_tol):
                constraint_error[constraint] = constraint_value - constraint_target
                scaled_constraint_error[constraint] = (constraint_value - constraint_target) / constraint_target
        else:
            constraint_value = prob[constraint]
            constraint_upper = constraints[constraint].get("upper", np.inf)
            constraint_lower = constraints[constraint].get("lower", -np.inf)
            if constraint_value > constraint_upper:
                if not np.isclose(constraint_value, constraint_upper, atol=feas_tol, rtol=feas_tol):
                    constraint_error[constraint] = constraint_value - constraint_upper
                    scaled_constraint_error[constraint] = (constraint_value - constraint_upper) / constraint_upper
            elif constraint_value < constraint_lower:
                if not np.isclose(constraint_value, constraint_lower, atol=feas_tol, rtol=feas_tol):
                    constraint_error[constraint] = constraint_value - constraint_lower
                    scaled_constraint_error[constraint] = (constraint_value - constraint_lower) / constraint_lower
    return constraint_error, scaled_constraint_error

def l1_merit_function(prob, objective, constraints, penalty_parameter, feas_tol=1e-6, maximize=False):
    phi = np.copy(prob[objective])

    if maximize:
        phi *= -1.0

    constraint_error, scaled_constraint_error = constraint_violation(prob, constraints, feas_tol)
    # for error in constraint_error.values():
    #     phi += penalty_parameter * np.absolute(error)
    for error in scaled_constraint_error.values():
        phi += penalty_parameter * np.absolute(error)

    return phi

def optimality(totals, objective, active_constraints, des_vars, multipliers, maximize=False):
    grad_f = {input: totals[objective, input] for input in des_vars.keys()}

    n = 0
    for input in grad_f.keys():
        n += grad_f[input].size

    grad_f_vec = np.zeros((n))
    offset = 0
    for input in grad_f.keys():
        input_size = grad_f[input].size
        grad_f_vec[offset:offset + input_size] = grad_f[input]
        offset += input_size

    if maximize:
        grad_f_vec *= -1.0

    n_con = len(active_constraints)
    active_cons_mat = np.zeros((n, n_con))
    multipliers_vec = np.zeros((n_con))
    multiplier_offset = 0
    for i, constraint in enumerate(active_constraints):
        if constraint in des_vars.keys():
            constraint_grad = {input: np.array([1.0]) if input == constraint else np.array([0.0]) for input in des_vars.keys()}
        else:
            constraint_grad = {input: totals[constraint, input] for input in des_vars.keys()}
        # print(f"{constraint} grad: {constraint_grad}")
        offset = 0
        for input in constraint_grad.keys():
            input_size = constraint_grad[input].size
            active_cons_mat[offset:offset + input_size, i] = constraint_grad[input]
            offset += input_size
        
        multiplier_size = multipliers[constraint].size
        multipliers_vec[multiplier_offset:multiplier_offset + multiplier_size] = multipliers[constraint]
        multiplier_offset += multiplier_size

    # multipliers_vec = np.zeros((n_con))
    # offset = 0
    # for input in multipliers.keys():
    #     input_size = multipliers[input].size
    #     multipliers_vec[offset:offset + input_size] = multipliers[input]
    #     offset += input_size

    optimality = np.linalg.norm(grad_f_vec - active_cons_mat @ multipliers_vec, np.inf)
    return optimality


def unscale_lagrange_multipliers(prob, objective, active_constraints, multipliers):
    for response in prob.driver._responses.values():
        if response['name'] == objective:
            obj_ref = response['ref']
            obj_ref0 = response['ref0']

    for constraint in active_constraints:
        if constraint in prob.driver._designvars:
            ref = prob.driver._designvars[constraint]['ref']
            ref0 = prob.driver._designvars[constraint]['ref0']
        else:
            for response in prob.driver._responses.values():
                if response['name'] == constraint:
                    ref = response['ref']
                    ref0 = response['ref0']

        if obj_ref is None:
            obj_ref = 1.0
        if obj_ref0 is None:
            obj_ref0 = 0.0

        if ref is None:
            ref = 1.0
        if ref0 is None:
            ref0 = 0.0

        multipliers[constraint] = multipliers[constraint] * (obj_ref - obj_ref0) / (ref - ref0)

    return multipliers

def estimate_lagrange_multipliers(prob, objective, active_constraints, totals, des_vars, unscaled=False):
    multipliers = dict()

    grad_f = {input: totals[objective, input] for input in des_vars.keys()}

    n = 0
    for input in grad_f.keys():
        n += grad_f[input].size

    grad_f_vec = np.zeros((n))
    offset = 0
    for input in grad_f.keys():
        input_size = grad_f[input].size
        grad_f_vec[offset:offset + input_size] = grad_f[input]
        offset += input_size

    n_con = len(active_constraints)
    active_cons_mat = np.zeros((n, n_con))
    for i, constraint in enumerate(active_constraints):
        if constraint in des_vars.keys():
            constraint_grad = {input: np.array([1.0]) if input == constraint else np.array([0.0]) for input in des_vars.keys()}
        else:
            constraint_grad = {input: totals[constraint, input] for input in des_vars.keys()}
        # print(f"{constraint} grad: {constraint_grad}")
        offset = 0
        for input in constraint_grad.keys():
            input_size = constraint_grad[input].size
            active_cons_mat[offset:offset + input_size, i] = constraint_grad[input]
            offset += input_size

    # print(f"grad_f_vec: {grad_f_vec}")
    # print(f"active_cons_mat: {active_cons_mat}")
    multipliers_vec, optimality, a, b = np.linalg.lstsq(active_cons_mat, grad_f_vec, rcond=None)
    # multipliers_vec, optimality, a, b = np.linalg.lstsq(active_cons_mat, grad_f_vec, rcond=-1)
    # print(f"lstsq output: {a}, {b}")
    # print(f"multipliers vec: {multipliers_vec}")
    # print(f"optimality: {np.linalg.norm(grad_f_vec - active_cons_mat @ multipliers_vec)}")
    # print(f"Estimated optimality squared: {optimality}")
    # print(f"Estimated optimality: {np.sqrt(optimality)}")
    offset = 0
    for constraint in active_constraints:
        constraint_size = 1
        multipliers[constraint] = multipliers_vec[offset:offset + constraint_size]
        offset += constraint_size

    if unscaled:
        unscale_lagrange_multipliers(prob, objective, active_constraints, multipliers)

    return multipliers
