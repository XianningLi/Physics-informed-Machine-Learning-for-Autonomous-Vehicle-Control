import casadi as ca
import numpy as np
import time

def mpc_controller(current_conditions, vx_ref, Y_ref):
    dt = 0.5
    Np = 10

    X_0 = 0
    Y_0 = current_conditions[2]
    psi_0 = current_conditions[3]
    vx_0 = current_conditions[4]
    vy_0 = current_conditions[5]
    omega_r_0 = current_conditions[6]
    acc_1 = current_conditions[7]
    delta_f_1 = current_conditions[8]
    X_CL_0 = current_conditions[9] - current_conditions[1]
    Y_CL_0 = current_conditions[10]
    v_CL_0 = current_conditions[11]

    m = 1270
    Iz = 1536.7
    a = 1.015
    b = 1.895
    Cf = 1250
    Cr = 755
    lane_width = 4

    X = ca.SX.sym('X')
    Y = ca.SX.sym('Y')
    psi = ca.SX.sym('psi')
    vx = ca.SX.sym('vx')
    vy = ca.SX.sym('vy')
    omega_r = ca.SX.sym('omega_r')
    states = ca.vertcat(X, Y, psi, vx, vy, omega_r)
    n_states = states.numel()

    acc = ca.SX.sym('acc')
    delta_f = ca.SX.sym('delta_f')
    controls = ca.vertcat(acc, delta_f)
    n_controls = controls.numel()

    X_dot = vx * ca.cos(psi) - vy * ca.sin(psi)
    Y_dot = vx * ca.sin(psi) + vy * ca.cos(psi)
    psi_dot = omega_r
    vx_dot = acc
    vy_dot = - 2 * (Cf + Cr) / (m * vx) * vy - (2 * (a * Cf - b * Cr) / (m * vx) + vx) * omega_r + 2 * Cf / m * delta_f
    wr_dot = - 2 * (a * Cf - b * Cr) / (Iz * vx) * vy - 2 * (a ** 2 * Cf + b ** 2 * Cr) / (Iz * vx) * omega_r + 2 * a * Cf / Iz * delta_f

    f_dynamic = ca.vertcat(X_dot, Y_dot, psi_dot, vx_dot, vy_dot, wr_dot)
    system_dynamics = ca.Function('system_dynamics', [states, controls], [f_dynamic])

    Qx = ca.diag([0, 60, 500, 50, 10, 100])
    Qu = ca.diag([5, 500])
    Qdelta = ca.diag([5, 500])
    Qt = ca.diag([0, 60, 1000, 70, 20, 200])

    vx_min, vx_max = 40/3.6, 120/3.6
    acc_min, acc_max = -3.0, 3.0
    delta_f_min, delta_f_max = -0.3, 0.3

    x0 = ca.DM([X_0, Y_0, psi_0, vx_0, vy_0, omega_r_0])
    x_ref = ca.DM([0, Y_ref, 0, vx_ref, 0, 0])

    U = ca.SX.sym('U', n_controls, Np)
    X = x0

    J = 0
    g = []
    lbg = []
    ubg = []
    lbx = []
    ubx = []
    for k in range(Np):
        diff = system_dynamics(X, U[:, k])
        X_next = X + diff * dt
        J += (X_next - x_ref).T @ Qx @ (X_next - x_ref)
        J += U[:, k].T @ Qu @ U[:, k]

        g.append(X_next[3])

        X = X_next

        lbg += [vx_min]
        ubg += [vx_max]
        lbx += [acc_min, delta_f_min]
        ubx += [acc_max, delta_f_max]

    J = J + (X - x_ref).T @ Qt @ (X - x_ref)

    nlp = {'f': J, 'x': ca.vec(U), 'g': ca.vertcat(*g)}

    opts = {'ipopt.print_level': 0, 'print_time': 0}
    solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

    sol = solver(lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)

    u_opt = np.array(sol['x']).reshape((Np, n_controls))
    u_opt = u_opt.T

    acc_k = u_opt[0][0]
    delta_f_k = u_opt[1][0]
    return [acc_k, delta_f_k]
