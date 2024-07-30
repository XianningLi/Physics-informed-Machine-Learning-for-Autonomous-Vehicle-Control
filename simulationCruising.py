import numpy as np
from MPC_controller import mpc_controller
from DPCModel import IntegratedNetwork
import time
import torch

def simulation(initial_conditions, simulation_time, Ts, Tc, controller_flag, vx_ref_array, Y_ref):
    model_LC = torch.load('controller_net.pth')
    model_LC.eval()
    model_LC.to("cpu")

    t_0 = initial_conditions[0]
    X_0 = initial_conditions[1]
    Y_0 = initial_conditions[2]
    psi_0 = initial_conditions[3]
    vx_0 = initial_conditions[4]
    vy_0 = initial_conditions[5]
    omega_r_0 = initial_conditions[6]
    acc_1 = initial_conditions[7]
    delta_f_1 = initial_conditions[8]
    X_CL_0 = initial_conditions[9]
    Y_CL_0 = initial_conditions[10]
    v_CL_0 = initial_conditions[11]

    m = 1270
    Iz = 1536.7
    a = 1.015
    b = 1.895
    Cf = 1250
    Cr = 755

    simulation_results = np.array([initial_conditions])
    num_steps = int(simulation_time / Ts)
    current_conditions = initial_conditions
    t_k = t_0
    X_k = X_0
    Y_k = Y_0
    psi_k = psi_0
    vx_k = vx_0
    vy_k = vy_0
    omega_r_k = omega_r_0
    acc_k_1 = acc_1
    delta_f_k_1 = delta_f_1
    X_CL_k = X_CL_0
    Y_CL_k = Y_CL_0
    v_CL_k = v_CL_0

    t_record = np.array([])

    for step in range(num_steps):
        acc_k = 0
        delta_f_k = 0
        vx_ref = vx_ref_array[step]
        if step % (Tc // Ts) == 0:
            if controller_flag == 0:
                start_time = time.time()
                [acc_k, delta_f_k] = mpc_controller(current_conditions, vx_ref, Y_ref)
                end_time = time.time()
                t_record = np.append(t_record, end_time - start_time)
            elif controller_flag == 1:
                features = current_conditions[1:7].copy()
                features[0] = 0
                features[1] -= Y_ref
                features = np.append(features, 0)
                features = np.append(features, vx_ref)

                features = torch.tensor(features, dtype=torch.float32)
                features = features.unsqueeze(0)
                start_time = time.time()
                initial_state, controller_output, reference = model_LC(features)
                end_time = time.time()
                t_record = np.append(t_record, end_time - start_time)
                acc_k = controller_output[0][0][0].item()
                delta_f_k = controller_output[0][0][1].item()
            else:
                print("Controller not found!")
        else:
            acc_k = acc_k_1
            delta_f_k = delta_f_k_1

        deriv_X = vx_k * np.cos(psi_k) - vy_k * np.sin(psi_k)
        deriv_Y = vx_k * np.sin(psi_k) + vy_k * np.cos(psi_k)
        deriv_psi = omega_r_k
        deriv_vx = acc_k
        deriv_vy = -2 * (Cf + Cr) * vy_k / (m * vx_k) - (2 * (a * Cf - b * Cr) / (m * vx_k) + vx_k) * omega_r_k + 2 * Cf * delta_f_k / m
        deriv_omega_r = -2 * (a * Cf - b * Cr) * vy_k / (Iz * vx_k) - 2 * (a ** 2 * Cf + b ** 2 * Cr) * omega_r_k / (Iz * vx_k) + 2 * a * Cf * delta_f_k / Iz
        deriv_X_CL = v_CL_k

        t_k = current_conditions[0] + Ts
        X_k = current_conditions[1] + deriv_X * Ts
        Y_k = current_conditions[2] + deriv_Y * Ts
        psi_k = current_conditions[3] + deriv_psi * Ts
        vx_k = current_conditions[4] + deriv_vx * Ts
        vy_k = current_conditions[5] + deriv_vy * Ts
        omega_r_k = current_conditions[6] + deriv_omega_r * Ts
        acc_k_1 = acc_k
        delta_f_k_1 = delta_f_k
        X_CL_k = current_conditions[9] + deriv_X_CL * Ts
        Y_CL_k = current_conditions[10]
        v_CL_k = current_conditions[11]

        current_conditions = np.array([t_k, X_k, Y_k, psi_k, vx_k, vy_k, omega_r_k, acc_k_1, delta_f_k_1, X_CL_k, Y_CL_k, v_CL_k])
        simulation_results = np.append(simulation_results, [current_conditions], axis=0)

    t_average = t_record.mean()
    return simulation_results, t_average
