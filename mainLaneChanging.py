from simulation import simulation
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

lane_width = 4

initial_speeds = [75, 80, 85]
reference_speeds = [75, 80, 85]
lane_changes = [(2, 3), (2, 4), (2, 5), (2, 6), (6, 5), (6, 4), (6, 3), (6, 2)]

rcParams['font.family'] = 'serif'
rcParams['font.serif'] = 'Times New Roman'
rcParams['mathtext.fontset'] = 'stix'

def compute_metrics(results, vx_ref, Y_ref):
    time = results[:, 0]
    vx = results[:, 4]
    Y = results[:, 2]
    acc = results[:, 7]
    delta_f = results[:, 8]
    calc_time = results[:, -1]

    speed_rmse = np.sqrt(np.mean((vx - vx_ref)**2))
    lateral_rmse = np.sqrt(np.mean((Y - Y_ref)**2))
    avg_calc_time = np.mean(calc_time)
    acc_variance = np.var(acc)
    delta_f_variance = np.var(delta_f)

    return speed_rmse, lateral_rmse, avg_calc_time, acc_variance, delta_f_variance

fig, axes = plt.subplots(3, 3, figsize=(18, 9))

mpc_metrics = []
dpc_metrics = []

for i, vx_0_kmh in enumerate(initial_speeds):
    for j, vx_ref_kmh in enumerate(reference_speeds):
        ax = axes[i, j]
        for initial_lane, target_lane in lane_changes:
            t_0 = 0
            X_0 = 0
            Y_0 = lane_width * initial_lane / 2
            psi_0 = 0
            vx_0 = vx_0_kmh / 3.6
            vy_0 = 0
            omega_r_0 = 0
            acc_1 = 0
            delta_f_1 = 0
            X_CL_0 = 25
            Y_CL_0 = lane_width * initial_lane / 2
            v_CL_0 = vx_0_kmh / 3.6
            initial_conditions = np.array([t_0, X_0, Y_0, psi_0, vx_0, vy_0, omega_r_0, acc_1, delta_f_1, X_CL_0, Y_CL_0, v_CL_0])

            vx_ref = vx_ref_kmh / 3.6
            Y_ref = lane_width * target_lane / 2

            MPC_results, mpc_calc_time = simulation(initial_conditions, 15, 0.01, 0.05, 0, vx_ref, Y_ref)
            MPC_X = MPC_results[:, 1]
            MPC_Y = MPC_results[:, 2]

            mpc_calc_time_column = np.full(MPC_results.shape[0], mpc_calc_time)
            MPC_results = np.column_stack((MPC_results, mpc_calc_time_column))

            mpc_metrics.append(compute_metrics(MPC_results, vx_ref, Y_ref))

            DPC_results, dpc_calc_time = simulation(initial_conditions, 15, 0.01, 0.05, 1, vx_ref, Y_ref)
            DPC_X = DPC_results[:, 1]
            DPC_Y = DPC_results[:, 2]

            dpc_calc_time_column = np.full(DPC_results.shape[0], dpc_calc_time)
            DPC_results = np.column_stack((DPC_results, dpc_calc_time_column))

            dpc_metrics.append(compute_metrics(DPC_results, vx_ref, Y_ref))

            ax.plot(MPC_X, MPC_Y, 'r-', label='MPC' if (initial_lane == 2 and target_lane == 3) else "")
            ax.plot(DPC_X, DPC_Y, 'b-', label='DPC' if (initial_lane == 2 and target_lane == 3) else "")

        ax.set_title(f'$v_{{0}}={vx_0_kmh}$km/h, $v_{{ref}}={vx_ref_kmh}$km/h')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.grid(True)
        ax.legend(loc='upper center', bbox_to_anchor=(0.845, 1.16), ncol=2)

def print_metrics(metrics, controller_name):
    speed_rmse = np.mean([m[0] for m in metrics])
    lateral_rmse = np.mean([m[1] for m in metrics])
    avg_calc_time = np.mean([m[2] for m in metrics])
    acc_variance = np.mean([m[3] for m in metrics])
    delta_f_variance = np.mean([m[4] for m in metrics])

    print(f"{controller_name} Performance Metrics:")
    print(f"Speed Tracking RMSE: {speed_rmse:.2f} m/s")
    print(f"Lateral Position Tracking RMSE: {lateral_rmse:.2f} m")
    print(f"Average Calculation Time: {avg_calc_time:.4f} s")
    print(f"Acceleration Variance: {acc_variance:.4f} (m/s^2)^2")
    print(f"Steering Angle Variance: {delta_f_variance:.4f} rad^2")
    print()

print_metrics(mpc_metrics, "MPC")
print_metrics(dpc_metrics, "DPC")

plt.tight_layout()
plt.show()
