from simulationCruising import simulation
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman']
rcParams['mathtext.fontset'] = 'stix'
rcParams['font.size'] = 14
rcParams['axes.labelsize'] = 14
rcParams['xtick.labelsize'] = 12
rcParams['ytick.labelsize'] = 12
rcParams['legend.fontsize'] = 14
rcParams['figure.titlesize'] = 16

lane_width = 4

t_0 = 0
X_0 = 0
Y_0 = lane_width / 2
psi_0 = 0
vx_0 = 80 / 3.6
vy_0 = 0
omega_r_0 = 0
acc_1 = 0
delta_f_1 = 0
X_CL_0 = 25
Y_CL_0 = lane_width / 2
v_CL_0 = 80 / 3.6
initial_conditions = np.array([t_0, X_0, Y_0, psi_0, vx_0, vy_0, omega_r_0, acc_1, delta_f_1, X_CL_0, Y_CL_0, v_CL_0])

Y_ref = lane_width / 2

simulation_time = 20
Ts = 0.01
Tc = 0.05

time = np.arange(0, simulation_time, Ts)
amplitude = 5 / 3.6
base_speed = 80 / 3.6

sin_wave = base_speed + amplitude * np.sin(2 * np.pi * time / simulation_time)
tri_wave = np.piecewise(time,
                        [time < 5, (time >= 5) & (time < 15), time >= 15],
                        [lambda t: base_speed + amplitude * (t / 5),
                         lambda t: base_speed + amplitude * (1 - (t - 5) / 5),
                         lambda t: base_speed + amplitude * (-1 + (t - 15) / 5)])
square_wave = base_speed + amplitude * np.sign(np.sin(2 * np.pi * time / simulation_time))

reference_speeds = {
    "Sin": sin_wave,
    "Tri": tri_wave,
    "Square": square_wave
}

results = {"MPC": {}, "DPC": {}}

for wave_type, ref_speed in reference_speeds.items():
    MPC_results, t_average_mpc = simulation(initial_conditions, simulation_time, Ts, Tc, 0, ref_speed, Y_ref)
    print(f"MPC Average Calculate Time = {t_average_mpc} s for {wave_type} wave")
    DPC_results, t_average_dpc = simulation(initial_conditions, simulation_time, Ts, Tc, 1, ref_speed, Y_ref)
    print(f"DPC Average Calculate Time = {t_average_dpc} s for {wave_type} wave")

    results["MPC"][wave_type] = MPC_results
    results["DPC"][wave_type] = DPC_results

fig, axes = plt.subplots(3, 1, figsize=(10, 10), dpi=300)

wave_types = ["Sin", "Tri", "Square"]
for i, wave_type in enumerate(wave_types):
    MPC_results = results["MPC"][wave_type]
    DPC_results = results["DPC"][wave_type]
    ref_speed = reference_speeds[wave_type]

    MPC_time = MPC_results[:, 0]
    MPC_vx = MPC_results[:, 4]

    DPC_time = DPC_results[:, 0]
    DPC_vx = DPC_results[:, 4]

    min_length = min(len(ref_speed), len(MPC_vx), len(DPC_vx))
    ref_speed = ref_speed[:min_length]
    MPC_vx = MPC_vx[:min_length]
    DPC_vx = DPC_vx[:min_length]
    MPC_time = MPC_time[:min_length]
    DPC_time = DPC_time[:min_length]

    MPC_error = ref_speed - MPC_vx
    DPC_error = ref_speed - DPC_vx

    MPC_MAE = np.mean(np.abs(MPC_error))
    MPC_RMSE = np.sqrt(np.mean(MPC_error ** 2))

    DPC_MAE = np.mean(np.abs(DPC_error))
    DPC_RMSE = np.sqrt(np.mean(DPC_error ** 2))

    print(f"{wave_type} Wave - MPC MAE: {MPC_MAE*3.6:.2f} km/h, MPC RMSE: {MPC_RMSE*3.6:.2f} km/h")
    print(f"{wave_type} Wave - DPC MAE: {DPC_MAE*3.6:.2f} km/h, DPC RMSE: {DPC_RMSE*3.6:.2f} km/h")

    ax = axes[i]
    ax.plot(MPC_time, ref_speed * 3.6, label="Reference Speed", color='g', linestyle='--')
    ax.plot(MPC_time, MPC_vx * 3.6, label=f"MPC - {wave_type} wave", color='r')
    ax.plot(DPC_time, DPC_vx * 3.6, label=f"DPC - {wave_type} wave", color='b')
    ax.set_title(f"Velocity Tracking for {wave_type} Wave")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Velocity (km/h)")
    ax.legend()
    ax.grid(True)
    ax.set_xlim([0, 20])

plt.tight_layout()
plt.show()
