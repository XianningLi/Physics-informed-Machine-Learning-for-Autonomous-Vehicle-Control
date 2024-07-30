from simulationGeneralization import simulation
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib import rcParams

rcParams['font.family'] = 'serif'
rcParams['font.serif'] = 'Times New Roman'
rcParams['mathtext.fontset'] = 'stix'

lane_width = 4

t_0 = 0
X_0 = 0
Y_0 = lane_width / 2 * 1
psi_0 = 0
vx_0 = 65 / 3.6
vy_0 = 0
omega_r_0 = 0
acc_1 = 0
delta_f_1 = 0
X_CL_0 = 25
Y_CL_0 = lane_width / 2
v_CL_0 = 70 / 3.6
initial_conditions = np.array([t_0, X_0, Y_0, psi_0, vx_0, vy_0, omega_r_0, acc_1, delta_f_1, X_CL_0, Y_CL_0, v_CL_0])

vx_ref = 90 / 3.6
Y_ref = lane_width * 3 / 2

MPC_results, MPC_t_record = simulation(initial_conditions, 15, 0.01, 0.05, 0, vx_ref, Y_ref)
DPC_results, DPC_t_record = simulation(initial_conditions, 15, 0.01, 0.05, 1, vx_ref, Y_ref)

MPC_time = MPC_results[:, 0]
MPC_X = MPC_results[:, 1]
MPC_Y = MPC_results[:, 2]
MPC_psi = MPC_results[:, 3]
MPC_vx = MPC_results[:, 4] * 3.6
MPC_vy = MPC_results[:, 5] * 3.6
MPC_omega_r = MPC_results[:, 6]
MPC_acc = MPC_results[:, 7]
MPC_delta_f = MPC_results[:, 8]
MPC_X_CL = MPC_results[:, 9]
MPC_Y_CL = MPC_results[:, 10]
MPC_v_CL = MPC_results[:, 11] * 3.6

DPC_time = DPC_results[:, 0]
DPC_X = DPC_results[:, 1]
DPC_Y = DPC_results[:, 2]
DPC_psi = DPC_results[:, 3]
DPC_vx = DPC_results[:, 4] * 3.6
DPC_vy = DPC_results[:, 5] * 3.6
DPC_omega_r = DPC_results[:, 6]
DPC_acc = DPC_results[:, 7]
DPC_delta_f = DPC_results[:, 8]
DPC_X_CL = DPC_results[:, 9]
DPC_Y_CL = DPC_results[:, 10]
DPC_v_CL = DPC_results[:, 11] * 3.6

fig, axs = plt.subplots(3, 3, figsize=(18, 9))

ax1 = plt.subplot2grid((3, 3), (0, 0), colspan=2)
ax1.plot(MPC_X, MPC_Y, color='r', label="MPC")
ax1.plot(DPC_X, DPC_Y, color='b', label="DPC")
ax1.set_title("X and Y Trajectories")
ax1.set_xlabel("X Position (m)")
ax1.set_ylabel("Y Position (m)")
ax1.legend()
ax1.grid(False)

ax1.set_facecolor('gray')
ax1.axhline(y=4, color='white', linestyle=(0, (10, 5)))
ax1.set_xlim([min(min(MPC_X), min(DPC_X)), max(max(MPC_X), max(DPC_X))])
ax1.set_ylim([0, lane_width * 2])
ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
ax1.yaxis.set_major_locator(MaxNLocator(integer=True))

axs[0, 0].get_xaxis().set_visible(False)
axs[0, 0].get_yaxis().set_visible(False)
axs[0, 1].get_xaxis().set_visible(False)
axs[0, 1].get_yaxis().set_visible(False)

axs[0, 2].plot(MPC_time[:len(MPC_t_record)], MPC_t_record, label="MPC", color='r')
axs[0, 2].plot(DPC_time[:len(DPC_t_record)], DPC_t_record, label="DPC", color='b')
axs[0, 2].set_title("Calculation Time")
axs[0, 2].set_xlabel("Time (s)")
axs[0, 2].set_ylabel("Time (s)")
axs[0, 2].legend()
axs[0, 2].grid(True)

axs[1, 0].plot(MPC_time, MPC_psi, label="MPC", color='r')
axs[1, 0].plot(DPC_time, DPC_psi, label="DPC", color='b')
axs[1, 0].set_title("Yaw Angle")
axs[1, 0].set_xlabel("Time (s)")
axs[1, 0].set_ylabel("Yaw Angle (rad)")
axs[1, 0].legend()
axs[1, 0].grid(True)

axs[1, 1].plot(MPC_time, MPC_vx, label="MPC", color='r')
axs[1, 1].plot(DPC_time, DPC_vx, label="DPC", color='b')
axs[1, 1].set_title("Longitudinal Velocity")
axs[1, 1].set_xlabel("Time (s)")
axs[1, 1].set_ylabel("Velocity (km/h)")
axs[1, 1].legend()
axs[1, 1].grid(True)

axs[1, 2].plot(MPC_time, MPC_vy, label="MPC", color='r')
axs[1, 2].plot(DPC_time, DPC_vy, label="DPC", color='b')
axs[1, 2].set_title("Lateral Velocity")
axs[1, 2].set_xlabel("Time (s)")
axs[1, 2].set_ylabel("Velocity (km/h)")
axs[1, 2].legend()
axs[1, 2].grid(True)

axs[2, 0].plot(MPC_time, MPC_omega_r, label="MPC", color='r')
axs[2, 0].plot(DPC_time, DPC_omega_r, label="DPC", color='b')
axs[2, 0].set_title("Yaw Rate")
axs[2, 0].set_xlabel("Time (s)")
axs[2, 0].set_ylabel("Yaw Rate (rad/s)")
axs[2, 0].legend()
axs[2, 0].grid(True)

axs[2, 1].plot(MPC_time, MPC_acc, label="MPC", color='r')
axs[2, 1].plot(DPC_time, DPC_acc, label="DPC", color='b')
axs[2, 1].set_title("Acceleration")
axs[2, 1].set_xlabel("Time (s)")
axs[2, 1].set_ylabel("Acceleration (m/s^2)")
axs[2, 1].legend()
axs[2, 1].grid(True)

axs[2, 2].plot(MPC_time, MPC_delta_f, label="MPC", color='r')
axs[2, 2].plot(DPC_time, DPC_delta_f, label="DPC", color='b')
axs[2, 2].set_title("Steering Angle")
axs[2, 2].set_xlabel("Time (s)")
axs[2, 2].set_ylabel("Steering Angle (rad)")
axs[2, 2].legend()
axs[2, 2].grid(True)

plt.tight_layout()
plt.show()
