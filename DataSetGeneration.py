from simulation import simulation
import numpy as np

def perform_simulation(initial_conditions, vx_0, vx_ref, Y_0, Y_ref, psi_0):
    initial_conditions[2] = Y_0
    initial_conditions[3] = psi_0
    initial_conditions[4] = vx_0
    results, t_average = simulation(initial_conditions, 10, 0.01, 0.4, 1, vx_ref, Y_ref)
    results = results[::20, 1:7]
    results[:, 0] = 0
    additional_columns = np.full((len(results), 2), [Y_ref, vx_ref])
    results = np.hstack((results, additional_columns))
    return results

lane_width = 4
t_0 = 0
X_0 = 0
Y_0 = lane_width / 2
psi_0 = 0
vx_0 = 70 / 3.6
vy_0 = 0
omega_r_0 = 0
acc_1 = 0
delta_f_1 = 0
X_CL_0 = 30
v_CL_0 = 60 / 3.6
initial_conditions = np.array([t_0, X_0, Y_0, psi_0, vx_0, vy_0, omega_r_0, acc_1, delta_f_1, X_CL_0, Y_0, v_CL_0])

Y_0_values_train = np.arange(2, 6.1, 1)
Y_ref_values_train = np.arange(2, 6.1, 1)
vx_0_values_train = np.arange(75, 85.1, 2) / 3.6
vx_ref_values_train = np.arange(75, 85.1, 2) / 3.6
psi_0_values_train = np.arange(-0.15, 0.16, 0.05)

Y_0_values_test = np.arange(6, 6.1, 0.5)
Y_ref_values_test = np.arange(2, 2.1, 0.5)
vx_0_values_test = np.array([78.1]) / 3.6
vx_ref_values_test = np.array([81.5]) / 3.6
psi_0_values_test = np.arange(0., 0.01, 0.05)

print(Y_0_values_train, Y_ref_values_train, vx_0_values_train, vx_ref_values_train)
print(Y_0_values_test, Y_ref_values_test)

all_results_train = []
all_results_test = []

for Y_0 in Y_0_values_train:
    for Y_ref in Y_ref_values_train:
        for vx_0 in vx_0_values_train:
            for vx_ref in vx_ref_values_train:
                for psi_0 in psi_0_values_train:
                    result = perform_simulation(initial_conditions, vx_0, vx_ref, Y_0, Y_ref, psi_0)
                    all_results_train.append(result)

for Y_0 in Y_0_values_test:
    for Y_ref in Y_ref_values_test:
        for vx_0 in vx_0_values_test:
            for vx_ref in vx_0_values_test:
                for psi_0 in psi_0_values_test:
                    result = perform_simulation(initial_conditions, vx_0, vx_ref, Y_0, Y_ref, psi_0)
                    all_results_test.append(result)

train_data = np.vstack(all_results_train)
test_data = np.vstack(all_results_test)

train_data[:, 1] -= train_data[:, 6]
train_data[:, 6] = 0
test_data[:, 1] -= test_data[:, 6]
test_data[:, 6] = 0

np.savetxt("Train Data.csv", train_data, delimiter=",", header="X,Y,psi,vx,vy,omega_r,Y_ref,vx_ref", comments="")
np.savetxt("Test Data.csv", test_data, delimiter=",", header="X,Y,psi,vx,vy,omega_r,Y_ref,vx_ref", comments="")

print("Simulation data saved successfully.")
