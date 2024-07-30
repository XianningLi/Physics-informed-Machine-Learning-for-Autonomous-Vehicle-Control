import torch

def vehicle_dynamics(state, control_input, params, dt):
    X, Y, psi, vx, vy, omega_r = torch.split(state, 1, dim=1)
    a, delta_f = torch.split(control_input, 1, dim=1)

    Cf = params['Cf']
    Cr = params['Cr']
    a_length = params['a']
    b_length = params['b']
    m = params['m']
    Iz = params['Iz']

    dX_dt = vx * torch.cos(psi) - vy * torch.sin(psi)
    dY_dt = vx * torch.sin(psi) + vy * torch.cos(psi)
    dpsi_dt = omega_r
    dvx_dt = a
    dvy_dt = (-2 * (Cf + Cr) / (m * vx)) * vy \
             - (2 * (a_length * Cf - b_length * Cr) / (m * vx) + vx) * omega_r \
             + (2 * Cf / m) * delta_f
    domega_r_dt = (-2 * (a_length * Cf - b_length * Cr) / (Iz * vx)) * vy \
                  - (2 * (a_length ** 2 * Cf + b_length ** 2 * Cr) / (Iz * vx)) * omega_r \
                  + (2 * a_length * Cf / Iz) * delta_f

    X_next = X + dt * dX_dt
    Y_next = Y + dt * dY_dt
    psi_next = psi + dt * dpsi_dt
    vx_next = vx + dt * dvx_dt
    vy_next = vy + dt * dvy_dt
    omega_r_next = omega_r + dt * domega_r_dt

    return torch.cat([X_next, Y_next, psi_next, vx_next, vy_next, omega_r_next], dim=1)
