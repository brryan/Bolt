import numpy as np
import arrayfire as af

single_mode_evolution = True

# Can be defined as 'electrostatic', 'user-defined'.
# The initial conditions need to be specified under initialize
# Ensure that the initial conditions specified satisfy
# Maxwell's constraint equations
fields_initialize = 'fft'

# Can be defined as 'electrostatic' and 'fdtd'
fields_solver = 'fft'

# Can be defined as 'strang' and 'lie'
solver_method_in_q = 'ASL'
riemann_solver = 'lax-friedrichs'
reconstruction_method = 'weno5'

# Dimensionality considered in velocity space:
p_dim = 1

# Number of devices(GPUs/Accelerators) on each node:
num_devices = 1

# Constants:
mass_particle      = 1
boltzmann_constant = 1
charge_electron    = 0

# Initial Conditions used in initialize:
rho_background         = 1
temperature_background = 1

p1_bulk_background = 0
p2_bulk_background = 0
p3_bulk_background = 0

pert_real = 0.01
pert_imag = 0

k_q1 = 2 * np.pi
k_q2 = 0
