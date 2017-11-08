import numpy as np
import arrayfire as af

# Can be defined as 'electrostatic', 'user-defined'.
# The initial conditions need to be specified under initialize
# Ensure that the initial conditions specified satisfy
# Maxwell's constraint equations
fields_initialize = 'electrostatic'

solver_method_in_q = 'FVM'
riemann_solver = 'lax-friedrichs'
reconstruction_method = 'minmod'

# Can be defined as 'electrostatic' and 'fdtd'
fields_solver = 'fdtd'

# Dimensionality considered in velocity space:
p_dim = 1

# Number of devices(GPUs/Accelerators) on each node:
num_devices = 1

# Constants:
mass_particle      = 1
boltzmann_constant = 1
charge_electron    = 0

# Initial Conditions used in initialize:
rho_background  = 1

# Variation of collisional-timescale parameter through phase space:
def tau(q1, q2, p1, p2, p3):
    return (np.inf)
