import numpy as np
import arrayfire as af

fields_type       = 'electrodynamic'
fields_initialize = 'fft + user-defined magnetic fields'
fields_solver     = 'fdtd'

# Dimensionality considered in velocity space:
p_dim = 3

# Method in q-space
solver_method_in_q = 'FVM'
solver_method_in_p = 'FVM'

riemann_solver_in_q = 'upwind-flux'
riemann_solver_in_p = 'upwind-flux'

reconstruction_method_in_q = 'weno5'
reconstruction_method_in_p = 'weno5'

# Units: l0, t0, m0, e0, n0, T0, v0
# Independent: n0, T0, m0, e0, k0, eps0
# Dependent  : l0, t0, v0

# Plasma parameters (given):
# Number density  ~ n; n = |n| units(n)
# Temperature     ~ T; T = |T| units(T)

# Fundamental consts: 
# Mass            ~ m_p; m_p = |m_p| units(m_p)
# Electric charge ~ e;   e   = |e|   units(e)
# Boltzmann const ~ k;   k   = |k|   units(k)
# Vacuum perm     ~ eps0; eps0 = |eps0| units(eps0)

# Now choosing units: 
n0  = 1 # |n| units(n)
T0  = 1 # |T| units(T)
m0  = 1 # |m_p| units(m)
e0  = 1 # |e| units(e)
k0  = 1 # |k| units(k)
eps = 1 # |eps0| units(eps0)
mu  = 1 # |mu0| units(mu0)

v0 = velocity_scales.thermal_speed(T0, m0, k0) ##??
l0 = length_scales.gyroradius(v0, B0, e0, m0)
t0 = 1/time_scales.cyclotron_frequency(B0, e0, m0)

t_final = 100 * t0
N_cfl   = 0.4

# Number of devices(GPUs/Accelerators) on each node:
num_devices = 1

# Constants:
mass               = [1 * m0, 1 * m0]
boltzmann_constant = 1 * k0
charge             = [-1 * e0, 1 * e0]

density = 1  * n0
v1_bulk = 10 * v0
beta    = 2
 
fields_enabled           = True
source_enabled           = False
instantaneous_collisions = False

# Variation of collisional-timescale parameter through phase space:
@af.broadcast
def tau(q1, q2, p1, p2, p3):
    return (np.inf * q1**0 * p1**0)