"""
Functions which are used in assigning the I.C's to
the system.
"""

import arrayfire as af
import numpy as np

def initialize_f(q1, q2, v1, v2, v3, params):

    k_q1  = params.k_q1
    alpha = params.alpha

    f =   (2 / (7 * np.sqrt(2 * np.pi))) * (1 + 5 * v1**2) \
        * (1 + alpha * (  (af.cos(2 * k_q1 * q1) + af.cos(3 * k_q1 * q1))/1.2 
                        + af.cos(k_q1 * q1)
                       )
          ) * af.exp(-0.5 * v1**2)

    af.eval(f)
    return (f)

def initialize_E(q1, q2, params):
    
    E1 = 0.01*q1**0
    E2 = 0.002*q1**0
    E3 = 0.0003*q1**0

    af.eval(E1, E2, E3)
    return (E1, E2, E3)

def initialize_B(q1, q2, params):
    
    B1 = 0.001*q1**0
    B2 = 0.002*q1**0
    B3 = 0.003*q1**0

    af.eval(B1, B2, B3)
    return (B1, B2, B3)

