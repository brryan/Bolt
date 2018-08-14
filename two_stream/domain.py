from numpy import pi
from params import k_q1

q1_start = 0
q1_end   = 2 * pi / k_q1
N_q1     = 32

q2_start = 0
q2_end   = 1
N_q2     = 32

p1_start = [-5]
p1_end   = [5]
N_p1     = 20

p2_start = [-0.5]
p2_end   = [0.5]
N_p2     = 20

p3_start = [-0.5]
p3_end   = [0.5]
N_p3     = 20

N_ghost = 3
