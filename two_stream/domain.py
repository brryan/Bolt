from numpy import pi
from params import k_q1

q1_start = 0
q1_end   = 2 * pi / k_q1
N_q1     = 256

q2_start = 0
q2_end   = 1
N_q2     = 128

p1_start = [-5]
p1_end   = [5]
N_p1     = 48

p2_start = [-5]
p2_end   = [5]
N_p2     = 48

p3_start = [-5]
p3_end   = [5]
N_p3     = 48

N_ghost = 3
