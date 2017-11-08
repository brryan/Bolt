import arrayfire as af

# Importing Riemann solver used in calculating fluxes:
from .riemann_solver import riemann_solver
from .reconstruct import reconstruct

# Equation to solve:
# df/dt + d(C_q1 * f)/dq1 + d(C_q2 * f)/dq2 = C[f]
# Grid convention considered:

#                  (i+1/2, j+1)
#              X-------o-------X
#              |               |
#              |               |
#   (i, j+1/2) o       o       o (i+1, j+1/2)
#              | (i+1/2, j+1/2)|
#              |               |
#              X-------o-------X
#                  (i+1/2, j)

# Using the finite volume method:
# d(f_{i+1/2, j+1/2})/dt  = ((- (C_q1 * f)_{i + 1, j + 1/2} + (C_q1 * f)_{i, j + 1/2})/dq1
                          #  (- (C_q2 * f)_{i + 1/2, j + 1} + (C_q2 * f)_{i + 1/2, j})/dq2
                          #  +  C[f_{i+1/2, j+1/2}]
                          # )

def df_dt_fvm(f, self):
    
    multiply = lambda a, b: a * b

    if(self.performance_test_flag == True):
        tic = af.time()

    left_plus_eps_flux, right_minus_eps_flux = \
        reconstruct(self, af.broadcast(multiply, self._C_q1, f), 'q1')
    bot_plus_eps_flux, top_minus_eps_flux = \
        reconstruct(self, af.broadcast(multiply, self._C_q2, f), 'q2')

    f_left_plus_eps, f_right_minus_eps = reconstruct(self, f, 'q1')
    f_bot_plus_eps, f_top_minus_eps    = reconstruct(self, f, 'q2')

    if(self.performance_test_flag == True):
        af.sync()
        toc = af.time()
        self.time_reconstruct += toc - tic

    # f_left_minus_eps of i-th cell is f_right_minus_eps of the (i-1)th cell
    f_left_minus_eps = af.shift(f_right_minus_eps, 0,  1)
    # Extending the same to bot:
    f_bot_minus_eps  = af.shift(f_top_minus_eps, 0, 0,  1)

    # Applying the shifts to the fluxes:
    left_minus_eps_flux = af.shift(right_minus_eps_flux, 0,  1)
    bot_minus_eps_flux  = af.shift(top_minus_eps_flux, 0, 0,  1)

    left_flux  = riemann_solver(self, left_minus_eps_flux, left_plus_eps_flux,
                                f_left_minus_eps, f_left_plus_eps
                               )

    bot_flux   = riemann_solver(self, bot_minus_eps_flux, bot_plus_eps_flux,
                                f_bot_minus_eps, f_bot_plus_eps
                               )

    right_flux = af.shift(left_flux, 0, -1)
    top_flux   = af.shift(bot_flux, 0, 0, -1)

    df_dt = - (right_flux - left_flux)/self.dq1 \
            - (top_flux   - bot_flux )/self.dq2 \
            + self._source(f, self.q1_center, self.q2_center,
                           self.p1, self.p2, self.p3, 
                           self.compute_moments, 
                           self.physical_system.params
                          ) 

    af.eval(df_dt)
    return(df_dt)
 