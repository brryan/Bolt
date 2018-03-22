import numpy as np
import matplotlib
matplotlib.use('agg')
import pylab as pl

import h5py
import domain
import params

# Optimized plot parameters to make beautiful plots:
pl.rcParams['figure.figsize']  = 12, 7.5
pl.rcParams['figure.dpi']      = 100
pl.rcParams['image.cmap']      = 'jet'
pl.rcParams['lines.linewidth'] = 1.5
pl.rcParams['font.family']     = 'serif'
pl.rcParams['font.weight']     = 'bold'
pl.rcParams['font.size']       = 20
pl.rcParams['font.sans-serif'] = 'serif'
pl.rcParams['text.usetex']     = True
pl.rcParams['axes.linewidth']  = 1.5
pl.rcParams['axes.titlesize']  = 'medium'
pl.rcParams['axes.labelsize']  = 'medium'

pl.rcParams['xtick.major.size'] = 8
pl.rcParams['xtick.minor.size'] = 4
pl.rcParams['xtick.major.pad']  = 8
pl.rcParams['xtick.minor.pad']  = 8
pl.rcParams['xtick.color']      = 'k'
pl.rcParams['xtick.labelsize']  = 'medium'
pl.rcParams['xtick.direction']  = 'in'

pl.rcParams['ytick.major.size'] = 8
pl.rcParams['ytick.minor.size'] = 4
pl.rcParams['ytick.major.pad']  = 8
pl.rcParams['ytick.minor.pad']  = 8
pl.rcParams['ytick.color']      = 'k'
pl.rcParams['ytick.labelsize']  = 'medium'
pl.rcParams['ytick.direction']  = 'in'

N_q1 = domain.N_q1
N_q2 = domain.N_q2
N_g  = domain.N_ghost

dq1 = (domain.q1_end - domain.q1_start) / N_q1
dq2 = (domain.q2_end - domain.q2_start) / N_q2

q1 = domain.q1_start + (0.5 + np.arange(N_q1)) * dq1
q2 = domain.q2_start + (0.5 + np.arange(N_q2)) * dq2

q2, q1 = np.meshgrid(q2, q1)

# Taking user input:
# ndim_plotting   = input('1D or 2D plotting?:')
N_s             = int(input('Enter number of species: '))

# quantities      = ['density', 'v1', 'v2', 'v3', 'temperature', 'pressure', 'q1', 'q2', 'q3',
#                    'E1', 'E2', 'E3', 'B1', 'B2', 'B3'
#                   ]
# Taking input on quantities to be plotted:
# quantities    = input('Enter quantities to be plotted separated by commas:')
# quantities    = quantities.split(',')
# N_quantities  = len(quantities)
# N_rows        = input('Enter number of rows:')
# N_columns     = input('Enter number of columns:')

def return_array_to_be_plotted(name, moments, fields):
    
    n       = moments[:, :, 0:N_s]
    
    v1_bulk = moments[:, :, 2*N_s:3*N_s] / n
    v2_bulk = moments[:, :, 3*N_s:4*N_s] / n
    v3_bulk = moments[:, :, 4*N_s:5*N_s] / n
    
    T       = (  moments[:, :, 1*N_s:2*N_s]
               - n * v1_bulk**2
               - n * v2_bulk**2
               - n * v3_bulk**2
              ) / (params.p_dim * n)

    heat_flux_1 = moments[:, :, 5*N_s:6*N_s] / n
    heat_flux_2 = moments[:, :, 6*N_s:7*N_s] / n
    heat_flux_3 = moments[:, :, 7*N_s:8*N_s] / n

    E1 = fields[:, :, 0]
    E2 = fields[:, :, 1]
    E3 = fields[:, :, 2]
    B1 = fields[:, :, 3]
    B2 = fields[:, :, 4]
    B3 = fields[:, :, 5]

    if(name == 'density'):
        return n

    elif(name == 'v1'):
        return v1_bulk

    elif(name == 'v2'):
        return v2_bulk

    elif(name == 'v3'):
        return v3_bulk

    elif(name == 'temperature'):
        return T

    elif(name == 'pressure'):
        return(n * T)

    elif(name == 'q1'):
        return heat_flux_1

    elif(name == 'v2'):
        return heat_flux_2

    elif(name == 'v3'):
        return heat_flux_3

    elif(name == 'E1'):
        return E1

    elif(name == 'E2'):
        return E2

    elif(name == 'E3'):
        return E3

    elif(name == 'B1'):
        return B1

    elif(name == 'B2'):
        return B2

    elif(name == 'B3'):
        return B3

    else:
        raise Exception('Not valid!')


# Declaration of the time array:
time_array = np.arange(0, params.t_final + params.dt_dump_moments, 
                       params.dt_dump_moments
                      )

# Traversal to determine the maximum and minimum:
def determine_min_max(quantity):
    # Declaring an initial value for the max and minimum for the quantity plotted:
    q_max = -1e10    
    q_min =  1e10

    for time_index, t0 in enumerate(time_array):
        
        h5f  = h5py.File('dump_moments/t=%.3f'%(t0) + '.h5', 'r')
        moments = np.swapaxes(h5f['moments'][:], 0, 1)
        h5f.close()

        array = return_array_to_be_plotted(quantity, moments, moments)

        if(np.max(array)>q_max):
            q_max = np.max(array)

        if(np.min(array)<q_min):
            q_min = np.min(array)

    return(q_min, q_max)

n_min, n_max   = determine_min_max('density')

def plot_2d():

    for time_index, t0 in enumerate(time_array):
        
        h5f  = h5py.File('dump_moments/t=%.3f'%(t0) + '.h5', 'r')
        moments = np.swapaxes(h5f['moments'][:], 0, 1)
        h5f.close()

        n  = return_array_to_be_plotted('density', moments, moments)

        fig = pl.figure()

        ax1 = fig.add_subplot(1, 1, 1)
        c1 = ax1.contourf(q1, q2, n[:, :, 0], np.linspace(0.99 * n_min, 1.01 * n_max, 100))
        ax1.set_aspect('equal')
        fig.colorbar(c1)
        fig.suptitle('Time = %.2f'%t0)
        pl.savefig('images/%04d'%time_index + '.png')
        pl.close(fig)
        pl.clf()

plot_2d()
