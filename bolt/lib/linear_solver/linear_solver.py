#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This is the module which contains the functions of the
linear solver of Bolt. It performs the FFT to map the given 
input onto the Fourier basis, and evolves each mode of the
input independantly. It is to be noted that this module
can only be applied to systems with periodic boundary conditions.
Additionally, this module isn't parallelized to run across
multiple devices/nodes.
"""

# In this code, we shall default to using the positionsExpanded form
# thoroughout. This means that the arrays defined in the system will
# be of the form: (N_q1, N_q2, N_p1*N_p2*N_p3)

# Importing dependencies:
import numpy as np
import arrayfire as af
from numpy.fft import fftfreq
import socket
from petsc4py import PETSc
from inspect import signature

# Importing solver functions:
from .EM_fields_solver import compute_electrostatic_fields
from .calculate_dfdp_background import calculate_dfdp_background
from .compute_moments import compute_moments as compute_moments_imported
from .file_io import dump, load
from .utils.bandwidth_test import bandwidth_test
from .utils.print_with_indent import indent

from . import timestep

class linear_solver(object):
    """
    An instance of this class' attributes contains methods which are used
    in evolving the system declared under physical system linearly. The 
    state of the system then may be determined from the attributes of the 
    system such as the distribution function and electromagnetic fields
    """

    def __init__(self, physical_system):
        """
        Constructor for the linear_solver object. It takes the physical
        system object as an argument and uses it in intialization and
        evolution of the system in consideration.

        Parameters
        ----------
        
        physical_system: The defined physical system object which holds
                         all the simulation information such as the initial
                         conditions, and the domain info is passed as an
                         argument in defining an instance of the
                         nonlinear_solver. This system is then evolved, and
                         monitored using the various methods under the
                         nonlinear_solver class.
        """
        self.physical_system = physical_system

        # Storing Domain Information:
        self.q1_start, self.q1_end = physical_system.q1_start,\
                                     physical_system.q1_end
        self.q2_start, self.q2_end = physical_system.q2_start,\
                                     physical_system.q2_end
        self.p1_start, self.p1_end = physical_system.p1_start,\
                                     physical_system.p1_end
        self.p2_start, self.p2_end = physical_system.p2_start,\
                                     physical_system.p2_end
        self.p3_start, self.p3_end = physical_system.p3_start,\
                                     physical_system.p3_end

        # Getting Domain Resolution
        self.N_q1, self.dq1 = physical_system.N_q1, physical_system.dq1
        self.N_q2, self.dq2 = physical_system.N_q2, physical_system.dq2
        self.N_p1, self.dp1 = physical_system.N_p1, physical_system.dp1
        self.N_p2, self.dp2 = physical_system.N_p2, physical_system.dp2
        self.N_p3, self.dp3 = physical_system.N_p3, physical_system.dp3

        # Checking that periodic B.C's are utilized:
        if(    physical_system.boundary_conditions.in_q1_left   != 'periodic' 
           and physical_system.boundary_conditions.in_q1_right  != 'periodic'
           and physical_system.boundary_conditions.in_q2_bottom != 'periodic'
           and physical_system.boundary_conditions.in_q2_top    != 'periodic'
          ):
            raise Exception('Only systems with periodic boundary conditions \
                             can be solved using the linear solver'
                           )

        # Initializing DAs which will be used in file-writing:
        self._da_dump_f = PETSc.DMDA().create([self.N_q1, self.N_q2],
                                              dof=(  self.N_p1 
                                                   * self.N_p2 
                                                   * self.N_p3
                                                  )
                                             )

        self._da_dump_moments = PETSc.DMDA().create([self.N_q1, self.N_q2],
                                                    dof=len(self.physical_system.\
                                                            moment_exponents
                                                           )
                                                   )


        # Printing backend details:
        PETSc.Sys.Print('\nBackend Details for Linear Solver:')
        PETSc.Sys.Print(indent('On Node: '+ socket.gethostname()))
        PETSc.Sys.Print(indent('Device Details:'))
        PETSc.Sys.Print(indent(af.info_str(), 2))
        PETSc.Sys.Print(indent('Device Bandwidth = ' + str(bandwidth_test(100)) + ' GB / sec'))
        PETSc.Sys.Print()

        # Creating PETSc Vecs which are used in dumping to file:
        self._glob_f       = self._da_dump_f.createGlobalVec()
        self._glob_f_value = self._da_dump_f.getVecArray(self._glob_f)

        self._glob_moments       = self._da_dump_moments.createGlobalVec()
        self._glob_moments_value = self._da_dump_moments.\
                                   getVecArray(self._glob_moments)

        # Setting names for the objects which will then be
        # used as the key identifiers for the HDF5 files:
        PETSc.Object.setName(self._glob_f, 'distribution_function')
        PETSc.Object.setName(self._glob_moments, 'moments')

        # Intializing position, wavenumber and velocity arrays:
        self.q1_center, self.q2_center = self._calculate_q_center()
        self.k_q1, self.k_q2           = self._calculate_k()
        self.p1, self.p2, self.p3      = self._calculate_p_center()

        # Assigning the advection terms along q1 and q2
        self._A_q1 = \
            self.physical_system.A_q(self.q1_center, self.q2_center,
                                     self.p1, self.p2, self.p3, 
                                     physical_system.params
                                    )[0]

        self._A_q2 = \
            self.physical_system.A_q(self.q1_center, self.q2_center,
                                     self.p1, self.p2, self.p3,
                                     physical_system.params
                                    )[1]

        # Assigning the function objects to methods of the solver:
        self._A_p    = self.physical_system.A_p
        self._source = self.physical_system.source

        if(len(signature(self._source).parameters) == 6):
            self.single_mode_evolution = True
        else:
            self.single_mode_evolution = False

        # Initializing f, f_hat and the other EM field quantities:
        self._initialize(physical_system.params)


    def _calculate_q_center(self):
        """
        Initializes the cannonical variables q1, q2 using a centered
        formulation. The size, and resolution are the same as declared
        under domain of the physical system object.
        """
        q1_center = self.q1_start + (0.5 + np.arange(self.N_q1)) * self.dq1
        q2_center = self.q2_start + (0.5 + np.arange(self.N_q2)) * self.dq2

        q2_center, q1_center = np.meshgrid(q2_center, q1_center)

        q2_center = af.to_array(q2_center)
        q1_center = af.to_array(q1_center)

        af.eval(q1_center, q2_center)
        return(q1_center, q2_center)

    def _calculate_k(self):
        """
        Initializes the wave numbers k_q1 and k_q2 which will be 
        used when solving in fourier space.
        """
        k_q1 = 2 * np.pi * fftfreq(self.N_q1, self.dq1)
        k_q2 = 2 * np.pi * fftfreq(self.N_q2, self.dq2)

        k_q2, k_q1 = np.meshgrid(k_q2, k_q1)

        k_q2 = af.to_array(k_q2)
        k_q1 = af.to_array(k_q1)

        af.eval(k_q1, k_q2)
        return(k_q1, k_q2)

    def _calculate_p_center(self):
        """
        Initializes the cannonical variables p1, p2 and p3 using a centered
        formulation. The size, and resolution are the same as declared
        under domain of the physical system object.
        """
        p1_center = \
            self.p1_start + (0.5 + np.arange(0, self.N_p1, 1)) * self.dp1
        
        p2_center = \
            self.p2_start + (0.5 + np.arange(0, self.N_p2, 1)) * self.dp2
        
        p3_center = \
            self.p3_start + (0.5 + np.arange(0, self.N_p3, 1)) * self.dp3

        p2_center, p1_center, p3_center = np.meshgrid(p2_center,
                                                      p1_center,
                                                      p3_center
                                                     )

        # Flattening the obtained arrays:
        p1_center = af.flat(af.to_array(p1_center))
        p2_center = af.flat(af.to_array(p2_center))
        p3_center = af.flat(af.to_array(p3_center))

        # Reordering such that variation in velocity is along axis 2:
        # This is done to be consistent with the positionsExpanded form:
        p1_center = af.reorder(p1_center, 2, 3, 0, 1)
        p2_center = af.reorder(p2_center, 2, 3, 0, 1)
        p3_center = af.reorder(p3_center, 2, 3, 0, 1)

        af.eval(p1_center, p2_center, p3_center)
        return(p1_center, p2_center, p3_center)

    def _initialize(self, params):
        """
        Called when the solver object is declared. This function is
        used to initialize the distribution function, and the field
        quantities using the options as provided by the user. The
        quantities are then mapped to the fourier basis by taking FFTs.
        The independant modes are then evolved by using the linear
        solver.
        """
        # af.broadcast(function, *args) performs batched operations on
        # function(*args):
        f     = af.broadcast(self.physical_system.initial_conditions.\
                             initialize_f, self.q1_center, self.q2_center,
                             self.p1, self.p2, self.p3, params
                             )
        # Taking FFT:
        f_hat = af.fft2(f)

        # Since (k_q1, k_q2) = (0, 0) will give the background distribution:
        # The division by (self.N_q1 * self.N_q2) is performed since the FFT
        # at (0, 0) returns (amplitude * (self.N_q1 * self.N_q2))
        self.f_background = af.abs(f_hat[0, 0, :])/ (self.N_q1 * self.N_q2)

        # Calculating derivatives of the background distribution function:
        self._calculate_dfdp_background()
   
        # Scaling Appropriately:
        # Except the case of (0, 0) the FFT returns
        # (0.5 * amplitude * (self.N_q1 * self.N_q2)):
        f_hat = 2 * f_hat / (self.N_q1 * self.N_q2)

        if(self.single_mode_evolution == True):
            # Background
            f_hat[0, 0, :] = 0

            # Finding the indices of the perturbation introduced:
            i_q1_max = np.unravel_index(af.imax(af.abs(f_hat))[1], 
                                        (self.N_q1, self.N_q2, 
                                         self.N_p1 * self.N_p2 * self.N_p3
                                        ),order = 'F'
                                       )[0]

            i_q2_max = np.unravel_index(af.imax(af.abs(f_hat))[1], 
                                        (self.N_q1, self.N_q2, 
                                         self.N_p1 * self.N_p2 * self.N_p3
                                        ),order = 'F'
                                       )[1]

            # Taking sum to get a scalar value:
            params.k_q1 = af.sum(self.k_q1[i_q1_max, i_q2_max])
            params.k_q2 = af.sum(self.k_q2[i_q1_max, i_q2_max])

            self.f_background = np.array(self.f_background).\
                                reshape(self.N_p1, self.N_p2, self.N_p3)

            self.p1 = np.array(self.p1).\
                      reshape(self.N_p1, self.N_p2, self.N_p3)
            self.p2 = np.array(self.p2).\
                      reshape(self.N_p1, self.N_p2, self.N_p3)
            self.p3 = np.array(self.p3).\
                      reshape(self.N_p1, self.N_p2, self.N_p3)

            self._A_q1 = np.array(self._A_q1).\
                         reshape(self.N_p1, self.N_p2, self.N_p3)
            self._A_q2 = np.array(self._A_q2).\
                         reshape(self.N_p1, self.N_p2, self.N_p3)

            params.rho_background = self.compute_moments('density',
                                                         self.f_background
                                                        )
            
            params.p1_bulk_background = self.compute_moments('mom_p1_bulk',
                                                             self.f_background
                                                            ) / params.rho_background
            params.p2_bulk_background = self.compute_moments('mom_p2_bulk',
                                                             self.f_background
                                                            ) / params.rho_background
            params.p3_bulk_background = self.compute_moments('mom_p3_bulk',
                                                             self.f_background
                                                            ) / params.rho_background
            
            params.T_background = ((1 / params.p_dim) * \
                                   self.compute_moments('energy',
                                                        self.f_background
                                                       )
                                   - params.rho_background * 
                                     params.p1_bulk_background**2
                                   - params.rho_background * 
                                     params.p2_bulk_background**2
                                   - params.rho_background * 
                                     params.p3_bulk_background**2
                                  ) / params.rho_background

            self.dfdp1_background = np.array(self.dfdp1_background).\
                                    reshape(self.N_p1, self.N_p2, self.N_p3)
            self.dfdp2_background = np.array(self.dfdp2_background).\
                                    reshape(self.N_p1, self.N_p2, self.N_p3)
            self.dfdp3_background = np.array(self.dfdp3_background).\
                                    reshape(self.N_p1, self.N_p2, self.N_p3)

            # Unable to recover pert_real and pert_imag accurately from the input.
            # There seems to be some error in recovering these quantities which
            # is dependant upon the resolution. Using user-defined quantities instead
            delta_f_hat =   params.pert_real * self.f_background \
                          + params.pert_imag * self.f_background * 1j 

            self.Y = np.array([delta_f_hat])
            compute_electrostatic_fields(self)
            self.Y = np.array([delta_f_hat, 
                               self.delta_E1_hat, self.delta_E2_hat, self.delta_E3_hat,
                               self.delta_B1_hat, self.delta_B2_hat, self.delta_B3_hat
                              ]
                             )

        else:
            # Using a vector Y to evolve the system:
            self.Y = af.constant(0, self.N_q1, self.N_q2,
                                 self.N_p1 * self.N_p2 * self.N_p3,
                                 7, dtype = af.Dtype.c64
                                )

            # Assigning the 0th indice along axis 3 to the f_hat:
            self.Y[:, :, :, 0] = f_hat

            # Initializing the EM field quantities:
            self.E3_hat = af.constant(0, self.N_q1, self.N_q2,
                                      self.N_p1 * self.N_p2 * self.N_p3, 
                                      dtype = af.Dtype.c64
                                     )

            self.B1_hat = af.constant(0, self.N_q1, self.N_q2,
                                      self.N_p1 * self.N_p2 * self.N_p3,
                                      dtype = af.Dtype.c64
                                     )
            
            self.B2_hat = af.constant(0, self.N_q1, self.N_q2,
                                      self.N_p1 * self.N_p2 * self.N_p3,
                                      dtype = af.Dtype.c64
                                     ) 
            
            self.B3_hat = af.constant(0, self.N_q1, self.N_q2,
                                      self.N_p1 * self.N_p2 * self.N_p3,
                                      dtype = af.Dtype.c64
                                     )
            
            # Initializing EM fields using Poisson Equation:
            if(self.physical_system.params.fields_initialize == 'electrostatic' or
               self.physical_system.params.fields_initialize == 'fft'
              ):
                compute_electrostatic_fields(self)

            # If option is given as user-defined:
            elif(self.physical_system.params.fields_initialize == 'user-defined'):
                E1, E2, E3 = \
                    self.physical_system.initial_conditions.initialize_E(self.q1_center, self.q2_center, self.physical_system.params)
                
                B1, B2, B3 = \
                    self.physical_system.initial_conditions.initialize_B(self.q1_center, self.q2_center, self.physical_system.params)

                # Scaling Appropriately
                self.E1_hat = af.tile(2 * af.fft2(E1) / (self.N_q1 * self.N_q2), 1, 1, self.N_p1 * self.N_p2 * self.N_p3)
                self.E2_hat = af.tile(2 * af.fft2(E2) / (self.N_q1 * self.N_q2), 1, 1, self.N_p1 * self.N_p2 * self.N_p3)
                self.E3_hat = af.tile(2 * af.fft2(E3) / (self.N_q1 * self.N_q2), 1, 1, self.N_p1 * self.N_p2 * self.N_p3)
                self.B1_hat = af.tile(2 * af.fft2(B1) / (self.N_q1 * self.N_q2), 1, 1, self.N_p1 * self.N_p2 * self.N_p3)
                self.B2_hat = af.tile(2 * af.fft2(B2) / (self.N_q1 * self.N_q2), 1, 1, self.N_p1 * self.N_p2 * self.N_p3)
                self.B3_hat = af.tile(2 * af.fft2(B3) / (self.N_q1 * self.N_q2), 1, 1, self.N_p1 * self.N_p2 * self.N_p3)
                
            else:
                raise NotImplementedError('Method invalid/not-implemented')

            # Assigning other indices along axis 3 to be
            # the EM field quantities:
            self.Y[:, :, :, 1] = self.E1_hat
            self.Y[:, :, :, 2] = self.E2_hat
            self.Y[:, :, :, 3] = self.E3_hat
            self.Y[:, :, :, 4] = self.B1_hat
            self.Y[:, :, :, 5] = self.B2_hat
            self.Y[:, :, :, 6] = self.B3_hat

            af.eval(self.Y)

        return

    # Injection of solver methods from other files:
    # Assigning function that is used in computing the derivatives
    # of the background distribution function:
    _calculate_dfdp_background = calculate_dfdp_background

    # Time-steppers:
    RK2_timestep = timestep.RK2_step
    RK4_timestep = timestep.RK4_step
    RK5_timestep = timestep.RK5_step

    # Routine which is used in computing the 
    # moments of the distribution function:
    compute_moments = compute_moments_imported

    # Methods used in writing the data to dump-files:
    dump_distribution_function = dump.dump_distribution_function
    dump_moments               = dump.dump_moments

    # Used to read the data from file
    load_distribution_function = load.load_distribution_function
