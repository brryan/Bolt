#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import arrayfire as af

def calculate_dfdp_background(self):
    """
    Calculates the derivative of the background distribution 
    with respect to the variables p1, p2, p3. This is used to
    solve for the contribution from the fields
    """
    f_b = af.moddims(self.f_background, self.N_p1, self.N_p2, self.N_p3)

    # Using a 4th order central difference stencil:
    dfdp1_background = (-af.shift(f_b, -2) + 8 * af.shift(f_b, -1)
                        +af.shift(f_b,  2) - 8 * af.shift(f_b,  1)
                       ) / (12 * self.dp1)

    dfdp2_background = (-af.shift(f_b, 0, -2) + 8 * af.shift(f_b, 0, -1)
                        +af.shift(f_b, 0,  2) - 8 * af.shift(f_b, 0,  1)
                       ) / (12 * self.dp2)

    dfdp3_background = (-af.shift(f_b, 0, 0, -2) + 8 * af.shift(f_b, 0, 0, -1)
                        +af.shift(f_b, 0, 0,  2) - 8 * af.shift(f_b, 0, 0,  1)
                       ) / (12 * self.dp3)

    # Reordering such that the variations in velocity are along axis 2
    self.dfdp1_background = af.reorder(af.flat(dfdp1_background), 2, 3, 0, 1)
    self.dfdp2_background = af.reorder(af.flat(dfdp2_background), 2, 3, 0, 1)
    self.dfdp3_background = af.reorder(af.flat(dfdp3_background), 2, 3, 0, 1)

    af.eval(self.dfdp1_background,
            self.dfdp2_background,
            self.dfdp3_background
           )

    return
