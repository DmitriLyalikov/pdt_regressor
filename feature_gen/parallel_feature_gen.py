"""
parallel_feature_gen.py

- Author: Dmitri Lyalikov
- Email: DLyalikov01@manhattan.edu
- Date of Last Revision: 05/04/2023

This python module generates training set data for the pdt-regression models.
For a parallel speedup, the multiprocessing module is used to allow computational processes to execute
on multiple cores
"""

import os
from multiprocessing import Process, Lock, Array, Value
import multiprocessing as mp
import numpy as np


def compute_kernel(l,
                   beta: int,
                   Smax_num: int,
                   num_point_integration: int,
                   noise_Percent_of_datamean: int,
                   Drop_height_values,
                   R_Cap_values,
                   R_e_values,
                   R_s_values,
                   Beta_values):
    Smin, Smax = find_Smin_Smax(beta, SF=0.02)
    for S in np.linspace(Smin, Smax, num=Smax_num):
        # solve ODE to generate the drop profile datapoint
        z0 = [0.0000001, 0.0000001, 0.0000001]
        t = np.linspace(0, S, num_point_integration)
        z = odeint(Drop_Profil, z0, t, args=(beta,))

        x, y = Add_Noise_Drop_Profile(z, noise_Percent_of_datamean)

        # Generate outputs
        Drop_Height = y[-1]
        R_Cap = x[-1]

        R_e, R_s = Find_Re_Rs(x, y, 5, Drop_Height)

        with l:
            Drop_height_values.append(Drop_Height)
            R_Cap_values.append(R_Cap)
            R_e_values.append(R_e)
            R_s_values.append(R_s)


if "__name__" == "__main__":

    num_processes_per_block = mp.cpu_count()  # number of processes that can run at a given time
    Beta_num = 501  # Number of total processes to spawn
    Smax_num = 101  # internal iteration count for compute kernel

    Drop_height_values = Array('d')
    R_Cap_values = Array('d')
    R_s_values = Array('d')
    R_e_values = Array('d')
    Beta_values = Array('d')

    lock = Lock()
    for Beta in np.linspace(Beta_min, Beta_max, num=Beta_num):
        # spawn a block of processes








