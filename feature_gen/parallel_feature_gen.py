"""
parallel_feature_gen.py

- Author: Dmitri Lyalikov
- Email: DLyalikov01@manhattan.edu
- Date of Last Revision: 05/04/2023

This python module generates training set data for the pdt-regression models.
For a parallel speedup, the multiprocessing module is used to allow computational processes to execute
on multiple cores
"""

import multiprocessing as mp
import numpy as np
from circle_fit import taubinSVD
import math
from scipy.integrate import odeint
import csv


def Drop_Profil(z, t, Beta):
    x = z[0]
    y = z[1]
    phi = z[2]
    dxdt = math.cos(phi)
    dydt = math.sin(phi)
    dphidt = 2 - Beta * y - math.sin(phi) / x
    dzdt = [dxdt, dydt, dphidt]
    return dzdt


# Adding noise to the drop profile
def Add_Noise_Drop_Profile(z, noise_Percent_of_datamean):
    y = z[:, 1]
    x = z[:, 0]
    noise = np.random.normal(0, 1, len(x))  # normal distribution mean=0 STD=1
    # Adding fraction of data average as noise
    x = x + noise / noise.max() * noise_Percent_of_datamean * x.mean()
    y = y + noise / noise.max() * noise_Percent_of_datamean * y.mean()
    # Sorting data from ymin to ymax for further analysis
    loc_y_incr = np.argsort(y)
    x = x[loc_y_incr]  # sorted from apex
    y = y[loc_y_incr]  # sorted from apex
    return x, y


# Finding Equator radius (Re) and Rs @ y=2Re
def Find_Re_Rs(x, y, n, Drop_Height):
    # at i_th x: we average from i-n to i+n which means 11 points
    global R_e
    R_e = 0
    i = n

    def Average_x(x, i, n):
        # i-n>=0
        s = 0
        for j in range(i - n, i + n + 1):
            s = s + x[j]
        return s / (2 * n + 1)

    def Recur_Equator_Radius(x, i, n):
        global R_e
        # R_e is the equator radius: must be defined here not outside
        # We use recursive approach: Start from Apex continue until x decreases
        # At i-th point we averagne x of x-n to x+n to subpress noise
        # We compare x_i_th vs x_i+1_th until it decrease to find equator
        if Average_x(x, i, n) < Average_x(x, i + 1, n) and i <= len(x) - n - 3:
            i = i + 1
            Output = Recur_Equator_Radius(x, i, n)
            if Output is not None:
                # Since recursive returns None!!! we use global Variable
                R_e = Output
        else:
            if i <= len(x) * 0.7:
                # I assumed 70% of drop is enough for R_e
                # print(i)
                # print(x[i])
                return x[i]
            else:
                return

    # A recursive function that returns equator radius
    Recur_Equator_Radius(x, i, n)

    if R_e == 0:
        # R_e=0: drop is not well-deformed e.g. Beta>0.7. Find R_e from cirle fitting
        # I selected 40% of the total number of points for circle fitting
        num_point_RH_Circlefit = round(0.4 * len(x))
        Points_RH_Circlefit = np.stack((x[:num_point_RH_Circlefit], y[:num_point_RH_Circlefit]), axis=1)
        xc, yc, R_e, sigma = taubinSVD(Points_RH_Circlefit)
    # Find R_s at y=2*R_e
    if R_e < 0.5 * Drop_Height:
        # res=index of y if y>2*R_e
        res = next(xx for xx, val in enumerate(y) if val > 2 * R_e)
        R_s = x[res]
    else:
        # Drop is too small
        R_s = x[-1] # R_cap
    return R_e, R_s


# This function returns Smin and Smax for integration based on Beta values
# Developed based on Helen, Payton and Dmitri Excel file
def find_Smin_Smax(Beta, SF):
    # SF is a safety factor. we report (1+SF)*Smin and (1-SF)*Smax
    # I used 2% safety factor
    Smin = 2.31 + 11.4 * Beta - 27.1 * Beta ** 2 + 16.5 * Beta ** 3
    if Beta > 0.3:
        Smax = 28.4 - 85.1 * Beta + 107 * Beta ** 2 - 46.9 * Beta ** 3
    else:
        Smax = 1.86 * Beta + 4.46
    return (1 + SF) * Smin, (1 - SF) * Smax

def compute_kernel(beta: int,):
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

        return [Drop_Height, R_Cap, R_s, R_e, beta]


if "__name__" == "__main__":
    print("here")
    num_processes_per_block = mp.cpu_count()  # number of processes that can run at a given time

    num_point_integration = 200
    noise_Percent_of_datamean = 0.01
    Beta_min = 0.1
    Beta_max = 0.8
    Beta_num = 10  # Number of total processes to spawn
    Smax_num = 10  # internal iteration count for compute kernel

    pool = mp.Pool(processes=num_processes_per_block)

    beta_values = np.linspace(Beta_min, Beta_max, num=Beta_num)

    arguments = [(beta,) for beta in beta_values]

    results = pool.map(compute_kernel, arguments)

    # write the results to a CSV file
    with open('output.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        for row in results:
            writer.writerow(row)

    print("done")







