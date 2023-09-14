#!/usr/bin/env python

import numpy as np
import csv

with np.load('./processed.npz') as dataset:
    t = dataset['t'] 
    X = dataset['X']
    Z = dataset['Z']
    Phi = dataset['Phi']
    Theta0 = dataset['Theta0']
    Theta1 = dataset['Theta1'] 
    dX = dataset['dX']
    dZ = dataset['dZ']
    dPhi = dataset['dPhi']
    dTheta0 = dataset['dTheta0']
    dTheta1 = dataset['dTheta1']
    ddX = dataset['ddX']
    ddZ = dataset['ddZ']
    ddPhi = dataset['ddPhi']
    ddTheta0 = dataset['ddTheta0']
    ddTheta1 = dataset['ddTheta1']
    Fx = dataset['Fx']
    Fz = dataset['Fz']
    Ty = dataset['Ty']

with open('./data_out/state_and_input.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerow(['ts', 'X', 'Z', 'Phi', 'Theta0', 'Theta1', 
                     'dX', 'dZ', 'dPhi', 'dTheta0', 'dTheta1', 
                     'ddX', 'ddZ', 'ddPhi', 'ddTheta0', 'ddTheta1',
                     'Fx', 'Fz', 'Ty'])
    for n in range(len(t)):
        writer.writerow([t[n], 
                         X[n], Z[n], Phi[n], Theta0[n], Theta1[n], 
                         dX[n], dZ[n], dPhi[n], dTheta0[n], dTheta1[n], 
                         ddX[n], ddZ[n], ddPhi[n], ddTheta0[n], ddTheta1[n],
                         Fx[n], Fz[n], Ty[n]])