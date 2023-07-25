#!/usr/bin/env python

#%%
import os
import pickle
import numpy as np
import mpmath as mp
import sympy as sm

#%%
# Load data
# Data paths
dataset_name = 'black_swing'
data_date = '0609'
data_dir = os.getcwd() + '/paramID_data/' + data_date + '/' + dataset_name  # TODO different in image_processing (extra '/' on end), maybe make same?

print('Dataset: ' + dataset_name)
print('Date: ' + data_date)
print('Path: ' + data_dir)

with np.load(data_dir + '/processed.npz') as dataset:
    p_vals = dataset['p_vals']
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

#%%
# Import EOM functions
# Constant parameters
m_L, m_E, L, D = sm.symbols('m_L m_E L D')  # m_L - total mass of cable, m_E - mass of weighted end
p = sm.Matrix([m_L, m_E, L, D])
# Configuration variables
theta_0, theta_1, x, z, phi = sm.symbols('theta_0 theta_1 x z phi')
theta = sm.Matrix([theta_0, theta_1])
dtheta_0, dtheta_1 = sm.symbols('dtheta_0 dtheta_1')
dtheta = sm.Matrix([dtheta_0, dtheta_1])
ddtheta_0, ddtheta_1 = sm.symbols('ddtheta_0 ddtheta_1')
ddtheta = sm.Matrix([ddtheta_0, ddtheta_1])
q = sm.Matrix([theta_0, theta_1, x, z, phi])
# Integration variables
s, d = sm.symbols('s d')

# Load EOM functions, replace constant parameters # TODO - change to loading from module
F_G = pickle.load(open("./generated_functions/G", "rb"))
F_B = pickle.load(open("./generated_functions/B", "rb"))
F_G = F_G.subs([(m_L,p_vals[0]),(m_E,p_vals[1]),(L,p_vals[2]),(D,p_vals[3])])
F_B = F_B.subs([(m_L,p_vals[0]),(m_E,p_vals[1]),(L,p_vals[2]),(D,p_vals[3])])
f_G = sm.lambdify(q, F_G, "mpmath")
f_B = sm.lambdify(q, F_B, "mpmath")
# Convenience functions to extract real floats from complex mpmath matrices
def eval_G(q): return np.array(f_G(q[0],q[1],q[2],q[3],q[4]).apply(mp.re).tolist(), dtype=float)
def eval_B(q): return np.array(f_B(q[0],q[1],q[2],q[3],q[4]).apply(mp.re).tolist(), dtype=float)

#%%
# Create q, qdot, qddot and F mega vectors (single 5*N column)
Q = np.vstack((Theta0, Theta1, X, Z, Phi))
dQ = np.vstack((dTheta0, dTheta1, dX, dZ, dPhi))
ddQ = np.vstack((ddTheta0, ddTheta1, ddX, ddZ, ddPhi))
F = np.vstack((-Fx, -Fz, -Ty)) # Force applied to the cable
num_samples = Q.shape[1]

#%%
# Calc delta vector
for sample in range(num_samples):
    if sample % 10 == 0:
        print(str(sample) + ' of ' + str(num_samples))

    # RHS = F - B*ddQ - G (no C matrix)
    RHS = (np.hstack((np.zeros((2,)),F[:,sample])) - eval_B(Q[:,sample])@ddQ[:,sample] - eval_G(Q[:,sample])).reshape((5,1))
    if sample == 0:
        delta = RHS
    else:
        delta = np.vstack((delta, RHS))

#%%
# Create Y matrix
c00 = Theta0 + Theta1/2.0
c01 = dTheta0 + dTheta1/2.0
c10 = Theta0/2.0 + Theta1/3.0
c11 = dTheta0/2.0 + dTheta1/3.0

for sample in range(num_samples):

    if sample == 0:
        Y = np.vstack((np.array(([c00[sample],c01[sample]],[c10[sample],c11[sample]])),
                       np.zeros((3,2))
        ))
    else:
        Y = np.vstack((Y, 
                       np.array(([c00[sample], c01[sample]],
                                 [c10[sample], c11[sample]])),
                       np.zeros((3,2))
        ))

#%%
# Evalaute Pi matrix
Pi = np.linalg.pinv(Y)@delta
print(Pi)
np.savetxt(data_dir + '/Pi.txt', Pi, delimiter=',')