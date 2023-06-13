#!/usr/bin/env python

#%%
import os
import pickle
import numpy as np
import mpmath as mp
import sympy as sm
import matplotlib.pyplot as plt

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
    Theta0 = dataset['Theta0']
    Theta1 = dataset['Theta1'] 
    dTheta0 = dataset['dTheta0']
    dTheta1 = dataset['dTheta1']
    ddTheta0 = dataset['ddTheta0']
    ddTheta1 = dataset['ddTheta1']

# Check Theta equilibrium
plt.plot(Theta0, label='Theta0')
plt.plot(Theta1, label='Theta1')
plt.show()
plt.plot(Theta0[-30:], label='Theta0')
plt.plot(Theta1[-30:], label='Theta1')
plt.show()

#%%
# Offset Theta to equilibrium state
Theta0 = Theta0 - np.mean(Theta0[:-30])
Theta1 = Theta1 - np.mean(Theta1[:-30])

#%%
# Theta vectors
Theta = np.vstack((Theta0, Theta1))
dTheta = np.vstack((dTheta0, dTheta1))
ddTheta = np.vstack((ddTheta0, ddTheta1))
num_samples = Theta.shape[1]
#Stiffness/damping identification coefficient
c00 = Theta0 + Theta1/2.0
c01 = dTheta0 + dTheta1/2.0
c10 = Theta0/2.0 + Theta1/3.0
c11 = dTheta0/2.0 + dTheta1/3.0

#%%
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

# %%
# Identify stiffness/damping only
# Load EOM functions, replace constant parameters
F_G = pickle.load(open("./generated_functions/fixed/G", "rb"))
F_B = pickle.load(open("./generated_functions/fixed/B", "rb"))
F_C = pickle.load(open("./generated_functions/fixed/C", "rb"))
F_G = F_G.subs([(m_L,p_vals[0]),(m_E,p_vals[1]),(L,p_vals[2]),(D,p_vals[3])])
F_B = F_B.subs([(m_L,p_vals[0]),(m_E,p_vals[1]),(L,p_vals[2]),(D,p_vals[3])])
F_C = F_C.subs([(m_L,p_vals[0]),(m_E,p_vals[1]),(L,p_vals[2]),(D,p_vals[3])])
f_G = sm.lambdify([theta], F_G, "mpmath")
f_B = sm.lambdify([theta], F_B, "mpmath")
f_C = sm.lambdify([theta, dtheta], F_C, "mpmath")
# Convenience functions to extract real floats from complex mpmath matrices
def eval_G(theta): return np.array(f_G(theta).apply(mp.re).tolist(), dtype=float)
def eval_B(theta): return np.array(f_B(theta).apply(mp.re).tolist(), dtype=float)
def eval_C(theta, dtheta): return np.array(f_C(theta, dtheta).apply(mp.re).tolist(), dtype=float)

#%%
# Calc delta vector
for sample in range(num_samples):
    if sample % 10 == 0:
        print(str(sample) + ' of ' + str(num_samples))

    # RHS = 0 - B*ddQ - C*dQ - G 
    RHS = - eval_B(Theta[:,sample])@ddTheta[:,sample].reshape((2,1)) \
          - eval_C(Theta[:,sample], dTheta[:,sample])@dTheta[:,sample].reshape((2,1)) \
          - eval_G(Theta[:,sample]).reshape((2,1))
    
    if sample == 0:
        delta = RHS
    else:
        delta = np.vstack((delta, RHS))

for sample in range(num_samples):
    
    Y_n = np.array(([c00[sample],c01[sample]],[c10[sample],c11[sample]]))

    if sample == 0:
        Y = Y_n
    else:
        Y = np.vstack((Y, Y_n))

# %%
# Identify mass, stiffness, damping
# Load EOM functions, replace constant parameters
F_Y = pickle.load(open("./generated_functions/Y", "rb"))
F_Y = F_Y.subs([(L,p_vals[2]),(D,p_vals[3])])
f_Y = sm.lambdify([theta, dtheta, ddtheta], F_Y, "mpmath")
# Convenience functions to extract real floats from complex mpmath matrices
def eval_Y(theta, dtheta, ddtheta): return np.array(f_Y(theta,dtheta,ddtheta).apply(mp.re).tolist(), dtype=float)

for sample in range(num_samples):
    if sample % 10 == 0:
        print(str(sample) + ' of ' + str(num_samples))
    
    Y_n = np.hstack((np.array(([c00[sample],c01[sample]],[c10[sample],c11[sample]])), 
                     eval_Y([Theta0[sample], Theta1[sample]], [dTheta0[sample], dTheta1[sample]], [ddTheta0[sample], ddTheta1[sample]])
    ))

    if sample == 0:
        Y = Y_n
    else:
        Y = np.vstack((Y, Y_n))

# Calc delta vector
# It is 0 -> cannot get a unique solution

# %%
# Identify cable mass, stiffness and damping
# Load EOM functions, replace constant parameters
F_dE_dmL = pickle.load(open("./generated_functions/fixed/dE_dmL", "rb"))
F_dE_dmL = F_dE_dmL.subs([(L,p_vals[2]),(D,p_vals[3])])
f_dE_dmL = sm.lambdify([theta, dtheta, ddtheta], F_dE_dmL, "mpmath")
F_E_mL_0 = pickle.load(open("./generated_functions/fixed/E_mL_0", "rb"))
F_E_mL_0 = F_E_mL_0.subs([(m_E,p_vals[1]),(L,p_vals[2]),(D,p_vals[3])])
f_E_mL_0 = sm.lambdify([theta, dtheta, ddtheta], F_E_mL_0, "mpmath")

# Convenience functions to extract real floats from complex mpmath matrices
def eval_dE_dmL(theta, dtheta, ddtheta): return np.array(f_dE_dmL(theta,dtheta,ddtheta).apply(mp.re).tolist(), dtype=float)
def eval_E_mL_0(theta, dtheta, ddtheta): return np.array(f_E_mL_0(theta,dtheta,ddtheta).apply(mp.re).tolist(), dtype=float)

#%%
# Calc delta vector & Y matrix
for sample in range(num_samples):
    if sample % 10 == 0:
        print(str(sample) + ' of ' + str(num_samples))

    RHS = - eval_E_mL_0([Theta0[sample], Theta1[sample]], [dTheta0[sample], dTheta1[sample]], [ddTheta0[sample], ddTheta1[sample]])
    
    if sample == 0:
        delta = RHS
    else:
        delta = np.vstack((delta, RHS))
    
    Y_n = np.hstack((np.array(([c00[sample],c01[sample]],[c10[sample],c11[sample]])), 
                     eval_dE_dmL([Theta0[sample], Theta1[sample]], [dTheta0[sample], dTheta1[sample]], [ddTheta0[sample], ddTheta1[sample]])
    ))

    if sample == 0:
        Y = Y_n
    else:
        Y = np.vstack((Y, Y_n))

#%%
# Evalaute Pi matrix
Pi = np.linalg.pinv(Y)@delta
print(Pi)
np.savetxt(data_dir + '/Pi.txt', Pi, delimiter=',')
# %%
