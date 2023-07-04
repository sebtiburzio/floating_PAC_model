#!/usr/bin/env python
#%%
import os
import pickle
import csv
import numpy as np
import mpmath as mp
import sympy as sm
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

def get_FK(q_repl,num_pts=21):
    s_evals = np.linspace(0,1,num_pts)
    FK_evals = np.zeros((s_evals.size,2,1))
    for i_s in range(s_evals.size):
       FK_evals[i_s] = np.array(eval_fk(q_repl,p_vals,s_evals[i_s],0.0))
    return FK_evals.squeeze()

# Plot FK based on theta config and optionally an fk target for comparison
def plot_FK(q_repl,i=None):
    FK_evals = get_FK(q_repl)
    fig, ax = plt.subplots()
    ax.plot(FK_evals[:,0],FK_evals[:,1],'tab:orange')
    ax.scatter(FK_evals[10,0],FK_evals[10,1],s=2,c='m',zorder=2.5)
    ax.scatter(FK_evals[-1,0],FK_evals[-1,1],s=2,c='m',zorder=2.5)
    plt.xlim(FK_evals[0,0]-1.1*p_vals[2],FK_evals[0,0]+1.1*p_vals[2])
    plt.ylim(FK_evals[0,1]-1.1*p_vals[2],FK_evals[0,1]+1.1*p_vals[2])

    if i is not None:
        plt.scatter(0,0,c='tab:red',marker='+')
        plt.scatter(fk_targets[i,0],fk_targets[i,1],c='tab:green',marker='+')
        plt.scatter(fk_targets[i,2],fk_targets[i,3],c='tab:blue',marker='+')

    fig.set_figwidth(8)
    ax.set_aspect('equal','box')
    ax.grid(True)
    plt.show()

def plot_fk_targets(i):
    plt.scatter(0,0,c='tab:red',marker='+')
    plt.scatter(fk_targets[i,0],fk_targets[i,1],c='tab:green',marker='+')
    plt.scatter(fk_targets[i,2],fk_targets[i,3],c='tab:blue',marker='+')
    plt.axis('equal')
    plt.grid(True)

def find_curvature(theta_guess, fk_target, epsilon=0.01, max_iterations=10):  
    error_2norm_last = np.inf
    for i in range(max_iterations):
        error = (np.vstack([eval_midpt(theta_guess,p_vals), eval_endpt(theta_guess,p_vals)]) - fk_target.reshape(4,1))
        error_2norm = np.linalg.norm(error)
        if error_2norm < epsilon:
            print("Converged after " + str(i) + " iterations")
            return theta_guess, True
        else:
            if np.isclose(error_2norm, error_2norm_last):
                print("Error stable after iteration " + str(i))
                return theta_guess, False
            elif error_2norm > error_2norm_last:
                print("Error increasing after iteration " + str(i))
                return theta_guess_last, False
            else:
                theta_guess_last = theta_guess
                error_2norm_last = error_2norm
                J = np.vstack([eval_J_midpt(theta_guess, p_vals), eval_J_endpt(theta_guess, p_vals)])
                theta_guess = theta_guess - (np.linalg.pinv(J)@error).squeeze()
    print("Max iterations reached (check why)")
    return theta_guess, False

# Rotate 2D vectors on XY plane around robot +Y axis
def rot_XZ_on_Y(XZ,angle):
    R_angle = np.array([[np.cos(-angle), np.sin(-angle)], 
                        [-np.sin(-angle), np.cos(-angle)]]).T
    if len(XZ.shape) == 1:
        return R_angle@XZ
    else:
        return np.einsum('ijk,ik->ij', R_angle, XZ)

#%%
# Import forward kinematics
# Constant parameters
m_L, m_E, L, D = sm.symbols('m_L m_E L D')  # m_L - total mass of cable, m_E - mass of weighted end
p = sm.Matrix([m_L, m_E, L, D])
# Configuration variables
theta_0, theta_1 = sm.symbols('theta_0 theta_1')
theta = sm.Matrix([theta_0, theta_1])
# Integration variables
s, d = sm.symbols('s d')
# Load functions
f_FK = sm.lambdify((theta,p,s,d), pickle.load(open("./generated_functions/fixed/fk", "rb")), "mpmath")
def eval_fk(theta, p_vals, s, d): return np.array(f_FK(theta, p_vals, s, d).apply(mp.re).tolist(), dtype=float)
f_FK_mid = sm.lambdify((theta,p), pickle.load(open("./generated_functions/fk_mid_fixed", "rb")), "mpmath")
f_FK_end = sm.lambdify((theta,p), pickle.load(open("./generated_functions/fk_end_fixed", "rb")), "mpmath")
f_J_mid = sm.lambdify((theta,p), pickle.load(open("./generated_functions/J_mid_fixed", "rb")), "mpmath")
f_J_end = sm.lambdify((theta,p), pickle.load(open("./generated_functions/J_end_fixed", "rb")), "mpmath")
def eval_midpt(theta, p_vals): return np.array(f_FK_mid(theta, p_vals).apply(mp.re).tolist(), dtype=float)
def eval_endpt(theta, p_vals): return np.array(f_FK_end(theta, p_vals).apply(mp.re).tolist(), dtype=float)
def eval_J_midpt(theta, p_vals): return np.array(f_J_mid(theta, p_vals).apply(mp.re).tolist(), dtype=float)
def eval_J_endpt(theta, p_vals): return np.array(f_J_end(theta, p_vals).apply(mp.re).tolist(), dtype=float)

#%%
# Data paths
dataset_name = 'orange_static_combined'
data_date = '0623'
data_dir = os.getcwd() + '/paramID_data/' + data_date + '/' + dataset_name  # TODO different in image_processing (extra '/' on end), maybe make same?

print('Dataset: ' + dataset_name)
print('Date: ' + data_date)
print('Path: ' + data_dir)

markers = np.loadtxt(data_dir + '/marker_positions.csv', delimiter=',', skiprows=1, usecols=range(1,7))
O_T_EE = np.loadtxt(data_dir + '/EE_pose.csv', delimiter=',', skiprows=1, usecols=range(1,17))
RMat_EE = np.array([[O_T_EE[:,0], O_T_EE[:,1],O_T_EE[:,2]],
                    [O_T_EE[:,4], O_T_EE[:,5],O_T_EE[:,6]],
                    [O_T_EE[:,8], O_T_EE[:,9],O_T_EE[:,10]]]).T
RPY_EE = R.from_matrix(RMat_EE).as_euler('xzy', degrees=False) 
Gamma = RPY_EE[:,2] # Note this is the rotated angle of the robot - simualted gravity direction would be -ve this

# Import marker positions
X_base_meas = markers[:,0]
Z_base_meas = markers[:,1]
X_mid_meas = markers[:,2]
Z_mid_meas = markers[:,3]
X_end_meas = markers[:,4]
Z_end_meas = markers[:,5]

p_vals = [0.4, 0.23, 0.75, 0.015] # cable properties: mass (length), mass (end), length, radius

#%%
# Transform marker points to fixed PAC frame (subtract X/Z, rotate back phi)
fk_targets_mid = np.vstack([X_mid_meas-X_base_meas,Z_mid_meas-Z_base_meas]).T
fk_targets_end = np.vstack([X_end_meas-X_base_meas,Z_end_meas-Z_base_meas]).T
fk_targets = np.hstack([rot_XZ_on_Y(fk_targets_mid,-Gamma), rot_XZ_on_Y(fk_targets_end,-Gamma)])

#%%
# Iterate IK over data
theta_extracted = np.zeros((fk_targets.shape[0],2,))
# theta_guess = np.array([9.5,-14.5]) # For black cable
theta_guess = np.array([11,-19]) # For orange cable
IK_converged = np.zeros((fk_targets.shape[0],1,))

for n in range(fk_targets.shape[0]):
    theta_n, convergence = find_curvature(theta_guess, fk_targets[n,:])
    theta_extracted[n,:] = theta_n
    theta_guess = theta_n
    IK_converged[n] = convergence
Theta0 = theta_extracted[:,0]
Theta1 = theta_extracted[:,1]

#%%
# Plot result over fk_targets
for i in range(len(theta_extracted)):
    plot_FK(theta_extracted[i],i)
plt.close()

# %%
# Export to csv for matlab
if not os.path.exists(data_dir + '/data_out'):
            os.makedirs(data_dir + '/data_out')
with open(data_dir + '/data_out/theta_equilibria.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerow(['Gamma', 'Theta0', 'Theta1', 'X_mid', 'Z_mid', 'X_end', 'Z_end'])
    for n in range(len(Gamma)):
        writer.writerow([Gamma[n], Theta0[n], Theta1[n], 
                        fk_targets[n,0], fk_targets[n,1], 
                        fk_targets[n,2], fk_targets[n,3]])
# %%
