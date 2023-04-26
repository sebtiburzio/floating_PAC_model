#!/usr/bin/env python
#%%

import sys
import pickle
import numpy as np
import mpmath as mp
import sympy as sm
from scipy import signal
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

#%%
# For plotting multiple data in a normalised time window
# Pass data as a list of arrays (eg [X] or [X,Z])
# Only works for data scaled to target sample rate
def plot_data(datas, t_s=0, t_f=1e3, datas2=None, ylims1=None, ylims2=None):
    window = np.asarray((t_target >= t_s) & (t_target < t_f)).nonzero()
    fig, ax1 = plt.subplots()

    for data in datas:
        ax1.tick_params(axis='y', labelcolor='tab:blue')
        ax1.plot(t_target[window], data[window], color='tab:blue')
    if ylims1 is not None:
        ax1.set_ylim(ylims1)

    if datas2 is not None:
        ax2 = ax1.twinx()
        for data in datas2:
            ax2.tick_params(axis='y', labelcolor='tab:orange')
            ax2.plot(t_target[window], data[window],color='tab:orange')
        if ylims2 is not None:
            ax2.set_ylim(ylims2)

    plt.show()

def plot_XZ(t_s=0, t_f=1e3):
    window = np.asarray((t_target >= t_s) & (t_target < t_f)).nonzero()
    plt.plot(X[window],Z[window],'tab:red')
    plt.plot(X_mid[window],Z_mid[window],'tab:green')
    plt.plot(X_end[window],Z_end[window],'tab:blue')
    plt.axis('equal')

def plot_fk_targets(t_s=0, t_f=1e3):
    window = np.asarray((t_target >= t_s) & (t_target < t_f)).nonzero()
    plt.scatter(0,0,c='tab:red',marker='+')
    plt.plot(fk_targets[window,0].squeeze(),fk_targets[window,1].squeeze(),'tab:green')
    plt.plot(fk_targets[window,2].squeeze(),fk_targets[window,3].squeeze(),'tab:blue')
    plt.axis('equal')

def plot_FK(q_repl):
    s_evals = np.linspace(0,1,21)
    FK_evals = np.empty((s_evals.size,2,1))
    FK_evals[:] = np.nan
    for i_s in range(s_evals.size):
       FK_evals[i_s] = np.array(f_FK(q_repl,p_vals,s_evals[i_s],0.0).apply(mp.re).tolist(),dtype=float)
    fig, ax = plt.subplots()
    # MANUALLY ADD TARGETS TO PLOT
    ax.scatter(fk_targets[n,0], fk_targets[n,1],s=2, c='tab:green')
    ax.scatter(fk_targets[n,2], fk_targets[n,3],s=2, c='tab:blue')
    # DONT FORGET TO REMOVE
    ax.plot(FK_evals[:,0],FK_evals[:,1],'tab:orange')
    plt.xlim(FK_evals[0,0]-0.8,FK_evals[0,0]+0.8)
    plt.ylim(FK_evals[0,1]-0.8,FK_evals[0,1]+0.2)
    fig.set_figwidth(8)
    ax.set_aspect('equal','box')
    ax.grid(True)
    plt.show()

def find_curvature(theta_guess, fk_target, epsilon=0.05, max_iterations=25):  
    theta_est = None
    for i in range(max_iterations):
        error = np.vstack([eval_midpt(theta_guess,p_vals), eval_endpt(theta_guess,p_vals)]) - fk_target.reshape(4,1)
        if np.linalg.norm(error) < epsilon:
            theta_est = theta_guess
            break
        else:
            J = np.vstack([eval_J_midpt(theta_guess, p_vals), eval_J_endpt(theta_guess, p_vals)])
            theta_guess = theta_guess - (np.linalg.pinv(J)@error).squeeze() # TODO - add step size
    if theta_est is None:
        print('Failed to converge\n')
        theta_est = np.array([np.nan, np.nan])
    return theta_est

#%%
# Import forward kinematics
# Constant parameters
m_L, m_E, L, D = sm.symbols('m_L m_E L D')  # m_L - total mass of cable, m_E - mass of weighted end
p = sm.Matrix([m_L, m_E, L, D])
# Configuration variables
theta_0, theta_1, x, z, phi = sm.symbols('theta_0 theta_1 x z phi')
theta = sm.Matrix([theta_0, theta_1])
q = sm.Matrix([theta_0, theta_1, x, z, phi])
# Integration variables
s, d = sm.symbols('s d')
# Load functions
f_FK_mid = sm.lambdify((theta,p), pickle.load(open("./generated_functions/fk_mid_fixed", "rb")), "mpmath")
f_FK_end = sm.lambdify((theta,p), pickle.load(open("./generated_functions/fk_end_fixed", "rb")), "mpmath")
f_J_mid = sm.lambdify((theta,p), pickle.load(open("./generated_functions/J_mid_fixed", "rb")), "mpmath")
f_J_end = sm.lambdify((theta,p), pickle.load(open("./generated_functions/J_end_fixed", "rb")), "mpmath")
f_FK = sm.lambdify((q,p,s,d), pickle.load(open("./generated_functions/fk", "rb")), "mpmath")
# Convenience functions to extract real floats from complex mpmath matrices
def eval_fk(q, p_vals, s, d): return np.array(f_FK(q, p_vals, s, d).apply(mp.re).tolist(), dtype=float)
def eval_midpt(theta, p_vals): return np.array(f_FK_mid(theta, p_vals).apply(mp.re).tolist(), dtype=float)
def eval_endpt(theta, p_vals): return np.array(f_FK_end(theta, p_vals).apply(mp.re).tolist(), dtype=float)
def eval_J_midpt(theta, p_vals): return np.array(f_J_mid(theta, p_vals).apply(mp.re).tolist(), dtype=float)
def eval_J_endpt(theta, p_vals): return np.array(f_J_end(theta, p_vals).apply(mp.re).tolist(), dtype=float)

#%%
# Import csv data
data_dir = './paramID_data/0417/rot_link6_w_mass' # + sys.argv[2]
O_T_EE = np.loadtxt(data_dir + '/EE_pose.csv', delimiter=',', skiprows=1)
markers = np.loadtxt(data_dir + '/marker_positions.csv', delimiter=',', skiprows=1)
W = np.loadtxt(data_dir + '/EE_wrench.csv', delimiter=',', skiprows=1)
# Rescale timestamps to seconds since first msg
timestamp_begin = np.max([np.min(O_T_EE[:,0]), np.min(markers[:,0]), np.min(W[:,0])])
timestamp_end = np.min([np.max(O_T_EE[:,0]), np.max(markers[:,0]), np.max(W[:,0])])
t_OTEE = (O_T_EE[:,0]-timestamp_begin)/1e9
t_markers = (markers[:,0]-timestamp_begin)/1e9
t_W = (W[:,0]-timestamp_begin)/1e9
t_end = (timestamp_end - timestamp_begin)/1e9

# Physical definitions for object set up
p_vals = [1.0, 1.0, 0.75, 0.015]
base_offset = 0.0 # Z-dir offset of cable attachment point from measured robot EE frame

# Copy relevant planar data
RPY_meas = R.from_matrix(np.array([[O_T_EE[:,1], O_T_EE[:,2],O_T_EE[:,3]],
                                   [O_T_EE[:,5], O_T_EE[:,6],O_T_EE[:,7]],
                                   [O_T_EE[:,9], O_T_EE[:,10],O_T_EE[:,11]]]).T).as_euler('xyz', degrees=False)
Phi_meas = RPY_meas[:,1]
X_meas = O_T_EE[:,13] + base_offset*np.cos(Phi_meas)
Z_meas = O_T_EE[:,15] + base_offset*np.sin(Phi_meas)
X_mid_meas = markers[:,1]
Z_mid_meas = markers[:,2]
X_end_meas = markers[:,3]
Z_end_meas = markers[:,4]
Fx_meas = W[:,1] # TODO - adjust F/T meas due to offset from FT frame?
Fz_meas = W[:,3]
Ty_meas = W[:,5]

# Interpolate to uniform sample times
freq_target = 30
t_target = np.arange(0, t_end, 1/freq_target)
X = np.interp(t_target, t_OTEE, X_meas)
Z = np.interp(t_target, t_OTEE, Z_meas) # TODO define offset from EE to object base
Phi = np.interp(t_target, t_OTEE, Phi_meas)
X_mid = np.interp(t_target, t_markers, X_mid_meas)
Z_mid = np.interp(t_target, t_markers, Z_mid_meas)
X_end = np.interp(t_target, t_markers, X_end_meas)
Z_end = np.interp(t_target, t_markers, Z_end_meas)
# Resample other data rates
t_90Hz = np.arange(0, t_end, 1/90)  # TODO interpolating 100Hz to 90Hz ok?
Fx_90Hz = np.interp(t_90Hz, t_W, Fx_meas)
Fz_90Hz = np.interp(t_90Hz, t_W, Fz_meas)
Ty_90Hz = np.interp(t_90Hz, t_W, Ty_meas)
Fx = signal.decimate(Fx_90Hz, 3)   # Downsample to 30Hz
Fz = signal.decimate(Fz_90Hz, 3)
Ty = signal.decimate(Ty_90Hz, 3)

#%%
# Transform marker points to fixed PAC frame (subtract X/Z, rotate back phi)
fk_targets_mid = np.vstack([X_mid-X,np.zeros_like(X),Z_mid-Z]).T
fk_targets_end = np.vstack([X_end-X,np.zeros_like(X),Z_end-Z]).T
R_Phi = R.from_euler('y', -Phi, degrees=False).as_matrix()
fk_targets_mid = np.einsum('ijk,ik->ij', R_Phi, fk_targets_mid)
fk_targets_end = np.einsum('ijk,ik->ij', R_Phi, fk_targets_end)
fk_targets = np.hstack([fk_targets_mid[:,(0,2)], fk_targets_end[:,(0,2)]])

theta_extracted = np.zeros((fk_targets.shape[0],2,))
theta_guess = np.array([1e-3, 1e-3])
#%%
for n in range(100): #range(fk_targets.shape[0]):
    print(n)
    theta_n = find_curvature(theta_guess, fk_targets[n,:])
    theta_extracted[n,:] = theta_n
    if np.isnan(theta_n).any():
        print('Failed to converge for sample' + str(n))
    else:
        theta_guess = theta_n

#%%
# Finite difference derivatives 
dX = np.diff(X)/(1/freq_target)
ddX = np.diff(dX)/(1/freq_target)
dZ = np.diff(Z)/(1/freq_target)
ddZ = np.diff(dZ)/(1/freq_target)
dPhi = np.diff(Phi)/(1/freq_target)
ddPhi = np.diff(dPhi)/(1/freq_target)

#%%
# Save data
# np.savez(data_dir + '/processed', t=t_target, 
#          X=X, Z=Z, Phi=Phi, Fx=Fx, Fz=Fz, Ty=Ty, 
#          dX=dX, ddX=ddX, dZ=dZ, 
#          ddZ=ddZ, dPhi=dPhi, ddPhi=ddPhi)

#%%
# Plotting examples

# Plot X
# plt.plot(t_target[120:200],X[120:200])
# plt.title('X')
# plt.show()
# plt.plot(t_W[380:620],Fx_meas[380:620])
# plt.title('Fx')
# plt.show()
# plt.plot(t_target[100:170],Ty[100:170])
# plt.title('Ty')
# plt.show()
# plt.plot(t_target[100:170],dX[100:170])
# plt.title('dX')
# plt.show()
# plt.plot(t_target[100:170],ddX[100:170])
# plt.title('ddX')
# plt.show()

# plt.plot(X_mid,Z_mid)
# plt.plot(X_end,Z_end)
# plt.plot(X,Z)
# plt.axis('equal')
# plt.xlim(0,0.7)

#%%
# # Filtering test
# plt.plot(t_OTEE[100:170],X_meas[100:170])
# plt.show()
# sos = signal.butter(5, 8, fs=freq_target, output='sos', btype='lowpass')
# X_filt = signal.sosfilt(sos, X)
# plt.plot(t_target[100:170],X_filt[100:170])
# plt.show()
# dX_filt = np.diff(X_filt)/(1/30)
# plt.plot(t_target[100:170],dX_filt[100:170])
# plt.show()
# ddX_filt = np.diff(dX_filt)/(1/30)
# plt.plot(t_target[100:170],ddX_filt[100:170])
# plt.show()

# %%
