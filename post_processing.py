#!/usr/bin/env python
#%%

import sys
import dill
import numpy as np
import mpmath as mp
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

def find_curvature(theta_guess, fk_target, epsilon=0.01, max_iterations=1000):   # TODO - fix horrific conversions between mp and np
    theta_est = None
    for i in range(max_iterations):
        error = np.vstack([np.array(f_FK_mid(theta_guess,p_vals).apply(mp.re).tolist(),dtype=float),
                           np.array(f_FK_end(theta_guess,p_vals).apply(mp.re).tolist(),dtype=float)]) - fk_target.reshape(4,1)
        if np.linalg.norm(error) < epsilon:
            theta_est = theta_guess
            break
        else:
            J = np.vstack([np.array(f_J_mid(theta_guess, p_vals).apply(mp.re).tolist(),dtype=float),
                           np.array(f_J_end(theta_guess, p_vals).apply(mp.re).tolist(),dtype=float)])
            theta_guess = theta_guess - (np.linalg.pinv(J)@error).squeeze()
    if theta_est is None:
        print('Failed to converge\n')
        theta_est = np.array([np.nan, np.nan])
    return theta_est

#%%
# Import csv data
data_dir = './paramID_data/0406/sine_x_w_depth' # + sys.argv[2]
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

# Copy relevant planar data
X_meas = O_T_EE[:,13]
Z_meas = O_T_EE[:,15]
RPY_meas = R.from_matrix(np.array([[O_T_EE[:,1], O_T_EE[:,2],O_T_EE[:,3]],
                                   [O_T_EE[:,5], O_T_EE[:,6],O_T_EE[:,7]],
                                   [O_T_EE[:,9], O_T_EE[:,10],O_T_EE[:,11]]]).T).as_euler('xyz', degrees=False)
Phi_meas = RPY_meas[:,1] # TODO - check this with data that varies phi
X_mid_meas = markers[:,1]
Z_mid_meas = markers[:,2]
X_end_meas = markers[:,3]
Z_end_meas = markers[:,4]
Fx_meas = W[:,1]
Fz_meas = W[:,3]
Ty_meas = W[:,5]

# Interpolate to uniform sample times
freq_target = 30
t_target = np.arange(0, t_end, 1/freq_target)
X = np.interp(t_target, t_OTEE, X_meas)
Z = np.interp(t_target, t_OTEE, Z_meas)
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
# Transform to base frame (subtract X/Z rotate phi)
fk_targets = np.array([X_mid-X,Z_mid-Z,X_end-X,Z_end-Z]).T
# TODO - need to rotate by phi when orientation changes
# TODO - need to add/subtract pi/2 to align model axis with robot frame?

###
def plot_FK(q_repl):
    s_evals = np.linspace(0,1,11)
    FK_evals = np.empty((s_evals.size,2,))
    FK_evals[:] = np.nan
    for i_s in range(s_evals.size):
       FK_evals[i_s] = f_FK(q_repl,p_vals,s_evals[i_s],0.0)
    fig, ax = plt.subplots()
    ax.plot(FK_evals[:,0],FK_evals[:,1])
    plt.xlim(FK_evals[0,0]-0.8,FK_evals[0,0]+0.8)
    plt.ylim(FK_evals[0,1]-0.8,FK_evals[0,1]+0.2)
    fig.set_figwidth(8)
    ax.set_aspect('equal','box')
    plt.show()
# Loading functions not working, so redfine here for now
import sympy as sm
# Constant parameters
L, D = sm.symbols('L D')  # m_L - total mass of cable, m_E - mass of weighted end
p = sm.Matrix([L, D])
p_vals = [0.75, 0.01]
# Configuration variables
theta_0, theta_1= sm.symbols('theta_0 theta_1')
theta = sm.Matrix([theta_0, theta_1])
# Object coordinates
fk_x, fk_z = sm.symbols('fk_x fk_z')
fk = sm.Matrix([fk_x, fk_z])
alpha = sm.symbols('alpha') # tip orientation in object base frame
# Integration variables
s, v, d = sm.symbols('s v d')
# Spine x,z in object base frame
alpha = theta_0*v + 0.5*theta_1*v**2
fk[0] = L*sm.integrate(sm.sin(alpha),(v, 0, s))
fk[1] = -L*sm.integrate(sm.cos(alpha),(v, 0, s)) # TODO - recheck all frames etc to make model match robot...
# FK of midpoint and endpoint in base frame (for curvature IK)
fk_mid_fixed = fk.subs(s, 0.5)
fk_end_fixed = fk.subs(s, 1)
J_mid_fixed = fk_mid_fixed.jacobian(sm.Matrix([theta_0, theta_1]))
J_end_fixed = fk_end_fixed.jacobian(sm.Matrix([theta_0, theta_1]))
f_FK_mid = sm.lambdify((theta,p), fk_mid_fixed, "mpmath")
f_FK_end = sm.lambdify((theta,p), fk_end_fixed, "mpmath")
f_J_mid = sm.lambdify((theta,p), J_mid_fixed, "mpmath")
f_J_end = sm.lambdify((theta,p), J_end_fixed, "mpmath")
# Full FK for plotting
f_FK = sm.lambdify((theta,p,s,d), fk, "mpmath")
###

theta_extracted = np.empty((fk_targets.shape[0],2,))
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
np.savez(data_dir + '/processed', t=t_target, 
         X=X, Z=Z, Phi=Phi, Fx=Fx, Fz=Fz, Ty=Ty, 
         dX=dX, ddX=ddX, dZ=dZ, 
         ddZ=ddZ, dPhi=dPhi, ddPhi=ddPhi)

#%%
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
