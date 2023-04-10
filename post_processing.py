#!/usr/bin/env python
#%%

import sys
import dill
import numpy as np
from scipy import signal
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

#%%
# For plotting multiple data in a normalised time window
def plot_data(datas, t_s=0, t_f=1e3, datas2=None, ylims1=None, ylims2=None):
    window = np.asarray((t_target >= t_s) & (t_target <= t_f)).nonzero()
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

def find_curvature(theta_guess, fk_target, epsilon=0.01, max_iterations=1000):   #TODO - check more inputs
    theta_est = None
    for i in range(max_iterations):
        error = np.vstack([f_FK_mid(theta_guess,p_vals),f_FK_end(theta_guess,p_vals)]) - fk_target
        if np.linalg.norm(error) < epsilon:
            theta_est = theta_guess
            break
        else:
            J = np.vstack([f_J_mid(theta_guess, p_vals),f_J_end(theta_guess, p_vals)])
    if theta_est is None:
        print('Failed to converge')
    return theta_est

#%%
# Import csv data
data_dir = './paramID_data/sine_x_fast' # + sys.argv[2]
O_T_EE = np.loadtxt(data_dir + '/EE_pose.csv', delimiter=',', skiprows=1)
W = np.loadtxt(data_dir + '/EE_wrench.csv', delimiter=',', skiprows=1)
# Rescale timestamps to seconds since first msg
timestamp_begin = np.min([np.min(O_T_EE[:,0]), np.min(W[:,0])])
timestamp_end = np.min([np.max(O_T_EE[:,0]), np.max(W[:,0])])
t_OTEE = (O_T_EE[:,0]-timestamp_begin)/1e9
t_W = (W[:,0]-timestamp_begin)/1e9
t_end = (timestamp_end - timestamp_begin)/1e9

# Copy relevant planar data
X_meas = O_T_EE[:,13]
Z_meas = O_T_EE[:,15]
RPY_meas = R.from_matrix(np.array([[O_T_EE[:,1], O_T_EE[:,2],O_T_EE[:,3]],
                                   [O_T_EE[:,5], O_T_EE[:,6],O_T_EE[:,7]],
                                   [O_T_EE[:,9], O_T_EE[:,10],O_T_EE[:,11]]]).T).as_euler('xyz', degrees=False)
Phi_meas = RPY_meas[:,1] # TODO - check this with data that varies phi
Fx_meas = W[:,1]
Fz_meas = W[:,3]
Ty_meas = W[:,5]

# Interpolate to uniform sample times
freq_target = 30
t_target = np.arange(0, t_end, 1/freq_target)
X = np.interp(t_target, t_OTEE, X_meas)
Z = np.interp(t_target, t_OTEE, Z_meas)
Phi = np.interp(t_target, t_OTEE, Phi_meas)
# Resample other data rates
t_90Hz = np.arange(0, t_end, 1/90)  # TODO interpolating 100Hz to 90Hz ok?
Fx_90Hz = np.interp(t_90Hz, t_W, Fx_meas)
Fz_90Hz = np.interp(t_90Hz, t_W, Fz_meas)
Ty_90Hz = np.interp(t_90Hz, t_W, Ty_meas)
Fx = signal.decimate(Fx_90Hz, 3)   # Downsample to 30Hz
Fz = signal.decimate(Fz_90Hz, 3)
Ty = signal.decimate(Ty_90Hz, 3)

#%%
# Curavture IK
f_FK_mid = dill.load(open('./generated_functions/f_FK_mf','rb'))
f_FK_end = dill.load(open('./generated_functions/f_FK_ef','rb'))
f_J_mid = dill.load(open('./generated_functions/f_J_mf','rb'))
f_J_end = dill.load(open('./generated_functions/f_J_ef','rb'))

p_vals = [1.0, 0.5, 1.0, 0.1]

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
