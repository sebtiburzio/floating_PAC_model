#!/usr/bin/env python
#%%

import sys
import numpy as np
from scipy import signal
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import dill

#%%
# Import csv data
data_dir = './sine_x_fast' #./' + sys.argv[2]
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
Phi_meas = RPY_meas[:,1] # TODO - check this was data that varies phi
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
# plt.plot(t_OTEE[100:170],X_meas[100:170])
# plt.show()
# plt.plot(t_target[100:170],X[100:170])
# plt.show()
# plt.plot(t_target[100:170],dX[100:170])
# plt.show()
# plt.plot(t_target[100:170],ddX[100:170])
# plt.show()

#%%
# Filtering test
# plt.plot(t_OTEE[100:170],X_meas[100:170])
# plt.show()
# sos = signal.butter(10, 10, fs=freq_target, output='sos', btype='lowpass')
# X_filt = signal.sosfilt(sos, X)
# plt.plot(t_target[100:170],X_filt[100:170])
# plt.show()
# dX_filt = np.diff(X_filt)/(1/30)
# plt.plot(t_target[100:170],dX_filt[100:170])
# plt.show()
# ddX_filt = np.diff(dX_filt)/(1/30)
# plt.plot(t_target[100:170],ddX_filt[100:170])
# plt.show()
