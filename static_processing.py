#!/usr/bin/env python
#%%
import os
import csv
import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

from utils import rot_XZ_on_Y, plot_FK, find_curvature
from generated_functions.fixed.fixed_base_functions import eval_fk, eval_midpt, eval_endpt, eval_J_midpt, eval_J_endpt
target_evaluators = [eval_midpt, eval_endpt, eval_J_midpt, eval_J_endpt]

#%%
# Data paths
dataset_name = 'orange_weighted_combined'
data_date = '0714'
data_dir = os.getcwd() + '/paramID_data/' + data_date + '/' + dataset_name

print('Dataset: ' + dataset_name)
print('Date: ' + data_date)
print('Path: ' + data_dir)

markers = np.loadtxt(data_dir + '/marker_positions.csv', delimiter=',', skiprows=1, usecols=range(1,7))
O_T_EE = np.loadtxt(data_dir + '/EE_pose.csv', delimiter=',', skiprows=1, usecols=range(1,17))
RMat_EE = np.array([[O_T_EE[:,0], O_T_EE[:,1],O_T_EE[:,2]],
                    [O_T_EE[:,4], O_T_EE[:,5],O_T_EE[:,6]],
                    [O_T_EE[:,8], O_T_EE[:,9],O_T_EE[:,10]]]).T
# Expect orientation to be pi around x axis then phi around y axis
RPY_EE = R.from_matrix(RMat_EE).as_euler('xzy', degrees=False)
Gamma = RPY_EE[:,2] # Note this is the rotated angle of the robot - simulated gravity direction would be -ve this
plt.plot(Gamma) # Check the RPY conversion worked

# Import marker positions
X_base_meas = markers[:,0]
Z_base_meas = markers[:,1]
X_mid_meas = markers[:,2]
Z_mid_meas = markers[:,3]
X_end_meas = markers[:,4]
Z_end_meas = markers[:,5]

p_vals = [0.4, 0.23, 0.75, 0.015] # cable properties: mass (length), mass (end), length, diameter

#%%
# Transform marker points to fixed PAC frame (subtract X/Z, rotate back phi)
fk_targets_mid = np.vstack([X_mid_meas-X_base_meas,Z_mid_meas-Z_base_meas]).T
fk_targets_end = np.vstack([X_end_meas-X_base_meas,Z_end_meas-Z_base_meas]).T
fk_targets = np.hstack([rot_XZ_on_Y(fk_targets_mid,-Gamma), rot_XZ_on_Y(fk_targets_end,-Gamma)])

#%%
# Iterate IK over data
theta_extracted = np.zeros((fk_targets.shape[0],2,))
# theta_guess = np.array([8.5,-12.2]) # For black cable
theta_guess = np.array([11,-18]) # For orange cable
IK_converged = np.zeros((fk_targets.shape[0],1,))

for n in range(fk_targets.shape[0]):
    theta_n, convergence = find_curvature(p_vals,theta_guess,target_evaluators,fk_targets[n,:])
    theta_extracted[n,:] = theta_n
    theta_guess = theta_n
    IK_converged[n] = convergence
Theta0 = theta_extracted[:,0]
Theta1 = theta_extracted[:,1]

#%%
# Plot result over fk_targets
for i in range(len(theta_extracted)):
    print(str(Gamma[i]*180/np.pi) + 'deg')
    plot_FK(p_vals,theta_extracted[i],eval_fk,fk_targets[i])
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
