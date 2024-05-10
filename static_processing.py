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
dataset_name = 'equilibria'
data_date = '0402-loop_static_id'
data_dir = os.getcwd() + '/paramID_data/' + data_date + '/' + dataset_name

print('Dataset: ' + dataset_name)
print('Date: ' + data_date)
print('Path: ' + data_dir)

data = np.loadtxt(data_dir + '/sequence_results.csv', delimiter=',', skiprows=1, usecols=range(0,10))
Gamma = data[:,3]
X_base_meas = data[:,4]
Z_base_meas = data[:,5]
X_mid_meas = data[:,6]
Z_mid_meas = data[:,7]
X_end_meas = data[:,8]
Z_end_meas = data[:,9]

with np.load('object_parameters/black_short_loop_100g.npz') as obj_params:
    p_vals = list(obj_params['p_vals']) # cable properties: mass (length), mass (end), length, diameter

#%%
# Transform marker points to fixed PAC frame (subtract X/Z, rotate back phi)
fk_targets_mid = np.vstack([X_mid_meas-X_base_meas,Z_mid_meas-Z_base_meas]).T
fk_targets_end = np.vstack([X_end_meas-X_base_meas,Z_end_meas-Z_base_meas]).T
fk_targets = np.hstack([rot_XZ_on_Y(fk_targets_mid,-Gamma), rot_XZ_on_Y(fk_targets_end,-Gamma)])

#%%
# Iterate IK over data
theta_extracted = np.zeros((fk_targets.shape[0],2,))
theta_guess = np.array([1e-3,1e-3]) # Assuming first point angle 0
IK_converged = np.zeros((fk_targets.shape[0],1,))

for n in range(fk_targets.shape[0]):
    theta_n, convergence = find_curvature(p_vals,theta_guess,target_evaluators,fk_targets[n,:])
    theta_extracted[n,:] = theta_n
    theta_guess = theta_n if n == 0 else theta_extracted[n-1,:] # HACK - assumes angles ordered and alternating +ve -ve
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
