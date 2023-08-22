#!/usr/bin/env python
#%%
import os
import csv
import cv2
import numpy as np
from scipy import signal
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter

from generated_functions.floating.floating_base_functions import eval_fk
from generated_functions.fixed.fixed_base_functions import eval_midpt, eval_endpt, eval_J_midpt, eval_J_endpt
target_evaluators = [eval_midpt, eval_endpt, eval_J_midpt, eval_J_endpt]
from utils import rot_XZ_on_Y, get_FK, find_curvature

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
    plt.grid(True)

def plot_fk_targets(t_s=0, t_f=1e3):
    window = np.asarray((t_target >= t_s) & (t_target < t_f)).nonzero()
    plt.scatter(0,0,c='tab:red',marker='+')
    plt.plot(fk_targets[window,0].squeeze(),fk_targets[window,1].squeeze(),'tab:green')
    plt.plot(fk_targets[window,2].squeeze(),fk_targets[window,3].squeeze(),'tab:blue')
    plt.axis('equal')
    plt.grid(True)

def plot_calib_check():
    img = cv2.cvtColor(cv2.imread(img_dir + '/' + str(ts_markers[0]) + '.jpg'), cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots()
    ax.imshow(img,alpha=0.5)
    # End effector measured by robot at first sample
    EE_XYZ = np.array([O_T_EE[0,12],O_T_EE[0,13],O_T_EE[0,14],1.0])
    EE_X_axis = np.array([O_T_EE[0,12]+0.05*np.cos(Phi_meas[0]),O_T_EE[0,13],O_T_EE[0,14]-0.05*np.sin(Phi_meas[0]),1.0])
    EE_Z_axis = np.array([O_T_EE[0,12]-0.05*np.sin(Phi_meas[0]),O_T_EE[0,13],O_T_EE[0,14]-0.05*np.cos(Phi_meas[0]),1.0])
    EE_proj = P@EE_XYZ
    EE_X_axis_proj = P@EE_X_axis
    EE_Z_axis_proj = P@EE_Z_axis
    # Base of cable incorporating offset from EE
    base_XYZ = np.array([X_meas[0],O_T_EE[0,13],Z_meas[0],1.0])
    base_proj = P@base_XYZ
    ax.scatter(base_proj[0]/base_proj[2],base_proj[1]/base_proj[2],s=2,c='yellow',zorder=2.5)
    # Planar axes
    ax.plot([EE_proj[0]/EE_proj[2],EE_X_axis_proj[0]/EE_X_axis_proj[2]],[EE_proj[1]/EE_proj[2],EE_X_axis_proj[1]/EE_X_axis_proj[2]],'r')
    ax.plot([EE_proj[0]/EE_proj[2],EE_Z_axis_proj[0]/EE_Z_axis_proj[2]],[EE_proj[1]/EE_proj[2],EE_Z_axis_proj[1]/EE_Z_axis_proj[2]],'b')
    ax.legend(['Cable base (X/Z_meas after offsets)','FT X axis', 'FT Z axis'])
    fig.suptitle('Robot data projected to camera image frame')
    fig.tight_layout()
    fig.subplots_adjust(top=1.15)

def plot_on_image(idx):
    img = cv2.cvtColor(cv2.imread(img_dir + '/' + Img[idx] + '.jpg'), cv2.COLOR_BGR2RGB)
    _, ax = plt.subplots()
    ax.imshow(img,alpha=0.5)
    # Base and marker positions
    base_XYZ = np.array([X[idx],Y_meas,Z[idx],1.0])
    base_proj = P@base_XYZ
    mid_XYZ = np.array([X_mid[idx],Y_meas,Z_mid[idx],1.0])
    mid_proj = P@mid_XYZ
    end_XYZ = np.array([X_end[idx],Y_meas,Z_end[idx],1.0])
    end_proj = P@end_XYZ
    ax.scatter(base_proj[0]/base_proj[2],base_proj[1]/base_proj[2],s=2,c='tab:red',zorder=2.5)
    ax.scatter(mid_proj[0]/mid_proj[2],mid_proj[1]/mid_proj[2],s=2,c='tab:green',zorder=2.5)
    ax.scatter(end_proj[0]/end_proj[2],end_proj[1]/end_proj[2],s=2,c='tab:blue',zorder=2.5)
    # Extracted FK
    XZ = get_FK(p_vals,[theta_extracted[idx,0],theta_extracted[idx,1],X[idx],Z[idx],Phi[idx]],eval_fk,21)
    curve_XYZ = np.vstack([XZ[:,0],Y_meas*np.ones((XZ.shape[0],)),XZ[:,1],np.ones((XZ.shape[0],))])
    FK_evals = P@curve_XYZ
    ax.plot(FK_evals[0]/FK_evals[2],FK_evals[1]/FK_evals[2],c='tab:orange' if IK_converged[idx] else 'orange')
    ax.scatter(FK_evals[0,10]/FK_evals[2,10],FK_evals[1,10]/FK_evals[2,10],s=2,c='m',zorder=2.5)
    ax.scatter(FK_evals[0,-1]/FK_evals[2,-1],FK_evals[1,-1]/FK_evals[2,-1],s=2,c='m',zorder=2.5)
    plt.show()

#%%
# Data paths
dataset_name = 'orange_short_weighted_swing'
data_date = '0801'
data_dir = os.getcwd() + '/paramID_data/' + data_date + '/' + dataset_name
if not os.path.exists(data_dir + '/videos'):
            os.makedirs(data_dir + '/videos')

print('Dataset: ' + dataset_name)
print('Date: ' + data_date)
print('Path: ' + data_dir)

#%%
# Import data
img_dir = data_dir + '/images'
# Camera intrinsic and extrinsic transforms
with np.load(data_dir + '/../TFs_adj.npz') as tfs:
    P = tfs['P']
    E_base = tfs['E_base']
    E_cam = tfs['E_cam']
    K_cam = tfs['K_cam']
# Timestamps
ts_OTEE = np.loadtxt(data_dir + '/EE_pose.csv', dtype=np.ulonglong, delimiter=',', skiprows=1, usecols=0)
ts_markers = np.loadtxt(data_dir + '/marker_positions.csv', dtype=np.ulonglong, delimiter=',', skiprows=1, usecols=0)

# ts_W = np.loadtxt(data_dir + '/EE_wrench.csv', dtype=np.ulonglong, delimiter=',', skiprows=1, usecols=0)
ts_W = np.array([ts_OTEE[0],ts_OTEE[-1]])

ts_begin = np.max([np.min(ts_OTEE), np.min(ts_markers), np.min(ts_W)])
ts_end = np.min([np.max(ts_OTEE), np.max(ts_markers), np.max(ts_W)])
cam_delay = 0.0
# Measurements
O_T_EE = np.loadtxt(data_dir + '/EE_pose.csv', delimiter=',', skiprows=1, usecols=range(1,17))
markers = np.loadtxt(data_dir + '/marker_positions.csv', delimiter=',', skiprows=1, usecols=range(1,10))

# W = np.loadtxt(data_dir + '/EE_wrench.csv', delimiter=',', skiprows=1, usecols=range(1,7))
W = np.array([[0,0,0,0,0,0],[0,0,0,0,0,0]])

# Physical definitions for object set up
print("REMEMBER TO SET THE OBJECT PROPERTIES!!!")
p_vals = [0.25, 0.23, 0.45, 0.015] # cable properties: mass (length), mass (end), length, diameter
base_offset = 0.015 # Offset distance of cable attachment point from measured robot EE frame (in EE frame)

# Copy relevant planar data
# Base position and orientation from robot state
RMat_EE = np.array([[O_T_EE[:,0], O_T_EE[:,1],O_T_EE[:,2]],
                    [O_T_EE[:,4], O_T_EE[:,5],O_T_EE[:,6]],
                    [O_T_EE[:,8], O_T_EE[:,9],O_T_EE[:,10]]]).T
# IMPORTANT : Extracting relevant Phi for the model assumes planar motion, parallel to XZ plane.
# Expectation is that the natural positon of the EE with Z pointing down and X pointing forward is reached with pi rotation around the robot X axis.
# The Phi angle of the model is then just the rotation around the robot Y axis. Doesn't work if there is also rotation around the Z axis.
RPY_EE = R.from_matrix(RMat_EE).as_euler('xzy', degrees=False) # Extrinsic Roll, Yaw, Pitch parametrisation. x=pi, z=0, y=Phi
Phi_meas = RPY_EE[:,2]
plt.plot(RPY_EE) # Worth checking that x=pi, z=0
plt.legend(['x','z','y'])
# Move robot EE position to cable attachment point. This also relies on the assumptions above.
X_meas = O_T_EE[:,12] - base_offset*np.sin(Phi_meas)
Z_meas = O_T_EE[:,14] - base_offset*np.cos(Phi_meas)
# Marker positions extracted from images
X_base_meas = markers[:,0]
Z_base_meas = markers[:,1]
X_mid_meas = markers[:,2]
Z_mid_meas = markers[:,3]
X_end_meas = markers[:,4]
Z_end_meas = markers[:,5]
# Y positions
Y_meas = np.mean(O_T_EE[:,13]) # This should be basically constant
Y_assumed = markers[0,6:9] # The [base, mid, end] Y positions used to extract 3D marker positions from the images (minor offset from nominal plane by object thickness)
# Force/torque as measured
Fx_meas = W[:,0]
Fz_meas = W[:,2]
Ty_meas = W[:,4]

#%% 
#------------------------------- Manual checks and changes - start -------------------------------#

# Check camera calibration
plot_calib_check()

#%%
# Trim dead time from beginning and end of data
fig, axs = plt.subplots(2,1)
axs[0].plot(ts_markers, X_end_meas)
axs[1].plot(ts_markers, Z_end_meas)
# axs[2].plot(ts_W, Fx_meas)
# axs[3].plot(ts_W, Fz_meas)
# axs[4].plot(ts_W, Ty_meas)
axs[-1].minorticks_on()
fig.suptitle('X_end, Z_end, Phi, Fx, Fz, Ty')

#%%
# Change these referring to plot, or skip to use full set of available data
ts_begin = 6.55e11 + 1.690213e18 
ts_end = 7.7e11 + 1.690213e18
cam_delay = 0.0 # Difference between timestamps of first movement visible in camera and robot state data. Usually 0.03s is close enough.

#%%
# Convert absolute ROS timestamps to relative seconds
t_OTEE = ts_OTEE/1e9-ts_begin/1e9
t_markers = ts_markers/1e9-ts_begin/1e9 -cam_delay # HACK - manual offset for measured camera delay
t_W = ts_W/1e9-ts_begin/1e9
t_end = ts_end/1e9-ts_begin/1e9

#%%
# Express force/torque in robot frame
Fx_sync = np.interp(t_OTEE, t_W, Fx_meas) # TODO - OK to interp to robot state times here?
Fz_sync = np.interp(t_OTEE, t_W, Fz_meas)
Ty_sync = np.interp(t_OTEE, t_W, Ty_meas)
F_sensor = np.vstack([Fx_sync, np.zeros(len(t_OTEE)), Fz_sync]).T
T_sensor = np.vstack([np.zeros(len(t_OTEE)), Ty_sync, np.zeros(len(t_OTEE))]).T
F_robot = np.einsum('ijk,ik->ij', RMat_EE, F_sensor)
T_robot = np.einsum('ijk,ik->ij', RMat_EE, T_sensor)
Fx_robot = F_robot[:,0]
Fz_robot = F_robot[:,2]
Ty_robot = T_robot[:,1] # TODO - Adjust F/T meas due to offset from FT frame?'
# TODO - this is force measured by sensor, opposite force applied to object. Maybe invert.

#-------------------------------- Manual checks and changes - end --------------------------------#

#%%
# Interpolate to uniform sample times
freq_target = 30
t_target = np.arange(0, t_end, 1/freq_target)
X = np.interp(t_target, t_OTEE, X_meas)
Z = np.interp(t_target, t_OTEE, Z_meas)
Phi = np.interp(t_target, t_OTEE, Phi_meas)
X_base = np.interp(t_target, t_markers, X_base_meas)
Z_base = np.interp(t_target, t_markers, Z_base_meas)
X_mid = np.interp(t_target, t_markers, X_mid_meas)
Z_mid = np.interp(t_target, t_markers, Z_mid_meas)
X_end = np.interp(t_target, t_markers, X_end_meas)
Z_end = np.interp(t_target, t_markers, Z_end_meas)
Fx = np.interp(t_target, t_OTEE, Fx_robot)
Fz = np.interp(t_target, t_OTEE, Fz_robot)
Ty = np.interp(t_target, t_OTEE, Ty_robot) 
# Match images to target timesteps
idxs = [(np.abs(t_markers - t_t)).argmin() for t_t in t_target] # find closest timestamp to each target timestep
Img = np.array([ts_markers[i] for i in idxs], dtype=str)

#%%
# Optionally take the object base position from the extracted marker instead of robot state
X = X_base
Z = Z_base

#%%
# Extract curvature from marker points
# Transform marker points to fixed PAC frame (subtract X/Z, rotate back phi)
fk_targets_mid = np.vstack([X_mid-X,Z_mid-Z]).T
fk_targets_end = np.vstack([X_end-X,Z_end-Z]).T
fk_targets = np.hstack([rot_XZ_on_Y(fk_targets_mid,-Phi), rot_XZ_on_Y(fk_targets_end,-Phi)])

theta_extracted = np.zeros((fk_targets.shape[0],2,))
theta_guess = np.array([-1,1e-3])
IK_converged = np.zeros((fk_targets.shape[0],1,))

# Iterate IK over data
for n in range(fk_targets.shape[0]):
    theta_n, convergence = find_curvature(p_vals, theta_guess, target_evaluators, fk_targets[n,:], 0.005)
    theta_extracted[n,:] = theta_n
    theta_guess = theta_n
    IK_converged[n] = convergence
Theta0 = theta_extracted[:,0]
Theta1 = theta_extracted[:,1]

#%%
# Derivatives
SG_window = 30
SG_order = 10
dX = signal.savgol_filter(X,SG_window,SG_order,deriv=1,delta=1/freq_target,mode='nearest')
ddX = signal.savgol_filter(X,SG_window,SG_order,deriv=2,delta=1/freq_target,mode='nearest')
dZ = signal.savgol_filter(Z,SG_window,SG_order,deriv=1,delta=1/freq_target,mode='nearest')
ddZ = signal.savgol_filter(Z,SG_window,SG_order,deriv=2,delta=1/freq_target,mode='nearest')
dPhi = signal.savgol_filter(Phi,SG_window,SG_order,deriv=1,delta=1/freq_target,mode='nearest')
ddPhi = signal.savgol_filter(Phi,SG_window,SG_order,deriv=2,delta=1/freq_target,mode='nearest')
dTheta0 = signal.savgol_filter(Theta0,SG_window,SG_order,deriv=1,delta=1/freq_target,mode='nearest')
ddTheta0 = signal.savgol_filter(Theta0,SG_window,SG_order,deriv=2,delta=1/freq_target,mode='nearest')
dTheta1 = signal.savgol_filter(Theta1,SG_window,SG_order,deriv=1,delta=1/freq_target,mode='nearest')
ddTheta1 = signal.savgol_filter(Theta1,SG_window,SG_order,deriv=2,delta=1/freq_target,mode='nearest')

#%%
# Curvature extraction animation

import matplotlib
matplotlib.use("Agg")

writer = FFMpegWriter(fps=30)

fig, ax = plt.subplots()
base, = plt.plot([], [], c='k')
curve, = plt.plot([], [], zorder=2.3)
mid = plt.scatter([], [], s=2, c='tab:green',zorder=2.5)
end = plt.scatter([], [], s=2, c='tab:blue',zorder=2.5)
Fx_vis,  = plt.plot([], [], c='r')
Fz_vis,  = plt.plot([], [], c='b')
Ty_vis,  = plt.plot([], [], c='g')
Ty_vis_angs = np.arange(0,np.pi/3,np.pi/10)
plt.xlim(-(p_vals[2]+0.1), (p_vals[2]+0.1))
plt.ylim(-(p_vals[2]+0.1), 0.1)
ax.set_aspect('equal')
ax.grid(True)

draw_FT = False
in_robot_frame = True
post = '_robot' if in_robot_frame else ''

with writer.saving(fig, data_dir + '/videos/curvature_anim' + post + '.mp4', 200):
    for i in range(fk_targets.shape[0]):
        if i % freq_target == 0:
            print('Generating animation, ' + str(i/freq_target) + ' of ' + str(t_end) + 's')
        # Draw base
        if in_robot_frame:
            base_vis = rot_XZ_on_Y(np.array([0,0.05]),Phi[i])
            base.set_data([X[i],X[i]+base_vis[0]], [Z[i],Z[i]+base_vis[1]])
        # Draw FK curve
        if in_robot_frame:
            XZ = get_FK(p_vals,[theta_extracted[i,0],theta_extracted[i,1],X[i],Z[i],Phi[i]],eval_fk,21)
        else:
            XZ = get_FK(p_vals,[theta_extracted[i,0],theta_extracted[i,1],0,0,0],eval_fk,21)
        curve.set_color('tab:orange' if IK_converged[i,0] else 'orange')
        curve.set_data(XZ[:,0], XZ[:,1])
        # Draw marker positions
        if in_robot_frame:
            mid.set_offsets([X_mid[i],Z_mid[i]])
            end.set_offsets([X_end[i],Z_end[i]])
        else:
            mid.set_offsets([fk_targets[i,0],fk_targets[i,1]])
            end.set_offsets([fk_targets[i,2],fk_targets[i,3]])
        # Draw Forces/Moments
        if draw_FT:
            if in_robot_frame:
                Fx_vis.set_data([X[i], X[i]+0.1*Fx[i]], [Z[i], Z[i]])
                Fz_vis.set_data([X[i], X[i]], [Z[i], Z[i]+0.1*Fz[i]])
                # Draw arc from X-axis illustrating torque sense - CW +ve
                Ty_vis.set_data(X[i]+np.abs(Ty[i])*np.cos(Ty_vis_angs),Z[i]-Ty[i]*np.sin(Ty_vis_angs))
            else:
                Fx_vis.set_data([0, 0.1*Fx[i]],[0, 0])
                Fz_vis.set_data([0, 0],[0, 0.1*Fz[i]])
                Ty_vis.set_data(np.abs(Ty[i])*np.cos(Ty_vis_angs),-Ty[i]*np.sin(Ty_vis_angs))
        # Centre plot
        if in_robot_frame:
            plt.xlim(XZ[0,0]-(p_vals[2]+0.1), XZ[0,0]+(p_vals[2]+0.1))
            plt.ylim(XZ[0,1]-(p_vals[2]+0.1), XZ[0,1]+0.1)

        writer.grab_frame()

    print("Finished")
    plt.close(fig)

matplotlib.use('module://matplotlib_inline.backend_inline') # TODO -figure out how to actually switch back to inline plotting

#%%
# Animation over camera image

import matplotlib
matplotlib.use("Agg")

writer = FFMpegWriter(fps=30)

fig, ax = plt.subplots()
image = plt.imshow(np.zeros((cv2.imread(img_dir + '/' + Img[0] + '.jpg').shape[0],cv2.imread(img_dir + '/' + Img[0] + '.jpg').shape[1],3)), alpha=0.5)
base = plt.scatter([], [], s=2, c='tab:red',marker='+',zorder=2.5)
mid = plt.scatter([], [], s=2, c='tab:green',zorder=2.5)
end = plt.scatter([], [], s=2, c='tab:blue',zorder=2.5)
curve, = plt.plot([], [])

delay = ('_delay' + str(int(cam_delay*1e3))) if cam_delay > 0.0 else ''

with writer.saving(fig, data_dir + '/videos/overlay_anim' + delay + '.mp4', 200):
    for idx in range(fk_targets.shape[0]):
        if idx % freq_target == 0:
            print('Generating animation, ' + str(idx/freq_target) + ' of ' + str(t_end) + 's')

        img_name = '/' + Img[idx] + '.jpg'
        img = cv2.cvtColor(cv2.imread(img_dir + img_name), cv2.COLOR_BGR2RGB)
        image.set_data(img)
        # Base and marker positions
        base_XYZ = np.array([X[idx],Y_meas,Z[idx],1.0])
        base_proj = P@base_XYZ
        mid_XYZ = np.array([X_mid[idx],Y_meas,Z_mid[idx],1.0])
        mid_proj = P@mid_XYZ
        end_XYZ = np.array([X_end[idx],Y_meas,Z_end[idx],1.0])
        end_proj = P@end_XYZ
        base.set_offsets([base_proj[0]/base_proj[2],base_proj[1]/base_proj[2]])
        mid.set_offsets([mid_proj[0]/mid_proj[2],mid_proj[1]/mid_proj[2]])
        end.set_offsets([end_proj[0]/end_proj[2],end_proj[1]/end_proj[2]])
        # Extracted FK
        XZ = get_FK(p_vals,[Theta0[idx],Theta1[idx],X[idx],Z[idx],Phi[idx]],eval_fk,21)
        curve_XYZ = np.vstack([XZ[:,0],Y_meas*np.ones((XZ.shape[0],)),XZ[:,1],np.ones((XZ.shape[0],))])
        FK_evals = P@curve_XYZ
        curve.set_color('tab:orange')# if IK_converged[idx,0] else 'orange')
        curve.set_data(FK_evals[0]/FK_evals[2],FK_evals[1]/FK_evals[2])

        writer.grab_frame()

    print("Finished")
    plt.close(fig)

matplotlib.use('module://matplotlib_inline.backend_inline')

#%%
# Save data
np.savez(data_dir + '/processed', p_vals=p_vals, t=t_target, 
         X=X, Z=Z, Y=Y_meas, Phi=Phi, Theta0=Theta0, Theta1=Theta1, 
         dX=dX,  dZ=dZ, dPhi=dPhi, dTheta0=dTheta0, dTheta1=dTheta1,
         ddX=ddX, ddZ=ddZ, ddPhi=ddPhi, ddTheta0=ddTheta0, ddTheta1=ddTheta1,
         Fx=Fx, Fz=Fz, Ty=Ty)
# Export to csv for matlab
if not os.path.exists(data_dir + '/data_out'):
            os.makedirs(data_dir + '/data_out')
with open(data_dir + '/data_out/theta_evolution.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerow(['ts', 'Theta0', 'Theta1', 'dTheta0', 'dTheta1', 'ddTheta0', 'ddTheta1'])
    for n in range(len(t_target)):
        writer.writerow([t_target[n], 
                         Theta0[n], Theta1[n],
                         dTheta0[n], dTheta1[n],
                         ddTheta0[n], ddTheta1[n]])
with open(data_dir + '/data_out/state_evolution.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerow(['ts', 'X', 'Z', 'Phi', 'Theta0', 'Theta1', 'dX', 'dZ', 'dPhi', 'dTheta0', 'dTheta1', 'ddX', 'ddZ', 'ddPhi', 'ddTheta0', 'ddTheta1'])
    for n in range(len(t_target)):
        writer.writerow([t_target[n], 
                            X[n], Z[n], Phi[n], Theta0[n], Theta1[n], 
                            dX[n], dZ[n], dPhi[n], dTheta0[n], dTheta1[n], 
                            ddX[n], ddZ[n], ddPhi[n], ddTheta0[n], ddTheta1[n]])
# # Only relevant for quasi static experiments
# with open(data_dir + '/data_out/theta_equilibria.csv', 'w', newline='') as csvfile:
#     writer = csv.writer(csvfile, delimiter=',')
#     writer.writerow(['Gamma', 'Theta0', 'Theta1', 'X_mid', 'Z_mid', 'X_end', 'Z_end'])
#     for n in range(len(t_target)):
#         writer.writerow([Phi[n], Theta0[n], Theta1[n], 
#                         fk_targets[n,0], fk_targets[n,1], 
#                         fk_targets[n,2], fk_targets[n,3]])

        
#%% Load matlab data
sim_data = np.loadtxt(data_dir + '/data_in/orange_weighted_swing_sim_B2.csv', dtype=np.float64, delimiter=',')
t_sim = sim_data[:,0]
Theta0_sim = sim_data[:,1]
Theta1_sim = sim_data[:,2]

#%%
# Animation over camera image (Using theta only)

import matplotlib
matplotlib.use("Agg")

writer = FFMpegWriter(fps=freq_target)

fig, ax = plt.subplots()
image = plt.imshow(np.zeros((cv2.imread(img_dir + '/' + Img[0] + '.jpg').shape[0],cv2.imread(img_dir + '/' + Img[0] + '.jpg').shape[1],3)), alpha=0.5)
base = plt.scatter([], [], s=2, c='tab:red',marker='+',zorder=2.5)
mid = plt.scatter([], [], s=2, c='tab:green',zorder=2.5)
end = plt.scatter([], [], s=2, c='tab:blue',zorder=2.5)
curve, = plt.plot([], [])

with writer.saving(fig, data_dir + '/videos/sim_overlay_B2' + '.mp4', 200):
    for idx in range(t_sim.shape[0]):
        if idx % freq_target == 0:
            print('Generating animation, ' + str(idx/freq_target) + ' of ' + str(t_sim.shape[0]/freq_target) + 's')

        data_idx = np.min([idx, len(Img)-1])
        img_name = '/' + Img[data_idx] + '.jpg'
        img = cv2.cvtColor(cv2.imread(img_dir + img_name), cv2.COLOR_BGR2RGB)
        image.set_data(img)
        XZ = get_FK(p_vals,[Theta0_sim[idx],Theta1_sim[idx],X[data_idx],Z[data_idx],Phi[data_idx]],eval_fk,21)
        curve_XYZ = np.vstack([XZ[:,0],Y_meas*np.ones((XZ.shape[0],)),XZ[:,1],np.ones((XZ.shape[0],))])
        FK_evals = P@curve_XYZ
        curve.set_color('tab:orange')
        curve.set_data(FK_evals[0]/FK_evals[2],FK_evals[1]/FK_evals[2])

        writer.grab_frame()

    print("Finished")
    plt.close(fig)

matplotlib.use('module://matplotlib_inline.backend_inline')
# %%
