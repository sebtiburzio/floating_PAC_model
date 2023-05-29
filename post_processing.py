#!/usr/bin/env python
#%%
import os
import pickle
import cv2
import numpy as np
import mpmath as mp
import sympy as sm
from scipy import signal
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter

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

def plot_FK(q_repl):
    FK_evals = get_FK(q_repl)
    fig, ax = plt.subplots()
    ax.plot(FK_evals[:,0],FK_evals[:,1],'tab:orange')
    ax.scatter(FK_evals[10,0],FK_evals[10,1],s=2,c='m',zorder=2.5)
    ax.scatter(FK_evals[-1,0],FK_evals[-1,1],s=2,c='m',zorder=2.5)
    plt.xlim(FK_evals[0,0]-0.8,FK_evals[0,0]+0.8)
    plt.ylim(FK_evals[0,1]-0.8,FK_evals[0,1]+0.2)
    fig.set_figwidth(8)
    ax.set_aspect('equal','box')
    ax.grid(True)
    plt.show()

def plot_calib_check():
    img = cv2.cvtColor(cv2.imread(img_dir + '/' + str(ts_markers[0]) + '.jpg'), cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots()
    ax.imshow(img,alpha=0.5)
    # End effector measured by robot
    EE_XYZ = np.array([O_T_EE[0,12],0.0,O_T_EE[0,14],1.0])
    EE_X_axis = np.array([O_T_EE[0,12]+0.05,0.0,O_T_EE[0,14],1.0]) # HACK - only if Phi_meas[0] = 0
    EE_Z_axis = np.array([O_T_EE[0,12],0.0,O_T_EE[0,14]-0.05,1.0]) #
    EE_proj = P@EE_XYZ
    EE_X_axis_proj = P@EE_X_axis
    EE_Z_axis_proj = P@EE_Z_axis
    # Base of cable incorporating offset from EE
    base_XYZ = np.array([X_meas[0],0.0,Z_meas[0],1.0])
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
    base_XYZ = np.array([X[idx],0.0,Z[idx],1.0])
    base_proj = P@base_XYZ
    mid_XYZ = np.array([X_mid[idx],0.0,Z_mid[idx],1.0])
    mid_proj = P@mid_XYZ
    end_XYZ = np.array([X_end[idx],0.0,Z_end[idx],1.0])
    end_proj = P@end_XYZ
    ax.scatter(base_proj[0]/base_proj[2],base_proj[1]/base_proj[2],s=2,c='tab:red',zorder=2.5)
    ax.scatter(mid_proj[0]/mid_proj[2],mid_proj[1]/mid_proj[2],s=2,c='tab:green',zorder=2.5)
    ax.scatter(end_proj[0]/end_proj[2],end_proj[1]/end_proj[2],s=2,c='tab:blue',zorder=2.5)
    # Extracted FK
    XZ = get_FK([theta_extracted[idx,0],theta_extracted[idx,1],X[idx],Z[idx],Phi[idx]],21)
    curve_XYZ = np.vstack([XZ[:,0],np.zeros((XZ.shape[0],)),XZ[:,1],np.ones((XZ.shape[0],))])
    FK_evals = P@curve_XYZ
    ax.plot(FK_evals[0]/FK_evals[2],FK_evals[1]/FK_evals[2],c='tab:orange' if IK_converged[idx] else 'orange')
    ax.scatter(FK_evals[0,10]/FK_evals[2,10],FK_evals[1,10]/FK_evals[2,10],s=2,c='m',zorder=2.5)
    ax.scatter(FK_evals[0,-1]/FK_evals[2,-1],FK_evals[1,-1]/FK_evals[2,-1],s=2,c='m',zorder=2.5)
    plt.show()

# Rotate 2D vectors on XY plane around robot +Y axis
def rot_XZ_on_Y(XZ,angle):
    R_angle = np.array([[np.cos(-angle), np.sin(-angle)], 
                        [-np.sin(-angle), np.cos(-angle)]]).T
    if len(XZ.shape) == 1:
        return R_angle@XZ
    else:
        return np.einsum('ijk,ik->ij', R_angle, XZ)

def get_FK(q_repl,num_pts=21):
    s_evals = np.linspace(0,1,num_pts)
    FK_evals = np.zeros((s_evals.size,2,1))
    for i_s in range(s_evals.size):
       FK_evals[i_s] = np.array(eval_fk(q_repl,p_vals,s_evals[i_s],0.0))
    return FK_evals.squeeze()

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
                return theta_guess, False
            else:
                J = np.vstack([eval_J_midpt(theta_guess, p_vals), eval_J_endpt(theta_guess, p_vals)])
                theta_guess = theta_guess - (np.linalg.pinv(J)@error).squeeze()
                error_2norm_last = error_2norm
    print("Max iterations reached (check why)")
    return theta_guess, False


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
# Data paths
dataset_name = 'sine_x_30FPS'
data_date = '0508_tripod'
data_dir = os.getcwd() + '/paramID_data/' + data_date + '/' + dataset_name  # TODO different in image_processing (extra '/' on end), maybe make same?

print('Dataset: ' + dataset_name)
print('Date: ' + data_date)
print('Path: ' + data_dir)

#%%
# Camera intrinsic and extrinsic transforms - copied from image_processing - TODO - make better (in both)
# K matrix from saved camera info
# HACK - extracting values manually from calibration files... mega hacky but... it'll do? Until anything at all changes and breaks it
# Probably save it as a nicer numpy format when doing bagfile extraction
K_line = np.loadtxt(data_dir + '/color_camera_info.txt', dtype='str', delimiter=',', skiprows=10, max_rows=1)
K_cam = np.array([[float(K_line[0][4:]), 0.0, float(K_line[2])], 
                  [0.0, float(K_line[4]), float(K_line[5])], 
                  [0.0, 0.0, 1.0]])

R_line = np.loadtxt(data_dir + '/../calib_' + data_date + '.launch', dtype='str', delimiter=' ', skiprows=5, max_rows=1)
R_cam = R.from_quat([float(R_line[11]), float(R_line[12]), float(R_line[13]), float(R_line[14])]).as_matrix() # cam to base frame
T_cam = np.array([[float(R_line[6][6:])],[float(R_line[7])],[float(R_line[8])]])
                 
E_base = np.hstack([R_cam, T_cam]) # cam to base frame
E_cam = np.hstack([R_cam.T, -R_cam.T@T_cam]) # base to cam frame
P = K_cam@E_cam

#%%
# Import data
img_dir = data_dir + '/images'
# Timestamps
ts_OTEE = np.loadtxt(data_dir + '/EE_pose.csv', dtype=np.ulonglong, delimiter=',', skiprows=1, usecols=0)
ts_markers = np.loadtxt(data_dir + '/marker_positions.csv', dtype=np.ulonglong, delimiter=',', skiprows=1, usecols=0)
ts_W = np.loadtxt(data_dir + '/EE_wrench.csv', dtype=np.ulonglong, delimiter=',', skiprows=1, usecols=0)
ts_begin = np.max([np.min(ts_OTEE), np.min(ts_markers), np.min(ts_W)])
ts_end = np.min([np.max(ts_OTEE), np.max(ts_markers), np.max(ts_W)])
cam_delay = 0.0
# Measurements
O_T_EE = np.loadtxt(data_dir + '/EE_pose.csv', delimiter=',', skiprows=1, usecols=range(1,17))
markers = np.loadtxt(data_dir + '/marker_positions.csv', delimiter=',', skiprows=1, usecols=range(1,7))
W = np.loadtxt(data_dir + '/EE_wrench.csv', delimiter=',', skiprows=1, usecols=range(1,7))

# Physical definitions for object set up
p_vals = [0.4, 0.23, 0.742, 0.015] # cable properties: mass (length), mass (end), length, radius
base_offset = -0.0085 # Z-dir offset of cable attachment point from measured robot EE frame

# Copy relevant planar data
# Base coordinates
RMat_EE = np.array([[O_T_EE[:,0], O_T_EE[:,1],O_T_EE[:,2]],
                    [O_T_EE[:,4], O_T_EE[:,5],O_T_EE[:,6]],
                    [O_T_EE[:,8], O_T_EE[:,9],O_T_EE[:,10]]]).T
RPY_EE = R.from_matrix(RMat_EE).as_euler('xyz', degrees=False)
Phi_meas = RPY_EE[:,1] # HACK - lose information about Pi rotation around X axis that results in inverted Y/Z axes
X_meas = O_T_EE[:,12] - base_offset*np.sin(Phi_meas) # Move robot EE position to cable attachment point
Z_meas = O_T_EE[:,14] + base_offset*np.cos(Phi_meas)
X_base_meas = markers[:,0]
Z_base_meas = markers[:,1]
X_mid_meas = markers[:,2]
Z_mid_meas = markers[:,3]
X_end_meas = markers[:,4]
Z_end_meas = markers[:,5]
# Force/torque as measured
Fx_meas = W[:,0]
Fz_meas = W[:,2]
Ty_meas = W[:,4]

#%% 
#------------------------------- Manual checks and changes - start -------------------------------#

# Check camera calibration
plot_calib_check()

#%%
# !!! The fudge zone !!!
# Static offsets to correct for camera calibration error
X_meas = X_meas - 0.008
plot_calib_check()
# !!! Leaving the fudge zone !!!

#%%
# Trim dead time from beginning and end of data
fig, axs = plt.subplots(5,1)
axs[0].plot(ts_markers, X_end_meas)
axs[1].plot(ts_markers, Z_end_meas)
axs[2].plot(ts_W, Fx_meas)
axs[3].plot(ts_W, Fz_meas)
axs[4].plot(ts_W, Ty_meas)
axs[-1].minorticks_on()
fig.suptitle('X_end, Z_end, Phi, Fx, Fz, Ty')

#%%
# Change these referring to plot, or skip to use full set of available data
ts_begin = 5.7e10 + 1.6835609e18 
ts_end = 7.0e10 + 1.6835609e18
cam_delay = 0.03 # Difference between timestamps of first movement visible in camera and robot state data

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
Fz_robot = F_robot[:,2] - (p_vals[0] + p_vals[1])*9.81 # Add weight of cable. TODO - include in data, remove here.
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
# Extract curvature from marker points
# Transform marker points to fixed PAC frame (subtract X/Z, rotate back phi)
fk_targets_mid = np.vstack([X_mid-X,Z_mid-Z]).T
fk_targets_end = np.vstack([X_end-X,Z_end-Z]).T
fk_targets = np.hstack([rot_XZ_on_Y(fk_targets_mid,-Phi), rot_XZ_on_Y(fk_targets_end,-Phi)])

theta_extracted = np.zeros((fk_targets.shape[0],2,))
theta_guess = np.array([1e-3, 1e-3])
IK_converged = np.zeros((fk_targets.shape[0],1,))

# Iterate IK over data
for n in range(fk_targets.shape[0]):
    theta_n, convergence = find_curvature(theta_guess, fk_targets[n,:])
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
# Save data
np.savez(data_dir + '/processed', p_vals=p_vals, t=t_target, 
         X=X, Z=Z, Phi=Phi, Theta0=Theta0, Theta1=Theta1, 
         dX=dX,  dZ=dZ, dPhi=dPhi, dTheta0=dTheta0, dTheta1=dTheta1,
         ddX=ddX, ddZ=ddZ, ddPhi=ddPhi, ddTheta0=ddTheta0, ddTheta1=ddTheta1,
         Fx=Fx, Fz=Fz, Ty=Ty)

#%%
# Curvature extraction animation
import matplotlib
matplotlib.use("Agg")

writer = FFMpegWriter(fps=freq_target)

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

draw_FT = True
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
            XZ = get_FK([theta_extracted[i,0],theta_extracted[i,1],X[i],Z[i],Phi[i]],21)
        else:
            XZ = get_FK([theta_extracted[i,0],theta_extracted[i,1],0,0,0],21)
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

writer = FFMpegWriter(fps=freq_target)

fig, ax = plt.subplots()
image = plt.imshow(np.zeros((720,1280,3)), alpha=0.5) # TODO - get img resolution?
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
        base_XYZ = np.array([X[idx],0.0,Z[idx],1.0])
        base_proj = P@base_XYZ
        mid_XYZ = np.array([X_mid[idx],0.0,Z_mid[idx],1.0])
        mid_proj = P@mid_XYZ
        end_XYZ = np.array([X_end[idx],0.0,Z_end[idx],1.0])
        end_proj = P@end_XYZ
        base.set_offsets([base_proj[0]/base_proj[2],base_proj[1]/base_proj[2]])
        mid.set_offsets([mid_proj[0]/mid_proj[2],mid_proj[1]/mid_proj[2]])
        end.set_offsets([end_proj[0]/end_proj[2],end_proj[1]/end_proj[2]])
        # Extracted FK
        XZ = get_FK([theta_extracted[idx,0],theta_extracted[idx,1],X[idx],Z[idx],Phi[idx]],21)
        curve_XYZ = np.vstack([XZ[:,0],np.zeros((XZ.shape[0],)),XZ[:,1],np.ones((XZ.shape[0],))])
        FK_evals = P@curve_XYZ
        curve.set_color('tab:orange' if IK_converged[idx,0] else 'orange')
        curve.set_data(FK_evals[0]/FK_evals[2],FK_evals[1]/FK_evals[2])

        writer.grab_frame()

    print("Finished")
    plt.close(fig)

matplotlib.use('module://matplotlib_inline.backend_inline')

# %%
