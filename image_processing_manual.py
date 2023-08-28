#!/usr/bin/env python
#%%
import os
import cv2
import csv
import numpy as np
import matplotlib.pyplot as plt

#%%
# Shows image in pop up without crashing jupyter
def showim(img): 
    res = cv2.resize(img, None, fx=0.75, fy=0.75, interpolation = cv2.INTER_CUBIC)
    cv2.imshow('image',res)
    k = cv2.waitKey(0) & 0xFF
    if k == 27:         # wait for ESC key to exit
        cv2.destroyAllWindows()

def plotim(idx,calib_vis=False,testpt=None):
    img = cv2.cvtColor(cv2.imread(img_dir + imgs[idx]), cv2.COLOR_BGR2RGB)
    fig = plt.figure(figsize=(14, 12))
    ax = fig.add_subplot(autoscale_on=False)
    ax.set_aspect('equal')
    ax.minorticks_on()
    ax.grid(which='both')
    ax.set_xlim(0,img.shape[1])
    ax.set_ylim(0,img.shape[0])

    ax.imshow(img)
    ax.scatter(EE_start_px[0],EE_start_px[1],s=5,c='red',zorder=2.5)
    
    if calib_vis:
        # Z-plane grid
        grid_btm = P@np.array([np.linspace(-0.2,0.2,5).T,-0.2*np.ones(5),np.zeros(5),np.ones(5)])
        grid_btm = grid_btm/grid_btm[2,:].reshape(1,-1)
        grid_top = P@np.array([np.linspace(-0.2,0.2,5).T,0.2*np.ones(5),np.zeros(5),np.ones(5)])
        grid_top = grid_top/grid_top[2,:].reshape(1,-1)
        grid_left = P@np.array([-0.2*np.ones(5),np.linspace(-0.2,0.2,5).T,np.zeros(5),np.ones(5)])
        grid_left = grid_left/grid_left[2,:].reshape(1,-1)
        grid_right = P@np.array([0.2*np.ones(5),np.linspace(-0.2,0.2,5).T,np.zeros(5),np.ones(5)])
        grid_right = grid_right/grid_right[2,:].reshape(1,-1)
        for i in range(5):
            ax.plot([grid_btm[0,i],grid_top[0,i]],[grid_btm[1,i],grid_top[1,i]],lw=1,c='aqua')
            ax.plot([grid_left[0,i],grid_right[0,i]],[grid_left[1,i],grid_right[1,i]],lw=1,c='aqua')
        # Y-plane grid
        grid_btm = P@np.array([np.linspace(0,1.0,11).T,np.zeros(11),np.zeros(11),np.ones(11)])
        grid_btm = grid_btm/grid_btm[2,:].reshape(1,-1)
        grid_top = P@np.array([np.linspace(0,1.0,11),np.zeros(11),np.ones(11),np.ones(11)])
        grid_top = grid_top/grid_top[2,:].reshape(1,-1)
        grid_left = P@np.array([np.zeros(11),np.zeros(11),np.linspace(0,1.0,11),np.ones(11)])
        grid_left = grid_left/grid_left[2,:].reshape(1,-1)
        grid_right = P@np.array([np.ones(11),np.zeros(11),np.linspace(0,1.0,11),np.ones(11)])
        grid_right = grid_right/grid_right[2,:].reshape(1,-1)
        for i in range(11):
            ax.plot([grid_btm[0,i],grid_top[0,i]],[grid_btm[1,i],grid_top[1,i]],lw=1,c='lime')
            ax.plot([grid_left[0,i],grid_right[0,i]],[grid_left[1,i],grid_right[1,i]],lw=1,c='lime')
        # Robot base links
        base_links = P@np.array([[0,0,0,1],[0,0,0.333,1]]).T #,[0,-0.15,0.333,1],[0,0.15,0.333,1] # Joint2 axis if Joint=0
        base_links = base_links/base_links[2,:].reshape(1,-1)
        ax.plot(base_links[0,:],base_links[1,:],lw=3,c='slategrey')

    if testpt is not None:
        ax.scatter(testpt[0],testpt[1],s=5,c='yellow',zorder=2.5)
    print(imgs[idx])

# Pixel to 3D conversions
def UV_to_XZplane(u,v,Y=0):
    rhs1 = np.hstack([P[:,:3],np.array([[-u,-v,-1]]).T])
    rhs1 = np.vstack([rhs1, np.array([0,1,0,0])])   # Intersect y=Y plane
    rhs2 = np.reshape(np.hstack([-P[:,3],[Y]]),(4,1))
    sol = np.linalg.inv(rhs1)@rhs2
    return sol[:3]

#%%
# Paths 
dataset_name = 'black_short_weighted_swing'
data_date = '0801'
data_dir = os.getcwd() + '/paramID_data/' + data_date + '/' + dataset_name

print('Dataset: ' + dataset_name)
print('Date: ' + data_date)
print('Path: ' + data_dir)

#%%
# Camera intrinsic and extrinsic transforms
with np.load(data_dir + '/../TFs_adj.npz') as tfs:
    P = tfs['P']
    E_base = tfs['E_base']
    E_cam = tfs['E_cam']
    K_cam = tfs['K_cam']

#%%
# Get img list and display first with grid
img_dir = data_dir + '/images/'
imgs = os.listdir(img_dir)
#imgs.sort()
# Plot initial EE location to check camera calibration
EE_start_XYZ = np.loadtxt(data_dir + '/EE_pose.csv', delimiter=',', skiprows=1, max_rows=1, usecols=range(13,17))
EE_start_px = P@EE_start_XYZ
EE_start_px = EE_start_px/EE_start_px[2]
plotim(0,True)
# TODO - adjust camera calibration here?

#%%
# Set Y positions of markers
base_Y = EE_start_XYZ[1] - 0.01
mid_Y = EE_start_XYZ[1] - 0.01
end_Y = EE_start_XYZ[1] - 0.015
print("Assuming base at Y=" + str(base_Y))
print("Assuming mid at Y=" + str(mid_Y))
print("Assuming end at Y=" + str(end_Y))
#%%

data = np.loadtxt(data_dir + '/sequence_experiment.csv', delimiter=',', skiprows=1)
ts = data[:,0]

X_base_meas = []
Z_base_meas = []
X_mid_meas = []
Z_mid_meas = []
X_end_meas = []
Z_end_meas = []
U_base = []
V_base = []
U_mid = []
V_mid = []
U_end = []
V_end = []
U_ang_start = []
V_ang_start = []
U_ang_end = []
V_ang_end = []
X_ang_start = []
Z_ang_start = []
X_ang_end = []
Z_ang_end = []
Base_angle = []

for i in range(len(ts)):
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.imshow(img_dir + str(ts) + '.jpg')
    plt.axis('off')
    UVs = plt.ginput(n=-1, timeout=0)
    plt.close()
    base_UV = [int(UVs[0][0]), int(UVs[0][1])]
    mid_UV = [int(UVs[1][0]), int(UVs[1][1])]
    end_UV = [int(UVs[2][0]), int(UVs[2][1])]
    base_XZ = UV_to_XZplane(base_UV[0], base_UV[1], base_Y)
    mid_XZ = UV_to_XZplane(mid_UV[0], mid_UV[1], mid_Y)
    end_XZ = UV_to_XZplane(end_UV[0], end_UV[1], end_Y)
    if len(UVs) > 3:
        base_ang_start_UV = [int(UVs[3][0]), int(UVs[3][1])]
        base_ang_end_UV = [int(UVs[4][0]), int(UVs[4][1])]
        base_ang_start_XZ = UV_to_XZplane(base_ang_start_UV[0], base_ang_start_UV[1], end_Y)
        base_ang_end_XZ = UV_to_XZplane(base_ang_end_UV[0], base_ang_end_UV[1], end_Y)
        base_ang = np.arctan2(-(base_ang_end_XZ[0]-base_ang_start_XZ[0]),-(base_ang_end_XZ[2]-base_ang_start_XZ[2])) # atan2(-delX,-delZ) because of robot axis shenanigans
    else:
        base_ang_start_UV = [0,0]
        base_ang_end_UV = [0,0]
        base_ang_start_XZ = [0.0,0.0,0.0]
        base_ang_end_XZ = [0.0,0.0,0.0]
        base_ang = 0.0

    X_base_meas.append(base_XZ[0,0])
    Z_base_meas.append(base_XZ[2,0])
    X_mid_meas.append(mid_XZ[0,0])
    Z_mid_meas.append(mid_XZ[2,0])
    X_end_meas.append(end_XZ[0,0])
    Z_end_meas.append(end_XZ[2,0])
    U_base.append(base_UV[0])
    V_base.append(base_UV[1])
    U_mid.append(mid_UV[0])
    V_mid.append(mid_UV[1])
    U_end.append(end_UV[0])
    V_end.append(end_UV[1])
    U_ang_start.append(base_ang_start_UV[0])
    V_ang_start.append(base_ang_start_UV[1])
    U_ang_end.append(base_ang_end_UV[0])
    V_ang_end.append(base_ang_end_UV[1])
    X_ang_start.append(base_ang_start_XZ[0,0])
    Z_ang_start.append(base_ang_start_XZ[2,0])
    X_ang_end.append(base_ang_end_XZ[0,0])
    Z_ang_end.append(base_ang_end_XZ[2,0])
    Base_angle.append(base_ang)

with open('./sequence_results.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerow(['ts', 'X_EE', 'Y_EE', 'Z_EE', 'Phi', 
                        'X_base_meas', 'Z_base_meas', 'X_mid_meas', 'Z_mid_meas', 'X_end_meas', 'Z_end_meas', 
                        'U_base', 'V_base', 'U_mid', 'V_mid', 'U_end', 'V_end', 
                        'Base_angle', 
                        'X_ang_start', 'Z_ang_start', 'X_ang_end', 'Z_ang_end', 
                        'U_ang_start', 'V_ang_start', 'U_ang_end', 'V_ang_end',
                        'Goal_X', 'Goal_Z', 'Endpt_Sol_X', 'Endpt_Sol_Z'])
    for n in range(len(ts)):
        writer.writerow([ts[n], data[n,1], data[n,2], data[n,3], data[n,4],
                            X_base_meas[n], Z_base_meas[n], X_mid_meas[n], Z_mid_meas[n], X_end_meas[n], Z_end_meas[n],
                            U_base[n], V_base[n], U_mid[n], V_mid[n], U_end[n], V_end[n],
                            Base_angle[n], 
                            X_ang_start[n], Z_ang_start[n], X_ang_end[n], Z_ang_end[n],
                            U_ang_start[n], V_ang_start[n], U_ang_end[n], V_ang_end[n],
                            data[n,5], data[n,6], data[n,7], data[n,8]])