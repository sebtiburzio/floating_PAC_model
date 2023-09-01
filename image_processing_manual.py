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
    img = cv2.cvtColor(cv2.imread(img_dir + str(imgs[idx]) + '.jpg'), cv2.COLOR_BGR2RGB)
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
        marked_UV = P@marked_XYZ
        marked_UV = marked_UV/marked_UV[2]

        ax.scatter(marked_UV[0,:],marked_UV[1,:],s=5,c='yellow',zorder=2.5)
        # Robot base links
        base_links = P@np.array([[0,0,0,1],[0,0,0.333,1]]).T #,[0,-0.15,0.333,1],[0,0.15,0.333,1] # Joint2 axis if Joint=0
        base_links = base_links/base_links[2,:].reshape(1,-1)
        ax.plot(base_links[0,:],base_links[1,:],lw=3,c='slategrey')

    if testpt is not None:
        ax.scatter(testpt[0],testpt[1],s=5,c='yellow',zorder=2.5)
    print(imgs[idx])

# Marked points for calibration visualation
marked_XYZ = np.array([
                       [0.154,0.149,0.0,1.0], # FR3 TERI base
                       [0.154,-0.150,0.0,1.0], # FR3 TERI base
                       [-0.238,0.149,0.0,1.0], # FR3 TERI base
                       [-0.238,-0.149,0.0,1.0], # FR3 TERI base
                       [0.154,0.149,-0.03,1.0], # FR3 TERI base
                       [0.154,-0.150,-0.03,1.0], # FR3 TERI base
                       [-0.238,0.149,-0.03,1.0], # FR3 TERI base
                       [-0.238,-0.149,-0.03,1.0], # FR3 TERI base
                       [0.055,0.0,0.14,1.0], # FR3 link0 arrow
                       [0.0715,0.0,0.00135,1.0], # FR3 link0 front edge
                    ]).T

# Pixel to 3D conversions
def UV_to_XZplane(u,v,Y=0):
    rhs1 = np.hstack([P[:,:3],np.array([[-u,-v,-1]]).T])
    rhs1 = np.vstack([rhs1, np.array([0,1,0,0])])   # Intersect y=Y plane
    rhs2 = np.reshape(np.hstack([-P[:,3],[Y]]),(4,1))
    sol = np.linalg.inv(rhs1)@rhs2
    return sol[:3]

#%%
# Paths 
dataset_name = 'black_unweighted_phicost'
data_date = '0829'
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
imgs = np.loadtxt(data_dir + '/sequence_experiment.csv', dtype=np.ulonglong, delimiter=',', skiprows=1, usecols=0)
# Plot initial EE location to check camera calibration
EE_start_XYZ = np.loadtxt(data_dir + '/sequence_experiment.csv', delimiter=',', skiprows=1, max_rows=1, usecols=range(1,4))
EE_start_px = P@np.hstack([EE_start_XYZ,1])
EE_start_px = EE_start_px/EE_start_px[2]
plotim(0,True)

#%%
# Set Y positions of markers
base_Y = EE_start_XYZ[1] - 0.01
mid_Y = EE_start_XYZ[1] - 0.01
end_Y = EE_start_XYZ[1] - 0.015
print("Assuming base at Y=" + str(base_Y))
print("Assuming mid at Y=" + str(mid_Y))
print("Assuming end at Y=" + str(end_Y))
print("DID YOU SET THE MARKER POSITIONS AT THE RIGHT RADII????")

#%%

import matplotlib
matplotlib.use('TkAgg')

ts = np.loadtxt(data_dir + '/sequence_experiment.csv', dtype=np.ulonglong, delimiter=',', skiprows=1, usecols=0)
data = np.loadtxt(data_dir + '/sequence_experiment.csv', delimiter=',', skiprows=1)

X_EE = data[:,1]
Y_EE = data[:,2]
Z_EE = data[:,3]
Phi = data[:,4]
Goal_X = data[:,5]
Goal_Z = data[:,6]
# Goal_Phi = data[:,7] # For endpoint orientation experiment
Endpt_Sol_X = data[:,7] # For endpoint grid experiment
Endpt_Sol_Z = data[:,8] # For endpoint grid experiment

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

for n in range(len(ts)):
    # Display current image
    fig, ax = plt.subplots(figsize=(12, 9))
    img = cv2.cvtColor(cv2.imread(img_dir + str(imgs[n]) + '.jpg'), cv2.COLOR_BGR2RGB)
    ax.imshow(img)
    # Plot current EE location
    EEpx = P@np.array([X_EE[n],Y_EE[n],Z_EE[n],1]).T
    EEpx = EEpx/EEpx[2]
    ax.scatter(EEpx[0],EEpx[1],s=5,c='red',zorder=2.5)
    # Plot current goal location
    goalpx = P@np.array([Goal_X[n],Y_EE[n],Goal_Z[n],1]).T
    goalpx = goalpx/goalpx[2]
    ax.scatter(goalpx[0],goalpx[1],marker='x',c='limegreen',zorder=2.5)
    # Plot current model endpt location
    modelpx = P@np.array([Endpt_Sol_X[n],Y_EE[n],np.max([Endpt_Sol_Z[n],0.0]),1]).T # clamp Z to 0 to use as reference if below bench
    modelpx = modelpx/modelpx[2]
    ax.scatter(modelpx[0],modelpx[1],s=5,c='yellow',zorder=2.5)
    # Plot Z=0 in XZ plane
    z0px_L = P@np.array([-1.0,Y_EE[n],0,1]).T
    z0px_L = z0px_L/z0px_L[2]
    z0px_R = P@np.array([1.0,Y_EE[n],0,1]).T
    z0px_R = z0px_R/z0px_R[2]
    plt.plot([z0px_L[0],z0px_R[0]],[z0px_L[1],z0px_R[1]],c='slategrey')
    # plt.axis('off')
    plt.xlim(-100,2020)
    plt.ylim(1080,-100)
    plt.tight_layout()
    # Get user input for marker locations
    UVs = plt.ginput(n=-1, timeout=0)
    plt.close()
    # Convert chosen pts to XZ plane
    base_UV = [int(UVs[0][0]), int(UVs[0][1])]
    mid_UV = [int(UVs[1][0]), int(UVs[1][1])]
    end_UV = [int(UVs[2][0]), int(UVs[2][1])]
    base_XZ = UV_to_XZplane(base_UV[0], base_UV[1], base_Y)
    mid_XZ = UV_to_XZplane(mid_UV[0], mid_UV[1], mid_Y)
    end_XZ = UV_to_XZplane(end_UV[0], end_UV[1], end_Y)
    # Determine base angle if applicable # TODO - it's actually the tip angle. But I already saved all the data as base angle.
    if len(UVs) > 3:
        base_ang_start_UV = [int(UVs[3][0]), int(UVs[3][1])]
        base_ang_end_UV = [int(UVs[4][0]), int(UVs[4][1])]
        base_ang_start_XZ = UV_to_XZplane(base_ang_start_UV[0], base_ang_start_UV[1], end_Y)
        base_ang_end_XZ = UV_to_XZplane(base_ang_end_UV[0], base_ang_end_UV[1], end_Y)
        base_ang = np.arctan2(-(base_ang_end_XZ[0]-base_ang_start_XZ[0]),-(base_ang_end_XZ[2]-base_ang_start_XZ[2])) # atan2(-delX,-delZ) because of robot axis shenanigans
    else:
        base_ang_start_UV = np.array([[0.0],[0.0]])
        base_ang_end_UV = np.array([[0.0],[0.0]])
        base_ang_start_XZ = np.array([[0.0],[0.0],[0.0]])
        base_ang_end_XZ = np.array([[0.0],[0.0],[0.0]])
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


with open(data_dir + '/sequence_results.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerow(['ts', 'X_EE', 'Y_EE', 'Z_EE', 'Phi', 
                        'X_base_meas', 'Z_base_meas', 'X_mid_meas', 'Z_mid_meas', 'X_end_meas', 'Z_end_meas', 
                        'U_base', 'V_base', 'U_mid', 'V_mid', 'U_end', 'V_end', 
                        'Base_angle', 
                        'X_ang_start', 'Z_ang_start', 'X_ang_end', 'Z_ang_end', 
                        'U_ang_start', 'V_ang_start', 'U_ang_end', 'V_ang_end',
                        'Goal_X', 'Goal_Z', 
                        # 'Goal_Phi',
                        'Endpt_Sol_X', 'Endpt_Sol_Z'
                    ])
    for n in range(len(ts)):
        writer.writerow([imgs[n], X_EE[n], Y_EE[n], Z_EE[n], Phi[n],
                            X_base_meas[n], Z_base_meas[n], X_mid_meas[n], Z_mid_meas[n], X_end_meas[n], Z_end_meas[n],
                            U_base[n], V_base[n], U_mid[n], V_mid[n], U_end[n], V_end[n],
                            float(Base_angle[n]), 
                            X_ang_start[n], Z_ang_start[n], X_ang_end[n], Z_ang_end[n],
                            U_ang_start[n], V_ang_start[n], U_ang_end[n], V_ang_end[n],
                            Goal_X[n], Goal_Z[n], 
                            # Goal_Phi[n],
                            Endpt_Sol_X[n], Endpt_Sol_Z[n]
                        ])
# %%
