#!/usr/bin/env python
#%%
import sys
import os
import cv2
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.ndimage import center_of_mass
from scipy.spatial.transform import Rotation as R

#%%
# Shows image in pop up without crashing jupyter
def showim(img): 
    res = cv2.resize(img, None, fx=0.75, fy=0.75, interpolation = cv2.INTER_CUBIC)
    cv2.imshow('image',res)
    k = cv2.waitKey(0) & 0xFF
    if k == 27:         # wait for ESC key to exit
        cv2.destroyAllWindows()

def plotim(idx):
    img = cv2.cvtColor(cv2.imread(img_dir + imgs[idx]), cv2.COLOR_BGR2RGB)
    fig = plt.figure(figsize=(14, 12))
    ax = fig.add_subplot(autoscale_on=False)
    ax.set_aspect('equal')
    ax.grid()
    ax.set_xlim(0,img.shape[1])
    ax.set_ylim(0,img.shape[0])
    ax.imshow(img)
    ax.scatter(EE_start[0],EE_start[1],s=5,c='y')

def plot_markers(idx, plot_mask=True, save=False):
    img_name = os.listdir(img_dir)[idx]
    img = cv2.cvtColor(cv2.imread(img_dir + img_name), cv2.COLOR_BGR2RGB)

    fig = plt.figure(figsize=(7.5, 6))
    ax = fig.add_subplot(autoscale_on=False)
    ax.set_aspect('equal')
    ax.grid()
    ax.set_xlim(0,img.shape[1])
    ax.set_ylim(0,img.shape[0])
    marker_colors = ['lightcoral','palegreen','lightskyblue']

    if plot_mask:
        marker_colors = ['orangered','forestgreen','dodgerblue']
        ax.plot(base_mask_pixels[idx][1],base_mask_pixels[idx][0],',',c='lightcoral')
        ax.plot(mid_mask_pixels[idx][1],mid_mask_pixels[idx][0],',',c='palegreen')
        ax.plot(end_mask_pixels[idx][1],end_mask_pixels[idx][0],',',c='lightskyblue')

    ax.imshow(np.asarray(img)) # TODO - way to make this not display when saving?
    ax.plot(base_positions_px[idx,1],base_positions_px[idx,0],ms=2,fillstyle='none',marker='o',mec=marker_colors[0])
    ax.plot(mid_positions_px[idx,1],mid_positions_px[idx,0],ms=2,fillstyle='none',marker='o',mec=marker_colors[1])
    ax.plot(end_positions_px[idx,1],end_positions_px[idx,0],ms=2,fillstyle='none',marker='o',mec=marker_colors[2])
    
    if save:
        if not os.path.exists(img_dir + 'detections'):
            os.makedirs(img_dir + 'detections')
        plt.savefig(img_dir + '/detections/' + img_name[:-4] + '_d.png', bbox_inches='tight', pad_inches=0.1)

# Pixel to 3D conversions
def UV_to_XZplane(u,v,Y=0):
    rhs1 = np.hstack([P[:,:3],np.array([[-u,-v,-1]]).T])
    rhs1 = np.vstack([rhs1, np.array([0,1,0,0])])   # Intersect y=Y plane
    rhs2 = np.reshape(np.hstack([-P[:,3],[Y]]),(4,1))
    sol = np.linalg.inv(rhs1)@rhs2
    return sol[:3]

# # Todo - troubleshoot/fix depth & clean up, or remove
# def UV_to_XYZ(u,v):
#     rhs1 = np.hstack([K_cam,np.array([[-u,-v,-1]]).T])
#     rhs1 = np.vstack([rhs1, np.array([0,0,1,0])])   # Intersect Z=depth plane
#     rhs2 = np.array([[0],[0,],[0],[dimg[v,u]/1000]])
#     sol = np.linalg.inv(rhs1)@rhs2
#     sol = E_base@sol
#     return sol[:3]

#%%
# Paths - TODO check if this works at all
if len(sys.argv) > 1: # arg is dataset name, assume running from dated data folder
    dataset_name = str(sys.argv[1])
    data_dir = os.getcwd() + '/' + dataset_name + '/'
    data_date = os.path.basename(os.getcwd())
else: # assume running from dataset folder, inside dated data folder
    data_dir = os.getcwd()
    dataset_name = os.path.basename(data_dir)
    data_date = os.path.basename(os.path.dirname(data_dir))
    
print('Dataset: ' + dataset_name)
print('Date: ' + data_date)
print('Path: ' + data_dir)

#%%
# Manual paths for interactive mode
dataset_name = 'rot_link6_orange'
data_date = '0605'
data_dir = os.getcwd() + '/paramID_data/' + data_date + '/' + dataset_name + '/'

print('Dataset: ' + dataset_name)
print('Date: ' + data_date)
print('Path: ' + data_dir)

#%%
# Camera intrinsic and extrinsic transforms
with np.load(data_dir + 'TFs.npz') as tfs:
    P = tfs['P']
    E_base = tfs['E_base']
    E_cam = tfs['E_cam']
    K_cam = tfs['K_cam']

#%%
# Get img list and display first with grid
img_dir = data_dir + 'images/'
imgs = os.listdir(img_dir)
imgs.sort()
# Plot initial EE location to check camera calibration
EE_start = P@np.loadtxt(data_dir + '/EE_pose.csv', delimiter=',', skiprows=1, max_rows=1, usecols=range(13,17))
EE_start = EE_start/EE_start[2]
plotim(0)

#%%
# Process each image in folder

base_positions_px = []
mid_positions_px = []
end_positions_px = []
base_positions_XZplane = []
mid_positions_XZplane = []
end_positions_XZplane = []
base_mask_pixels = []
mid_mask_pixels = []
end_mask_pixels = []
marker_ts = []

# For depth
# mid_positions_XYZ = []
# end_positions_XYZ = []
# dimg_list = os.listdir(data_dir + 'depth/')
# didx = 0

# Mask range for markers
lower_R = np.array([0,80,50])
upper_R = np.array([10,255,255])
lower_G = np.array([20,80,20])
upper_G = np.array([80,255,255])
lower_B = np.array([90,100,20])
upper_B = np.array([130,255,255])

for img_name in imgs:
    img = cv2.imread(img_dir + img_name)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    mask_R = cv2.inRange(hsv, lower_R, upper_R)
    mask_G = cv2.inRange(hsv, lower_G, upper_G)
    mask_B = cv2.inRange(hsv, lower_B, upper_B)

    # Crop above base (change depending on img orientation)
    mask_R[:,:200] = 0
    mask_R[:,350:] = 0
    mask_G[:,:500] = 0
    mask_G[:,1000:] = 0
    mask_B[:,:900] = 0
    # Crop trouble areas creating some spurious readings - TODO proper solution, outlier resistant
    mask_R[:200,:] = 0
    mask_G[:200,:] = 0
    mask_B[:200,:] = 0

    # Remove noise from mask
    mask_R = cv2.morphologyEx(mask_R, cv2.MORPH_OPEN, np.ones((5,5)))
    mask_R = cv2.morphologyEx(mask_R, cv2.MORPH_CLOSE, np.ones((5,5)))
    mask_G = cv2.morphologyEx(mask_G, cv2.MORPH_OPEN, np.ones((5,5)))
    mask_G = cv2.morphologyEx(mask_G, cv2.MORPH_CLOSE, np.ones((5,5)))
    mask_B = cv2.morphologyEx(mask_B, cv2.MORPH_OPEN, np.ones((5,5)))
    mask_B = cv2.morphologyEx(mask_B, cv2.MORPH_CLOSE, np.ones((5,5)))
    # Remove noise from mask TODO - is this here twice on purpose?
    mask_R = cv2.morphologyEx(mask_R, cv2.MORPH_OPEN, np.ones((5,5)))
    mask_R = cv2.morphologyEx(mask_R, cv2.MORPH_CLOSE, np.ones((5,5)))
    mask_G = cv2.morphologyEx(mask_G, cv2.MORPH_OPEN, np.ones((5,5)))
    mask_G = cv2.morphologyEx(mask_G, cv2.MORPH_CLOSE, np.ones((5,5)))
    mask_B = cv2.morphologyEx(mask_B, cv2.MORPH_OPEN, np.ones((5,5)))
    mask_B = cv2.morphologyEx(mask_B, cv2.MORPH_CLOSE, np.ones((5,5)))

    # Mask non-marker pixels
    masked_R = cv2.bitwise_and(img,img,mask=mask_R)
    masked_G = cv2.bitwise_and(img,img,mask=mask_G)
    masked_B = cv2.bitwise_and(img,img,mask=mask_B)

    # Locate markers at COM of mask
    base_pos_px = np.round(center_of_mass(mask_R)).astype(int)
    mid_pos_px = np.round(center_of_mass(mask_G)).astype(int)
    end_pos_px = np.round(center_of_mass(mask_B)).astype(int)
    # Convert px location to world frame
    base_pos_XZplane = UV_to_XZplane(base_pos_px[1],base_pos_px[0],Y=0.05) # TODO - measure actual R marker distance from plane
    mid_pos_XZplane = UV_to_XZplane(mid_pos_px[1],mid_pos_px[0],Y=0) # TODO - offset cable thickness?
    end_pos_XZplane = UV_to_XZplane(end_pos_px[1],end_pos_px[0],Y=0)

    # # Convert px location to world frame, using depth measurement
    # dimg = cv2.imread(data_dir + 'depth/' + dimg_list[didx], cv2.IMREAD_UNCHANGED)
    # mid_pos_XYZ = UV_to_XYZ(mid_pos_px[1],mid_pos_px[0])
    # end_pos_XYZ = UV_to_XYZ(end_pos_px[1],end_pos_px[0])
    # mid_positions_XYZ.append(mid_pos_XYZ)
    # end_positions_XYZ.append(end_pos_XYZ)
    # didx += 1

    base_positions_px.append(base_pos_px)
    mid_positions_px.append(mid_pos_px)
    end_positions_px.append(end_pos_px)
    base_positions_XZplane.append(base_pos_XZplane)
    mid_positions_XZplane.append(mid_pos_XZplane)
    end_positions_XZplane.append(end_pos_XZplane)
    base_mask_pixels.append(np.where(masked_R[:,:,0] > 0))
    mid_mask_pixels.append(np.where(masked_G[:,:,0] > 0))
    end_mask_pixels.append(np.where(masked_B[:,:,0] > 0))
    marker_ts.append(img_name[:-4])

base_positions_px = np.array(base_positions_px)
mid_positions_px = np.array(mid_positions_px)
end_positions_px = np.array(end_positions_px)
base_positions_XZplane = np.array(base_positions_XZplane)
mid_positions_XZplane = np.array(mid_positions_XZplane)
end_positions_XZplane = np.array(end_positions_XZplane)
t_markers = np.array(marker_ts, dtype=np.ulonglong)
t_markers = (t_markers - t_markers[0])/1e9

# # For depth
# mid_positions_XYZ = np.array(mid_positions_XYZ)
# end_positions_XYZ = np.array(end_positions_XYZ)

# Check for false detections
plt.plot(base_positions_px[:,0])
plt.plot(mid_positions_px[:,0])
plt.plot(end_positions_px[:,0])
plt.plot(base_positions_px[:,1])
plt.plot(mid_positions_px[:,1])
plt.plot(end_positions_px[:,1])

#%%
# Export to csv
with open(data_dir + 'marker_positions.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerow(['ts', 'base_pos_x', 'base_pos_z', 'mid_pos_x', 'mid_pos_z', 'end_pos_x', 'end_pos_z'])
    for n in range(len(marker_ts)):
        writer.writerow([marker_ts[n], 
                         float(base_positions_XZplane[n,0]), float(base_positions_XZplane[n,2]),
                         float(mid_positions_XZplane[n,0]), float(mid_positions_XZplane[n,2]), 
                         float(end_positions_XZplane[n,0]), float(end_positions_XZplane[n,2])])

#%%
# Animated plot (requires %matplotlib ipympl) TODO - crashes after a while (ipympl)
# Init plot
fig = plt.figure(figsize=(5, 4))
ax = fig.add_subplot(autoscale_on=False)
ax.set_aspect('equal')
ax.grid()
ax.set_xlim(0,1920)
ax.set_ylim(0,1080)
ax.fill
scatter = ax.scatter([],[])

def animate(frame):
    scatter.set_offsets([np.array(end_positions_px)[frame,1],np.array(end_positions_px)[frame,0]])
    ax.set_title(t_markers[frame])
    return scatter,

ani = animation.FuncAnimation(fig, animate, len(mid_positions_px), interval=1000/30, blit=True)
plt.show()

#%%
# For multiple points
# # Find centroid of markers using k-means clustering
# Y, X = np.where(res[:,:,0] > 0)
# Z = np.vstack((X,Y)).T
# Z = np.float32(Z)
# _,_,centres=cv2.kmeans(Z,5,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

# centres.sort(axis=0)
# marker_positions.append(np.hstack([centres[2,:], centres[4,:]]))
# marker_ts.append(img_name[:-4])

# plt.scatter(center[:,0],center[:,1])
# plt.xlim(0,1920)
# plt.ylim(0,1080)
# plt.show()

# Init plot
# fig = plt.figure(figsize=(5, 4))
# ax = fig.add_subplot(autoscale_on=False)
# ax.set_aspect('equal')
# ax.grid()
# ax.set_xlim(0,1920)
# ax.set_ylim(0,1080)
# #scatter = ax.scatter([np.array(marker_positions)[0,0],np.array(marker_positions)[0,2]], [np.array(marker_positions)[0,1],np.array(marker_positions)[0,3]])
# scatter = ax.scatter([],[])

# def animate(frame):
#     scatter.set_offsets([np.array(marker_positions)[frame,0],np.array(marker_positions)[frame,1]])
#     ax.set_title(t_markers[frame])
#     return scatter,

# ani = animation.FuncAnimation(fig, animate, len(marker_positions), interval=1/30, blit=True)
# plt.show()