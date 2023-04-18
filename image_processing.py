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

def plot_markers(idx, plot_mask=True, save=False):
    img_name = os.listdir(img_folder)[idx]
    img = cv2.cvtColor(cv2.imread(img_folder + img_name), cv2.COLOR_BGR2RGB)

    fig = plt.figure(figsize=(7.5, 6))
    ax = fig.add_subplot(autoscale_on=False)
    ax.set_aspect('equal')
    ax.grid()
    ax.set_xlim(0,img.shape[1])
    ax.set_ylim(0,img.shape[0])
    marker_colors = ['palegreen','lightskyblue']

    if plot_mask:
        marker_colors = ['forestgreen','dodgerblue']
        ax.plot(mid_mask_pixels[idx][1],mid_mask_pixels[idx][0],',',c='palegreen')
        ax.plot(end_mask_pixels[idx][1],end_mask_pixels[idx][0],',',c='lightskyblue')

    ax.imshow(np.asarray(img)) # TODO - way to make this not display when saving?
    ax.plot(mid_positions_px[idx,1],mid_positions_px[idx,0],ms=2,fillstyle='none',marker='o',mec=marker_colors[0])
    ax.plot(end_positions_px[idx,1],end_positions_px[idx,0],ms=2,fillstyle='none',marker='o',mec=marker_colors[1])
    # ax.plot(110,460,'y,') # For checking difference between projection and image
    
    if save:
        if not os.path.exists(img_folder + 'detections'):
            os.makedirs(img_folder + 'detections')
        plt.savefig(img_folder + '/detections/' + img_name[:-4] + '_d.png', bbox_inches='tight', pad_inches=0.1)

# Pixel to 3D conversions
def UV_to_XZplane(u,v):
    rhs1 = np.hstack([P[:,:3],np.array([[-u,-v,-1]]).T])
    rhs1 = np.vstack([rhs1, np.array([0,1,0,0])])   # Intersect Y=0 plane
    rhs2 = np.reshape(np.hstack([-P[:,3],[0]]),(4,1))
    sol = np.linalg.inv(rhs1)@rhs2
    return sol[:3]

# Todo - torubleshoot/fix depth & clean up, or remove
# def UV_to_XYZ(u,v):
#     rhs1 = np.hstack([K_cam,np.array([[-u,-v,-1]]).T])
#     rhs1 = np.vstack([rhs1, np.array([0,0,1,0])])   # Intersect Z=depth plane
#     rhs2 = np.array([[0],[0,],[0],[dimg[v,u]/1000]])
#     sol = np.linalg.inv(rhs1)@rhs2
#     sol = E_base@sol
#     return sol[:3]

#%%
# Paths - TODO check if this works at all
if len(sys.argv) > 1:   # arg is dataset name, assume running from dated data folder
    dataset_name = str(sys.argv[1])
    data_dir = os.getcwd() + '/' + dataset_name + '/'
    data_date = os.path.basename(os.getcwd())
else:   # assume running from dataset folder, inside dated data folder
    data_dir = os.getcwd()
    dataset_name = os.path.basename(data_dir)
    data_date = os.path.basename(os.path.dirname(data_dir))
    
print('Dataset: ' + dataset_name)
print('Date: ' + data_date)
print('Path: ' + data_dir)

#%%
# Manual paths for interactive mode
dataset_name = 'rot_link6_w_mass'
data_date = '0417'
data_dir = os.getcwd() + '/paramID_data/' + data_date + '/' + dataset_name + '/'

print('Dataset: ' + dataset_name)
print('Date: ' + data_date)
print('Path: ' + data_dir)

#%%
# Camera intrinsic and extrinsic transforms
# K matrix from saved camera info
# TODO - extracting values manually from calibration files... mega hacky but... it'll do? Until anything at all changes and breaks it
K_line = np.loadtxt(data_dir + 'color_camera_info.txt', dtype='str', delimiter=',', skiprows=10, max_rows=1)
K_cam = np.array([[float(K_line[0][4:]), 0.0, float(K_line[2])], 
                  [0.0, float(K_line[4]), float(K_line[5])], 
                  [0.0, 0.0, 1.0]])

R_line = np.loadtxt(data_dir + '../calib_' + data_date + '.launch', dtype='str', delimiter=' ', skiprows=5, max_rows=1)
R_cam = R.from_quat([float(R_line[11]), float(R_line[12]), float(R_line[13]), float(R_line[14])]).as_matrix() # cam to base frame
T_cam = np.array([[float(R_line[6][6:])],[float(R_line[7])],[float(R_line[8])]])
                 
E_base = np.hstack([R_cam, T_cam]) # cam to base frame
E_cam = np.hstack([R_cam.T, -R_cam.T@T_cam]) # base to cam frame
P = K_cam@E_cam

#%%
# Process each image in folder
img_folder = data_dir + 'images/'

mid_positions_px = []
end_positions_px = []
mid_positions_XZplane = []
end_positions_XZplane = []
mid_mask_pixels = []
end_mask_pixels = []
marker_ts = []

# For depth
# mid_positions_XYZ = []
# end_positions_XYZ = []
# dimg_list = os.listdir(data_dir + 'depth/')
# didx = 0

# Mask range for green and blue markers
lower_G = np.array([20,80,20])
upper_G = np.array([80,255,255])
lower_B = np.array([90,100,20])
upper_B = np.array([130,255,255])

for img_name in os.listdir(img_folder):
    img = cv2.imread(img_folder + img_name)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    mask_G = cv2.inRange(hsv, lower_G, upper_G)
    mask_B = cv2.inRange(hsv, lower_B, upper_B)

    # Crop above base
    mask_G[:,1000:] = 0
    mask_B[:,900:] = 0
    # Crop trouble areas creating some spurious readings - TODO proper solution, outlier resistant
    mask_G[:,:125] = 0
    # mask_B[:25,:] = 0

    # Remove noise from mask
    mask_G = cv2.morphologyEx(mask_G, cv2.MORPH_OPEN, np.ones((5,5)))
    mask_G = cv2.morphologyEx(mask_G, cv2.MORPH_CLOSE, np.ones((5,5)))
    mask_B = cv2.morphologyEx(mask_B, cv2.MORPH_OPEN, np.ones((5,5)))
    mask_B = cv2.morphologyEx(mask_B, cv2.MORPH_CLOSE, np.ones((5,5)))
    # Remove noise from mask TODO - is this here twice on purpose?
    mask_G = cv2.morphologyEx(mask_G, cv2.MORPH_OPEN, np.ones((5,5)))
    mask_G = cv2.morphologyEx(mask_G, cv2.MORPH_CLOSE, np.ones((5,5)))
    mask_B = cv2.morphologyEx(mask_B, cv2.MORPH_OPEN, np.ones((5,5)))
    mask_B = cv2.morphologyEx(mask_B, cv2.MORPH_CLOSE, np.ones((5,5)))

    # Mask non-marker pixels
    masked_G = cv2.bitwise_and(img,img,mask=mask_G)
    masked_B = cv2.bitwise_and(img,img,mask=mask_B)

    # Locate markers at COM of mask
    mid_pos_px = np.round(center_of_mass(mask_G)).astype(int)
    end_pos_px = np.round(center_of_mass(mask_B)).astype(int)
    # Convert px location to world frame, assuming on Y=0 plane
    mid_pos_XZplane = UV_to_XZplane(mid_pos_px[1],mid_pos_px[0])
    end_pos_XZplane = UV_to_XZplane(end_pos_px[1],end_pos_px[0])

    # # Convert px location to world frame, using depth measurement
    # dimg = cv2.imread(data_dir + 'depth/' + dimg_list[didx], cv2.IMREAD_UNCHANGED)
    # mid_pos_XYZ = UV_to_XYZ(mid_pos_px[1],mid_pos_px[0])
    # end_pos_XYZ = UV_to_XYZ(end_pos_px[1],end_pos_px[0])
    # mid_positions_XYZ.append(mid_pos_XYZ)
    # end_positions_XYZ.append(end_pos_XYZ)

    mid_positions_px.append(mid_pos_px)
    end_positions_px.append(end_pos_px)
    mid_positions_XZplane.append(mid_pos_XZplane)
    end_positions_XZplane.append(end_pos_XZplane)
    mid_mask_pixels.append(np.where(masked_G[:,:,0] > 0))
    end_mask_pixels.append(np.where(masked_B[:,:,0] > 0))
    marker_ts.append(img_name[:-4])

mid_positions_px = np.array(mid_positions_px)
end_positions_px = np.array(end_positions_px)
mid_positions_XZplane = np.array(mid_positions_XZplane)
end_positions_XZplane = np.array(end_positions_XZplane)
t_markers = np.array(marker_ts, dtype=np.float64)
t_markers = (t_markers - t_markers[0])/1e9

# For depth
# mid_positions_XYZ = np.array(mid_positions_XYZ)
# end_positions_XYZ = np.array(end_positions_XYZ)

#%%
# Export to csv
with open(data_dir + 'marker_positions.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerow(['ts', 'mid_pos_x', 'mid_pos_z', 'end_pos_x', 'end_pos_z'])
    for n in range(len(marker_ts)):
        writer.writerow([marker_ts[n], 
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