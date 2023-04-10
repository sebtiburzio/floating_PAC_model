#!/usr/bin/env python
#%%
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

def plot_markers(idx, plot_mask=False):
    img_name = os.listdir(img_folder)[idx]
    img = cv2.cvtColor(cv2.imread(img_folder + img_name), cv2.COLOR_BGR2RGB)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(autoscale_on=False)
    ax.set_aspect('equal')
    ax.grid()
    ax.set_xlim(0,1920)
    ax.set_ylim(0,1080)
    marker_colors = ['palegreen','lightskyblue']

    if plot_mask:
        marker_colors = ['forestgreen','dodgerblue']
        ax.plot(mid_mask_pixels[idx][1],mid_mask_pixels[idx][0],',',c='palegreen')
        ax.plot(end_mask_pixels[idx][1],end_mask_pixels[idx][0],',',c='lightskyblue')

    ax.imshow(np.asarray(img))
    ax.plot(mid_positions_px[idx,1],mid_positions_px[idx,0],ms=2,fillstyle='none',marker='o',mec=marker_colors[0])
    ax.plot(end_positions_px[idx,1],end_positions_px[idx,0],ms=2,fillstyle='none',marker='o',mec=marker_colors[1])
    # ax.scatter(np.array(mid_positions_px)[-1,1],np.array(mid_positions_px)[-1,0],s=10,c='None',marker='o',edgecolors='lawngreen')
    # ax.scatter(np.array(end_positions_px)[-1,1],np.array(end_positions_px)[-1,0],s=10,c='None',marker='o',edgecolors='skyblue')

#%%
# Camera intrinsic and extrinsic transforms
# K matrix (TODO - read from camera info.txt)
K_cam = np.array([[1365.853271484375, 0.0, 975.876708984375], 
                  [0.0, 1365.0545654296875, 559.9922485351562], 
                  [0.0, 0.0, 1.0]])
R_cam = R.from_quat([0.523796, 0.6114, -0.381026, 0.454584]).as_matrix() # cam to base frame # TODO - import from calib file
T_cam = np.array([[0.292646],[0.606908],[0.5813980]])
E_cam = np.hstack([R_cam.T, -R_cam.T@T_cam]) # base to cam frame
P = K_cam@E_cam

def px_to_space(u,v):
    rhs1 = np.hstack([P[:,:3],np.array([[-u,-v,-1]]).T])
    rhs1 = np.vstack([rhs1, np.array([0,1,0,0])])   # Intersect Y=0 plane
    rhs2 = np.reshape(np.hstack([-P[:,3],[0]]),(4,1))
    sol = np.linalg.inv(rhs1)@rhs2
    return sol[:3]

#%%
# Process each image in folder
img_folder = './paramID_data/0406/sine_x_w_depth/images/'

mid_positions_px = []
end_positions_px = []
mid_positions_3D = []
end_positions_3D = []
mid_mask_pixels = []
end_mask_pixels = []
marker_ts = []

# Mask range for green and blue markers
lower_G = np.array([25,64,30])
upper_G = np.array([85,255,255])
lower_B = np.array([85,64,30])
upper_B = np.array([135,255,255])

for img_name in os.listdir(img_folder):
    img = cv2.imread(img_folder + img_name)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    mask_G = cv2.inRange(hsv, lower_G, upper_G)
    mask_B = cv2.inRange(hsv, lower_B, upper_B)

    # Crop above base
    mask_G[:,:250] = 0
    mask_B[:,:250] = 0
    # Crop edge - creating some spurious blue readings - TODO proper solution, outlier resistant
    mask_B[:25,:] = 0

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
    mid_pos_px = center_of_mass(mask_G)
    end_pos_px = center_of_mass(mask_B)
    # Convert px location to 3D, assuming on Y=0 plane
    mid_pos_3D = px_to_space(mid_pos_px[1],mid_pos_px[0])
    end_pos_3D = px_to_space(end_pos_px[1],end_pos_px[0])

    mid_positions_px.append(mid_pos_px)
    end_positions_px.append(end_pos_px)
    mid_positions_3D.append(mid_pos_3D)
    end_positions_3D.append(end_pos_3D)
    mid_mask_pixels.append(np.where(masked_G[:,:,0] > 0))
    end_mask_pixels.append(np.where(masked_B[:,:,0] > 0))
    marker_ts.append(img_name[:-4])

mid_positions_px = np.array(mid_positions_px)
end_positions_px = np.array(end_positions_px)
mid_positions_3D = np.array(mid_positions_3D)
end_positions_3D = np.array(end_positions_3D)
t_markers = np.array(marker_ts, dtype=np.float64)
t_markers = (t_markers - t_markers[0])/1e9

#%%
# Export to csv
with open(img_folder + '../marker_positions.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerow(['ts', 'mid_pos_x', 'mid_pos_z', 'end_pos_x', 'end_pos_z'])
    for n in range(len(marker_ts)):
        writer.writerow([marker_ts[n], 
                         float(mid_positions_3D[n,0]), float(mid_positions_3D[n,2]), 
                         float(end_positions_3D[n,0]), float(end_positions_3D[n,2])])

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