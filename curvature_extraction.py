#!/usr/bin/env python
#%%
import os
import cv2
import numpy as np
import sympy as sm
import dill
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.ndimage import center_of_mass

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
    ax.plot(np.array(mid_positions_px)[idx,1],np.array(mid_positions_px)[idx,0],ms=2,fillstyle='none',marker='o',mec=marker_colors[0])
    ax.plot(np.array(end_positions_px)[idx,1],np.array(end_positions_px)[idx,0],ms=2,fillstyle='none',marker='o',mec=marker_colors[1])
    # ax.scatter(np.array(mid_positions_px)[-1,1],np.array(mid_positions_px)[-1,0],s=10,c='None',marker='o',edgecolors='lawngreen')
    # ax.scatter(np.array(end_positions_px)[-1,1],np.array(end_positions_px)[-1,0],s=10,c='None',marker='o',edgecolors='skyblue')

def find_curvature(theta_guess, fk_target, epsilon=0.01, max_iterations=1000):   #TODO - check more inputs
    theta_est = None
    for i in range(max_iterations):
        error = np.vstack([f_FK_mid(theta_guess,p_vals),f_FK_end(theta_guess,p_vals)]) - fk_target
        if np.linalg.norm(error) < epsilon:
            theta_est = theta_guess
            break
        else:
            J = np.vstack([f_J_mid(theta_guess, p_vals),f_J_end(theta_guess, p_vals)])
    if theta_est is None:
        print('Failed to converge')
    return theta_est

#%%
# Process each image in folder
img_folder = './paramID_data/0406/sine_x_w_depth/images_subset/'

mid_positions_px = []
end_positions_px = []
mid_mask_pixels = []
end_mask_pixels = []
marker_ts = []

lower_G = np.array([25,64,30])
upper_G = np.array([85,255,255])
lower_B = np.array([85,64,30])
upper_B = np.array([135,255,255])
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

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

    # Remove noise from mask
    mask_G = cv2.morphologyEx(mask_G, cv2.MORPH_OPEN, np.ones((5,5)))
    mask_G = cv2.morphologyEx(mask_G, cv2.MORPH_CLOSE, np.ones((5,5)))
    mask_B = cv2.morphologyEx(mask_B, cv2.MORPH_OPEN, np.ones((5,5)))
    mask_B = cv2.morphologyEx(mask_B, cv2.MORPH_CLOSE, np.ones((5,5)))

    # Mask non-marker pixels
    masked_G = cv2.bitwise_and(img,img,mask=mask_G)
    masked_B = cv2.bitwise_and(img,img,mask=mask_B)

    mid_positions_px.append(center_of_mass(mask_G))
    end_positions_px.append(center_of_mass(mask_B))
    mid_mask_pixels.append(np.where(masked_G[:,:,0] > 0))
    end_mask_pixels.append(np.where(masked_B[:,:,0] > 0))
    marker_ts.append(img_name[:-4])

t_markers = np.array(marker_ts, dtype=np.float64)
t_markers = (t_markers - t_markers[0])/1e9

# Convert pixels to space
# Animated plot of marker positions to check realistic?

# Transform to base frame (substract X/Z rotate phi)
# Extract curvature - initial guess close to zero, subsequent guess is previous estimate

# Interpolate sample times
# Filtering? (or filter before curvature extraction?)
# Calculate derivatives

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
# Curavture IK

f_FK_mid = dill.load(open('./generated_functions/f_FK_mf','rb'))
f_FK_end = dill.load(open('./generated_functions/f_FK_ef','rb'))
f_J_mid = dill.load(open('./generated_functions/f_J_mf','rb'))
f_J_end = dill.load(open('./generated_functions/f_J_ef','rb'))

p_vals = [1.0, 0.5, 1.0, 0.1]

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