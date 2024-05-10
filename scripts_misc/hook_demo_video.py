#!/usr/bin/env python
# Need to put this in the top directory to use, or figure out module imports better

#%%
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter

from generated_functions.floating.floating_base_functions import eval_fk
from utils import get_FK

#%%
# Data paths
dataset_name = '650_0'
data_date = '0402-loop_demo_static_id'
data_dir = os.getcwd() + '/paramID_data/' + data_date + '/' + dataset_name
if not os.path.exists(data_dir + '/videos'):
            os.makedirs(data_dir + '/videos')

print('Dataset: ' + dataset_name)
print('Date: ' + data_date)
print('Path: ' + data_dir)

#%%
# Import data
img_dir = data_dir + '/images'
ts_markers = np.loadtxt(data_dir + '/marker_positions.csv', dtype=np.ulonglong, delimiter=',', skiprows=1, usecols=0)
Img = np.array(ts_markers, dtype=str)
# Load EE pt sequence
sequence = np.loadtxt(data_dir + '/sequence.csv', dtype=np.float64, delimiter=',')
X_seq = sequence[:,0]
Z_seq = sequence[:,1]
Phi_seq = sequence[:,2]
Theta0_seq = sequence[:,3]
Theta1_seq = sequence[:,4]
Goals_X = sequence[:,5]
Goals_Z = sequence[:,6]
Goals_Alpha = sequence[:,7]
Endpt_Sols_X = sequence[:,8] 
Endpt_Sols_Z = sequence[:,9]
Endpt_Sols_Alpha = sequence[:,10]
# Camera intrinsic and extrinsic transforms
with np.load(data_dir + '/../TFs_adj.npz') as tfs:
    P = tfs['P']
    E_base = tfs['E_base']
    E_cam = tfs['E_cam']
    K_cam = tfs['K_cam']

# Manual set up of some parameters
with np.load('object_parameters/black_short_loop_100g.npz') as obj_params:
    p_vals = list(obj_params['p_vals']) # cable properties: mass (length), mass (end), length, diameter
Y_plane = 0
radial_constraint = 0.65
data_fps = 6
# Manually find the timestamp where the intermediate goals in the sequence are reached
# goal_stamps = ['1712070093366770459','1712070096707564821','1712070097710131047','1712070101049976012','end'] # For '600_0'
# goal_stamps = ['1712069839797070669','1712069843137408694','1712069844307512169','1712069847481138975','end'] # For '600_45'
goal_stamps = ['1712069552170984409','1712069555840234545','1712069558845864594','1712069562186332750','end'] # For '650_0'
# goal_stamps = ['1712069552170984409','1712069555840234545','1712069558845864594','1712069562186332750','1712069568868238491'] # For exporting sequence frames
# goal_stamps = ['1712069245479911705','1712069247984799705','1712069251993981851','1712069255168738209','end'] # For '650_45'
# goal_stamps = ['1712056584864818872','1712056589206704778','1712056592212611278','1712056595385432737','end'] # For '700_0'
# goal_stamps = ['1712056836422758340','1712056839425849955','1712056844607239619','1712056847782117413','end'] # For '700_45'


#%%

fig, ax = plt.subplots()
# fig.set_size_inches(19.2, 14.4) # For exporting sequence frames

image = plt.imshow(cv2.cvtColor(cv2.imread(img_dir + '/' + '1712069534963297943' + '.jpg'), cv2.COLOR_BGR2RGB))

constraint, = plt.plot([], [])
constraint.set_color('tab:red')
constraint.set_linewidth(0.75)
# constraint.set_linewidth(3.0) # For exporting sequence frames
constraint_XYZ = np.vstack([radial_constraint * np.cos(np.linspace(0,2*np.pi,100)), Y_plane*np.ones(100), radial_constraint * np.sin(np.linspace(0,2*np.pi,100)) + 0.333, np.ones(100)])
constraint_UV = P@constraint_XYZ
constraint.set_data(constraint_UV[0]/constraint_UV[2],constraint_UV[1]/constraint_UV[2])
curve, = plt.plot([], [])
curve.set_color('yellow')
curve.set_linestyle('--')
# curve.set_linewidth(3) # For exporting sequence frames
s = 0
curve_XZ = get_FK(p_vals,[Theta0_seq[s],Theta1_seq[s],X_seq[s],Z_seq[s],Phi_seq[s]],eval_fk,21)
curve_XYZ = np.vstack([curve_XZ[:,0],Y_plane*np.ones((curve_XZ.shape[0],)),curve_XZ[:,1],np.ones((curve_XZ.shape[0],))])
curve_UV = P@curve_XYZ
curve.set_data(curve_UV[0]/curve_UV[2],curve_UV[1]/curve_UV[2])

# goal = plt.scatter([], [], s=1, c='tab:green',marker='x',zorder=2.5)
# goal_XYZ = np.vstack([Goals_X, Y_plane*np.ones(5), Goals_Z, np.ones(5)])
# goal_UV = P@goal_XYZ
# goal.set_offsets(np.array([goal_UV[0,:]/goal_UV[2,:],goal_UV[1,:]/goal_UV[2,:]]).T)
# goals, = plt.plot([], [])
# goals.set_linestyle(':')
# goals.set_color('tab:green')
# goals.set_linewidth(0.5)
# goals.set_data(goal_UV[0,:]/goal_UV[2,:],goal_UV[1,:]/goal_UV[2,:])

# curves = [plt.plot([], []), plt.plot([], []), plt.plot([], []), plt.plot([], []), plt.plot([], [])]
# for s in range(len(curves)):
#     curves[s][0].set_color('yellow')
#     curve_XZ = get_FK([0.42,0.12,0.43,0.02],[Theta0_seq[s],Theta1_seq[s],X_seq[s],Z_seq[s],Phi_seq[s]],eval_fk,21)
#     curve_XYZ = np.vstack([curve_XZ[:,0],Y_plane*np.ones((curve_XZ.shape[0],)),curve_XZ[:,1],np.ones((curve_XZ.shape[0],))])
#     curve_UV = P@curve_XYZ
#     curves[s][0].set_data(curve_UV[0]/curve_UV[2],curve_UV[1]/curve_UV[2])

#%%

import matplotlib
matplotlib.use("Agg")

fps=24
writer = FFMpegWriter(fps)

with writer.saving(fig, data_dir + '/videos/hook_demo_' + '.mp4', 400):
    for idx in range(Img.shape[0]):
        if idx % fps == 0:
            print('Generating animation, ' + str(idx) + ' of ' + str(Img.shape[0]) + ' frames')

        img_name = '/' + Img[idx] + '.jpg'
        img = cv2.cvtColor(cv2.imread(img_dir + img_name), cv2.COLOR_BGR2RGB)
        image.set_data(img)

        if str(Img[idx]) == goal_stamps[s]:
            s = s + 1
            curve_XZ = get_FK(p_vals,[Theta0_seq[s],Theta1_seq[s],X_seq[s],Z_seq[s],Phi_seq[s]],eval_fk,21)
            curve_XYZ = np.vstack([curve_XZ[:,0],Y_plane*np.ones((curve_XZ.shape[0],)),curve_XZ[:,1],np.ones((curve_XZ.shape[0],))])
            curve_UV = P@curve_XYZ
            curve.set_data(curve_UV[0]/curve_UV[2],curve_UV[1]/curve_UV[2])

        writer.grab_frame()

    print("Finished")
    plt.close(fig)
    
# %%
