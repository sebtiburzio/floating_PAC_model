#!/usr/bin/env python

# Need to put this in the top directory to use, or figure out module imports better
#%%
import os
import cv2
import csv
import numpy as np
import matplotlib.pyplot as plt

from generated_functions.floating.floating_base_functions import eval_fk
from utils import get_FK

# Camera intrinsic and extrinsic transforms
with np.load('./TFs_adj.npz') as tfs:
    P = tfs['P']
    E_base = tfs['E_base']
    E_cam = tfs['E_cam']
    K_cam = tfs['K_cam']

states = np.loadtxt('./states.csv', dtype=np.float64, delimiter=',')
Theta0 = states[:,0]
Theta1 = states[:,1]
X = states[:,2]
Z = states[:,3]
Phi = states[:,4]

img = cv2.cvtColor(cv2.imread('./composite.jpg'), cv2.COLOR_BGR2RGB)

p_vals = [1,1,0.6,0.02]
Y_meas = -0.25

fig = plt.figure(figsize=(25, 20))
ax = fig.add_subplot(autoscale_on=False)
ax.set_aspect('equal')
ax.set_xlim(0,img.shape[1])
ax.set_ylim(0,img.shape[0])



for i in [1,2,3,4,5]:
    print(i)
    img = cv2.cvtColor(cv2.imread('./' + str(i) + '.png'), cv2.COLOR_BGR2RGB)
    ax.imshow(np.asarray(img))
    XZ = get_FK(p_vals,[Theta0[0],Theta1[0],X[0],Z[0],Phi[0]],eval_fk,21)    
    curve_XYZ = np.vstack([XZ[:,0],Y_meas*np.ones((XZ.shape[0],)),XZ[:,1],np.ones((XZ.shape[0],))])
    FK_evals = P@curve_XYZ
    ax.plot(FK_evals[0]/FK_evals[2],FK_evals[1]/FK_evals[2],ls='--',color='yellow',linewidth=7)

    ax.scatter(FK_evals[0,0]/FK_evals[2,0],FK_evals[1,0]/FK_evals[2,0],c='orangered',s=50,zorder=2.5)
    ax.scatter(FK_evals[0,10]/FK_evals[2,10],FK_evals[1,10]/FK_evals[2,10],c='limegreen',s=50,zorder=2.5)
    ax.scatter(FK_evals[0,20]/FK_evals[2,20],FK_evals[1,20]/FK_evals[2,20],c='dodgerblue',s=50,zorder=2.5)

    plt.savefig('./out' + str(i) + '.png', bbox_inches='tight', pad_inches=0.1)
plt.close()
# %%
