#!/usr/bin/env python
#%%

import cv2
import numpy as np
import matplotlib.pyplot as plt

def showim(img):
    res = cv2.resize(img,None,fx=0.75, fy=0.75, interpolation = cv2.INTER_CUBIC)
    cv2.imshow('image',res)
    k = cv2.waitKey(0) & 0xFF
    if k == 27:         # wait for ESC key to exit
        cv2.destroyAllWindows()

#%%
img = cv2.imread('./paramID_data/sine_x_fast/images/1679419822965192813.jpg')

#%%
showim(img)

# %%
# HSV threshold segmentation
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
lower = np.array([20,64,30])
upper = np.array([95,255,255])
mask = cv2.inRange(hsv, lower, upper)

# Crop trouble areas
mask[:,:480] = 0
mask[:200,:] = 0

# Remove noise from mask
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5)))
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5)))

# Mask non-makrer pixels
res = cv2.bitwise_and(img,img, mask= mask)

# Find centroid of markers using k-means clustering
Y, X = np.where(res[:,:,0] > 0)
Z = np.vstack((X,Y)).T
Z = np.float32(Z)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
ret,_,center=cv2.kmeans(Z,5,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

plt.scatter(center[:,0],center[:,1])
plt.xlim(0,1920)
plt.ylim(0,1080)
plt.show()

showim(res)
# %%
