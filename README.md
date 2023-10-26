# floating_PAC_model

This repository contains Python code used mostly for data processing, as well as a dynamic model using SymPy mirroring the [Matlab version](https://github.com/sebtiburzio/PAC_model_matlab). 

## Model

The model is functionally the same as the Matlab version, however is currently much slower so mostly only useable for kinematics. The most likely candidate for improving this is the computation of the Fresnel functions (noting that these are manually replaced with an approximation in the Matlab version).

Scripts defining the model are in `scripts_model` which outputs to `generated_functions`.

## Data Processing

Several scripts process data collected in the experiments (manipulator EE state `EE_pose.csv`, force/torque readings `EE_wrench.csv`, RGB images `images\` and associated extrinsic camera calibration transform `TFs_adj.npz`). [Another repo](https://github.com/sebtiburzio/dlo_parameter_id) is more related to collecting this data.

These are interactive scripts that generally have parameters that need to be adjusted while stepping through.

### image_processing.py

Uses the directory of image frames captured in the experiment and outputs the positions of the base, middle and end coloured markers on the object, in the XZ plane of the object model. The ROS timestamps are embedded as the image names.

![image](https://github.com/sebtiburzio/floating_PAC_model/assets/95340175/f309ddf7-ced5-4aa6-a411-365695cc84a1)

### image_processing_manual.py

Version where marker points can be manually selected in each frame instead of automatically by HSV filtering.

### post_processing.py

Processes the time series of manipulator EE state, FT measurements and marker positions extracted from the images. Resample data to consistent rate, extract curvature configurations of Theta from the marker positions, calculate derivatives. Output is saved to `data_out\` and `processed.npz`. Also can generate animations of the state in XY plane or projected over the recorded image frames.

https://github.com/sebtiburzio/floating_PAC_model/assets/95340175/37dc3b1c-74fa-4f70-a370-83a530ec3c7e

https://github.com/sebtiburzio/floating_PAC_model/assets/95340175/2147ab62-2974-4144-902b-fff92b42dbe9

### static_processing.py

Version for processing a set of static equilibrium data points.
