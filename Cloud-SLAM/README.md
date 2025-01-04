# Cloud-SLAM
## This section of the code is a partial Cloud SLAM deployment of VC-SLAM on cloud servers，Please deploy the cloud code and environment first.
### 1. Download and configure Droid Slam yourself. The code link is: https://github.com/princeton-vl/DROID-SLAM.git.
### 2. Make the following modifications in the Droid Slam code: add the instruction 'self.video.tstamp [ix]=self.video.tstamp [ix+1]' to the function rm_keyfram in the factor_graph.py function.
### 3. Configure the cloud operating environment and run the following command in your terminal:
> conda env create -f cloud_slam_env.yml
### 4. 

