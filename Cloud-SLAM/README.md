# Cloud-SLAM
## This section of the code is a partial Cloud SLAM deployment of VC-SLAM on cloud servers，Please deploy the cloud code and environment first.
### 1. Download and configure Droid Slam yourself. The code link is: https://github.com/princeton-vl/DROID-SLAM.git.
### 2. Make the following modifications in the Droid Slam code: add the instruction 'self.video.tstamp [ix]=self.video.tstamp [ix+1]' to the function rm_keyfram in the factor_graph.py function.
### 3. There are several places where the path needs to be modified in Cloud SLAM/src/edgecloud/scripts/cloud_stam_kan.py, which are # 27# 28 ，# 43，# Lines 95 and # 473, please configure according to the installation path by yourself.
```
#27 sys.path.append('/data/xjluo/DROID-SLAM/droid_slam')

#28 sys.path.append('/data/xjluo/DROID-SLAM/droid_slam')

#95 def put_pth(pth, remote_path, file_path="/data/xjluo/cloud_udf/trans_work"):

#473 cloudslalm = Cloud_Slam('/data/xjluo/cloud_udf/src/edgecloud/scripts/confs/base.conf')
```
### 4. In addition, in terms of communication transmission, there are many paths that need to be changed to edge related file paths by oneself. It will not be pointed out here one by one. This project uses SSH transmission, so the edge needs to have SSH function for cloud access. The specific function is put_pth, and SSH needs to be configured by oneself.
### 5. Configure the cloud operating environment and run the following command in your terminal:
```
conda env create -f cloud_slam_env.yml
```
### 6. Then compile the cloud ROS package with the following instructions:
```
cd src

catkin_init_workspace

cd ..

catkin_make
```
### 7. Based on ROS communication mechanism, open a new terminal to run:
```
roscore
```
### 8. Activate the cloud runtime environment and run the cloud_stlam_ksan.py：
```
conda activate cloud-slam

python cloud_slam_kan.py
```
