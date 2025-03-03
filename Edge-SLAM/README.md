# Udf-Edge
简要介绍一下，代码分成三部分，其中边端有两部分（Cloud-Edge-SLAM和Udf-Edge），云端有一部分cloud_udf

三部分都是基于ROS搭建的，所以执行本项目先确保边端云端电脑有安装ROS，最好版本一致

### 云端环境的搭建
先自行装好Droid-Slam，然后在其factor_graph.py函数中的function rm_keyfram里添加 self.video.tstamp[ix] = self.video.tstamp[ix+1]
然后编译云端ros包，指令如下
```
cd src
catkin_init_workspace
cd ..
catkin_make
```
在src/edgecloud/scripts/cloud_slam.py里有几处要修改路径，分别是#27 and #28 and #79行
除此之外，在通信传输上，有较多路径需要自己更改，这里不一一指出，本项目采用ssh传输的，所以边端需具备ssh功能供云端访问，具体函数是put_pth
运行指令：
```
conda activate droidenv5 # 激活你的conda环境
python src/edgecloud/scripts/cloud_slam.py #运行主函数
```
### 边端环境的搭建
Cloud-Edge-SLAM上有readme.md介绍了基本的安装步骤，这里不重复介绍
至于Udf-Edge，先编译Ros包，然后执行主函数即可
```
python src/cap/edge/edge.py
```
就是这样，运行后报错再根据错误修改即可，可检查路径是否有误
### 其他
有问题可以联系邮箱：luocby@gmail.com
