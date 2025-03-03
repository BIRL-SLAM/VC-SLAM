import os
import torch
import numpy as np
import torch.nn.functional as F
#from models.fields import CAPUDFNetwork
from models.kan_network import CAPUDFNetwork1
from models.dataset import Dataset
from pyhocon import ConfigFactory
from scipy.spatial.transform import Rotation as spR
import time
from msg_action.msg import CloudMap
from msg_action.msg import KeyFrame
from msg_action.msg import MapPoint
from msg_action.msg import Descriptor
from msg_action.msg import KeyPoint
from msg_action.msg import Observation
from std_msgs.msg import Header
from geometry_msgs.msg import Pose
from geometry_msgs.msg import Point
from geometry_msgs.msg import Quaternion
from msg_action.msg import CloudSlamAction, CloudSlamResult
import rosbag
from collections import OrderedDict

class Runner:
    def __init__(self,conf):
        self.device = torch.device('cpu')   
        self.conf = conf
        self.iter_step = 0
        self.conf_path = conf
        f = open(self.conf_path)
        conf_text = f.read()
        f.close()

        self.conf = ConfigFactory.parse_string(conf_text)

        self.point_dir = self.conf['general.point_dir']

        self.others_dir = self.conf['general.others_dir']

        self.new_udf = self.conf['general.new_udf']
        # Networks
        #self.udf_network = CAPUDFNetwork(**self.conf['model.udf_network'])
        self.udf_network = CAPUDFNetwork1()
        #self.optimizer = torch.optim.Adam(self.udf_network.parameters(), lr=0.001)
        self.optimizer = torch.optim.AdamW(self.udf_network.parameters(), lr=0.001,weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.8)
        
    def run(self):
        test = True
        while test:
            te = os.access(self.point_dir+'test.xyz',os.F_OK)
            #print(self.point_dir)
            if te :
                #time_1 = time.time()
                time.sleep(2) #等test.xyz加载好
                time_edge_s = time.time()
                self.load_checkpoint('/home/birl/Udf-Edge/Cloud-Edge-SLAM/Cloud-Edge-SLAM-master/trans_pth/test.pth')
                process_time_s = time.time()
                self.dataset = Dataset(self.point_dir,"test")
                process_time_e = time.time()
                process_t = process_time_e - process_time_s
                print("process time: {}".format(process_t))
                if (self.new_udf):
                    idx_keyframe = np.load(self.others_dir + 'idx_keyframe.npy')
                    poses_keyframe = np.load(self.others_dir + 'poses_keyframe.npy')
                    depths_keyframe = np.load(self.others_dir + 'depths_keyframe.npy')
                    intrinsics_keyframe = np.load(self.others_dir + 'intrinsics_keyframe.npy')
                    intrisics_fullsize = np.load(self.others_dir + 'intrisics_fullsize.npy')
                    gen_time_s = time.time()
                    pc_v, cam_mpt_uvd_v = self.gen_extra_pointcloud(1.0, idx_keyframe, poses_keyframe, depths_keyframe, intrinsics_keyframe, intrisics_fullsize)
                    gen_time_e = time.time()
                    gen_t = gen_time_e - gen_time_s
                    print("gen time: {}".format(gen_t))
                    tstamps_keyframe = np.load(self.others_dir + 'tstamps_keyframe.npy')
                    result = CloudSlamResult()
                    map_time_s = time.time()
                    result.map = self.data_to_message(tstamps_keyframe, poses_keyframe, idx_keyframe, pc_v, cam_mpt_uvd_v)
                    map_time_e = time.time()
                    map_t = map_time_e - map_time_s
                    print("map time: {}".format(map_t))
                    id = np.loadtxt(self.others_dir + 'id.txt')
                    result.map.edge_front_map_mnid = int(id[0])
                    result.map.edge_back_map_mnid  = int(id[1])
                    #print("-------------------------result++++++++++++++++++++:",result)
                    bag = rosbag.Bag('/home/birl/Udf-Edge/Cloud-Edge-SLAM/Cloud-Edge-SLAM-master/trans_bag/test.bag', 'w')
                    bag.write('/test', result)
                    bag.close()
                    #os.remove(self.others_dir + 'idx_keyframe.npy')
                    #os.remove(self.others_dir + 'poses_keyframe.npy')
                    #os.remove(self.others_dir + 'depths_keyframe.npy')
                    #os.remove(self.others_dir + 'intrinsics_keyframe.npy')
                    #os.remove(self.others_dir + 'intrisics_fullsize.npy')
                    #os.remove(self.others_dir + 'tstamps_keyframe.npy')
                    #os.remove(self.others_dir + 'id.txt')
                    test = True
                    time_edge_e = time.time()
                    time_edge = time_edge_e - time_edge_s
                    print("edge_udf time: {}".format(time_edge))
                    time_all = []
                    time_all.append(process_t)
                    time_all.append(gen_t)
                    time_all.append(map_t)
                    time_all.append(time_edge)
                    np.set_printoptions(suppress=True)
                    np.set_printoptions(precision=4)   
                    np.savetxt('time.txt',time_all,fmt='%f')
                else:
                    self.gen_extra_point(1.0)
                    test = True
                print("---------------------------Finsh-----------------------")
                #time_2 = time.time()
                #time_all=time_2-time_1
                #print("All Time: {}s".format(time_all))
            else:
                test = True

    def gen_extra_pointcloud(self, low_range, indexes, poses, depths, intrisics, intrisics_fs):

        res = []
        num_points = 10000
        gen_nums = 0
        pc_list = []
        uv_list = []
        i = 0
        scale = self.dataset.get_scale()
        center = self.dataset.get_center()
        while gen_nums < num_points:
            
            points, samples = self.dataset.get_train_data(5000)
            offsets = samples - points
            std = torch.std(offsets)
            std = torch.std(points)

            extra_std = std * low_range
            rands = torch.normal(0.0, extra_std, size=points.shape)
            #samples = points + rands..clone().detach().cpu().float().requires_grad_(True)
            samples = points + torch.tensor(rands).cpu().float()
            samples.requires_grad = True
            gradients_sample = self.udf_network.gradient(samples).squeeze() # 5000x3
            udf_sample = self.udf_network.udf(samples)                      # 5000x1
            grad_norm = F.normalize(gradients_sample, dim=1)                # 5000x3
            sample_moved = samples - grad_norm * udf_sample                 # 5000x3

            index = udf_sample < 0.01
            index = index.squeeze(1)
            sample_moved = sample_moved[index]

            res.append(sample_moved.detach().cpu().numpy())
            
            pc, uv = self.construct_covis_graph(res[i]*scale + center, indexes, poses, depths, intrisics, intrisics_fs, gen_nums)
            i = i + 1

            pc_list.append(pc) #test
            uv_list.append(uv) #test
            gen_nums += pc.shape[0] #test

        #res = np.concatenate(res) * scale + center
        pc_list = np.concatenate(pc_list) #test
        uv_list = np.concatenate(uv_list) #test
        #pc_list, uv_list = self.construct_covis_graph(res, indexes, poses, depths, intrisics, intrisics_fs, 0)
        #np.savetxt(os.path.join('test2.xyz'), res)
        return pc_list, uv_list #test

    def gen_extra_point(self, low_range):

        res = []
        gen_nums = 0
        num_points = 10000

        while gen_nums < num_points:
            
            points, samples = self.dataset.get_train_data(5000)
            offsets = samples - points
            std = torch.std(offsets)
            std = torch.std(points)

            extra_std = std * low_range
            rands = torch.normal(0.0, extra_std, size=points.shape)   
            samples = points + torch.tensor(rands).cpu().float()

            samples.requires_grad = True
            gradients_sample = self.udf_network.gradient(samples).squeeze() # 5000x3
            udf_sample = self.udf_network.udf(samples)                      # 5000x1
            grad_norm = F.normalize(gradients_sample, dim=1)                # 5000x3
            sample_moved = samples - grad_norm * udf_sample                 # 5000x3

            index = udf_sample < 0.01
            index = index.squeeze(1)
            sample_moved = sample_moved[index]
            
            gen_nums += sample_moved.shape[0]

            res.append(sample_moved.detach().cpu().numpy())

        scale = self.dataset.get_scale()
        center = self.dataset.get_center()
        res = np.concatenate(res) * scale + center
        res = res[:num_points]

        np.savetxt(os.path.join(self.point_dir, 'test2.xyz'), res)


    def load_checkpoint(self, pth_path):
        checkpoint = torch.load(os.path.join(pth_path), map_location=self.device)
        ###########################################################################
        # new_state_dict = OrderedDict()
        # for k, v in checkpoint['udf_network_fine'].items():
        #     name = k[7:] # remove `module.`
        #     new_state_dict[name] = v
        # self.udf_network.load_state_dict(new_state_dict)
        ###########################################################################
        self.udf_network.load_state_dict(checkpoint['udf_network_fine'])
        #os.remove(pth_path)

    def construct_covis_graph(self, pointcloud, indexes, poses, depths, intrisics, intrisics_fs ,gen_nums):
        """ Construct covisibility graph with data from droid-slam

        Args:
            pointcloud (ndarray): Nx3
            indexes (ndarray): keyframe index in dataset
            poses (ndarray): num_poses x 7
            depths (ndarray):  60x80 depth maps from DROID-SLAM
            intrisics (ndarray): _description_
            intrisics_fs (ndarray): _description_

        Returns:
            pointcloud (ndarray): Nx3, delete not observed mappoints
            nd_idx_cam_idx_mpt_uv (ndarray): Nx[ kf idx in dataset, mpt idx in pointcloud, image coordinate(u, v) ]
        """
        ## data
        fx, fy, cx, cy = intrisics[0]
        cam_droid = np.array([  [fx, 0, cx], \
                                [0, fy, cy], \
                                [0, 0, 1] ])

        fx, fy, cx, cy = intrisics_fs
        cam_fullsize = np.array([ [fx, 0, cx], \
                                [0, fy, cy], \
                                [0,  0,  1] ])

        t_cw = poses[:,:3]
        q_cw = poses[:,3:]

        idx_mappts = np.arange(gen_nums, len(pointcloud) + gen_nums)
        idx_mappts_i = None
        T = np.identity(4)

        table_visibility = np.zeros([len(pointcloud), len(poses)], dtype=bool)
        idx_cam_idx_mpt_uv = []

        list_data = None

        list_data = list(range(len(poses)))
        for i in list_data:
            # transform points to camera frame
            Ri = spR.from_quat(q_cw[i]).as_matrix()
            ti = t_cw[i]
            T[:3,:3] = Ri
            T[:3,3] = ti.ravel()
            points_c = Ri.dot(pointcloud.T) + ti.reshape([3,1])
            points_c = points_c.T

            # positive depth mask
            mask_p_depth = points_c[:,-1]>1e-3
            points_c = points_c[mask_p_depth]
            idx_mappts_i = idx_mappts[mask_p_depth]

            # projection
            uvds_droid = cam_droid.dot(points_c.T)
            uvds_droid[:2,:] = uvds_droid[:2,:] / uvds_droid[-1,:].reshape([1,-1])
            uvds_droid = uvds_droid.T

            uvds_fullsize = cam_fullsize.dot(points_c.T)
            uvds_fullsize[:2,:] = uvds_fullsize[:2,:] / uvds_fullsize[-1,:].reshape([1,-1])
            uvds_fullsize = uvds_fullsize.T
        
            # fov mask
            mask_fov =  (uvds_droid[:,0]>=0) & (uvds_droid[:,0]<depths[0].shape[1]) \
                & (uvds_droid[:,1]>=0) & (uvds_droid[:,1]<depths[0].shape[0])
            uvds_droid = uvds_droid[mask_fov]
            uvds_fullsize = uvds_fullsize[mask_fov]
            idx_mappts_i = idx_mappts_i[mask_fov]

            # z test mask
            uv_droid = uvds_droid[:,:2].astype(np.int32)
            query = depths[i][ uv_droid[:,1], uv_droid[:, 0] ]
            mask_z_test = uvds_droid[:,2] < query * 1.

            uv_droid = uv_droid[mask_z_test]
            uvds_droid = uvds_droid[mask_z_test]
            uvds_fullsize = uvds_fullsize[mask_z_test]
            idx_mappts_i = idx_mappts_i[mask_z_test]

            # set marks
            table_visibility[idx_mappts_i - gen_nums, i] = True

            # save points
            ## camera idx
            idx_cam_i = np.ones([len(idx_mappts_i), 1]) * indexes[i]
            idx_cam_idx_mpt_uv.append( np.column_stack([ idx_cam_i.reshape([-1, 1]), \
                                                        idx_mappts_i.reshape([-1, 1]), \
                                                        uvds_fullsize ]) )

        nd_idx_cam_idx_mpt_uv = np.row_stack( idx_cam_idx_mpt_uv )
        # delete unobserved map points
        observed_table = table_visibility.sum(axis=1).ravel().astype(bool)
        id_new = np.zeros_like(observed_table, dtype=np.int32)
        count = gen_nums
        for i in range(len(id_new)):
            if observed_table[i]:
                id_new[i] = count
                count += 1

        pointcloud = pointcloud[observed_table]

        nd_idx_cam_idx_mpt_uv[:, 1] = id_new[ (nd_idx_cam_idx_mpt_uv[:, 1] - gen_nums).astype(np.int32) ]

        return pointcloud, nd_idx_cam_idx_mpt_uv

    def data_to_message(self, time_data, pose_data, kf_index_data, map_point_xyz_data, kf_map_point_uv):
        sum_time = 0
        sum_time_s = time.time()

        pub_cloud_map = CloudMap()
        pub_cloud_map.edge_front_map_mnid = 0
        pub_cloud_map.edge_back_map_mnid = 0
        pub_cloud_map.header = Header()

        keyframe_time = 0
        keyframe_time_s = time.time()

        ros_keyframes = []
        cov_mappoint_infos_list = np.full(
            fill_value=np.nan,
            dtype=np.float16,
            shape=(
                map_point_xyz_data.shape[0],                                 #map_point_xyz_data.shape[0]
                time_data.shape[0],
                kf_map_point_uv.shape[1],
            ),
        )
        cov_mappoint_infos_list_index = np.full(
            fill_value=0, dtype=np.int32, shape=(map_point_xyz_data.shape[0]) #map_point_xyz_data.shape[0]
        )
        # cov_mappoint_infos_list = [
        #     ([None] * max_cov_mappoint_size) for i in range(map_point_xyz_data.shape[0])
        # ]
        # cov_mappoint_infos_list_index = [0] * map_point_xyz_data.shape[0]
        for keyframe_i in range(time_data.shape[0]):
            # print("Process KeyFrame: {} / {}".format(keyframe_i, time_data.shape[0]))
            # time stamp
            time_stamp = time_data[keyframe_i]
            ros_time_stamp = time_stamp

            # pose
            pose = pose_data[keyframe_i]
            ros_pose = Pose(
                position=Point(x=pose[0], y=pose[1], z=pose[2]),
                orientation=Quaternion(x=pose[3], y=pose[4], z=pose[5], w=pose[6]),
            )

            # cur no descriptor

            # keypoint
            ros_keypoints = []
            kf_observations = kf_map_point_uv[kf_map_point_uv[:, 0] == keyframe_i]
            map_point_index = kf_observations[:, 1].astype(np.int32)
            for keypoint_i in range(kf_observations.shape[0]):
                ros_keypoint = KeyPoint()
                ros_keypoint.x = kf_observations[keypoint_i][2]
                ros_keypoint.y = kf_observations[keypoint_i][3]
                tmp_map_point_index = int(kf_observations[keypoint_i][1])
                # if 1:
                try:
                    cov_mappoint_infos_list[tmp_map_point_index][
                        cov_mappoint_infos_list_index[tmp_map_point_index]
                    ] = kf_observations[keypoint_i]
                    cov_mappoint_infos_list_index[tmp_map_point_index] += 1
                except:
                    raise BaseException("max_cov_mappoint_size 不够大")
                ros_keypoints.append(ros_keypoint)

            ros_keyframe = KeyFrame()
            ros_keyframe.mTimeStamp = float(ros_time_stamp)
            ros_keyframe.mnId = keyframe_i  # TODO 最好这部分也传回来
            ros_keyframe.pose_cw = ros_pose
            ros_keyframe.descriptors = []  # TODO descriptors
            ros_keyframe.key_points = ros_keypoints
            ros_keyframe.mvp_map_points_index = map_point_index.tolist()

            ros_keyframes.append(ros_keyframe)
            pass

        keyframe_time_e = time.time()
        keyframe_time = keyframe_time_e - keyframe_time_s
        print("keyframe time: {}".format(keyframe_time))

        map_point_time = 0
        map_point_time_s = time.time()
        ros_mappoints = []
        te = 0
        for map_point_i in range(map_point_xyz_data.shape[0]):
            print(
                "Process MapPoint: {} / {}".format(map_point_i, map_point_xyz_data.shape[0])
            )
            # point
            map_data = map_point_xyz_data[map_point_i]
            ros_point = Point(x=map_data[0], y=map_data[1], z=map_data[2])

            # observations
            ros_observations = []
            refer_keyframe_id = -1  # TODO 只用新数据就没有得到这个

            # cov_mappoint_infos = kf_map_point_uv[kf_map_point_uv[:, 1] == map_point_i]
            cov_mappoint_infos = cov_mappoint_infos_list[map_point_i]
            for ( info ) in cov_mappoint_infos:  # TODO 目前共视图缺乏2D点匹配uv，无法添加其他keyframe的Observation
                # if info is None:
                if np.isnan(info[0]):
                    te = te +1
                    break
                ros_observation = Observation()
                keyframe_id = int(info[0])
                refer_keyframe_id = keyframe_id  # TODO change
                keyframe = ros_keyframes[keyframe_id]
                refer_keypoint_index = keyframe.mvp_map_points_index.index(map_point_i)
                ros_observation.keyframe_id = keyframe_id
                ros_observation.refer_keypoint_index = refer_keypoint_index
                ros_observations.append(ros_observation)

            # 如果没有被任何一帧看到，跳过该mappoint，但是不能在这里跳过，因为会打乱keyframe的map point index
            # 所以只能去读取的地方continue
            # if refer_keyframe_id == -1:
            #     continue

            ros_mappoint = MapPoint()
            ros_mappoint.mnId = map_point_i
            ros_mappoint.point = ros_point
            ros_mappoint.num_obs = len(ros_observations)
            ros_mappoint.observations = ros_observations
            ros_mappoint.ref_keyframe_id = refer_keyframe_id

            ros_mappoints.append(ros_mappoint)
            pass

        map_point_time_e = time.time()
        map_point_time = map_point_time_e - map_point_time_s
        print("map point time: {}".format(map_point_time))

        pub_cloud_map.key_frames = ros_keyframes
        pub_cloud_map.map_points = ros_mappoints

        sum_time_e = time.time()
        sum_time = sum_time_e - sum_time_s
        print("sum time: {}".format(sum_time))
        print(te)
        return pub_cloud_map

if __name__ == '__main__':
    conf = '/home/birl/Udf-Edge/Udf-Edge/src/cap-edge/confs/base.conf'
    time_1 = time.time()
    runner = Runner(conf)
    runner.run()
    
