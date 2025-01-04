from scipy.spatial.transform import Rotation as spR
import numpy as np
import os
opj = os.path.join
import open3d as o3d
import torch
from lietorch import SE3
import droid_backends
from tqdm import tqdm
# custom
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


# ----------process data----------
def create_point_actor(points, colors=None):
    """
    Args:
        pointcloud (ndarray): Nx3
    Returns:
        pointcloud (ndarray): Nx3， downsampled
    """
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    # point_cloud.colors = o3d.utility.Vector3dVector(colors)
    return point_cloud

def down_sample_ratio(pcd, desired_num_points, bigger_voxel_size, smaller_voxel_size, tolerance_ratio=0.05):
    """ Use the method voxel_down_sample defined in open3d and do bisection iteratively 
        to get the appropriate voxel_size which yields the points with the desired number.
        
        Args:
            pcd                (ndarray): shape (n,3)
            desired_num_points (int    ): the desired number of points after down sampling
            bigger_voxel_size  (float  ): the initial bigger voxel size to do bisection
            smaller_voxel_size (float  ): the initial smaller voxel size to do bisection
            tolerance_ratio    (float  ): tolerance ratio of number of points, ratio of desired_num_points
        Returns:
            downsampled_pcd: down sampled points with the original data type

    """
    # bigger_voxel_size should be larger than smaller_voxel_size
    if bigger_voxel_size < smaller_voxel_size:
        bigger_voxel_size = int(10*smaller_voxel_size)

    assert len(pcd.points) > desired_num_points, "desired_num_points should be less than or equal to the num of points in the given array."
    
    pcd_less = pcd.voxel_down_sample(bigger_voxel_size)
    if len(pcd_less.points) >= desired_num_points:
        bigger_voxel_size = int(10*bigger_voxel_size)
    
    pcd_more = pcd.voxel_down_sample(smaller_voxel_size)
    if len(pcd_more.points) <= desired_num_points:
        smaller_voxel_size = int(0.1*smaller_voxel_size)

    midVoxelSize = (bigger_voxel_size + smaller_voxel_size) / 2.
    pcd_tmp = pcd.voxel_down_sample(midVoxelSize)
    count = 0
    while np.abs(len(pcd_tmp.points) - desired_num_points) > int(tolerance_ratio*desired_num_points):
        if len(pcd_tmp.points) < desired_num_points:
            bigger_voxel_size = midVoxelSize
        else:
            smaller_voxel_size = midVoxelSize
        midVoxelSize = (bigger_voxel_size + smaller_voxel_size) / 2.
        pcd_tmp = pcd.voxel_down_sample(midVoxelSize)
        print(count, midVoxelSize)
        count += 1

    return pcd_tmp

def downsample_voxel(pointcloud, voxel_size=0.05, do_ratio=False, ratio=0.1, tolerance_ratio=0.1):
    """ Down sample point cloud by voxelization

    Args:
        pointcloud (ndarray): Nx[x,y,z]
        voxel_size (float, optional): size of voxel.
        do_ratio   (bool): if downsample to a specify ratio of number of points
        raito      (float): ratio of number of points
        tolerance_ratio    (float  ): tolerance ratio of number of points, ratio of desired_num_points
    Returns:
        pointcloud (ndarray): Nx[x,y,z]
    """
    point_actor = create_point_actor(pointcloud)
    if not do_ratio:
        point_actor = point_actor.voxel_down_sample(voxel_size)
    elif do_ratio:
        point_actor = down_sample_ratio(point_actor, int(ratio*len(point_actor.points)), 
            0.01, 0.1, tolerance_ratio=tolerance_ratio)
    pointcloud = np.asarray(point_actor.points)
    return pointcloud

def construct_covis_graph(pointcloud, indexes, poses, depths, intrisics, intrisics_fs):
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

    idx_mappts = np.arange(len(pointcloud))
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
        table_visibility[idx_mappts_i, i] = True

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
    count = 0
    for i in range(len(id_new)):
        if observed_table[i]:
            id_new[i] = count
            count += 1

    pointcloud = pointcloud[observed_table]

    nd_idx_cam_idx_mpt_uv[:, 1] = id_new[ nd_idx_cam_idx_mpt_uv[:, 1].astype(np.int32) ]

    return pointcloud, nd_idx_cam_idx_mpt_uv

def extract_points_world(video, num_keyframes, calib, thresh_scale=2, device="cuda:0"):
    """ Extract 3D points in world frame """

    fx, fy, cx, cy = calib

    torch.cuda.set_device(device)
    extract_points_world.video = video
    extract_points_world.filter_thresh = 0.005 * thresh_scale

    list_pts = []
    list_duvs = []
    list_clrs = []
    with torch.no_grad():

        # convert poses to 4x4 matrix
        poses = video.poses
        disps = video.disps[:num_keyframes]
        Ps = SE3(poses).inv().matrix().cpu().numpy()

        images = video.images
        images = images.cpu()[:,[2,1,0],3::8,3::8].permute(0,2,3,1) / 255.0
        points = droid_backends.iproj(SE3(poses).inv().data, disps, video.intrinsics[0]).cpu()

        thresh = extract_points_world.filter_thresh * torch.ones_like(disps.mean(dim=[1,2]))
        
        cu_idx_keyframes = torch.arange(0, num_keyframes, device=device)

        count = droid_backends.depth_filter(
            video.poses, video.disps, video.intrinsics[0], cu_idx_keyframes, thresh)

        count = count.cpu()
        disps = disps.cpu()[:num_keyframes]
        masks = ((count >= 2) & (disps > .5*disps.mean(dim=[1,2], keepdim=True)))

        camera_matrix = np.array([ [fx, 0, cx], \
                                   [0, fy, cy], \
                                   [0, 0, 1] ])
        for i in range(num_keyframes):

            mask = masks[i].reshape(-1)
            pts_w = points[i].reshape(-1, 3)[mask].cpu().numpy()
            clr = images[i].reshape(-1, 3)[mask].cpu().numpy()
            list_pts.append(pts_w)

            # get uvs of pts
            pose_i = poses[i].cpu().numpy()
            tcw = pose_i[:3]
            Rcw = spR.from_quat(pose_i[3:]).as_matrix()
            pts_c = camera_matrix.dot( Rcw.dot(pts_w.T) + tcw.reshape([3,1]) ).T
            uv = pts_c[:,:2] / pts_c[:,-1].reshape([-1,1])
            list_duvs.append(np.column_stack([pts_c[:,-1].reshape([-1,1]), uv]))
            list_clrs.append(clr)
        #list_xyz = []
        #for i in range(num_keyframes):
        #    list_xyz.append(np.column_stack([list_pts[i].reshape([-1,3])]))
        #list_xyz = np.row_stack(list_xyz)
        #np.save("ceshi.npy",list_xyz)
        #pcd = o3d.geometry.PointCloud()
        #pcd.points = o3d.utility.Vector3dVector(list_xyz)
        #o3d.io.write_point_cloud("ceshi.ply", pcd)
    return list_pts, list_duvs, list_clrs

def extract_data(droid, args, graph, calib, tstamps):
    # Output for constructing ORBSLAM map
    idx_keyframe = droid.video.tstamp.cpu().numpy().astype(np.int32)
    num_keyframe = len(idx_keyframe[idx_keyframe>0])+1
    idx_keyframe = idx_keyframe[:num_keyframe] * args.stride # stride for recovering the original frame index
    tstamps_keyframe = np.array(tstamps, dtype=np.float64)[idx_keyframe//args.stride]
    images_keyframe = np.transpose(droid.video.images.cpu().numpy()[:num_keyframe], (0, 2, 3, 1))
    poses_keyframe = droid.video.poses.cpu().numpy()[:num_keyframe]
    depths_keyframe = np.reciprocal(droid.video.disps.cpu().numpy())[:num_keyframe]
    intrinsics_keyframe = droid.video.intrinsics.cpu().numpy()[:num_keyframe]

    intrinsics_fullsize = calib # TODO
    
    list_xyzs_w, list_duvs, list_clrs = extract_points_world(droid.video, num_keyframe, intrinsics_fullsize)
    list_index = []
    list_index_xyz_duv = []
    for i in range(num_keyframe):
        list_index.append(np.ones([len(list_xyzs_w[i]), 1])*idx_keyframe[i])
        list_index_xyz_duv.append( np.column_stack([list_index[i].reshape([-1,1]),\
                                                   list_xyzs_w[i].reshape([-1,3]),\
                                                   list_duvs[i].reshape([-1,3])
                                                   ])
                                )
    index_xyz_duv = np.row_stack(list_index_xyz_duv)
    colors = np.row_stack(list_clrs)

    graph_ijw = np.column_stack(graph)

    return idx_keyframe, tstamps_keyframe, images_keyframe, poses_keyframe, depths_keyframe, \
        intrinsics_keyframe, intrinsics_fullsize, index_xyz_duv, colors, graph_ijw


# ---------- data to message ----------
import time
def data_to_message(time_data, pose_data, kf_index_data, map_point_xyz_data, kf_map_point_uv):
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
            map_point_xyz_data.shape[0],
            time_data.shape[0],
            kf_map_point_uv.shape[1],
        ),
    )
    cov_mappoint_infos_list_index = np.full(
        fill_value=0, dtype=np.int32, shape=(map_point_xyz_data.shape[0])
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
        for (
            info
        ) in cov_mappoint_infos:  # TODO 目前共视图缺乏2D点匹配uv，无法添加其他keyframe的Observation
            # if info is None:
            if np.isnan(info[0]):
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

    return pub_cloud_map
