import numpy as np
import open3d as o3d

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

def create_point_actor(points, colors=None):
    """
    Args:
        pointcloud (ndarray): Nx3
    Returns:
        pointcloud (ndarray): Nx3, downsampled
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

if __name__ == '__main__':
    pointcloud = np.load('ceshi3.npy')
    pc_v = downsample_voxel(pointcloud, do_ratio=True, ratio=0.05, tolerance_ratio=0.05)
    np.save("ce2.npy",pc_v)
