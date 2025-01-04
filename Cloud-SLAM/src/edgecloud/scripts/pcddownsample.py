import open3d as o3d
 
# 读取文件
pcd = o3d.io.read_point_cloud("/data/xjluo/Download/data/bun000.ply")  # path为文件路径
 
pcd_new = o3d.geometry.PointCloud.uniform_down_sample(pcd, 30)
# value值的设定为整数，即value个点中选取一个点

o3d.visualization.draw_geometries([pcd_new])
