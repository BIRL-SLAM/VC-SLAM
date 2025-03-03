import torch
import numpy as np
import os
from scipy.spatial import cKDTree
import numba as nb
import time
import multiprocessing as mp
from pathos.multiprocessing import ProcessingPool as Pool

# def for_speed(QUERY_EACH,POINT_NUM_GT,pointcloud,POINT_NUM,sigmas):
#     sample = []
#     sample_near = []
#     time_s = time.time()
#     print(QUERY_EACH)
#     for i in range(QUERY_EACH):
#         scale = 0.25 if 0.25 * np.sqrt(POINT_NUM_GT / 20000) < 0.25 else 0.25 * np.sqrt(POINT_NUM_GT / 20000)
#         tt = pointcloud + scale*np.expand_dims(sigmas,-1) * np.random.normal(0.0, 1.0, size=pointcloud.shape)
#         sample.append(tt) #*
#         tt = tt.reshape(-1,POINT_NUM,3)

#         sample_near_tmp = []
#         time_a2 = 0
#         for j in range(tt.shape[0]):
#             time_s1 = time.time()
#             nearest_idx = search_nearest_point(torch.tensor(tt[j]).float().cpu(), torch.tensor(pointcloud).float().cpu())
#             time_e1 = time.time()
#             time_a1 = time_e1 - time_s1
#             time_a2 += time_a1
#             nearest_points = pointcloud[nearest_idx]
#             nearest_points = np.asarray(nearest_points).reshape(-1,3)
#             sample_near_tmp.append(nearest_points)
#         print("sum time: {}".format(time_a2))
#         sample_near_tmp = np.asarray(sample_near_tmp)
#         sample_near_tmp = sample_near_tmp.reshape(-1,3)
#         sample_near.append(sample_near_tmp) #*
#     time_e = time.time()
#     time_a = time_e - time_s
#     print("sum time: {}".format(time_a))
#     sample = np.asarray(sample) #*
#     sample_near = np.asarray(sample_near) #*

def for_speed(POINT_NUM_GT,pointcloud,sigmas,sample,sample_near,POINT_NUM):
    scale = 0.25 if 0.25 * np.sqrt(POINT_NUM_GT / 20000) < 0.25 else 0.25 * np.sqrt(POINT_NUM_GT / 20000)
    tt = pointcloud + scale*np.expand_dims(sigmas,-1) * np.random.normal(0.0, 1.0, size=pointcloud.shape)
    sample.append(tt) #*
    tt = tt.reshape(-1,POINT_NUM,3)

    sample_near_tmp = []
    for j in range(tt.shape[0]):
        nearest_idx = search_nearest_point(torch.tensor(tt[j]).float().cpu(), torch.tensor(pointcloud).float().cpu())
        nearest_points = pointcloud[nearest_idx]
        nearest_points = np.asarray(nearest_points).reshape(-1,3)
        sample_near_tmp.append(nearest_points)
    sample_near_tmp = np.asarray(sample_near_tmp)
    sample_near_tmp = sample_near_tmp.reshape(-1,3)
    sample_near.append(sample_near_tmp) #*
    return sample, sample_near

def search_nearest_point(point_batch, point_gt):
    num_point_batch, num_point_gt = point_batch.shape[0], point_gt.shape[0]
    point_batch = point_batch.unsqueeze(1).repeat(1, num_point_gt, 1)
    point_gt = point_gt.unsqueeze(0).repeat(num_point_batch, 1, 1)

    distances = torch.sqrt(torch.sum((point_batch-point_gt) ** 2, axis=-1) + 1e-12) 
    dis_idx = torch.argmin(distances, axis=1).detach().cpu().numpy()

def process_data(data_dir, dataname):

    pointcloud = np.loadtxt(os.path.join(data_dir, dataname) + '.xyz')

    shape_scale = np.max([np.max(pointcloud[:,0])-np.min(pointcloud[:,0]),np.max(pointcloud[:,1])-np.min(pointcloud[:,1]),np.max(pointcloud[:,2])-np.min(pointcloud[:,2])])
    shape_center = [(np.max(pointcloud[:,0])+np.min(pointcloud[:,0]))/2, (np.max(pointcloud[:,1])+np.min(pointcloud[:,1]))/2, (np.max(pointcloud[:,2])+np.min(pointcloud[:,2]))/2]
    pointcloud = pointcloud - shape_center
    pointcloud = pointcloud / shape_scale

    POINT_NUM = pointcloud.shape[0] // 60 #"//"向下取整 原始点云数量/60，再取整
    POINT_NUM_GT = pointcloud.shape[0] // 60 * 60 #取整后*60
    QUERY_EACH = 1000000//POINT_NUM_GT #？

    point_idx = np.random.choice(pointcloud.shape[0], POINT_NUM_GT, replace = False) #从[0,原始数据点]中取POINT_NUM_GT个数，组成一维数组
    pointcloud = pointcloud[point_idx,:]
    ptree = cKDTree(pointcloud)
    sigmas = []
    for p in np.array_split(pointcloud,100,axis=0): #将数量拆成100组
        d = ptree.query(p,51) #ckdtree查询
        sigmas.append(d[0][:,-1])
    
    sigmas = np.concatenate(sigmas)
    sample = []
    sample_near = []
    #for_speed(QUERY_EACH,POINT_NUM_GT,pointcloud,POINT_NUM,sigmas)
    time_s = time.time()
    pool = Pool(2)
    sample = pool.map(for_speed,[POINT_NUM_GT]*4,[pointcloud]*4,[sigmas]*4,[sample]*4,[sample_near]*4,[POINT_NUM]*4)
    # for i in range(QUERY_EACH):
    #     scale = 0.25 if 0.25 * np.sqrt(POINT_NUM_GT / 20000) < 0.25 else 0.25 * np.sqrt(POINT_NUM_GT / 20000)
    #     tt = pointcloud + scale*np.expand_dims(sigmas,-1) * np.random.normal(0.0, 1.0, size=pointcloud.shape)
    #     sample.append(tt) #*
    #     tt = tt.reshape(-1,POINT_NUM,3)

    #     sample_near_tmp = []
    #     for j in range(tt.shape[0]):
    #         nearest_idx = search_nearest_point(torch.tensor(tt[j]).float().cpu(), torch.tensor(pointcloud).float().cpu())
    #         nearest_points = pointcloud[nearest_idx]
    #         nearest_points = np.asarray(nearest_points).reshape(-1,3)
    #         sample_near_tmp.append(nearest_points)
    #     sample_near_tmp = np.asarray(sample_near_tmp)
    #     sample_near_tmp = sample_near_tmp.reshape(-1,3)
    #     sample_near.append(sample_near_tmp) #*
    time_e = time.time()
    time_a = time_e - time_s
    print("sum time: {}".format(time_a))
    sample = np.asarray(sample) #*
    sample_near = np.asarray(sample_near) #*

if __name__ == '__main__':
    process_data('/zclin/perfect_test/','test')