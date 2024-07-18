import os
from typing import Union, List

import torch
import numpy as np
import open3d as o3d
from sklearn.neighbors import NearestNeighbors
from torch.utils import data

from src import PointNet, TNet, New_PointNet
from train_utils import train_and_eval
from Data import POINTDataset
import transforms as T
import torch.nn as nn
from torchvision import models


class PresetEval:
    def __init__(self, num_samples):
        self.transforms = T.Compose([
            T.DownsampleV(voxel_size=0.05, target_num_points=num_samples),
            T.ToTensor()
        ])
    def __call__(self, point, target):
        return self.transforms(point, target)

def read_point_cloud(file_path):
    # 读取点云数据
    point_cloud = np.loadtxt(file_path, usecols=(0, 1, 2))
    return point_cloud

class DownsampleFPS(object):
    def __init__(self, num_samples):
        self.num_samples = num_samples

    def farthest_point_sample(self, points, npoint):
        N, C = points.shape
        centroids = np.zeros((npoint,), dtype=np.int32)
        distance = np.ones((N,)) * 1e10
        farthest = np.random.randint(0, N)
        for i in range(npoint):
            centroids[i] = farthest
            centroid = points[farthest, :]
            dist = np.sum((points - centroid) ** 2, axis=1)
            mask = dist < distance
            distance[mask] = dist[mask]
            farthest = np.argmax(distance)
        return centroids

    def __call__(self, point, bim):
        num_points = point.shape[0]
        num_bim = bim.shape[0]
        
        if num_points != self.num_samples:
            sampled_indices = self.farthest_point_sample(point, self.num_samples)
            point = point[sampled_indices, :]
            
        if num_bim != self.num_samples:
            bim_indices = self.farthest_point_sample(bim, self.num_samples)
            bim = bim[bim_indices, :]    
        return point, bim

class TwoPoint:
    def __init__(self, num_samples):
        self.transforms = T.Compose([
            T.DownsampleV(voxel_size=0.05, target_num_points=num_samples),
            T.ToTensor()
        ])
    def __call__(self, point, bim):
        return self.transforms(point, bim)

class DownsampleV(object):
    def __init__(self, voxel_size, target_num_points):
        self.voxel_size = voxel_size
        self.target_num_points = target_num_points

    def voxel_downsample(self, point_cloud):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_cloud)
        downsampled_pcd = pcd.voxel_down_sample(voxel_size=self.voxel_size)
        return np.asarray(downsampled_pcd.points)

    def __call__(self, points, ):
        num_points = points.shape[0]
        if num_points > self.target_num_points:
            indices = np.random.choice(num_points, self.target_num_points, replace=False)
            points = points[indices, :]
        elif num_points < self.target_num_points:
            indices = np.random.choice(num_points, self.target_num_points, replace=True)
            points = points[indices, :]
        return points

def estimate_normals(points, radius):
    # 创建一个Open3D PointCloud对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # 估计法线
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30))
    
    # 返回估计的法线
    normals = np.asarray(pcd.normals)
    return normals

def match_features(source_features, target_features, k=1):
    nn = NearestNeighbors(n_neighbors=k)
    nn.fit(target_features)
    distances, indices = nn.kneighbors(source_features)
    return distances, indices

def compute_initial_transformation(source_points, target_points, indices):
    max_iterations = 10000
    threshold = 0.005
    best_inliers = 1000
    best_transformation = np.eye(4)
    
    for _ in range(max_iterations):
        sample_indices = np.random.choice(len(indices), 3, replace=False)
        src_pts = source_points[sample_indices]
        tgt_pts = target_points[indices[sample_indices]]
        
        src_centroid = np.mean(src_pts, axis=0)
        tgt_centroid = np.mean(tgt_pts, axis=0)
        H = (src_pts - src_centroid).T @ (tgt_pts - tgt_centroid)
        U, S, Vt = np.linalg.svd(H)
        R_matrix = Vt.T @ U.T
        if np.linalg.det(R_matrix) < 0:
            Vt[-1, :] *= -1
            R_matrix = Vt.T @ U.T
        t = tgt_centroid.T - R_matrix @ src_centroid.T
        transformation = np.eye(4)
        transformation[:3, :3] = R_matrix
        transformation[:3, 3] = t
        
        transformed_src_pts = (R_matrix @ source_points.T).T + t
        distances = np.linalg.norm(transformed_src_pts - target_points[indices], axis=1)
        inliers = np.sum(distances < threshold)
        
        if inliers > best_inliers:
            best_inliers = inliers
            best_transformation = transformation
    
    return best_transformation

def execute_global_registration(source_points, target_points, source_features, target_features, voxel_size):
    # 创建Open3D点云对象
    source_pcd = o3d.geometry.PointCloud()
    target_pcd = o3d.geometry.PointCloud()
    
    source_pcd.points = o3d.utility.Vector3dVector(source_points)
    target_pcd.points = o3d.utility.Vector3dVector(target_points)
    
    # 将深度学习特征转换为Open3D特征对象
    source_fpfh = o3d.pipelines.registration.Feature()
    target_fpfh = o3d.pipelines.registration.Feature()
    
    source_fpfh.data = source_features.T
    target_fpfh.data = target_features.T
    
    distance_threshold = voxel_size * 1.5
    
    # 执行RANSAC配准
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_pcd, target_pcd, source_fpfh, target_fpfh, True, distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 3,
        [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
        ],
        o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    
    return result.transformation

def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    assert os.path.exists(args.weights), f"weights {args.weights} not found."

    # 读取数据并加载
    rtcd_file = "/home/ubuntu/PointNetfeature/POINT-TE/POINT-TE-Point/Point80.txt"
    rtcd_point_1 = read_point_cloud(rtcd_file)
    bim_file = "/home/ubuntu/PointNetfeature/POINT-TE/POINT-TE-Point/Point99.txt"
    bim_point_1 = read_point_cloud(bim_file)
    
    # 数据预处理（降采样）
    Data_pre = DownsampleV(voxel_size=5, target_num_points=2024)
    Data_sample = DownsampleV(voxel_size=10, target_num_points=2024)
    rtcd_point= Data_pre(rtcd_point_1)
    bim_point = Data_pre(bim_point_1)
    rtcd_sample= Data_sample(rtcd_point_1)
    bim_sample = Data_sample(bim_point_1)
    print(rtcd_point.shape, bim_point.shape)
    
   # 估计法线
    source_normals = estimate_normals(rtcd_point_1, radius=0.5*2)
    target_normals = estimate_normals(bim_point_1, radius=0.5*2)
    
    # 加载模型并调用预训练权重，作为pre_train
    model = PointNet()
    pre_model = New_PointNet()
    pretrain_weights = torch.load(args.weights, map_location='cpu')
    if "model" in pretrain_weights:
        model.load_state_dict(pretrain_weights["model"])
    else:
        model.load_state_dict(pretrain_weights)
        
    model_dict = model.state_dict()
    premodel_dict = pre_model.state_dict()
    
    premodel_dict = {k: v for k, v in model_dict.items() if k in premodel_dict}
    premodel_dict.update(premodel_dict)
    pre_model.load_state_dict(premodel_dict)

    pre_model.to(device)
    
    # 提取特征
    rtcd_f = train_and_eval.sub_feat(pre_model, rtcd_point, device)
    bim_f = train_and_eval.sub_feat(pre_model, bim_point, device)
    print("after sub FT shape is : ", rtcd_f.shape, bim_f.shape)
    
    
    
    # 特征融合
    # distances, indices = match_features(rtcd_point_1, bim_point_1)
    # print(indices.shape)
    
    # 初步配准
    # initial_transformation = compute_initial_transformation(rtcd_point_1, bim_point_1, indices.flatten())
    # print("11111",initial_transformation.shape)
    initial_transformation = execute_global_registration(rtcd_sample, bim_sample, rtcd_f, bim_f, 0.5)
    print(initial_transformation.shape)
    
    # 创建Open3D点云对象
    source_pcd = o3d.geometry.PointCloud()
    source_pcd.points = o3d.utility.Vector3dVector(rtcd_point_1)
    source_pcd.normals = o3d.utility.Vector3dVector(source_normals)

    target_pcd = o3d.geometry.PointCloud()
    target_pcd.points = o3d.utility.Vector3dVector(bim_point_1)
    target_pcd.normals = o3d.utility.Vector3dVector(target_normals)
    
    

    # ICP精细配准
    result_icp = o3d.pipelines.registration.registration_icp(
    source_pcd, target_pcd, 0.5 * 0.4, initial_transformation,
    o3d.pipelines.registration.TransformationEstimationPointToPlane())

    # 应用变换
    source_pcd.transform(result_icp.transformation)
    source_pcd.paint_uniform_color([1, 0.706, 0])  # BLUE
    target_pcd.paint_uniform_color([0, 0.651, 0.929])  # YELLOW
    # 可视化配准结果
    o3d.visualization.draw_geometries([source_pcd, target_pcd])
    
    
def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="pytorch u2net validation")

    parser.add_argument("--data-path", default="./", help="DUTS root")
    parser.add_argument("--weights", default="./save_weights/model_best.pth")
    parser.add_argument("--device", default="cuda:0", help="training device")
    parser.add_argument('--print-freq', default=10, type=int, help='print frequency')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)
