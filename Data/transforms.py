import random
import numpy as np
import torch
import open3d as o3d
from torchvision.transforms import functional as F
from torchvision.transforms import transforms as T


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, point, target=None):
        for t in self.transforms:
            point, target = t(point, target)
        return point, target


class ToTensor(object):
    def __call__(self, point, target):
        point = F.to_tensor(point)
        target = F.to_tensor(target)
        return point, target


class Normalize(object):
    def __call__(self, point, target):
        centroid = torch.mean(point, dim=0)
        point = point - centroid
        max_distance = torch.max(torch.sqrt(torch.sum(point**2, dim=1)))
        point = point / max_distance
        return point, target        


class Downsample(object):
    def __init__(self, num_samples):
        self.num_samples = num_samples

    def __call__(self, points, target):
        num_points = points.shape[0]
        if num_points > self.num_samples:
            indices = torch.randperm(num_points)[:self.num_samples]
            points = points[indices, :]
        elif num_points < self.num_samples:
            indices = torch.randint(0, num_points, (self.num_samples,))
            points = points[indices, :]
        return points, target

class Flaten(object):
    def __init__(self) -> None:
        pass
    def __call__(self, point, target):
        target = target.reshape(target.shape[0], -1)
        return point, target
    
    

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

    def __call__(self, point, target):
        num_points = point.shape[0]
        if num_points != self.num_samples:
            sampled_indices = self.farthest_point_sample(point, self.num_samples)
            point = point[sampled_indices, :]
        return point, target
    

class DownsampleFPS_gpu(object):
    def __init__(self, num_samples):
        self.num_samples = num_samples

    def farthest_point_sample(self, xyz, npoint):
        device = xyz.device
        xyz = xyz.squeeze(1)  
        print(xyz.shape)
        B, N, C = xyz.shape  # shape is torch.Size([1, 828861, 3])
        centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
        distance = torch.ones(B, N).to(device) * 1e10
        farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
        batch_indices = torch.arange(B, dtype=torch.long).to(device)
        for i in range(npoint):
            centroids[:, i] = farthest
            centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
            dist = torch.sum((xyz - centroid) ** 2, -1)
            mask = dist < distance
            distance[mask] = dist[mask]
            farthest = torch.max(distance, -1)[1]
        return centroids

    def __call__(self, point, target):
        num_points = point.shape[0]
        if num_points != self.num_samples:
            points_tensor = point.unsqueeze(0).float().to('cuda')  # Add batch dimension and move to GPU
            sampled_indices = self.farthest_point_sample(points_tensor, self.num_samples)
            point = points_tensor[0, sampled_indices].cpu()  # Remove batch dimension and move to CPU
        return point, target
    
class DownsampleV(object):
    def __init__(self, voxel_size, target_num_points):
        self.voxel_size = voxel_size
        self.target_num_points = target_num_points

    def voxel_downsample(self, point_cloud):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_cloud)
        downsampled_pcd = pcd.voxel_down_sample(voxel_size=self.voxel_size)
        return np.asarray(downsampled_pcd.points)

    def __call__(self, points, target):
        num_points = points.shape[0]
        if num_points > self.target_num_points:
            indices = np.random.choice(num_points, self.target_num_points, replace=False)
            points = points[indices, :]
        elif num_points < self.target_num_points:
            indices = np.random.choice(num_points, self.target_num_points, replace=True)
            points = points[indices, :]
        return points, target