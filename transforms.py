import random
import torch
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

    def __call__(self, point, target):
        num_points = 2292345
        if num_points > self.num_samples:
            indices = torch.randperm(num_points)[:self.num_samples]
            point = point[indices, :]
        elif num_points < self.num_samples:
            indices = torch.randint(0, num_points, (self.num_samples,))
            point = point[indices, :]
        return point, target

class Flaten(object):
    def __init__(self) -> None:
        pass
    def __call__(self, point, target):
        target = target.reshape(target.shape[0], -1)
        return point, target