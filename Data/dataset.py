import os
import numpy as np
import torch.utils.data as data
import transforms as T

class POINTDataset(data.Dataset):
    def __init__(self, root: str, train: bool = True, transforms=None):
        assert os.path.exists(root), f"path '{root}' does not exist."
        if train:
            self.point_root = os.path.join(root, "POINT-TR", "POINT-TR-Point")
            self.mask_root = os.path.join(root, "POINT-TR", "POINT-TR-Mask")
        else:
            self.point_root = os.path.join(root, "POINT-TE", "POINT-TE-Point")
            self.mask_root = os.path.join(root, "POINT-TE", "POINT-TE-Mask")
        assert os.path.exists(self.point_root), f"path '{self.point_root}' does not exist."
        assert os.path.exists(self.mask_root), f"path '{self.mask_root}' does not exist."
        

        point_names = [p for p in os.listdir(self.point_root) if p.endswith(".txt")]
        mask_names = [p for p in os.listdir(self.mask_root) if p.endswith(".txt")]
        assert len(point_names) > 0, f"not find any images in {self.point_root}."
        re_mask_names = []
        for p in point_names:
            mask_name = p.replace(".txt", ".txt")
            assert mask_name in mask_names, f"{p} has no corresponding mask."
            re_mask_names.append(mask_name)
        mask_names = re_mask_names

        self.point_path = [os.path.join(self.point_root, n) for n in point_names]
        self.masks_path = [os.path.join(self.mask_root, n) for n in mask_names]

        self.transforms = transforms

    def __getitem__(self, idx):
        point_path = self.point_path[idx]
        mask_path = self.masks_path[idx]
        point = np.loadtxt(point_path)
        assert point is not None, f"failed to read image: {point_path}"
        target = np.loadtxt(mask_path)
        assert target is not None, f"failed to read mask: {mask_path}"

        if self.transforms is not None:
            point, target = self.transforms(point, target)
        return point, target

    def __len__(self):
        return len(self.point_path)

    @staticmethod
    def collate_fn(batch):
        points, targets = list(zip(*batch))
        batched_imgs = cat_list(points, fill_value=0)
        batched_targets = cat_list(targets, fill_value=0)

        return batched_imgs, batched_targets


class ConstractiveDataset(data.Dataset):
    def __init__(self, root: str, train: bool = True, transforms=None):
        assert os.path.exists(root), f"path '{root}' does not exist."
        if train:
            self.point_root = os.path.join(root, "POINT-TR", "POINT-TR-Point")
            self.bim_root = os.path.join(root, "POINT-TR", "POINT-TR-Bim")
            self.mask_root = os.path.join(root, "POINT-TR", "POINT-TR-Mask")
        else:
            self.point_root = os.path.join(root, "POINT-TE", "POINT-TE-Point")
            self.bim_root = os.path.join(root, "POINT-TE", "POINT-TE-Bim")
            self.mask_root = os.path.join(root, "POINT-TE", "POINT-TE-Mask")
        assert os.path.exists(self.point_root), f"path '{self.point_root}' does not exist."
        assert os.path.exists(self.bim_root), f"path '{self.bim_root}' does not exist."
        assert os.path.exists(self.mask_root), f"path '{self.mask_root}' does not exist."
        

        point_names = [p for p in os.listdir(self.point_root) if p.endswith(".txt")]
        bim_names = [p for p in os.listdir(self.bim_root) if p.endswith(".txt")]
        mask_names = [p for p in os.listdir(self.mask_root) if p.endswith(".txt")]
        assert len(point_names) > 0, f"not find any images in {self.point_root}."
        re_bim_names = []
        re_mask_names = []
        for p in point_names:
            bim_name = p.replace(".txt", ".txt")
            assert bim_name in bim_name, f"{p} has no corresponding mask."
            mask_name = p.replace(".txt", ".txt")
            assert mask_name in mask_names, f"{p} has no corresponding mask."
            re_bim_names.append(bim_name)
            re_mask_names.append(mask_name)
        bim_names = re_bim_names
        mask_names = re_mask_names

        self.point_path = [os.path.join(self.point_root, n) for n in point_names]
        self.bim_path = [os.path.join(self.bim_root, n) for n in bim_names]
        self.masks_path = [os.path.join(self.mask_root, n) for n in mask_names]

        self.transforms = transforms

    def __getitem__(self, idx):
        point_path = self.point_path[idx]
        bim_path = self.bim_path [idx]
        mask_path = self.masks_path[idx]
        point = np.loadtxt(point_path)
        assert point is not None, f"failed to read image: {point_path}"
        bim = np.loadtxt(bim_path)
        assert bim is not None, f"failed to read image: {bim_path}"
        target = np.loadtxt(mask_path)
        assert target is not None, f"failed to read mask: {mask_path}"

        if self.transforms is not None:
            point, bim, target = self.transforms(point, bim, target)
        return point, bim, target

    def __len__(self):
        return len(self.point_path)

    @staticmethod
    def collate_fn(batch):
        points, bim, targets = list(zip(*batch))
        batched_imgs = cat_list(points, fill_value=0)
        batched_bims = cat_list(bim, fill_value=0)
        batched_targets = cat_list(targets, fill_value=0)

        return batched_imgs, batched_bims, batched_targets


def cat_list(points, fill_value=0):
    max_size = tuple(max(s) for s in zip(*[img.shape for img in points]))
    batch_shape = (len(points),) + max_size
    batched_imgs = points[0].new(*batch_shape).fill_(fill_value)
    for img, pad_img in zip(points, batched_imgs):
        pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
    return batched_imgs


class PresetTrain:
    def __init__(self, num_samples):
        self.transforms = T.Compose([
            T.DownsampleV(voxel_size=0.05, target_num_points=num_samples),
            T.ToTensor(),
            T.Flaten()
        ])

    def __call__(self, pot, target):
        return self.transforms(pot, target)


class PresetEval:
    def __init__(self):
        self.transforms = T.Compose([
            T.DownsampleV(voxel_size=0.05, target_num_points=2024),
            T.ToTensor(),
            T.Flaten()
        ])

    def __call__(self, pot, target):
        return self.transforms(pot, target)


if __name__ == '__main__':
    train_dataset = POINTDataset('./', train=True, transforms=PresetTrain(num_samples=2024))
    print("train data len is: ", len(train_dataset))
    val_dataset = POINTDataset('./', train=False, transforms=PresetEval())
    print("val data len is: ", len(val_dataset))
    i, t = train_dataset[0]
    print("after downsample shape is: ", i.shape, t.shape)
    # train_dataset = ConstractiveDataset('./', train=True, transforms=PresetTrain(num_samples=2024))
    # print("train data len is: ", len(train_dataset))
    # val_dataset = ConstractiveDataset('./', train=False, transforms=PresetEval())
    # print("val data len is: ", len(val_dataset))

    # i, b, t = train_dataset[0]
    # print("after downsample shape is: ", i.shape, b.shape, t.shape)
