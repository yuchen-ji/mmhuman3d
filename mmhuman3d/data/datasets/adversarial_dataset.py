import numpy as np
from torch.utils.data import Dataset

from .builder import DATASETS, build_dataset
import torch
import cv2

# 20.02.18, add 2d keypoints heatmap
def gen_heatmap(heatmap_size, image_size, joints):
    # print("image_size: ", image_size)
    
    sigma = 2
    tmp_size = sigma * 3
    num_joints = len(joints)
    
    # target: [num_joints, heatmap_size, heatmap_size]
    target = np.zeros((num_joints, heatmap_size[0], heatmap_size[1]), dtype=np.float32)
    # target_weight: [num_joints, 1] (1-dim, 1:visible, 0:invisible)
    # target_weight = np.ones((num_joints,1), dtype=np.float32)
    target_conf = np.ones((num_joints, 1), dtype=np.float32)
    target_conf[:, 0] = joints[..., 2]
    
    # 生成一张图像的heatmap
    for joint_id in range(num_joints):
        feat_stride = image_size / heatmap_size
        # 高斯核中心在热力图上的位置
        mu_x = int(joints[joint_id][0] / feat_stride[0] + 0.5)
        mu_y = int(joints[joint_id][1] / feat_stride[1] + 0.5)
        # gaussian range, ul:left-top, br:right-down
        ul = [int(mu_x-tmp_size), int(mu_y-tmp_size)]
        br = [int(mu_x+tmp_size+1), int(mu_y+tmp_size+1)]
        # 排除热力图部分全部在图像之外的点，并将该点的vis设置为0
        if ul[0] >= heatmap_size[0] or ul[1] >= heatmap_size[1] or br[0] < 0 or br[1] < 0:
            target_conf[joint_id] = 0
            continue
        
        # gaussian kernel size
        size = 2 * tmp_size + 1
        x = np.arange(0, size, 1, np.float32)
        y = x[:, np.newaxis]
        x0 = y0 = size // 2
        # guassian distribution: z = e^(-((x-x0)^2+(y-y0)^2) / (2*σ^2)）
        g = np.exp(-((x-x0)**2+(y-y0)**2) / (2*sigma**2) ) 
        g_x = max(0, -ul[0]), min(br[0], heatmap_size[0]) - ul[0]
        g_y = max(0, -ul[1]), min(br[1], heatmap_size[1]) - ul[1]
        # image range
        img_x = max(ul[0], 0) , min(br[0], image_size[0])
        img_y = max(ul[1], 0) , min(br[1], image_size[1])
                
        # select confident joints
        v = target_conf[joint_id]
        if v > 0.5:
            target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
                
    return target, target_conf


@DATASETS.register_module()
class AdversarialDataset(Dataset):
    """Mix Dataset for the adversarial training in 3D human mesh estimation
    task.

    The dataset combines data from two datasets and
    return a dict containing data from two datasets.
    Args:
        train_dataset (:obj:`Dataset`): Dataset for 3D human mesh estimation.
        adv_dataset (:obj:`Dataset`): Dataset for adversarial learning.
    """

    def __init__(self, train_dataset: Dataset, adv_dataset: Dataset):
        super().__init__()
        self.train_dataset = build_dataset(train_dataset)
        self.adv_dataset = build_dataset(adv_dataset)
        self.num_train_data = len(self.train_dataset)
        self.num_adv_data = len(self.adv_dataset)

    def __len__(self):
        """Get the size of the dataset."""
        return self.num_train_data

    def __getitem__(self, idx: int):
        """Given index, get the data from train dataset and randomly sample an
        item from adversarial dataset.

        Return a dict containing data from train and adversarial dataset.
        """
        data = self.train_dataset[idx]
        
        # By Yuchen, 23.02.18, add 2d keypoints heatmap
        heatmap_size = np.array([56, 56])
        image_size = np.array([data['img'].shape[1], data['img'].shape[2]])
        joints = data['keypoints2d'][:].numpy()
        heatmap, heatmap_conf = gen_heatmap(heatmap_size, image_size, joints)

        # # visualize heatmap, By Yuchen, 23.04.08, 生成的heatmap和图中的关键点位置符合
        # tmpmap = np.mean(heatmap, axis=0)
        # am = np.amax(tmpmap)
        # tmpmap /= am / 255
        # tmpmap = cv2.applyColorMap(np.uint8(tmpmap), cv2.COLORMAP_HSV)
        # cv2.imwrite('/workspaces/mmhuman3d/workspace/demo/222.png', tmpmap)
        # img_data = data['img'].numpy() * 255
        # cv2.imwrite('/workspaces/mmhuman3d/workspace/demo/gt.png', img_data.transpose(1,2,0))

        data['heatmap2d'] = torch.from_numpy(heatmap).float()
        data['heatmap2d_conf'] = torch.from_numpy(heatmap_conf)
        
        adv_idx = np.random.randint(low=0, high=self.num_adv_data, dtype=int)
        adv_data = self.adv_dataset[adv_idx]
        for k, v in adv_data.items():
            data['adv_' + k] = v
        return data
