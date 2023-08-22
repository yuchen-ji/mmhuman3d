import os, sys
import shutil
import numpy as np    
import matplotlib.pyplot as plt
import cv2
import mmcv
import pickle

def process_h36m(dirname):
    split = ['S1', "S5", "S6", "S7", "S8", "S9", "S11"]
    
    for each in split:  
        fnames = os.listdir(os.path.join(dirname, each))
        fnames = filter(lambda x: x.endswith(".jpg"), fnames)
        categeries = []
        for imname in fnames:
            cate_dir = os.path.join(dirname, each, imname[:-11])
            if imname[:-11] not in categeries:
                categeries.append(imname[:-11])
                if not os.path.exists(cate_dir):
                    os.mkdir(cate_dir)
                    
            src = os.path.join(dirname, each, imname)
            dst = os.path.join(cate_dir, imname)
            shutil.move(src, dst)
            
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
        
def img2video():
    mmcv.frames2video('workspace/demo/input_results', 'test.avi', fps=10, filename_tmpl='{:06d}.png')

if __name__ == "__main__":
    # dirname = 'data/datasets/h36m'
    # process_h36m(dirname)
    
    # joints = np.array([[100, 100, 1]])
    # heatmap_size = np.array([128, 128])
    # image_size = np.array([128, 128])
    # heatmap = gen_heatmap(heatmap_size, image_size, joints)
    
    # data = np.load('data/preprocessed_datasets/pw3d_test.npz', allow_pickle=True)
    # smpl = data['smpl']
    # image_path = data['image_path']
    
    # with open('data/datasets/pw3d/sequenceFiles/test/downtown_arguing_00.pkl', 'rb') as f:
    #     data = pickle.load(f)
    # print(data)
    
    img2video()