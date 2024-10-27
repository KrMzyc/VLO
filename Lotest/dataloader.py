import numpy as np
import os
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

class Kitti(Dataset):
    def __init__(self, scan_dir, pose_dir, transform, test_sequence=None, split='train'):
        self.scan_dir = scan_dir # lidar scans
        self.pose_dir = pose_dir # ground-truth poses
        self.split = split
        self.transform = transform
        self.projected = {} # dict to store transformed scans
        
        if self.split == 'train':
            self.sequence_idx = '07' # TODO: make train/val data include multiple sequences instead??
        elif self.split == 'validate':
            self.sequence_idx = '04'
        elif self.split == 'test':
            self.sequence_idx = test_sequence
            
        self.velo_files = self.load_velo_files(self.sequence_idx)
        self.poses = self.load_poses(self.sequence_idx)

    def __len__(self):
        return len(self.velo_files)
    
    def __repr__(self):
        return "Total frames: {}, total poses: {} in {} sequence {}".format(len(self.velo_files),
                                                                    len(self.poses), self.split, self.sequence_idx)
    
    def __getitem__(self, index: int):
        if index == 0:
            prev_index = 0
        else:
            prev_index = index - 1

        curr_scan, prev_scan = self.load_velo(index), self.load_velo(prev_index) # velo scans
        
        # if scans already projected, grab from memory; else transform scans and add to memory
        if index in self.projected.keys():
            curr_img = self.projected[index]
        else:
            curr_img = self.transform(curr_scan)
            self.projected[index] = curr_img
            
        if prev_index in self.projected.keys():
            prev_img = self.projected[prev_index]
        else:
            prev_img = self.transform(prev_scan)
            self.projected[prev_index] = prev_img
        
        # grab poses and compute relative pose
        curr_pose, prev_pose = self.poses[index], self.poses[prev_index]
        rel_pose = np.linalg.inv(prev_pose) @ curr_pose 
        return curr_img, prev_img, rel_pose
    
    def load_velo_files(self, seq_idx):
        sequence_dir = os.path.join(self.scan_dir, seq_idx, 'velodyne_points', 'data')
        sequence_files = sorted(os.listdir(sequence_dir))
        velo_files = [os.path.join(sequence_dir, frame) 
                              for frame in sequence_files]
        return velo_files
    
    def load_velo(self, item: int): 
        """Load velodyne [x,y,z,i] scan data from binary files."""
        filename = self.velo_files[item]
        scan = np.fromfile(filename, dtype=np.float32)
        scan = scan.reshape((-1,4)) 
        return scan
    
    def load_poses(self, sequence):
        pose_file = os.path.join(self.pose_dir, sequence + '.txt')
        poses = [] # store 4x4 pose matrices
        try:
            with open(pose_file, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    pose_vector = np.fromstring(line, dtype=float, sep=' ')
                    pose_matrix = pose_vector.reshape(3, 4)
                    pose_matrix = np.vstack((pose_matrix, [0, 0, 0, 1]))
                    poses.append(pose_matrix)      
        except FileNotFoundError:
            print('Ground truth poses are not available for sequence ' +
                  sequence + '.')
        return poses  
    
    # def visualise(self, index: int):
    #     img, _, _ = self.__getitem__(index)
    #     img = img.permute(1,2,0).numpy()
    #     img_intensity = img[:,:,4] * (255.0 / img[:,:,4].max()) # [xyz range intensity normals]
    #     img_range = img[:,:,3] * (255.0 / img[:,:,3].max())
        
    #     fig, axs = plt.subplots(2, figsize=(12,6), dpi=100)
    #     axs[0].imshow(img_intensity)
    #     axs[0].set_title("Intensity map")
    #     axs[1].imshow(img_range) # invert normalize TODO: invert or not??
    #     axs[1].set_title("Depth map")
    #     plt.show()

    def visualise(self, index: int):
        img, _, _ = self.__getitem__(index)
        img = img.permute(1, 2, 0).numpy()

        # 获取强度图像和深度图像
        img_intensity = img[:, :, 4] * (255.0 / img[:, :, 4].max())  # [xyz range intensity normals]
        img_range = img[:, :, 3]

        # 增强深度图亮度（通过平方根或对数变换）
        img_range_normalized = img_range / img_range.max()  # 归一化
        img_range_enhanced = np.sqrt(img_range_normalized)  # 通过平方根增强较小的深度值
        img_range_enhanced *= 255.0 / img_range_enhanced.max()  # 将增强后的深度图重新映射到 0-255 范围

        img_range = img[:,:,3] * (255.0 / img[:,:,3].max())

        # 可视化强度图和增强后的深度图
        fig, axs = plt.subplots(3, figsize=(12, 6), dpi=100)
        axs[0].imshow(img_intensity)
        axs[0].set_title("Intensity map")

        axs[1].imshow(img_range)  # 显示增强后的深度图
        axs[1].set_title("Depth map")

        axs[2].imshow(img_range_enhanced)  # 显示增强后的深度图
        axs[2].set_title("Enhanced Depth map")
        plt.show()
