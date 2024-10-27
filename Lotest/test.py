from dataloader import *
from lidarenc import *
from pcsampler import *
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

def main():
    fov_up = 2.5 # deg
    fov_down = -19.8
    fov_left = 60
    fov_right = 60
    num_lasers = 192 # H
    img_length = 648 # W

    lidar_transform = transforms.Compose([LidarEncoder(fov_up, fov_down, fov_left, fov_right, num_lasers, img_length), 
                                      transforms.ToTensor()])
    scan_dir = '/home/krmzyc/Downloads/kitti'
    pose_dir = '/home/krmzyc/Downloads/kitti/poses'
    train_data = Kitti(scan_dir, pose_dir, transform=lidar_transform)
    val_data = Kitti(scan_dir, pose_dir, split='validate', transform=lidar_transform)

    train_dataloader = DataLoader(dataset=train_data, batch_size=2, shuffle=True, drop_last=True)
    val_dataloader = DataLoader(dataset=val_data, batch_size=2, drop_last=True)

    train_data.visualise(100)

if __name__ == "__main__":
    main()

