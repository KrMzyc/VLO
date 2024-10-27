import numpy as np
import open3d as o3d

class LidarEncoder():
    def __init__(self, fov_up, fov_down, fov_left, fov_right, num_lasers: int, img_length: int):
        self.num_lasers = num_lasers
        self.img_length = img_length
        
        # 上下视场角度（单位：弧度）
        self.fov_up_rad = (fov_up / 180) * np.pi
        self.fov_down_rad = (fov_down / 180) * np.pi
        self.fov_rad = abs(self.fov_up_rad) + abs(self.fov_down_rad)
        
        # 左右视场角度（单位：弧度）
        self.fov_left_rad = (fov_left / 180) * np.pi
        self.fov_right_rad = (fov_right / 180) * np.pi
        self.h_fov_rad = abs(self.fov_left_rad) + abs(self.fov_right_rad)

    def get_u_v(self, point):
        assert point.shape[0] == 3  # XYZ

        x, y, z = point
        r = np.sqrt(x ** 2 + y ** 2 + z ** 2)  # range
        yaw = np.arctan2(y, x)  # 水平角度
        pitch = np.arcsin(z / r)  # 垂直角度

        # 限制水平角度在左右视场角范围内
        if yaw < -self.fov_left_rad or yaw > self.fov_right_rad:
            return None  # 超出水平视场角，不进行投影

        # 将 yaw 映射到 [0, 1] 范围
        v = (yaw + self.fov_left_rad) / self.h_fov_rad  # 水平归一化
        u = 1.0 - (pitch + abs(self.fov_down_rad)) / self.fov_rad  # 垂直归一化

        # 将归一化的 u, v 映射到图像大小
        v *= self.img_length
        u *= self.num_lasers

        # round and limit for use as index
        v = np.floor(v)
        v = min(self.img_length - 1, v)
        v = max(0.0, v)
        pixel_v = int(v)  # col (水平)

        u = np.floor(u)
        u = min(self.num_lasers - 1, u)
        u = max(0.0, u)
        pixel_u = int(u)  # row (垂直)

        return pixel_u, pixel_v, r


    def estimate_normals(self, pointcloud):
        pointcloud_xyz = pointcloud[:, :-1]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pointcloud_xyz)
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=4))  # PCA w/ knn

        normals = np.asarray(pcd.normals)
        return normals

    def __call__(self, pointcloud):
        assert pointcloud.shape[1] == 4  # XYZI
        N = pointcloud.shape[0]
        projection = np.zeros((self.num_lasers, self.img_length, 8))  # feature_channels = 8
        normals = self.estimate_normals(pointcloud)  # estimate normals

        # Create image projection
        for i in range(N):
            point = pointcloud[i, :-1]  # grab XYZ
            intensity = pointcloud[i, -1]
            normal = normals[i, :]  # xyz normals
            result = self.get_u_v(point)
            if result is not None:  # 只对视场内的点进行投影
                pixel_u, pixel_v, r = result
                projection[pixel_u, pixel_v] = np.concatenate(([point[0], point[1], point[2], r, intensity], normal))

        return projection
