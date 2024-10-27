import numpy as np
from tqdm.notebook import tqdm

class PointCloudSampler():
    def __init__(self, output_n: int):
        self.output_n = output_n
    
    def sample_index(self, xyz): 
        """ Returns sample indices for pointcloud """
        N = xyz.shape[0]
        centroids = np.zeros(self.output_n)
        distance = np.ones(N) * 1e10
        farthest = np.random.randint(0, N)
        print("Sampling pointclouds ...")
        for i in tqdm(range(self.output_n)):
            # Update the i-th farthest point
            centroids[i] = farthest
            # Take the xyz coordinate of the farthest point
            centroid = xyz[farthest, :]
            # Calculate the Euclidean distance from all points in the point set to this farthest point
            dist = np.sum((xyz - centroid) ** 2, -1)
            # Update distances to record the minimum distance of each point in the sample from all existing sample points
            mask = dist < distance
            distance[mask] = dist[mask]
            # Find the farthest point from the updated distances matrix, and use it as the farthest point for the next iteration
            farthest = np.argmax(distance, -1)
        return centroids.astype(int)
    
    def __call__(self, pointcloud):
        assert pointcloud.shape[1] == 4 # XYZI
        xyz = pointcloud[:,:-1] # extract xyz
        centroids = self.sample_index(xyz)
        
        sampled_pointcloud = pointcloud[centroids, :]
        
        return sampled_pointcloud