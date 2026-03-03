from torch.utils.data import Dataset
import numpy as np
import torch
import cv2


class TinyNerfDataset(Dataset):
    def __init__(self, directory):
        data = np.load(directory)
        self.images = torch.from_numpy(data['images']).float()
        self.poses = torch.from_numpy(data['poses']).float()
        self.focal = torch.from_numpy(data['focal']).float()

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        return (self.images[idx], self.poses[idx])
    
    def get(self, idx):
        return self.images[idx].numpy() * 255, self.poses[idx], self.focal
        


if __name__ == "__main__":
    tinynerf = TinyNerfDataset("tiny_nerf_data.npz")

    image, poses, focal = tinynerf.get(5)
    img_raw, _ = tinynerf[5]
    cv2.imwrite("showing.png", image)
    print("poses:: ", poses)
    print("focal:: ", focal)