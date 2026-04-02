import os
from pathlib import Path
import yaml

from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import torch
import cv2


class TinyNerfDataset(Dataset):
    def __init__(self, directory, detail = False):
        self.directory = directory
        data = np.load(directory)
        self.images = torch.from_numpy(data['images']).float()
        self.poses = torch.from_numpy(data['poses']).float()
        self.focal = torch.from_numpy(data['focal']).float()
        self.detail = detail
        
        
    def get_K(self):
        H, W, C = self.images[0].numpy().shape
        
        return np.array([[self.focal, 0, W/2], 
                          [0, self.focal, H/2], 
                          [0, 0, 1]], dtype=np.float64)


    def __len__(self):
        return len(self.images)
    
    
    def __getitem__(self, idx):
        
        if self.detail:
             return (self.images[idx], self.poses[idx], self.get_K())
        else:            
            return (self.images[idx], self.poses[idx])
    
    
    def get(self, idx):
        
        if self.detail:
            return self.images[idx].numpy() * 255, self.poses[idx], self.get_K()
        else:
            return self.images[idx].numpy() * 255, self.poses[idx], self.focal
    
    
    def initialise_images_for_colmap(self, path):
        os.makedirs(path, exist_ok=True)
        
        for i in range(len(self)):
            image_raw = self.get(i)[0]
            img = cv2.cvtColor(image_raw, cv2.COLOR_BGR2RGB)
            cv2.imwrite(os.path.join(path, f"{i:03d}.png"), img.astype(np.uint8))
            
        return path
            



class TLessDataset(Dataset):
    def __init__(self, directory, number = 1, detail = False):
        number_folder = f"{number:02d}"
        
        self.image_dir = os.path.join(directory, number_folder, "rgb")
        gt_dir = os.path.join(directory, number_folder, "gt.yml")
        info_dir = os.path.join(directory, number_folder, "info.yml")
        
        self.image_paths = list(Path(self.image_dir).glob("*.png"))
        self.image_paths.sort()
        
        with open(info_dir, "r") as f:
            self.info = yaml.safe_load(f)
        
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])
        
        self.detail = detail
        
    
    def initialise_images_for_colmap(self, path):
        return self.image_dir
    
    
    def __len__(self):
        return len(self.image_paths)
    
    
    def proj_mat(self, cam_K):
        np.array(cam_K).reshape(3, 3)
    
    
    def get(self, idx):
        poses = np.zeros((4, 4), dtype= np.float64)
        
        cam_R_w2c = np.array(self.info[idx]["cam_R_w2c"]).reshape(3, 3)
        cam_t_w2c = np.array(self.info[idx]["cam_t_w2c"])
        
        poses[:3, :3] = cam_R_w2c
        poses[:3, 3] = cam_t_w2c
        poses[3, 3] = 1
        
        cam_K = self.info[idx]["cam_K"]
        focal = cam_K[0]
        
        img = cv2.imread(self.image_paths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if self.detail:
            return img, poses, np.array(cam_K).reshape(3, 3)
        else:
            return img, poses, focal
    
    
    def __getitem__(self, index):
        raw_img, poses, focal = self.get(index)
        
        if self.detail:
            return raw_img, poses, focal
        else:
            return raw_img, poses
        
        
        
def tless_test():
    
    tinynerf = TLessDataset("/home/yeongyoo/03_Dataset/01_t-less_v2/test_kinect/", number = 5, detail= True)

    image, poses, focal = tinynerf.get(5)
    img_raw, poses, focal = tinynerf[5]
    cv2.imwrite("showing.png", image)
    print(img_raw.shape)
    print("poses:: ", poses)
    print("focal:: ", focal)
    
    
def tinynerf_test():
    tinynerf = TinyNerfDataset("tiny_nerf_data.npz", detail= True)

    image, poses, focal = tinynerf.get(10)
    img_raw, _, _ = tinynerf[10]
    print(img_raw.shape)
    cv2.imwrite("showing.png", image)
    print("poses:: ", poses)
    print("focal:: ", focal)

if __name__ == "__main__":
    tless_test()