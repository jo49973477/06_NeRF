import os

import torch.optim as optim
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pytorch_msssim import ssim

from gsplat import rasterization
from gsplat.strategy import DefaultStrategy

from omegaconf import DictConfig, OmegaConf
import hydra
from tqdm import tqdm
import wandb

from utils import get_projmat_from_K
from config import GSConfig
from dataset import *


class GaussianSplatting:
    
    def __init__(self, cfg, dataloader: DataLoader, strategy = None):
        self.num_points = cfg.num_points
        self.means = nn.Parameter(torch.rand(self.num_points, 3).cuda())      # 위치 (x, y, z)
        self.scales = nn.Parameter(torch.rand(self.num_points, 3).cuda())    # 크기 (가로, 세로, 높이)
        self.quats = nn.Parameter(torch.rand(self.num_points, 4).cuda())      # 회전 (쿼터니언)
        self.opacities = nn.Parameter(torch.rand(self.num_points).cuda())     # 불투명도 (알파값)
        self.colors = nn.Parameter(torch.rand(self.num_points, 3).cuda())     # 색상 (RGB 또는 SH 계수)
        
        self.max_steps = cfg.max_steps
        self.camera_width = cfg.camera_width
        self.camera_height = cfg.camera_height
        
        
        self.lambda_ssim = cfg.ssim_coefficient
        self.dataloader = dataloader
        
        self.optimizer = optim.Adam([self.means, self.scales, self.quats, self.opacities, self.colors], 
                                    lr= cfg.lr)
        
        self.strategy = DefaultStrategy() if strategy is None else strategy
        self.strategy_state = self.strategy.initialize_state()
        
        self.save_path = cfg.save_directory
        
        
        wandb.init(
            project="3DGS-TEST", 
            name= cfg.name,
            config=vars(cfg) 
        )
    
    
    def make_image(self, viewmat, projmat):
        rendered_image, _ = rasterization(
            means= self.means,
            quats= self.quats,
            scales= self.scales,
            opacities= self.opacities,
            colors= self.colors,
            viewmats= viewmat,
            projmats= projmat,
            image_width= self.camera_width,
            image_height= self.camera_height,
        )
        
        return rendered_image

    def save_weights(self):
        
        os.makedirs(self.save_path, exist_ok=True)
        save_path = os.path.join(self.save_path, "orange_latest.pt")
        
        torch.save({
            'means': self.means.detach().cpu(),
            'scales': self.scales.detach().cpu(),
            'quats': self.quats.detach().cpu(),
            'opacities': self.opacities.detach().cpu(),
            'colors': self.colors.detach().cpu(),
        }, save_path)
        
        print(f"💾 [저장 완료] 3090의 땀방울이 {save_path} 에 안전하게 봉인되었습니다!!")
    
    def training_step(self, batch, global_step):
        
        img, pose, focal_matrix = batch
        gt_img = img.float().cuda()
        pose = pose.float().cuda()
        focal_matrix = focal_matrix.float().cuda()
        
        scales_act = torch.exp(self.scales)
        quats_act = self.quats / self.quats.norm(dim=-1, keepdim=True)
        opacities_act = torch.sigmoid(self.opacities)
        colors_act = torch.sigmoid(self.colors)
        
        pred_img, _, render_info = rasterization(
            means= self.means,
            quats= quats_act,
            scales= scales_act,
            opacities= opacities_act,
            colors= colors_act,
            viewmats= pose,
            Ks= focal_matrix,
            width= self.camera_width,
            height= self.camera_height,
            packed=True,
        )
        
        pred_img_perm = pred_img.permute(0, 3, 1, 2)
        gt_img_perm = gt_img.permute(0, 3, 1, 2)
        
        loss_l1 = torch.abs(pred_img - gt_img).mean()
        loss_ssim = 1.0 - ssim(pred_img_perm, gt_img_perm, data_range=1.0, size_average=True)
        
        loss = (1- self.lambda_ssim) * loss_l1 + self.lambda_ssim * loss_ssim
            
        render_info["means2d"].retain_grad()
        
        
        loss.backward()
        
        params_dict = {
            "means": self.means,
            "scales": self.scales,
            "quats": self.quats,
            "opacities": self.opacities,
            "colors": self.colors
        }
        
        self.strategy.step_post_backward(
            params= params_dict,
            optimizers={
                "means": self.optimizer,
                "scales": self.optimizer,
                "quats": self.optimizer,
                "opacities": self.optimizer,
                "colors": self.optimizer
            }, 
            state=self.strategy_state, 
            step=global_step, 
            info=render_info,
            packed=True,
        )
        
        self.means = params_dict["means"]
        self.scales = params_dict["scales"]
        self.quats = params_dict["quats"]
        self.opacities = params_dict["opacities"]
        self.colors = params_dict["colors"]
        
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        return pred_img[0], loss.item()
        
    
    def train(self):
        
        pbar = tqdm(range(self.max_steps), desc="3DGS Training Epochs")
        global_step = 0
        
        for epoch in pbar:
            for batch_idx, batch in enumerate(self.dataloader):
                rendered_iamge, loss = self.training_step(batch, global_step)
                wandb.log({
                    "train/loss": loss,
                    "train/step": global_step
                })
                global_step += 1
            
            
            pbar.set_description(f"Rendering... Loss: {loss:.5f}")
            img_to_log = rendered_iamge.detach().cpu().numpy()
            wandb.log({"render/image": wandb.Image(img_to_log, caption=f"Step {epoch}")})
                
        self.save_weights()
                
if __name__ == "__main__":
    
    hydra.initialize(version_base=None, config_path="conf_gsplat")
    cfg = hydra.compose(config_name="tiny_config")
    raw_config = OmegaConf.to_container(cfg, resolve=True)
    main_cfg = GSConfig(**raw_config)


    trainset = TinyNerfDataset(main_cfg.directory, detail = True) if main_cfg.dataset == "tiny" else TLessDataset(main_cfg.directory, detail = True)
    
    
    trainloader = DataLoader(trainset, 
                             batch_size=1, 
                             shuffle=True, 
                             num_workers=4)
    
    # preparing device
    gs = GaussianSplatting(main_cfg, trainloader)
    gs.train()
