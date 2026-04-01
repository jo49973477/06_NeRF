import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb

import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from omegaconf import DictConfig, OmegaConf
import hydra

from dataset import TinyNerfDataset, TLessDataset
from config import MainConfig
from net import NeRF_MLP
from embedding import PositionalEncoder
from utils import get_rays, sample_points_along_rays, volume_rendering


class NeRFLightning(pl.LightningModule): # 이름부터 NeRF로 바꾸자!
    def __init__(self, model, cfg, dataset):
        super().__init__()
        self.model = model
        self.cfg = cfg
        self.dataset = dataset
        self.focal = dataset.focal
        self.img_size = dataset[0][0].shape[0]
        self.embedder_pts = PositionalEncoder(L=10)
        self.embedder_views = PositionalEncoder(L=4)
        
        # model은 너무 커서 하이퍼파라미터 로깅에서 빼는 게 맞아. 잘했어!
        self.save_hyperparameters(ignore=['model'])

    def forward(self, x, d):
        return self.model(x, d)

    def training_step(self, batch, batch_idx):
        img, pose = batch
        img = img[0]
        pose = pose[0]

        # 1. 광선 쏘기 및 Batch 추출
        ray_center, ray_direction = get_rays(self.img_size, self.img_size, self.focal, pose)
        ray_center = ray_center.reshape(-1, 3)
        ray_direction = ray_direction.reshape(-1, 3)
        target_rgb = img.reshape(-1, 3)

        rand_idx = torch.randperm(ray_center.shape[0])[:self.cfg.n_rand]
        batch_ray_center = ray_center[rand_idx]
        batch_ray_direction = ray_direction[rand_idx]
        batch_target = target_rgb[rand_idx]

        # 2. Ray Marching & Positional Encoding
        pts, z_vals = sample_points_along_rays(
            batch_ray_center, batch_ray_direction, self.cfg.near, self.cfg.far, self.cfg.n_samples
        )
    
        viewdirs = batch_ray_direction / torch.norm(batch_ray_direction, dim=-1, keepdim=True)
        viewdirs = viewdirs[..., None, :].expand(pts.shape) 
        
        # (주의: embedder 인스턴스는 클래스 내부나 외부에서 제대로 주입되어야 함!)
        pts_flat = self.embedder_pts(pts)         
        dirs_flat = self.embedder_views(viewdirs) 

        # 3. 인공지능 통과 (nerf_model이 아니라 self.model!)
        raw_outputs = self.model(pts_flat, dirs_flat)

        # 4. 잃어버린 볼륨 렌더링 되찾기! (DDPM 코드가 있던 자리)
        # 여기서 최종 픽셀의 예측 색상(rgb_pred)을 뽑아내야 해!
        rgb_pred, _ = volume_rendering(raw_outputs, z_vals, batch_ray_direction)

        # 5. NeRF의 Loss는 예측 색상과 진짜 색상의 차이!
        loss = F.mse_loss(rgb_pred, batch_target)
        
        # 6. PL의 마법: 이거 한 줄이면 WandB, TensorBoard에 그래프 다 그려짐!
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        return loss 

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.cfg.lr)

        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999)

        return [optimizer], [scheduler]
    
if __name__ == "__main__":
    # setting the configuration parameters
    hydra.initialize(version_base=None, config_path="conf")
    cfg = hydra.compose(config_name="tless_config")
    raw_config = OmegaConf.to_container(cfg, resolve=True)
    main_cfg = MainConfig(**raw_config)


    trainset = TinyNerfDataset(main_cfg.directory) if main_cfg.dataset == "tiny" else TLessDataset(main_cfg.directory)
    
    
    trainloader = DataLoader(trainset, 
                             batch_size=1, 
                             shuffle=True, 
                             num_workers=2)
    
    # preparing device
    model = NeRF_MLP()
    
    # 2. LightningModule 생성 
    model = NeRFLightning(model=model, cfg=main_cfg, dataset = trainset)

    # 3. WandB 로거 연결
    wandb_logger = WandbLogger(
        project="TLess_NeRF",
        name="take001"
    )

    # 4. 트레이너(Trainer) 소환! ⚡️
    # 여기가 핵심이야. GPU 관리, 에폭 관리 다 얘가 해줌.
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/",                 
        filename="nerf-{epoch:02d}-{train_loss:.4f}", 
        monitor="train_loss",                   
        mode="min",                             
        save_top_k=3,                           
        save_last=True,                         
    )

    trainer = pl.Trainer(
        max_epochs=main_cfg.epochs,
        accelerator="auto",      # GPU 있으면 쓰고, 없으면 CPU 씀 (똑똑함)
        devices=1,               # GPU 개수
        logger=wandb_logger,     # WandB 연결
        log_every_n_steps=10     # 10 스텝마다 로그 기록
    )

    print("🚀 Lightning 학습 시작!")
    trainer.fit(model, train_dataloaders=trainloader)