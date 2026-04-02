import os
import torch.optim as optim
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pytorch_msssim import ssim
import numpy as np

from gsplat import rasterization
from gsplat.strategy import DefaultStrategy
from sklearn.neighbors import NearestNeighbors
import pycolmap
from omegaconf import DictConfig, OmegaConf
import hydra
from tqdm import tqdm
import wandb
import cv2

from utils import get_projmat_from_K
from config import GSConfig
from dataset import *


class GaussianSplatting:
    def __init__(self, cfg, dataloader: DataLoader, strategy=None, colmap_image_path="colmap_image_path", random = False):
        path = dataloader.dataset.initialise_images_for_colmap(colmap_image_path)
        if random:
            means_tensor = (torch.rand(num_points, 3) - 0.5) * 2.0 
            colors_tensor = torch.rand(num_points, 3)
        else:
            means_tensor, colors_tensor = self.get_colmap(path)
        
        # [수정 1] 색상 초기화 시 Logit 적용 (시그모이드 역함수)
        # 0~1 사이의 값을 clamp로 안전하게 만든 뒤 logit을 씌워야 렌더링 시 원래 색이 나옵니다.
        colors_tensor = torch.logit(colors_tensor.clamp(1e-6, 1-1e-6))
        
        points_np = means_tensor.detach().cpu().numpy()
        nn_engine = NearestNeighbors(n_neighbors=4, n_jobs=-1).fit(points_np)
        distances, _ = nn_engine.kneighbors(points_np)
        dist_avg = distances[:, 1:].mean(axis=1)
        dist_avg_tensor = torch.from_numpy(dist_avg).float().cuda()
        scales_tensor = torch.log(dist_avg_tensor * 1.0).unsqueeze(-1).repeat(1, 3)
        
        self.max_steps = cfg.max_steps
        self.camera_width = cfg.camera_width
        self.camera_height = cfg.camera_height
        
        num_points = len(means_tensor)
        
        self.param_dict = {
            "means": nn.Parameter(means_tensor),
            "colors": nn.Parameter(colors_tensor),
            "scales": nn.Parameter(scales_tensor),
            "quats": nn.Parameter(torch.rand(num_points, 4).cuda()),
            # [수정 2] 투명도 역시 Logit 적용
            "opacities": nn.Parameter(torch.logit(torch.rand(num_points).clamp(1e-6, 1-1e-6)).cuda())
        }
        
        lrs = {
            "means": 1.6e-4,    
            "scales": 1e-3,    
            "quats": 1e-3,      
            "opacities": 1e-2,  
            "colors": 2.5e-3    
        }
        
        
        # .items()로 정상 순회하도록 보장
        self.optimizers = {k: optim.Adam([v], lr=lrs[k], eps=1e-15) for k, v in self.param_dict.items()}
        
        self.lambda_ssim = cfg.ssim_coefficient
        self.dataloader = dataloader
        
        self.strategy = DefaultStrategy() if strategy is None else strategy if strategy is None else strategy
        
        self.strategy_state = self.strategy.initialize_state()
        
        self.save_path = cfg.save_directory
        
        wandb.init(
            project="3DGS-TEST", 
            name=cfg.name,
            config=vars(cfg) 
        )
    
    def get_colmap(self, path, output_dir="colmap_output"):
        sparse_path = os.path.join(output_dir, "sparse")
        points_bin_path = os.path.join(sparse_path, "points3D.bin")
        points_txt_path = os.path.join(sparse_path, "points3D.txt")
        
        # 1. 🚀 핵심: 이미 완성된 COLMAP 결과가 있는지 얌체처럼 확인하기
        if os.path.exists(points_bin_path) or os.path.exists(points_txt_path):
            print(f"😎 [개이득] 이미 완성된 COLMAP 데이터를 발견했습니다! 지루한 연산을 건너뜁니다: {sparse_path}")
            reconstruction = pycolmap.Reconstruction(sparse_path)
            
        else:
            # 2. 저장된 파일이 없다면 눈물을 머금고 처음부터 연산
            print("⏳ [안내] 저장된 COLMAP 데이터가 없습니다. 뼈 빠지는 연산을 시작합니다...")
            os.makedirs(output_dir, exist_ok=True)
            
            database_path = os.path.join(output_dir, "database.db")
            if os.path.exists(database_path):
                os.remove(database_path) # 이전 실행 찌꺼기 방지
                
            pycolmap.extract_features(database_path, path)
            pycolmap.match_exhaustive(database_path)
            
            maps = pycolmap.incremental_mapping(database_path, path, output_dir)
            
            if len(maps) < 1:
                raise ValueError("😱[ERROR] SfM으로 backbone을 만드는 데 실패했어요.")
            
            # 연산이 끝난 소중한 결과를 다음번을 위해 파일로 저장해 둡니다.
            os.makedirs(sparse_path, exist_ok=True)
            maps[0].write(sparse_path)
            reconstruction = maps[0]
            
        # 3. 3D 포인트 및 색상 추출 (기존과 동일)
        points3d = reconstruction.points3D.values()
        
        xyz = np.array([p.xyz for p in points3d])
        rgb = np.array([p.color for p in points3d]) / 255.0
    
        means = torch.from_numpy(xyz).float()
        colors = torch.from_numpy(rgb).float()
        print(f"🎉[SUCCEEDED] SfM feature point {len(means)}개를 성공적으로 로드했습니다!")
        
        return means.cuda(), colors.cuda()
    
    
    def save_weights(self):
        os.makedirs(self.save_path, exist_ok=True)
        save_path = os.path.join(self.save_path, "tless_latest.pt")
        
        # [수정 3] 언패킹 에러 방지를 위해 .items() 추가
        torch.save({k: v.detach().cpu() for k, v in self.param_dict.items()}, save_path)
        
        print(f"💾 [저장 완료] 3090의 땀방울이 {save_path} 에 안전하게 봉인되었습니다!!")
    
    
    def get_viewmat(self, c2w):
        # 1. 텐서 복사 (원본 손상 방지)
        c2w_opencv = c2w.clone()
        
        # 2. 🚨 핵심: OpenGL(TinyNeRF) -> OpenCV(gsplat) 카메라 축 변환
        # Y축(1번 열)과 Z축(2번 열)의 부호를 반대로 뒤집어줍니다.
        c2w_opencv[..., :3, 1] *= -1.0
        c2w_opencv[..., :3, 2] *= -1.0
        
        # 3. 이후는 동일하게 역행렬(W2C) 계산
        R = c2w_opencv[..., :3, :3]
        T = c2w_opencv[..., :3, 3:4]
        
        R_inv = R.transpose(-2, -1)
        T_inv = -torch.bmm(R_inv, T)
        
        viewmat = torch.zeros(c2w.shape[0], 4, 4, device=c2w.device, dtype=c2w.dtype)
        viewmat[..., 3, 3] = 1.0
        viewmat[..., :3, :3] = R_inv
        viewmat[..., :3, 3:4] = T_inv
        
        return viewmat
    
    
    def training_step(self, batch, global_step):
        img, pose, focal_matrix = batch
        gt_img = img.float().cuda()
        
        # [방어 코드] 혹시라도 이미지가 0~255 스케일로 들어온다면 0~1로 정규화
        if gt_img.max() > 2.0:
            gt_img = gt_img / 255.0

        pose = pose.float().cuda()
        focal_matrix = focal_matrix.float().cuda()
        
        scales_act = torch.exp(self.param_dict["scales"])
        quats_act = self.param_dict["quats"] / self.param_dict["quats"].norm(dim=-1, keepdim=True)
        opacities_act = torch.sigmoid(self.param_dict["opacities"])
        colors_act = torch.sigmoid(self.param_dict["colors"])
        
        # pose = self.get_viewmat(pose)
        
        # [수정 5] Tiny NeRF 데이터셋을 위한 흑색 배경 명시
        
        if self.param_dict["means"].shape[0] == 0:
            print(f"\n[💥 비상 탈출] {global_step} 스텝에서 점이 0개가 되었습니다!!")
            return gt_img[0], loss.item() # 엔진으로 넘기지 않고 강제 리턴!
        
        pred_img, _, render_info = rasterization(
            means=self.param_dict["means"],
            quats=quats_act,
            scales=scales_act,
            opacities=opacities_act,
            colors=colors_act,
            viewmats=pose,
            Ks=focal_matrix,
            width=self.camera_width,
            height=self.camera_height,
            packed=True,
        )
        
        self.strategy.step_pre_backward(
            params=self.param_dict,
            optimizers=self.optimizers, 
            state=self.strategy_state, 
            step=global_step, 
            info=render_info,
        )
        
        pred_img_perm = pred_img.permute(0, 3, 1, 2)
        gt_img_perm = gt_img.permute(0, 3, 1, 2)
        
        loss_l1 = torch.abs(pred_img - gt_img).mean()
        loss_ssim = 1.0 - ssim(pred_img_perm, gt_img_perm, data_range=1.0, size_average=True)
        
        loss = (1 - self.lambda_ssim) * loss_l1 + self.lambda_ssim * loss_ssim
            
        render_info["means2d"].retain_grad()
        loss.backward()
        
        if torch.isnan(loss):
            print(f"\n[💥 비상] {global_step} 스텝에서 NaN이 발생했습니다! 학습을 즉시 건너뜁니다.")
            self.optimizers.zero_grad() # 오염된 값 초기화
            return gt_img[0], 0.0
        
        self.strategy.step_post_backward(
            params=self.param_dict,
            optimizers=self.optimizers, 
            state=self.strategy_state, 
            step=global_step, 
            info=render_info,
            packed=True,
        )
        
        for opt in self.optimizers.values():
            opt.step()
            opt.zero_grad()
        
        return pred_img[0], loss.item()
        
    def train(self):
        # [수정 6] Epoch 기준이 아닌 Step 기준으로 루프 변경
        pbar = tqdm(range(self.max_steps), desc="3DGS Training Steps")
        global_step = 0
        
        # 데이터로더를 무한 반복 가능한 이터레이터로 변환
        dataloader_iter = iter(self.dataloader)
        
        while global_step < self.max_steps:
            try:
                batch = next(dataloader_iter)
            except StopIteration:
                dataloader_iter = iter(self.dataloader)
                batch = next(dataloader_iter)

            rendered_image, loss = self.training_step(batch, global_step)
            
            # 매 스텝 로깅하면 느려지므로 10 스텝마다 기록
            if global_step % 10 == 0:
                wandb.log({
                    "train/loss": loss,
                    "train/step": global_step
                })
                pbar.set_description(f"Rendering... Loss: {loss:.5f}")
            
            # 이미지는 500 스텝마다 로깅 (메모리 절약)
            if global_step % 500 == 0:
                img_to_log = rendered_image.detach().cpu().clamp(0, 1).numpy()
                wandb.log({"render/image": wandb.Image(img_to_log, caption=f"Step {global_step}")})
                
            global_step += 1
            pbar.update(1)
                
        self.save_weights()
                
if __name__ == "__main__":
    hydra.initialize(version_base=None, config_path="conf_gsplat")
    cfg = hydra.compose(config_name="tless_config")
    raw_config = OmegaConf.to_container(cfg, resolve=True)
    main_cfg = GSConfig(**raw_config)

    trainset = TinyNerfDataset(main_cfg.directory, detail=True) if main_cfg.dataset == "tiny" else TLessDataset(main_cfg.directory, detail=True)
    
    trainloader = DataLoader(trainset, 
                             batch_size=1, 
                             shuffle=True, 
                             num_workers=0)
    
    gs = GaussianSplatting(main_cfg, trainloader)
    gs.train()