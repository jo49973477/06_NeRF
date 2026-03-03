import torch
import numpy as np
import imageio
import matplotlib.pyplot as plt

from omegaconf import DictConfig, OmegaConf
import hydra

from config import MainConfig
from utils import *
from net import NeRF_MLP
from embedding import PositionalEncoder
from dataset import TinyNerfDataset


def render_image(model, H, W, focal, pose, cfg, chunk_size=4096, ):
    """
    새로운 카메라 포즈(pose)에서 본 3D 이미지를 렌더링하는 함수!
    chunk_size: 한 번에 GPU에 넣을 광선의 개수 (GPU 성능에 따라 조절)
    """

    embedder_pts = PositionalEncoder(L=10)
    embedder_views = PositionalEncoder(L=4)
    
    # 1. 새로운 카메라 위치에서 화면 전체(HxW)로 Ray 64만 개 쏘기
    rays_o, rays_d = get_rays(H, W, focal, pose)
    rays_o = rays_o.reshape(-1, 3).to("cuda")
    rays_d = rays_d.reshape(-1, 3).to("cuda")

    all_rgb = [] # 조각난 픽셀 색상들을 모을 바구니

    for i in range(0, rays_o.shape[0], chunk_size):

        batch_rays_o = rays_o[i : i + chunk_size]
        batch_rays_d = rays_d[i : i + chunk_size]

        # --- 훈련 스텝과 똑같은 렌더링 과정 ---
        pts, z_vals = sample_points_along_rays(
            batch_rays_o, batch_rays_d, cfg.near, cfg.far, cfg.n_samples
        )
        
        viewdirs = batch_rays_d / torch.norm(batch_rays_d, dim=-1, keepdim=True)
        viewdirs = viewdirs[..., None, :].expand(pts.shape)
        
        # 모델 내부의 인코더 사용
        pts_flat = embedder_pts(pts)
        dirs_flat = embedder_views(viewdirs)
        
        # 뇌세포 통과 및 볼륨 렌더링
        raw_outputs = model(pts_flat, dirs_flat)
        rgb_pred, _ = volume_rendering(raw_outputs, z_vals, batch_rays_d)
        
        # 결과물을 바구니에 담기 (CPU로 내려서 메모리 확보)
        all_rgb.append(rgb_pred.cpu())

    # 3. 🧩 조각난 픽셀들 다시 하나의 이미지로 조립하기!
    # (H*W, 3) 모양으로 합친 다음, 원래 해상도 (H, W, 3)으로 되돌림
    final_image = torch.cat(all_rgb, dim=0).reshape(H, W, 3)
    
    # 파이토치 텐서(0~1)를 넘파이 배열(0~255)로 변환
    final_image = (final_image.detach().numpy() * 255).astype(np.uint8)
    
    return final_image


def pose_spherical(theta, phi, radius):
    """
    theta: 가로 회전 각도 (0~360)
    phi: 세로 회전 각도 (위/아래 고도, 예: 위에서 내려다보기)
    radius: 물체와의 거리 (줌 인/아웃)
    """
    # [수학 도우미] 각도를 파이토치 4x4 행렬로 바꿔주는 람다 함수들
    trans_t = lambda t: torch.tensor([ # z-axis movement
        [1,0,0,0], [0,1,0,0], [0,0,1,t], [0,0,0,1]
    ], dtype=torch.float32)

    rot_phi = lambda phi: torch.tensor([ # x-axis rotation
        [1,0,0,0],
        [0, np.cos(phi), -np.sin(phi), 0],
        [0, np.sin(phi),  np.cos(phi), 0],
        [0,0,0,1]
    ], dtype=torch.float32)

    rot_theta = lambda th: torch.tensor([ # y-axis rotation
        [np.cos(th), 0, -np.sin(th), 0],
        [0, 1, 0, 0],
        [np.sin(th), 0,  np.cos(th), 0],
        [0, 0, 0, 1]
    ], dtype=torch.float32)

    # 1. 거리를 벌리고 -> 2. 고도를 맞추고 -> 3. 옆으로 돈다! (행렬 곱셈 @)
    c2w = trans_t(radius)
    c2w = rot_phi(phi / 180. * np.pi) @ c2w
    c2w = rot_theta(theta / 180. * np.pi) @ c2w

    # [주의 🚨] 좌표계 맞추기! 
    # 파이토치와 그래픽스(OpenGL)의 XYZ 축 방향이 달라서 한 번 뒤집어줘야 함!
    c2w = torch.tensor([[-1,0,0,0], [0,0,1,0], [0,1,0,0], [0,0,0,1]], dtype=torch.float32) @ c2w
    
    return c2w


if __name__ == "__main__":
# ---------------------------------------------------------
# 2. 🎬 촬영 및 비디오 굽기 (Action!)
# ---------------------------------------------------------

    hydra.initialize(version_base=None, config_path="conf")
    cfg = hydra.compose(config_name="config")
    raw_config = OmegaConf.to_container(cfg, resolve=True)
    main_cfg = MainConfig(**raw_config)

    model = NeRF_MLP()
    
    # 2) Lightning이 저장한 .ckpt 파일 불러오기 (경로/파일이름 확인 필수!)
    ckpt_path = "Tiny_NeRF/svrnnn3n/checkpoints/epoch=99-step=10600.ckpt" # <-- 여기 수정!
    checkpoint = torch.load(ckpt_path, weights_only=False)
    
    # 3) Lightning이 붙인 'model.' 꼬리표 떼어내기 (딕셔너리 정리)
    clean_state_dict = {}
    for k, v in checkpoint['state_dict'].items():
        if k.startswith('model.'):
            # 'model.' 6글자를 지운 원래 이름표로 복구!
            clean_state_dict[k.replace('model.', '')] = v
            
    # 4) 정리된 기억을 깡통 모델에 주입!
    model.load_state_dict(clean_state_dict)
    
    # 5) 실전 모드 ON: GPU로 올리고, Dropout 같은 거 끄기 (eval)
    model.to('cuda')
    model.eval()
    
    print("✅ 모델 가중치 로드 및 렌더링 준비 완료!")

    tinynerf = TinyNerfDataset("tiny_nerf_data.npz")
    image, test_pose, focal = tinynerf.get(100)
    img_rendered = render_image(model, H=100, W=100, focal=138.8889, cfg=main_cfg, pose=test_pose.to())

    # 화면에 출력!
    plt.imshow(img_rendered * 255)
    plt.title("NeRF Rendered Image")
    plt.show()

    frames = []

    # 0도부터 360도까지, 40장의 프레임(사진)을 찍을 거야
    for th in np.linspace(0., 360., 40, endpoint=False):
        
        # 카메라 위치 생성: 고도는 -30도(약간 위에서 내려다봄), 거리는 4.0
        pose = pose_spherical(th, -30., 4.0)
        
        # 아까 짰던 Inference 함수로 렌더링!
        # (H와 W, focal은 지휘관 데이터셋에 맞게 넣어주면 됨. 예: 100, 100)
        img = render_image(model, H=100, W=100, focal=138.8889, pose=pose, cfg=main_cfg, chunk_size=4096, )
        
        frames.append(img)
        print(f"📸 {th:.1f}도 렌더링 완료!")

    # 3. 📼 이미지들을 묶어서 .mp4 동영상으로 저장!
    # (imageio 패키지가 필요해: pip install imageio[ffmpeg])
    video_path = 'nerf_360_spin.mp4'
    imageio.mimwrite(video_path, frames, fps=30, quality=8)

    print(f"✨ 찢었다! [{video_path}] 비디오 저장 완료! ✨")