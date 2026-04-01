import torch
import torch.nn.functional as F
import numpy as np

def get_rays(H, W, focal, c2w):
    device = c2w.device
    
    i, j = torch.meshgrid(
        torch.arange(W, dtype=torch.float32, device=device), 
        torch.arange(H, dtype=torch.float32, device=device), 
        indexing='xy'
    )
    
    dirs = torch.stack([
        (i - W * 0.5) / focal,
        -(j - H * 0.5) / focal,  # 그래픽스에서 Y축은 보통 위로 가니까 뒤집어줌 (-)
        -torch.ones_like(i)      # 카메라는 -Z 방향을 바라본다 (OpenGL 기준)
    ], dim=-1) # dirs의 모양: (H, W, 3)
    
    rays_d = torch.sum(dirs[..., None, :] * c2w[:3, :3], dim=-1)
    
    rays_o = c2w[:3, 3].expand(rays_d.shape)
    
    # 결과물: 출발점 배열 (H, W, 3), 방향 배열 (H, W, 3)
    return rays_o, rays_d



def sample_points_along_rays(rays_o, rays_d, near, far, N_samples, perturb=True):
    device = rays_o.device
    """
    rays_o: 광선의 출발점들 [N_rays, 3]
    rays_d: 광선의 방향들 [N_rays, 3]
    near, far: 카메라에서 얼마나 떨어진 곳부터/까지 찍을 것인가 (float)
    N_samples: 광선 하나당 점을 몇 개 찍을 것인가 (예: 64)
    perturb: 점을 찍는 위치를 랜덤하게 흔들 것인가? (True/False)
    """
    
    t_vals = torch.linspace(near, far, N_samples, device=device) # 모양: [N_samples]
    
    # 2. 모든 Ray에 똑같은 t 값을 복사 (아까 배운 expand!)
    # 광선이 N_rays개 있으니까, t_vals도 N_rays개만큼 복사해줌.
    z_vals = t_vals.expand(rays_o.shape[0], N_samples).clone() # 모양: [N_rays, N_samples]
    
    # 3. 🎲 Stratified Sampling (랜덤 흔들기 - NeRF의 핵심!)
    if perturb:
        # 구간의 절반 정도씩 앞뒤로 경계를 구함
        mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = torch.cat([mids, z_vals[..., -1:]], -1)
        lower = torch.cat([z_vals[..., :1], mids], -1)
        
        # 각 구간 안에서 랜덤한 위치를 하나씩 뽑음 (Uniform Distribution)
        t_rand = torch.rand(z_vals.shape, device=device)
        z_vals = lower + (upper - lower) * t_rand
        
    # 4. 실제 3D 좌표 계산: P = O + t * D
    # PyTorch의 브로드캐스팅(None 추가) 마법을 써서 차원을 맞춘 뒤 한 방에 계산!
    # rays_o[:, None, :] -> [N_rays, 1, 3]
    # rays_d[:, None, :] -> [N_rays, 1, 3]
    # z_vals[:, :, None] -> [N_rays, N_samples, 1]
    
    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
    
    # 결과물: 3D 점들의 좌표 pts [N_rays, N_samples, 3], 그 점들의 거리 z_vals [N_rays, N_samples]
    return pts, z_vals



def volume_rendering(raw_outputs, z_vals, rays_d):
    """
    raw_outputs -> 모델을 돌린 결과물 [:, 4]
    z_vals -> 점들의 거리 [N_rays, N_samples]
    rays_d -> ray들의 방향 벡터
    """
    dists = z_vals[:, 1:] - z_vals[:, :-1]
    last_dist = torch.tensor([1e10], device=raw_outputs.device).expand(dists[..., :1].shape)
    dists = torch.cat([dists, last_dist], dim=-1)
    dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

    rgb = torch.sigmoid(raw_outputs[..., :3])
    sigma = F.relu(raw_outputs[..., 3])

    alpha = 1. - torch.exp(- sigma * dists)
    transmittance = torch.cumprod(
        torch.cat([torch.ones((alpha.shape[0], 1), device=alpha.device), 1.0 - alpha + 1e-10], dim=-1), 
        dim=-1
    )[:, :-1]

    # 5. ⚖️ 가중치(Weight) 계산 및 최종 색상 덧셈(Sum)
    # 진짜 내 눈에 들어오는 영향력 = 내 눈까지 온 투과율(T) * 그 지점의 불투명도(alpha)
    weights = alpha * transmittance # [N_rays, N_samples]
    
    # 가중치와 RGB를 곱해서 64개의 점을 1개의 픽셀 색상으로 다 더해버림! (적분 끝!)
    pixel_color = torch.sum(weights[..., None] * rgb, dim=-2) # [N_rays, 3]

    return pixel_color, weights

def get_projmat_from_K(K, W, H, znear=0.01, zfar=100.0):
    """
    T-Less 데이터셋처럼 3x3 Intrinsic Matrix(K)가 이미 있을 때,
    이걸 3DGS/gsplat이 좋아하는 4x4 Projection Matrix로 변환하는 상남자의 함수!!
    """
    # 1. 3x3 K 행렬에서 알맹이(초점 거리와 주점)만 쏙쏙 빼온다!
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    # 2. 4x4 그래픽스용 빈 깡통 텐서(행렬) 준비!
    P = np.zeros((4, 4), dtype=np.float32)

    # 3. [X, Y축 스케일링] 픽셀 단위 초점 거리를 화면 비율(-1 ~ 1)로 압축!
    P[0, 0] = 2.0 * fx / W
    P[1, 1] = 2.0 * fy / H

    # 4. [주점 오프셋] 렌즈의 중심(cx, cy)이 정중앙(W/2, H/2)이 아닐 경우의 찌그러짐 보정!!
    # (OpenCV의 좌상단 0,0 좌표계를 그래픽스의 정중앙 0,0 좌표계로 밀어버림)
    P[0, 2] = (2.0 * cx / W) - 1.0
    P[1, 2] = (2.0 * cy / H) - 1.0 
    
    # 5. [Z축 클리핑] 원근감과 렌더링 한계선(Near/Far) 설정 (Vulkan/DirectX/3DGS 근본 공식)
    P[2, 2] = zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    
    # 6. Z값을 w로 복사해서 원근 투영(Perspective Divide)을 발동시키는 핵심 트리거!
    P[3, 2] = 1.0

    return P