import torch
import numpy as np
import trimesh
from skimage.measure import marching_cubes

from net import NeRF_MLP

# 1. 학습된 NeRF 모델 불러오기 (모델 구조는 본인이 짠 클래스에 맞게)
model = NeRF_MLP()
model.load_state_dict(torch.load('nerf_weights.pt'))
model.eval()

# 2. 3D 공간(Bounding Box) 설정 및 Voxel Grid 생성
# 주의: N이 너무 크면 RAM/VRAM이 터집니다! 보통 256~512를 씁니다.
N = 256 
x = torch.linspace(-1.0, 1.0, N)
y = torch.linspace(-1.0, 1.0, N)
z = torch.linspace(-1.0, 1.0, N)
grid_x, grid_y, grid_z = torch.meshgrid(x, y, z, indexing='ij')

# (N*N*N, 3) 형태로 좌표계 플래튼(Flatten)
coords = torch.stack([grid_x, grid_y, grid_z], dim=-1).reshape(-1, 3).cuda()

# 3. MLP에 좌표를 찔러넣어서 밀도(Density, 보통 sigma) 값 추출
print("MLP Querying... (메모리 터짐 주의)")
densities = torch.zeros(N * N * N, device='cpu')

chunk_size = 1024 * 64 # 한 번에 추론할 좌표 개수 (OOM 방지용 청크 처리)
with torch.no_grad():
    for i in range(0, coords.shape[0], chunk_size):
        chunk_coords = coords[i:i+chunk_size]
        
        # 방향 벡터(viewdirs)는 표면 추출에 필요 없으므로 임의의 값(예: 0)을 넣거나 생략
        # 모델의 출력 중 밀도(sigma) 값만 가져옵니다.
        _, chunk_sigma = model(chunk_coords) 
        
        densities[i:i+chunk_size] = chunk_sigma.cpu()

# 3D 그리드 형태로 복구
densities = densities.reshape(N, N, N).numpy()

# 4. Marching Cubes 알고리즘으로 표면(Mesh) 추출
# threshold(iso-level)는 실험적으로 깎아보면서 맞춰야 합니다.
print("Running Marching Cubes...")
threshold = 15.0 # 이 값보다 밀도가 높으면 '물체'로 간주
verts, faces, normals, values = marching_cubes(densities, level=threshold)

# 5. Voxel 좌표계(0 ~ N-1)를 원래 3D 공간 좌표계(-1.0 ~ 1.0)로 정규화
verts = verts / (N - 1) * 2.0 - 1.0

# 6. Trimesh를 이용해 PLY 파일로 굽기
print("Exporting to PLY...")
mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)
mesh.export('nerf_extracted_mesh.ply')

print("완료! MeshLab이나 Blender에서 확인해보세요!")