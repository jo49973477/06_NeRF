import torch
import numpy as np
from plyfile import PlyData, PlyElement

def convert_to_universal_ply(pt_path, ply_path):
    print("🚀 범용 뷰어 호환 모드로 데이터 변환을 시작합니다...")
    
    # 1. 파일 로드
    checkpoint = torch.load(pt_path, map_location='cpu')
    means = checkpoint['means'].numpy()
    
    # 2. 날것의 색상 데이터(Logit)를 0~1로 복원
    colors_logit = checkpoint['colors']
    colors_rgb = torch.sigmoid(colors_logit).numpy()
    
    # 3. 0~1 값을 0~255 사이의 정수(uint8)로 확실하게 변환
    colors_uint8 = (colors_rgb * 255.0).clip(0, 255).astype(np.uint8)
    
    # 4. 가장 단순하고 직관적인 범용 이름표 사용
    dtype_universal = [
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('red', 'u1'), ('green', 'u1'), ('blue', 'u1') # u1 = 0~255 정수
    ]
    
    elements = np.empty(means.shape[0], dtype=dtype_universal)
    
    elements['x'] = means[:, 0]
    elements['y'] = means[:, 1]
    elements['z'] = means[:, 2]
    elements['red'] = colors_uint8[:, 0]
    elements['green'] = colors_uint8[:, 1]
    elements['blue'] = colors_uint8[:, 2]
    
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(ply_path)
    print(f"✅ {ply_path} 생성 완료! 이제 어느 뷰어에서든 색이 보일 겁니다.")

if __name__ == "__main__":
    convert_to_universal_ply("3dgs_checkpoint_tless/tless_latest.pt", "universal_output.ply")