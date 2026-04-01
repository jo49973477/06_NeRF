import torch
import numpy as np
from plyfile import PlyData, PlyElement

def convert_pt_to_ply(pt_path, ply_path):
    print(f"🔥 [1단계] {pt_path} 에서 3090 대장님의 영혼석을 깨웁니다...")
    
    # 1. pt 파일 로드! (CPU로 안전하게 모셔오기)
    checkpoint = torch.load(pt_path, map_location='cpu')
    
    # 2. 텐서를 Numpy 배열로 변환!
    means = checkpoint['means'].numpy()
    scales = checkpoint['scales'].numpy()
    quats = checkpoint['quats'].numpy()
    opacities = checkpoint['opacities'].numpy()
    colors = checkpoint['colors'].numpy()
    
    # (법선 벡터는 3DGS에서 보통 안 쓰니까 0으로 채움)
    normals = np.zeros_like(means) 
    
    print(f"🍊 [2단계] {means.shape[0]}개의 가우시안 점들을 조립 중...")

    # 3. 전 세계 3DGS 뷰어들이 환장하는 '엄격한' 데이터 이름표(dtype) 세팅!!
    # (이름표가 하나라도 틀리면 WebGL 뷰어들이 뱉어냄!)
    dtype_full = [
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
        ('f_dc_0', 'f4'), ('f_dc_1', 'f4'), ('f_dc_2', 'f4'), # 색상 (RGB or SH)
        ('opacity', 'f4'),
        ('scale_0', 'f4'), ('scale_1', 'f4'), ('scale_2', 'f4'),
        ('rot_0', 'f4'), ('rot_1', 'f4'), ('rot_2', 'f4'), ('rot_3', 'f4')
    ]
    
    # 4. 빈 깡통(배열) 만들고 알맹이 채워 넣기!
    elements = np.empty(means.shape[0], dtype=dtype_full)
    
    # 위치 (x, y, z)
    elements['x'] = means[:, 0]
    elements['y'] = means[:, 1]
    elements['z'] = means[:, 2]
    
    # 법선 (nx, ny, nz)
    elements['nx'] = normals[:, 0]
    elements['ny'] = normals[:, 1]
    elements['nz'] = normals[:, 2]
    
    # 색상 (f_dc_0, f_dc_1, f_dc_2)
    # 💡 팁: RGB 텐서를 그냥 쓰면 웹 뷰어에서 약간 어둡게 보일 수 있음. 
    # 원래 3DGS는 Spherical Harmonics(SH) 공식을 쓰기 때문! (RGB = SH * 0.28209 + 0.5)
    # 캡틴의 colors가 순수 0~1 RGB라면, SH DC 성분으로 역변환해주는 상남자 센스!
    sh_dc = (colors - 0.5) / 0.28209
    elements['f_dc_0'] = sh_dc[:, 0]
    elements['f_dc_1'] = sh_dc[:, 1]
    elements['f_dc_2'] = sh_dc[:, 2]
    
    # 불투명도
    elements['opacity'] = opacities
    
    # 스케일
    elements['scale_0'] = scales[:, 0]
    elements['scale_1'] = scales[:, 1]
    elements['scale_2'] = scales[:, 2]
    
    # 회전 (쿼터니언)
    elements['rot_0'] = quats[:, 0]
    elements['rot_1'] = quats[:, 1]
    elements['rot_2'] = quats[:, 2]
    elements['rot_3'] = quats[:, 3]
    
    print("💾 [3단계] PLY 파일로 압축 중... 잠시만 기다려주세요!")
    
    # 5. PlyData 객체로 말아서 저장!!
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(ply_path)
    
    print(f"🎉 [성공] {ply_path} 변환 완료!! 이제 뷰어에 드래그 앤 드롭 하세요!!")

if __name__ == "__main__":
    # 사용법: 파일 이름만 캡틴 거에 맞게 바꿔서 실행!
    convert_pt_to_ply(
        pt_path="fruitninja_checkpoints/orange_latest.pt", 
        ply_path="orange_output.ply"
    )