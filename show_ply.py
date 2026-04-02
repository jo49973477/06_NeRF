import open3d as o3d
import numpy as np

def view_ply_open3d(file_path):
    # PLY 파일 로드
    pcd = o3d.io.read_point_cloud(file_path)
    
    # 점들이 너무 적다면(예: 40개) 점 크기를 키워서 보여줍니다.
    print(f"점 개수: {len(pcd.points)}")
    
    # 시각화 창 띄우기
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="3DGS Point Cloud Viewer")
    vis.add_geometry(pcd)
    
    # 점 크기 조절 (40개면 크기를 10.0 정도로 키워야 보입니다)
    render_option = vis.get_render_option()
    render_option.point_size = 5.0 
    render_option.background_color = np.array([0.1, 0.1, 0.1]) # 어두운 배경
    
    vis.run()
    vis.destroy_window()


view_ply_open3d("universal_output.ply")