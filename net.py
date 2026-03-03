import torch
import torch.nn as nn
        

class NeRF_MLP(nn.Module):
    
    def __init__(self, input_ch=63, input_ch_views=27, hidden=256, layers=8, skips=[4]):
        super().__init__() # 1. 필수!
        
        self.skips = skips
        self.layers = layers
        
        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, hidden)] + 
            [nn.Linear(hidden, hidden) if i not in self.skips 
             else nn.Linear(hidden + input_ch, hidden) for i in range(layers-1)]
        )
        
        self.sigma_out = nn.Linear(hidden, 1)
        
        self.feature_linear = nn.Linear(hidden, hidden)
        
        self.views_linears = nn.Sequential(
            nn.Linear(hidden + input_ch_views, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 3) # 최종 RGB 3채널 출력
        )


    def forward(self, x, d):
        """
        x: 위치 인코딩된 좌표 [N_rays, N_samples, 63]
        d: 위치 인코딩된 방향 [N_rays, N_samples, 27]
        """
        h = x
        
        # 8개의 층을 통과 (스킵 커넥션 포함)
        for i, layer in enumerate(self.pts_linears):
            h = layer(h)
            h = nn.functional.relu(h)
            if i in self.skips:
                h = torch.cat([x, h], dim=-1) # 원본 x를 옆에다 찰싹 붙여줌!
                
        # 밀도 출력 (1채널)
        sigma = self.sigma_out(h)
        
        # 방향 정보 합쳐서 RGB 출력 (3채널)
        feature = self.feature_linear(h)
        h = torch.cat([feature, d], dim=-1) # 방향 정보(d) 합치기!
        rgb = self.views_linears(h)
        
        # rgb(3) + sigma(1) = 4채널로 합쳐서 리턴
        return torch.cat([rgb, sigma], dim=-1)