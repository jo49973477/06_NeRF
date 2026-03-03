import torch

class PositionalEncoder:
    def __init__(self, L=10):
        self.L = L

        self.freq_bands = (2.0 ** torch.arange(L)) * torch.pi 

    def __call__(self, p):
        """
        p: 3D 좌표 [1024, 64, 3]
        """
        freq_bands = self.freq_bands.to(p.device)
        p_freq = p[..., None] * freq_bands
        p_freq = p_freq.flatten(start_dim=-2, end_dim=-1)
        return torch.cat([p, torch.sin(p_freq), torch.cos(p_freq)], dim=-1)
    

if __name__ == "__main__":
    pe = PositionalEncoder()

    print(pe(torch.Tensor([0.5])))