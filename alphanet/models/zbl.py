# 文件路径: AlphaNet-lammps/alphanet/models/zbl.py
import torch
from torch import nn
import math

class ZBLPotential(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 默认参数 (通用拟合值)
        default_w = [0.187, 0.3769, 0.189, 0.081, 0.003, 0.037, 0.0546, 0.0715]
        default_b = [3.20, 1.10, 0.102, 0.958, 1.28, 1.14, 1.69, 5.0]
        
        # 从配置读取，支持微调 ZBL 参数
        w = getattr(config, 'zbl_w', default_w)
        if w is None: w = default_w
        
        b = getattr(config, 'zbl_b', default_b)
        if b is None: b = default_b
        
        gamma = getattr(config, 'zbl_gamma', 1.001)
        alpha = getattr(config, 'zbl_alpha', 0.6032)
        
        self.register_buffer('fzbl_w', torch.tensor(w, dtype=torch.get_default_dtype()))
        self.register_buffer('fzbl_b', torch.tensor(b, dtype=torch.get_default_dtype()))
        
        # 归一化权重
        with torch.no_grad():
            w_tensor = self.fzbl_w
            w_tensor = w_tensor.clamp(min=0.0)
            w_tensor = w_tensor / (w_tensor.sum() + 1e-12)
            self.fzbl_w.copy_(w_tensor)

        self.register_buffer('fzbl_gamma', torch.tensor(gamma, dtype=torch.get_default_dtype()))
        self.register_buffer('fzbl_alpha', torch.tensor(alpha, dtype=torch.get_default_dtype()))
        
        # 物理常数
        self.register_buffer('fzbl_E2', torch.tensor(14.399645478425, dtype=torch.get_default_dtype()))  # eV·Å
        self.register_buffer('fzbl_A0', torch.tensor(0.529177210903, dtype=torch.get_default_dtype()))    # Å
        
        # 平滑截断参数
        self.r_cut = 1.0 

    def forward(self, dist: torch.Tensor, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        """
        Pair ZBL Energy V_ij
        """
        r_e = dist
        
        # 确保计算精度一致
        dtype = r_e.dtype
        w = self.fzbl_w.to(dtype=dtype)
        b = self.fzbl_b.to(dtype=dtype)
        gamma = self.fzbl_gamma.to(dtype=dtype)
        alpha = self.fzbl_alpha.to(dtype=dtype)
        E2 = self.fzbl_E2.to(dtype=dtype)
        A0 = self.fzbl_A0.to(dtype=dtype)
        
        denom = torch.pow(z_j, alpha) + torch.pow(z_i, alpha)
        denom = torch.clamp(denom, min=1e-12)
        a_vals = gamma * 0.8854 * A0 / denom
        x = r_e / a_vals
        
        # V(r) = (Z1*Z2*e^2/r) * phi(x)
        # phi(x) = sum(w * exp(-b*x))
        exp_terms = torch.exp(-x.unsqueeze(1) * b.unsqueeze(0)) # [Edges, M]
        phi_vals = exp_terms.matmul(w)                          # [Edges]
        
        V_edge = (z_j * z_i * E2) * (phi_vals / r_e)
        
        # Cutoff smoothing (0 to r_cut)
        
        xrc = (r_e / self.r_cut).clamp(min=0.0, max=1.0)
        c = 0.5 * (torch.cos(math.pi * xrc) + 1.0)
        c = torch.where(r_e >= self.r_cut, torch.zeros_like(c), c)
        
        return V_edge * c