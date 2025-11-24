import torch
import torch.nn as nn
from torch import Tensor
from typing import Dict, Tuple, Optional, List
import math
from math import pi
from torch_scatter import scatter, scatter_add
import os

# 导入你的 AlphaNet 组件
from alphanet.models.alphanet import AlphaNet
from alphanet.config import All_Config

class AlphaNetJITWrapper(nn.Module):
    """
    专为 LAMMPS CLI 设计的 JIT Wrapper。
    包含：
    1. AlphaNet 主推理逻辑
    2. ZBL 短距离排斥保护 (C1 连续)
    3. 邻居列表统计与打印 (每 100 步)
    """
    def __init__(self, model: AlphaNet, cutoff: float):
        super().__init__()
        # --- 复制 AlphaNet 子模块 ---
        self.z_emb = model.z_emb
        self.z_emb_ln = model.z_emb_ln
        self.radial_emb = model.radial_emb
        self.radial_lin = model.radial_lin
        self.neighbor_emb = model.neighbor_emb
        self.S_vector = model.S_vector
        self.lin = model.lin
        self.message_layers = model.message_layers
        self.FTEs = model.FTEs
        self.last_layer = model.last_layer
        self.last_layer_quantum = model.last_layer_quantum

        # --- 复制参数 ---
        self.a = model.a
        self.b = model.b
        self.kernel1 = model.kernel1
        
        # [关键修改] 将 ModuleList 转换为 ParameterList 以便 JIT 编译
        self.kernels_real = nn.ParameterList([p for p in model.kernels_real])
        self.kernels_imag = nn.ParameterList([p for p in model.kernels_imag])

        # --- 基础配置 ---
        self.cutoff = float(cutoff)
        self.pi = pi
        self.eps = 1e-8
        self.hidden_channels = model.hidden_channels
        self.chi1 = model.chi1
        self.complex_type = model.complex_type
        self.readout = model.readout
        
        # --- LAMMPS 元数据 (必须) ---
        self.ndescriptors = 1
        self.nparams = 1
        self.rcutfac = self.cutoff / 2.0
        self.register_buffer("r_max", torch.tensor(self.cutoff))

        # --- ZBL 参数 ---
        self.zbl = True
        if self.zbl:
            self.register_buffer('fzbl_w', model.fzbl_w)
            self.register_buffer('fzbl_b', model.fzbl_b)
            self.register_buffer('fzbl_gamma', model.fzbl_gamma)
            self.register_buffer('fzbl_alpha', model.fzbl_alpha)
            self.register_buffer('fzbl_E2', model.fzbl_E2)
            self.register_buffer('fzbl_A0', model.fzbl_A0)

        # --- 状态计数器 (注册为 buffer 以便保存) ---
        self.register_buffer("step_counter", torch.tensor(0, dtype=torch.long))
        self.register_buffer("prev_num_edges", torch.tensor(-1, dtype=torch.long))

    def forward(self, data: Dict[str, Tensor]) -> Tuple[Tensor, Tensor, Tensor]:
        # 1. 解包数据 (LAMMPS 传入)
        pos = data["positions"]
        z = data["node_attrs"]
        batch = data["batch"]
        edge_index = data["edge_index"]
        edge_vec = data["vectors"].requires_grad_() # 必须开启梯度
        dist = torch.linalg.norm(edge_vec, dim=1)
        
        # natoms 是一个 tensor，通常只包含一个标量 [nlocal]
        natoms_tensor = data["natoms"]
        nlocal = natoms_tensor[0]

        # ---------------------------------------------------------
        # 邻居统计与变动检测 (JIT 兼容写法)
        # ---------------------------------------------------------
        self.step_counter += 1
        current_num_edges = edge_index.size(1)
        
        if self.step_counter % 100 == 0:
            # 打印基本步数信息
            print("Step:", self.step_counter.item(), "| Total Edges:", current_num_edges)
            
            # 简单的变动检测
            prev = self.prev_num_edges.item()
            if prev != -1:
                if current_num_edges != prev:
                    print("Warning: Neighbor list size changed!", prev, "->", current_num_edges)
            
            # 计算配位数统计 (仅针对本地原子)
            if current_num_edges > 0:
                degrees = torch.bincount(edge_index[1])
                if degrees.size(0) >= nlocal:
                    local_degs = degrees[:nlocal].float()
                    if local_degs.numel() > 0:
                        avg = torch.mean(local_degs).item()
                        mn = torch.min(local_degs).item()
                        mx = torch.max(local_degs).item()
                        print(" Neighbors/Atom: Avg=", avg, "Min=", mn, "Max=", mx)
            
            self.prev_num_edges[0] = current_num_edges

        # ---------------------------------------------------------
        # 2. AlphaNet 核心推理
        # ---------------------------------------------------------
        z_emb = self.z_emb_ln(self.z_emb(z))
        radial_emb = self.radial_emb(dist)
        radial_hidden = self.radial_lin(radial_emb)
        
        # 平滑截断 + Mask
        rbounds = 0.5 * (torch.cos(dist * self.pi / self.cutoff) + 1.0)
        mask_cut = (dist < self.cutoff).to(rbounds.dtype)
        rbounds = rbounds * mask_cut
        radial_hidden = rbounds.unsqueeze(-1) * radial_hidden

        s = self.neighbor_emb(z, z_emb, edge_index, radial_hidden)
        vec = torch.zeros(s.size(0), 3, s.size(1), device=s.device, dtype=s.dtype)
        
        # j, i = edge_index[0], edge_index[1] (LAMMPS mliap 格式)
        j, i = edge_index[0], edge_index[1]
        edge_diff = edge_vec / (dist.unsqueeze(1) + self.eps)
        
        edge_vec_mean = scatter(edge_vec, i, reduce='mean', dim=0) 
        edge_cross = torch.cross(edge_vec, edge_vec_mean[i])
        edge_vertical = torch.cross(edge_diff, edge_cross)
        edge_frame = torch.cat((edge_diff.unsqueeze(-1), edge_cross.unsqueeze(-1), edge_vertical.unsqueeze(-1)), dim=-1)
        S_i_j = self.S_vector(s, edge_diff.unsqueeze(-1), edge_index, radial_hidden)
        
        scalrization1 = torch.sum(S_i_j[i].unsqueeze(2) * edge_frame.unsqueeze(-1), dim=1)
        scalrization2 = torch.sum(S_i_j[j].unsqueeze(2) * edge_frame.unsqueeze(-1), dim=1)
        scalrization1[:, 1, :] = torch.abs(scalrization1[:, 1, :].clone())
        scalrization2[:, 1, :] = torch.abs(scalrization2[:, 1, :].clone())
        
        scalar3 = (self.lin(torch.permute(scalrization1, (0, 2, 1))) + 
                  torch.permute(scalrization1, (0, 2, 1))[:, :, 0].unsqueeze(2)).squeeze(-1) / math.sqrt(self.hidden_channels)
        scalar4 = (self.lin(torch.permute(scalrization2, (0, 2, 1))) + 
                  torch.permute(scalrization2, (0, 2, 1))[:, :, 0].unsqueeze(2)).squeeze(-1) / math.sqrt(self.hidden_channels)
        
        edge_weight = torch.cat((scalar3, scalar4), dim=-1) * rbounds.unsqueeze(-1)
        edge_weight = torch.cat((edge_weight, radial_hidden, radial_emb), dim=-1)
        
        quantum = torch.einsum('ik,bi->bk', self.kernel1, z_emb)
        real, imagine = torch.split(quantum, self.chi1, dim=-1)
        quantum = torch.complex(real, imagine)

        rope: Optional[Tensor] = None
        
        # ==========================================================
        # [核心修复] 使用 zip 替代索引循环
        # TorchScript 不支持 self.modules[k] 这种动态索引，必须用 zip 迭代
        # ==========================================================
        for message_layer, fte, kernel_real, kernel_imag in zip(
            self.message_layers, 
            self.FTEs, 
            self.kernels_real, 
            self.kernels_imag
        ):
            if rope is None:
                rope, ds, dvec = message_layer(s, vec, edge_index, radial_emb, edge_weight, edge_diff, None)
            else:
                rope, ds, dvec = message_layer(s, vec, edge_index, radial_emb, edge_weight, edge_diff, rope)
            
            s = s + ds
            vec = vec + dvec
            
            kerneli = torch.complex(kernel_real, kernel_imag) 
            quantum = torch.einsum('ikl,bi,bl->bk', kerneli, s.to(self.complex_type), quantum)
            quantum = quantum / quantum.abs().to(self.complex_type)
            
            ds, dvec = fte(s, vec)
            s = s + ds
            vec = vec + dvec
        # ==========================================================

        s_per_atom = self.last_layer(s) + self.last_layer_quantum(torch.cat([quantum.real, quantum.imag], dim=-1)) / self.chi1
        node_energy = (self.a[z].unsqueeze(1) * s_per_atom + self.b[z].unsqueeze(1)).squeeze(-1)
        
        # ---------------------------------------------------------
        # 3. ZBL 保护逻辑
        # ---------------------------------------------------------
        if self.zbl:
            r_e = dist
            Z_j = z[j]
            Z_i = z[i]
            
            w = self.fzbl_w
            b_params = self.fzbl_b
            # 类型转换以匹配
            if s.dtype == torch.float64:
                w = w.double()
                b_params = b_params.double()
            
            gamma = self.fzbl_gamma
            alpha = self.fzbl_alpha
            E2 = self.fzbl_E2
            A0 = self.fzbl_A0
            
            denom = torch.pow(Z_j, alpha) + torch.pow(Z_i, alpha)
            denom = torch.clamp(denom, min=1e-12)
            a_vals = gamma * 0.8854 * A0 / denom
            x = r_e / a_vals
            
            exp_terms = torch.exp(-x.unsqueeze(1) * b_params.unsqueeze(0))
            phi_vals = exp_terms.matmul(w)
            
            V_edge = (Z_j * Z_i * E2) * (phi_vals / r_e)
            
            # ZBL Cutoff (Hard cutoff 1.0 Å)
            r_cut_zbl = 1.0
            xrc = (r_e / r_cut_zbl).clamp(min=0.0, max=1.0)
            # Cosine taper
            c = 0.5 * (torch.cos(torch.pi * xrc) + 1.0)
            # 强制截断掩码
            c = torch.where(r_e >= r_cut_zbl, torch.zeros_like(c), c)
            
            V_edge = V_edge * c
            
            # 分配给原子 (0.5 * edge energy)
            zbl_per_atom = scatter_add(V_edge, i, dim=0, dim_size=node_energy.size(0)) * 0.5
            node_energy = node_energy + zbl_per_atom
        
        # 汇总能量
        s_total = scatter(node_energy, batch, dim=0, reduce=self.readout).sum()
        
        # 计算力
        grad_outputs: List[Optional[Tensor]] = [torch.ones_like(s_total)]
        edge_forces = torch.autograd.grad(
            outputs=[s_total],
            inputs=[edge_vec], 
            grad_outputs=grad_outputs,
            retain_graph=False,
            create_graph=False,
            allow_unused=True,
        )[0]
        
        if edge_forces is None:
            edge_forces = torch.zeros_like(edge_vec)
        
        pair_forces = -1.0 * edge_forces 
        
        return s_total, node_energy, pair_forces

def main():
    # --- 配置路径 ---
    config_path = "/mnt/bn/bangchen/repos/AlphaNet/pretrained/AQCAT25/aqcat.json"
    checkpoint_path = "/mnt/bn/bangchen/repos/AlphaNet/pretrained/AQCAT25/aqcat_1021.ckpt"
    output_path = "alphanet_lammps.pt"
    
    print(f"Loading config: {config_path}")
    config = All_Config().from_json(config_path)
    
    print("Initializing base model...")
    raw_model = AlphaNet(config.model)
    
    print(f"Loading weights: {checkpoint_path}")
    
    # --- [核心修复] 稳健的权重加载逻辑 (防止 Segfault) ---
    # 优先尝试 JIT 加载 (因为之前的报错暗示它可能是 TorchScript 格式)
    try:
        print("Attempting to load as TorchScript (JIT) model...")
        jit_model = torch.jit.load(checkpoint_path, map_location="cpu")
        state_dict = jit_model.state_dict()
        print("Success! Loaded via torch.jit.load")
    except Exception as e_jit:
        print(f"JIT load failed, falling back to standard torch.load... ({str(e_jit)[:100]})")
        try:
            ckpt = torch.load(checkpoint_path, map_location="cpu")
            state_dict = ckpt['state_dict'] if isinstance(ckpt, dict) and 'state_dict' in ckpt else ckpt
            print("Success! Loaded via torch.load")
        except Exception as e_load:
            print("CRITICAL ERROR: Failed to load checkpoint with both methods.")
            raise e_load

    # 修正键名 (移除 'model.' 前缀)
    new_state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
    
    # 加载权重到 raw_model
    try:
        raw_model.load_state_dict(new_state_dict, strict=False)
    except RuntimeError as e:
        print("Warning during load_state_dict (ignore if just shape mismatch in unused layers):", e)
    
    # 转换为 Float64
    print("Converting to float64 (Double Precision)...")
    raw_model.double()
    raw_model.eval()
    
    # 包裹 JIT 逻辑
    print("Wrapping with JIT logic (ZBL + Print Stats)...")
    jit_wrapper = AlphaNetJITWrapper(raw_model, cutoff=config.model.cutoff)
    jit_wrapper.eval()
    
    # 编译
    print("Compiling TorchScript...")
    try:
        scripted_model = torch.jit.script(jit_wrapper)
    except Exception as e:
        print("\n!!! JIT Compilation Error !!!")
        print(e)
        return
    
    # 保存
    print(f"Saving to {output_path}...")
    scripted_model.save(output_path)
    print("✅ Done! You can now run LAMMPS directly.")

if __name__ == "__main__":
    main()