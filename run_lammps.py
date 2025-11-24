import torch_scatter
import torch_geometric

import sys
import os
import torch
import numpy as np
import cupy as cp
from torch.utils.dlpack import to_dlpack
from cupy import from_dlpack

import lammps as lmp_lib
import lammps.mliap

# 假设 _ORIGINAL_JIT_LOAD 已经在外部定义
# from somewhere import _ORIGINAL_JIT_LOAD 

class AlphaNetAdapter:
    def __init__(self, jit_model_path, device):
        print(f"🔌 Loading internal JIT model from {jit_model_path}...")
        self.device = torch.device(device)
        
        # 拦截加载逻辑
        self.model = _ORIGINAL_JIT_LOAD(jit_model_path, map_location=self.device)
        self.model.eval()
        
        # 复制元数据
        self.ndescriptors = getattr(self.model, "ndescriptors", 1)
        self.nparams = getattr(self.model, "nparams", 1)
        self.rcutfac = getattr(self.model, "rcutfac", 2.5)
        self.element_types = getattr(self.model, "element_types", ["Al"])
        
        self.dtype = torch.float64 
        self.model.to(self.dtype)

        # [新增] 用于统计邻居变化的计数器
        self.step_counter = 0
        self.prev_num_edges = -1

    def compute_forces(self, data):
        natoms = data.nlocal
        ntotal = data.ntotal
        
        if natoms == 0: return

        # ==========================================
        # 1. 数据提取
        # ==========================================
        elems_np = data.elems.get()[:ntotal]
        species_tensor = torch.as_tensor(elems_np + 1, device=self.device, dtype=torch.long)

        # 邻居列表
        pair_j_np = data.pair_j.get()
        pair_i_np = data.pair_i.get()
        
        # i 是中心原子，j 是邻居原子 (LAMMPS 格式通常 j 是邻居列表中的索引)
        # 注意：LAMMPS 的 pair_i/pair_j 定义可能随 neighbor style 变化
        # 通常 pair_i 是 local atom index, pair_j 是 neighbor index
        edge_index = torch.stack([
            torch.as_tensor(pair_j_np, device=self.device, dtype=torch.long),
            torch.as_tensor(pair_i_np, device=self.device, dtype=torch.long)
        ], dim=0)
        
        # 邻居向量
        rij_np = data.rij.get()
        rij_tensor = torch.as_tensor(rij_np, device=self.device, dtype=self.dtype)
        
        # ---------------------------------------------------------
        # [新增] 邻居统计与变动检测
        # ---------------------------------------------------------
        self.step_counter += 1
        current_num_edges = edge_index.shape[1]
        
        # 每 10 步或者是边数发生变化时打印
        if self.step_counter % 100 == 0:
            
            # 计算每个原子的配位数 (Degree)
            # edge_index[1] 通常是源节点（或者目标节点，取决于具体定义，但在无向图中只看度数是一样的）
            # 统计每个 i 有多少个 j
            degrees = torch.bincount(edge_index[1], minlength=ntotal)
            
            # 只看本地原子 (nlocal) 的配位数统计
            local_degrees = degrees[:natoms].float()
            
            avg_neigh = local_degrees.mean().item()
            min_neigh = local_degrees.min().item()
            max_neigh = local_degrees.max().item()
            
            change_str = " (UNCHANGED)"
            if self.prev_num_edges != -1 and current_num_edges != self.prev_num_edges:
                diff = current_num_edges - self.prev_num_edges
                change_str = f" (CHANGED: {diff:+d})"
            
            print(f"🔍 [Step {self.step_counter}] Neighbors Stats:")
            print(f"   Total Edges: {current_num_edges}{change_str}")
            print(f"   Neighbors/Atom: Avg={avg_neigh:.2f}, Min={int(min_neigh)}, Max={int(max_neigh)}")
            
            # 更新上一状态
            self.prev_num_edges = current_num_edges
        # ---------------------------------------------------------

        # 构造输入
        pos_tensor = torch.zeros((ntotal, 3), device=self.device, dtype=self.dtype)
        batch = torch.zeros(ntotal, device=self.device, dtype=torch.long)
        natoms_tensor = torch.tensor([ntotal], device=self.device, dtype=torch.long)
        cell_tensor = torch.eye(3, device=self.device, dtype=self.dtype).unsqueeze(0)

        input_dict = {
            "positions": pos_tensor,
            "node_attrs": species_tensor,
            "edge_index": edge_index,
            "vectors": rij_tensor,
            "batch": batch,
            "natoms": natoms_tensor,
            "cell": cell_tensor
        }

        # ==========================================
        # 2. 推理
        # ==========================================
        with torch.enable_grad():
            input_dict["vectors"].requires_grad_(True)
            total_e, atom_e, pair_forces = self.model(input_dict)
            pair_forces = -1.0 * pair_forces

        # ==========================================
        # 3. 写回结果
        # ==========================================
        atom_e_cpu = atom_e[:natoms].detach().cpu().numpy()
        data.eatoms[:natoms] = cp.asarray(atom_e_cpu)
        
        my_total_e = atom_e[:natoms].sum()
        data.energy = my_total_e.item()
        
        force_tensor = pair_forces.detach().to(torch.float64)
        force_cupy = from_dlpack(to_dlpack(force_tensor))
        
        data.update_pair_forces_gpu(force_cupy)

    def compute_descriptors(self, data): pass
    def compute_gradients(self, data): pass

# ... (Monkey Patch 和 Main 部分保持不变)

# ---------------------------------------------------------
# Monkey Patch
# ---------------------------------------------------------
_ORIGINAL_JIT_LOAD = torch.jit.load
_ORIGINAL_TORCH_LOAD = torch.load

def hijack_load(f, *args, **kwargs):
    f_str = str(f)
    if "alphanet_lammps.pt" in f_str:
        print(f"⚡ Intercepted load request for '{f_str}'. Returning Adapter object!")
        return adapter
    return _ORIGINAL_JIT_LOAD(f, *args, **kwargs)

torch.jit.load = hijack_load
torch.load = hijack_load

# ---------------------------------------------------------
# 主程序
# ---------------------------------------------------------
if __name__ == "__main__":
    input_file = "test.in"
    model_file = "alphanet_lammps.pt"

    if not os.path.exists(input_file):
        print(f"❌ Error: {input_file} not found!")
        sys.exit(1)

    print(f"✅ PyTorch {torch.__version__} loaded.")
    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"✅ Using device: {device_name}")

    print("🛠️  Initializing Adapter...")
    adapter = AlphaNetAdapter(model_file, device_name)

    cmd_args = [
        "-k", "on", "g", "1",              
        "-sf", "kk",                       
        "-pk", "kokkos", "newton", "on", "neigh", "half"
    ]

    try:
        print(f"🚀 Initializing LAMMPS...")
        lmp = lmp_lib.lammps(cmdargs=cmd_args)

        print("🔌 Activating ML-IAP Kokkos interface...")
        lammps.mliap.activate_mliappy_kokkos(lmp)
        
        print(f"📂 Executing {input_file}...")
        lmp.file(input_file) 
        
        print("🎉 LAMMPS simulation finished successfully.")

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"❌ Error: {e}")
        sys.exit(1)