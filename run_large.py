import sys
import os
import subprocess

# =========================================================
# 0. [核心修复] 在导入任何库之前，仅通过环境变量探测 Rank 并锁定 GPU
#    绝对不要在这里 import mpi4py 或 torch！
# =========================================================
def strict_gpu_binding():
    # 1. 尝试从常见的 MPI 启动器环境变量中获取 Local Rank
    # OpenMPI (mpirun), MPICH, Slurm 等通常都会设置这些
    val = os.environ.get('OMPI_COMM_WORLD_LOCAL_RANK') or \
          os.environ.get('MV2_COMM_WORLD_LOCAL_RANK') or \
          os.environ.get('MPI_LOCALRANKID') or \
          os.environ.get('SLURM_LOCALID') or \
          os.environ.get('PMI_RANK')

    if val is not None:
        local_rank = int(val)
        
        # 2. 简单的 GPU 数量检测 (通过 nvidia-smi)
        # 如果失败默认 8 (取模安全)
        try:
            out = subprocess.check_output("nvidia-smi -L", shell=True).decode()
            num_gpus = len([x for x in out.strip().split('\n') if x])
        except:
            num_gpus = 8
            
        # 3. 计算目标 GPU ID
        target_gpu = local_rank % num_gpus
        
        # 4. [关键] 设置可见设备
        # 此时 Python 还没加载 CUDA，这个设置会生效！
        os.environ["CUDA_VISIBLE_DEVICES"] = str(target_gpu)
        
        print(f"🔒 [Pre-Init] Process detected Local Rank {local_rank}. Force binding to GPU {target_gpu} (CUDA_VISIBLE_DEVICES={target_gpu})")
        return True
    else:
        print("⚠️ [Pre-Init] No MPI environment variables found. GPU binding might fail if using MPI.")
        return False

# 立即执行绑定
is_mpi_bound = strict_gpu_binding()

# =========================================================
# 1. 现在的导入是安全的
# =========================================================
import torch
import numpy as np
import cupy as cp
import torch_scatter
import torch_geometric
import lammps as lmp_lib
import lammps.mliap

# 尝试导入 mpi4py 仅用于后续通信，不用于绑定
try:
    from mpi4py import MPI
    HAS_MPI = True
    my_rank = MPI.COMM_WORLD.Get_rank()
except ImportError:
    HAS_MPI = False
    my_rank = 0

try:
    from ase.data import atomic_numbers
except ImportError:
    if my_rank == 0: print("❌ ASE not found.")
    sys.exit(1)

# ---------------------------------------------------------
# 适配器类
# ---------------------------------------------------------
class AlphaNetAdapter:
    def __init__(self, jit_model_path):
        # 因为已经屏蔽了其他卡，这里只能看到一张卡，即 "cuda:0"
        self.device = torch.device("cuda:0")
        
        if my_rank == 0:
            print(f"🔌 Loading JIT model from {jit_model_path}...")
            
        self.model = _ORIGINAL_JIT_LOAD(jit_model_path, map_location=self.device)
        self.model.eval()
        
        self.ndescriptors = getattr(self.model, "ndescriptors", 1)
        self.nparams = getattr(self.model, "nparams", 1)
        self.rcutfac = getattr(self.model, "rcutfac", 2.5)
        self.element_types = getattr(self.model, "element_types", ["Al"])
        
        self.dtype = torch.float64 
        self.model.to(self.dtype)

    def compute_forces(self, data):
        natoms = data.nlocal
        ntotal = data.ntotal
        if natoms == 0: return

        # --- 数据提取 (.get + 映射 + Tensor) ---
        elems_np = data.elems.get()[:ntotal]
        # Al: 0 -> 13
        species_tensor = torch.as_tensor(elems_np + 1, device=self.device, dtype=torch.long)

        pair_j_np = data.pair_j.get()
        pair_i_np = data.pair_i.get()
        edge_index = torch.stack([
            torch.as_tensor(pair_j_np, device=self.device, dtype=torch.long),
            torch.as_tensor(pair_i_np, device=self.device, dtype=torch.long)
        ], dim=0)
        
        rij_np = data.rij.get()
        rij_tensor = torch.as_tensor(rij_np, device=self.device, dtype=self.dtype)
        
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

        # --- 推理 ---
        with torch.enable_grad():
            input_dict["vectors"].requires_grad_(True)
            total_e, atom_e, pair_forces = self.model(input_dict)
            pair_forces = -1.0 * pair_forces

        # --- 写回 ---
        atom_e_cpu = atom_e[:natoms].detach().cpu().numpy()
        data.eatoms[:natoms] = cp.asarray(atom_e_cpu)
        
        my_total_e = atom_e[:natoms].sum()
        data.energy = my_total_e.item()
        
        from torch.utils.dlpack import to_dlpack
        from cupy import from_dlpack
        force_cupy = from_dlpack(to_dlpack(pair_forces.detach().to(torch.float64)))
        data.update_pair_forces_gpu(force_cupy)

    def compute_descriptors(self, data): pass
    def compute_gradients(self, data): pass

# ---------------------------------------------------------
# Monkey Patch
# ---------------------------------------------------------
_ORIGINAL_JIT_LOAD = torch.jit.load
_ORIGINAL_TORCH_LOAD = torch.load

def hijack_load(f, *args, **kwargs):
    f_str = str(f)
    #if "alphanet_lammps.pt" in f_str:
    if my_rank == 0: print(f"⚡ [Rank {my_rank}] Intercepting model load -> Adapter")
    return adapter
    #return _ORIGINAL_JIT_LOAD(f, *args, **kwargs)

torch.jit.load = hijack_load
torch.load = hijack_load

# ---------------------------------------------------------
# 主程序
# ---------------------------------------------------------
if __name__ == "__main__":
    input_file = "test_large.in" 
    model_file = "alphanet_lammps.pt"

    if not os.path.exists(input_file):
        if my_rank == 0: print(f"❌ Error: {input_file} not found!")
        sys.exit(1)

    if my_rank == 0:
        print(f"✅ PyTorch {torch.__version__} loaded.")
        if is_mpi_bound:
            print("🚀 GPU Binding Strategy: Environment Variable (Success)")
        else:
            print("⚠️ GPU Binding Strategy: Serial/Fallback (Check if not running MPI)")

    adapter = AlphaNetAdapter(model_file)

    # Kokkos 参数: 每个进程只负责 "1" 个 GPU (因为它只能看到 1 个)
    cmd_args = [
        "-k", "on", "g", "1",              
        "-sf", "kk",                       
        "-pk", "kokkos", "newton", "on", "neigh", "half"
    ]

    try:
        if my_rank == 0: print(f"🚀 Initializing LAMMPS...")
        lmp = lmp_lib.lammps(cmdargs=cmd_args)

        if my_rank == 0: print("🔌 Activating ML-IAP Kokkos interface...")
        lammps.mliap.activate_mliappy_kokkos(lmp)
        
        if my_rank == 0: print(f"📂 Executing {input_file}...")
        lmp.file(input_file) 
        
        if my_rank == 0: print("🎉 LAMMPS simulation finished successfully.")

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"❌ Error on Rank {my_rank}: {e}")
        sys.exit(1)
