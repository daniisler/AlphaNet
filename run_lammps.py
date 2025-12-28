import sys
import os
import torch
import lammps as lmp_lib
import lammps.mliap

try:
    import alphanet.infer.lammps_mliap_alphanet
except ImportError:
    sys.path.append(os.getcwd()) 
    import alphanet.infer.lammps_mliap_alphanet


_ORIGINAL_JIT_LOAD = torch.jit.load
_ORIGINAL_TORCH_LOAD = torch.load

GLOBAL_LOADED_MODEL = None

def hijack_load(f, *args, **kwargs):
    """
    拦截 LAMMPS (或其 Python 接口) 的加载请求。
    直接返回我们预加载好的 Python 对象。
    """
    f_str = str(f)
    print(f"⚡ Intercepted load request for '{f_str}'")
    
    if GLOBAL_LOADED_MODEL is not None:
        print("   => Returning pre-loaded LAMMPS_MLIAP_ALPHANET Object!")
        return GLOBAL_LOADED_MODEL
    
    print("   => Warning: Global model not set. Fallback to original load.")
    # 根据文件后缀猜测加载方式
    if f_str.endswith('.pt') or f_str.endswith('.pth'):
        return _ORIGINAL_TORCH_LOAD(f, *args, **kwargs)
    return _ORIGINAL_JIT_LOAD(f, *args, **kwargs)

# 替换 torch 的加载函数，以便被 LAMMPS 调用时触发钩子
torch.jit.load = hijack_load
torch.load = hijack_load

# ---------------------------------------------------------
# 主程序
# ---------------------------------------------------------
if __name__ == "__main__":
    input_file = "sl.in"             # LAMMPS 输入文件
    model_file = "sl.pt" # create_lammps_model.py 生成的文件

    if not os.path.exists(input_file):
        print(f"❌ Error: {input_file} not found!")
        sys.exit(1)
        
    if not os.path.exists(model_file):
        print(f"❌ Error: {model_file} not found!")
        sys.exit(1)

    print(f"✅ PyTorch {torch.__version__} loaded.")
    
    # 1. 确定设备
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"✅ Using device: CUDA (GPU)")
    else:
        device = torch.device("cpu")
        print(f"⚠️ Using device: CPU")

    # 2. 预加载模型对象 (Pre-load)
    print(f"🛠️  Loading Python object from {model_file}...")
    try:
        # 使用原始的 torch.load 加载保存的 Python 对象
        loaded_object = _ORIGINAL_TORCH_LOAD(model_file, map_location=device)
        
        # 确保模型及其内部参数都在正确的设备上
        if hasattr(loaded_object, 'model'):
            loaded_object.model.to(device)
            loaded_object.device = device # 更新对象内部记录的 device
            loaded_object.model.eval()
            
        # 注册到全局变量，供钩子使用
        GLOBAL_LOADED_MODEL = loaded_object
        print("   Model loaded and registered successfully.")
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"❌ Failed to load model: {e}")
        sys.exit(1)

    # 3. 配置 LAMMPS
    # -k on g 1: 开启 Kokkos 并使用 1 个 GPU
    # neigh half: Kokkos 默认要求，我们已经在 Python 端通过“智能对称化”解决了这个问题
    # newton on:  必须开启，用于跨进程的力通信
    cmd_args = [
        "-k", "on", "g", "1",              
        "-sf", "kk",                       
        "-pk", "kokkos", "newton", "on", "neigh", "half"
    ]

    try:
        print(f"🚀 Initializing LAMMPS...")
        lmp = lmp_lib.lammps(cmdargs=cmd_args)

        print("🔌 Activating ML-IAP Kokkos interface...")
        # 这行代码会触发 C++ 调用 Python 来加载模型
        # 此时会命中我们的 hijack_load，并返回 GLOBAL_LOADED_MODEL
        lammps.mliap.activate_mliappy_kokkos(lmp)
        
        print(f"📂 Executing {input_file}...")
        lmp.file(input_file) 
        
        print("🎉 LAMMPS simulation finished successfully.")

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"❌ Error during simulation: {e}")
        sys.exit(1)