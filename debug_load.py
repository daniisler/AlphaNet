import torch
import torch_scatter  # 必须导入，因为你的模型依赖它
import sys

print(f"PyTorch Version: {torch.__version__}")
print(f"Torch Scatter Version: {torch_scatter.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")

model_path = "alphanet_lammps.pt"  # 你的模型文件名

print(f"\nAttempting to load {model_path} using torch.jit.load...")
try:
    # 模拟 LAMMPS 的加载方式
    model = torch.jit.load(model_path)
    model.eval()
    print("✅ Success! Model loaded successfully in Python.")

    # 尝试一次推理（可选）
    # 这里的输入维度需要根据你的实际情况构造，哪怕是随机数
    # print("Testing inference...")
    # dummy_input = ...
    # model(dummy_input)

except Exception as e:
    print(f"❌ Failed to load model: {e}")
    sys.exit(1)