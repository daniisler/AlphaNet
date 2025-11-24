from ase import Atoms
from ase.build import molecule
from ase.constraints import FixBondLengths
from ase.calculators.lj import LennardJones
from ase.optimize import FIRE
import numpy as np
from ase.io import write
import random

# ----------------配置----------------
filename = "methane_ase.data"
n_mol = 20
box_size = 20.0
# ------------------------------------

print(f"🛠️  Building system with ASE: {n_mol} CH4 in {box_size}x{box_size}x{box_size} box...")

# 1. 创建一个空的盒子
atoms = Atoms(cell=[box_size, box_size, box_size], pbc=True)

# 2. 放入甲烷分子
# 为了防止一开始就重叠，我们先用简单的格子点作为初始位置
grid_n = int(np.ceil(n_mol**(1/3)))
spacing = box_size / grid_n

count = 0
for x in range(grid_n):
    for y in range(grid_n):
        for z in range(grid_n):
            if count >= n_mol: break
            
            # 生成一个标准甲烷分子
            mol = molecule('CH4')
            
            # 随机旋转
            mol.rotate(random.random() * 360, 'x')
            mol.rotate(random.random() * 360, 'y')
            mol.rotate(random.random() * 360, 'z')
            
            # 移动到格子点，加一点点随机扰动
            jitter = (random.random() - 0.5) * 1.0
            pos = np.array([(x+0.5)*spacing, (y+0.5)*spacing, (z+0.5)*spacing]) + jitter
            mol.translate(pos)
            
            atoms += mol
            count += 1

# 3. 设置原子类型 ID (C=1, H=2)
# ASE 默认按序号排，我们需要手动指定
# 获取所有符号
symbols = atoms.get_chemical_symbols()
# 定义映射: C->1, H->2
type_map = {'C': 1, 'H': 2}
# 生成 type 数组
types = [type_map[s] for s in symbols]
# 赋值给 atoms 对象 (用于 lammps-data 输出)
atoms.set_tags(types) 

# 4. 粗优化 (Coarse Optimization)
# 使用简单的 Lennard-Jones 势把靠太近的原子推开
# sigma=2.0, epsilon=0.05 这是一个非常软的排斥力，足以推开重叠，但不会破坏结构
calc = LennardJones(sigma=2.0, epsilon=0.05)
atoms.calc = calc

print("🔧 Running coarse optimization with Lennard-Jones potential...")
# 固定 C-H 键长，防止简单的 LJ 力场把分子拆散了
# CH4 中，每 5 个原子是一组，索引 0-4, 5-9 ...
# 0是C, 1,2,3,4是H。
constraints = []
for i in range(n_mol):
    base = i * 5
    c_idx = base
    constraints.append(FixBondLengths([(c_idx, base+1), (c_idx, base+2), (c_idx, base+3), (c_idx, base+4)]))

atoms.set_constraint(constraints)

# 运行优化
opt = FIRE(atoms, logfile=None) # 使用 FIRE 算法，不输出太多日志
opt.run(fmax=0.1, steps=100)    # 只要力小于 0.1 eV/A 就停，或者跑 100 步

print(f"✅ Optimization done. Potential Energy: {atoms.get_potential_energy():.4f} eV")

# 5. 写入 LAMMPS data 文件
# atom_style='atomic' 对应 lammps 里的 atom_style atomic
# 必须显式传入 masses
masses = {1: 12.011, 2: 1.008}

# 注意：write lammps-data 时，ASE 会根据 atomic number 自动分配 type。
# 为了确保 C=1, H=2，最稳妥的方法是手动指定 specorder
write(filename, atoms, format='lammps-data', specorder=['C', 'H'], atom_style='atomic')

print(f"💾 Saved to {filename}")