from ase.io import read, write
from ase.constraints import FixAtoms
import sys

def convert_poscar_to_lammps():
    try:
        # 1. 读取 POSCAR (ASE 会自动解析 Selective Dynamics)
        print("📂 Reading POSCAR...")
        atoms = read('POSCAR', format='vasp')
    except Exception as e:
        print(f"❌ Error reading POSCAR: {e}")
        sys.exit(1)

    # 2. 生成 LAMMPS data 文件
    print("💾 Writing structure.data...")
    # atom_style atomic 对应不带电荷的格式
    write('structure.data', atoms, format='lammps-data')

    # 3. 提取固定原子 (Fixed Atoms)
    # ASE 将 VASP 的 F F F 解析为 FixAtoms 约束
    fixed_indices = []
    for constraint in atoms.constraints:
        if isinstance(constraint, FixAtoms):
            # 获取被固定原子的索引 (0-based)
            fixed_indices.extend(constraint.get_indices())
    
    # 去重并排序
    fixed_indices = sorted(list(set(fixed_indices)))

    # 4. 生成 LAMMPS 分组文件 in.groups
    print("🔒 Generating fixed atom groups (in.groups)...")
    with open('in.groups', 'w') as f:
        if fixed_indices:
            # LAMMPS 是 1-based 索引，所以要 +1
            lammps_ids = [str(i + 1) for i in fixed_indices]
            # 即使原子很多，group 命令也能分行处理，但为了保险使用 id 列表
            f.write(f"# Group definition generated from POSCAR\n")
            f.write(f"group fixed_atoms id {' '.join(lammps_ids)}\n")
            f.write(f"group mobile_atoms subtract all fixed_atoms\n")
            print(f"   -> Found {len(fixed_indices)} fixed atoms.")
        else:
            # 如果没有固定原子
            f.write("group fixed_atoms empty\n")
            f.write("group mobile_atoms union all\n")
            print("   -> No fixed atoms found.")

    # 5. 打印对应的 pair_coeff 提示
    elements = sorted(list(set(atoms.get_chemical_symbols())))
    print("\n⚠️  Please update your LAMMPS input with:")
    print(f"    pair_coeff * * {' '.join(elements)}")

if __name__ == "__main__":
    convert_poscar_to_lammps()