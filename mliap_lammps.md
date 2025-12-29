# LAMMPS Installation Guide: ML-IAP with KOKKOS (CUDA) Support

This guide details the process for building LAMMPS with the Machine Learning Interatomic Potential (`ML-IAP`) package, enabled with Python bindings and KOKKOS acceleration for NVIDIA GPUs.

## Prerequisites
* **Git** & **CMake**
* **C++ Compiler** (compatible with your MPI/CUDA version)
* **MPI Implementation** (e.g., OpenMPI, MPICH)
* **CUDA Toolkit** (version 11.x or 12.x)
* **Python 3.x**

---

## 1. Get the Source Code
Clone the repository and check out the specific commit hash used for this build to ensure reproducibility.

```bash
git clone [https://github.com/lammps/lammps.git](https://github.com/lammps/lammps.git)
cd lammps

# Checkout specific commit for stability/reproducibility
git checkout ccca772

```

## 2. Prepare the Build Environment

We will use an out-of-source build directory to keep the source tree clean. We will also start with the `kokkos-cuda` CMake preset.

```bash
mkdir build-mliap
cd build-mliap

# Copy the KOKKOS CUDA preset to the build directory
cp ../cmake/presets/kokkos-cuda.cmake ./

```

### ⚠️ Important Configuration Step

Before generating the build files, you must edit `kokkos-cuda.cmake` to match your specific GPU architecture.

1. Open `kokkos-cuda.cmake` in a text editor.
2. Locate the architecture flag (e.g., `-DKokkos_ARCH_...`).
3. Change it to match your GPU (e.g., `Kokkos_ARCH_VOLTA70`, `Kokkos_ARCH_AMPERE80`, `Kokkos_ARCH_HOPPER90`).
* *Reference:* [LAMMPS KOKKOS Build Options](https://docs.lammps.org/Build_extras.html#kokkos)



---

## 3. Configure and Compile

Run CMake to configure the build with ML-IAP, SNAP, and Python support enabled.

```bash
cmake -C kokkos-cuda.cmake \
  -D CMAKE_BUILD_TYPE=Release \
  -D CMAKE_INSTALL_PREFIX=$(pwd) \
  -D BUILD_MPI=ON \
  -D PKG_ML-IAP=ON \
  -D PKG_ML-SNAP=ON \
  -D MLIAP_ENABLE_PYTHON=ON \
  -D PKG_PYTHON=ON \
  -D BUILD_SHARED_LIBS=ON \
  ../cmake

```

**Key Flags Explained:**

* `PKG_ML-IAP=ON`: Enables the Machine Learning Interatomic Potentials package.
* `MLIAP_ENABLE_PYTHON=ON`: Allows ML-IAP to call Python functions (essential for PyTorch/PyG models).
* `BUILD_SHARED_LIBS=ON`: Builds LAMMPS as a shared library (`.so`), required for the Python module.

### Compile

Compile the code using multiple cores (adjust `-j 8` based on your CPU cores).

```bash
make -j 8

```

---

## 4. Python Environment Setup

Install the LAMMPS Python interface and the necessary dependencies.

```bash
# Install the lammps python module into your current environment
make install-python

# Install dependencies for your ML model
pip install -r lmp_requirements.txt

# Install CuPy (Ensure the version matches your CUDA version)
# For CUDA 12.x:
pip install cupy-cuda12x 
# For CUDA 11.x, use: pip install cupy-cuda11x

```

---

## 5. Running LAMMPS

Below are the commands to run LAMMPS using the KOKKOS accelerator package on GPUs.

### Single GPU Execution

Run on 1 GPU without MPI.

```bash
lmp -k on g 1 -sf kk -pk kokkos newton on neigh half gpu/aware on -in test.in

```

### Multi-GPU Execution

Run on 2 GPUs using MPI.

```bash
mpirun -np 2 lmp -k on g 2 -sf kk -pk kokkos newton on neigh half gpu/aware on -in sl.in

```

### Runtime Flags Breakdown

* `-k on g X`: Enable KOKKOS and use **X** GPUs per node.
* `-sf kk`: **Suffix KOKKOS**. Automatically appends `/kk` to styles in the input script (e.g., `pair_style` becomes `pair_style/kk`).
* `-pk kokkos`: Modifies global KOKKOS parameters:
* `newton on`: Turns on Newton's 3rd law optimizations (often faster for GPUs).
* `neigh half`: Uses a half-neighbor list (often more efficient on GPUs).
* `gpu/aware on`: Optimizes MPI communication if using CUDA-aware MPI.



