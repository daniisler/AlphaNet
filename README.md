<p align="center">
  <img src="./old_README/logo.png" alt="AlphaNet Logo" width="80" height="auto">
</p>

<h1 align="center">AlphaNet</h1>

<p align="center">
  <b>Local Frame-based Equivariant Neural Network Potential for Atomistic Simulations</b><br>
  <i>Efficient and accurate machine learning potential for molecular dynamics</i>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2501.07155">
    <img src="https://img.shields.io/badge/arXiv-2501.07155-B31B1B.svg" alt="arXiv">
  </a>
  <img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.1+-EE4C2C.svg" alt="PyTorch">
  <img src="https://img.shields.io/badge/License-GNU-green.svg" alt="License">
</p>

---

## 📚 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Changelog](#changelog)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage Examples](#usage-examples)
- [LAMMPS Integration](#lammps-integration)
- [Pretrained Models](#pretrained-models)
- [FAQ](#faq)
- [Citation](#citation)

---

## 🎯 Overview

**AlphaNet** is a local frame-based equivariant neural network model designed for efficient and accurate atomistic simulations. It leverages local geometric structures of atomic environments through the construction of equivariant local frames and learnable frame transitions.

### Key Innovations

- 🏗️ **Local Frame Equivariance** - Constructs equivariant frames from local geometry
- 🔄 **Learnable Frame Transitions** - Dynamically adapts to different atomic environments
- 🔬 **Multi-body Message Passing** - Inspired by quantum mechanics, uses matrix product state contractions for efficient multi-body interactions
- ⚡ **Computational Efficiency** - Achieves excellent balance between accuracy and speed
- 📈 **Scalability** - Works across various system sizes and datasets

---

## ✨ Features

- ✅ Joint prediction of energy, forces, and stress
- ✅ ZBL repulsive potential integration (short-range interactions)
- ✅ LAMMPS molecular dynamics support
- ✅ PyTorch Lightning training framework
- ✅ Multi-GPU training support
- ✅ ASE calculator interface
- ✅ Data import from DeepMD and extxyz formats

---

## 📝 Changelog

### v0.1.2-beta
- **Added LAMMPS ML-IAP interface** - Run MD simulations with AlphaNet potentials
- **Model architecture refinements** - Improved network structure for better performance
- **Added finetuning capability** - Fine-tune pretrained models on new datasets

---

## 🛠️ Installation

### Requirements

- Python ≥ 3.8
- CUDA Toolkit 11.x or 12.x (recommended for GPU acceleration)
- Git

### Step 1: Create Conda Environment

```bash
conda create -n alphanet python=3.9
conda activate alphanet
```

### Step 2: Clone Repository

```bash
git clone https://github.com/zmyybc/AlphaNet.git
cd AlphaNet
git checkout lammps  # For LAMMPS integration
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Install AlphaNet

```bash
pip install -e .
```

> 💡 **Tip**: The `-e` flag installs in editable mode, so code changes are reflected without reinstallation.

### Verify Installation

```bash
alpha-train --help
```

---

## 🚀 Quick Start

### 1. Prepare Dataset

AlphaNet supports two data formats:

#### Convert from DeepMD format

```bash
python scripts/dp2pic_batch.py
```

#### Convert from extxyz format

```bash
python scripts/xyz2pic.py --input data.xyz --output dataset/
```

#### Dataset Directory Structure

```
dataset/
├── my_dataset_1/           # Custom dataset name
│   ├── raw/               # Raw data files
│   └── processed/         # Preprocessed data (auto-generated)
└── my_dataset_2/
    ├── raw/
    └── processed/
```

### 2. Configure Training

Create config file `config.json`:

```json
{
  "data": {
    "dataset_name": "my_dataset_1",
    "root": "dataset/",
    "train_size": null,
    "valid_size": null
  },
  "model": {
    "num_layers": 3,
    "hidden_channels": 128,
    "cutoff": 5.0,
    "compute_forces": true,
    "compute_stress": false,
    "zbl": false
  },
  "train": {
    "epochs": 1000,
    "batch_size": 32,
    "lr": 0.0005,
    "energy_coef": 1.0,
    "force_coef": 100.0,
    "device": "cuda"
  }
}
```

> 📖 **Detailed config options**: See comments in `alphanet/config.py`

### 3. Start Training

```bash
alpha-train config.json
```

#### Multi-GPU Training

```bash
alpha-train config.json --num_devices 4
```

#### Resume from Checkpoint

```bash
alpha-train config.json --resume --ckpt_path path/to/checkpoint.ckpt
```

#### Finetune Pretrained Model

```bash
alpha-train config.json --finetune path/to/pretrained.ckpt
```

---

## 💻 Usage Examples

### Model Evaluation

```bash
alpha-eval -c config.json -m path/to/model.ckpt
```

### Convert Checkpoint Format

Convert Lightning checkpoint to standard PyTorch format:

```bash
alpha-conv -i model.ckpt -o model_state_dict.pt
```

### Using ASE Calculator in Python

```python
from alphanet.infer.calc import AlphaNetCalculator
from alphanet.config import All_Config
from ase.build import bulk

# Create copper crystal structure
atoms = bulk('Cu', 'fcc', a=3.6, cubic=True)

# Load AlphaNet calculator
calculator = AlphaNetCalculator(
    ckpt_path='./pretrained/OMA/alex_0410.ckpt',
    device='cuda',
    precision='32',
    config=All_Config().from_json('./pretrained/OMA/oma.json'),
)

# Compute energy and forces
atoms.calc = calculator
energy = atoms.get_potential_energy()
forces = atoms.get_forces()

print(f"Energy: {energy:.4f} eV")
print(f"Forces shape: {forces.shape}")
```

---

## 🔗 LAMMPS Integration

AlphaNet supports LAMMPS as a machine learning potential with GPU acceleration.

### Quick Usage

1. **Prepare Model File**

```bash
python alphanet/create_lammps_model.py \
    --config ./pretrained/OMA/oma.json \
    --checkpoint ./pretrained/OMA/alex_0410.ckpt \
    --output ./alphanet_lammps.pt
```

2. **Create LAMMPS Input** `run.in`

```lammps
units         metal
atom_style    atomic
newton        on

# Read structure
read_data     structure.data

# Use AlphaNet potential
pair_style    mliap unified alphanet_lammps.pt 0
pair_coeff    * * 

# Run settings
timestep      0.001
run           1000
```

3. **Run Simulation**

```bash
python run_lammps.py
```

### Detailed Installation Guide

See [mliap_lammps.md](mliap_lammps.md) for complete LAMMPS build and configuration instructions.

---

## 🤖 Pretrained Models

| Model | Dataset | Description |
|-------|---------|-------------|
| [AlphaNet-oma-v1.5](pretrained/OMA) | OMAT24 + sALEX + MPtrj | General-purpose material potential |

### Using Pretrained Models

```python
from alphanet.infer.calc import AlphaNetCalculator

calc = AlphaNetCalculator(
    ckpt_path='./pretrained/OMA/alex_0410.ckpt',
    config=All_Config().from_json('./pretrained/OMA/oma.json'),
    device='cuda'
)
```

---

## ❓ FAQ

### Q: Getting NaN loss during training?

Ensure gradient clipping is enabled. If training in custom code:

```python
from pytorch_lightning import Trainer

trainer = Trainer(
    gradient_clip_val=1.0,  # Enable gradient clipping
    ...
)
```

### Q: How to adjust energy/force/stress loss weights?

Modify in config file:

```json
{
  "train": {
    "energy_coef": 4.0,    # Energy loss weight
    "force_coef": 100.0,   # Force loss weight
    "stress_coef": 100.0   # Stress loss weight
  }
}
```

### Q: Which elements are supported?

Current pretrained models support various elements depending on the training dataset. For new materials, we recommend finetuning with datasets containing your target elements.

### Q: How to report issues?

Please open an issue on [GitHub Issues](https://github.com/zmyybc/AlphaNet/issues).

---

## 📄 License

This project is licensed under the GNU License - see [LICENSE](LICENSE) file.

## 🙏 Acknowledgments

Thanks to all contributors and the community. Special thanks to:
- PyTorch Lightning team
- LAMMPS development team
- ASE development team

## 📖 Citation

If you use AlphaNet in your research, please cite:

```bibtex
@article{alphanet2025,
  title={AlphaNet: Scaling Up Local-frame-based Interatomic Potential},
  author={...},
  journal={arXiv preprint arXiv:2501.07155},
  year={2025}
}
```

---

<p align="center">
  <i>Made with ❤️ for the atomistic simulation community</i>
</p>
