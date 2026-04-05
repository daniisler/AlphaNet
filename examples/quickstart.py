#!/usr/bin/env python3
"""
AlphaNet Quick Start Script

Helps new users verify installation and get started quickly.
"""

import os
import sys
import json
import subprocess
from pathlib import Path

def print_header():
    print("=" * 60)
    print("🚀 AlphaNet Quick Start")
    print("=" * 60)
    print()

def check_installation():
    """Check if AlphaNet is properly installed"""
    print("📋 Checking installation status...")
    
    try:
        import alphanet
        print("   ✅ AlphaNet package installed")
    except ImportError:
        print("   ❌ AlphaNet package not installed")
        print("      Please run: pip install -e .")
        return False
    
    try:
        import torch
        print(f"   ✅ PyTorch {torch.__version__} installed")
        if torch.cuda.is_available():
            print(f"      CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            print("      ⚠️  CUDA not available, will use CPU")
    except ImportError:
        print("   ❌ PyTorch not installed")
        return False
    
    print()
    return True

def create_example_dataset():
    """Create example dataset directory structure"""
    print("📁 Creating example dataset directory...")
    
    dataset_dir = Path("dataset/example_data")
    dataset_dir.mkdir(parents=True, exist_ok=True)
    (dataset_dir / "raw").mkdir(exist_ok=True)
    
    print(f"   ✅ Created: {dataset_dir}")
    print("   💡 Place your data files in dataset/example_data/raw/")
    print()

def show_commands():
    """Display common commands"""
    print("📚 Common Commands:")
    print()
    print("1️⃣  Train a model:")
    print("   alpha-train examples/config_basic.json")
    print()
    print("2️⃣  Multi-GPU training:")
    print("   alpha-train examples/config_basic.json --num_devices 4")
    print()
    print("3️⃣  Resume training:")
    print("   alpha-train config.json --resume --ckpt_path checkpoint.ckpt")
    print()
    print("4️⃣  Finetune a model:")
    print("   alpha-train config.json --finetune pretrained.ckpt")
    print()
    print("5️⃣  Evaluate a model:")
    print("   alpha-eval -c config.json -m model.ckpt")
    print()
    print("6️⃣  Convert checkpoint:")
    print("   alpha-conv -i model.ckpt -o model.pt")
    print()

def main():
    print_header()
    
    # Check installation
    if not check_installation():
        print("❌ Installation check failed, please complete installation first")
        sys.exit(1)
    
    # Create example directory
    create_example_dataset()
    
    # Show commands
    show_commands()
    
    print("=" * 60)
    print("✨ Ready to go! Check examples/ directory for config files")
    print("=" * 60)

if __name__ == "__main__":
    main()
