import argparse
import os
import torch
from pathlib import Path

# Import the AlphaNet model wrapper and config
from alphanet.models.model import AlphaNetWrapper
from alphanet.config import All_Config

# Import the Python-level LAMMPS interface class
try:
    from alphanet.infer.lammps_mliap_alphanet import LAMMPS_MLIAP_ALPHANET
except ImportError:
    print("Could not import LAMMPS_MLIAP_ALPHANET.")
    print("Please ensure 'alphanet/infer/lammps_mliap_alphanet.py' exists.")
    exit(1)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert an AlphaNet model to LAMMPS ML-IAP format (Python Pickle)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config", "-c", required=True, type=str,
        help="Path to the model configuration JSON file",
    )
    parser.add_argument(
        "--checkpoint", "-m", required=True, type=str,
        help="Path to the trained model checkpoint (.ckpt)",
    )
    parser.add_argument(
        "--output", "-o", required=True, type=str,
        help="Output path to save the model (e.g., alphanet_lammps.pt)",
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="Device to load the model on ('cpu' or 'cuda')",
    )
    parser.add_argument(
        "--dtype", type=str, default="float64",
        choices=["float32", "float64"],
        help="Data type for the model",
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    device = torch.device(args.device)
    
    print(f"1. Loading configuration from {args.config}...")
    config_obj = All_Config().from_json(args.config)
    
    config_obj.model.dtype = "64" if args.dtype == "float64" else "32"
    
    print(f"2. Initializing AlphaNetWrapper (precision: {args.dtype}, device: {args.device})...")
    model_wrapper = AlphaNetWrapper(config_obj.model) 
    
    print(f"3. Loading weights from {args.checkpoint}...")
    ckpt = torch.load(args.checkpoint, map_location=device)
    
    if 'state_dict' in ckpt:
        state_dict = {k.replace('model.', ''): v for k, v in ckpt['state_dict'].items()}
        model_wrapper.model.load_state_dict(state_dict, strict=False)
    else:
        model_wrapper.load_state_dict(ckpt, strict=False)

    if args.dtype == "float64":
        model_wrapper.double()
    else:
        model_wrapper.float()
        
    model_wrapper.to(device).eval()
    
    print("4. Creating LAMMPS ML-IAP Interface Object...")
    lammps_interface_object = LAMMPS_MLIAP_ALPHANET(model_wrapper)
    
    if device.type == 'cuda':
        lammps_interface_object.model.cuda()
    
    print(f"5. Saving Python object to {args.output}...")
    # Using standard torch.save for Python pickle compatibility
    torch.save(lammps_interface_object, args.output)
    
    print("\n--- Success ---")
    print(f"Created LAMMPS model file: {args.output}")
    print("Usage in LAMMPS: pair_style mliap model/python ...")

if __name__ == "__main__":
    main()