import argparse
import os
import torch
from pathlib import Path

# Import the AlphaNet model wrapper and config
from alphanet.models.model import AlphaNetWrapper
from alphanet.config import All_Config

# Import the NEWLY created ML-IAP wrapper
# This is the module we will JIT compile
try:
    from alphanet.infer.lammps_mliap_alphanet import AlphaNetEdgeForcesWrapper
except ImportError:
    print("Could not import AlphaNetEdgeForcesWrapper.")
    print("Please ensure 'alphanet/calculators/lammps_mliap_alphanet.py' exists.")
    exit(1)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert an AlphaNet model to LAMMPS ML-IAP format (.pt)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        "-c",
        required=True,
        type=str,
        help="Path to the model configuration JSON file",
    )
    parser.add_argument(
        "--checkpoint",
        "-m",
        required=True,
        type=str,
        help="Path to the trained model checkpoint (.ckpt)",
    )
    parser.add_argument(
        "--output",
        "-o",
        required=True,
        type=str,
        help="Output path to save the JIT-compiled LAMMPS model (e.g., alphanet_lammps.pt)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to load the model on ('cpu' or 'cuda')",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float64",
        choices=["float32", "float64"],
        help="Data type for the model (LAMMPS ML-IAP recommends float64)",
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    device = torch.device(args.device)
    
    print(f"1. Loading configuration from {args.config}...")
    config_obj = All_Config().from_json(args.config)
    
    # Override config with command-line arguments for consistency
    config_obj.model.dtype = "64" if args.dtype == "float64" else "32"
    config_obj.model.compute_forces = True  # Required
    config_obj.model.compute_stress = False # Not used by ML-IAP
    
    print(f"2. Initializing AlphaNetWrapper (precision: {args.dtype}, device: {args.device})...")
    # We use AlphaNetWrapper to easily load the model and state dict
    model_wrapper = AlphaNetWrapper(config_obj.model) 
    
    print(f"3. Loading weights from {args.checkpoint}...")
    ckpt = torch.load(args.checkpoint, map_location=device)
    
    # Handle checkpoints from PyTorch Lightning
    if 'state_dict' in ckpt:
        state_dict = ckpt['state_dict']
        # Remove 'model.' prefix if present
        state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
        model_wrapper.model.load_state_dict(state_dict,strict=False)
        print("   Successfully loaded weights from 'state_dict'.")
    else:
        # Handle raw model state_dict
        try:
            model_wrapper.load_state_dict(ckpt, strict=False)
            print("   Successfully loaded weights from raw state_dict.")
        except Exception as e:
            raise ValueError(
                f"Could not load state_dict from checkpoint. "
                f"Ensure it is a PyTorch Lightning .ckpt file. Error: {e}"
            )

    # Set precision and evaluation mode
    if args.dtype == "float64":
        model_wrapper.double()
    else:
        model_wrapper.float()
        
    model_wrapper.to(device).eval()
    
    print("4. Creating LAMMPS ML-IAP EdgeForces wrapper...")
    # Extract the *internal* AlphaNet model
    internal_model = model_wrapper.model
    
    # Wrap it with the JIT-scriptable ML-IAP wrapper
    lammps_model = AlphaNetEdgeForcesWrapper(internal_model)
    lammps_model.to(device).eval()

    print("5. JIT-compiling model (torch.jit.script)...")
    try:
        lammps_model_compiled = torch.jit.script(lammps_model)
    except Exception as e:
        print("\n--- JIT Compilation FAILED ---")
        print(f"Error: {e}")
        print("This often means the AlphaNetEdgeForcesWrapper.forward method")
        print("contains Python-only syntax (like dynamic attributes).")
        print("Please check the implementation in 'alphanet/calculators/lammps_mliap_alphanet.py'.")
        return

    print(f"6. Saving compiled model to {args.output}...")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lammps_model_compiled.save(args.output)
    
    print("\n--- Success ---")
    print(f"Created LAMMPS ML-IAP model: {args.output}")
    print("You can now use this file with 'pair_style mliap/torch' in LAMMPS.")

if __name__ == "__main__":
    main()