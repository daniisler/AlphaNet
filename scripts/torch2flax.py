#!/usr/bin/env python3
import os
import logging
import numpy as np
import jax
import jax.numpy as jnp
from flax import serialization
from flax.core import unfreeze
import torch
import re
from collections import defaultdict
from typing import Dict, List, Any, Tuple, Union
from utils import process_positions_and_edges
from alphanet import AlphaNet_flax as AlphaNet

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

pos = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=jnp.float32)
z = jnp.array([1, 1, 1], dtype=jnp.int32)
natoms = jnp.array([3], dtype=jnp.int32)
batch = jnp.array([0, 0, 0], dtype=jnp.int32)
cell = jnp.array([[10, 0.0, 0.0], [0.0, 10, 0.0], [0.0, 0.0, 10]], dtype=jnp.float32)[jnp.newaxis, :, :]

graph_data = process_positions_and_edges(
    pos=pos,
    z=z,
    natoms=natoms,
    batch=batch,
    cell=cell,
    use_pbc=True,
    cutoff=5.0
)

class PyTorchToFlaxConverter:
    def __init__(self, flax_model, pt_path: str, flax_output_dir: str):
        """
        Args:
            flax_model: initialized Flax model
            pt_path: PyTorch path (.bin/.pth/.safetensors)
            flax_output_dir: output directory
        """
        self.flax_model = flax_model
        self.pt_path = pt_path
        self.output_dir = flax_output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.index_pattern = re.compile(r'\.(\d+)\.')
        
        self.special_mappings = {
            'model.a': 'a',
            'model.b': 'b',
            'model.kernel1': 'kernel1',
            'model.kernels_real': 'kernels_real',
            'model.kernels_imag': 'kernels_imag',
            
            'model.z_emb.weight': 'z_emb/embedding',
            'model.neighbor_emb.embedding.weight': 'neighbor_emb/Embed_0/embedding',
            'model.radial_emb.bessel_weights': 'radial_emb/bessel_weights',
            
            'model.radial_lin.0.weight': 'radial_lin/layers_0/kernel',
            'model.radial_lin.0.bias': 'radial_lin/layers_0/bias',
            'model.radial_lin.2.weight': 'radial_lin/layers_2/kernel',
            'model.radial_lin.2.bias': 'radial_lin/layers_2/bias',
            
            'model.S_vector.lin1.0.weight': 's_vector/Dense_0/kernel',
            'model.S_vector.lin1.0.bias': 's_vector/Dense_0/bias',
            
            'model.lin.0.weight': 'lin/layers_0/kernel',
            'model.lin.0.bias': 'lin/layers_0/bias',
            'model.lin.2.weight': 'lin/layers_2/kernel',
            'model.lin.2.bias': 'lin/layers_2/bias',
            
            'model.last_layer.weight': 'last_layer/kernel',
            'model.last_layer.bias': 'last_layer/bias',
            'model.last_layer_quantum.weight': 'last_layer_quantum/kernel',
            'model.last_layer_quantum.bias': 'last_layer_quantum/bias',
            
            'model.FTEs.0.vec_proj.weight': 'ftes_0/vec_proj/kernel',
            'model.FTEs.1.vec_proj.weight': 'ftes_1/vec_proj/kernel',
            'model.FTEs.2.vec_proj.weight': 'ftes_2/vec_proj/kernel',
            'model.FTEs.3.vec_proj.weight': 'ftes_3/vec_proj/kernel',
            
            'model.message_layers.0.fc_dx.weight': 'message_layers_0/fc_dx/kernel',
            'model.message_layers.0.fc_dx.bias': 'message_layers_0/fc_dx/bias',
            'model.message_layers.1.fc_dx.weight': 'message_layers_1/fc_dx/kernel',
            'model.message_layers.1.fc_dx.bias': 'message_layers_1/fc_dx/bias',
            'model.message_layers.2.fc_dx.weight': 'message_layers_2/fc_dx/kernel',
            'model.message_layers.2.fc_dx.bias': 'message_layers_2/fc_dx/bias',
            'model.message_layers.3.fc_dx.weight': 'message_layers_3/fc_dx/kernel',
            'model.message_layers.3.fc_dx.bias': 'message_layers_3/fc_dx/bias',
            
            
            'model.FTEs.0.xvec_proj.0.weight': 'ftes_0/xvec_proj/layers_0/kernel',
            'model.FTEs.0.xvec_proj.0.bias': 'ftes_0/xvec_proj/layers_0/bias',
            'model.FTEs.0.xvec_proj.2.weight': 'ftes_0/xvec_proj/layers_2/kernel',
            'model.FTEs.0.xvec_proj.2.bias': 'ftes_0/xvec_proj/layers_2/bias',
            'model.FTEs.1.xvec_proj.0.weight': 'ftes_1/xvec_proj/layers_0/kernel',
            'model.FTEs.1.xvec_proj.0.bias': 'ftes_1/xvec_proj/layers_0/bias',
            'model.FTEs.1.xvec_proj.2.weight': 'ftes_1/xvec_proj/layers_2/kernel',
            'model.FTEs.1.xvec_proj.2.bias': 'ftes_1/xvec_proj/layers_2/bias',
            'model.FTEs.2.xvec_proj.0.weight': 'ftes_2/xvec_proj/layers_0/kernel',
            'model.FTEs.2.xvec_proj.0.bias': 'ftes_2/xvec_proj/layers_0/bias',
            'model.FTEs.2.xvec_proj.2.weight': 'ftes_2/xvec_proj/layers_2/kernel',
            'model.FTEs.2.xvec_proj.2.bias': 'ftes_2/xvec_proj/layers_2/bias',
            'model.FTEs.3.xvec_proj.0.weight': 'ftes_3/xvec_proj/layers_0/kernel',
            'model.FTEs.3.xvec_proj.0.bias': 'ftes_3/xvec_proj/layers_0/bias',
            'model.FTEs.3.xvec_proj.2.weight': 'ftes_3/xvec_proj/layers_2/kernel',
            'model.FTEs.3.xvec_proj.2.bias': 'ftes_3/xvec_proj/layers_2/bias',
        }

    def _load_pytorch_state_dict(self) -> Dict[str, torch.Tensor]:
       
        if not os.path.exists(self.pt_path):
            raise FileNotFoundError(f"PyTorch weights file {self.pt_path} does not exist")
        
        logger.info(f"loading PyTorch weights: {self.pt_path}")
        state_dict = torch.load(self.pt_path, map_location="cpu")
        
        if hasattr(state_dict, "state_dict"):
            state_dict = state_dict.state_dict()
        
        logger.info(f"loaded {len(state_dict)} params successfully")
        return state_dict

    def _convert_param_name(self, pt_key: str) -> Tuple[str, bool]:
      
        if pt_key in self.special_mappings:
            return self.special_mappings[pt_key], False

        if 'message_layers' in pt_key:
            return self._convert_message_layer_name(pt_key)

        if 'FTEs' in pt_key:
            return self._convert_fte_name(pt_key)
        
        key = pt_key.replace("model.", "")

        key = self.index_pattern.sub(lambda m: f"_{m.group(1)}/", key)

        key = key.replace("S_vector", "s_vector")
        key = key.replace("weight", "kernel")
        key = key.replace("bias", "bias")
        
        if "layernorm" in key:
            key = key.replace("weight", "scale")
        
        return key, "kernel" in key and "layernorm" not in key

    def _convert_message_layer_name(self, pt_key: str) -> Tuple[str, bool]:
        
        parts = pt_key.split('.')
        layer_idx = parts[2]
        param_name = '.'.join(parts[3:])
        
        flax_prefix = f"message_layers_{layer_idx}/"
        
        if "diachi1" in pt_key:
            return flax_prefix + "diachi1", False
        elif "kernel_real" in pt_key:
            return flax_prefix + "kernel_real", False
        elif "kernel_imag" in pt_key:
            return flax_prefix + "kernel_imag", False
        elif "unitary" in pt_key:
            return flax_prefix + "unitary", False
        elif "scale.weight" in pt_key:
            return flax_prefix + "scale/kernel", True
        elif "scale.bias" in pt_key:
            return flax_prefix + "scale/bias", False
        elif "rbf_proj.weight" in pt_key:
            return flax_prefix + "rbf_proj/kernel", True
        elif "rbf_proj.bias" in pt_key:
            return flax_prefix + "rbf_proj/bias", False
        elif "x_layernorm.weight" in pt_key:
            return flax_prefix + "x_layernorm/scale", False
        elif "x_layernorm.bias" in pt_key:
            return flax_prefix + "x_layernorm/bias", False
        elif "dx_layer_norm.weight" in pt_key:
            return flax_prefix + "dx_layer_norm/scale", False
        elif "dx_layer_norm.bias" in pt_key:
            return flax_prefix + "dx_layer_norm/bias", False
        elif "fc_mps.weight" in pt_key:
            return flax_prefix + "fc_mps/kernel", True
        elif "fc_mps.bias" in pt_key:
            return flax_prefix + "fc_mps/bias", False
        elif "fc_dx.weight" in pt_key:
            return flax_prefix + "fc_dx/kernel", True
        elif "fc_dx.bias" in pt_key:
            return flax_prefix + "fc_dx/bias", False
        elif "dia.weight" in pt_key:
            return flax_prefix + "dia/kernel", True
        elif "dia.bias" in pt_key:
            return flax_prefix + "dia/bias", False
        
        if "dir_proj" in pt_key or "x_proj" in pt_key:
            sub_idx = parts[-2]
            param_type = parts[-1]
            
            layer_type = parts[3]
            if param_type == "weight":
                flax_name = f"{layer_type}/layers_{sub_idx}/kernel"
                return flax_prefix + flax_name, True
            else:
                flax_name = f"{layer_type}/layers_{sub_idx}/bias"
                return flax_prefix + flax_name, False
        
       
        if "diagonal" in pt_key:
            sub_idx = parts[-2]
            param_type = parts[-1]
            
            if param_type == "weight":
                flax_name = f"diagonal/layers_{sub_idx}/kernel"
                return flax_prefix + flax_name, True
            else:
                flax_name = f"diagonal/layers_{sub_idx}/bias"
                return flax_prefix + flax_name, False
        
       
        if "scale2.0.weight" in pt_key:
            return flax_prefix + "scale2/kernel", True
        elif "scale2.0.bias" in pt_key:
            return flax_prefix + "scale2/bias", False
        
        key = pt_key.replace("model.", "")
        key = key.replace(".", "/")
        key = key.replace("weight", "kernel")
        return flax_prefix + key, "kernel" in key

    def _convert_fte_name(self, pt_key: str) -> Tuple[str, bool]:
        parts = pt_key.split('.')
        layer_idx = parts[1]
        param_name = '.'.join(parts[2:])
        flax_prefix = f"ftes_{layer_idx}/"
        
        if "vec_proj.weight" in pt_key:
            return flax_prefix + "vec_proj/kernel", True
        
        if "xvec_proj" in pt_key:
            sub_idx = parts[-2]
            param_type = parts[-1]
            
            if param_type == "weight":
                flax_name = f"xvec_proj/layers_{sub_idx}/kernel"
                return flax_prefix + flax_name, True
            else:
                flax_name = f"xvec_proj/layers_{sub_idx}/bias"
                return flax_prefix + flax_name, False
        key = pt_key.replace("model.", "")
        key = key.replace(".", "/")
        key = key.replace("weight", "kernel")
        return flax_prefix + key, "kernel" in key

    def _reshape_parameter(self, pt_key: str, tensor: torch.Tensor) -> np.ndarray:
        array = tensor.numpy()
        flax_key, needs_transpose = self._convert_param_name(pt_key)
        if "kernel_real" in pt_key or "kernel_imag" in pt_key:
            return array
        
        if needs_transpose and array.ndim == 2:
            return array.T
        if "dir_proj" in pt_key or "x_proj" in pt_key or "diagonal" in pt_key:
            if array.ndim == 2:
                return array.T
        if "vec_proj" in pt_key and array.ndim == 2:
            return array.T
            
        if "xvec_proj" in pt_key and array.ndim == 2:
            return array.T
            
        if "last_layer" in pt_key and array.ndim == 2:
            return array.T
            
        if "radial_lin" in pt_key and array.ndim == 2:
            return array.T
            
        if "lin" in pt_key and array.ndim == 2:
            return array.T
        
        return array

    def convert(self) -> Dict:

        pt_state_dict = self._load_pytorch_state_dict()

        rng = jax.random.PRNGKey(0)
        flax_params = self.flax_model.init(rng, graph_data)

        logger.info("Flax params tree:")
        self._analyze_flax_structure(flax_params['params'])

        flax_state_dict = unfreeze(flax_params)
        missing_keys = []
        shape_mismatches = []
        conversion_map = {}
        
        for pt_key, pt_value in pt_state_dict.items():
            if "num_batches_tracked" in pt_key or "running_" in pt_key:
                continue
            flax_key, transpose_needed = self._convert_param_name(pt_key)
            conversion_map[pt_key] = flax_key
            flax_value = self._reshape_parameter(pt_key, pt_value)

            keys = flax_key.split('/')
            target = flax_state_dict['params'] 
            
            path_found = True
            current_path = []
            
            for k in keys[:-1]:
                current_path.append(k)
                if k not in target:
                    path_found = False
                    logger.debug(f"Path does not exist: {'/'.join(current_path)}")
                    break
                target = target[k]
            
            if not path_found:
                missing_keys.append(pt_key)
                logger.info(f"No path: {pt_key} -> {flax_key}")
                continue
                
            final_key = keys[-1]
            if final_key in target:
                if target[final_key].shape != flax_value.shape:
                    shape_mismatches.append((pt_key, flax_key, pt_value.shape, target[final_key].shape))
                    logger.warning(f"shape miss match: {pt_key} -> {flax_key} "
                                  f"(PyTorch: {pt_value.shape} -> Flax: {target[final_key].shape})")
                
                    if pt_value.numel() == target[final_key].size:
                        logger.info(f"Reshaping: {pt_key}")
                        flax_value = pt_value.numpy().reshape(target[final_key].shape)
                        target[final_key] = flax_value
                    else:
                        continue
                else:
                    target[final_key] = flax_value
            else:
                missing_keys.append(pt_key)
                logger.info(f"Can not find: {pt_key} -> {flax_key}")

        self._log_unmatched_keys(missing_keys, shape_mismatches)

        self._log_conversion_results(conversion_map, missing_keys, shape_mismatches)

        self._save_flax_checkpoint(flax_state_dict)
        return flax_state_dict

    def _log_unmatched_keys(self, missing_keys, shape_mismatches):
       
        if missing_keys or shape_mismatches:
            logger.warning("=" * 80)
            logger.warning("miss match params:")
            logger.warning("=" * 80)
          
            if missing_keys:
                logger.warning(f"missing keys ({len(missing_keys)}):")
                for i, key in enumerate(missing_keys[:10]): 
                    logger.warning(f"  {i+1}. {key}")
                if len(missing_keys) > 10:
                    logger.warning(f"  another {len(missing_keys)-10} keys...")

            if shape_mismatches:
                logger.warning(f"shape miss match ({len(shape_mismatches)}):")
                for i, (pt_key, flax_key, pt_shape, flax_shape) in enumerate(shape_mismatches[:5]):  
                    logger.warning(f"  {i+1}. {pt_key} -> {flax_key}")
                    logger.warning(f"      PyTorchshape: {pt_shape}, Flaxshape: {flax_shape}")
                if len(shape_mismatches) > 5:
                    logger.warning(f"  Another {len(shape_mismatches)-5} shape miss match")
            
            logger.warning("=" * 80)
        else:
            logger.info("All matched！")

    def _analyze_flax_structure(self, params, depth=0, prefix=""):
       
        indent = "  " * depth
        for key, value in params.items():
            if isinstance(value, dict):
                logger.info(f"{indent}{prefix}{key}/ (dict)")
                self._analyze_flax_structure(value, depth+1, prefix=f"{prefix}{key}/")
            elif hasattr(value, 'shape'):
                logger.info(f"{indent}{prefix}{key}: {value.shape}")
            else:
                logger.info(f"{indent}{prefix}{key}: {type(value)}")

    def _log_conversion_results(self, conversion_map, missing_keys, shape_mismatches):
        
        conversion_file = os.path.join(self.output_dir, "conversion_map.txt")
        with open(conversion_file, 'w') as f:
            f.write("Map PyTorch to Flax:\n")
            f.write("-" * 80 + "\n")
            for pt_key, flax_key in conversion_map.items():
                f.write(f"{pt_key} -> {flax_key}\n")
        
      
        if missing_keys:
            missing_file = os.path.join(self.output_dir, "missing_keys.txt")
            with open(missing_file, 'w') as f:
                f.write(f"missing {len(missing_keys)} params:\n")
                f.write("-" * 80 + "\n")
                for key in missing_keys:
                    f.write(f"{key}\n")
            
            logger.warning(f"Con not match {len(missing_keys)} parameters, see: {missing_file}")
        else:
            logger.info("All parameters match")
        
      
        if shape_mismatches:
            shape_file = os.path.join(self.output_dir, "shape_mismatches.txt")
            with open(shape_file, 'w') as f:
                f.write(f"shape mismatch ({len(shape_mismatches)}):\n")
                f.write("-" * 100 + "\n")
                f.write("PyTorch\tFlax\tPyTorch shape\tFlax shape\n")
                f.write("-" * 100 + "\n")
                for item in shape_mismatches:
                    f.write(f"{item[0]}\t{item[1]}\t{item[2]}\t{item[3]}\n")
            
            logger.warning(f" {len(shape_mismatches)} shape miss match : {shape_file}")

    def _save_flax_checkpoint(self, state_dict: Dict):
        
        output_path = os.path.join(self.output_dir, 'flax_model.ckpt')
        with open(output_path, 'wb') as f:
            f.write(serialization.to_bytes(state_dict))
        logger.info(f"Flax has been saved to {output_path}")

class Config:
    hidden_channels = 176
    num_radial = 8
    cutoff = 5.0
    num_layers = 4
    head = 16
    chi1 = 24
    chi2 = 6
    mp_chi1 = 24
    hidden_channels_chi = 96
    dtype = "32"
    eps = 1e-8
    a = 1
    b = 1
    readout = 'sum'
    use_sigmoid = False
    output_dim = 1
    compute_forces = True
    compute_stress = True
    main_chi1 = 24
    has_dropout_flag = False
    has_norm_before_flag = True
    has_norm_after_flag = False
    reduce_mode = 'sum'

config = Config()

# 使用示例
if __name__ == "__main__":
   
    flax_model = AlphaNet(config)
    
    converter = PyTorchToFlaxConverter(
        flax_model=flax_model,
        pt_path="alex_0410.ckpt",  # PyTorch模型路径
        flax_output_dir="./flax_model"
    )
    
    # 3. 执行转换
    flax_params = converter.convert()
    
    logger.info("参数转换完成")