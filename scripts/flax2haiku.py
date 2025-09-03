import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
import haiku as hk
from typing import Optional, Dict, Any
import orbax.checkpoint as orbax
import os
from rich import print
from rich.tree import Tree
from rich.table import Table
from rich.panel import Panel
from rich.console import Console
from dataclasses import dataclass
import pickle
import sys
from alpha_jax1 import AlphaNet_flax
from alpha_hiku import AlphaNet_hiku
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
    dtype = "64"
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

@dataclass
class MoleculeData:
   
    pos: jnp.ndarray         
    batch: jnp.ndarray        
    z: jnp.ndarray            
    edge_index: jnp.ndarray   
    edge_attr: jnp.ndarray    
    edge_vec: jnp.ndarray    
    shift: Optional[jnp.ndarray] = None  
    cell: Optional[jnp.ndarray] = None 
    
    def __post_init__(self):
       
        num_atoms = self.pos.shape[0]
        num_edges = self.edge_index.shape[1]
        
        assert self.batch.shape == (num_atoms,), "batch shape mismatch"
        assert self.z.shape == (num_atoms,), "z shape mismatch"
        assert self.edge_attr.shape == (num_edges,), "edge_attr shape mismatch"
        assert self.edge_vec.shape == (num_edges, 3), "edge_vec shape mismatch"
        
        if self.shift is not None:
            assert self.shift.shape == (num_edges, 3), "shift shape mismatch"
        
        if self.cell is not None:
            assert self.cell.shape == (3, 3), "cell shape mismatch"

def create_dummy_data(num_atoms=10, num_edges=20, dtype=jnp.float32, periodic=False):
   
    
    pos = jnp.array(np.random.randn(num_atoms, 3), dtype=dtype)
  
    batch = jnp.zeros(num_atoms, dtype=jnp.int32)
    z = jnp.array(np.random.randint(0, 95, num_atoms), dtype=jnp.int32)
    edge_index = np.zeros((2, num_edges), dtype=np.int32)
    for i in range(num_edges):
        edge_index[0, i] = np.random.randint(0, num_atoms)
        edge_index[1, i] = np.random.randint(0, num_atoms)
    edge_index = jnp.array(edge_index)
    edge_attr = jnp.array(np.random.rand(num_edges), dtype=dtype)
    edge_vec = jnp.array(np.random.randn(num_edges, 3), dtype=dtype)
    shift = None
    cell = None
    if periodic:
        shift = jnp.array(np.random.randn(num_edges, 3), dtype=dtype)
        cell = jnp.array(np.random.randn(3, 3), dtype=dtype)
    
    return MoleculeData(
        pos=pos,
        batch=batch,
        z=z,
        edge_index=edge_index,
        edge_attr=edge_attr,
        edge_vec=edge_vec,
        shift=shift,
        cell=cell
    )

class ParameterConverter:
    def __init__(self, config):
        self.config = config
        self.mapping = self._create_mapping()
    
    def flax_to_haiku(self, flax_params: Dict[str, Any]) -> Dict[str, Any]:
       
        haiku_params = {}
        
        for flax_path, haiku_path in self.mapping.items():
            flax_value = self._get_nested_value(flax_params, flax_path)
            
            if flax_value is not None:
                self._set_nested_value(haiku_params, haiku_path, flax_value)
        
        return haiku_params
    
    def _get_nested_value(self, params, path):
        parts = path.split('/')
        current = params
        
        for part in parts:
            if part not in current:
                return None
            current = current[part]
        
        return current
    
    def _set_nested_value(self, params, path, value):
        parts = path.split('/')
        current = params
        
        for i, part in enumerate(parts[:-1]):
            if part not in current:
                current[part] = {}
            current = current[part]
        
        current[parts[-1]] = value
    
    def _create_mapping(self):
        mapping = {
           
            'params/z_emb/embedding': 'alpha_net_hiku/~/embed/embeddings',

            'params/radial_emb/bessel_weights': 'alpha_net_hiku/~/rbf_emb/bessel_weights',
            'params/radial_lin/layers_0/kernel': 'alpha_net_hiku/~/linear/w',
            'params/radial_lin/layers_0/bias':   'alpha_net_hiku/~/linear/b',
            'params/radial_lin/layers_2/kernel': 'alpha_net_hiku/~/linear_1/w',
            'params/radial_lin/layers_2/bias':   'alpha_net_hiku/~/linear_1/b',

            'params/neighbor_emb/Embed_0/embedding': 'alpha_net_hiku/~/neighbor_emb/embed/embeddings',

            'params/s_vector/Dense_0/kernel': 'alpha_net_hiku/~/s_vector/linear/w',
            'params/s_vector/Dense_0/bias':   'alpha_net_hiku/~/s_vector/linear/b',

            'params/lin/layers_0/kernel': 'alpha_net_hiku/~/linear_2/w',
            'params/lin/layers_0/bias':   'alpha_net_hiku/~/linear_2/b',
            'params/lin/layers_2/kernel': 'alpha_net_hiku/~/linear_3/w',
            'params/lin/layers_2/bias':   'alpha_net_hiku/~/linear_3/b',

            'params/kernel1':      'alpha_net_hiku/kernel1',
            'params/kernels_real': 'alpha_net_hiku/kernels_real',
            'params/kernels_imag': 'alpha_net_hiku/kernels_imag',
            'params/a': 'alpha_net_hiku/a',
            'params/b': 'alpha_net_hiku/b',

            'params/last_layer/kernel':          'alpha_net_hiku/~/linear_4/w',
            'params/last_layer/bias':            'alpha_net_hiku/~/linear_4/b',
            'params/last_layer_quantum/kernel':  'alpha_net_hiku/~/linear_5/w',
            'params/last_layer_quantum/bias':    'alpha_net_hiku/~/linear_5/b',
        }

        def suf(i): 
            return "" if i == 0 else f"_{i}"

        for i in range(self.config.num_layers):
            emp = f'alpha_net_hiku/~/equi_message_passing{suf(i)}'
            fte = f'alpha_net_hiku/~/fte{suf(i)}'

            mapping.update({
                f'params/message_layers_{i}/x_layernorm/scale':         f'{emp}/layer_norm/scale',
                f'params/message_layers_{i}/x_layernorm/bias':          f'{emp}/layer_norm/offset',
                f'params/message_layers_{i}/x_proj/layers_0/kernel':    f'{emp}/~_build_x_proj/linear/w',
                f'params/message_layers_{i}/x_proj/layers_0/bias':      f'{emp}/~_build_x_proj/linear/b',
                f'params/message_layers_{i}/x_proj/layers_2/kernel':    f'{emp}/~_build_x_proj/linear_1/w',
                f'params/message_layers_{i}/x_proj/layers_2/bias':      f'{emp}/~_build_x_proj/linear_1/b',
                f'params/message_layers_{i}/rbf_proj/kernel':           f'{emp}/linear/w',
                f'params/message_layers_{i}/rbf_proj/bias':             f'{emp}/linear/b',
                f'params/message_layers_{i}/dir_proj/layers_0/kernel':  f'{emp}/~_build_dir_proj/linear/w',
                f'params/message_layers_{i}/dir_proj/layers_0/bias':    f'{emp}/~_build_dir_proj/linear/b',
                f'params/message_layers_{i}/dir_proj/layers_2/kernel':  f'{emp}/~_build_dir_proj/linear_1/w',
                f'params/message_layers_{i}/dir_proj/layers_2/bias':    f'{emp}/~_build_dir_proj/linear_1/b',
                f'params/message_layers_{i}/scale/kernel':              f'{emp}/~message/linear/w',
                f'params/message_layers_{i}/scale/bias':                f'{emp}/~message/linear/b',
                f'params/message_layers_{i}/kernel_real':               f'{emp}/kernel_real',
                f'params/message_layers_{i}/kernel_imag':               f'{emp}/kernel_imag',
                f'params/message_layers_{i}/diachi1':                   f'{emp}/diachi1',
                f'params/message_layers_{i}/diagonal/layers_0/kernel':  f'{emp}/~_build_diagonal/linear/w',
                f'params/message_layers_{i}/diagonal/layers_0/bias':    f'{emp}/~_build_diagonal/linear/b',
                f'params/message_layers_{i}/diagonal/layers_2/kernel':  f'{emp}/~_build_diagonal/linear_1/w',
                f'params/message_layers_{i}/diagonal/layers_2/bias':    f'{emp}/~_build_diagonal/linear_1/b',
                f'params/message_layers_{i}/dia/kernel':                f'{emp}/~message/linear_1/w',
                f'params/message_layers_{i}/dia/bias':                  f'{emp}/~message/linear_1/b',
                f'params/message_layers_{i}/fc_mps/kernel':             f'{emp}/~/linear/w',
                f'params/message_layers_{i}/fc_mps/bias':               f'{emp}/~/linear/b',
                f'params/message_layers_{i}/dx_layer_norm/scale':       f'{emp}/layer_norm_1/scale',
                f'params/message_layers_{i}/dx_layer_norm/bias':        f'{emp}/layer_norm_1/offset',
                f'params/message_layers_{i}/scale2/kernel':             f'{emp}/linear_1/w',
                f'params/message_layers_{i}/scale2/bias':               f'{emp}/linear_1/b',
                
                # f'params/message_layers_{i}/<your_name>/kernel':      f'{emp}/~message/linear_3/w',
                # f'params/message_layers_{i}/<your_name>/bias':        f'{emp}/~message/linear_3/b',
            })

           
            mapping.update({
                f'params/ftes_{i}/vec_proj/kernel':                 f'{fte}/~/linear/w',
                f'params/ftes_{i}/xvec_proj/layers_0/kernel':       f'{fte}/~/linear_1/w',
                f'params/ftes_{i}/xvec_proj/layers_0/bias':         f'{fte}/~/linear_1/b',
                f'params/ftes_{i}/xvec_proj/layers_2/kernel':       f'{fte}/~/linear_2/w',
                f'params/ftes_{i}/xvec_proj/layers_2/bias':         f'{fte}/~/linear_2/b',
            })

        return mapping
def main():
   
    config = Config()
    
    console = Console()
    
    console.print("\n[bold cyan]step 1: create dummy data[/bold cyan]")
    dummy_data = create_dummy_data(
        num_atoms=5, 
        num_edges=8,
        dtype=jnp.float32 if config.dtype == "32" else jnp.float64,
        periodic=False
    )
    console.print(f"Created {dummy_data.pos.shape[0]} atoms and {dummy_data.edge_index.shape[1]} edges")
    
    console.print("\n[bold cyan]step2: init Flax model [/bold cyan]")
    flax_model = AlphaNet_flax(config)
    rng = jax.random.PRNGKey(0)
    flax_params = flax_model.init(rng, dummy_data)
    
    console.print("\n[bold yellow]Flax model architexture :[/bold yellow]")
    console.print(flax_params)

    def remove_empty_items(params):
        
        if isinstance(params, dict):
            return {k: remove_empty_items(v) for k, v in params.items() if v is not None}
        elif isinstance(params, list):
            return [remove_empty_items(v) for v in params if v is not None]
        elif hasattr(params, "shape") and params.size == 0:
            return None  
        else:
            return params
    
    cleaned_flax_params = remove_empty_items(flax_params)
    checkpoint_dir = os.path.abspath("flax_checkpoint")
    os.makedirs(checkpoint_dir, exist_ok=True)
    orbax_checkpointer = orbax.PyTreeCheckpointer()
    checkpoint_path = os.path.join(checkpoint_dir, "checkpoint")
    orbax_checkpointer.save(
        checkpoint_path, 
        cleaned_flax_params
    )
    console.print(f"Flax has been saved to: {checkpoint_dir}")
    console.print("\n[bold cyan]step 3: loaded Flax checkpoint[/bold cyan]")
    restored_params = orbax_checkpointer.restore(checkpoint_path)
    console.print("Flax has been loaded")
    
    console.print("\n[bold cyan]step 4: Convert to Haiku formatt[/bold cyan]")
    converter = ParameterConverter(config)
    haiku_params = converter.flax_to_haiku(restored_params)
    console.print("completed!")
    haiku_checkpoint_dir = os.path.abspath("haiku_checkpoint")
    os.makedirs(haiku_checkpoint_dir, exist_ok=True)
    
    haiku_params_path = os.path.join(haiku_checkpoint_dir, "haiku_params.pkl")
    with open(haiku_params_path, "wb") as f:
        pickle.dump(haiku_params, f)
    console.print(f" Converted Haiku params has been saved to: {haiku_checkpoint_dir}")
    
    console.print("\n[bold cyan]step5: initializ and loade Haiku model[/bold cyan]")

    def haiku_fn(data):
        model = AlphaNet_hiku(config)
        return model(data)

    transformed = hk.transform(haiku_fn)
    rng = jax.random.PRNGKey(0)

    console.print("[bold yellow]initializing Haiku parameters...[/bold yellow]")
    init_params = transformed.init(rng, dummy_data)

    console.print("[bold yellow]loading converted params...[/bold yellow]")
    with open(haiku_params_path, "rb") as f:
        haiku_params_converted = pickle.load(f)

    def flatten_leaf_arrays(tree, prefix=""):
        leaves = {}
        if isinstance(tree, dict):
            for k, v in tree.items():
                new_prefix = f"{prefix}/{k}" if prefix else k
                leaves.update(flatten_leaf_arrays(v, new_prefix))
        else:
            if hasattr(tree, "shape"): 
                leaves[prefix] = tree
        return leaves

    def convert_dict_keys(old_dict):
        new_dict = {}
        for key, value in old_dict.items():
            parts = key.split('/')
            module_path = '/'.join(parts[:-1])

            if module_path not in new_dict:
                new_dict[module_path] = {}

            param_name = parts[-1]
            new_dict[module_path][param_name] = value
        
        return new_dict

    base_leaves = flatten_leaf_arrays(init_params)
    override_leaves = flatten_leaf_arrays(haiku_params_converted)

    missing = sorted(set(base_leaves) - set(override_leaves)) 
    extra   = sorted(set(override_leaves) - set(base_leaves))  

    shape_mismatch = []
    for k in override_leaves:
        if k in base_leaves and base_leaves[k].shape != override_leaves[k].shape:
            shape_mismatch.append((k, base_leaves[k].shape, override_leaves[k].shape))

    if missing:
        print("[red]Missing; Flax does not provide some of the parameters[/red]")
        for p in missing[:50]:
            print("  ", p)
        raise ValueError(f"In all {len(missing)} missing")

    if extra:
        print("[yellow]Flax provide some extra parameters[/yellow]")
        for p in extra[:50]:
            print("  ", p)

    if shape_mismatch:
        for p, s0, s1 in shape_mismatch[:50]:
            print(f"[yellow]{p} shape miss match: Haiku{tuple(s0)} vs Flax{tuple(s1)}[/yellow]")
        raise ValueError(f" {len(shape_mismatch)} miss match")

    
    merged_leaves = base_leaves.copy()
    merged_leaves.update(override_leaves)

    merged_params = convert_dict_keys(merged_leaves)
    
if __name__ == "__main__":
   
    try:
        main()
    except Exception as e:
        print(f"[bold red]error: {e}[/bold red]")
        import traceback
        traceback.print_exc()
        sys.exit(1)