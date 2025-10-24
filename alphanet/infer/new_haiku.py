import jax
import jax.numpy as jnp
from jax import value_and_grad
from ase.calculators.calculator import Calculator
from ase.data import atomic_numbers
import haiku as hk
import pickle
import numpy as np
from alphanet.models.graph_jax import process_positions_and_edges
from alphanet.models.alpha_haiku import AlphaNet_hiku
import time
from matscipy.neighbours import neighbour_list
from functools import partial
import hashlib

class AlphaNetCalculator(Calculator):
    implemented_properties = ['energy', 'free_energy', 'forces', 'stress']

    def __init__(self, ckpt_path, config, device='cpu', precision='32', **kwargs):
        Calculator.__init__(self, **kwargs)
        self.config = config
        self.rng = jax.random.PRNGKey(0)

        def forward_fn(graph_data):
            model = AlphaNet_hiku(config)
            return model(graph_data)

        self.transform = hk.transform(forward_fn)
        graph_data = create_dummy_data()
        init_params = self.transform.init(self.rng, graph_data)

        self.apply_fn = jax.jit(self.transform.apply)
        self.process_positions_and_edges_fn = jax.jit(process_positions_and_edges)

        self.precision = jnp.float32 if precision == "32" else jnp.float64
        self.dtype = np.float32 if precision == "32" else np.float64

        if ckpt_path.endswith('ckpt') or ckpt_path.endswith('pkl'):
            with open(ckpt_path, 'rb') as f:
                raw_params = pickle.load(f)
        else:
            raise ValueError("Unknown checkpoint format")
        
        base_leaves = flatten_leaf_arrays(init_params)
        override_leaves = flatten_leaf_arrays(raw_params)
        check_params(override_leaves, base_leaves)
        merged_leaves = base_leaves.copy()
        merged_leaves.update(override_leaves)
        self.params = convert_dict_keys(merged_leaves)

        self.device = jax.devices('cpu')[0] if device == 'cpu' else jax.devices('gpu')[0]

        self._compiled_functions = {}
        self._last_hash = None

    def _get_config_hash(self, atoms):

        config_str = f"{len(atoms)}-{''.join(atom.symbol for atom in atoms)}-{atoms.get_cell().tobytes()}"
        return hashlib.md5(config_str.encode()).hexdigest()

    def _compile_energy_and_grad_fn(self, atoms):

        z = jnp.array([atomic_numbers[atom.symbol] for atom in atoms], dtype=jnp.int32)
        natoms = jnp.array([len(z)], dtype=jnp.int32)
        batch = jnp.zeros_like(z)
        
        cell = jnp.asarray(atoms.get_cell(complete=True)[:], dtype=self.precision)
       
        index_i, index_j, shift = neighbour_list(
            quantities="ijS", 
            atoms=atoms, 
            cutoff=self.config.cutoff
        )
        edge_index =jnp.stack([jnp.array(index_j), jnp.array(index_i)])
        

        def energy_fn(pos, displacement):
            graph_data = self.process_positions_and_edges_fn(
                pos=pos,
                z=z,
                natoms=natoms,
                batch=batch,
                cell=cell,
                edge_index=edge_index,
                shift=shift,
                displacement=displacement,
            )
           
            return self.apply_fn(self.params, self.rng, graph_data)

        if self.config.compute_forces and self.config.compute_stress:
            grad_fn = jax.jit(value_and_grad(energy_fn, (0, 1)))
        elif self.config.compute_forces:
            grad_fn = jax.jit(value_and_grad(energy_fn, 0))
        else:
            grad_fn = jax.jit(energy_fn)
        
        return grad_fn, edge_index

    def calculate(self, atoms=None, properties=None, system_changes=[]):
        Calculator.calculate(self, atoms, properties, system_changes)
        properties = properties or ['energy']
        if not self.atoms.pbc.any():
            print("Non-periodic system detected. Automatically adding a large vacuum box for calculation.")
           
            # Add 20 Å of vacuum padding around the molecule
            padding = 20.0
            new_cell_dims = atoms.get_positions().ptp(axis=0) + padding
            atoms.set_cell(np.diag(new_cell_dims))
            atoms.center()
            atoms.pbc = True # Treat it as periodic now
        config_hash = self._get_config_hash(atoms)
        if config_hash != self._last_hash:
            grad_fn, edge_index = self._compile_energy_and_grad_fn(atoms)
            self._compiled_functions[config_hash] = (grad_fn, edge_index)
            self._last_hash = config_hash
        else:
            grad_fn, edge_index = self._compiled_functions[config_hash]
        
        pos = jnp.array(atoms.get_positions(), dtype=self.precision)
        displacement = jnp.zeros((1, 3, 3), dtype=pos.dtype)

        if self.config.compute_forces:
            if self.config.compute_stress:
                (energy, (minus_forces, pseudo_stress)) = grad_fn(pos, displacement)
            else:
                (energy, minus_forces) = grad_fn(pos, displacement)
                pseudo_stress = None
        else:
            energy = grad_fn(pos, displacement)
            minus_forces = None
            pseudo_stress = None

        self.results['energy'] = np.array(energy)
        self.results['free_energy'] = np.array(energy)

        if minus_forces is not None:
            self.results['forces'] = -np.array(minus_forces)

        if pseudo_stress is not None:
            stress = pseudo_stress / atoms.get_volume()
            stress_matrix = np.array(stress)[0]
            self.results['stress'] = np.array([
                stress_matrix[0, 0],
                stress_matrix[1, 1],
                stress_matrix[2, 2],
                0.5 * (stress_matrix[1, 2] + stress_matrix[2, 1]),
                0.5 * (stress_matrix[0, 2] + stress_matrix[2, 0]),
                0.5 * (stress_matrix[0, 1] + stress_matrix[1, 0])
            ])

def create_dummy_data():

    pos = jnp.array([[0.0, 0.0, 0.0], [0.75, 0.0, 0.0]], dtype=jnp.float32)
    z = jnp.array([1, 1], dtype=jnp.int32)
    natoms = jnp.array([2], dtype=jnp.int32)
    batch = jnp.array([0, 0], dtype=jnp.int32)
    displacement = jnp.zeros((1, 3, 3), dtype=pos.dtype)
    cell = jnp.array([[10.0, 0.0, 0.0],
                      [0.0, 10.0, 0.0],
                      [0.0, 0.0, 10.0]], dtype=jnp.float32)
    index_i, index_j, shift = neighbour_list(quantities = "ijS",  positions =pos, cell =cell,  pbc = np.array([True,True,True]),cutoff = 1.0)
    edge_index = jnp.stack([jnp.array(index_j), jnp.array(index_i)])
    graph_data = process_positions_and_edges(
                pos=pos,
                z=z,
                natoms=natoms,
                batch=batch,
                cell=cell,
                edge_index = edge_index,
                shift =shift,
                displacement=displacement,
            )
    return graph_data

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

def check_params(override_leaves, base_leaves):
    shape_mismatch = []
    missing = sorted(set(base_leaves) - set(override_leaves))
    extra   = sorted(set(override_leaves) - set(base_leaves))
    for k in override_leaves:
        if k in base_leaves and base_leaves[k].shape != override_leaves[k].shape:
            shape_mismatch.append((k, base_leaves[k].shape, override_leaves[k].shape))

    if missing:
        print("Missing leaf parameters")
        for p in missing[:50]:
            print("  ", p)
        raise ValueError(f"Total of {len(missing)} missing leaves.")

    if extra:
        print("Extra leaf parameters")
        for p in extra[:50]:
            print("  ", p)

    if shape_mismatch:
        for p, s0, s1 in shape_mismatch[:50]:
            print(f"{p} shape mismatch: {tuple(s0)} vs {tuple(s1)}")
        raise ValueError(f"Total of {len(shape_mismatch)} shape mismatches found.")