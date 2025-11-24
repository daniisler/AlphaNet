import logging
import time
import math
from typing import Dict, Tuple, Optional, List
from math import pi

import torch
import torch.nn.functional
from torch import nn, Tensor
from torch_scatter import scatter, scatter_add
from ase.data import chemical_symbols

# Import AlphaNet core components
from alphanet.models.alphanet import AlphaNet
from alphanet.models.model import AlphaNetWrapper
from alphanet.models.graph import GraphData

# Try to import the LAMMPS ML-IAP base class
try:
    from lammps.mliap.mliap_unified_abc import MLIAPUnified
except ImportError:
    class MLIAPUnified:
        """Dummy class if lammps Python package is not installed."""
        def __init__(self):
            pass
    print("Warning: LAMMPS MLIAP-Unified interface not found. Creating dummy class.")

class AlphaNetEdgeForcesWrapper(torch.nn.Module):
    """
    JIT-compatible wrapper that distributes ZBL energy to atoms.
    """
    def __init__(self, model: AlphaNet):
        super().__init__()
        
        # --- Copy Submodules ---
        self.z_emb = model.z_emb
        self.z_emb_ln = model.z_emb_ln
        self.radial_emb = model.radial_emb
        self.radial_lin = model.radial_lin
        self.neighbor_emb = model.neighbor_emb
        self.S_vector = model.S_vector
        self.lin = model.lin
        self.message_layers = model.message_layers
        self.FTEs = model.FTEs
        self.last_layer = model.last_layer
        self.last_layer_quantum = model.last_layer_quantum

        # --- Copy Parameters & Buffers ---
        self.a = model.a
        self.b = model.b
        self.kernel1 = model.kernel1

        self.kernels_real = nn.ParameterList([p for p in model.kernels_real])
        self.kernels_imag = nn.ParameterList([p for p in model.kernels_imag])

        # --- Copy Metadata ---
        self.cutoff = model.cutoff
        self.pi = model.pi
        self.eps = model.eps
        self.hidden_channels = model.hidden_channels
        self.chi1 = model.chi1
        self.complex_type = model.complex_type
        self.readout = model.readout

        # LAMMPS Metadata
        self.ndescriptors = 1
        self.nparams = 1
        self.rcutfac = float(model.cutoff) / 2.0 
        self.num_species = 94
        self.element_types = [chemical_symbols[i] for i in range(1, 95)]
        
        self.register_buffer("atomic_numbers", torch.arange(1, 95)) 
        self.register_buffer("r_max", torch.tensor(self.cutoff))

        # --- ZBL Buffers ---
        self.zbl = True #model.zbl 
        
        if self.zbl:
            self.register_buffer('fzbl_w', model.fzbl_w)
            self.register_buffer('fzbl_b', model.fzbl_b)
            self.register_buffer('fzbl_gamma', model.fzbl_gamma)
            self.register_buffer('fzbl_alpha', model.fzbl_alpha)
            self.register_buffer('fzbl_E2', model.fzbl_E2)
            self.register_buffer('fzbl_A0', model.fzbl_A0)
        else:
            # Dummy buffers for JIT compatibility
            dummy = torch.empty(0, dtype=torch.get_default_dtype())
            self.register_buffer('fzbl_w', dummy)
            self.register_buffer('fzbl_b', dummy)
            self.register_buffer('fzbl_gamma', dummy)
            self.register_buffer('fzbl_alpha', dummy)
            self.register_buffer('fzbl_E2', dummy)
            self.register_buffer('fzbl_A0', dummy)

        self.eval() 

    def forward(self, data: Dict[str, Tensor]) -> Tuple[Tensor, Tensor, Tensor]:
        # 1. Unpack data
        pos = data["positions"]
        z = data["node_attrs"]
        batch = data["batch"]
        # natoms = data["natoms"] # 未使用
        # cell = data["cell"]     # 未使用
        edge_index = data["edge_index"]
        
        # Enable gradients for edge_vec to compute forces
        edge_vec = data["vectors"].requires_grad_() 
        dist = torch.linalg.norm(edge_vec, dim=1)
        
        # 2. AlphaNet Core Logic
        z_emb = self.z_emb_ln(self.z_emb(z))
        radial_emb = self.radial_emb(dist)
        radial_hidden = self.radial_lin(radial_emb)
        
        rbounds = 0.5 * (torch.cos(dist * self.pi / self.cutoff) + 1.0)
        radial_hidden = rbounds.unsqueeze(-1) * radial_hidden

        s = self.neighbor_emb(z, z_emb, edge_index, radial_hidden)
        vec = torch.zeros(s.size(0), 3, s.size(1), device=s.device, dtype=s.dtype)
        
        j = edge_index[0]
        i = edge_index[1]
        edge_diff = edge_vec / (dist.unsqueeze(1) + self.eps)
        
        # Geometric calculation
        edge_vec_mean = scatter(edge_vec, i, reduce='mean', dim=0) 
        edge_cross = torch.cross(edge_vec, edge_vec_mean[i])
        
        edge_vertical = torch.cross(edge_diff, edge_cross)
        edge_frame = torch.cat((edge_diff.unsqueeze(-1), edge_cross.unsqueeze(-1), edge_vertical.unsqueeze(-1)), dim=-1)
        S_i_j = self.S_vector(s, edge_diff.unsqueeze(-1), edge_index, radial_hidden)
        
        scalrization1 = torch.sum(S_i_j[i].unsqueeze(2) * edge_frame.unsqueeze(-1), dim=1)
        scalrization2 = torch.sum(S_i_j[j].unsqueeze(2) * edge_frame.unsqueeze(-1), dim=1)
        scalrization1[:, 1, :] = torch.abs(scalrization1[:, 1, :].clone())
        scalrization2[:, 1, :] = torch.abs(scalrization2[:, 1, :].clone())
        
        scalar3 = (self.lin(torch.permute(scalrization1, (0, 2, 1))) + 
                  torch.permute(scalrization1, (0, 2, 1))[:, :, 0].unsqueeze(2)).squeeze(-1) / math.sqrt(self.hidden_channels)
        scalar4 = (self.lin(torch.permute(scalrization2, (0, 2, 1))) + 
                  torch.permute(scalrization2, (0, 2, 1))[:, :, 0].unsqueeze(2)).squeeze(-1) / math.sqrt(self.hidden_channels)
        
        edge_weight = torch.cat((scalar3, scalar4), dim=-1) * rbounds.unsqueeze(-1)
        edge_weight = torch.cat((edge_weight, radial_hidden, radial_emb), dim=-1)
        
        quantum = torch.einsum('ik,bi->bk', self.kernel1, z_emb)
        real, imagine = torch.split(quantum, self.chi1, dim=-1)
        quantum = torch.complex(real, imagine)

        rope: Optional[Tensor] = None
        
        for message_layer, fte, kernel_real, kernel_imag in zip(
            self.message_layers, self.FTEs, self.kernels_real, self.kernels_imag
        ):
            if rope is None:
                rope, ds, dvec = message_layer(s, vec, edge_index, radial_emb, edge_weight, edge_diff, None)
            else:
                rope, ds, dvec = message_layer(s, vec, edge_index, radial_emb, edge_weight, edge_diff, rope)
            
            s = s + ds
            vec = vec + dvec
            
            kerneli = torch.complex(kernel_real, kernel_imag) 
            quantum = torch.einsum('ikl,bi,bl->bk', kerneli, s.to(self.complex_type), quantum)
            quantum = quantum / quantum.abs().to(self.complex_type)
            
            ds, dvec = fte(s, vec)
            s = s + ds
            vec = vec + dvec

        s_per_atom = self.last_layer(s) + self.last_layer_quantum(torch.cat([quantum.real, quantum.imag], dim=-1)) / self.chi1
        
        # AlphaNet ML Energy (per atom)
        # Squeeze here to make it 1D [Natoms]
        node_energy = (self.a[z].unsqueeze(1) * s_per_atom + self.b[z].unsqueeze(1)).squeeze(-1)
        
        # --- ZBL Potential Calculation ---
        if self.zbl:
            r_e = dist
            Z_j = z[j]
            Z_i = z[i]
            
            w = self.fzbl_w
            b_params = self.fzbl_b
            # Ensure types match
            if s.dtype == torch.float64:
                w = w.double()
                b_params = b_params.double()
            
            # ... (Simplified loading for brevity, assuming buffers are correct type/device)
            gamma = self.fzbl_gamma
            alpha = self.fzbl_alpha
            E2 = self.fzbl_E2
            A0 = self.fzbl_A0
            
            denom = torch.pow(Z_j, alpha) + torch.pow(Z_i, alpha)
            denom = torch.clamp(denom, min=1e-12)
            a_vals = gamma * 0.8854 * A0 / denom
            x = r_e / a_vals
            
            # Broadcasting for ZBL calc
            exp_terms = torch.exp(-x.unsqueeze(1) * b_params.unsqueeze(0))
            phi_vals = exp_terms.matmul(w)
            
            V_edge = (Z_j * Z_i * E2) * (phi_vals / r_e)
            
            # Cutoff smoothing
            r_cut = 1.0 # Hardcoded smoothing cutoff start?
            xrc = (r_e / r_cut).clamp(min=0.0, max=1.0)
            c = 0.5 * (torch.cos(torch.pi * xrc) + 1.0)
            c = torch.where(r_e >= r_cut, torch.zeros_like(c), c)
            V_edge = V_edge * c
            
            # --- [KEY CHANGE START] Distribute ZBL to Atoms ---
            # Accumulate half of the edge energy to the target node (atom i).
            # Since neighbor lists typically contain both i->j and j->i,
            # adding 0.5 * V_edge for every incoming edge sums to the total correct energy per atom.
            # dim_size is important to handle cases where an atom might be isolated (no edges)
            zbl_per_atom = scatter_add(V_edge, i, dim=0, dim_size=node_energy.size(0)) * 0.5
            
            # Add ZBL contribution to the ML atomic energy
            node_energy = node_energy + zbl_per_atom
            # --- [KEY CHANGE END] ---
        
        # --- Aggregation ---
        # Re-calculate total energy based on the UPDATED node_energy
        s_total = scatter(node_energy, batch, dim=0, reduce=self.readout).sum()
        
        # 3. Calculate Pair Forces
        grad_outputs: List[Optional[Tensor]] = [torch.ones_like(s_total)]
        edge_forces = torch.autograd.grad(
            outputs=[s_total],
            inputs=[edge_vec], 
            grad_outputs=grad_outputs,
            retain_graph=False,
            create_graph=False,
            allow_unused=True,
        )[0]
        
        if edge_forces is None:
            edge_forces = torch.zeros_like(edge_vec)
        
        pair_forces = -1 * edge_forces 
        
        # Return updated node_energy (ML + ZBL)
        return s_total, node_energy, pair_forces

    @torch.jit.export
    def compute_descriptors(self, data: Dict[str, Tensor]) -> None:
        pass

    @torch.jit.export
    def compute_gradients(self, data: Dict[str, Tensor]) -> None:
        pass

class LAMMPS_MLIAP_ALPHANET(MLIAPUnified):
    """
    Python-level interface for AlphaNet to LAMMPS via the ML-IAP interface.
    This class is used when running LAMMPS *as a library* from Python.
    """
    def __init__(self, model_wrapper: AlphaNetWrapper, **kwargs):
        super().__init__()
        
        # We must script the *instance* here for the Python interface
        internal_model = model_wrapper.model
        # Ensure model is in eval mode before scripting
        internal_model.eval() 
        edge_wrapper = AlphaNetEdgeForcesWrapper(internal_model).eval()
        self.model = torch.jit.script(edge_wrapper)
        
        # Metadata required by LAMMPS ML-IAP
        self.element_types = [chemical_symbols[i] for i in range(1, 95)]
        self.num_species = 94
        self.rcutfac = 0.5 * float(model_wrapper.model.cutoff) # rcut = rcutfac * 2
        self.ndescriptors = 1 # dummy value
        self.nparams = 1 # dummy value
        
        self.dtype = model_wrapper.precision
        self.device = "cpu" # will be set in _initialize_device
        self.initialized = False
        self.step = 0

    def _initialize_device(self, data):
        """Infer device from LAMMPS data object."""
        using_kokkos = "kokkos" in data.__class__.__module__.lower()

        if using_kokkos:
            device = torch.as_tensor(data.elems).device
            if device.type == "cpu":
                logging.warning("LAMMPS is running on CPU, so AlphaNet will also run on CPU.")
        else:
            device = torch.device("cpu")

        self.device = device
        self.model = self.model.to(device)
        self.model.eval() # Ensure model is in evaluation mode
        logging.info(f"AlphaNet ML-IAP model initialized on device: {device}")
        self.initialized = True

    def compute_forces(self, data):
        """
        Main function called by LAMMPS to compute energy and forces.
        """
        natoms = data.nlocal
        ntotal = data.ntotal
        nghosts = ntotal - natoms
        npairs = data.npairs
        species = torch.as_tensor(data.elems, dtype=torch.int64)

        if not self.initialized:
            self.model.to(self.dtype) # Ensure model dtype is correct
            self._initialize_device(data)

        self.step += 1
        
        if natoms == 0 or npairs <= 1:
            return

        with torch.no_grad():
            batch = self._prepare_batch(data, natoms, nghosts, species)
            
            with torch.enable_grad():
                total_energy, atom_energies, pair_forces = self.model(batch)

            if self.device.type != "cpu" and self.device.type != "mps":
                torch.cuda.synchronize()

            self._update_lammps_data(data, atom_energies, pair_forces, natoms, total_energy)

    def _prepare_batch(self, data, natoms, nghosts, species) -> Dict[str, Tensor]:
        """
        Convert LAMMPS data object to the dictionary format
        required by AlphaNetEdgeForcesWrapper.
        """
        positions = torch.as_tensor(data.x).to(self.dtype).to(self.device)[:natoms]
        node_attrs = species.to(torch.int64).to(self.device) 
        batch_tensor = torch.zeros(natoms, dtype=torch.int64, device=self.device)
        natoms_tensor = torch.tensor([natoms], dtype=torch.int64, device=self.device)
        cell_tensor = torch.as_tensor(data.cell).to(self.dtype).to(self.device)

        return {
            "positions": positions,
            "vectors": torch.as_tensor(data.rij).to(self.dtype).to(self.device),
            "node_attrs": node_attrs,
            "edge_index": torch.stack([
                torch.as_tensor(data.pair_j, dtype=torch.int64).to(self.device),
                torch.as_tensor(data.pair_i, dtype=torch.int64).to(self.device),
            ], dim=0),
            "batch": batch_tensor,
            "natoms": natoms_tensor,
            "cell": cell_tensor,
        }

    def _update_lammps_data(self, data, atom_energies, pair_forces, natoms, total_energy):
        """
        Write computed energies and pair forces back to the LAMMPS data object.
        """
        if self.dtype == torch.float32:
            pair_forces = pair_forces.double()
            atom_energies = atom_energies.double()
            total_energy = total_energy.double()

        eatoms = torch.as_tensor(data.eatoms)
        eatoms.copy_(atom_energies[:natoms])
        data.energy = total_energy
        
        data.update_pair_forces_gpu(pair_forces) 

    @torch.jit.export
    def compute_descriptors(self, data: Dict[str, Tensor]) -> None:
        """
        Dummy method required by LAMMPS ML-IAP interface.
        Does nothing but must exist.
        """
        pass

    @torch.jit.export
    def compute_gradients(self, data: Dict[str, Tensor]) -> None:
        """
        Dummy method required by LAMMPS ML-IAP interface.
        Does nothing but must exist.
        """
        pass