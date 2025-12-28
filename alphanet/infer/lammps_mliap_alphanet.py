import logging
import math
from typing import Dict, Tuple, Optional
from math import pi

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch_scatter import scatter
from ase.data import chemical_symbols

# Import custom model modules
from alphanet.models.alphanet import AlphaNet
from alphanet.models.model import AlphaNetWrapper
# from alphanet.models.graph import GraphData, get_max_neighbors_mask  # Uncomment if needed

# Try to import LAMMPS interface
try:
    from lammps.mliap.mliap_unified_abc import MLIAPUnified
except ImportError:
    class MLIAPUnified:
        def __init__(self): pass
    print("Warning: LAMMPS MLIAP-Unified interface not found. Creating dummy class.")


class LAMMPS_MP(torch.autograd.Function):
    """
    Handles MPI communication for gradients between Local and Ghost atoms within LAMMPS.
    """
    @staticmethod
    def forward(ctx, *args):
        feats, data = args
        ctx.vec_len = feats.shape[-1]
        ctx.data = data
        
        out = torch.empty_like(feats)
        if not feats.is_contiguous():
            feats = feats.contiguous()
            
        # Forward exchange: Ghost atoms get data from their Local owners on other procs
        data.forward_exchange(feats, out, ctx.vec_len)
        return out

    @staticmethod
    def backward(ctx, *grad_outputs):
        (grad,) = grad_outputs
        
        gout = grad.clone()
        if not gout.is_contiguous():
            gout = gout.contiguous()
            
        # Reverse exchange: Sum gradients from Ghost atoms back to their Local owners
        ctx.data.reverse_exchange(grad, gout, ctx.vec_len)
        
        return gout, None


class AlphaNetEdgeForcesWrapper(torch.nn.Module):
    """
    Wrapper for AlphaNet to compute forces via edge gradients directly.
    """
    def __init__(self, model: AlphaNet):
        super().__init__()
        
        # Copy submodules
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

        # Copy Parameters
        self.a = model.a
        self.b = model.b
        self.kernel1 = model.kernel1
        
        if isinstance(model.kernels_real, torch.Tensor):
             self.kernels_real = nn.ParameterList([nn.Parameter(p) for p in model.kernels_real])
        else:
             self.kernels_real = model.kernels_real

        if isinstance(model.kernels_imag, torch.Tensor):
             self.kernels_imag = nn.ParameterList([nn.Parameter(p) for p in model.kernels_imag])
        else:
             self.kernels_imag = model.kernels_imag

        # Metadata
        self.cutoff = model.cutoff
        self.pi = model.pi
        self.eps = 1e-9
        self.hidden_channels = model.hidden_channels
        self.chi1 = model.chi1
        self.complex_type = model.complex_type
        self.rcutfac = float(model.cutoff)
        
        self.register_buffer("atomic_numbers", torch.arange(1, 95)) 
        self.eval() 

    def handle_lammps(self, tensor: Tensor, lammps_class: Optional[object], natoms: Tensor) -> Tensor:
        """
        Syncs tensor data for ghost atoms if running inside LAMMPS with MPI.
        """
        if lammps_class is None: 
            return tensor
            
        n_local = int(natoms[0])
        n_total = int(natoms[1])
        n_current = tensor.size(0)
        current_dim = tensor.size(1)
        
        # Pad if necessary
        if n_current == n_local and n_total > n_local:
            padding = torch.zeros((n_total - n_local, current_dim), dtype=tensor.dtype, device=tensor.device)
            tensor_full = torch.cat([tensor, padding], dim=0)
        else:
            tensor_full = tensor

        if n_total > n_local:
             if tensor_full.device != self.a.device:
                 tensor_full = tensor_full.to(self.a.device)
             
             orig_dtype = tensor_full.dtype
             if orig_dtype != torch.float64:
                 tensor_full = tensor_full.to(torch.float64)
                 
             target_dim = getattr(lammps_class, "ndescriptors", current_dim)
             
             # Pad dimension if needed
             if current_dim < target_dim:
                 pad_width = target_dim - current_dim
                 col_padding = torch.zeros((tensor_full.size(0), pad_width), dtype=tensor_full.dtype, device=tensor_full.device)
                 tensor_ready = torch.cat([tensor_full, col_padding], dim=1)
             else:
                 tensor_ready = tensor_full

             if not tensor_ready.is_contiguous():
                 tensor_ready = tensor_ready.contiguous()
             
             if tensor_ready.is_cuda: 
                 torch.cuda.synchronize()
                 
             # Sync data using custom autograd function
             tensor_synced = LAMMPS_MP.apply(tensor_ready, lammps_class)
             
             if current_dim < target_dim:
                 tensor_synced = tensor_synced[:, :current_dim]
             if tensor_synced.dtype != orig_dtype:
                 tensor_synced = tensor_synced.to(orig_dtype)
             return tensor_synced
             
        return tensor_full

    def forward(self, data: Dict[str, Tensor]) -> Tuple[Tensor, Tensor, Tensor]:
        pos = data["positions"] 
        z = data["node_attrs"]
        edge_index = data["edge_index"] 
        edge_vec = data["vectors"]
        
        # Ensure vectors require grad for force computation
        if not edge_vec.requires_grad:
            edge_vec = edge_vec.requires_grad_()
        
        lammps_ptr = data.get("lammps_ptr") 
        natoms_info = data["natoms"]
        
        n_local = int(natoms_info[0])
        n_total = pos.size(0)
            
        dist = torch.linalg.norm(edge_vec, dim=1)
        z_emb = self.z_emb_ln(self.z_emb(z))
        radial_emb = self.radial_emb(dist)
        radial_hidden = self.radial_lin(radial_emb)
        rbounds = 0.5 * (torch.cos(dist * self.pi / self.cutoff) + 1.0)
        radial_hidden = rbounds.unsqueeze(-1) * radial_hidden

        s = self.neighbor_emb(z, z_emb, edge_index, radial_hidden)
        
        vec = torch.zeros(n_local, 3, s.size(1), device=s.device, dtype=s.dtype)
        s = s[:n_local] 
        
        j = edge_index[0] 
        i = edge_index[1] 
        edge_diff = edge_vec / (dist.unsqueeze(1) + self.eps)
        
        edge_vec_mean = scatter(edge_vec, i, reduce='mean', dim=0, dim_size=n_total) 
        edge_cross = torch.cross(edge_vec, edge_vec_mean[i])
        edge_vertical = torch.cross(edge_diff, edge_cross)
        edge_frame = torch.cat((edge_diff.unsqueeze(-1), edge_cross.unsqueeze(-1), edge_vertical.unsqueeze(-1)), dim=-1)
        
        # Sync initial s features
        s_full = self.handle_lammps(s, lammps_ptr, natoms_info)
        
        S_i_j = self.S_vector(s_full, edge_diff.unsqueeze(-1), edge_index, radial_hidden)[:n_local]
        sij_flat = S_i_j.reshape(n_local, -1)
        sij_full_flat = self.handle_lammps(sij_flat, lammps_ptr, natoms_info)
        S_i_j = sij_full_flat.reshape(n_total, 3, -1)
        
        scalrization1 = torch.sum(S_i_j[i].unsqueeze(2) * edge_frame.unsqueeze(-1), dim=1)
        scalrization2 = torch.sum(S_i_j[j].unsqueeze(2) * edge_frame.unsqueeze(-1), dim=1)
        scalrization1[:, 1, :] = torch.square(scalrization1[:, 1, :].clone())
        scalrization2[:, 1, :] = torch.square(scalrization2[:, 1, :].clone())
        
        scalar3 = (self.lin(torch.permute(scalrization1, (0, 2, 1))) + 
                  torch.permute(scalrization1, (0, 2, 1))[:, :, 0].unsqueeze(2)).squeeze(-1) / math.sqrt(self.hidden_channels)
        scalar4 = (self.lin(torch.permute(scalrization2, (0, 2, 1))) + 
                  torch.permute(scalrization2, (0, 2, 1))[:, :, 0].unsqueeze(2)).squeeze(-1) / math.sqrt(self.hidden_channels)
        
        edge_weight = torch.cat((scalar3, scalar4), dim=-1) * rbounds.unsqueeze(-1)
        edge_weight = torch.cat((edge_weight, radial_hidden, radial_emb), dim=-1)
        
        quantum = torch.einsum('ik,bi->bk', self.kernel1, z_emb[:n_local])
        real, imagine = torch.split(quantum, self.chi1, dim=-1)
        quantum = torch.complex(real, imagine)

        rope: Optional[Tensor] = None
        
        # Message Passing Layers
        for i_layer, (message_layer, fte, kernel_real, kernel_imag) in enumerate(zip(
            self.message_layers, self.FTEs, self.kernels_real, self.kernels_imag
        )):
            # Sync s
            s_full = self.handle_lammps(s, lammps_ptr, natoms_info)

            # Sync vec
            vec_flat = vec.reshape(n_local, -1)
            vec_full_flat = self.handle_lammps(vec_flat, lammps_ptr, natoms_info)
            vec_full = vec_full_flat.reshape(n_total, 3, -1)
            
            # Sync rope
            if rope is not None:
                if rope.is_complex():
                    rope_real = torch.view_as_real(rope).flatten(1) 
                    rope_full_real = self.handle_lammps(rope_real, lammps_ptr, natoms_info)
                    rope_full = torch.view_as_complex(rope_full_real.view(n_total, -1, 2))
                else:
                    rope_full = self.handle_lammps(rope, lammps_ptr, natoms_info)
            else:
                rope_full = None
            
            new_rope_full, ds_full, dvec_full = message_layer(s_full, vec_full, edge_index, radial_emb, edge_weight, edge_diff, rope_full)
            
            rope = new_rope_full[:n_local]
            ds = ds_full[:n_local]
            dvec = dvec_full[:n_local]
            s = s + ds 
            vec = vec + dvec
           
            kerneli = torch.complex(kernel_real, kernel_imag) 
            quantum = torch.einsum('ikl,bi,bl->bk', kerneli, s.to(self.complex_type), quantum)
            quantum = quantum / (self.eps + quantum.abs().to(self.complex_type))
            ds_fte, dvec_fte = fte(s, vec)
            s = s + ds_fte
            vec = vec + dvec_fte
           
        # Final Readout
        s_per_atom = self.last_layer(s) + self.last_layer_quantum(torch.cat([quantum.real, quantum.imag], dim=-1)) / self.chi1
        node_energy = (self.a[z[:n_local]].unsqueeze(1) * s_per_atom + self.b[z[:n_local]].unsqueeze(1)).squeeze(-1)
        
        if node_energy.shape[0] > n_local:
             s_total = node_energy[:n_local].sum()
        else:
             s_total = node_energy.sum()

        # Compute Gradients (Force = -dE/dr, computed here as gradients w.r.t edge vectors)
        if s_total.grad_fn is not None:
            grads = torch.autograd.grad(
                outputs=[s_total],
                inputs=[edge_vec], 
                grad_outputs=[torch.ones_like(s_total)],
                retain_graph=False,
                create_graph=False, 
                allow_unused=True,
            )[0]
            if grads is None:
                grads = torch.zeros_like(edge_vec)
        else:
            grads = torch.zeros_like(edge_vec)
        
        pair_forces = grads 
        
        return s_total, node_energy, pair_forces


class LAMMPS_MLIAP_ALPHANET(MLIAPUnified):
    def __init__(self, model_wrapper: AlphaNetWrapper, **kwargs):
        super().__init__()
        
        internal_model = model_wrapper.model
        internal_model.double()
        internal_model.eval() 
        edge_wrapper = AlphaNetEdgeForcesWrapper(internal_model).eval()
        edge_wrapper.double()
        
        self.model = edge_wrapper 
        self.element_types = [chemical_symbols[i] for i in range(1, 95)]
        self.num_species = 94
        self.rcutfac = 0.5 * float(model_wrapper.model.cutoff) 
        self.hidden_dim = internal_model.hidden_channels
        
        target_dim = 3 * self.hidden_dim
        self.ndescriptors = target_dim
        self.nparams = target_dim
        self.dtype = model_wrapper.precision
        self.device = "cpu" 
        self.initialized = False
        self.step = 0

    def _initialize_device(self, data):
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.model = self.model.to(self.device)
        self.model.eval()
        self.initialized = True
    
    def compute_forces(self, data):
        natoms = data.nlocal
        ntotal = data.ntotal
        nghosts = ntotal - natoms
        npairs = data.npairs
        
        if not self.initialized:
            self.model.to(self.dtype)
            self._initialize_device(data)
        
        if hasattr(data.elems, 'get'): 
            elems_np = data.elems.get()
        else: 
            elems_np = data.elems
            
        species = torch.as_tensor(elems_np, dtype=torch.int64, device=self.device)
        self.step += 1
        
        if natoms == 0 or npairs <= 1: 
            return

        # 1. Prepare data batch
        batch = self._prepare_batch(data, natoms, nghosts, species)
        
        # 2. Forward pass
        batch["vectors"].requires_grad_(True)
        total_energy, atom_energies, pair_forces = self.model(batch)
        
        if self.device.type == "cuda": 
            torch.cuda.synchronize()

        # 3. Update LAMMPS with energies and forces
        self._update_lammps_data(data, atom_energies, pair_forces, natoms, total_energy)

    def _update_lammps_data(self, data, atom_energies, pair_forces, natoms, total_energy):
        if self.dtype == torch.float32:
            pair_forces = pair_forces.double()
            atom_energies = atom_energies.double()
            total_energy = total_energy.double()

        atom_energies_cpu = atom_energies[:natoms].detach().cpu().numpy()
        
        # Update atom energies
        if hasattr(data.eatoms, 'get'): 
             try: data.eatoms[:natoms] = atom_energies_cpu
             except: pass
        else:
             try:
                 eatoms_tensor = torch.as_tensor(data.eatoms)
                 eatoms_tensor[:natoms].copy_(torch.from_numpy(atom_energies_cpu))
             except:
                 pass 

        data.energy = total_energy.item()
        
        final_forces = pair_forces.detach()
       
        if not final_forces.is_contiguous():
            final_forces = final_forces.contiguous()

        # Update pair forces (GPU or CPU)
        if self.device.type == 'cuda':
            try:
                from torch.utils.dlpack import to_dlpack
                from cupy import from_dlpack
                force_cupy = from_dlpack(to_dlpack(final_forces))
                data.update_pair_forces_gpu(force_cupy)
            except ImportError:
                data.update_pair_forces_gpu(final_forces.cpu().numpy())
        else:
            final_forces_np = final_forces.numpy()
            data.update_pair_forces_gpu(final_forces_np)

    def _prepare_batch(self, data, natoms, nghosts, species) -> Dict[str, object]:
        positions = torch.zeros((natoms + nghosts, 3), dtype=self.dtype, device=self.device)
        node_attrs = species + 1
        batch_tensor = torch.zeros(natoms, dtype=torch.int64, device=self.device)
        natoms_tensor = torch.tensor([natoms, natoms+nghosts], dtype=torch.int64, device=self.device)
        
        if hasattr(data.rij, 'get'): rij_data = data.rij.get()
        else: rij_data = data.rij
        
        if hasattr(data.pair_i, 'get'): pair_i = data.pair_i.get()
        else: pair_i = data.pair_i
        
        if hasattr(data.pair_j, 'get'): pair_j = data.pair_j.get()
        else: pair_j = data.pair_j

        rij_tensor = torch.as_tensor(rij_data, dtype=self.dtype, device=self.device)
        target_tensor = torch.as_tensor(pair_i, dtype=torch.int64, device=self.device)
        source_tensor = torch.as_tensor(pair_j, dtype=torch.int64, device=self.device)
        
        edge_index = torch.stack([source_tensor, target_tensor], dim=0)
        
        return {
            "positions": positions,
            "vectors": rij_tensor,
            "node_attrs": node_attrs,
            "edge_index": edge_index,
            "batch": batch_tensor,
            "natoms": natoms_tensor, 
            "lammps_ptr": data,
        }

    def compute_descriptors(self, data: Dict[str, Tensor]) -> None: 
        pass

    def compute_gradients(self, data: Dict[str, Tensor]) -> None: 
        pass