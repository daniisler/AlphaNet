import math
from math import pi
from typing import Optional, Tuple, List

import torch
from torch import nn, Tensor
from torch.nn import Embedding
from torch_geometric.nn.conv import MessagePassing

from alphanet.models.graph import GraphData
import numpy as np


def scatter(src: torch.Tensor, index: torch.Tensor, dim: int = -1, 
            out: Optional[torch.Tensor] = None, dim_size: Optional[int] = None, 
            reduce: str = "sum") -> torch.Tensor:
    """
    Drop-in replacement for torch_scatter.scatter using native PyTorch functions.
    """
    if out is not None:
        dim_size = out.size(dim)
    else:
        if dim_size is None:
            dim_size = int(index.max()) + 1 if index.numel() > 0 else 0

    out_size = list(src.size())
    out_size[dim] = dim_size

    if index.dim() != src.dim():
        curr_dims = index.dim()
        target_dims = src.dim()
        for _ in range(target_dims - curr_dims):
            index = index.unsqueeze(-1)
        index = index.expand_as(src)

    reduce = reduce.lower()
    
    if reduce in ['sum', 'add']:
        if out is None:
            out = torch.zeros(out_size, dtype=src.dtype, device=src.device)
        return out.scatter_add_(dim, index, src)
    
    if reduce == 'mean':
        mode = 'mean'
        init_val = 0.0
    elif reduce in ['min', 'amin']:
        mode = 'amin'
        init_val = float('inf')
    elif reduce in ['max', 'amax']:
        mode = 'amax'
        init_val = float('-inf')
    else:
        raise ValueError(f"Unknown reduce mode: {reduce}")

    if out is None:
        out = torch.full(out_size, init_val, dtype=src.dtype, device=src.device)

    out.scatter_reduce_(dim, index, src, reduce=mode, include_self=False)
    return out


class rbf_emb(nn.Module):
    r_max: float
    prefactor: float

    def __init__(self, num_basis=8, r_max=5.0, trainable=True):
        super(rbf_emb, self).__init__()
        self.trainable = trainable
        self.num_basis = num_basis
        self.r_max = r_max
        self.prefactor = 2.0 / self.r_max

        bessel_weights = (
            torch.linspace(start=1.0, end=num_basis, steps=num_basis) * math.pi
        )
        if self.trainable:
            self.bessel_weights = nn.Parameter(bessel_weights)
        else:
            self.register_buffer("bessel_weights", bessel_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        numerator = torch.sin(self.bessel_weights * x.unsqueeze(-1) / self.r_max)
        return self.prefactor * (numerator / x.unsqueeze(-1))


class NeighborEmb(MessagePassing):
    propagate_type = {'x': Tensor, 'norm': Tensor}
    
    def __init__(self, hid_dim: int):
        super(NeighborEmb, self).__init__(aggr='add')
        self.embedding = nn.Embedding(95, hid_dim)
        self.hid_dim = hid_dim
        self.ln_emb = nn.LayerNorm(hid_dim, elementwise_affine=False)

    def forward(self, z: Tensor, s: Tensor, edge_index: Tensor, embs: Tensor) -> Tensor:
        s_neighbors = self.ln_emb(self.embedding(z))
        s_neighbors = self.propagate(edge_index, x=s_neighbors, norm=embs)
        s = s + s_neighbors
        return s

    def message(self, x_j: Tensor, norm: Tensor) -> Tensor:
        return norm.view(-1, self.hid_dim) * x_j


class S_vector(MessagePassing):
    propagate_type = {'x': Tensor, 'norm': Tensor}
    
    def __init__(self, hid_dim: int):
        super(S_vector, self).__init__(aggr='add')
        self.hid_dim = hid_dim
        self.lin1 = nn.Sequential(
            nn.Linear(hid_dim, hid_dim),
            nn.LayerNorm(hid_dim, elementwise_affine=False),
            nn.SiLU())

    def forward(self, s: Tensor, v: Tensor, edge_index: Tensor, emb: Tensor) -> Tensor:
        s = self.lin1(s)
        emb = emb.unsqueeze(1) * v
        v = self.propagate(edge_index, x=s, norm=emb)
        return v.view(-1, 3, self.hid_dim)

    def message(self, x_j: Tensor, norm: Tensor) -> Tensor:
        x_j = x_j.unsqueeze(1)
        a = norm.view(-1, 3, self.hid_dim) * x_j
        return a.view(-1, 3 * self.hid_dim)


class EquiMessagePassing(MessagePassing):
    propagate_type = {
        'xh': Tensor, 'vec': Tensor, 'rbfh_ij': Tensor, 'r_ij': Tensor
    }

    def __init__(
            self,
            hidden_channels,
            num_radial,
            hidden_channels_chi=96,
            head: int = 16,
            chi1: int = 32,
            chi2: int = 8,
            has_dropout_flag: bool = False,
            has_norm_before_flag=True,
            has_norm_after_flag=False,
            complex_type=torch.complex64,
            reduce_mode='sum',
            device=torch.device('cuda') if torch.cuda.is_available() else torch.device("cpu")
    ):
        super(EquiMessagePassing, self).__init__(aggr="add", node_dim=0)

        self.device = device
        self.complex_type = complex_type
        self.reduce_mode = reduce_mode
        self.chi1 = chi1
        self.chi2 = chi2
        self.head = head
        self.hidden_channels = hidden_channels
        self.hidden_channels_chi = hidden_channels_chi
        self.scale = nn.Linear(self.hidden_channels, self.hidden_channels_chi * 2)
        self.num_radial = num_radial
        
        self.dir_proj = nn.Sequential(
            nn.Linear(3 * self.hidden_channels + self.num_radial, self.hidden_channels * 3), 
            nn.SiLU(inplace=True),
            nn.Linear(self.hidden_channels * 3, self.hidden_channels * 3)
        )

        self.x_proj = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels * 3),
        )
        self.rbf_proj = nn.Linear(num_radial, hidden_channels * 3)
        self.x_layernorm = nn.LayerNorm(hidden_channels)
        self.diagonal = nn.Sequential(
            nn.Linear(hidden_channels * 3, hidden_channels_chi // 2),
            nn.SiLU(),
            nn.Linear(hidden_channels_chi // 2, self.chi2),
        )
        self.has_dropout_flag = has_dropout_flag
        self.has_norm_before_flag = has_norm_before_flag
        self.has_norm_after_flag = has_norm_after_flag

        if self.has_norm_after_flag:
            self.dx_layer_norm = nn.LayerNorm(self.chi1)
        if self.has_norm_before_flag:
            self.dx_layer_norm = nn.LayerNorm(self.chi1 + self.hidden_channels)
            
        self.dropout = nn.Dropout(p=0.5)
        self.diachi1 = torch.nn.Parameter(torch.randn((self.chi1), device=self.device))
        self.scale2 = nn.Sequential(
            nn.Linear(self.chi1, hidden_channels // 2),
        )

        self.kernel_real = torch.nn.Parameter(torch.randn((self.head + 1, (self.hidden_channels_chi) // self.head, self.chi2)))
        self.kernel_imag = torch.nn.Parameter(torch.randn((self.head + 1, (self.hidden_channels_chi) // self.head, self.chi2)))
        
        self.fc_mps = nn.Linear(self.chi1, self.chi1)
        self.fc_dx = nn.Linear(self.chi1, hidden_channels)
        self.dia = nn.Linear(self.chi1, self.chi1)
      
        self.unitary = torch.nn.Parameter(torch.randn((self.chi1, self.chi1), device=self.device))
        self.activation = nn.SiLU()

        self.inv_sqrt_3 = 1 / math.sqrt(3.0)
        self.inv_sqrt_h = 1 / math.sqrt(hidden_channels)
        
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.x_proj[0].weight)
        self.x_proj[0].bias.data.fill_(0)
        nn.init.xavier_uniform_(self.x_proj[2].weight)
        self.x_proj[2].bias.data.fill_(0)
        nn.init.xavier_uniform_(self.rbf_proj.weight)
        self.rbf_proj.bias.data.fill_(0)
        self.x_layernorm.reset_parameters()

        nn.init.xavier_uniform_(self.dir_proj[0].weight)
        self.dir_proj[0].bias.data.fill_(0)
        nn.init.xavier_uniform_(self.dir_proj[2].weight)
        self.dir_proj[2].bias.data.fill_(0)

    def forward(
        self,
        x: Tensor,
        vec: Tensor,
        edge_index: Tensor,
        edge_rbf: Tensor,
        weight: Tensor,
        edge_vector: Tensor,
        rope: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor, Tensor]:
        if rope is not None:
            real, imag = torch.split(x, [self.hidden_channels // 2, self.hidden_channels // 2], dim=-1)
            dy_pre = torch.complex(real=real, imag=imag)
            dy_pre = dy_pre * rope
            x = torch.cat([dy_pre.real, dy_pre.imag], dim=-1)
            
        xh = self.x_proj(self.x_layernorm(x))
        rbfh = self.rbf_proj(edge_rbf)
        weight = self.dir_proj(weight)
        rbfh = rbfh * weight
        
        dx, dvec = self.propagate(
            edge_index,
            xh=xh,
            vec=vec,
            rbfh_ij=rbfh,
            r_ij=edge_vector,
            size=None,
        )
        
        if self.has_norm_before_flag:
            dx = self.dx_layer_norm(dx)

        dx, dy = torch.split(dx, [self.chi1, self.hidden_channels], dim=-1)

        if self.has_norm_after_flag:
            dx = self.dx_layer_norm(dx)

        dx = self.scale2(dx)
        dx = torch.complex(torch.cos(dx), torch.sin(dx))
        
        return dx, dy, dvec

    def message(self, xh_j, vec_j, rbfh_ij, r_ij):
        x, xh2, xh3 = torch.split(xh_j * rbfh_ij, self.hidden_channels, dim=-1)
        xh2 = xh2 * self.inv_sqrt_3
        
        real, imagine = torch.split(self.scale(x), self.hidden_channels_chi, dim=-1)
        real = real.reshape(x.shape[0], self.head, (self.hidden_channels_chi) // self.head)
        imagine = imagine.reshape(x.shape[0], self.head, (self.hidden_channels_chi) // self.head)
        if self.has_dropout_flag:
            real = self.dropout(real)
            imagine = self.dropout(imagine)

        phi = torch.complex(real, imagine)
        q = phi
        a = torch.ones(q.shape[0], 1, (self.hidden_channels_chi) // self.head, device=self.device, dtype=self.complex_type)
        kernel = (torch.complex(self.kernel_real, self.kernel_imag) / math.sqrt((self.hidden_channels) // self.head)).expand(q.shape[0], -1, -1, -1)
        
        equation = 'ijl, ijlk->ik'
        conv = torch.einsum(equation, torch.cat([a, q], dim=1), kernel.to(self.complex_type))
        a = 1.0 * self.activation(self.diagonal(rbfh_ij))
        b = a.unsqueeze(-1) * self.diachi1.unsqueeze(0).unsqueeze(0) + torch.ones(kernel.shape[0], self.chi2, self.chi1, device=self.device)
        dia = self.dia(b)
        
        equation = 'ik,ikl->il'
        kernel = torch.einsum(equation, conv, dia.to(self.complex_type))
        kernel_real, kernel_imag = kernel.real, kernel.imag
        kernel_real, kernel_imag = self.fc_mps(kernel_real), self.fc_mps(kernel_imag)
        kernel = torch.angle(torch.complex(kernel_real, kernel_imag))
        
        agg = torch.cat([kernel, x], dim=-1)
        vec = vec_j * xh2.unsqueeze(1) + xh3.unsqueeze(1) * r_ij.unsqueeze(2)
        vec = vec * self.inv_sqrt_h

        return agg, vec

    def aggregate(
            self,
            features: Tuple[torch.Tensor, torch.Tensor],
            index: torch.Tensor,
            ptr: Optional[torch.Tensor],
            dim_size: Optional[int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x, vec = features
        x = scatter(x, index, dim=self.node_dim, dim_size=dim_size, reduce=self.reduce_mode)
        vec = scatter(vec, index, dim=self.node_dim, dim_size=dim_size, reduce='sum')
        return x, vec

    def update(
            self, inputs: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return inputs


class FTE(nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.hidden_channels = hidden_channels

        self.vec_proj = nn.Linear(
            hidden_channels, hidden_channels * 2, bias=False
        )
        self.act = nn.SiLU()
        self.xvec_proj = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels * 3)
        )

        self.inv_sqrt_2 = 1 / math.sqrt(2.0)
        self.inv_sqrt_h = 1 / math.sqrt(hidden_channels)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.vec_proj.weight)
        nn.init.xavier_uniform_(self.xvec_proj[0].weight)
        self.xvec_proj[0].bias.data.fill_(0)
        nn.init.xavier_uniform_(self.xvec_proj[2].weight)
        self.xvec_proj[2].bias.data.fill_(0)

    def forward(self, x, vec):
        vec = self.vec_proj(vec)
        vec1, vec2 = torch.split(vec, self.hidden_channels, dim=-1)

        scalar = torch.sum(vec1**2, dim=-2)
        vec_dot = (vec1 * vec2).sum(dim=1)
        vec_dot = vec_dot * self.inv_sqrt_h

        x_vec_h = self.xvec_proj(torch.cat([x, scalar], dim=-1))
        xvec1, xvec2, xvec3 = torch.split(x_vec_h, self.hidden_channels, dim=-1)

        dx = xvec1 + xvec2 + vec_dot
        dx = dx * self.inv_sqrt_2

        dvec = xvec3.unsqueeze(1) * vec2
        return dx, dvec


class AlphaNet(nn.Module):
    def __init__(self, config, device=torch.device('cuda') if torch.cuda.is_available() else torch.device("cpu")):
        super(AlphaNet, self).__init__()

        self.device = device
        self.complex_type = torch.complex64 if config.dtype == "32" else torch.complex128
        self.eps = 1e-9
        self.num_layers = config.num_layers
        self.hidden_channels = config.hidden_channels
        self.a = nn.Parameter(torch.ones(108) * config.a)
        self.b = nn.Parameter(torch.ones(108) * config.b)
        self.cutoff = config.cutoff
        self.readout = config.readout
        self.chi1 = config.main_chi1
        
        self.use_sigmoid = config.use_sigmoid
        self.num_targets = config.output_dim if config.output_dim != 0 else 1
        self.compute_forces = config.compute_forces
        self.compute_stress = config.compute_stress
        
        self.z_emb_ln = nn.LayerNorm(config.hidden_channels, elementwise_affine=False)
        self.z_emb = Embedding(95, config.hidden_channels)
        self.kernel1 = torch.nn.Parameter(torch.randn((config.hidden_channels, self.chi1 * 2), device=self.device))
        self.radial_emb = rbf_emb(config.num_radial, config.cutoff)
        self.radial_lin = nn.Sequential(
            nn.Linear(config.num_radial, config.hidden_channels),
            nn.SiLU(inplace=True),
            nn.Linear(config.hidden_channels, config.hidden_channels))
        self.pi = pi 
        self.neighbor_emb = NeighborEmb(config.hidden_channels)
        self.S_vector = S_vector(config.hidden_channels)
        self.lin = nn.Sequential(
            nn.Linear(3, config.hidden_channels // 4),
            nn.SiLU(inplace=True),
            nn.Linear(config.hidden_channels // 4, 1))
        
        self.message_layers = nn.ModuleList()
        self.FTEs = nn.ModuleList()
        self.kernels_real = []
        self.kernels_imag = []
        self.zbl = config.zbl
        
        if self.zbl:
            self.register_buffer('fzbl_w', torch.tensor([0.187,0.3769,0.189,0.081,0.003,0.037,0.0546,0.0715], dtype=torch.get_default_dtype()))
            self.register_buffer('fzbl_b', torch.tensor([3.20,1.10,0.102,0.958,1.28,1.14,1.69,5], dtype=torch.get_default_dtype()))
            with torch.no_grad():
                w = getattr(self, 'fzbl_w')
                w = w.clamp(min=0.0)
                w = w / (w.sum() + 1e-12)
                self.fzbl_w.copy_(w)

            self.register_buffer('fzbl_gamma', torch.tensor(1.001, dtype=torch.get_default_dtype()))
            self.register_buffer('fzbl_alpha', torch.tensor(0.6032, dtype=torch.get_default_dtype()))
            self.register_buffer('fzbl_E2', torch.tensor(14.399645478425, dtype=torch.get_default_dtype()))  # eV·Å
            self.register_buffer('fzbl_A0', torch.tensor(0.529177210903, dtype=torch.get_default_dtype()))    # Å

        for _ in range(config.num_layers):
            self.message_layers.append(
                EquiMessagePassing(
                    hidden_channels=config.hidden_channels,
                    num_radial=config.num_radial,
                    head=config.head,
                    chi2=config.chi2,
                    chi1=config.mp_chi1,
                    has_dropout_flag=config.has_dropout_flag,
                    has_norm_before_flag=config.has_norm_before_flag,
                    has_norm_after_flag=config.has_norm_after_flag,
                    hidden_channels_chi=config.hidden_channels_chi,
                    complex_type=self.complex_type,
                    device=device,
                    reduce_mode=config.reduce_mode
                )
            )
            self.FTEs.append(FTE(config.hidden_channels))
            
            kernel_real = torch.randn((config.hidden_channels, self.chi1, self.chi1))
            kernel_imag = torch.randn((config.hidden_channels, self.chi1, self.chi1))
            self.kernels_real.append(kernel_real)
            self.kernels_imag.append(kernel_imag)
            
        self.kernels_real = torch.nn.Parameter(torch.stack(self.kernels_real))
        self.kernels_imag = torch.nn.Parameter(torch.stack(self.kernels_imag))
        self.last_layer = nn.Linear(config.hidden_channels, self.num_targets)
        self.last_layer_quantum = nn.Linear(self.chi1 * 2, self.num_targets)
        
        self.inv_sqrt_2 = 1 / math.sqrt(2.0)
        self.reset_parameters()

    def reset_parameters(self):
        self.z_emb.reset_parameters()
        for layer in self.message_layers:
            layer.reset_parameters()
        for layer in self.FTEs:
            layer.reset_parameters()
        self.last_layer.reset_parameters()
        
        for layer in self.radial_lin:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        for layer in self.lin:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def forward(
        self,
        data: GraphData,
        prefix: str,
        compute_forces: Optional[bool] = None,
        compute_stress: Optional[bool] = None,
        return_atom_energy: bool = False,
    ):
        compute_forces = self.compute_forces if compute_forces is None else compute_forces
        compute_stress = self.compute_stress if compute_stress is None else compute_stress
        pos = data.pos
        batch = data.batch
        z = data.z.long()
        edge_index = data.edge_index
        vecs = data.edge_vec
        
        dist = torch.linalg.norm(vecs, dim=1)
        z_emb = self.z_emb_ln(self.z_emb(z))
        radial_emb = self.radial_emb(dist)
        radial_hidden = self.radial_lin(radial_emb)
        rbounds = 0.5 * (torch.cos(dist * self.pi / self.cutoff) + 1.0)
        radial_hidden = rbounds.unsqueeze(-1) * radial_hidden

        s = self.neighbor_emb(z, z_emb, edge_index, radial_hidden)
        vec = torch.zeros(s.size(0), 3, s.size(1), device=s.device)
        
        j = edge_index[0]
        i = edge_index[1]
        edge_diff = vecs
        edge_diff = edge_diff / (dist.unsqueeze(1) + self.eps)
        
        edge_vec_mean = scatter(vecs, i, reduce='mean', dim=0) 
        edge_cross = torch.cross(vecs, edge_vec_mean[i])
        edge_vertical = torch.cross(edge_diff, edge_cross)
        edge_frame = torch.cat((edge_diff.unsqueeze(-1), edge_cross.unsqueeze(-1), edge_vertical.unsqueeze(-1)), dim=-1)

        S_i_j = self.S_vector(s, edge_diff.unsqueeze(-1), edge_index, radial_hidden)
        
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
        
        equation = 'ik,bi->bk'
        quantum = torch.einsum(equation, self.kernel1, z_emb)
        real, imagine = torch.split(quantum, self.chi1, dim=-1)
        quantum = torch.complex(real, imagine)

        rope = None
        for id, (message_layer, fte) in enumerate(zip(self.message_layers, self.FTEs)):
            if rope is None:
                rope, ds, dvec = message_layer(s, vec, edge_index, radial_emb, edge_weight, edge_diff, None)
            else:
                rope, ds, dvec = message_layer(s, vec, edge_index, radial_emb, edge_weight, edge_diff, rope)
            
            s = s + ds
            vec = vec + dvec
            
            kernel_real = self.kernels_real[id]
            kernel_imag = self.kernels_imag[id]
            equation = 'ikl,bi,bl->bk'
            kerneli = torch.complex(kernel_real, kernel_imag)
            quantum = torch.einsum(equation, kerneli, s.to(self.complex_type), quantum)
            quantum = quantum / (self.eps + quantum.abs().to(self.complex_type))
            
            ds, dvec = fte(s, vec)
            s = s + ds
            vec = vec + dvec

        s = self.last_layer(s) + self.last_layer_quantum(torch.cat([quantum.real, quantum.imag], dim=-1)) / self.chi1
        
        if s.dim() == 2:
            s = (self.a[z].unsqueeze(1) * s + self.b[z].unsqueeze(1))
        elif s.dim() == 1:
            s = (self.a[z] * s + self.b[z]).unsqueeze(1)
        else:
            raise ValueError(f"Unexpected shape of s: {s.shape}")

        atom_energy = s.squeeze(-1) if s.dim() == 2 and s.size(-1) == 1 else s
        total_energy = scatter(atom_energy, batch, dim=0, reduce=self.readout).squeeze()
        
        if self.use_sigmoid:
            if return_atom_energy:
                raise ValueError("Per-atom energy is not defined when sigmoid readout is enabled")
            total_energy = torch.sigmoid((total_energy - 0.5) * 5)
            
        if compute_forces and compute_stress:
            if data.displacement is not None:
              stress, forces = self.cal_stress_and_force(total_energy, pos, data.displacement, data.cell, prefix)
              stress = stress.view(-1, 3)
            else:
                stress = None
                forces = None
            if return_atom_energy:
                return total_energy, forces, stress, atom_energy
            return total_energy, forces, stress
        elif compute_forces:
            forces = self.cal_forces(total_energy, pos, prefix)
            if return_atom_energy:
                return total_energy, forces, None, atom_energy
            return total_energy, forces, None

        if return_atom_energy:
            return total_energy, None, None, atom_energy
        return total_energy, None, None
    
    def cal_forces(self, energy, positions, prefix: str = 'infer'):
        graph = (prefix == "train")
        grad_outputs = torch.jit.annotate(List[Optional[torch.Tensor]], [torch.ones_like(energy)])
        forces = torch.autograd.grad(
            outputs=[energy],
            inputs=[positions],
            grad_outputs=grad_outputs,
            create_graph=graph,
            retain_graph=graph,
            allow_unused=True
        )[0]
        assert forces is not None, "Gradient should not be None"
        return -forces
    
    def cal_stress_and_force(self, energy: Tensor, positions: Tensor, displacement: Optional[Tensor], cell: Tensor, prefix: str) -> Tuple[Tensor, Tensor]:
        if displacement is None:
             raise ValueError("displacement cannot be None for stress calculation")      
        graph = (prefix == "train")
        grad_outputs = torch.jit.annotate(List[Optional[torch.Tensor]], [torch.ones_like(energy)])
        output = torch.autograd.grad(
            [energy],
            [displacement, positions],
            grad_outputs=grad_outputs,
            create_graph=graph,
            retain_graph=graph,
            allow_unused=True
        )
        virial = output[0] if output[0] is not None else torch.zeros((3, 3), device=cell.device) 
        assert virial is not None, "Virial tensor should not be None"
        volume = torch.abs(torch.linalg.det(cell))
        volume_expanded = volume.reshape(-1, 1, 1)
        stress = virial / volume_expanded
        force = output[1]
        
        assert force is not None, "Forces tensor should not be None"
        return stress, -force
