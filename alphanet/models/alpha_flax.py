# -*- coding: utf-8 -*-
"""
Created on Mon Jul 28 15:54:29 2025

@author: Bangchen Yin
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax import linen as nn
from typing import Optional, Tuple, List, NamedTuple, Any
import math
import numpy as np

class Config:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
def segment_mean(data, segment_ids, num_segments):

    segment_ids = segment_ids.astype(jnp.int32)
    ones = jnp.ones((data.shape[0], *((1,) * (data.ndim - 1))), dtype=data.dtype)
    segment_count = jax.ops.segment_sum(ones, segment_ids, num_segments)

    segment_sum = jax.ops.segment_sum(data, segment_ids, num_segments)

    segment_count = jnp.where(segment_count == 0, 1, segment_count)
    
    return segment_sum / segment_count

class rbf_emb(nn.Module):
    num_basis: int = 8
    r_max: float = 5.0
    trainable: bool = True
    
    @nn.compact
    def __call__(self, x):
        prefactor = 2.0 / self.r_max
        init_values = jnp.pi * jnp.arange(1, self.num_basis + 1)
        
        if self.trainable:

            bessel_weights = self.param('bessel_weights', 
                                       nn.initializers.constant(init_values), 
                                       (self.num_basis,))
        else:

            bessel_weights = init_values

        x_expanded = x[..., jnp.newaxis]

        numerator = jnp.sin(bessel_weights * x_expanded / self.r_max)

        result = prefactor * (numerator / x_expanded)
        
        return result

class NeighborEmb(nn.Module):
    hid_dim: int
    
    @nn.compact
    def __call__(self, z, s, edge_index, embs):
        embedding = nn.Embed(95, self.hid_dim)
        s_neighbors = embedding(z)

        s_neighbors = nn.LayerNorm(use_bias=False, use_scale=False, epsilon=1e-6)(s_neighbors)
        
        source, target = edge_index 
        messages = embs * s_neighbors[source]
        s_neighbors = jax.ops.segment_sum(messages, target, s.shape[0])
        s = s + s_neighbors
        return s

class S_vector(nn.Module):
    hid_dim: int

    @nn.compact
    def __call__(
        self,
        s: jnp.ndarray,         # (N, H)
        v: jnp.ndarray,         # (N, 3, H)
        edge_index: jnp.ndarray,# (2, E)  dtype = int
        emb: jnp.ndarray        # (N, H)
    ) -> jnp.ndarray:          # returns (N, 3, H)
        """
        Flax implementation that mirrors the PyG MessagePassing(add) behavior.
        Assumes edge_index[0] = source, edge_index[1] = target.
        """
        
        s = nn.Dense(self.hid_dim)(s)   # (N,H)
        s = nn.LayerNorm(use_bias=False, use_scale=False)(s)
        s = nn.silu(s)

        source = edge_index[0]  # (E,)
        target = edge_index[1]  # (E,)

        target = target.astype(jnp.int32)
        source = source.astype(jnp.int32)
        node_transform = emb[:, None, :] * v
        
        source_transform = node_transform
       
        source_s = s[source][:, None, :]

    
        messages = source_transform * source_s
        E = messages.shape[0]
        messages_flat = messages.reshape(E, 3 * self.hid_dim)
        
        N = s.shape[0]
        agg_shape = (N, 3 * self.hid_dim)
        aggregated = jnp.zeros(agg_shape, dtype=messages_flat.dtype).at[target].add(messages_flat)

        out = aggregated.reshape(N, 3, self.hid_dim)
        return out


class EquiMessagePassing(nn.Module):
    hidden_channels: int
    num_radial: int
    head: int = 16
    chi1: int = 32
    chi2: int = 8
    hidden_channels_chi: int = 96
    complex_type: type = jnp.complex64
    has_dropout_flag: bool = False
    has_norm_before_flag: bool = True
    has_norm_after_flag: bool = False
    reduce_mode: str = 'sum'
    
    def setup(self):
      
        self.scale = nn.Dense(self.hidden_channels_chi * 2)
        self.dir_proj = nn.Sequential([
            nn.Dense(self.hidden_channels * 3),
            nn.silu,
            nn.Dense(self.hidden_channels * 3)
        ])
        
        self.x_proj = nn.Sequential([
            nn.Dense(self.hidden_channels),
            nn.silu,
            nn.Dense(self.hidden_channels * 3)
        ])
        
        self.rbf_proj = nn.Dense(self.hidden_channels * 3)
        self.x_layernorm = nn.LayerNorm()
        
        self.diagonal = nn.Sequential([
            nn.Dense(self.hidden_channels_chi // 2),
            nn.silu,
            nn.Dense(self.chi2)
        ])
        
        if self.has_norm_after_flag or self.has_norm_before_flag:
            self.dx_layer_norm = nn.LayerNorm()
            
        self.scale2 = nn.Dense(self.hidden_channels // 2)

        self.kernel_real = self.param(
            'kernel_real', 
            nn.initializers.lecun_normal(), 
            (self.head + 1, self.hidden_channels_chi // self.head, self.chi2)
        )
        
        self.kernel_imag = self.param(
            'kernel_imag', 
            nn.initializers.lecun_normal(), 
            (self.head + 1, self.hidden_channels_chi // self.head, self.chi2)
        )
        
        self.diachi1 = self.param(
            'diachi1', 
            nn.initializers.normal(), 
            (self.chi1,)
        )
        
       
        self.activation = jax.nn.silu
        self.fc_mps = nn.Dense(self.chi1)
        
        self.dia = nn.Dense(self.chi1)
        
        self.inv_sqrt_3 = 1 / math.sqrt(3.0)
        self.inv_sqrt_h = 1 / math.sqrt(self.hidden_channels)
    
    def __call__(self, x, vec, edge_index, edge_rbf, weight, edge_vector, rope=None):
        if rope is not None:
            real, imag = jnp.split(x, 2, axis=-1)
            dy_pre = real + 1j * imag
            dy_pre = dy_pre * rope
            x = jnp.concatenate([jnp.real(dy_pre), jnp.imag(dy_pre)], axis=-1)
            
        x = self.x_layernorm(x)
        xh = self.x_proj(x)
        
        rbfh = self.rbf_proj(edge_rbf)
        weight_proj = self.dir_proj(weight)
        rbfh = rbfh * weight_proj
        
        #row, col = edge_index
        col, row = edge_index
        messages_x, messages_vec = self.message(
            xh[col], vec[col], rbfh, edge_vector
        )

        dx = jax.ops.segment_sum(messages_x, row, x.shape[0])
        dvec = jax.ops.segment_sum(messages_vec, row, vec.shape[0])
        
        if self.has_norm_before_flag:
            dx = self.dx_layer_norm(dx)
            
        dx, dy = dx[..., :self.chi1], dx[..., self.chi1:]
        
        if self.has_norm_after_flag:
            dx = self.dx_layer_norm(dx)
            
        dx = self.scale2(dx)
        dx = jnp.cos(dx) + 1j * jnp.sin(dx)
        
        return dx, dy, dvec
    
    def message(self, xh_j, vec_j, rbfh_ij, r_ij):

        x, xh2, xh3 = jnp.split(xh_j * rbfh_ij, 3, axis=-1)
        xh2 = xh2 * self.inv_sqrt_3

        scale_out = self.scale(x)
        real, imag = jnp.split(scale_out, 2, axis=-1)
        
        real = real.reshape(x.shape[0], self.head, -1)

        imag = imag.reshape(x.shape[0], self.head, -1)

        phi = real + 1j * imag
        q = phi

        a = jnp.ones((q.shape[0], 1, self.hidden_channels_chi // self.head), dtype=self.complex_type)

        scale_factor = 1 / math.sqrt(self.hidden_channels_chi // self.head)
        kernel_real_part = self.kernel_real * scale_factor
        kernel_imag_part = self.kernel_imag * scale_factor
        kernel = kernel_real_part + 1j * kernel_imag_part
        kernel = jnp.broadcast_to(kernel, (q.shape[0],) + kernel.shape)

        q_expanded = jnp.concatenate([a, q], axis=1)
        conv = jnp.einsum('ijl,ijlk->ik', q_expanded, kernel)

        a_diag = self.activation(self.diagonal(rbfh_ij))
        b = a_diag[..., jnp.newaxis] * self.diachi1 + 1.0
        dia = self.dia(b)
        dia_complex = dia + 0j

        kernel = jnp.einsum('ik,ikl->il', conv, dia_complex)

        kernel_real = self.fc_mps(jnp.real(kernel))
        kernel_imag = self.fc_mps(jnp.imag(kernel))
        kernel_angle = jnp.angle(kernel_real + 1j * kernel_imag)

        agg = jnp.concatenate([kernel_angle, x], axis=-1)

        vec_part1 = vec_j * xh2[:, jnp.newaxis, :]

        r_ij_expanded = r_ij[:, :, jnp.newaxis]  # (num_edges, 3, 1)
        xh3_expanded = xh3[:, jnp.newaxis, :]  # (num_edges, 1, hidden_channels)
        vec_part2 = xh3_expanded * r_ij_expanded

        vec = vec_part1 + vec_part2

        vec = vec * self.inv_sqrt_h
        
        return agg, vec


class FTE(nn.Module):
    hidden_channels: int

    def setup(self):
        self.vec_proj = nn.Dense(self.hidden_channels * 2, use_bias=False)
        self.xvec_proj = nn.Sequential([
            nn.Dense(self.hidden_channels),
            nn.silu,
            nn.Dense(self.hidden_channels * 3)
        ])
        self.inv_sqrt_2 = 1 / math.sqrt(2.0)
        self.inv_sqrt_h = 1 / math.sqrt(self.hidden_channels)

    def __call__(self, x: jnp.ndarray, vec: jnp.ndarray):

        vec_proj = self.vec_proj(vec)
        vec1, vec2 = jnp.split(vec_proj, 2, axis=-1)

        scalar = jnp.linalg.norm(vec1, axis=1, ord=1)
        vec_dot = jnp.sum(vec1 * vec2, axis=1) * self.inv_sqrt_h

        x_vec_h = self.xvec_proj(jnp.concatenate([x, scalar], axis=-1))
      
        xvec1, xvec2, xvec3 = jnp.split(x_vec_h, 3, axis=-1)

        dx = (xvec1 + xvec2 + vec_dot) * self.inv_sqrt_2

        dvec = xvec3[:, None] * vec2

        return dx, dvec

class AlphaNet_flax(nn.Module):
    config: Any
    
    def setup(self):
 
        if self.config.dtype == "64": # <--- 修正
            self.dtype = jnp.float64
        else:
            self.dtype = jnp.float32
        self.z_emb = nn.Embed(95, self.config.hidden_channels)
        self.z_emb_ln = nn.LayerNorm(use_bias=False, use_scale=False)
        self.radial_emb = rbf_emb(
            num_basis=self.config.num_radial, 
            r_max=self.config.cutoff
        )
        
        self.radial_lin = nn.Sequential([
            nn.Dense(self.config.hidden_channels),
            nn.silu,
            nn.Dense(self.config.hidden_channels)
        ])
        
        self.neighbor_emb = NeighborEmb(self.config.hidden_channels)
        self.s_vector = S_vector(self.config.hidden_channels)
        
        self.lin = nn.Sequential([
            nn.Dense(self.config.hidden_channels // 4),
            nn.silu,
            nn.Dense(1)
        ])
        
     
        self.message_layers = [
            EquiMessagePassing(
                hidden_channels=self.config.hidden_channels,
                num_radial=self.config.num_radial,
                head=self.config.head,
                chi2=self.config.chi2,
                chi1=self.config.mp_chi1,
                has_dropout_flag=self.config.has_dropout_flag,
                has_norm_before_flag=self.config.has_norm_before_flag,
                has_norm_after_flag=self.config.has_norm_after_flag,
                hidden_channels_chi=self.config.hidden_channels_chi,
                complex_type=jnp.complex64 if self.config.dtype == "32" else jnp.complex128,
                reduce_mode=self.config.reduce_mode
            ) for _ in range(self.config.num_layers)
        ]
        
        self.ftes = [FTE(self.config.hidden_channels) for _ in range(self.config.num_layers)]

        self.kernel1 = self.param(
            'kernel1',
            nn.initializers.lecun_normal(),
            (self.config.hidden_channels, self.config.main_chi1 * 2)
        )
        
        self.kernels_real = self.param(
            'kernels_real',
            nn.initializers.lecun_normal(),
            (self.config.num_layers, self.config.hidden_channels, self.config.main_chi1, self.config.main_chi1)
        )
        
        self.kernels_imag = self.param(
            'kernels_imag',
            nn.initializers.lecun_normal(),
            (self.config.num_layers, self.config.hidden_channels, self.config.main_chi1, self.config.main_chi1)
        )

        self.last_layer = nn.Dense(self.config.output_dim if self.config.output_dim != 0 else 1)
        self.last_layer_quantum = nn.Dense(1)

        self.a = self.param(
            'a',
            nn.initializers.ones,
            (108,)
        ) * self.config.a
        
        self.b = self.param(
            'b',
            nn.initializers.ones,
            (108,)
        ) * self.config.b
        self.zbl = self.config.zbl # <--- 修正
        if self.zbl:
            
            fzbl_w_init = jnp.array([0.187, 0.3769, 0.189, 0.081, 0.003, 0.037, 0.0546, 0.0715], dtype=self.dtype)
            fzbl_w_init = jnp.clip(fzbl_w_init, a_min=0.0)
            fzbl_w_init = fzbl_w_init / (jnp.sum(fzbl_w_init) + 1e-12)
            
            self.fzbl_w = fzbl_w_init
            self.fzbl_b = jnp.array([3.20, 1.10, 0.102, 0.958, 1.28, 1.14, 1.69, 5], dtype=self.dtype)
            self.fzbl_gamma = jnp.array(1.001, dtype=self.dtype)
            self.fzbl_alpha = jnp.array(0.6032, dtype=self.dtype)
            self.fzbl_E2 = jnp.array(14.399645478425, dtype=self.dtype)
            self.fzbl_A0 = jnp.array(0.529177210903, dtype=self.dtype)
        self.inv_sqrt_2 = 1 / math.sqrt(2.0)
        self.pi = jnp.pi
    def __call__(self, data,prefix='infer'):

        pos = data.pos
        batch = data.batch
        z = data.z.astype(jnp.int64)
        edge_index = data.edge_index
        dist = data.edge_attr
        vecs = data.edge_vec
        

        z_emb = self.z_emb(z)
        z_emb = self.z_emb_ln(z_emb)
        radial_emb = self.radial_emb(dist)
        radial_hidden = self.radial_lin(radial_emb)

        rbounds = 0.5 * (jnp.cos(dist * self.pi / self.config.cutoff) + 1.0)
        radial_hidden = radial_hidden * rbounds[..., None]
        s = self.neighbor_emb(z, z_emb, edge_index, radial_hidden)
        vec = jnp.zeros((s.shape[0], 3, s.shape[1]))

        j = edge_index[0]
        i = edge_index[1]
        edge_diff = vecs / (dist[:, None] + self.config.eps)
        mean = segment_mean(pos[j], i, pos.shape[0])
        edge_cross = jnp.cross(pos[i] - mean[i], pos[j] - mean[i])
        edge_vertical = jnp.cross(edge_diff, edge_cross)
        
        edge_frame = jnp.stack([
            edge_diff, edge_cross, edge_vertical
        ], axis=-1)

        S_i_j = self.s_vector(s,jnp.expand_dims(edge_diff, axis=-1), edge_index, radial_hidden)

        scalrization1 = jnp.sum(
            S_i_j[i][:, :, None, :] * edge_frame[..., None],  
            axis=1
        )

        scalrization2 = jnp.sum(
            S_i_j[j][:, :, None, :] * edge_frame[..., None],
            axis=1
        )

        scalrization1 = scalrization1.at[:, 1, :].set(jnp.abs(scalrization1[:, 1, :]))
        scalrization2 = scalrization2.at[:, 1, :].set(jnp.abs(scalrization2[:, 1, :]))
        scalar3 = self.lin(jnp.transpose(scalrization1, (0, 2, 1)))
        scalar3 += jnp.transpose(scalrization1, (0, 2, 1))[..., 0][:, :, None]
        scalar3 = scalar3.squeeze(-1) / math.sqrt(self.config.hidden_channels)
        
        scalar4 = self.lin(jnp.transpose(scalrization2, (0, 2, 1)))
        scalar4 += jnp.transpose(scalrization2, (0, 2, 1))[..., 0][:, :, None]
        scalar4 = scalar4.squeeze(-1) / math.sqrt(self.config.hidden_channels)
        edge_weight = jnp.concatenate([scalar3, scalar4], axis=-1) * rbounds[:, None]
        edge_weight = jnp.concatenate([
            edge_weight, 
            radial_hidden, 
            radial_emb
        ], axis=-1)
        quantum = jnp.einsum('ik,bi->bk', self.kernel1, z_emb)
        real, imag = jnp.split(quantum, 2, axis=-1)
        quantum = real + 1j * imag 

        rope = None
        for idx in range(self.config.num_layers):
            message_layer = self.message_layers[idx]
            fte = self.ftes[idx]
            
            if rope is None:
                rope, ds, dvec = message_layer(s, vec, edge_index, radial_emb, edge_weight, edge_diff, None)
             
            else:
                rope, ds, dvec = message_layer(s, vec, edge_index, radial_emb, edge_weight, edge_diff, rope)
                
            s += ds
            vec += dvec

            kernel_real = self.kernels_real[idx]
            kernel_imag = self.kernels_imag[idx]
            kerneli = kernel_real + 1j * kernel_imag 
            s_complex = s + 0j
            quantum = jnp.einsum(
                'ikl,bi,bl->bk', 
                kerneli, 
                s_complex, 
                quantum
            )
            
            quantum = quantum / jnp.abs(quantum)

            ds, dvec = fte(s, vec)
            s += ds
            vec += dvec

        s = self.last_layer(s) + self.last_layer_quantum(jnp.concatenate([jnp.real(quantum), jnp.imag(quantum)], axis=-1)) / self.config.main_chi1
        V_graph = 0
        if self.zbl:
            r_e = dist
            Z_j = z[j]
            Z_i = z[i]

            # load constants (buffers) and cast to pos dtype/device
            w = self.fzbl_w  # (M,)
            b = self.fzbl_b  # (M,)
            gamma = self.fzbl_gamma
            alpha = self.fzbl_alpha
            E2 = self.fzbl_E2
            A0 = self.fzbl_A0

            # compute screening length a per edge: a = gamma * 0.8854 * a0 / (Z1^alpha + Z2^alpha)
            denom = jnp.power(Z_j, alpha) + jnp.power(Z_i, alpha)   # (E,)
            denom = jnp.clip(denom, a_min=1e-12)
            a_vals = gamma * 0.8854 * A0 / denom                    # (E,)
            x = r_e / a_vals                                        # (E,)

            # compute phi(x) = sum_i w_i * exp(-b_i * x)  (vectorized)
            # exp(- x[:,None] * b[None,:]) -> (E, M)
            exp_terms = jnp.exp(- x[:, jnp.newaxis] * b[jnp.newaxis, :])   # (E, M)
            phi_vals = exp_terms @ w                                    # (E,)

            # pair potential per edge: V_e = Z1*Z2 * E2 * phi / r
            V_edge = (Z_j * Z_i * E2) * (phi_vals / r_e)              # (E,)
            r_cut = 1.0  # you can make this self.fzbl_rcut buffer if you want configurable value

            # compute taper coefficient: cosine cutoff (smooth)
            # for r in [0, r_cut]: c = 0.5*(cos(pi * r / r_cut) + 1)
            # for r >= r_cut: c = 0
            # for safety, clamp r/r_cut in [0, 1]
            xrc = jnp.clip(r_e / r_cut, 0.0, 1.0)   # (E,)
            # cosine taper
            c = 0.5 * (jnp.cos(jnp.pi * xrc) + 1.0)    # (E,)
            # enforce zero beyond r_cut explicitly (cos already gives 0 at x=1 but clamp keeps numeric safe)
            c = jnp.where(r_e >= r_cut, jnp.zeros_like(c), c)

            # apply taper to edge potential
            V_edge = V_edge * c
            
            # aggregate edge energies to graph-level using jax.ops.segment_sum
            # Note: JAX uses segment_sum instead of scatter_add
            graph_idx = batch[i]  # map receiver node -> graph index (E,)
            V_graph = jax.ops.segment_sum(V_edge, graph_idx, num_segments=1) / 2.0
        a_values = self.a[z]
        b_values = self.b[z]
        
        if s.ndim == 2:
            s = a_values[:, None] * s + b_values[:, None]
        else:
            s = a_values * s + b_values
            s = s[:, None]
        assert jnp.ndim(batch) == 1, "Batch must be 1D"
        #assert jnp.all(batch >= 0)
       # num_segments = jnp.max(data.batch) + 1
        s_out = jax.ops.segment_sum(s, batch, num_segments=1)
       # if self.zbl:
        #    s_out[0] = s_out[0] + V_graph[0]
       
        return s_out[0] #+ V_graph[0]
        if self.config.compute_forces and self.config.compute_stress:
            
            if data.displacement is not None:
              stress, forces = self.cal_stress_and_force(s, pos, data.displacement, data.cell, prefix)
              stress = stress.view(-1, 3)
            else:
                stress = None
                forces = None
            return s, forces, stress
        elif self.config.compute_forces:
            forces = self.cal_forces(s, pos, prefix)
            return s, forces, None
        return s, None, None
    
   