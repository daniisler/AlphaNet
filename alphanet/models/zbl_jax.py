import jax.numpy as jnp
import jax

def zbl_interaction(dist, z_i, z_j, w, b, gamma, alpha, E2, A0, r_cut=1.0):
    """
    计算 ZBL Pair Potential (JAX functional implementation)
    """
    r_e = dist
    
    # 防止除零
    denom = jnp.power(z_j, alpha) + jnp.power(z_i, alpha)
    denom = jnp.clip(denom, a_min=1e-12)
    
    a_vals = gamma * 0.8854 * A0 / denom
    x = r_e / a_vals
    
    # phi(x) = sum(w * exp(-b*x))
    # w: (M,), b: (M,), x: (Edges,)
    # exp(- x[:, None] * b[None, :]) -> (Edges, M)
    exp_terms = jnp.exp(-x[:, jnp.newaxis] * b[jnp.newaxis, :])
    phi_vals = jnp.dot(exp_terms, w)
    
    V_edge = (z_j * z_i * E2) * (phi_vals / r_e)
    
    # Cutoff smoothing
    xrc = jnp.clip(r_e / r_cut, 0.0, 1.0)
    c = 0.5 * (jnp.cos(jnp.pi * xrc) + 1.0)
    c = jnp.where(r_e >= r_cut, jnp.zeros_like(c), c)
    
    return V_edge * c

def get_default_zbl_params(dtype=jnp.float32):
    fzbl_w = jnp.array([0.187, 0.3769, 0.189, 0.081, 0.003, 0.037, 0.0546, 0.0715], dtype=dtype)
    # Normalize
    fzbl_w = jnp.clip(fzbl_w, a_min=0.0)
    fzbl_w = fzbl_w / (jnp.sum(fzbl_w) + 1e-12)
    
    fzbl_b = jnp.array([3.20, 1.10, 0.102, 0.958, 1.28, 1.14, 1.69, 5.0], dtype=dtype)
    
    params = {
        "w": fzbl_w,
        "b": fzbl_b,
        "gamma": jnp.array(1.001, dtype=dtype),
        "alpha": jnp.array(0.6032, dtype=dtype),
        "E2": jnp.array(14.399645478425, dtype=dtype),
        "A0": jnp.array(0.529177210903, dtype=dtype)
    }
    return params