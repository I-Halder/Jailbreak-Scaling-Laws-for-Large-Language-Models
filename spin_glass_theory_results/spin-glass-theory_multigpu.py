#!/usr/bin/env python3
"""
Usage (single GPU):
python spin-glass-theory_multigpu.py \
    --N 24 --beta 3.0 --j0 1.0 --m_unsafe 3 \
    --h_values 0 \
    --k_values 1,2,4,8,16,32 \
    --n_disorder 64 --n_sel 8 \
    --pd_B 8 --pd_num_perms 32 \
    --device cuda --threads 1 \
    --out ./theory_figures/spin-glass-multigpu-figures/spin-glass-theory-multigpu.png

Usage (multi-GPU data/model parallel):
torchrun --nproc_per_node=2 spin-glass-theory_multigpu.py \
  --N 24 --beta 10.0 --j0 1.0 --m_unsafe 1 \
  --h_values 0,0.05,0.1,0.15,0.2 \
  --k_values 1,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50,52,54,56,58,60,62,64 \
  --n_disorder 1024 --n_sel 8 \
  --pd_B 8 --pd_num_perms 32 \
  --device cuda --threads 1 \
  --max_states_per_rank 20000000 \
  --out ./spin-glass-theory-multigpu-N24-disorder1024-m1-curvefit.png
  
Finite-N verification scaffold for weak-field theorem in spin-glass theory.

Key fix: Properly compute Delta_q = q_{l+1} - q_l from finite-N overlap structure
to get the effective field parameter lambda = beta * h * N * Delta_q.
"""

import argparse
import atexit
import builtins
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.distributed as dist
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit

# -------------------------
# Numerical helpers
# -------------------------

def logmeanexp(logx: np.ndarray, axis: int = 0) -> np.ndarray:
    """Compute log(mean(exp(logx))) stably along axis."""
    m = np.max(logx, axis=axis, keepdims=True)
    all_neginf = ~np.isfinite(m)
    shifted = logx - m
    exp_shifted = np.exp(shifted, where=np.isfinite(shifted), out=np.zeros_like(shifted))
    mean_exp = np.mean(exp_shifted, axis=axis, keepdims=True)
    out = m + np.log(mean_exp, where=(mean_exp > 0), out=np.full_like(mean_exp, -np.inf))
    out[all_neginf] = -np.inf
    return np.squeeze(out, axis=axis)


@dataclass
class DistContext:
    enabled: bool
    rank: int
    world_size: int
    local_rank: int

    @property
    def is_root(self) -> bool:
        return self.rank == 0


def maybe_init_distributed() -> Tuple[bool, int, int, int]:
    """Initialize torch.distributed if launched with torchrun."""
    # Ensure IPv4-friendly defaults when torchrun did not set them (helps on some clusters).
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")

    if dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", rank % max(world_size, 1)))
        return True, rank, world_size, local_rank

    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", rank % max(world_size, 1)))
        dist.init_process_group(backend="nccl", init_method="env://")
        torch.cuda.set_device(local_rank)
        return True, rank, world_size, local_rank

    return False, 0, 1, 0


def global_logsumexp(logw_local: torch.Tensor, dist_ctx: DistContext) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute log-sum-exp over all ranks without gathering full tensors."""
    if not dist_ctx.enabled:
        logZ = torch.logsumexp(logw_local, dim=0)
        return logZ, torch.exp(logw_local - logZ)

    max_local = torch.max(logw_local)
    max_global = max_local.clone()
    dist.all_reduce(max_global, op=dist.ReduceOp.MAX)

    exp_sum_local = torch.exp(logw_local - max_global).sum()
    exp_sum_global = exp_sum_local.clone()
    dist.all_reduce(exp_sum_global, op=dist.ReduceOp.SUM)

    logZ = max_global + torch.log(exp_sum_global)
    probs_local = torch.exp(logw_local - logZ)
    return logZ, probs_local


def broadcast_long_tensor(tensor: torch.Tensor, dist_ctx: DistContext) -> torch.Tensor:
    """Broadcast a 1D int64 tensor from rank 0; handles variable length."""
    if not dist_ctx.enabled:
        return tensor

    device = torch.device("cuda", dist_ctx.local_rank) if torch.cuda.is_available() else torch.device("cpu")
    length = torch.tensor([tensor.numel()], device=device, dtype=torch.int64)
    dist.broadcast(length, src=0)

    if not dist_ctx.is_root:
        tensor = torch.empty((length.item(),), device=device, dtype=torch.int64)

    if length.item() > 0:
        dist.broadcast(tensor, src=0)
    return tensor


def broadcast_from_rank0(tensor: torch.Tensor) -> torch.Tensor:
    """Broadcast tensor from rank 0 to all ranks (no-op for single GPU)."""
    if not dist.is_initialized():
        return tensor
    dist.broadcast(tensor, src=0)
    return tensor


def gather_variable_tensor(tensor: torch.Tensor, world_size: int, dst: int = 0):
    """Gather tensors with potentially different first-dim sizes to dst."""
    if not dist.is_initialized():
        return tensor

    device = tensor.device
    length = torch.tensor([tensor.shape[0]], device=device, dtype=torch.int64)
    lengths = [torch.zeros_like(length) for _ in range(world_size)]
    dist.all_gather(lengths, length)
    max_len = int(torch.stack(lengths).max().item())

    if tensor.ndim == 1:
        pad_shape = (max_len,)
    else:
        pad_shape = (max_len,) + tensor.shape[1:]

    pad = max_len - tensor.shape[0]
    if pad > 0:
        pad_tensor = torch.zeros(pad_shape, device=device, dtype=tensor.dtype)
        pad_tensor[: tensor.shape[0]] = tensor
        tensor_padded = pad_tensor
    else:
        tensor_padded = tensor

    gather_list = [torch.zeros_like(tensor_padded) for _ in range(world_size)] if dist.get_rank() == dst else None
    dist.gather(tensor_padded, gather_list=gather_list, dst=dst)

    if dist.get_rank() != dst:
        return None

    parts = []
    for buf, l in zip(gather_list, lengths):
        parts.append(buf[: l.item()])
    return torch.cat(parts, dim=0)


def bootstrap_curves_and_nu(
    logk: np.ndarray,
    loggap_samples: np.ndarray,
    lambda_samples: np.ndarray,
    B: int = 400,
    seed: int = 0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float, float, float]:
    """
    Bootstrap over samples. For each bootstrap draw:
      - compute curve1: log E[gap] via logmeanexp
      - compute Pi_k = 1 - exp(curve1)
      - compute curve2: log(-log(Pi_k))
      - fit nu from curve1 vs logk (nu = -slope)
      - compute mean lambda
    """
    rng = np.random.default_rng(seed)
    n = loggap_samples.shape[0]
    n_k = loggap_samples.shape[1]

    curve1 = np.zeros((B, n_k), dtype=np.float64)
    curve2 = np.zeros((B, n_k), dtype=np.float64)
    nus = np.zeros((B,), dtype=np.float64)
    lambdas = np.zeros((B,), dtype=np.float64)

    for b in range(B):
        idx = rng.integers(0, n, size=n)
        sample = loggap_samples[idx]  # [n, n_k]
        lam_sample = lambda_samples[idx]
        
        lm = logmeanexp(sample, axis=0)  # log E[gap]
        curve1[b] = lm

        gap = np.exp(lm)  # E[gap] in (0,1]
        Pi = 1.0 - gap

        # clamp Pi into (0,1) for stable log(-log(.))
        Pi = np.clip(Pi, 1e-300, 1.0 - 1e-16)
        y2 = np.log(-np.log(Pi))
        curve2[b] = y2

        slope, _ = np.polyfit(logk, lm, deg=1)
        nus[b] = -slope
        lambdas[b] = np.mean(lam_sample)

    return (
        curve1.mean(axis=0),
        curve1.std(axis=0, ddof=1),
        curve2.mean(axis=0),
        curve2.std(axis=0, ddof=1),
        float(nus.mean()),
        float(nus.std(ddof=1)),
        float(lambdas.mean()),
        float(lambdas.std(ddof=1)),
    )


# -------------------------
# Spin configurations
# -------------------------

def all_spin_configs(N: int, device: torch.device, dtype=torch.float32) -> torch.Tensor:
    """Return S of shape [2^N, N] with entries in {-1, +1}."""
    M = 1 << N
    ints = torch.arange(M, dtype=torch.int64)
    bits = ((ints[:, None] >> torch.arange(N, dtype=torch.int64)[None, :]) & 1)
    S = bits.to(dtype=dtype) * 2.0 - 1.0
    return S.to(device)


def all_spin_configs_chunk(
    N: int,
    start: int,
    end: int,
    device: torch.device,
    dtype=torch.float32
) -> torch.Tensor:
    """Return a slice of spin configs in [start, end) with entries in {-1, +1}."""
    ints = torch.arange(start, end, dtype=torch.int64, device=device)
    bits = ((ints[:, None] >> torch.arange(N, dtype=torch.int64, device=device)[None, :]) & 1)
    S = bits.to(dtype=dtype) * 2.0 - 1.0
    return S


def all_spin_configs_batched(
    N: int,
    start: int,
    end: int,
    batch_size: int,
    device: torch.device,
    dtype=torch.float32
) -> torch.Tensor:
    """Build all spin configs in [start, end) using fixed-size batches."""
    total = end - start
    S = torch.empty((total, N), device=device, dtype=dtype)
    offset = 0
    for bstart in range(start, end, batch_size):
        bend = min(end, bstart + batch_size)
        chunk = all_spin_configs_chunk(N, bstart, bend, device=device, dtype=dtype)
        S[offset:offset + (bend - bstart)] = chunk
        offset += (bend - bstart)
    return S


def sample_spin_configs(
    N: int,
    count: int,
    device: torch.device,
    dtype=torch.float32,
    seed: int = 0,
) -> torch.Tensor:
    """Return a random subset of spin configs of shape [count, N] with entries in {-1,+1}."""
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    bits = torch.randint(0, 2, size=(count, N), device=device, generator=g, dtype=torch.int64)
    S = bits.to(dtype=dtype) * 2.0 - 1.0
    return S


def spins_to_int(S: torch.Tensor) -> torch.Tensor:
    """Encode spins {-1,+1} into int64 bitmask (bit i = 1 iff spin_i = +1)."""
    bits = (S > 0).to(torch.int64)
    N = S.shape[-1]
    weights = (1 << torch.arange(N, device=S.device, dtype=torch.int64))
    return (bits * weights).sum(dim=-1)


# -------------------------
# SK model
# -------------------------

def sample_sk_J(N: int, j0: float, device: torch.device, dtype=torch.float32) -> torch.Tensor:
    """SK couplings: J_ij ~ N(0, j0^2/N), symmetric, zero diagonal."""
    std = j0 / math.sqrt(N)
    A = torch.randn((N, N), device=device, dtype=dtype) * std
    J = torch.triu(A, diagonal=1)
    J = J + J.T
    return J


def sk_energy(S: torch.Tensor, J: torch.Tensor) -> torch.Tensor:
    """E(sigma) = - sum_{i<j} J_ij sigma_i sigma_j = -0.5 * sigma^T J sigma."""
    SJ = S @ J
    return -0.5 * (S * SJ).sum(dim=1)


@torch.no_grad()
def greedy_descent_to_minima(
    S: torch.Tensor,
    J: torch.Tensor,
    max_steps: int = 128,
    tol: float = 1e-12
) -> torch.Tensor:
    """Vectorized greedy single-spin-flip descent for all configs."""
    S_cur = S.clone()
    for _ in range(max_steps):
        Hloc = S_cur @ J
        dE = 2.0 * S_cur * Hloc
        min_dE, idx = dE.min(dim=1)
        mask = (min_dE < -tol)
        if not mask.any():
            break
        rows = torch.nonzero(mask, as_tuple=False).squeeze(1)
        cols = idx[rows]
        S_cur[rows, cols] *= -1.0
    return S_cur


# -------------------------
# Clusters and weights
# -------------------------

@dataclass
class ClusterDecomp:
    state_cluster_id: torch.Tensor  # [M] int64
    centers: torch.Tensor           # [K,N]
    weights_teacher: torch.Tensor   # [K]


@torch.no_grad()
def compute_teacher_clusters_and_weights(
    S: torch.Tensor,
    J: torch.Tensor,
    E: torch.Tensor,
    beta: float
) -> ClusterDecomp:
    """Clusters = basins of greedy descent to local minima; weights = sum Gibbs probs in basin."""
    device = S.device
    dtype = S.dtype

    logw = -beta * E
    logZ = torch.logsumexp(logw, dim=0)
    p = torch.exp(logw - logZ)

    S_min = greedy_descent_to_minima(S, J)
    min_code = spins_to_int(S_min)

    uniq_codes, inv = torch.unique(min_code, sorted=True, return_inverse=True)
    K = uniq_codes.numel()

    inv_cpu = inv.detach().cpu().numpy()
    first = np.full((K,), -1, dtype=np.int64)
    for i, cid in enumerate(inv_cpu):
        if first[cid] < 0:
            first[cid] = i
    first_indices = torch.tensor(first, device=device, dtype=torch.int64)
    centers = S_min[first_indices].to(dtype=dtype)

    weights = torch.zeros((K,), device=device, dtype=dtype)
    weights.scatter_add_(0, inv, p)
    weights = weights / weights.sum().clamp_min(1e-12)

    return ClusterDecomp(state_cluster_id=inv, centers=centers, weights_teacher=weights)


@torch.no_grad()
def compute_teacher_clusters_and_weights_distributed(
    S_local: torch.Tensor,
    J: torch.Tensor,
    E_local: torch.Tensor,
    beta: float,
    dist_ctx: DistContext
) -> ClusterDecomp:
    """Cluster decomposition without gathering all states to rank 0."""
    if not dist_ctx.enabled:
        return compute_teacher_clusters_and_weights(S_local, J, E_local, beta)

    device = S_local.device
    dtype = S_local.dtype

    # Teacher probabilities across all ranks
    logw_local = -beta * E_local
    logZ, p_local = global_logsumexp(logw_local, dist_ctx)

    # Local minima per rank
    S_min_local = greedy_descent_to_minima(S_local, J)
    min_code_local = spins_to_int(S_min_local)

    # Unique minima per rank (keep first occurrence for center)
    uniq_codes_local, inv_local = torch.unique(min_code_local, sorted=True, return_inverse=True)
    first_indices = torch.zeros_like(uniq_codes_local)
    for idx, code in enumerate(uniq_codes_local):
        pos = torch.nonzero(min_code_local == code, as_tuple=False).squeeze(1)[0]
        first_indices[idx] = pos
    centers_local = S_min_local[first_indices]

    # Gather unique codes and centers to rank 0
    codes_all = gather_variable_tensor(uniq_codes_local, dist_ctx.world_size, dst=0)
    centers_all = gather_variable_tensor(centers_local, dist_ctx.world_size, dst=0)

    if dist_ctx.is_root:
        # Build mapping code -> center (first occurrence wins)
        code_to_center = {}
        for code, center in zip(codes_all.cpu().tolist(), centers_all.cpu()):
            if code not in code_to_center:
                code_to_center[code] = center.to(dtype=dtype)

        global_codes = torch.tensor(list(code_to_center.keys()), device=device, dtype=torch.int64)
        centers = torch.stack(list(code_to_center.values()), dim=0).to(device=device, dtype=dtype)

        # Sort for consistent searchsorted on all ranks
        order = torch.argsort(global_codes)
        global_codes = global_codes[order]
        centers = centers[order]
    else:
        global_codes = torch.empty((0,), device=device, dtype=torch.int64)
        centers = torch.empty((0, S_local.shape[1]), device=device, dtype=dtype)

    # Broadcast global codes and centers
    k_tensor = torch.tensor([global_codes.numel()], device=device, dtype=torch.int64)
    dist.broadcast(k_tensor, src=0)
    K = int(k_tensor.item())

    if not dist_ctx.is_root:
        global_codes = torch.empty((K,), device=device, dtype=torch.int64)
        centers = torch.empty((K, S_local.shape[1]), device=device, dtype=dtype)

    if K > 0:
        dist.broadcast(global_codes, src=0)
        dist.broadcast(centers, src=0)

    # Map local states to cluster ids
    if K > 0:
        state_cluster_id = torch.searchsorted(global_codes, min_code_local)
    else:
        state_cluster_id = torch.zeros_like(min_code_local)

    # Teacher weights (global probability mass per cluster)
    weights_local = torch.zeros((K,), device=device, dtype=dtype)
    if K > 0:
        weights_local.scatter_add_(0, state_cluster_id, p_local)
    dist.all_reduce(weights_local, op=dist.ReduceOp.SUM)
    weights_teacher = weights_local / weights_local.sum().clamp_min(1e-12)

    return ClusterDecomp(
        state_cluster_id=state_cluster_id,
        centers=centers,
        weights_teacher=weights_teacher,
    )


# -------------------------
# Overlap estimation (NEW!)
# -------------------------

@torch.no_grad()
def estimate_delta_q(
    S: torch.Tensor,
    decomp: ClusterDecomp,
    unsafe_cluster_idx: torch.Tensor
) -> float:
    """
    Estimate Delta_q = q_{l+1} - q_l from actual overlap structure.
    
    q_{l+1} ≈ average overlap of states in unsafe basins with their own center
    q_l ≈ average overlap between different unsafe cluster centers
    
    Returns Delta_q for use in lambda = beta * h * N * Delta_q
    """
    device = S.device
    N = S.shape[1]
    
    if unsafe_cluster_idx.numel() == 0:
        return 0.0
    
    # Get unsafe centers
    unsafe_centers = decomp.centers[unsafe_cluster_idx.to(device)]  # [m, N]
    m = unsafe_centers.shape[0]
    
    # Estimate q_{l+1}: average overlap of states with their own cluster center
    q_lplus1_samples = []
    for i, cluster_idx in enumerate(unsafe_cluster_idx):
        # Get states in this cluster
        mask = (decomp.state_cluster_id == cluster_idx.item())
        if mask.sum() == 0:
            continue
        states_in_cluster = S[mask.to(device)]  # [n_states, N]
        center = unsafe_centers[i:i+1]  # [1, N]
        
        # Compute overlaps: R(s, center) = (1/N) * sum_j s_j * center_j
        overlaps = (states_in_cluster @ center.T).squeeze(1) / N  # [n_states]
        q_lplus1_samples.append(overlaps.mean().item())
    
    q_lplus1 = float(np.mean(q_lplus1_samples)) if q_lplus1_samples else 0.5
    
    # print("Value of q_lplus1_samples: ", q_lplus1_samples)
    
    # Estimate q_l: average overlap between different cluster centers
    if m > 1:
        # Compute pairwise overlaps between centers
        center_overlaps = (unsafe_centers @ unsafe_centers.T) / N  # [m, m]
        # Take off-diagonal elements
        mask = ~torch.eye(m, device=device, dtype=torch.bool)
        q_l = center_overlaps[mask].abs().mean().item()
    else:
        # Single cluster: use overlap with other (safe) centers as proxy
        all_centers = decomp.centers
        safe_centers = all_centers[[i for i in range(all_centers.shape[0]) 
                                   if i not in unsafe_cluster_idx.tolist()]]
        if safe_centers.shape[0] > 0:
            cross_overlaps = (unsafe_centers @ safe_centers.T) / N
            q_l = cross_overlaps.abs().mean().item()
        else:
            q_l = 0.0
    
    Delta_q = max(q_lplus1 - q_l, 0.01)  # Ensure positive
    
    return Delta_q


@torch.no_grad()
def estimate_delta_q_distributed(
    S_local: torch.Tensor,
    decomp: ClusterDecomp,
    unsafe_cluster_idx: torch.Tensor,
    dist_ctx: DistContext
) -> float:
    """Distributed Delta_q using local states and global centers."""
    if not dist_ctx.enabled:
        return estimate_delta_q(S_local, decomp, unsafe_cluster_idx)

    device = S_local.device
    N = S_local.shape[1]

    if unsafe_cluster_idx.numel() == 0:
        return 0.0

    unsafe_centers = decomp.centers[unsafe_cluster_idx]
    m = unsafe_centers.shape[0]

    # q_{l+1}: average overlap of states with their own unsafe center
    sums = torch.zeros((m,), device=device, dtype=torch.float64)
    counts = torch.zeros((m,), device=device, dtype=torch.float64)

    for idx, cid in enumerate(unsafe_cluster_idx):
        mask = (decomp.state_cluster_id == cid)
        if mask.any():
            states = S_local[mask]
            overlaps = (states @ unsafe_centers[idx].unsqueeze(1))[:, 0] / N
            sums[idx] = overlaps.sum(dtype=torch.float64)
            counts[idx] = torch.tensor(float(overlaps.numel()), device=device, dtype=torch.float64)

    dist.all_reduce(sums, op=dist.ReduceOp.SUM)
    dist.all_reduce(counts, op=dist.ReduceOp.SUM)

    valid = counts > 0
    q_lplus1 = (sums[valid] / counts[valid]).mean().item() if valid.any() else 0.5

    # q_l: center overlaps only
    if m > 1:
        center_overlaps = (unsafe_centers @ unsafe_centers.T) / N
        mask = ~torch.eye(m, device=device, dtype=torch.bool)
        q_l = center_overlaps[mask].abs().mean().item()
    else:
        q_l = 0.0

    return max(q_lplus1 - q_l, 0.01)


# -------------------------
# Size-biased ordering and GEM/PD MLE
# -------------------------

@torch.no_grad()
def size_biased_permutation(weights: torch.Tensor, B: int, rng: torch.Generator) -> torch.Tensor:
    """Sample size-biased ordering without replacement."""
    w = weights.clamp_min(0)
    s = w.sum()
    if s <= 0:
        return torch.empty((0,), device=weights.device, dtype=torch.int64)
    w = w / s

    K = w.numel()
    B = min(B, K)
    chosen = []
    w_work = w.clone()
    for _ in range(B):
        idx = torch.multinomial(w_work, num_samples=1, replacement=True, generator=rng).item()
        chosen.append(idx)
        w_work[idx] = 0.0
        s2 = w_work.sum()
        if s2 <= 0:
            break
        w_work = w_work / s2

    return torch.tensor(chosen, device=weights.device, dtype=torch.int64)


def log_beta_pdf(v: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    eps = 1e-12
    v = v.clamp(eps, 1 - eps)
    return (a - 1) * torch.log(v) + (b - 1) * torch.log(1 - v) - (
        torch.lgamma(a) + torch.lgamma(b) - torch.lgamma(a + b)
    )


@torch.no_grad()
def gem_mle_m_from_size_biased_weights(
    weights: torch.Tensor,
    B: int,
    num_perms: int,
    grid: int,
    rng: torch.Generator
) -> float:
    """MLE for m in GEM(m): V_i ~ Beta(1-m, i m)."""
    device = weights.device
    dtype = weights.dtype

    w = weights.clamp_min(0)
    s = w.sum()
    if s <= 0:
        return float("nan")
    w = w / s

    if w.numel() < 2:
        return float("nan")

    m_grid = torch.linspace(0.01, 0.99, steps=grid, device=device, dtype=dtype)
    ll_acc = torch.zeros_like(m_grid)

    for _ in range(num_perms):
        order = size_biased_permutation(w, B=B, rng=rng)
        if order.numel() == 0:
            continue
        w_ord = w[order]  # NO RENORMALIZATION

        rem = torch.tensor(1.0, device=device, dtype=dtype)
        V_list = []
        for wi in w_ord:
            vi = (wi / rem).clamp(1e-12, 1 - 1e-12)
            V_list.append(vi)
            rem = (rem - wi).clamp_min(1e-12)
        V = torch.stack(V_list)
        B_eff = V.numel()
        i = torch.arange(1, B_eff + 1, device=device, dtype=dtype)

        a = (1.0 - m_grid)[:, None]
        b = (m_grid[:, None] * i[None, :])
        ll = log_beta_pdf(V[None, :], a, b).sum(dim=1)
        ll_acc += ll

    ll_acc = ll_acc / max(1, num_perms)
    return float(m_grid[torch.argmax(ll_acc)].item())


# -------------------------
# Student model (microscopic)
# -------------------------

@torch.no_grad()
def compute_student_probs_microscopic(
    S: torch.Tensor,
    E_teacher: torch.Tensor,
    beta: float,
    h: float,
    unsafe_centers: torch.Tensor
) -> torch.Tensor:
    """
    Student energy: E_h(s) = E_teacher(s) - h * sum_{a in unsafe} sum_j s_j * center_a^j
    
    This implements the Hamiltonian from the paper:
    H^{(h)}_J(sigma) = H_J(sigma) - sum_{i=1}^m h * sum_{j=1}^N sigma_j sigma*_{i}(J)_j
    
    Returns p_h over states.
    """
    logw = -beta * E_teacher
    if unsafe_centers.numel() > 0 and abs(h) > 0:
        # Field = h * sum over unsafe centers of (sum over spins of s_j * center_j)
        # This is h * sum_a <s, center_a> where <.,.> is the inner product
        dots = S @ unsafe_centers.T  # [M, m]
        field = h * dots.sum(dim=1)  # [M]
        logw = -beta * (E_teacher - field)
    logZ = torch.logsumexp(logw, dim=0)
    return torch.exp(logw - logZ)


@torch.no_grad()
def compute_student_probs_microscopic_sharded(
    S_local: torch.Tensor,
    E_teacher_local: torch.Tensor,
    beta: float,
    h: float,
    unsafe_centers: torch.Tensor,
    dist_ctx: DistContext
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Student probabilities with sharded states and global normalization."""
    logw = -beta * E_teacher_local
    if unsafe_centers.numel() > 0 and abs(h) > 0:
        dots = S_local @ unsafe_centers.T
        field = h * dots.sum(dim=1)
        logw = -beta * (E_teacher_local - field)

    logZ, p_local = global_logsumexp(logw, dist_ctx)
    return p_local, logZ


# -------------------------
# Main
# -------------------------

def parse_float_list(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def parse_int_list(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def with_suffix(path: str, suffix: str) -> str:
    base, ext = os.path.splitext(path)
    if ext.lower() in [".png", ".pdf", ".jpg", ".jpeg", ".svg"]:
        return base + suffix + ext
    return path + suffix

def model_log_corrected(x, a, b, l):
    x = np.asarray(x, dtype=np.float64)
    return -a * np.log(x) - 0.5*(l**2)*x + b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, required=True)
    ap.add_argument("--beta", type=float, required=True)
    ap.add_argument("--j0", type=float, required=True)
    ap.add_argument("--m_unsafe", type=int, required=True)

    ap.add_argument("--h_values", type=str, required=True)
    ap.add_argument("--k_values", type=str, required=True)

    ap.add_argument("--n_disorder", type=int, default=32)
    ap.add_argument("--n_sel", type=int, default=8)

    ap.add_argument("--pd_B", type=int, default=8)
    ap.add_argument("--pd_num_perms", type=int, default=32)
    ap.add_argument("--pd_grid", type=int, default=199)

    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", type=str, default="float32", choices=["float32", "float64"])
    ap.add_argument("--threads", type=int, default=1)

    ap.add_argument("--bootstrap_B", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="spin_glass_theory_fixed.png")
    ap.add_argument("--max_states_per_rank", type=int, default=1_000_000,
                    help="Cap on number of spin states per rank; if exceeded, random subset is sampled")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)

    dist_enabled, rank, world_size, local_rank = maybe_init_distributed()
    dist_ctx = DistContext(dist_enabled, rank, world_size, local_rank)
    device = torch.device("cuda", local_rank) if dist_enabled else torch.device(args.device)
    dtype = torch.float64 if args.dtype == "float64" else torch.float32

    # Capture printed output to JSON for the root rank.
    log_messages: List[str] = []
    _orig_print = builtins.print

    def _print_and_log(*args, **kwargs):
        _orig_print(*args, **kwargs)
        try:
            msg = " ".join(str(a) for a in args)
            log_messages.append(msg)
        except Exception as e:  # pragma: no cover - defensive
            _orig_print(f"[WARN] Failed to log message: {e}")

    if dist_ctx.is_root:
        builtins.print = _print_and_log

        def _flush_logs():
            try:
                out_path = Path(args.out)
                log_path = out_path.with_suffix(out_path.suffix + ".logs.json") if out_path.suffix else out_path.with_suffix(".logs.json")
                log_path.parent.mkdir(parents=True, exist_ok=True)
                with log_path.open("w", encoding="utf-8") as f:
                    json.dump(log_messages, f, indent=2)
            except Exception as e:  # pragma: no cover - defensive
                _orig_print(f"[WARN] Failed to write logs: {e}")

        atexit.register(_flush_logs)

    N = args.N
    beta = args.beta
    j0 = args.j0
    m_unsafe = args.m_unsafe

    h_values = parse_float_list(args.h_values)
    k_values = sorted(parse_int_list(args.k_values))
    logk = np.log(np.asarray(k_values, dtype=np.float64))

    rng_pd = torch.Generator(device="cpu")
    rng_pd.manual_seed(args.seed)

    M = 1 << N
    if dist_enabled:
        chunk = (M + world_size - 1) // world_size
        start = rank * chunk
        end = min(M, (rank + 1) * chunk)
        num_states = end - start
        if num_states > args.max_states_per_rank:
            if dist_ctx.is_root:
                print(f"[INFO] Requested {num_states} states per rank exceeds cap {args.max_states_per_rank}; processing in batches.")
            S_local = all_spin_configs_batched(
                N=N,
                start=start,
                end=end,
                batch_size=args.max_states_per_rank,
                device=device,
                dtype=dtype,
            )
        else:
            S_local = all_spin_configs_chunk(N, start, end, device=device, dtype=dtype)
    else:
        if M > args.max_states_per_rank:
            print(f"[INFO] Requested {M} states exceeds cap {args.max_states_per_rank}; processing in batches.")
            S_local = all_spin_configs_batched(
                N=N,
                start=0,
                end=M,
                batch_size=args.max_states_per_rank,
                device=device,
                dtype=dtype,
            )
        else:
            S_local = all_spin_configs(N, device=device, dtype=dtype)

    teacher_m_hats = []
    per_h_loggap_samples: Dict[float, List[np.ndarray]] = {h: [] for h in h_values}
    per_h_lambda_samples: Dict[float, List[float]] = {h: [] for h in h_values}
    per_h_student_m_hats: Dict[float, List[float]] = {h: [] for h in h_values}
    per_h_delta_q_samples: Dict[float, List[float]] = {h: [] for h in h_values}

    for t in range(args.n_disorder):
        if dist_enabled:
            if rank == 0:
                J = sample_sk_J(N, j0=j0, device=torch.device("cuda", local_rank), dtype=dtype)
            else:
                J = torch.empty((N, N), device=device, dtype=dtype)
            J = broadcast_from_rank0(J)
        else:
            J = sample_sk_J(N, j0=j0, device=device, dtype=dtype)

        E_local = sk_energy(S_local, J)

        decomp = compute_teacher_clusters_and_weights_distributed(
            S_local=S_local,
            J=J,
            E_local=E_local,
            beta=beta,
            dist_ctx=dist_ctx,
        )

        wT = decomp.weights_teacher
        K = wT.numel()

        # Teacher m_l
        if dist_ctx.is_root:
            m_hat_T = gem_mle_m_from_size_biased_weights(
                weights=wT.detach().cpu(),
                B=args.pd_B,
                num_perms=args.pd_num_perms,
                grid=args.pd_grid,
                rng=rng_pd
            )
            if np.isfinite(m_hat_T):
                teacher_m_hats.append(m_hat_T)

        # selections
        for sidx in range(args.n_sel):
            if dist_ctx.is_root:
                sel_gen = torch.Generator(device="cpu")
                sel_gen.manual_seed(args.seed * 10_000 + t * 100 + sidx)
                order = size_biased_permutation(wT.detach().cpu(), B=min(m_unsafe, K), rng=sel_gen)
                unsafe_cluster_idx = order[: min(m_unsafe, order.numel())].to(torch.int64).to(device)
            else:
                unsafe_cluster_idx = torch.empty((0,), device=device, dtype=torch.int64)

            unsafe_cluster_idx = broadcast_long_tensor(unsafe_cluster_idx, dist_ctx)

            unsafe_centers = (
                decomp.centers[unsafe_cluster_idx]
                if unsafe_cluster_idx.numel() > 0
                else torch.empty((0, N), device=device, dtype=dtype)
            )

            Delta_q = estimate_delta_q_distributed(S_local, decomp, unsafe_cluster_idx, dist_ctx)

            unsafe_state_mask_local = torch.isin(decomp.state_cluster_id, unsafe_cluster_idx)
            unsafe_state_idx_local = torch.nonzero(unsafe_state_mask_local, as_tuple=False).squeeze(1)

            for h in h_values:
                lambda_eff = beta * h * N * Delta_q

                pS_local, _ = compute_student_probs_microscopic_sharded(
                    S_local=S_local,
                    E_teacher_local=E_local,
                    beta=beta,
                    h=h,
                    unsafe_centers=unsafe_centers,
                    dist_ctx=dist_ctx,
                )

                A_local = pS_local[unsafe_state_idx_local].sum()
                if dist_ctx.enabled:
                    dist.all_reduce(A_local, op=dist.ReduceOp.SUM)
                    A_val = float(min(max(A_local.item(), 0.0), 1.0)) if dist_ctx.is_root else 0.0
                else:
                    A_val = float(min(max(A_local.item(), 0.0), 1.0))

                one_minus_A = 1.0 - A_val

                if dist_ctx.is_root:
                    lg = []
                    if one_minus_A <= 0.0:
                        lg = [-np.inf for _ in k_values]
                    else:
                        log_one_minus_A = math.log(one_minus_A)
                        for k in k_values:
                            lg.append(k * log_one_minus_A)
                    per_h_loggap_samples[h].append(np.asarray(lg, dtype=np.float64))
                    per_h_lambda_samples[h].append(lambda_eff)
                    per_h_delta_q_samples[h].append(Delta_q)

                wS_local = torch.zeros((K,), device=device, dtype=dtype)
                if K > 0:
                    wS_local.scatter_add_(0, decomp.state_cluster_id, pS_local)

                wS_global = wS_local.clone()
                if dist_ctx.enabled:
                    dist.all_reduce(wS_global, op=dist.ReduceOp.SUM)
                wS_global = wS_global / wS_global.sum().clamp_min(1e-12)

                if dist_ctx.is_root:
                    m_hat_S = gem_mle_m_from_size_biased_weights(
                        weights=wS_global.detach().cpu(),
                        B=args.pd_B,
                        num_perms=max(8, args.pd_num_perms // 2),
                        grid=args.pd_grid,
                        rng=rng_pd
                    )
                    if np.isfinite(m_hat_S):
                        per_h_student_m_hats[h].append(m_hat_S)

    if dist_ctx.enabled:
        dist.barrier()
        if not dist_ctx.is_root:
            return

    # Teacher summary
    teacher_m_hats = np.asarray(teacher_m_hats, dtype=np.float64)
    teacher_m_mean = float(np.mean(teacher_m_hats)) if teacher_m_hats.size else float("nan")
    teacher_m_std = float(np.std(teacher_m_hats, ddof=1)) if teacher_m_hats.size > 1 else float("nan")
    nu_theory = m_unsafe * (1.0 - teacher_m_mean) if np.isfinite(teacher_m_mean) else float("nan")

    print("\n" + "="*80)
    print("SPIN-GLASS THEORY: WEAK-FIELD REGIME VERIFICATION")
    print("="*80)
    print(f"\nParameters: N={N}, beta={beta}, j0={j0}, m={m_unsafe}")
    print(f"Disorder samples: {args.n_disorder}, Selections per disorder: {args.n_sel}")
    
    print("\n" + "-"*80)
    print("TEACHER PROPERTIES (No magnetic field)")
    print("-"*80)
    print(f"Teacher m_l (GEM/PD MLE): {teacher_m_mean:.6f} ± {teacher_m_std:.6f}   (n={teacher_m_hats.size})")
    print(f"Theoretical exponent nu = m(1-m_l) = {m_unsafe}*(1-{teacher_m_mean:.6f}) = {nu_theory:.6f}")

    # Figure 1: log E[1-Pi_k] vs logk
    fig1, ax1 = plt.subplots(figsize=(7.2, 5.2))
    # Figure 2: log(-log Pi_k) vs logk
    fig2, ax2 = plt.subplots(figsize=(7.2, 5.2))

    print("\n" + "-"*80)
    print("STUDENT PROPERTIES (With magnetic field h)")
    print("-"*80)
    
    from scipy.special import gamma
    # def get_C_m(m, m_l):
    #     product = 1.0
    #     for i in range(1, m + 1):
    #         numerator = gamma(1 + (i - 1) * m_l) * gamma(1 + (i* m_l))
    #         denominator = gamma(i * m_l) * gamma(1 + (i - 1) * m_l + 1)
    #         product *= numerator / denominator
    #     return product
    
    def get_C_m(m, m_l):
        product = 1.0
        for i in range(1, m + 1):
            numerator = gamma(1 + (i - 1) * m_l)
            denominator = gamma(i * m_l)
            product *= numerator / denominator
        return product
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(h_values)))
    
    for ii, h in enumerate(h_values):   
        X = np.stack(per_h_loggap_samples[h], axis=0)  # [n_samples, n_k]
        lambda_arr = np.array(per_h_lambda_samples[h])
        delta_q_arr = np.array(per_h_delta_q_samples[h])
        n_samples = X.shape[0]

        c1_mean, c1_std, c2_mean, c2_std, nu_mean, nu_std, lambda_mean, lambda_std = bootstrap_curves_and_nu(
            logk=logk,
            loggap_samples=X,
            lambda_samples=lambda_arr,
            B=args.bootstrap_B,
            seed=args.seed + int(abs(h) * 1e9) % 1_000_000
        )

        mS = np.asarray(per_h_student_m_hats[h], dtype=np.float64)
        mS_mean = float(np.mean(mS)) if mS.size else float("nan")
        mS_std = float(np.std(mS, ddof=1)) if mS.size > 1 else float("nan")
        
        delta_q_mean = float(np.mean(delta_q_arr))
        delta_q_std = float(np.std(delta_q_arr, ddof=1)) if delta_q_arr.size > 1 else 0.0

        # effective m diagnostic
        m_eff_student = nu_mean / (1.0 - mS_mean) if np.isfinite(mS_mean) and (1.0 - mS_mean) > 1e-9 else float("nan")

        print(f"\nh = {h:g}  (samples={n_samples})")
        print(f"  Delta_q = q_{{l+1}} - q_l (estimated): {delta_q_mean:.6f} ± {delta_q_std:.6f}")
        print(f"  lambda = beta*h*N*Delta_q          : {lambda_mean:.6f} ± {lambda_std:.6f}")
        print(f"  fitted nu from log(1-Pi_k)         : {nu_mean:.6f} ± {nu_std:.6f}")
        print(f"  student m_l (GEM/PD MLE)           : {mS_mean:.6f} ± {mS_std:.6f}   (n={mS.size})")
        print(f"  m_eff = nu/(1-m_l) from student    : {m_eff_student:.6f}  [theory: m={m_unsafe}]")

        # Theory prediction at h=0: log(1-Pi_k) = log C_m - nu*log(k)
        # Theory prediction for small h: log(1-Pi_k) = log C_m - nu*log(k) - lambda^2*k/2
        
        print(f"\n  Weak-field theory check:")
        for i, kk in enumerate(k_values):
            theory_h0 = -nu_theory * np.log(kk)  # log C_m absorbed in normalization
            theory_h = theory_h0 - 0.5 * lambda_mean**2 * kk
            
            ratio_to_h0 = np.exp(c1_mean[i] - theory_h0) if np.isfinite(theory_h0) else np.nan
            ratio_to_h = np.exp(c1_mean[i] - theory_h) if np.isfinite(theory_h) else np.nan
            
            print(f"    k={kk:>3d}: log(1-Pi_k) = {c1_mean[i]:+.4f} ± {c1_std[i]:.4f}")
            print(f"           theory(h=0)   = {theory_h0:+.4f},  ratio = {ratio_to_h0:.3f}")
            print(f"           theory(h={h}) = {theory_h:+.4f},  ratio = {ratio_to_h:.3f}")

        Cm = get_C_m(m_unsafe, teacher_m_mean)
        log_Cm = np.log(Cm)
        print("Value of log_Cm: ", log_Cm)
        # Plot figure 1
        label1 = f"h={h:g}, λ={lambda_mean:.3f}, ν={nu_mean:.2f}"
        ax1.errorbar(logk, c1_mean, yerr=c1_std, marker="o", linestyle="-", capsize=3, label=label1, color=colors[ii])
        slope1, intercept1 = np.polyfit(logk, c1_mean, deg=1)
        # ax1.plot(logk, intercept1 + slope1 * logk, linestyle="--", alpha=0.5)
        # fit_line = - nu_theory * logk + nu_theory * np.log(2**3) + c1_mean[3]
        fit_line = - nu_theory * logk + log_Cm - nu_theory * np.mean(lambda_arr) # theoretical fit
        
        print("Shape of lambda_arr: ", lambda_arr.shape)
        print("Lambda mean: ", np.mean(lambda_arr))
        ax1.plot(logk, fit_line, linestyle="--", alpha=0.5, label=f"theory(h={h})", color=colors[ii])

        # Plot figure 2
        label2 = f"h={h:g}, λ={lambda_mean:.3f}"
        ax2.errorbar(logk, c2_mean, yerr=c2_std, marker="o", linestyle="-", capsize=3, label=label2, color=colors[ii])
        slope2, intercept2 = np.polyfit(logk, c2_mean, deg=1)
        # ax2.plot(logk, intercept2 + slope2 * logk, linestyle="--", alpha=0.5)
        # fit_line = - nu_theory * logk + nu_theory * np.log(2**3) + c2_mean[3]
        fit_line = - nu_theory * logk + log_Cm - nu_theory * np.mean(lambda_arr) # theoretical fit
        y_val = c2_mean
        popt, pcov = curve_fit(model_log_corrected, k_values, y_val, p0=(-1, 0.0, 1.0))
        #fit_line = model_log_corrected(k_values, *popt)
        a, b, l = popt
        print("Fitting parameters: ", popt)
        ax2.plot(logk,fit_line,linestyle="--",alpha=0.5,label=rf"h={h} fitting:$\hat{{\nu}}={a:.2f}$, $\hat{{\lambda}}={l:.2f}$",color=colors[ii])

    ax1.set_xlabel("log k", fontsize=11)
    ax1.set_ylabel("log(1 - Pi_k)  =  log E[(1-A)^k]", fontsize=11)
    ax1.set_title(f"Weak-Field Theory Check: log(1-Pi_k) vs log k\n(p=2, N={N}, beta={beta}, j0={j0}, m={m_unsafe})", fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8, loc='best')

    ax2.set_xlabel("log k", fontsize=11)
    ax2.set_ylabel("log(-log(Pi_k))", fontsize=11)
    ax2.set_title(f"log(-log Pi_k) vs log k\n(p=2, N={N}, beta={beta}, j0={j0}, m={m_unsafe})", fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8, loc='best')

    out1 = args.out
    out2 = with_suffix(args.out, "_loglogPi")

    os.makedirs(os.path.dirname(out1) or ".", exist_ok=True)
    fig1.tight_layout()
    fig2.tight_layout()
    fig1.savefig(out1, dpi=200)
    fig2.savefig(out2, dpi=200)

    print("\n" + "="*80)
    print(f"Saved plot 1 (log(1-Pi_k) vs log k)     to: {out1}")
    print(f"Saved plot 2 (log(-log Pi_k) vs log k)  to: {out2}")
    print("="*80 + "\n")
    
    


if __name__ == "__main__":
    main()