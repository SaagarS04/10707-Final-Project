"""
TransFusion: Baseball Pitch Sequence Model + MCMC Game Simulator
================================================================
Architecture:
    TransFusion = Transformer encoder (context) + Diffusion decoder (continuous pitch features)
                + autoregressive classification heads (pitch type → diffusion → outcome)

    CORRECTED CAUSAL HEAD DEPENDENCY:
        The full per-pitch causal chain reflects physical causality:

            context[t]
                │
                ├──► pitch_type_head → pt_logits[t] → predicts pitch_types[t+1]
                │         │
                │    pt_emb[t+1]  ← ground-truth (train) / sampled (inference)
                │         │
                │    denoiser(x_t, t, context[t], pitch_type[t+1])
                │         │
                │    → continuous pitch features[t+1]   ← physical reality of the pitch
                │         │
                │    pitch_feat_emb[t+1] = pitch_feat_proj(pitch_feats[t+1])
                │         │
                └──► outcome_head(context[t] ⊕ pt_emb[t+1] ⊕ pitch_feat_emb[t+1])
                          → oc_logits[t] → predicts outcomes[t+1]
                                │
                    context[t+1] ← encoder_step(pitch_feats, game_state, pitch_type, outcome)

        Physical causality:
          1. Pitcher decides WHAT to throw (pitch type)
          2. The pitch has physical properties (velocity, location, break) — determined
             at release, before the batter does anything → DIFFUSION generates these
          3. The batter and umpire observe those properties → OUTCOME results

        The outcome head conditions on actual pitch physics (plate_x, plate_z, etc.),
        which are the primary determinants of balls, strikes, and contact. This is
        strictly more informative than conditioning on pitch type alone.

        At training: pitch_feats in step 2 are ground-truth Statcast values (teacher-forcing,
                     no compounding errors from denoiser sampling).
        At inference: pitch_feats = denoiser output, projected into outcome conditioning space.

    ARCHITECTURAL ADDITIONS vs. baseline TransFusion:
        • RoPE (Rotary Position Embedding) — replaces sinusoidal absolute PE.
          Applied inside each attention head to Q and K; attention scores depend
          on RELATIVE pitch distance, not absolute index. Two position streams:
          game-level (pitch 0…~300) and at-bat level (pitch 0…~12).

        • PitcherBatterCrossAttention — lightweight cross-attention adapter at
          every Transformer layer. The pitch sequence (Q) attends to a joint
          pitcher+batter conditioning vector (K, V). Re-evaluates the matchup
          interaction at each layer depth rather than as a fixed input bias.
          Overhead: T queries × 1 key-value per layer — negligible.

        • Corrected causal ordering — diffusion before outcome (see above).
          pitch_feat_proj: Linear(n_cont → embed_dim) + GELU bridges denoiser
          output into the outcome head's conditioning space.

Training:
    pt_loss:   context[t]                                    → pitch_types[t+1]
    diff_loss: context[t] + pitch_types[t+1]                 → continuous features[t+1]
    oc_loss:   context[t] + pitch_types[t+1]
                          + pitch_feats[t+1]  (ground truth) → outcomes[t+1]

Curriculum:
    Phase 1 [1, phase2_start):      VAE only — player latent spaces
    Phase 2 [phase2_start, phase3_start): encoder + pt_head + denoiser
                                     (entire outcome pathway frozen until denoiser matures)
    Phase 3 [phase3_start, end]:    all components including outcome head + pitch_feat_proj

Simulation (MCMC):
    Given N context innings of a real game, autoregressively simulate the remainder
    pitch-by-pitch. At each step:
        1. Sample pitch_type  from pitch_type_head(context[t])
        2. Sample pitch_feats from denoiser(context[t], pt_emb)   ← physics first
        3. Sample outcome     from outcome_head(context[t], pt_emb, pitch_feat_emb)
        4. Advance game state (count, outs, runners, score) deterministically
        5. context[t+1] ← encoder_step(pitch_feats, game_state, pitch_type, outcome)
        6. Repeat until 27 outs (or extras)
    Run K parallel simulations → aggregate win probabilities.
"""

import argparse
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from new_dataset_builder import (
    BaseballDatasetBuilder,
    PitchSequenceDataset,
    collate_fn,
    Encoders,
    PITCH_CONTINUOUS_COLS,
    GAME_STATE_COLS,
    PITCHER_STAT_COLS,
    BATTER_STAT_COLS,
    GAME_CTX_COLS,
)

import warnings
warnings.filterwarnings("ignore")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAVE_MPL = True
except ImportError:
    _HAVE_MPL = False


# =============================================================================
# 0.  CONFIG
# =============================================================================

@dataclass
class ModelConfig:
    # Input dims (overridden from dataset at runtime)
    pitch_feat_dim:   int = 30
    pitcher_feat_dim: int = 17
    batter_feat_dim:  int = 14
    game_feat_dim:    int = 3

    # Vocab sizes (overridden from encoders at runtime)
    num_pitch_types: int = 20
    num_outcomes:    int = 30
    num_events:      int = 25
    num_batters:     int = 5000
    num_pitchers:    int = 3500

    # Transformer
    d_model:     int   = 256
    n_heads:     int   = 8
    n_layers:    int   = 6
    d_ff:        int   = 1024
    dropout:     float = 0.2
    max_seq_len: int   = 400

    # Diffusion
    n_diffusion_steps: int   = 50
    beta_start:        float = 1e-4
    beta_end:          float = 0.02

    # Embeddings
    embed_dim: int = 128

    # VAE latent dimensions
    pitcher_latent_dim: int   = 64   # latent space for pitcher season stats
    batter_latent_dim:  int   = 64   # latent space for batter season stats

    # VAE loss weight — how much KL + recon contribute to total loss
    lambda_vae: float = 0.10

    # Loss weights
    lambda_diffusion:  float = 0.20
    lambda_pitch_type: float = 0.40
    lambda_outcome:    float = 0.40

    # Post-training calibration temperatures (fitted on val set after training).
    # 1.0 = uncalibrated.  Stored in checkpoint so simulation uses them automatically.
    # pt_temperature:  scales pitch_type_head logits before softmax
    # oc_temperature:  scales outcome_head logits before softmax
    pt_temperature: float = 1.0
    oc_temperature: float = 1.0

    # Frequency-weighted loss: rare outcomes get higher weight so the model
    # doesn't ignore tail classes.  Computed from training data at dataset
    # build time and stored here.  None = uniform (original label smoothing).
    pt_class_weights: Optional[List[float]] = None
    oc_class_weights: Optional[List[float]] = None


@dataclass
class TrainConfig:
    cache_dir:      str   = "./baseball_cache"
    checkpoint_dir: str   = "./checkpoints"
    start_dt:       str   = "2015-03-22"
    end_dt:         str   = "2026-05-01"
    val_start_dt:   str   = "2025-03-20"
    test_start_dt:  str   = "2026-03-25"
    epochs:         int   = 100
    batch_size:     int   = 16
    lr:             float = 3e-4
    weight_decay:   float = 1e-4
    grad_clip:      float = 1.0
    warmup_steps:   int   = 1000
    log_every:      int   = 50
    val_every:      int   = 1
    num_workers:    int   = 0
    seed:           int   = 42
    device:         str   = "auto"

    # Curriculum training phase boundaries
    # Phase 1 [1, phase2_start):             VAE only — learn player latent spaces first
    # Phase 2 [phase2_start, phase3_start):  encoder + pitch_type head + diffusion (VAE active)
    # Phase 3 [phase3_start, epochs]:        all components including outcome head
    phase2_start:   int   = 4   # epoch at which pitch/diffusion heads are unfrozen
    phase3_start:   int   = 13   # epoch at which outcome head is unfrozen


@dataclass
class SimConfig:
    checkpoint:       str   = "./checkpoints/best.pt"
    cache_dir:        str   = "./baseball_cache"
    start_dt:         str   = "2015-03-22"
    end_dt:           str   = "2026-05-01"
    val_start_dt:     str   = "2025-03-20"
    test_start_dt:    str   = "2026-03-25"
    context_innings:  float = 3.0
    n_simulations:    int   = 500
    split:            str   = "test"
    out_dir:          str   = "./sim_results"
    max_game_pitches: int   = 600   # raised from 400 to handle extra-inning games (20 inn × ~30 pitches)
    device:           str   = "auto"
    temperature:      float = 0.85


# =============================================================================
# 1.  DIFFUSION SCHEDULE
# =============================================================================

class CosineNoiseSchedule(nn.Module):
    def __init__(self, T: int, s: float = 0.008):
        super().__init__()
        steps = T + 1
        x = torch.linspace(0, T, steps)
        alphas_cumprod = torch.cos(((x / T) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        betas = betas.clamp(0.0001, 0.9999)

        alphas         = 1.0 - betas
        alpha_bar      = torch.cumprod(alphas, dim=0)
        alpha_bar_prev = F.pad(alpha_bar[:-1], (1, 0), value=1.0)

        self.register_buffer("betas",                    betas)
        self.register_buffer("alphas",                   alphas)
        self.register_buffer("alpha_bar",                alpha_bar)
        self.register_buffer("alpha_bar_prev",           alpha_bar_prev)
        self.register_buffer("sqrt_alpha_bar",           torch.sqrt(alpha_bar))
        self.register_buffer("sqrt_one_minus_alpha_bar", torch.sqrt(1.0 - alpha_bar))
        self.T = T

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor,
                 noise: Optional[torch.Tensor] = None):
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_ab  = self.sqrt_alpha_bar[t].view(-1, *([1] * (x0.dim() - 1)))
        sqrt_oab = self.sqrt_one_minus_alpha_bar[t].view(-1, *([1] * (x0.dim() - 1)))
        return sqrt_ab * x0 + sqrt_oab * noise, noise

    @torch.no_grad()
    def ddim_sample(self, model_fn, shape: Tuple, device: torch.device,
                    n_steps: int = 20, eta: float = 0.0) -> torch.Tensor:
        step_size = self.T // n_steps
        timesteps = list(range(0, self.T, step_size))[::-1]
        x = torch.randn(shape, device=device)
        for i, t_val in enumerate(timesteps):
            t_batch = torch.full((shape[0],), t_val, device=device, dtype=torch.long)
            eps     = model_fn(x, t_batch)
            ab      = self.alpha_bar[t_val]
            ab_p    = self.alpha_bar[timesteps[i + 1]] if i + 1 < len(timesteps) else torch.tensor(1.0)
            ab_p    = ab_p.to(device) if isinstance(ab_p, torch.Tensor) else torch.tensor(ab_p, device=device)
            x0_pred = (x - torch.sqrt(1 - ab) * eps) / torch.sqrt(ab)
            x0_pred = x0_pred.clamp(-5.0, 5.0)
            sigma   = eta * torch.sqrt((1 - ab_p) / (1 - ab) * (1 - ab / ab_p))
            dir_x   = torch.sqrt(1 - ab_p - sigma ** 2) * eps
            x       = torch.sqrt(ab_p) * x0_pred + dir_x
            if eta > 0:
                x = x + sigma * torch.randn_like(x)
        return x


# =============================================================================
# 1b.  PLAYER VAEs
# =============================================================================

class PlayerVAE(nn.Module):
    """
    Variational Autoencoder for player season statistics.

    Encodes a raw stat vector (pitcher or batter) into a learned latent space
    z ~ N(μ, σ²) via the reparameterization trick. The decoder reconstructs
    the original stats from z, providing a reconstruction loss signal.

    During training: encode stats → sample z → decode → compute KL + recon.
    During inference: encode stats → use μ directly (deterministic, no noise).

    The latent vector z replaces the raw stat vector in the ContextEncoder's
    global conditioning, giving the Transformer a smoother, more generalizable
    representation of each player. Players with limited data map close to the
    prior N(0, I), providing graceful degradation rather than noisy raw stats.
    """

    def __init__(self, input_dim: int, latent_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.latent_dim = latent_dim

        # Encoder: stats → (μ, log σ²)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.fc_mu     = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder: z → reconstructed stats
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (μ, log σ²) for the input stat vector."""
        h      = self.encoder(x)
        mu     = self.fc_mu(h)
        logvar = self.fc_logvar(h).clamp(-4.0, 4.0)  # stability clamp
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """z = μ + ε·σ  (reparameterization trick; deterministic at inference)."""
        if self.training:
            std = torch.exp(0.5 * logvar)
            return mu + torch.randn_like(std) * std
        return mu

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (z, μ, log σ²)."""
        mu, logvar = self.encode(x)
        z          = self.reparameterize(mu, logvar)
        return z, mu, logvar

    @staticmethod
    def kl_loss(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """KL( N(μ,σ²) ‖ N(0,I) ), averaged over the batch."""
        return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    def vae_loss(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full VAE loss = recon MSE + KL.  Returns (z, recon_loss, kl_loss)."""
        z, mu, logvar = self.forward(x)
        recon_loss    = F.mse_loss(self.decode(z), x)
        kl            = self.kl_loss(mu, logvar)
        return z, recon_loss, kl


# =============================================================================
# 2.  MODEL COMPONENTS
# =============================================================================

class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE) — Su et al. 2021.

    Unlike absolute sinusoidal PE which is added to the input, RoPE is applied
    directly inside each attention head to the Q and K vectors. This means
    attention scores only depend on the *relative* distance between positions,
    not their absolute indices — a significant benefit for pitch sequence modeling
    where what matters is "3 pitches ago in this at-bat" not "pitch 47 overall."

    We maintain two position streams simultaneously:
      • game_pos  — absolute pitch index within the game  (0 … ~300)
      • ab_pos    — pitch index within the current at-bat (0 … ~12)

    Both streams use the same rotation angles but are summed, so the model sees
    both scales of relative position. The rotation is applied as complex
    multiplication in the (cos, sin) basis.

    Usage: call apply_rope(q, k, seq_len) inside your attention forward pass.
    """

    def __init__(self, head_dim: int, max_len: int = 512, base: float = 10000.0):
        super().__init__()
        assert head_dim % 2 == 0, "head_dim must be even for RoPE"
        half = head_dim // 2
        theta = 1.0 / (base ** (torch.arange(0, half, dtype=torch.float) / half))
        self.register_buffer("theta", theta)          # [half]
        self.max_len  = max_len
        self.head_dim = head_dim

        # Pre-cache cos/sin for positions 0..max_len
        self._build_cache(max_len)

    def _build_cache(self, max_len: int):
        pos = torch.arange(max_len, dtype=torch.float)          # [L]
        freqs = torch.outer(pos, self.theta)                    # [L, half]
        cos = torch.cos(freqs)                                  # [L, half]
        sin = torch.sin(freqs)                                  # [L, half]
        # Store as [1, L, 1, half] so it broadcasts over (B, L, n_heads, half)
        self.register_buffer("cos_cache", cos.unsqueeze(0).unsqueeze(2))
        self.register_buffer("sin_cache", sin.unsqueeze(0).unsqueeze(2))

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        """Rotate the second half of each head's features by -90°."""
        half = x.shape[-1] // 2
        x1, x2 = x[..., :half], x[..., half:]
        return torch.cat([-x2, x1], dim=-1)

    def apply(self, q: torch.Tensor, k: torch.Tensor,
              seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply RoPE rotation to query and key tensors.
        q, k: [B, n_heads, T, head_dim]
        Returns rotated (q, k) of the same shape.
        """
        if seq_len > self.max_len:
            self._build_cache(seq_len)

        cos = self.cos_cache[:, :seq_len]   # [1, T, 1, half]
        sin = self.sin_cache[:, :seq_len]

        # RoPE: x_rot = x * cos + rotate_half(x) * sin
        # Transpose to [B, T, n_heads, head_dim] for broadcasting, then back
        def _rot(x):
            x = x.transpose(1, 2)           # [B, T, n_heads, head_dim]
            half = self.head_dim // 2
            x_full_cos = torch.cat([cos, cos], dim=-1)  # [1, T, 1, head_dim]
            x_full_sin = torch.cat([sin, sin], dim=-1)
            out = x * x_full_cos + self._rotate_half(x) * x_full_sin
            return out.transpose(1, 2)      # [B, n_heads, T, head_dim]

        return _rot(q), _rot(k)


class PitcherBatterCrossAttention(nn.Module):
    """
    Per-layer cross-attention adapter that lets the pitch sequence attend to
    a joint pitcher-batter conditioning vector.

    Motivation: your current design computes pitcher_z and batter_z once and
    adds them as a fixed bias before layer 0. This means every layer sees the
    same static player representation regardless of what the sequence has learned
    so far. Frontier models (AlphaFold, some sports models) show that re-evaluating
    entity relationships at each layer significantly improves predictive accuracy
    for interactions between two parties — here, pitcher vs. batter matchup.

    Architecture:
        kv = LayerNorm(concat(pitcher_latent, batter_latent) → proj → [B, 1, d_model])
        q  = LayerNorm(context[t])          → [B, T, d_model]
        out = MultiheadAttention(Q=q, K=kv, V=kv)  (T queries, 1 key/value)
        context += dropout(out)  (residual)

    The KV has sequence length 1 (the matchup is a single vector), so this adds
    only T×d_model query projections per step — very cheap. At inference it is
    called once per encoder forward pass, not per DDIM step.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float,
                 pitcher_latent_dim: int, batter_latent_dim: int):
        super().__init__()
        joint_dim = pitcher_latent_dim + batter_latent_dim
        self.kv_proj = nn.Linear(joint_dim, d_model)
        self.norm_q  = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.attn    = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads,
            dropout=dropout, batch_first=True,
        )
        self.drop    = nn.Dropout(dropout)

    def forward(
        self,
        context:        torch.Tensor,   # [B, T, d_model]
        pitcher_latent: torch.Tensor,   # [B, pitcher_latent_dim]
        batter_latent:  torch.Tensor,   # [B, batter_latent_dim]
    ) -> torch.Tensor:                  # [B, T, d_model]
        joint   = torch.cat([pitcher_latent, batter_latent], dim=-1)  # [B, joint_dim]
        kv      = self.norm_kv(self.kv_proj(joint)).unsqueeze(1)      # [B, 1, d_model]
        q       = self.norm_q(context)                                 # [B, T, d_model]
        out, _  = self.attn(q, kv, kv)                                # [B, T, d_model]
        return context + self.drop(out)


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half  = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device) / (half - 1)
        ).float()
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        return torch.cat([args.sin(), args.cos()], dim=-1)


class RoPETransformerLayer(nn.Module):
    """
    A single causal Transformer layer with:
      1. RoPE applied to Q and K inside self-attention (relative position bias).
      2. Pitcher-batter cross-attention adapter after self-attention (entity interaction).
      3. Pre-norm architecture matching the original TransformerEncoderLayer(norm_first=True).

    The cross-attention adapter lets the pitch sequence re-evaluate its relationship
    to the specific pitcher-batter matchup at *every* layer, not just at the input.
    This is the key architectural change from frontier work on entity-pair modeling.
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float,
                 rope: "RotaryPositionalEmbedding",
                 pitcher_latent_dim: int, batter_latent_dim: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads  = n_heads
        self.head_dim = d_model // n_heads
        self.scale    = self.head_dim ** -0.5
        self.rope     = rope

        # Self-attention (manual so we can inject RoPE into Q/K)
        self.norm1   = nn.LayerNorm(d_model)
        self.q_proj  = nn.Linear(d_model, d_model, bias=False)
        self.k_proj  = nn.Linear(d_model, d_model, bias=False)
        self.v_proj  = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)
        self.drop_attn = nn.Dropout(dropout)

        # Pitcher-batter cross-attention
        self.cross_attn = PitcherBatterCrossAttention(
            d_model, n_heads, dropout, pitcher_latent_dim, batter_latent_dim
        )

        # Feed-forward
        self.norm2 = nn.LayerNorm(d_model)
        self.ff    = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward_incremental(
        self,
        x:              torch.Tensor,   # [1, 1, d_model]  — single new step
        pitcher_latent: torch.Tensor,   # [1, pitcher_latent_dim]
        batter_latent:  torch.Tensor,   # [1, batter_latent_dim]
        kv_cache:       "KVCache",
        layer_idx:      int,
    ) -> torch.Tensor:                  # [1, 1, d_model]
        """
        Single-step forward with KV-caching.  Used during simulation to avoid
        re-computing attention over all prior pitches at each new step.
        The query is the new pitch; keys/values come from the full cached history.
        """
        B, T_new, D = x.shape   # B=1, T_new=1
        H, Hd = self.n_heads, self.head_dim
        T_all = kv_cache.length + 1   # including current position

        r = self.norm1(x)
        q = self.q_proj(r).reshape(B, T_new, H, Hd).transpose(1, 2)  # [1,H,1,Hd]
        k = self.k_proj(r).reshape(B, T_new, H, Hd).transpose(1, 2)
        v = self.v_proj(r).reshape(B, T_new, H, Hd).transpose(1, 2)

        # Apply RoPE at position T_all-1 (the new position)
        # We create a position-specific rotation rather than a full sequence rotation
        seq_len_for_rope = T_all
        if seq_len_for_rope > self.rope.max_len:
            self.rope._build_cache(seq_len_for_rope)
        cos_q = self.rope.cos_cache[:, T_all-1:T_all]   # [1,1,1,half]
        sin_q = self.rope.sin_cache[:, T_all-1:T_all]

        def _rot_single(x_in):
            x_in = x_in.transpose(1, 2)   # [1,1,H,Hd]
            half  = Hd // 2
            c = torch.cat([cos_q, cos_q], dim=-1)
            s = torch.cat([sin_q, sin_q], dim=-1)
            out = x_in * c + self.rope._rotate_half(x_in) * s
            return out.transpose(1, 2)

        q = _rot_single(q)

        # For cached K/V we need to apply RoPE at their original positions
        # We store the raw (un-rotated) K/V and rotate on retrieval
        k_full, v_full = kv_cache.update(layer_idx, k, v)
        # Apply RoPE to full K sequence [1,H,T_all,Hd]
        T_k = k_full.shape[2]
        if T_k > self.rope.max_len:
            self.rope._build_cache(T_k)
        cos_k = self.rope.cos_cache[:, :T_k]   # [1,T_k,1,half]
        sin_k = self.rope.sin_cache[:, :T_k]

        k_t = k_full.transpose(1, 2)   # [1,T_k,H,Hd]
        ck  = torch.cat([cos_k, cos_k], dim=-1)
        sk  = torch.cat([sin_k, sin_k], dim=-1)
        k_rot = (k_t * ck + self.rope._rotate_half(k_t) * sk).transpose(1, 2)

        scores = (q @ k_rot.transpose(-2, -1)) * self.scale   # [1,H,1,T_k]
        weights = self.drop_attn(scores.softmax(dim=-1))
        attn_out = (weights @ v_full).transpose(1, 2).reshape(B, T_new, D)

        x = x + self.out_proj(attn_out)
        x = self.cross_attn(x, pitcher_latent, batter_latent)
        x = x + self.ff(self.norm2(x))
        return x

    def forward(
        self,
        x:              torch.Tensor,   # [B, T, d_model]
        causal_mask:    torch.Tensor,   # [T, T]  additive mask (-inf / 0)
        key_pad:        torch.Tensor,   # [B, T]  True = padding
        pitcher_latent: torch.Tensor,   # [B, pitcher_latent_dim]
        batter_latent:  torch.Tensor,   # [B, batter_latent_dim]
    ) -> torch.Tensor:
        B, T, D = x.shape
        H, Hd   = self.n_heads, self.head_dim

        # ── Pre-norm self-attention with RoPE ────────────────────────────
        r = self.norm1(x)
        q = self.q_proj(r).reshape(B, T, H, Hd).transpose(1, 2)   # [B,H,T,Hd]
        k = self.k_proj(r).reshape(B, T, H, Hd).transpose(1, 2)
        v = self.v_proj(r).reshape(B, T, H, Hd).transpose(1, 2)

        q, k = self.rope.apply(q, k, T)   # inject relative positions

        # Additive causal mask + key-padding mask
        attn_bias = causal_mask.unsqueeze(0).unsqueeze(0)          # [1,1,T,T]
        if key_pad is not None:
            # [B,T] → [B,1,1,T] → broadcast over heads and queries
            pad_bias = key_pad.float().masked_fill(key_pad, float("-inf"))
            pad_bias = pad_bias.unsqueeze(1).unsqueeze(2)
            attn_bias = attn_bias + pad_bias

        scores = (q @ k.transpose(-2, -1)) * self.scale + attn_bias
        weights = self.drop_attn(scores.softmax(dim=-1))
        attn_out = (weights @ v).transpose(1, 2).reshape(B, T, D)  # [B,T,D]
        x = x + self.out_proj(attn_out)

        # ── Pitcher-batter cross-attention (per-layer matchup interaction) ─
        x = self.cross_attn(x, pitcher_latent, batter_latent)

        # ── Pre-norm feed-forward ─────────────────────────────────────────
        x = x + self.ff(self.norm2(x))
        return x


class KVCache:
    """
    Key-value cache for incremental causal attention during simulation.

    During training and context encoding the full sequence is processed in one
    pass.  During simulation we extend the sequence one pitch at a time.
    Without caching, each new pitch would require re-attending to all prior
    positions — O(T²) cost.  With caching we store the K and V projections for
    all prior positions and only compute attention for the new query — O(T) cost.

    Usage:
        cache = KVCache(n_layers, n_heads, head_dim, max_len, device)
        for each new pitch t:
            new_ctx = encoder.forward_incremental(x_t, pitcher_latent, batter_latent, cache)
    """

    def __init__(self, n_layers: int, n_heads: int, head_dim: int,
                 max_len: int, device: torch.device):
        self.n_layers = n_layers
        self.n_heads  = n_heads
        self.head_dim = head_dim
        self.max_len  = max_len
        self.device   = device
        self.length   = 0   # number of cached positions

        # k_cache[l] and v_cache[l]: [1, n_heads, max_len, head_dim]
        self.k_cache = [
            torch.zeros(1, n_heads, max_len, head_dim, device=device)
            for _ in range(n_layers)
        ]
        self.v_cache = [
            torch.zeros(1, n_heads, max_len, head_dim, device=device)
            for _ in range(n_layers)
        ]

    def update(self, layer: int,
               k_new: torch.Tensor,    # [1, n_heads, 1, head_dim]
               v_new: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Append new K/V to cache and return full [1, n_heads, T, head_dim]."""
        t = self.length
        self.k_cache[layer][:, :, t:t+1, :] = k_new
        self.v_cache[layer][:, :, t:t+1, :] = v_new
        return self.k_cache[layer][:, :, :t+1, :], self.v_cache[layer][:, :, :t+1, :]

    def advance(self):
        """Call after all layers have processed one new position."""
        self.length += 1


class ContextEncoder(nn.Module):
    """
    Causal Transformer encoder over pitch sequences.

    CORRECTED CAUSAL ORDERING vs. original design:
    ------------------------------------------------
    Outcome[t] is NO LONGER part of the per-step encoder input. Previously the
    encoder saw pitch_type[t] AND outcome[t] at each step, then the diffusion
    conditioned on outcome. This was causally inverted: the physical features of
    a pitch (velocity, location, break) exist before the batter reacts — the
    outcome is a consequence of those features, not a cause.

    New chain:
        context[t]
            ├──► pt_head  → pitch_type[t+1]
            │         │ pt_emb[t+1]
            │    denoiser(ctx[t], pt_emb[t+1]) → pitch_feats[t+1]   ← physical cause
            │         │ pitch_feat_emb[t+1]
            └──► oc_head(ctx[t], pt_emb[t+1], pitch_feat_emb[t+1]) → outcome[t+1]

    The encoder still sees outcome[t] (the *previous* pitch's result) via the
    pitch_seq features, which include prev_delta_run_exp etc.

    ARCHITECTURAL ADDITIONS vs. original:
    --------------------------------------
    1. RoPE — replaces sinusoidal absolute PE with relative position embedding
       applied inside each attention head. The model learns "3 pitches ago"
       rather than "pitch 47 overall," which generalises better across games
       of different lengths and across at-bat boundaries.

    2. PitcherBatterCrossAttention — added at each Transformer layer so the
       sequence can re-evaluate the pitcher-batter matchup at every depth of
       processing, not just as a fixed input bias.

    Per-step input:  pitch_seq[t] + batter_ctx[t] + batter_id_emb + ptype_emb[t]
                     (outcome removed — now downstream of diffusion)
    Global bias:     pitcher_latent + pitcher_id_emb + game_ctx + batting_order_mean
                     + batter_latent (added once at input; cross-attn adds per layer)
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        # outcome_emb is still needed — encoder reads outcome[t] from the sequence
        # (it's embedded in pitch_seq via prev_delta_run_exp etc.), and we keep
        # the embedding table for use in _incremental_encode_step during simulation.
        # But outcome is NO LONGER concatenated into per-step input to the encoder.
        per_step_in = (
            cfg.pitch_feat_dim
            + cfg.batter_feat_dim
            + cfg.embed_dim   # batter_id
            + cfg.embed_dim   # pitch_type[t]
            # outcome removed — it is now DOWNSTREAM of diffusion
        )
        self.step_proj = nn.Linear(per_step_in, cfg.d_model)

        global_in = (
            cfg.pitcher_latent_dim
            + cfg.embed_dim   # pitcher_id
            + cfg.game_feat_dim
            + cfg.embed_dim   # batting_order mean
            + cfg.batter_latent_dim
        )
        self.global_proj = nn.Linear(global_in, cfg.d_model)

        self.batter_emb  = nn.Embedding(cfg.num_batters,     cfg.embed_dim, padding_idx=0)
        self.pitcher_emb = nn.Embedding(cfg.num_pitchers,    cfg.embed_dim, padding_idx=0)
        self.ptype_emb   = nn.Embedding(cfg.num_pitch_types, cfg.embed_dim, padding_idx=0)
        self.outcome_emb = nn.Embedding(cfg.num_outcomes,    cfg.embed_dim, padding_idx=0)
        self.order_emb   = nn.Embedding(cfg.num_batters,     cfg.embed_dim, padding_idx=0)

        # Shared RoPE instance — all layers use the same rotation frequencies
        self.rope = RotaryPositionalEmbedding(
            head_dim = cfg.d_model // cfg.n_heads,
            max_len  = cfg.max_seq_len,
        )

        self.layers = nn.ModuleList([
            RoPETransformerLayer(
                d_model            = cfg.d_model,
                n_heads            = cfg.n_heads,
                d_ff               = cfg.d_ff,
                dropout            = cfg.dropout,
                rope               = self.rope,
                pitcher_latent_dim = cfg.pitcher_latent_dim,
                batter_latent_dim  = cfg.batter_latent_dim,
            )
            for _ in range(cfg.n_layers)
        ])
        self.out_norm = nn.LayerNorm(cfg.d_model)

    def forward(
        self,
        pitch_seq:      torch.Tensor,  # [B, T, F_pitch]
        pitch_types:    torch.Tensor,  # [B, T]
        batter_ctx:     torch.Tensor,  # [B, T, F_batter]
        batter_ids:     torch.Tensor,  # [B, T]
        pitcher_latent: torch.Tensor,  # [B, pitcher_latent_dim]
        pitcher_id:     torch.Tensor,  # [B]
        batting_order:  torch.Tensor,  # [B, 9]
        game_ctx:       torch.Tensor,  # [B, F_game]
        batter_latent:  torch.Tensor,  # [B, batter_latent_dim]
        mask:           torch.Tensor,  # [B, T] True=valid
    ) -> torch.Tensor:                 # [B, T, d_model]

        B, T, _ = pitch_seq.shape

        b_emb  = self.batter_emb(batter_ids)
        pt_emb = self.ptype_emb(pitch_types)
        # outcome embedding removed from per-step input

        step_in = torch.cat([pitch_seq, batter_ctx, b_emb, pt_emb], dim=-1)
        x = self.step_proj(step_in)

        p_emb     = self.pitcher_emb(pitcher_id)
        ord_emb   = self.order_emb(batting_order).mean(dim=1)
        global_in = torch.cat([pitcher_latent, p_emb, game_ctx, ord_emb, batter_latent], dim=-1)
        g_vec     = self.global_proj(global_in).unsqueeze(1)
        x = x + g_vec   # global bias at input; cross-attn refines per layer

        # Causal mask (no sinusoidal PE — RoPE is applied inside each layer)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(T, device=x.device)
        key_pad     = ~mask

        for layer in self.layers:
            x = layer(x, causal_mask, key_pad, pitcher_latent, batter_latent)

        return self.out_norm(x)

    def forward_incremental(
        self,
        x_new:          torch.Tensor,   # [1, 1, d_model]  — projected new step input
        pitcher_latent: torch.Tensor,   # [1, pitcher_latent_dim]
        batter_latent:  torch.Tensor,   # [1, batter_latent_dim]
        kv_cache:       KVCache,
    ) -> torch.Tensor:                  # [1, 1, d_model]
        """
        Incremental single-step encode with KV-cache.
        x_new is already projected through step_proj and has global bias added.
        Returns the output context vector for the new position.
        """
        x = x_new
        for i, layer in enumerate(self.layers):
            x = layer.forward_incremental(x, pitcher_latent, batter_latent, kv_cache, i)
        kv_cache.advance()
        return self.out_norm(x)


class ClassificationHeads(nn.Module):
    """
    Autoregressive classification heads implementing the CORRECTED causal chain:

        context[t]
            │
            ├──► pitch_type_head → pt_logits[t]   → predicts pitch_type[t+1]
            │         │ pt_emb[t+1]  (teacher-forced / sampled)
            │         │
            │    denoiser(context[t], pt_emb[t+1]) → pitch_feats[t+1]
            │         │ pitch_feat_emb[t+1]  = pitch_feat_proj(pitch_feats)
            │         │
            └──► outcome_head(context[t] ⊕ pt_emb[t+1] ⊕ pitch_feat_emb[t+1])
                          → oc_logits[t]  → predicts outcome[t+1]

    Physical causality: the pitcher releases the ball (pitch_type → physical features),
    then the batter and umpire react (outcome). Outcome is DOWNSTREAM of physics,
    not upstream. The outcome head now conditions on the actual generated (or
    teacher-forced) continuous pitch features, giving it the location and movement
    information that most directly determines whether a pitch is a ball or strike.

    Training:
        pitch_feats used in forward_oc = ground truth continuous features (teacher-forcing)
        pitch_feats projected via pitch_feat_proj [n_cont → embed_dim]

    Inference:
        pitch_feats = denoiser output → projected → concatenated for outcome head
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        # Pitch type head: context[t] → pt_logits
        self.pitch_type_head = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model // 2),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model // 2, cfg.num_pitch_types),
        )

        # Dedicated embedding for pitch_type → outcome conditioning.
        # Separate from encoder's ptype_emb to avoid gradient interference.
        self.pt_to_oc_emb = nn.Embedding(cfg.num_pitch_types, cfg.embed_dim, padding_idx=0)

        # NEW: project continuous pitch features into outcome conditioning space.
        # This is the key change: outcome head now sees actual pitch physics.
        n_cont = len(PITCH_CONTINUOUS_COLS)
        self.pitch_feat_proj = nn.Sequential(
            nn.Linear(n_cont, cfg.embed_dim),
            nn.GELU(),
        )

        # Outcome head: context[t] ⊕ pt_emb[t+1] ⊕ pitch_feat_emb[t+1] → oc_logits
        # Input is d_model + embed_dim (pt) + embed_dim (pitch feat)
        self.outcome_head = nn.Sequential(
            nn.Linear(cfg.d_model + cfg.embed_dim + cfg.embed_dim, cfg.d_model // 2),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model // 2, cfg.num_outcomes),
        )

    def forward_pt(self, context: torch.Tensor) -> torch.Tensor:
        """context [B, T, d_model] → pt_logits [B, T, num_pitch_types]"""
        return self.pitch_type_head(context)

    def forward_oc(
        self,
        context:     torch.Tensor,   # [B, T, d_model]
        pitch_types: torch.Tensor,   # [B, T]  — teacher-forced or sampled
        pitch_feats: torch.Tensor,   # [B, T, n_cont]  — ground-truth (train) or denoised (infer)
    ) -> torch.Tensor:               # [B, T, num_outcomes]
        """
        Predict outcome conditioned on context, pitch type, AND physical pitch features.
        pitch_feats are the continuous Statcast values (location, velocity, break, etc.)
        which are the primary determinants of whether a pitch is a ball or strike.
        """
        pt_emb      = self.pt_to_oc_emb(pitch_types)        # [B, T, embed_dim]
        feat_emb    = self.pitch_feat_proj(pitch_feats)      # [B, T, embed_dim]
        oc_input    = torch.cat([context, pt_emb, feat_emb], dim=-1)
        return self.outcome_head(oc_input)


class DiffusionDenoiser(nn.Module):
    """
    Denoiser conditioned on context and pitch type ONLY.

    CORRECTED vs. original: outcome conditioning has been REMOVED.

    Physical causality:
        pitch_type determines the delivery → physical features are generated
        → batter/umpire observe those features → outcome results

    The denoiser generates the physical reality of the pitch (velocity, location,
    break). It should not know the outcome — that would be conditioning the
    cause on its own effect. Removing outcome from the denoiser forces it to
    learn genuine physical distributions for each pitch type and game context,
    rather than back-solving from the outcome label.

    Net input: [x_noisy + time_emb, context, pt_emb]  →  3×d_model
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        t_dim = cfg.d_model

        self.time_emb = nn.Sequential(
            SinusoidalTimeEmbedding(t_dim),
            nn.Linear(t_dim, t_dim),
            nn.SiLU(),
            nn.Linear(t_dim, t_dim),
        )

        self.ptype_proj = nn.Linear(cfg.embed_dim, t_dim)
        self.ptype_emb  = nn.Embedding(cfg.num_pitch_types, cfg.embed_dim, padding_idx=0)
        # outcome_emb REMOVED — diffusion no longer conditions on outcome

        self.x_proj = nn.Linear(len(PITCH_CONTINUOUS_COLS), t_dim)

        # net input: [x+time, context, pt_emb] → 3 × t_dim
        self.net = nn.Sequential(
            nn.Linear(t_dim * 3, t_dim * 2),
            nn.SiLU(),
            nn.LayerNorm(t_dim * 2),
            nn.Linear(t_dim * 2, t_dim),
            nn.SiLU(),
            nn.LayerNorm(t_dim),
            nn.Linear(t_dim, len(PITCH_CONTINUOUS_COLS)),
        )

    def forward(self, x_t: torch.Tensor, t: torch.Tensor,
                context: torch.Tensor, pitch_type: torch.Tensor) -> torch.Tensor:
        t_emb  = self.time_emb(t)
        pt_emb = self.ptype_proj(self.ptype_emb(pitch_type))
        x_emb  = self.x_proj(x_t)
        h = torch.cat([x_emb + t_emb, context, pt_emb], dim=-1)
        return self.net(h)



class InPlayEventMLP(nn.Module):
    """
    Conditional in-play event classifier.

    Replaces the flat prior `in_play_probs.json` with a small MLP that
    conditions on batter features and game context to predict the distribution
    over in-play outcomes (single, double, triple, HR, field_out, …).

    Input: batter_ctx [batter_feat_dim] ⊕ game_ctx [game_feat_dim]
           ⊕ pitch_feat [n_cont_selected]
    Output: softmax over IN_PLAY_EVENT_CLASSES

    Trained as a lightweight auxiliary task using the same Statcast data,
    loaded from a separate checkpoint at simulation time.  The model is
    intentionally small (2 hidden layers) to avoid overfitting — batted ball
    outcomes have substantial irreducible variance.

    IN_PLAY_EVENT_CLASSES (ordered, must match _IN_PLAY_EVENTS_ARR):
        single, double, triple, home_run,
        field_out, force_out, double_play, grounded_into_double_play,
        field_error, sac_fly
    """

    N_EVENTS = 10   # must match len(_IN_PLAY_EVENTS_ARR)

    def __init__(self, batter_feat_dim: int, game_feat_dim: int,
                 n_pitch_feats: int = 4, hidden_dim: int = 64):
        super().__init__()
        # Use a small subset of continuous pitch features:
        #   plate_x (index 9), plate_z (10), release_speed (0), arm_angle (13)
        # These are the most predictive of batted ball type.
        self.pitch_feat_indices = [0, 9, 10, 13]   # speed, plate_x/z, arm_angle
        in_dim = batter_feat_dim + game_feat_dim + len(self.pitch_feat_indices)

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.N_EVENTS),
        )
        self.batter_feat_dim = batter_feat_dim
        self.game_feat_dim   = game_feat_dim

    def forward(self, batter_ctx: torch.Tensor,
                game_ctx: torch.Tensor,
                pitch_feats: torch.Tensor) -> torch.Tensor:
        """
        batter_ctx:  [B, batter_feat_dim]
        game_ctx:    [B, game_feat_dim]
        pitch_feats: [B, n_cont]  — full continuous pitch feature vector
        returns:     [B, N_EVENTS] — unnormalised logits
        """
        pf_sel = pitch_feats[:, self.pitch_feat_indices]
        x = torch.cat([batter_ctx, game_ctx, pf_sel], dim=-1)
        return self.net(x)

    @torch.no_grad()
    def sample(self, batter_ctx: torch.Tensor,
               game_ctx: torch.Tensor,
               pitch_feats: torch.Tensor,
               temperature: float = 1.0) -> np.ndarray:
        """Sample in-play event indices for a batch of situations."""
        logits = self.forward(batter_ctx, game_ctx, pitch_feats)
        probs  = torch.softmax(logits / temperature, dim=-1)
        idxs   = torch.multinomial(probs, 1).squeeze(-1).cpu().numpy()
        return idxs


# Global singleton — loaded once at simulation start
_IN_PLAY_MLP: Optional["InPlayEventMLP"] = None

def _load_in_play_mlp(cache_dir: str, device: torch.device,
                       batter_feat_dim: int, game_feat_dim: int) -> Optional["InPlayEventMLP"]:
    """Load InPlayEventMLP from cache if available; else return None (use prior)."""
    global _IN_PLAY_MLP
    path = Path(cache_dir) / "in_play_mlp.pt"
    if not path.exists():
        print(f"[in-play-mlp] {path} not found — using flat prior fallback")
        return None
    ckpt = torch.load(path, map_location=device)
    mlp  = InPlayEventMLP(
        batter_feat_dim=batter_feat_dim,
        game_feat_dim=game_feat_dim,
    ).to(device)
    mlp.load_state_dict(ckpt["model_state"])
    mlp.eval()
    _IN_PLAY_MLP = mlp
    print(f"[in-play-mlp] Loaded from {path}")
    return mlp


def train_in_play_mlp(
    cache_dir:      str  = "./baseball_cache",
    train_end_dt:   str  = "2026-02-01",
    epochs:         int  = 20,
    batch_size:     int  = 4096,
    lr:             float = 1e-3,
    hidden_dim:     int  = 64,
    device_str:     str  = "auto",
    batter_feat_dim: int = 24,
    game_feat_dim:  int  = 4,
) -> "InPlayEventMLP":
    """
    Train the InPlayEventMLP from Statcast data and save to
    baseball_cache/in_play_mlp.pt.

    Can be called standalone:
        python new_transfusion.py train_in_play_mlp

    Or imported and called from a notebook after the main dataset is built.
    The training data is derived from the same statcast.parquet used for the
    main model, so no extra data download is needed.
    """
    import pandas as _pd
    device = _resolve_device(device_str)

    _IN_PLAY_EVENTS = [
        "single","double","triple","home_run",
        "field_out","force_out","double_play","grounded_into_double_play",
        "field_error","sac_fly",
    ]
    event2idx = {e: i for i, e in enumerate(_IN_PLAY_EVENTS)}
    N = len(_IN_PLAY_EVENTS)

    # ── Load Statcast and filter to in-play terminal pitches ──────────────
    from new_dataset_builder import StatScaler, BATTER_STAT_COLS, GAME_CTX_COLS
    path = Path(cache_dir) / "statcast.parquet"
    print(f"[in-play-mlp] Loading {path} ...")
    df = _pd.read_parquet(path)
    df["game_date"] = _pd.to_datetime(df["game_date"], errors="coerce")
    df = df[df["game_date"] < _pd.Timestamp(train_end_dt)].copy()
    df = df[df["events"].isin(_IN_PLAY_EVENTS)].dropna(subset=["events"]).copy()
    print(f"[in-play-mlp] In-play terminal pitches: {len(df):,}")

    # ── Load batter stats and game context scalers ────────────────────────
    bscaler_path = Path(cache_dir) / "batter_scaler.pkl"
    bstats_path  = Path(cache_dir) / "batter_stats_statcast.parquet"
    if not bscaler_path.exists() or not bstats_path.exists():
        raise FileNotFoundError("Run BaseballDatasetBuilder.build() first to create scaler files.")
    import pickle as _pk
    with open(bscaler_path, "rb") as f:
        batter_scaler = _pk.load(f)
    batter_stats_df = _pd.read_parquet(bstats_path)

    # Build batter LUT
    batter_lut = {}
    for _, row in batter_stats_df.iterrows():
        batter_lut[(int(row["batter"]), int(row["game_year"]))] = row.to_dict()

    # ── Build feature matrix ──────────────────────────────────────────────
    PITCH_COLS_USED = ["release_speed","plate_x","plate_z","arm_angle"]

    # batter features
    def _get_batter_ctx(row) -> np.ndarray:
        bid    = int(row["batter"])  if _pd.notna(row.get("batter"))    else 0
        season = int(row["game_year"]) if _pd.notna(row.get("game_year")) else 2023
        brow   = batter_lut.get((bid, season), {})
        return batter_scaler.transform_row(brow, BATTER_STAT_COLS)

    # game context: approximate with [0.5, 0.5, 0, 1.0] (neutral)
    game_ctx_neutral = np.array([0.5, 0.5, 0.0, 1.0], dtype=np.float32)

    print("[in-play-mlp] Building feature matrix ...")
    rows_b, rows_p, rows_y = [], [], []
    for _, row in df.iterrows():
        b_feat = _get_batter_ctx(row)
        p_feat = np.array([
            float(row.get(c, 0.0)) if _pd.notna(row.get(c)) else 0.0
            for c in PITCH_COLS_USED
        ], dtype=np.float32)
        y = event2idx.get(str(row["events"]), -1)
        if y < 0: continue
        rows_b.append(b_feat)
        rows_p.append(p_feat)
        rows_y.append(y)

    B_arr = np.stack(rows_b).astype(np.float32)
    P_arr = np.stack(rows_p).astype(np.float32)
    G_arr = np.tile(game_ctx_neutral, (len(rows_y), 1))
    Y_arr = np.array(rows_y, dtype=np.int64)
    print(f"[in-play-mlp] Samples: {len(Y_arr):,}")

    # ── Train ─────────────────────────────────────────────────────────────
    mlp = InPlayEventMLP(
        batter_feat_dim=batter_feat_dim,
        game_feat_dim=game_feat_dim,
        hidden_dim=hidden_dim,
    ).to(device)

    opt = torch.optim.Adam(mlp.parameters(), lr=lr)
    n   = len(Y_arr)
    mlp.train()
    for ep in range(1, epochs + 1):
        perm = np.random.permutation(n)
        total_loss, nb = 0.0, 0
        for i in range(0, n, batch_size):
            idx = perm[i:i+batch_size]
            bx  = torch.tensor(B_arr[idx], device=device)
            gx  = torch.tensor(G_arr[idx], device=device)
            px  = torch.tensor(P_arr[idx], device=device)
            y   = torch.tensor(Y_arr[idx], device=device)
            opt.zero_grad()
            # Expand pitch feats to full n_cont with zeros for unused dims
            pf_full = torch.zeros(len(idx), 21, device=device)  # 21 = n_cont
            # map PITCH_COLS_USED → indices [0,9,10,13]
            pf_full[:, 0]  = px[:, 0]   # release_speed
            pf_full[:, 9]  = px[:, 1]   # plate_x
            pf_full[:, 10] = px[:, 2]   # plate_z
            pf_full[:, 13] = px[:, 3]   # arm_angle
            loss = F.cross_entropy(mlp(bx, gx, pf_full), y)
            loss.backward()
            opt.step()
            total_loss += loss.item(); nb += 1
        if ep % 5 == 0 or ep == 1:
            print(f"  [in-play-mlp] epoch {ep:3d}  loss={total_loss/nb:.4f}")

    # ── Save ──────────────────────────────────────────────────────────────
    out = Path(cache_dir) / "in_play_mlp.pt"
    torch.save({
        "model_state": mlp.cpu().state_dict(),
        "batter_feat_dim": batter_feat_dim,
        "game_feat_dim":   game_feat_dim,
        "hidden_dim":      hidden_dim,
        "event_classes":   _IN_PLAY_EVENTS,
    }, out)
    print(f"[in-play-mlp] Saved → {out}")
    return mlp

# =============================================================================
# 3.  FULL TRANSFUSION MODEL
# =============================================================================

class TransFusion(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg      = cfg
        self.encoder  = ContextEncoder(cfg)
        self.denoiser = DiffusionDenoiser(cfg)
        self.heads    = ClassificationHeads(cfg)
        self.schedule = CosineNoiseSchedule(cfg.n_diffusion_steps)
        self.n_cont   = len(PITCH_CONTINUOUS_COLS)

        # Player VAEs — jointly trained with the main model.
        # They replace raw pitcher_ctx / batter_ctx in the global conditioning,
        # providing a smoother latent space that generalises across players and
        # degrades gracefully for players with limited per-season data.
        self.pitcher_vae = PlayerVAE(
            input_dim  = cfg.pitcher_feat_dim,
            latent_dim = cfg.pitcher_latent_dim,
            hidden_dim = max(64, cfg.pitcher_latent_dim * 4),
        )
        self.batter_vae = PlayerVAE(
            input_dim  = cfg.batter_feat_dim,
            latent_dim = cfg.batter_latent_dim,
            hidden_dim = max(64, cfg.batter_latent_dim * 4),
        )

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        pitch_seq     = batch["pitch_seq"]
        pitch_types   = batch["pitch_types"]
        outcomes      = batch["outcomes"]
        batter_ctx    = batch["batter_ctx"]
        batter_ids    = batch["batter_ids"]
        pitcher_ctx   = batch["pitcher_ctx"]
        pitcher_id    = batch["pitcher_id"]
        batting_order = batch["batting_order"]
        game_ctx      = batch["game_ctx"]
        mask          = batch["mask"]

        B, T, _ = pitch_seq.shape

        # ── 1. Encode players through VAEs ────────────────────────────────
        pitcher_z, pitcher_recon, pitcher_kl = self.pitcher_vae.vae_loss(pitcher_ctx)

        batter_flat = batter_ctx.reshape(B * T, -1)
        batter_z_flat, batter_mu_flat, batter_lv_flat = self.batter_vae(batter_flat)
        batter_kl    = PlayerVAE.kl_loss(batter_mu_flat, batter_lv_flat)
        batter_z_mean = batter_z_flat.reshape(B, T, self.cfg.batter_latent_dim).mean(dim=1)

        vae_loss = pitcher_recon + pitcher_kl + batter_kl

        # ── 2. Encode full sequence causally (outcome NOT in encoder input) ─
        # Outcome is now downstream of diffusion — see chain below.
        context = self.encoder(
            pitch_seq, pitch_types,
            batter_ctx, batter_ids,
            pitcher_z, pitcher_id,
            batting_order, game_ctx,
            batter_z_mean, mask,
        )  # [B, T, d_model]

        # ── 3. Pitch type loss: context[t] → pitch_types[t+1] ─────────────
        pt_logits = self.heads.forward_pt(context)
        # Frequency-weighted CE: rare pitch types get higher weight.
        # Falls back to uniform if pt_class_weights not yet populated.
        _pt_w = (
            torch.tensor(self.cfg.pt_class_weights, dtype=torch.float32, device=pitch_seq.device)
            if self.cfg.pt_class_weights else None
        )
        pt_loss = F.cross_entropy(
            pt_logits[:, :-1].reshape(-1, self.cfg.num_pitch_types),
            pitch_types[:, 1:].reshape(-1),
            weight=_pt_w,
            ignore_index=0,
        )

        # ── 4. Diffusion loss: context[t] + pitch_type[t+1] → pitch_feats[t+1]
        # Outcome is NOT a conditioning input to the denoiser (causal fix).
        x0     = pitch_seq[:, 1:, :self.n_cont].reshape(B * (T - 1), self.n_cont)
        t_diff = torch.randint(0, self.cfg.n_diffusion_steps, (B * (T - 1),), device=x0.device)
        x_t, noise = self.schedule.q_sample(x0, t_diff)

        ctx_for_diff = context[:, :-1].reshape(B * (T - 1), -1)
        pt_for_diff  = pitch_types[:, 1:].reshape(B * (T - 1))

        noise_pred = self.denoiser(x_t, t_diff, ctx_for_diff, pt_for_diff)

        valid     = mask[:, 1:].reshape(-1)
        diff_loss = F.mse_loss(noise_pred[valid], noise[valid])

        # ── 5. Outcome loss: context[t] + pt_emb[t+1] + pitch_feats[t+1] → outcomes[t+1]
        # pitch_feats[t+1] are the ground-truth continuous features (teacher-forcing).
        # At inference these will be replaced by the denoiser's generated output.
        pitch_feats_for_oc = pitch_seq[:, 1:, :self.n_cont]   # [B, T-1, n_cont]

        oc_logits = self.heads.forward_oc(
            context[:, :-1],        # [B, T-1, d_model]
            pitch_types[:, 1:],     # [B, T-1]
            pitch_feats_for_oc,     # [B, T-1, n_cont]  ← physical features as conditioning
        )
        # Frequency-weighted CE: rare outcomes (hit_into_play_score, hbp…)
        # get higher weight so the model doesn't collapse to ball/strike modes.
        _oc_w = (
            torch.tensor(self.cfg.oc_class_weights, dtype=torch.float32, device=pitch_seq.device)
            if self.cfg.oc_class_weights else None
        )
        oc_loss = F.cross_entropy(
            oc_logits.reshape(-1, self.cfg.num_outcomes),
            outcomes[:, 1:].reshape(-1),
            weight=_oc_w,
            ignore_index=0,
        )

        # ── 6. Weighted total loss ─────────────────────────────────────────
        loss = (
            self.cfg.lambda_diffusion  * diff_loss
            + self.cfg.lambda_pitch_type * pt_loss
            + self.cfg.lambda_outcome    * oc_loss
            + self.cfg.lambda_vae        * vae_loss
        )

        return {
            "loss":          loss,
            "diff_loss":     diff_loss.detach(),
            "pt_loss":       pt_loss.detach(),
            "oc_loss":       oc_loss.detach(),
            "vae_loss":      vae_loss.detach(),
            "pitcher_recon": pitcher_recon.detach(),
            "pitcher_kl":    pitcher_kl.detach(),
            "batter_kl":     batter_kl.detach(),
        }

    @torch.no_grad()
    def _get_player_latents(
        self,
        pitcher_ctx:  torch.Tensor,  # [B, F_pitcher]
        batter_ctx:   torch.Tensor,  # [B, T, F_batter]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Deterministic (μ, no sampling) player latents for inference."""
        pitcher_mu, _ = self.pitcher_vae.encode(pitcher_ctx)
        B, T, F = batter_ctx.shape
        batter_mu_flat, _ = self.batter_vae.encode(batter_ctx.reshape(B * T, F))
        batter_z_mean     = batter_mu_flat.reshape(B, T, self.cfg.batter_latent_dim).mean(dim=1)
        return pitcher_mu, batter_z_mean

    @torch.no_grad()
    def encode_context(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        pitcher_z, batter_z_mean = self._get_player_latents(
            batch["pitcher_ctx"], batch["batter_ctx"]
        )
        return self.encoder(
            batch["pitch_seq"], batch["pitch_types"],
            batch["batter_ctx"], batch["batter_ids"],
            pitcher_z,
            batch["pitcher_id"],
            batch["batting_order"], batch["game_ctx"],
            batter_z_mean,
            batch["mask"],
        )

    @torch.no_grad()
    def predict_next_pitch_type(self, context_vec: torch.Tensor,
                                temperature: float = 0.90) -> torch.Tensor:
        """
        Step 1 of the autoregressive chain.
        context_vec: [B, d_model] — context at current step t
        returns:     [B, num_pitch_types] — probability distribution
        """
        pt_logits = self.heads.forward_pt(context_vec)
        return F.softmax(pt_logits / temperature, dim=-1)

    @torch.no_grad()
    def sample_pitch_features(self, context_vec: torch.Tensor,
                               pitch_type: torch.Tensor,
                               ddim_steps: int = 10) -> torch.Tensor:
        """
        Step 2 of the autoregressive chain (CORRECTED ORDER: diffusion before outcome).
        context_vec: [B, d_model]
        pitch_type:  [B]
        returns:     [B, n_cont] — denoised continuous pitch features
        """
        B      = context_vec.shape[0]
        device = context_vec.device

        def _model_fn(x_t, t_batch):
            return self.denoiser(x_t, t_batch, context_vec, pitch_type)

        return self.schedule.ddim_sample(
            _model_fn, (B, self.n_cont), device, n_steps=ddim_steps
        )

    @torch.no_grad()
    def predict_next_outcome(self, context_vec: torch.Tensor,
                             pitch_type: torch.Tensor,
                             pitch_feats: torch.Tensor,
                             temperature: float = 0.85) -> torch.Tensor:
        """
        Step 3 of the autoregressive chain (CORRECTED ORDER: after diffusion).
        context_vec: [B, d_model]
        pitch_type:  [B]
        pitch_feats: [B, n_cont]  — output of sample_pitch_features
        returns:     [B, num_outcomes] — probability distribution over outcome
        """
        oc_logits = self.heads.forward_oc(
            context_vec.unsqueeze(1),   # [B, 1, d_model]
            pitch_type.unsqueeze(1),    # [B, 1]
            pitch_feats.unsqueeze(1),   # [B, 1, n_cont]
        ).squeeze(1)                    # [B, num_outcomes]
        return F.softmax(oc_logits / temperature, dim=-1)


# =============================================================================
# 4a. RE24 RUN EXPECTANCY TABLE
# =============================================================================

_RE24_TABLE_CACHE: Dict[Tuple[int, bool, bool, bool], float] = {
    (0, False, False, False): 0.461, (0, True,  False, False): 0.831,
    (0, False, True,  False): 1.068, (0, True,  True,  False): 1.373,
    (0, False, False, True):  1.426, (0, True,  False, True):  1.798,
    (0, False, True,  True):  1.920, (0, True,  True,  True):  2.282,
    (1, False, False, False): 0.243, (1, True,  False, False): 0.489,
    (1, False, True,  False): 0.644, (1, True,  True,  False): 0.908,
    (1, False, False, True):  0.865, (1, True,  False, True):  1.140,
    (1, False, True,  True):  1.352, (1, True,  True,  True):  1.520,
    (2, False, False, False): 0.095, (2, True,  False, False): 0.214,
    (2, False, True,  False): 0.305, (2, True,  True,  False): 0.343,
    (2, False, False, True):  0.413, (2, True,  False, True):  0.471,
    (2, False, True,  True):  0.570, (2, True,  True,  True):  0.736,
}
_RE24_MAX_CACHE: float = 2.282


def _load_re24_table(cache_dir: str) -> None:
    global _RE24_TABLE_CACHE, _RE24_MAX_CACHE
    import pandas as _pd
    path = Path(cache_dir) / "re24_table.parquet"
    if not path.exists():
        warnings.warn(f"[re24] re24_table.parquet not found at {path}; using fallback constants.")
        return
    df = _pd.read_parquet(str(path))
    _RE24_TABLE_CACHE = {
        (int(r.outs), bool(r.on_1b), bool(r.on_2b), bool(r.on_3b)): float(r.re24_value)
        for _, r in df.iterrows()
    }
    _RE24_MAX_CACHE = max(_RE24_TABLE_CACHE.values()) if _RE24_TABLE_CACHE else 2.292
    print(f"[re24] Loaded empirical RE24 table from {path} (max={_RE24_MAX_CACHE:.3f})")


def _phi(outs: int, on_1b: bool, on_2b: bool, on_3b: bool) -> float:
    re = _RE24_TABLE_CACHE.get((outs, on_1b, on_2b, on_3b), 0.0)
    return math.log(max(re, 1e-6)) - math.log(_RE24_MAX_CACHE)


# =============================================================================
# 4.  GAME STATE ENGINE
# =============================================================================

BALL_OUTCOMES = {"ball", "blocked_ball", "pitchout", "intent_ball"}
STRIKE_OUTCOMES = {
    "called_strike", "swinging_strike", "swinging_strike_blocked",
    "foul_tip", "missed_bunt",
}
FOUL_OUTCOMES    = {"foul", "foul_bunt"}
IN_PLAY_OUTCOMES = {"hit_into_play", "hit_into_play_score", "hit_into_play_no_out"}

EVENT_TABLE = {
    "field_out":                 (1, 0, "out"),
    "strikeout":                 (1, 0, "out"),
    "strikeout_double_play":     (2, 0, "out"),
    "double_play":               (2, 0, "out"),
    "grounded_into_double_play": (2, 0, "out"),
    "force_out":                 (1, 0, "out"),
    "field_error":               (0, 0, "single"),
    "fielders_choice":           (1, 0, "single"),
    "fielders_choice_out":       (1, 0, "out"),
    "sac_fly":                   (1, 1, "out"),
    "sac_fly_double_play":       (2, 1, "out"),
    "sac_bunt":                  (1, 0, "out"),
    "other_out":                 (1, 0, "out"),
    "single":                    (0, 0, "single"),
    "double":                    (0, 0, "double"),
    "triple":                    (0, 0, "triple"),
    "home_run":                  (0, 0, "hr"),
    "walk":                      (0, 0, "walk"),
    "intent_walk":               (0, 0, "walk"),
    "hit_by_pitch":              (0, 0, "hbp"),
    "catcher_interf":            (0, 0, "walk"),
}


class GameState:
    def __init__(self):
        self.inning      = 1
        self.is_top      = True
        self.outs        = 0
        self.on_1b       = False
        self.on_2b       = False
        self.on_3b       = False
        self.home_score  = 0
        self.away_score  = 0
        self.batting_idx = 0
        self.balls       = 0
        self.strikes     = 0

    def is_game_over(self, min_innings: int = 9, max_innings: int = 20) -> bool:
        # Hard innings cap — prevents infinite extra-inning simulations.
        # MLB record is 26 innings (1920); 20 is a safe practical ceiling.
        if self.inning > max_innings:
            return True
        # Regulation end: top of 10th+ and scores differ → away team lost bottom 9th+
        # (home team wins in bottom half via _check_walkoff; away team wins here)
        if self.inning > min_innings and self.is_top and self.home_score != self.away_score:
            return True
        # End of regulation bottom 9th with home team losing — game over
        if self.inning == min_innings and not self.is_top and self.home_score < self.away_score:
            return False   # home team still batting, not over yet
        return False

    def is_walkoff(self) -> bool:
        return (not self.is_top and self.inning >= 9 and self.home_score > self.away_score)

    def advance_runners(self, bases: int) -> int:
        runs = 0
        new_1b = new_2b = new_3b = False
        if bases == 1:
            if self.on_3b: runs += 1
            if self.on_2b: new_3b = True
            if self.on_1b: new_2b = True
            new_1b = True
        elif bases == 2:
            if self.on_3b: runs += 1
            if self.on_2b: runs += 1
            if self.on_1b: new_3b = True
            new_2b = True
        elif bases == 3:
            if self.on_3b: runs += 1
            if self.on_2b: runs += 1
            if self.on_1b: runs += 1
            new_3b = True
        elif bases == 4:
            if self.on_3b: runs += 1
            if self.on_2b: runs += 1
            if self.on_1b: runs += 1
            runs += 1
        self.on_1b, self.on_2b, self.on_3b = new_1b, new_2b, new_3b
        return runs

    def apply_walk(self) -> int:
        runs = 0
        if self.on_1b and self.on_2b and self.on_3b:
            runs = 1
        elif self.on_1b and self.on_2b:
            self.on_3b = True
        elif self.on_1b:
            self.on_2b = True
        self.on_1b = True
        return runs

    def apply_event(self, event_str: str) -> int:
        outs_added, base_runs, code = EVENT_TABLE.get(event_str, (1, 0, "out"))
        runs = base_runs
        if code == "out":
            self.outs += outs_added
            if self.outs >= 3:
                self._end_half_inning()
        elif code == "single":
            runs += self.advance_runners(1)
            self._add_score(runs)
        elif code == "double":
            runs += self.advance_runners(2)
            self._add_score(runs)
        elif code == "triple":
            runs += self.advance_runners(3)
            self._add_score(runs)
        elif code == "hr":
            runs += self.advance_runners(4)
            self._add_score(runs)
        elif code in ("walk", "hbp"):
            runs += self.apply_walk()
            self._add_score(runs)
        else:
            self.outs += outs_added
        self.balls = 0
        self.strikes = 0
        self.batting_idx = (self.batting_idx + 1) % 9
        return runs

    def apply_pitch_outcome(self, outcome_str: str) -> Optional[str]:
        if outcome_str in BALL_OUTCOMES:
            self.balls += 1
            if self.balls >= 4:
                return "walk"
        elif outcome_str in STRIKE_OUTCOMES:
            self.strikes += 1
            if self.strikes >= 3:
                return "strikeout"
        elif outcome_str in FOUL_OUTCOMES:
            if self.strikes < 2:
                self.strikes += 1
        return None

    def _add_score(self, runs: int):
        if self.is_top:
            self.away_score += runs
        else:
            self.home_score += runs
            if self.inning >= 9 and self.home_score > self.away_score:
                self._walkoff = True

    def _check_walkoff(self) -> bool:
        return getattr(self, "_walkoff", False)

    def _end_half_inning(self):
        self.outs  = 0
        self.on_1b = self.on_2b = self.on_3b = False
        self.balls = self.strikes = 0
        if self.is_top:
            self.is_top = False
        else:
            self.inning += 1
            self.is_top = True
            if self.inning > 9:
                self.on_2b = True  # automatic runner (MLB rule since 2020)

    def to_feature_vec(self, pitch_scaler) -> np.ndarray:
        raw = {
            "balls":        float(self.balls),
            "strikes":      float(self.strikes),
            "outs_when_up": float(self.outs),
            "inning":       float(self.inning),
            "home_score":   float(self.home_score),
            "away_score":   float(self.away_score),
            "on_1b":        float(self.on_1b),
            "on_2b":        float(self.on_2b),
            "on_3b":        float(self.on_3b),
            "run_diff":     float(self.home_score - self.away_score),
        }
        return pitch_scaler.transform_row(raw, GAME_STATE_COLS)


def inning_number_to_context_outs(context_innings: float) -> int:
    full_innings = int(context_innings)
    half = (context_innings - full_innings) >= 0.4
    return full_innings * 6 + (3 if half else 0)


def find_context_split(game_df, context_innings: float) -> int:
    target_outs = inning_number_to_context_outs(context_innings)
    if target_outs == 0:
        return 0
    outs_seen = 0
    for i, row in game_df.iterrows():
        event = row.get("events", None)
        if event is not None and str(event) != "nan":
            outs_change, _, _ = EVENT_TABLE.get(str(event), (0, 0, "none"))
            outs_seen += outs_change
            if outs_seen >= target_outs:
                return i + 1
    return len(game_df)


# =============================================================================
# 5.  TRAINING
# =============================================================================

def get_lr_scheduler(optimizer, warmup_steps: int, total_steps: int):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.01, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _model_tag(cfg_model: ModelConfig) -> str:
    return (
        f"d{cfg_model.d_model}"
        f"_L{cfg_model.n_layers}"
        f"_H{cfg_model.n_heads}"
        f"_ff{cfg_model.d_ff}"
        f"_T{cfg_model.n_diffusion_steps}"
    )


def _save_loss_plots(history, cfg_model, cfg_train, out_dir):
    if not _HAVE_MPL:
        print("[train] matplotlib not available — skipping loss plots")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    tag    = _model_tag(cfg_model)
    n_ep   = len(history["train_loss"])
    epochs = list(range(1, n_ep + 1))

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(epochs, history["train_loss"], label="Train Loss", color="steelblue", linewidth=1.8)
    ax.plot(epochs, history["val_loss"],   label="Val Loss",   color="tomato",    linewidth=1.8)
    best_ep = int(np.argmin(history["val_loss"])) + 1
    ax.axvline(best_ep, color="tomato", linestyle="--", alpha=0.55, label=f"Best val epoch={best_ep}")

    # Mark curriculum phase boundaries
    phase_colors = {2: "#e67e22", 3: "#27ae60"}
    phase_labels = {2: f"Phase 2 (ep {cfg_train.phase2_start})", 3: f"Phase 3 (ep {cfg_train.phase3_start})"}
    for ph, ep in [(2, cfg_train.phase2_start), (3, cfg_train.phase3_start)]:
        if ep <= n_ep:
            ax.axvline(ep, color=phase_colors[ph], linestyle=":", linewidth=1.4,
                       alpha=0.8, label=phase_labels[ph])

    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.set_title(
        f"TransFusion — Total Loss  (curriculum: ph1→{cfg_train.phase2_start}→{cfg_train.phase3_start})\n"
        f"d={cfg_model.d_model}  L={cfg_model.n_layers}  H={cfg_model.n_heads}  "
        f"ff={cfg_model.d_ff}  T={cfg_model.n_diffusion_steps}"
    )
    ax.legend(); ax.grid(True, alpha=0.3); fig.tight_layout()
    fname1 = out_dir / f"transfusion_loss_total_{tag}_ep{n_ep}.png"
    fig.savefig(fname1, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"[train] Loss plot (total)      → {fname1}")

    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    components = [
        ("diff_loss", "Diffusion MSE",  "steelblue"),
        ("pt_loss",   "Pitch-Type CE",  "seagreen"),
        ("oc_loss",   "Outcome CE",     "darkorange"),
        ("vae_loss",  "VAE (recon+KL)", "mediumpurple"),
    ]
    for ax, (key, label, color) in zip(axes, components):
        train_vals = history.get(f"train_{key}", [])
        val_vals   = history.get(f"val_{key}",   [])
        ax.plot(epochs[:len(train_vals)], train_vals, label="Train", color=color, linewidth=1.8)
        if val_vals:
            ax.plot(epochs[:len(val_vals)], val_vals, label="Val", color=color,
                    linewidth=1.8, linestyle="--", alpha=0.8)
        ax.axvline(best_ep, color="tomato", linestyle=":", alpha=0.5)
        ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.set_title(label)
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    fig.suptitle(
        f"TransFusion — Component Losses  |  d={cfg_model.d_model}  L={cfg_model.n_layers}  "
        f"H={cfg_model.n_heads}  ff={cfg_model.d_ff}  T={cfg_model.n_diffusion_steps}  "
        f"VAE p{cfg_model.pitcher_latent_dim}/b{cfg_model.batter_latent_dim}",
        fontsize=10,
    )
    fig.tight_layout()
    fname2 = out_dir / f"transfusion_loss_components_{tag}_ep{n_ep}.png"
    fig.savefig(fname2, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"[train] Loss plot (components) → {fname2}")


def _set_training_phase(model: "TransFusion", phase: int) -> None:
    """
    Configure which parameter groups are trainable for each curriculum phase.

    Corrected causal chain:
        pt_head → pitch_type → denoiser → pitch_feats → oc_head → outcome

    Phase 1 — VAE warmup  [ep 1 → phase2_start):
        Only pitcher_vae and batter_vae are updated.  Encoder and all heads
        are frozen so player latent spaces stabilise before sequence training.

    Phase 2 — Encoder + pitch_type + diffusion  [phase2_start → phase3_start):
        VAEs continue refining.  Encoder, pitch_type_head, and denoiser are
        unfrozen.  The entire outcome pathway (outcome_head, pt_to_oc_emb,
        pitch_feat_proj) stays frozen — the denoiser must produce meaningful
        physical features before the outcome head can learn from them.
        Under the corrected causal ordering, unlocking the outcome head before
        the denoiser is trained would expose it to random-quality pitch_feats,
        teaching it to ignore that conditioning signal entirely.

    Phase 3 — Full joint training  [phase3_start → end):
        Everything unfrozen.  outcome_head now conditions on:
          • context[t]         — well-trained encoder representation
          • pt_emb[t+1]        — well-trained pitch type embedding
          • pitch_feat_emb[t+1]— meaningful denoiser physical output
        This is the earliest point where the outcome head receives useful signal
        from all three conditioning inputs simultaneously.
    """
    PHASE_NAMES = {
        1: "Phase 1 — VAE warmup (encoder/heads frozen)",
        2: "Phase 2 — encoder + pitch_type + diffusion (outcome pathway frozen)",
        3: "Phase 3 — full joint training (all components active)",
    }
    print(f"\n[curriculum] Entering {PHASE_NAMES[phase]}\n")

    for p in model.parameters():
        p.requires_grad_(False)

    def _unfreeze(*modules):
        for m in modules:
            for p in m.parameters():
                p.requires_grad_(True)

    if phase == 1:
        _unfreeze(model.pitcher_vae, model.batter_vae)

    elif phase == 2:
        _unfreeze(
            model.pitcher_vae, model.batter_vae,
            model.encoder,                       # RoPE + cross-attn layers
            model.denoiser,                      # pitch physical features
            model.heads.pitch_type_head,         # pitch type prediction
            # Entire outcome pathway frozen:
            #   heads.outcome_head
            #   heads.pt_to_oc_emb
            #   heads.pitch_feat_proj   ← new; projects denoiser output into oc head
        )

    elif phase == 3:
        _unfreeze(
            model.pitcher_vae, model.batter_vae,
            model.encoder,
            model.denoiser,
            model.heads.pitch_type_head,
            model.heads.outcome_head,        # now gets real physics from denoiser
            model.heads.pt_to_oc_emb,
            model.heads.pitch_feat_proj,     # project pitch_feats → embed_dim for oc_head
        )

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total     = sum(p.numel() for p in model.parameters())
    print(f"[curriculum] Trainable params: {n_trainable:,} / {n_total:,} "
          f"({100*n_trainable/n_total:.1f}%)\n")



def calibrate_temperature(
    model:      "TransFusion",
    val_loader: DataLoader,
    device:     torch.device,
    cfg_model:  "ModelConfig",
    ckpt_dir:   Path,
    max_iter:   int = 100,
    lr:         float = 0.01,
) -> Tuple[float, float]:
    """
    Post-training temperature scaling calibration (Guo et al. 2017).

    Fits one scalar temperature per classification head on the validation set
    by minimising NLL (cross-entropy) of the true labels.  Does NOT change
    model weights — only the temperature scalars T_pt and T_oc are optimised.

    Why two separate temperatures:
        The pitch-type head and outcome head are trained with different loss
        weights and at different curriculum phases, so they are likely to be
        miscalibrated to different degrees.  Fitting them independently gives
        each head the freedom to reach its own optimal calibration without
        interfering with the other.

    Effect on metrics:
        Accuracy is unchanged (argmax is invariant to positive scaling).
        Brier score improves because overconfident predictions (probabilities
        near 0 or 1 that are wrong) are pulled toward empirical frequencies.
        Log-loss improves directly since we optimise NLL.

    The fitted temperatures are:
        • Stored in cfg_model.pt_temperature / cfg_model.oc_temperature
        • Saved into the best.pt checkpoint (overwriting the old file)
        • Used automatically by predict_next_pitch_type / predict_next_outcome
          via the temperature argument already present in those methods

    Args:
        model:      fully trained TransFusion, in eval mode
        val_loader: validation DataLoader (not shuffled)
        device:     compute device
        cfg_model:  ModelConfig — pt_temperature and oc_temperature updated in place
        ckpt_dir:   directory containing best.pt — checkpoint is re-saved after fitting
        max_iter:   LBFGS iterations per temperature (100 is typically more than enough)
        lr:         LBFGS learning rate

    Returns:
        (T_pt, T_oc) — the fitted temperature scalars
    """
    print("\n[calibrate] Starting post-training temperature scaling on val set...")
    model.eval()

    # ── 1. Collect all logits and targets from the val set ────────────────
    all_pt_logits: List[torch.Tensor] = []
    all_oc_logits: List[torch.Tensor] = []
    all_pt_targets: List[torch.Tensor] = []
    all_oc_targets: List[torch.Tensor] = []

    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            pitch_seq     = batch["pitch_seq"]
            pitch_types   = batch["pitch_types"]
            outcomes      = batch["outcomes"]
            batter_ctx    = batch["batter_ctx"]
            mask          = batch["mask"]
            B, T, _ = pitch_seq.shape

            # VAE encode
            pitcher_z, _, _ = model.pitcher_vae.vae_loss(batch["pitcher_ctx"])
            batter_flat = batter_ctx.reshape(B * T, -1)
            batter_z_flat, bm, blv = model.batter_vae(batter_flat)
            batter_z_mean = batter_z_flat.reshape(B, T, model.cfg.batter_latent_dim).mean(dim=1)

            # Encoder forward (no outcome in input — corrected chain)
            context = model.encoder(
                pitch_seq, pitch_types,
                batter_ctx, batch["batter_ids"],
                pitcher_z, batch["pitcher_id"],
                batch["batting_order"], batch["game_ctx"],
                batter_z_mean, mask,
            )

            # Pitch type logits: context[t] → pt_logits (predicts pitch_types[t+1])
            pt_logits = model.heads.forward_pt(context[:, :-1])           # [B, T-1, n_pt]
            pt_targets = pitch_types[:, 1:]                                # [B, T-1]

            # Ground-truth pitch features for outcome conditioning
            pitch_feats_gt = pitch_seq[:, 1:, :model.n_cont]              # [B, T-1, n_cont]

            # Outcome logits: context[t] + pt[t+1] + feats[t+1] → oc_logits
            oc_logits = model.heads.forward_oc(
                context[:, :-1], pitch_types[:, 1:], pitch_feats_gt
            )                                                              # [B, T-1, n_oc]

            # Flatten and filter padding (ignore_index=0)
            valid = mask[:, 1:].reshape(-1)                               # [B*(T-1)]

            pt_flat = pt_logits.reshape(-1, model.cfg.num_pitch_types)[valid]
            pt_tgt  = pt_targets.reshape(-1)[valid]
            oc_flat = oc_logits.reshape(-1, model.cfg.num_outcomes)[valid]
            oc_tgt  = outcomes[:, 1:].reshape(-1)[valid]

            # Filter <UNK> (token 0) from targets
            pt_mask = pt_tgt != 0
            oc_mask = oc_tgt != 0

            all_pt_logits.append(pt_flat[pt_mask].cpu())
            all_pt_targets.append(pt_tgt[pt_mask].cpu())
            all_oc_logits.append(oc_flat[oc_mask].cpu())
            all_oc_targets.append(oc_tgt[oc_mask].cpu())

    pt_logits_all  = torch.cat(all_pt_logits)   # [N_pt, n_pitch_types]
    pt_targets_all = torch.cat(all_pt_targets)  # [N_pt]
    oc_logits_all  = torch.cat(all_oc_logits)   # [N_oc, n_outcomes]
    oc_targets_all = torch.cat(all_oc_targets)  # [N_oc]

    print(f"[calibrate] Collected {len(pt_targets_all):,} pitch-type samples, "
          f"{len(oc_targets_all):,} outcome samples")

    # ── 2. Fit temperature for each head via LBFGS ────────────────────────
    def _fit_temperature(logits: torch.Tensor, targets: torch.Tensor,
                         label: str) -> float:
        """Minimise NLL w.r.t. a single temperature scalar using LBFGS."""
        T_param = nn.Parameter(torch.ones(1))
        optimizer = torch.optim.LBFGS([T_param], lr=lr, max_iter=max_iter,
                                       tolerance_grad=1e-7, tolerance_change=1e-9,
                                       line_search_fn="strong_wolfe")

        nll_before = F.cross_entropy(logits, targets).item()

        def _closure():
            optimizer.zero_grad()
            loss = F.cross_entropy(logits / T_param.clamp(min=1e-2), targets)
            loss.backward()
            return loss

        optimizer.step(_closure)

        T_val = float(T_param.clamp(min=1e-2).item())
        nll_after = F.cross_entropy(logits / T_val, targets).item()

        print(f"[calibrate] {label:12s}  T={T_val:.4f}  "
              f"NLL: {nll_before:.4f} → {nll_after:.4f}  "
              f"(Δ = {nll_after - nll_before:+.4f})")
        return T_val

    T_pt = _fit_temperature(pt_logits_all, pt_targets_all, "pitch_type")
    T_oc = _fit_temperature(oc_logits_all, oc_targets_all, "outcome")

    # ── 3. Compute pre/post Brier scores on win-probability proxy ─────────
    # Use pitch-type calibration as a proxy (direct win-prob Brier requires sim)
    def _brier(logits, targets, T):
        probs = torch.softmax(logits / T, dim=-1)
        one_hot = torch.zeros_like(probs)
        one_hot.scatter_(1, targets.unsqueeze(1), 1.0)
        return float(((probs - one_hot) ** 2).sum(dim=-1).mean())

    bs_pt_before = _brier(pt_logits_all, pt_targets_all, 1.0)
    bs_pt_after  = _brier(pt_logits_all, pt_targets_all, T_pt)
    bs_oc_before = _brier(oc_logits_all, oc_targets_all, 1.0)
    bs_oc_after  = _brier(oc_logits_all, oc_targets_all, T_oc)

    print(f"[calibrate] Brier score (pitch_type): {bs_pt_before:.4f} → {bs_pt_after:.4f}")
    print(f"[calibrate] Brier score (outcome):    {bs_oc_before:.4f} → {bs_oc_after:.4f}")

    # ── 4. Store temperatures in config and re-save best checkpoint ───────
    cfg_model.pt_temperature = T_pt
    cfg_model.oc_temperature = T_oc

    best_ckpt_path = ckpt_dir / "best.pt"
    if best_ckpt_path.exists():
        ckpt = torch.load(best_ckpt_path, map_location="cpu")
        ckpt["cfg_model"]["pt_temperature"] = T_pt
        ckpt["cfg_model"]["oc_temperature"] = T_oc
        ckpt["pt_temperature"] = T_pt   # top-level for easy inspection
        ckpt["oc_temperature"] = T_oc
        torch.save(ckpt, best_ckpt_path)
        print(f"[calibrate] Saved calibrated temperatures to {best_ckpt_path}")
    else:
        print(f"[calibrate] WARNING: {best_ckpt_path} not found — temperatures not saved to checkpoint")

    # Also save a standalone calibration record
    cal_path = ckpt_dir / "calibration.json"
    import json as _json
    with open(cal_path, "w") as f:
        _json.dump({
            "pt_temperature": T_pt,
            "oc_temperature": T_oc,
            "pt_nll_before":  float(F.cross_entropy(pt_logits_all, pt_targets_all).item()),
            "pt_nll_after":   float(F.cross_entropy(pt_logits_all / T_pt, pt_targets_all).item()),
            "oc_nll_before":  float(F.cross_entropy(oc_logits_all, oc_targets_all).item()),
            "oc_nll_after":   float(F.cross_entropy(oc_logits_all / T_oc, oc_targets_all).item()),
            "pt_brier_before": bs_pt_before,
            "pt_brier_after":  bs_pt_after,
            "oc_brier_before": bs_oc_before,
            "oc_brier_after":  bs_oc_after,
        }, f, indent=2)
    print(f"[calibrate] Saved calibration record to {cal_path}")

    model.train()
    return T_pt, T_oc


@torch.no_grad()
def _evaluate(model: "TransFusion", loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    totals = {"loss": 0.0, "diff_loss": 0.0, "pt_loss": 0.0, "oc_loss": 0.0, "vae_loss": 0.0}
    count  = 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        losses = model(batch)
        for k in totals:
            totals[k] += losses[k].item()
        count += 1
    model.train()
    n = max(count, 1)
    return {k: v / n for k, v in totals.items()}

@torch.no_grad()
def _evaluate_test(
    model:     "TransFusion",
    loader:    DataLoader,
    device:    torch.device,
    cfg_model: "ModelConfig",
    ckpt_dir:  Path,
) -> Dict[str, float]:
    """
    Full evaluation on the held-out test set.

    Computes:
        loss, diff_loss, pt_loss, oc_loss, vae_loss  — same as val metrics
        pt_brier_uncalibrated  — pitch-type Brier before temperature scaling
        pt_brier_calibrated    — pitch-type Brier with T_pt applied
        oc_brier_uncalibrated  — outcome Brier before temperature scaling
        oc_brier_calibrated    — outcome Brier with T_oc applied
        n_pitches              — total non-padding pitches evaluated

    Results are saved to checkpoints/test_metrics.json.
    This function must be called AFTER calibrate_temperature() so that
    cfg_model.pt_temperature and cfg_model.oc_temperature are set.
    """
    model.eval()

    T_pt = cfg_model.pt_temperature
    T_oc = cfg_model.oc_temperature

    totals = {
        "loss": 0.0, "diff_loss": 0.0, "pt_loss": 0.0,
        "oc_loss": 0.0, "vae_loss": 0.0,
    }
    count = 0

    all_pt_logits:  List[torch.Tensor] = []
    all_pt_targets: List[torch.Tensor] = []
    all_oc_logits:  List[torch.Tensor] = []
    all_oc_targets: List[torch.Tensor] = []
    total_pitches = 0

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}

        # Standard loss forward (uses ground-truth teacher-forcing throughout)
        losses = model(batch)
        for k in totals:
            totals[k] += losses[k].item()
        count += 1

        # Collect logits for Brier score computation
        pitch_seq   = batch["pitch_seq"]
        pitch_types = batch["pitch_types"]
        outcomes    = batch["outcomes"]
        batter_ctx  = batch["batter_ctx"]
        mask        = batch["mask"]
        B, T, _     = pitch_seq.shape

        pitcher_z, _, _ = model.pitcher_vae.vae_loss(batch["pitcher_ctx"])
        batter_flat = batter_ctx.reshape(B * T, -1)
        bz, bm, blv = model.batter_vae(batter_flat)
        batter_z_mean = bz.reshape(B, T, model.cfg.batter_latent_dim).mean(dim=1)

        context = model.encoder(
            pitch_seq, pitch_types,
            batter_ctx, batch["batter_ids"],
            pitcher_z, batch["pitcher_id"],
            batch["batting_order"], batch["game_ctx"],
            batter_z_mean, mask,
        )

        pt_logits = model.heads.forward_pt(context[:, :-1])
        pitch_feats_gt = pitch_seq[:, 1:, :model.n_cont]
        oc_logits = model.heads.forward_oc(
            context[:, :-1], pitch_types[:, 1:], pitch_feats_gt
        )

        valid = mask[:, 1:].reshape(-1)
        pt_flat = pt_logits.reshape(-1, model.cfg.num_pitch_types)[valid]
        pt_tgt  = pitch_types[:, 1:].reshape(-1)[valid]
        oc_flat = oc_logits.reshape(-1, model.cfg.num_outcomes)[valid]
        oc_tgt  = outcomes[:, 1:].reshape(-1)[valid]

        pt_mask = pt_tgt != 0
        oc_mask = oc_tgt != 0

        all_pt_logits.append(pt_flat[pt_mask].cpu())
        all_pt_targets.append(pt_tgt[pt_mask].cpu())
        all_oc_logits.append(oc_flat[oc_mask].cpu())
        all_oc_targets.append(oc_tgt[oc_mask].cpu())
        total_pitches += int(valid.sum())

    model.train()
    n = max(count, 1)
    result = {k: v / n for k, v in totals.items()}
    result["n_pitches"] = total_pitches

    # ── Brier scores (multi-class: mean squared error of full probability vector)
    def _brier(logits: torch.Tensor, targets: torch.Tensor, T: float) -> float:
        probs    = torch.softmax(logits / T, dim=-1)
        one_hot  = torch.zeros_like(probs)
        one_hot.scatter_(1, targets.unsqueeze(1), 1.0)
        return float(((probs - one_hot) ** 2).sum(dim=-1).mean())

    pt_all = torch.cat(all_pt_logits)
    pt_tgt_all = torch.cat(all_pt_targets)
    oc_all = torch.cat(all_oc_logits)
    oc_tgt_all = torch.cat(all_oc_targets)

    result["pt_brier_uncalibrated"] = _brier(pt_all, pt_tgt_all, 1.0)
    result["pt_brier_calibrated"]   = _brier(pt_all, pt_tgt_all, T_pt)
    result["oc_brier_uncalibrated"] = _brier(oc_all, oc_tgt_all, 1.0)
    result["oc_brier_calibrated"]   = _brier(oc_all, oc_tgt_all, T_oc)
    result["pt_temperature_used"]   = T_pt
    result["oc_temperature_used"]   = T_oc

    # ── Save ─────────────────────────────────────────────────────────────────
    out = ckpt_dir / "test_metrics.json"
    _save_json(result, out)
    print(f"[test]  Saved test metrics → {out}")

    return result


def train(cfg_train: TrainConfig, cfg_model: ModelConfig):
    torch.manual_seed(cfg_train.seed)
    np.random.seed(cfg_train.seed)
    random.seed(cfg_train.seed)

    device   = _resolve_device(cfg_train.device)
    ckpt_dir = Path(cfg_train.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    print(f"[train] Device: {device}")

    builder = BaseballDatasetBuilder(
        start_dt=cfg_train.start_dt, end_dt=cfg_train.end_dt,
        val_start_dt=cfg_train.val_start_dt, test_start_dt=cfg_train.test_start_dt,
        cache_dir=cfg_train.cache_dir, max_seq_len=cfg_model.max_seq_len,
        min_pitches_per_game=100,
    )
    train_ds, val_ds, test_ds, encoders = builder.build()

    cfg_model.pitch_feat_dim   = train_ds.pitch_feat_dim
    cfg_model.pitcher_feat_dim = train_ds.pitcher_feat_dim
    cfg_model.batter_feat_dim  = train_ds.batter_feat_dim
    cfg_model.game_feat_dim    = train_ds.game_feat_dim
    cfg_model.num_pitch_types  = encoders.num_pitch_types
    cfg_model.num_outcomes     = encoders.num_outcomes
    cfg_model.num_events       = encoders.num_events
    cfg_model.num_batters      = encoders.num_batters
    cfg_model.num_pitchers     = encoders.num_pitchers

    print(f"[train] pitch_feat_dim={cfg_model.pitch_feat_dim}  "
          f"pitcher_feat_dim={cfg_model.pitcher_feat_dim}  "
          f"batter_feat_dim={cfg_model.batter_feat_dim}")
    print(f"[train] pitcher_latent={cfg_model.pitcher_latent_dim}  "
          f"batter_latent={cfg_model.batter_latent_dim}  "
          f"lambda_vae={cfg_model.lambda_vae}")

    # ── Compute inverse-frequency class weights from training data ─────────
    # Rare outcomes (contact, HBP) get higher weight; common ones (ball,
    # called_strike) get lower weight.  We clip to avoid extreme weights from
    # classes with < 10 examples, and normalise so mean weight = 1.0.
    def _class_weights(dataset: PitchSequenceDataset,
                       enc_fn, n_classes: int,
                       key: str) -> List[float]:
        counts = np.zeros(n_classes, dtype=np.float64)
        for gid in dataset.game_ids:
            gdf = dataset.game_groups[gid]
            for v in gdf[key]:
                idx = enc_fn(v)
                if 0 < idx < n_classes:
                    counts[idx] += 1
        # inverse frequency; class 0 (<UNK>) gets weight 0 (handled by ignore_index)
        counts[0] = 0
        safe = np.where(counts > 10, counts, np.inf)
        inv  = np.where(safe < np.inf, 1.0 / safe, 0.0)
        mean = inv[inv > 0].mean() if (inv > 0).any() else 1.0
        w = (inv / mean).tolist()
        return w

    from new_dataset_builder import PITCH_TYPE_COL, OUTCOME_COL
    print("[train] Computing class weights from training data...")
    cfg_model.pt_class_weights = _class_weights(
        train_ds, encoders.enc_pitch_type,
        encoders.num_pitch_types, PITCH_TYPE_COL)
    cfg_model.oc_class_weights = _class_weights(
        train_ds, encoders.enc_outcome,
        encoders.num_outcomes, OUTCOME_COL)
    print(f"[train] pt class weights: min={min(w for w in cfg_model.pt_class_weights if w>0):.3f}  "
          f"max={max(cfg_model.pt_class_weights):.3f}")
    print(f"[train] oc class weights: min={min(w for w in cfg_model.oc_class_weights if w>0):.3f}  "
          f"max={max(cfg_model.oc_class_weights):.3f}")

    train_loader = DataLoader(
        train_ds, batch_size=cfg_train.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=cfg_train.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg_train.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=cfg_train.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg_train.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=cfg_train.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model     = TransFusion(cfg_model).to(device)
    n_params  = sum(p.numel() for p in model.parameters())
    print(f"[train] TransFusion parameters (total): {n_params:,}")

    total_steps = cfg_train.epochs * len(train_loader)
    scaler_amp  = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    _save_json(vars(cfg_model), ckpt_dir / "model_config.json")

    best_val_loss = float("inf")
    global_step   = 0
    current_phase = 0   # tracks last activated phase to avoid redundant switches

    def _rebuild_optimizer_and_scheduler(step_offset: int = 0):
        """Create a fresh AdamW over currently-trainable params + a new LR scheduler."""
        trainable = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(
            trainable, lr=cfg_train.lr,
            weight_decay=cfg_train.weight_decay,
            betas=(0.9, 0.98), eps=1e-9,
        )
        # Remaining steps from this point forward; scheduler tracks position via step_offset
        remaining = max(total_steps - step_offset, 1)
        sch = get_lr_scheduler(opt, cfg_train.warmup_steps, total_steps)
        # Fast-forward scheduler to current position so LR is continuous across transitions
        for _ in range(step_offset):
            sch.step()
        return opt, sch

    history: Dict[str, List[float]] = {
        "train_loss": [], "val_loss": [],
        "train_diff_loss": [], "train_pt_loss": [], "train_oc_loss": [], "train_vae_loss": [],
        "val_diff_loss":   [], "val_pt_loss":   [], "val_oc_loss":   [], "val_vae_loss":   [],
        "phase": [],   # records the active phase each epoch for the loss plot
    }

    # Start in phase 1; optimizer/scheduler built after first phase is set
    _set_training_phase(model, 1)
    current_phase = 1
    optimizer, scheduler = _rebuild_optimizer_and_scheduler(step_offset=0)

    for epoch in range(1, cfg_train.epochs + 1):

        # ── Curriculum phase transitions ───────────────────────────────────
        if epoch == cfg_train.phase2_start and current_phase < 2:
            _set_training_phase(model, 2)
            current_phase = 2
            optimizer, scheduler = _rebuild_optimizer_and_scheduler(global_step)
        elif epoch == cfg_train.phase3_start and current_phase < 3:
            _set_training_phase(model, 3)
            current_phase = 3
            optimizer, scheduler = _rebuild_optimizer_and_scheduler(global_step)
        # ──────────────────────────────────────────────────────────────────

        model.train()
        epoch_losses = {"loss": 0.0, "diff_loss": 0.0, "pt_loss": 0.0, "oc_loss": 0.0, "vae_loss": 0.0}
        t0 = time.time()

        for step, batch in tqdm(enumerate(train_loader), total=len(train_loader),
                                desc=f"Epoch {epoch}/{cfg_train.epochs}"):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                losses = model(batch)
            scaler_amp.scale(losses["loss"]).backward()
            scaler_amp.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg_train.grad_clip)
            scaler_amp.step(optimizer)
            scaler_amp.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            global_step += 1
            for k in epoch_losses:
                epoch_losses[k] += losses[k].item()
            if global_step % cfg_train.log_every == 0:
                lr = scheduler.get_last_lr()[0]
                print(f"  ep={epoch:3d} step={global_step:6d} "
                      f"loss={losses['loss'].item():.4f} "
                      f"diff={losses['diff_loss'].item():.4f} "
                      f"pt={losses['pt_loss'].item():.4f} "
                      f"oc={losses['oc_loss'].item():.4f} "
                      f"vae={losses['vae_loss'].item():.4f} "
                      f"lr={lr:.2e}")

        n = len(train_loader)
        print(f"[epoch {epoch:3d}] train_loss={epoch_losses['loss']/n:.4f}  ({time.time()-t0:.0f}s)")

        history["train_loss"].append(epoch_losses["loss"] / n)
        history["train_diff_loss"].append(epoch_losses["diff_loss"] / n)
        history["train_pt_loss"].append(epoch_losses["pt_loss"] / n)
        history["train_oc_loss"].append(epoch_losses["oc_loss"] / n)
        history["train_vae_loss"].append(epoch_losses["vae_loss"] / n)
        history["phase"].append(current_phase)

        if epoch % cfg_train.val_every == 0:
            val_metrics = _evaluate(model, val_loader, device)
            val_loss    = val_metrics["loss"]
            print(f"[epoch {epoch:3d}] val_loss={val_loss:.4f}  "
                  f"diff={val_metrics['diff_loss']:.4f}  "
                  f"pt={val_metrics['pt_loss']:.4f}  "
                  f"oc={val_metrics['oc_loss']:.4f}  "
                  f"vae={val_metrics['vae_loss']:.4f}")

            history["val_loss"].append(val_loss)
            history["val_diff_loss"].append(val_metrics["diff_loss"])
            history["val_pt_loss"].append(val_metrics["pt_loss"])
            history["val_oc_loss"].append(val_metrics["oc_loss"])
            history["val_vae_loss"].append(val_metrics["vae_loss"])

            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_loss": val_loss,
                "cfg_model": vars(cfg_model),
            }, ckpt_dir / "latest.pt")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "val_loss": val_loss,
                    "cfg_model": vars(cfg_model),
                }, ckpt_dir / "best.pt")
                print(f"  ✓ New best val_loss={val_loss:.4f}")

    print(f"[train] Done. Best val_loss={best_val_loss:.4f}")
    _save_loss_plots(history, cfg_model, cfg_train, Path(cfg_train.checkpoint_dir) / "plots")

    # ── Post-training temperature scaling calibration ─────────────────────
    # Load the best checkpoint weights before calibrating (not the last epoch)
    print("\n[train] Loading best checkpoint for calibration...")
    best_ckpt = torch.load(ckpt_dir / "best.pt", map_location=device)
    model.load_state_dict(best_ckpt["model_state"])
    model.eval()

    T_pt, T_oc = calibrate_temperature(
        model=model,
        val_loader=val_loader,
        device=device,
        cfg_model=cfg_model,
        ckpt_dir=ckpt_dir,
    )
    print(f"[train] Calibration complete. T_pt={T_pt:.4f}  T_oc={T_oc:.4f}")

    # ── Test set evaluation ────────────────────────────────────────────────
    # Run AFTER calibration so reported Brier scores use calibrated temperatures.
    # This is the number that goes in Table 2 of the paper.
    print("\n[train] Evaluating on held-out test set...")
    test_metrics = _evaluate_test(
        model=model,
        loader=test_loader,
        device=device,
        cfg_model=cfg_model,
        ckpt_dir=ckpt_dir,
    )
    print(
        f"[test]  loss={test_metrics['loss']:.4f}  "
        f"diff={test_metrics['diff_loss']:.4f}  "
        f"pt={test_metrics['pt_loss']:.4f}  "
        f"oc={test_metrics['oc_loss']:.4f}  "
        f"vae={test_metrics['vae_loss']:.4f}"
    )
    print(
        f"[test]  Brier (pt, calibrated)={test_metrics['pt_brier_calibrated']:.4f}  "
        f"Brier (oc, calibrated)={test_metrics['oc_brier_calibrated']:.4f}"
    )


# =============================================================================
# 6.  MCMC SIMULATION
# =============================================================================

@dataclass
class SimResult:
    game_pk:           int
    context_innings:   float
    n_simulations:     int
    home_win_prob:     float
    away_win_prob:     float
    tie_prob:          float
    mean_home_runs:    float
    mean_away_runs:    float
    std_home_runs:     float
    std_away_runs:     float
    actual_home_score: Optional[int]
    actual_away_score: Optional[int]
    actual_home_win:   Optional[bool]


@dataclass
class BatterLine:
    """Expected batting statistics for one lineup slot, averaged over K simulations."""
    slot:   int    # 0-indexed batting order slot
    AB:     float = 0.0
    H:      float = 0.0
    HR:     float = 0.0
    doubles: float = 0.0
    triples: float = 0.0
    BB:     float = 0.0
    HBP:    float = 0.0
    K:      float = 0.0
    RBI:    float = 0.0
    # std across simulations
    AB_std:  float = 0.0
    H_std:   float = 0.0
    HR_std:  float = 0.0
    RBI_std: float = 0.0
    K_std:   float = 0.0

    @property
    def AVG(self) -> float:
        return self.H / self.AB if self.AB > 0 else 0.0

    @property
    def OBP(self) -> float:
        pa = self.AB + self.BB + self.HBP
        return (self.H + self.BB + self.HBP) / pa if pa > 0 else 0.0

    @property
    def SLG(self) -> float:
        tb = (self.H - self.doubles - self.triples - self.HR) \
             + 2 * self.doubles + 3 * self.triples + 4 * self.HR
        return tb / self.AB if self.AB > 0 else 0.0

    def to_dict(self) -> Dict:
        r = lambda x: round(float(x), 2)   # ensure Python float, not numpy float32
        return {
            "slot": int(self.slot), "AB": r(self.AB), "H": r(self.H),
            "2B": r(self.doubles), "3B": r(self.triples),
            "HR": r(self.HR), "BB": r(self.BB),
            "HBP": r(self.HBP), "K": r(self.K),
            "RBI": r(self.RBI),
            "AVG": f"{self.AVG:.3f}", "OBP": f"{self.OBP:.3f}", "SLG": f"{self.SLG:.3f}",
            "AB_std": r(self.AB_std), "H_std": r(self.H_std),
            "HR_std": r(self.HR_std), "RBI_std": r(self.RBI_std),
            "K_std": r(self.K_std),
        }


@dataclass
class PitcherLine:
    """
    Expected pitching statistics for the simulated portion of the game,
    averaged over K simulations. Keyed by team ('home'/'away').
    """
    team:      str    # 'home' or 'away'
    outs:      float = 0.0   # outs recorded = IP * 3
    H:         float = 0.0
    HR:        float = 0.0
    BB:        float = 0.0
    K:         float = 0.0
    R:         float = 0.0   # runs allowed (all earned in our model)
    outs_std:  float = 0.0
    R_std:     float = 0.0

    @property
    def IP(self) -> float:
        full = int(self.outs) // 3
        rem  = int(round(self.outs)) % 3
        return float(f"{full}.{rem}")

    def to_dict(self) -> Dict:
        r = lambda x: round(float(x), 2)
        return {
            "team": self.team, "IP": float(self.IP),
            "H": r(self.H), "HR": r(self.HR),
            "BB": r(self.BB), "K": r(self.K),
            "R": r(self.R),
            "outs_std": r(self.outs_std), "R_std": r(self.R_std),
        }


@dataclass
class BoxScore:
    game_pk:         int
    context_innings: float
    n_simulations:   int
    away_batters:    List[BatterLine]
    home_batters:    List[BatterLine]
    away_pitcher:    PitcherLine
    home_pitcher:    PitcherLine

    def print_table(self):
        """Pretty-print the box score to stdout."""
        def _batter_table(lines: List[BatterLine], label: str):
            print(f"\n  {label} BATTING")
            print(f"  {'Slot':>4}  {'AB':>5}  {'H':>5}  {'2B':>5}  {'3B':>5}  "
                  f"{'HR':>5}  {'BB':>5}  {'K':>5}  {'RBI':>5}  {'AVG':>5}  {'OBP':>5}  {'SLG':>5}")
            print("  " + "-" * 80)
            for b in lines:
                print(f"  {b.slot+1:>4}  {b.AB:>5.1f}  {b.H:>5.1f}  {b.doubles:>5.1f}  "
                      f"{b.triples:>5.1f}  {b.HR:>5.1f}  {b.BB:>5.1f}  {b.K:>5.1f}  "
                      f"{b.RBI:>5.1f}  {b.AVG:>5.3f}  {b.OBP:>5.3f}  {b.SLG:>5.3f}")

        def _pitcher_table(line: PitcherLine, label: str):
            print(f"\n  {label} PITCHING (simulated portion)")
            print(f"  {'IP':>6}  {'H':>5}  {'HR':>5}  {'BB':>5}  {'K':>5}  {'R':>5}")
            print("  " + "-" * 40)
            print(f"  {line.IP:>6}  {line.H:>5.1f}  {line.HR:>5.1f}  "
                  f"{line.BB:>5.1f}  {line.K:>5.1f}  {line.R:>5.1f}")

        print(f"\n{'='*55}")
        print(f"  BOX SCORE  game={self.game_pk}  "
              f"context={self.context_innings}inn  N={self.n_simulations}")
        print(f"{'='*55}")
        _batter_table(self.away_batters, "AWAY")
        _batter_table(self.home_batters, "HOME")
        _pitcher_table(self.away_pitcher, "AWAY")
        _pitcher_table(self.home_pitcher, "HOME")
        print()

    def to_dict(self) -> Dict:
        return {
            "game_pk": self.game_pk,
            "context_innings": self.context_innings,
            "n_simulations": self.n_simulations,
            "away_batters": [b.to_dict() for b in self.away_batters],
            "home_batters": [b.to_dict() for b in self.home_batters],
            "away_pitcher": self.away_pitcher.to_dict(),
            "home_pitcher": self.home_pitcher.to_dict(),
        }


def _aggregate_box_scores(
    batter_stats: np.ndarray,   # [K, 2, 9, 9]  (K, team, slot, stat)
    pitcher_stats: np.ndarray,  # [K, 2, 6]     (K, team, stat)
    game_pk: int,
    context_innings: float,
    K: int,
) -> BoxScore:
    """
    Aggregate per-simulation stat arrays into a BoxScore with means and stds.

    batter_stats axis-3 order:  AB, H, HR, 2B, 3B, BB, HBP, K, RBI
    pitcher_stats axis-2 order: outs, H, HR, BB, K, R
    """
    BATTER_STAT_NAMES = ["AB","H","HR","2B","3B","BB","HBP","K","RBI"]
    PITCHER_STAT_NAMES = ["outs","H","HR","BB","K","R"]
    BS = len(BATTER_STAT_NAMES)
    PS = len(PITCHER_STAT_NAMES)

    def _make_batter_lines(team_idx: int) -> List[BatterLine]:
        lines = []
        for slot in range(9):
            arr = batter_stats[:, team_idx, slot, :]   # [K, 9]
            mu  = arr.mean(axis=0)
            sd  = arr.std(axis=0)
            lines.append(BatterLine(
                slot=slot,
                AB=mu[0], H=mu[1], HR=mu[2], doubles=mu[3], triples=mu[4],
                BB=mu[5], HBP=mu[6], K=mu[7], RBI=mu[8],
                AB_std=sd[0], H_std=sd[1], HR_std=sd[2], RBI_std=sd[8], K_std=sd[7],
            ))
        return lines

    def _make_pitcher_line(team_idx: int, team: str) -> PitcherLine:
        arr = pitcher_stats[:, team_idx, :]   # [K, 6]
        mu  = arr.mean(axis=0)
        sd  = arr.std(axis=0)
        return PitcherLine(
            team=team,
            outs=mu[0], H=mu[1], HR=mu[2], BB=mu[3], K=mu[4], R=mu[5],
            outs_std=sd[0], R_std=sd[5],
        )

    return BoxScore(
        game_pk=game_pk,
        context_innings=context_innings,
        n_simulations=K,
        away_batters=_make_batter_lines(0),
        home_batters=_make_batter_lines(1),
        away_pitcher=_make_pitcher_line(0, "away"),
        home_pitcher=_make_pitcher_line(1, "home"),
    )


def simulate_games_mh(cfg_sim: SimConfig, cfg_model: ModelConfig,
                       lam: float = 1.0, n_steps: int = 500, burn_in: int = 100):
    device  = _resolve_device(cfg_sim.device)
    out_dir = Path(cfg_sim.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt      = torch.load(cfg_sim.checkpoint, map_location=device)
    saved_cfg = ckpt.get("cfg_model", {})
    for k, v in saved_cfg.items():
        if hasattr(cfg_model, k):
            setattr(cfg_model, k, v)

    model = TransFusion(cfg_model).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # Calibrated temperatures from checkpoint (saved_cfg loop above already
    # populated cfg_model.pt_temperature and cfg_model.oc_temperature)
    pt_temp = cfg_model.pt_temperature
    oc_temp = cfg_model.oc_temperature
    print(f"[mh-sim] Loaded checkpoint (epoch {ckpt.get('epoch','?')})")
    print(f"[mh-sim] λ={lam}  steps={n_steps}  burn_in={burn_in}")
    print(f"[mh-sim] Calibrated temperatures: T_pt={pt_temp:.4f}  T_oc={oc_temp:.4f}")

    builder = BaseballDatasetBuilder(
        start_dt=cfg_sim.start_dt, end_dt=cfg_sim.end_dt,
        val_start_dt=cfg_sim.val_start_dt, test_start_dt=cfg_sim.test_start_dt,
        cache_dir=cfg_sim.cache_dir, max_seq_len=cfg_model.max_seq_len,
        min_pitches_per_game=100,
    )
    train_ds, val_ds, test_ds, encoders = builder.build()
    _load_re24_table(cfg_sim.cache_dir)
    _load_in_play_probs(cfg_sim.cache_dir)
    dataset: PitchSequenceDataset = {"train": train_ds, "val": val_ds, "test": test_ds}[cfg_sim.split]
    print(f"[mh-sim] {len(dataset)} games, split={cfg_sim.split}")

    results = []
    for game_idx in range(len(dataset)):
        sample  = dataset[game_idx]
        game_pk = sample["game_pk"].item()
        game_df = dataset.game_groups[game_pk]

        context_end_idx = find_context_split(game_df, cfg_sim.context_innings)
        actual_home     = int(game_df["home_score"].iloc[-1])
        actual_away     = int(game_df["away_score"].iloc[-1])

        if context_end_idx == 0:
            ctx_batch   = _make_empty_context_batch(sample, device)
            ctx_out_idx = 0
        else:
            T_available     = sample["pitch_seq"].shape[0]
            context_end_idx = min(context_end_idx, T_available)
            ctx_batch       = _make_context_batch(sample, context_end_idx, device)
            ctx_out_idx     = context_end_idx - 1

        with torch.no_grad():
            context_memory = model.encode_context(ctx_batch)
        ctx_vec_1 = context_memory[:, ctx_out_idx, :]

        context_state = _reconstruct_game_state(game_df, context_end_idx)
        current_traj  = _simulate_one_game(
            model, encoders, dataset.pitch_scaler, dataset.batter_scaler,
            ctx_vec_1.expand(1, -1), context_state,
            sample["batting_order"].tolist(), device, cfg_sim.max_game_pitches,
            pt_temperature=pt_temp,
            oc_temperature=oc_temp,
        )

        win_votes = 0; n_accepted = 0; post_burn_count = 0

        for step in range(n_steps + burn_in):
            boundaries = current_traj["half_inning_boundaries"]
            m_tau      = len(boundaries)
            if m_tau == 0:
                proposed_traj = current_traj; accepted = True
            else:
                split_idx  = random.randint(0, m_tau - 1)
                split_info = boundaries[split_idx]
                split_ctx_vec = _get_context_at_boundary(
                    model, sample, ctx_vec_1, split_info, device,
                    dataset.pitch_scaler, dataset.batter_scaler, encoders,
                    context_end_idx, cfg_sim.max_game_pitches,
                )
                proposed_traj = _simulate_one_game(
                    model, encoders, dataset.pitch_scaler, dataset.batter_scaler,
                    split_ctx_vec.expand(1, -1), split_info["game_state"],
                    sample["batting_order"].tolist(), device, cfg_sim.max_game_pitches,
                    prefix_phi_sum=split_info["prefix_phi_sum"],
                    pt_temperature=pt_temp,
                    oc_temperature=oc_temp,
                )
                m_tau_prime = len(proposed_traj["half_inning_boundaries"])
                if lam == 0.0:
                    log_alpha = 0.0
                else:
                    suffix_phi_current  = current_traj["total_phi"] - split_info["prefix_phi_sum"]
                    suffix_phi_proposed = proposed_traj["total_phi"] - split_info["prefix_phi_sum"]
                    log_ratio_m = math.log(m_tau) - math.log(max(m_tau_prime, 1))
                    log_alpha   = log_ratio_m + lam * (suffix_phi_proposed - suffix_phi_current)
                accepted = math.log(random.random() + 1e-10) < log_alpha
                if accepted:
                    current_traj = proposed_traj; n_accepted += 1

            if step >= burn_in:
                post_burn_count += 1
                if current_traj["home_score"] > current_traj["away_score"]:
                    win_votes += 1

        home_win_prob = win_votes / max(post_burn_count, 1)
        accept_rate   = n_accepted / max(n_steps, 1)
        result = SimResult(
            game_pk=game_pk, context_innings=cfg_sim.context_innings,
            n_simulations=post_burn_count,
            home_win_prob=home_win_prob, away_win_prob=1.0 - home_win_prob,
            tie_prob=0.0,
            mean_home_runs=float(current_traj["home_score"]),
            mean_away_runs=float(current_traj["away_score"]),
            std_home_runs=0.0, std_away_runs=0.0,
            actual_home_score=actual_home, actual_away_score=actual_away,
            actual_home_win=actual_home > actual_away,
        )
        results.append(result)
        if (game_idx + 1) % 10 == 0 or game_idx == 0:
            correct = ((result.actual_home_win and result.home_win_prob > 0.5)
                       or (not result.actual_home_win and result.away_win_prob > 0.5))
            print(f"  [{game_idx+1:4d}/{len(dataset)}] game={game_pk}  "
                  f"P(home)={home_win_prob:.3f}  accept={accept_rate:.2f}  "
                  f"{'✓' if correct else '✗'}")

    tag      = f"mh_lam{lam}_s{n_steps}_b{burn_in}"
    out_path = out_dir / f"sim_results_{cfg_sim.context_innings}inn_{tag}.json"
    with open(out_path, "w") as f:
        json.dump([vars(r) for r in results], f, indent=2)
    print(f"[mh-sim] Saved {len(results)} results to {out_path}")
    _print_summary(results)
    return results


def _simulate_one_game(
    model, encoders, pitch_scaler, batter_scaler,
    ctx_vec_1: torch.Tensor,
    start_gs: GameState,
    batting_order_raw: List[int],
    device: torch.device,
    max_pitches: int,
    prefix_phi_sum: float = 0.0,
    pt_temperature: float = 0.85,   # calibrated temperature for pitch_type_head
    oc_temperature: float = 0.85,   # calibrated temperature for outcome_head
) -> Dict:
    """
    Autoregressive simulation executing the corrected causal chain each step:
        context[t] → pitch_type → pitch_features (diffusion) → outcome → context[t+1]
    """
    oc_idx2str = {v: k for k, v in encoders.outcome.items()}

    gs          = _clone_game_state(start_gs)
    current_ctx = ctx_vec_1.squeeze(0).clone()

    half_inning_boundaries = []
    total_phi              = prefix_phi_sum
    pitch_count            = 0
    hi_phi_sum             = 0.0

    while not gs.is_game_over() and pitch_count < max_pitches:
        ctx_active = current_ctx.unsqueeze(0)  # [1, d_model]

        # Step 1: decide what pitch is thrown — calibrated pt_temperature
        pt_probs  = model.predict_next_pitch_type(ctx_active, temperature=pt_temperature)
        pt_sample = torch.multinomial(pt_probs, 1).squeeze(-1)  # [1]

        # Step 2: generate continuous pitch features (BEFORE outcome — causal fix)
        pitch_feat = model.sample_pitch_features(ctx_active, pt_sample, ddim_steps=10)

        # Step 3: outcome conditioned on physics — calibrated oc_temperature
        oc_probs  = model.predict_next_outcome(ctx_active, pt_sample, pitch_feat, temperature=oc_temperature)
        oc_sample = torch.multinomial(oc_probs, 1).squeeze(-1)  # [1]

        oc = oc_idx2str.get(oc_sample.item(), "ball")

        phi_s       = _phi(gs.outs, gs.on_1b, gs.on_2b, gs.on_3b)
        hi_phi_sum += phi_s
        total_phi  += phi_s

        prev_inning = gs.inning
        prev_is_top = gs.is_top

        terminal_event = gs.apply_pitch_outcome(oc)
        if oc in IN_PLAY_OUTCOMES:
            terminal_event = _sample_in_play_event()
        if terminal_event is not None:
            gs.apply_event(terminal_event)

        if gs.inning != prev_inning or gs.is_top != prev_is_top:
            half_inning_boundaries.append({
                "pitch_idx":      pitch_count,
                "game_state":     _clone_game_state(gs),
                "ctx_vec":        current_ctx.clone(),
                "prefix_phi_sum": total_phi,
                "hi_phi":         hi_phi_sum,
            })
            hi_phi_sum = 0.0

        if gs._check_walkoff():
            break

        # Step 4: update context for next step
        gs_feats   = torch.tensor(gs.to_feature_vec(pitch_scaler), device=device, dtype=torch.float32)
        batter_tok = batting_order_raw[gs.batting_idx] if gs.batting_idx < len(batting_order_raw) else 0
        b_ctx      = torch.tensor(
            batter_scaler.transform_row({}, BATTER_STAT_COLS), device=device, dtype=torch.float32,
        )
        new_ctx = _incremental_encode_step(
            model=model, prev_ctx=current_ctx.unsqueeze(0),
            pitch_feats=pitch_feat, gs_feats=gs_feats.unsqueeze(0),
            b_ctx=b_ctx.unsqueeze(0),
            batter_id=torch.tensor([batter_tok], device=device),
            pt_token=pt_sample.unsqueeze(0), oc_token=oc_sample.unsqueeze(0),
        )
        current_ctx = new_ctx.squeeze(0)
        pitch_count += 1

    return {
        "home_score":             gs.home_score,
        "away_score":             gs.away_score,
        "half_inning_boundaries": half_inning_boundaries,
        "total_phi":              total_phi,
        "pitch_count":            pitch_count,
    }


def _get_context_at_boundary(
    model, sample, initial_ctx_vec, boundary_info, device,
    pitch_scaler, batter_scaler, encoders, context_end_idx, max_pitches,
) -> torch.Tensor:
    return boundary_info["ctx_vec"].unsqueeze(0)


def simulate_games(cfg_sim: SimConfig, cfg_model: ModelConfig):
    device  = _resolve_device(cfg_sim.device)
    out_dir = Path(cfg_sim.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt      = torch.load(cfg_sim.checkpoint, map_location=device)
    saved_cfg = ckpt.get("cfg_model", {})
    for k, v in saved_cfg.items():
        if hasattr(cfg_model, k):
            setattr(cfg_model, k, v)

    model = TransFusion(cfg_model).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # Apply calibrated temperatures from checkpoint (fitted after training on val set).
    # cfg_model fields were already populated by the saved_cfg loop above, so
    # cfg_model.pt_temperature and cfg_model.oc_temperature now hold the fitted values.
    # We override cfg_sim.temperature so the simulation log is accurate, but pass the
    # head-specific temperatures directly to the predict_* calls below.
    pt_temp = cfg_model.pt_temperature   # for pitch_type_head
    oc_temp = cfg_model.oc_temperature   # for outcome_head
    print(f"[sim] Loaded checkpoint from {cfg_sim.checkpoint}  (epoch {ckpt.get('epoch','?')})")
    print(f"[sim] Calibrated temperatures: T_pt={pt_temp:.4f}  T_oc={oc_temp:.4f}"
          f"  (1.0 = uncalibrated)")

    builder = BaseballDatasetBuilder(
        start_dt=cfg_sim.start_dt, end_dt=cfg_sim.end_dt,
        val_start_dt=cfg_sim.val_start_dt, test_start_dt=cfg_sim.test_start_dt,
        cache_dir=cfg_sim.cache_dir, max_seq_len=cfg_model.max_seq_len,
        min_pitches_per_game=100,
    )
    train_ds, val_ds, test_ds, encoders = builder.build()
    _load_in_play_probs(cfg_sim.cache_dir)

    dataset: PitchSequenceDataset = {"train": train_ds, "val": val_ds, "test": test_ds}[cfg_sim.split]
    print(f"[sim] Simulating {len(dataset)} games from '{cfg_sim.split}' split")
    print(f"[sim] Context: {cfg_sim.context_innings} innings  |  N simulations: {cfg_sim.n_simulations}  |  Temperature: {cfg_sim.temperature}")

    results    = []
    box_scores = []
    for game_idx in range(len(dataset)):
        sample  = dataset[game_idx]
        game_pk = sample["game_pk"].item()
        game_df = dataset.game_groups[game_pk]

        context_end_idx = find_context_split(game_df, cfg_sim.context_innings)
        actual_home     = int(game_df["home_score"].iloc[-1])
        actual_away     = int(game_df["away_score"].iloc[-1])

        if context_end_idx == 0:
            ctx_batch   = _make_empty_context_batch(sample, device)
            ctx_out_idx = 0
        else:
            T_available     = sample["pitch_seq"].shape[0]
            context_end_idx = min(context_end_idx, T_available)
            ctx_batch       = _make_context_batch(sample, context_end_idx, device)
            ctx_out_idx     = context_end_idx - 1

        with torch.no_grad():
            context_memory = model.encode_context(ctx_batch)

        context_state = _reconstruct_game_state(game_df, context_end_idx)
        K       = cfg_sim.n_simulations
        ctx_vec = context_memory[:, ctx_out_idx, :].expand(K, -1)

        # Attach batter LUT to scaler so sim can do real per-batter context lookup
        dataset.batter_scaler._batter_lut_sim = dataset._batter_lut   # type: ignore[attr-defined]

        h_scores, a_scores, batter_stats, pitcher_stats = _run_parallel_simulations(
            model=model, encoders=encoders,
            pitch_scaler=dataset.pitch_scaler,
            batter_scaler=dataset.batter_scaler,
            pitcher_scaler=dataset.pitcher_scaler,
            ctx_vec=ctx_vec, game_state_template=context_state,
            sample=sample, game_df=game_df,
            context_end_idx=context_end_idx,
            K=K, device=device, max_pitches=cfg_sim.max_game_pitches,
            cfg_model=cfg_model,
            pt_temperature=pt_temp,
            oc_temperature=oc_temp,
        )

        home_wins = sum(h > a for h, a in zip(h_scores, a_scores))
        away_wins = sum(a > h for h, a in zip(h_scores, a_scores))
        ties      = sum(h == a for h, a in zip(h_scores, a_scores))

        if ties > 0:
            # Ties should be rare (only from innings cap). Log them as a diagnostic.
            import warnings as _w
            _w.warn(f"[sim] game {game_pk}: {ties}/{K} simulations ended tied "
                    f"(innings cap hit) — assigning 0.5 win prob each")

        result = SimResult(
            game_pk=game_pk, context_innings=cfg_sim.context_innings,
            n_simulations=K,
            home_win_prob=(home_wins + ties * 0.5) / K,
            away_win_prob=(away_wins + ties * 0.5) / K,
            tie_prob=ties / K,
            mean_home_runs=float(np.mean(h_scores)),
            mean_away_runs=float(np.mean(a_scores)),
            std_home_runs=float(np.std(h_scores)),
            std_away_runs=float(np.std(a_scores)),
            actual_home_score=actual_home, actual_away_score=actual_away,
            actual_home_win=actual_home > actual_away,
        )
        results.append(result)

        box = _aggregate_box_scores(
            batter_stats, pitcher_stats, game_pk, cfg_sim.context_innings, K)
        box_scores.append(box)

        if (game_idx + 1) % 10 == 0 or game_idx == 0:
            correct = ((result.actual_home_win and result.home_win_prob > 0.5)
                       or (not result.actual_home_win and result.away_win_prob > 0.5))
            print(f"  [{game_idx+1:4d}/{len(dataset)}] game={game_pk}  "
                  f"P(home)={result.home_win_prob:.3f}  "
                  f"actual={'H' if actual_home > actual_away else 'A'}  "
                  f"{'✓' if correct else '✗'}")

    out_path = out_dir / f"sim_results_{cfg_sim.context_innings}inn.json"
    with open(out_path, "w") as f:
        json.dump([vars(r) for r in results], f, indent=2, cls=_NumpyEncoder)
    print(f"[sim] Saved {len(results)} results to {out_path}")

    box_path = out_dir / f"box_scores_{cfg_sim.context_innings}inn.json"
    with open(box_path, "w") as f:
        json.dump([b.to_dict() for b in box_scores], f, indent=2, cls=_NumpyEncoder)
    print(f"[sim] Saved {len(box_scores)} box scores to {box_path}")

    # Print the first game's box score as a quick sanity check
    if box_scores:
        box_scores[0].print_table()

    _print_summary(results)
    return results, box_scores


# ---------------------------------------------------------------------------
# Vectorized simulation helpers
# ---------------------------------------------------------------------------

_IN_PLAY_EVENTS_ARR = np.array([
    "single","double","triple","home_run",
    "field_out","force_out","double_play","grounded_into_double_play",
    "field_error","sac_fly",
])
_IN_PLAY_PROBS_ARR = np.array([0.230, 0.060, 0.008, 0.040,
                                0.450, 0.080, 0.040, 0.030,
                                0.010, 0.005])
_IN_PLAY_PROBS_ARR = _IN_PLAY_PROBS_ARR / _IN_PLAY_PROBS_ARR.sum()
_IN_PLAY_CUMPROBS  = np.cumsum(_IN_PLAY_PROBS_ARR)


def _load_in_play_probs(cache_dir: str) -> None:
    global _IN_PLAY_PROBS_ARR, _IN_PLAY_CUMPROBS
    candidates = [
        Path(cache_dir) / "in_play_probs.json",
        Path(__file__).parent / "data" / "fallback_in_play_probs.json",
    ]
    for path in candidates:
        if path.exists():
            with open(path) as f:
                probs_dict = json.load(f)
            probs = np.array([probs_dict.get(e, 0.0) for e in _IN_PLAY_EVENTS_ARR], dtype=float)
            s = probs.sum()
            if s > 0:
                probs /= s
            _IN_PLAY_PROBS_ARR = probs
            _IN_PLAY_CUMPROBS  = np.cumsum(_IN_PLAY_PROBS_ARR)
            print(f"[in-play] Loaded in-play probs from {path}")
            return
    warnings.warn("[in-play] No in_play_probs.json found; using hard-coded fallback prior.")


def _vec_sample_in_play(n: int) -> np.ndarray:
    u    = np.random.rand(n)
    idxs = np.searchsorted(_IN_PLAY_CUMPROBS, u)
    idxs = np.clip(idxs, 0, len(_IN_PLAY_EVENTS_ARR) - 1)
    return _IN_PLAY_EVENTS_ARR[idxs]


def _sample_in_play_event() -> str:
    return _vec_sample_in_play(1)[0]


def _run_parallel_simulations(
    model, encoders, pitch_scaler, batter_scaler, pitcher_scaler,
    ctx_vec, game_state_template, sample, game_df, context_end_idx,
    K, device, max_pitches, cfg_model,
    pt_temperature: float = 0.85,   # calibrated temperature for pitch_type_head
    oc_temperature: float = 0.85,   # calibrated temperature for outcome_head
    temperature:    float = 0.85,   # fallback if per-head temps not provided (legacy)
) -> Tuple[List[float], List[float]]:
    """
    Vectorized K-way parallel simulation executing the corrected causal chain:
        context[t] → pitch_type → pitch_features (diffusion) → outcome → context[t+1]

    pt_temperature and oc_temperature are the post-training calibrated scalars
    loaded from the checkpoint.  They replace the single shared temperature.
    """
    gs0               = game_state_template
    batting_order_arr = np.array(sample["batting_order"].tolist(), dtype=np.int64)
    n_batters         = len(batting_order_arr)

    oc_idx2str = {v: k for k, v in encoders.outcome.items()}

    OC_BALL    = 0
    OC_STRIKE  = 1
    OC_FOUL    = 2
    OC_IN_PLAY = 3
    OC_OTHER   = 4

    def _classify_outcome(oc_str: str) -> int:
        if oc_str in BALL_OUTCOMES or oc_str in {"hit_by_pitch","automatic_ball",
                "bunt_foul_tip","foul_pitchout","pitchout","intent_ball","blocked_ball"}:
            return OC_BALL
        if oc_str in STRIKE_OUTCOMES:  return OC_STRIKE
        if oc_str in FOUL_OUTCOMES:    return OC_FOUL
        if oc_str in IN_PLAY_OUTCOMES: return OC_IN_PLAY
        return OC_OTHER

    oc_class = np.array(
        [_classify_outcome(oc_idx2str.get(i, "ball")) for i in range(max(oc_idx2str.keys()) + 1)],
        dtype=np.int32,
    )

    inning      = np.full(K, gs0.inning,      dtype=np.int32)
    is_top      = np.full(K, gs0.is_top,      dtype=bool)
    outs        = np.full(K, gs0.outs,        dtype=np.int32)
    balls       = np.full(K, gs0.balls,       dtype=np.int32)
    strikes     = np.full(K, gs0.strikes,     dtype=np.int32)
    on_1b       = np.full(K, gs0.on_1b,       dtype=bool)
    on_2b       = np.full(K, gs0.on_2b,       dtype=bool)
    on_3b       = np.full(K, gs0.on_3b,       dtype=bool)
    home_score  = np.full(K, gs0.home_score,  dtype=np.int32)
    away_score  = np.full(K, gs0.away_score,  dtype=np.int32)
    batting_idx = np.full(K, gs0.batting_idx, dtype=np.int32)

    active_mask  = np.ones(K, dtype=bool)
    walkoff_mask = np.zeros(K, dtype=bool)
    current_ctx  = ctx_vec.clone()
    pitch_count  = 0

    # ── Box score tracking ────────────────────────────────────────────────
    # batter_stats[k, team, slot, stat] where team 0=away, 1=home
    # stat order: AB, H, HR, 2B, 3B, BB, HBP, K, RBI  (9 stats)
    N_BSTATS = 9
    BS_AB=0; BS_H=1; BS_HR=2; BS_2B=3; BS_3B=4; BS_BB=5; BS_HBP=6; BS_K=7; BS_RBI=8
    batter_stats  = np.zeros((K, 2, 9, N_BSTATS), dtype=np.float32)

    # pitcher_stats[k, team, stat] where team 0=away, 1=home
    # stat order: outs_recorded, H, HR, BB, K, R  (6 stats)
    N_PSTATS = 6
    PS_OUTS=0; PS_H=1; PS_HR=2; PS_BB=3; PS_K=4; PS_R=5
    pitcher_stats = np.zeros((K, 2, N_PSTATS), dtype=np.float32)

    # Track which base each batting-slot runner occupies per sim.
    # runner_slot[k, base] = batting slot index (-1 = no runner), base 0=1b,1=2b,2=3b
    runner_slot = np.full((K, 3), -1, dtype=np.int32)
    # ─────────────────────────────────────────────────────────────────────

    # ── Real batter context lookup — replace the zero imputation ────────────
    # Look up each lineup slot's season stats from the dataset's batter LUT.
    # game_year is taken from the first pitch in the game_df so we get the
    # correct season's stats, not a cross-season average.
    game_year = int(game_df["game_year"].iloc[0]) if "game_year" in game_df.columns else 2025
    batter_ctx_cache: Dict[int, np.ndarray] = {}

    _batter_lut_for_sim = getattr(batter_scaler, '_batter_lut_sim', None)

    def _get_batter_ctx_np(mlbam_id: int) -> np.ndarray:
        """Return scaled batter feature vector, with dict cache."""
        if mlbam_id not in batter_ctx_cache:
            row_dict = (
                _batter_lut_for_sim.get((mlbam_id, game_year), {})
                if _batter_lut_for_sim is not None else {}
            )
            batter_ctx_cache[mlbam_id] = batter_scaler.transform_row(row_dict, BATTER_STAT_COLS)
        return batter_ctx_cache[mlbam_id]

    # Pre-compute ctx for each lineup slot (0-indexed)
    batting_order_ids = sample["batting_order"].tolist()   # list of MLBAM IDs
    slot_ctx = np.stack([
        _get_batter_ctx_np(int(bid)) if bid != 0 else
        batter_scaler.transform_row({}, BATTER_STAT_COLS)
        for bid in batting_order_ids
    ])  # [9, batter_feat_dim]
    slot_ctx_tensor = torch.tensor(slot_ctx, dtype=torch.float32, device=device)  # [9, F_b]

    # League-average reliever context — used when pitcher_ctx switches in extras
    # or after TTO threshold. Represents a "generic MLB reliever."
    league_avg_pitcher_ctx = torch.zeros(1, cfg_model.pitcher_feat_dim, device=device)

    zero_b_ctx = torch.tensor(
        batter_scaler.transform_row({}, BATTER_STAT_COLS), dtype=torch.float32, device=device,
    )

    gs_cols = GAME_STATE_COLS
    gs_mean = np.array([pitch_scaler.mean_.get(c, 0.0) for c in gs_cols], dtype=np.float32)
    gs_std  = np.array([pitch_scaler.std_.get(c, 1.0)  for c in gs_cols], dtype=np.float32)
    gs_std  = np.where(gs_std < 1e-6, 1.0, gs_std)
    _ci     = {c: i for i, c in enumerate(gs_cols)}

    def _build_gs_feats_batch(active: np.ndarray) -> torch.Tensor:
        n   = len(active)
        raw = np.zeros((n, len(gs_cols)), dtype=np.float32)
        raw[:, _ci["balls"]]        = balls[active]
        raw[:, _ci["strikes"]]      = strikes[active]
        raw[:, _ci["outs_when_up"]] = outs[active]
        raw[:, _ci["inning"]]       = inning[active]
        raw[:, _ci["home_score"]]   = home_score[active]
        raw[:, _ci["away_score"]]   = away_score[active]
        raw[:, _ci["on_1b"]]        = on_1b[active].astype(np.float32)
        raw[:, _ci["on_2b"]]        = on_2b[active].astype(np.float32)
        raw[:, _ci["on_3b"]]        = on_3b[active].astype(np.float32)
        raw[:, _ci["run_diff"]]     = (home_score[active] - away_score[active]).astype(np.float32)
        return torch.tensor((raw - gs_mean) / gs_std, dtype=torch.float32, device=device)

    while np.any(active_mask) and pitch_count < max_pitches:
        active = np.where(active_mask)[0]
        n_act  = len(active)

        ctx_active = current_ctx[active]  # [n_act, d]

        # Step 1: sample pitch type — uses calibrated pt_temperature
        pt_probs   = model.predict_next_pitch_type(ctx_active, temperature=pt_temperature)
        pt_samples = torch.multinomial(pt_probs, 1).squeeze(-1)  # [n_act]

        # Step 2: generate continuous pitch features (BEFORE outcome — causal fix)
        pitch_feats = model.sample_pitch_features(
            ctx_active, pt_samples, ddim_steps=10)  # [n_act, F_cont]

        # Step 3: sample outcome — uses calibrated oc_temperature, conditioned on physics
        oc_probs   = model.predict_next_outcome(ctx_active, pt_samples, pitch_feats, temperature=oc_temperature)
        oc_samples = torch.multinomial(oc_probs, 1).squeeze(-1)  # [n_act]

        oc_np         = oc_samples.cpu().numpy()
        oc_np_clipped = np.clip(oc_np, 0, len(oc_class) - 1)
        cls           = oc_class[oc_np_clipped]

        is_ball    = cls == OC_BALL
        is_strike  = cls == OC_STRIKE
        is_foul    = cls == OC_FOUL
        is_in_play = cls == OC_IN_PLAY

        balls[active]   += is_ball.astype(np.int32)
        strikes[active] += is_strike.astype(np.int32)
        foul_adds        = is_foul & (strikes[active] < 2)
        strikes[active] += foul_adds.astype(np.int32)

        walk_from_ball = is_ball   & (balls[active]   >= 4)
        ko_from_strike = is_strike & (strikes[active] >= 3)

        n_in_play = int(is_in_play.sum())
        if n_in_play > 0 and in_play_mlp is not None:
            # Conditional in-play sampling: conditions on batter features and pitch physics
            ip_b_idxs  = batting_idx[active][is_in_play] % n_batters
            ip_b_ctx   = slot_ctx_tensor[ip_b_idxs]                   # [n_ip, F_batter]
            ip_g_ctx   = torch.zeros(n_in_play, cfg_model.game_feat_dim, device=device)
            ip_pf      = pitch_feats[is_in_play]                       # [n_ip, n_cont]
            ip_idxs    = in_play_mlp.sample(ip_b_ctx, ip_g_ctx, ip_pf)
            in_play_events = _IN_PLAY_EVENTS_ARR[ip_idxs]
        elif n_in_play > 0:
            in_play_events = _vec_sample_in_play(n_in_play)
        else:
            in_play_events = np.array([])
        in_play_ptr = 0

        for local_i in range(n_act):
            sim_i  = active[local_i]
            team_i = 0 if is_top[sim_i] else 1   # batting team: 0=away, 1=home
            pit_i  = 1 - team_i                   # pitching team (opposite)
            slot_i = batting_idx[sim_i] % 9       # current batter's order slot

            if walk_from_ball[local_i]:
                # --- credit BB to batter, advance runners, credit RBI if run scores ---
                batter_stats[sim_i, team_i, slot_i, BS_BB] += 1
                runs_before = home_score[sim_i] + away_score[sim_i]
                _vec_apply_walk(sim_i, inning, is_top, home_score, away_score,
                                on_1b, on_2b, on_3b, walkoff_mask)
                runs_scored = (home_score[sim_i] + away_score[sim_i]) - runs_before
                batter_stats[sim_i, team_i, slot_i, BS_RBI] += runs_scored
                pitcher_stats[sim_i, pit_i, PS_BB]  += 1
                pitcher_stats[sim_i, pit_i, PS_R]   += runs_scored
                # advance runner_slot: bases-loaded walk pushes 3b runner home
                if on_1b[sim_i] and runner_slot[sim_i, 0] >= 0:
                    # push chain: 1b→2b→3b→home
                    if on_2b[sim_i]:
                        if on_3b[sim_i]:
                            runner_slot[sim_i, 2] = -1   # 3b runner scored
                        runner_slot[sim_i, 2] = runner_slot[sim_i, 1]
                    runner_slot[sim_i, 1] = runner_slot[sim_i, 0]
                runner_slot[sim_i, 0] = slot_i   # batter now on 1b
                balls[sim_i] = strikes[sim_i] = 0
                batting_idx[sim_i] = (batting_idx[sim_i] + 1) % 9

            elif ko_from_strike[local_i]:
                # --- credit K to batter and pitcher ---
                batter_stats[sim_i, team_i, slot_i, BS_AB] += 1
                batter_stats[sim_i, team_i, slot_i, BS_K]  += 1
                pitcher_stats[sim_i, pit_i, PS_K]    += 1
                pitcher_stats[sim_i, pit_i, PS_OUTS] += 1
                outs[sim_i] += 1
                balls[sim_i] = strikes[sim_i] = 0
                batting_idx[sim_i] = (batting_idx[sim_i] + 1) % 9
                if outs[sim_i] >= 3:
                    _vec_end_half_inning(sim_i, inning, is_top, outs,
                                         on_1b, on_2b, on_3b, balls, strikes)
                    runner_slot[sim_i, :] = -1   # clear runners on half-inning end

            elif is_in_play[local_i]:
                ev = in_play_events[in_play_ptr]; in_play_ptr += 1
                # --- resolve event stats before mutating game state ---
                batter_stats[sim_i, team_i, slot_i, BS_AB] += 1
                outs_before  = outs[sim_i]
                score_before = home_score[sim_i] + away_score[sim_i]

                _vec_apply_event_with_stats(
                    sim_i, ev, inning, is_top, outs,
                    on_1b, on_2b, on_3b, home_score, away_score,
                    balls, strikes, batting_idx, walkoff_mask,
                    batter_stats, pitcher_stats, runner_slot,
                    team_i, pit_i, slot_i,
                    BS_H, BS_HR, BS_2B, BS_3B, BS_RBI,
                    PS_OUTS, PS_H, PS_HR, PS_R,
                )
                # clear runners if half-inning ended
                if outs[sim_i] < outs_before or (outs[sim_i] == 0 and outs_before >= 3):
                    runner_slot[sim_i, :] = -1

        # Step 4: update context for all active sims
        gs_feats    = _build_gs_feats_batch(active)
        b_idxs      = batting_idx[active] % n_batters
        batter_toks = torch.tensor(
            [batting_order_arr[b] for b in b_idxs], dtype=torch.long, device=device)
        # Real per-batter context from lineup slot — replaces zero imputation
        b_ctx_batch = slot_ctx_tensor[b_idxs]   # [n_act, F_batter]

        new_ctxs = _incremental_encode_step_batch(
            model=model, prev_ctx=ctx_active,
            pitch_feats=pitch_feats, gs_feats=gs_feats,
            b_ctx=b_ctx_batch, batter_ids=batter_toks,
            pt_tokens=pt_samples, oc_tokens=oc_samples,
        )
        current_ctx[active] = new_ctxs

        # Game ends when: (a) scores differ at start of top of extra inning
        # (away team won the bottom half), or (b) hard innings cap reached to
        # prevent infinite tied extra-inning simulations. When the cap fires on
        # a tie, we break the tie by adding 1 run for the home team (home field
        # advantage tiebreaker — approximates the ~54% home win rate in extras).
        regulation_over = (inning > 9) & is_top & (home_score != away_score)
        innings_capped  = (inning > 20)   # hard cap: 20 innings max
        # Break ties at cap: home team wins the coin flip
        tie_at_cap = innings_capped & (home_score == away_score)
        home_score[tie_at_cap] += 1   # home team wins tied capped games
        game_over   = regulation_over | innings_capped
        active_mask = ~game_over & ~walkoff_mask
        pitch_count += 1

    return home_score.tolist(), away_score.tolist(), batter_stats, pitcher_stats


def _vec_apply_walk(sim_i, inning, is_top, home_score, away_score,
                    on_1b, on_2b, on_3b, walkoff_mask):
    if on_1b[sim_i] and on_2b[sim_i] and on_3b[sim_i]:
        runs = 1
    elif on_1b[sim_i] and on_2b[sim_i]:
        on_3b[sim_i] = True; runs = 0
    elif on_1b[sim_i]:
        on_2b[sim_i] = True; runs = 0
    else:
        runs = 0
    on_1b[sim_i] = True
    if runs > 0:
        if is_top[sim_i]:
            away_score[sim_i] += runs
        else:
            home_score[sim_i] += runs
            if inning[sim_i] >= 9 and home_score[sim_i] > away_score[sim_i]:
                walkoff_mask[sim_i] = True


def _vec_end_half_inning(sim_i, inning, is_top, outs, on_1b, on_2b, on_3b, balls, strikes):
    outs[sim_i]  = 0
    on_1b[sim_i] = on_2b[sim_i] = on_3b[sim_i] = False
    balls[sim_i] = strikes[sim_i] = 0
    if is_top[sim_i]:
        is_top[sim_i] = False
    else:
        inning[sim_i] += 1
        is_top[sim_i]  = True
        if inning[sim_i] > 9:
            on_2b[sim_i] = True


def _vec_apply_event(sim_i, event_str, inning, is_top, outs,
                      on_1b, on_2b, on_3b, home_score, away_score,
                      balls, strikes, batting_idx, walkoff_mask):
    outs_added, base_runs, code = EVENT_TABLE.get(event_str, (1, 0, "out"))
    runs = base_runs
    if code == "out":
        outs[sim_i] += outs_added
        if outs[sim_i] >= 3:
            _vec_end_half_inning(sim_i, inning, is_top, outs, on_1b, on_2b, on_3b, balls, strikes)
    else:
        bases = {"single":1,"double":2,"triple":3,"hr":4,"walk":1,"hbp":1}.get(code, 0)
        new_1b = new_2b = new_3b = False
        if bases == 1:
            if on_3b[sim_i]: runs += 1
            if on_2b[sim_i]: new_3b = True
            if on_1b[sim_i]: new_2b = True
            new_1b = True
        elif bases == 2:
            if on_3b[sim_i]: runs += 1
            if on_2b[sim_i]: runs += 1
            if on_1b[sim_i]: new_3b = True
            new_2b = True
        elif bases == 3:
            if on_3b[sim_i]: runs += 1
            if on_2b[sim_i]: runs += 1
            if on_1b[sim_i]: runs += 1
            new_3b = True
        elif bases == 4:
            if on_3b[sim_i]: runs += 1
            if on_2b[sim_i]: runs += 1
            if on_1b[sim_i]: runs += 1
            runs += 1
        on_1b[sim_i], on_2b[sim_i], on_3b[sim_i] = new_1b, new_2b, new_3b
        if is_top[sim_i]:
            away_score[sim_i] += runs
        else:
            home_score[sim_i] += runs
            if inning[sim_i] >= 9 and home_score[sim_i] > away_score[sim_i]:
                walkoff_mask[sim_i] = True
        balls[sim_i] = strikes[sim_i] = 0
    batting_idx[sim_i] = (batting_idx[sim_i] + 1) % 9


def _vec_apply_event_with_stats(
    sim_i, event_str, inning, is_top, outs,
    on_1b, on_2b, on_3b, home_score, away_score,
    balls, strikes, batting_idx, walkoff_mask,
    batter_stats, pitcher_stats, runner_slot,
    team_i, pit_i, slot_i,
    BS_H, BS_HR, BS_2B, BS_3B, BS_RBI,
    PS_OUTS, PS_H, PS_HR, PS_R,
):
    """
    Apply an in-play event and credit stats to the appropriate batter/pitcher.
    Mutates all the game-state arrays in-place (same as _vec_apply_event)
    and additionally updates batter_stats / pitcher_stats / runner_slot.
    """
    outs_added, base_runs, code = EVENT_TABLE.get(event_str, (1, 0, "out"))

    if code == "out":
        # No hit, count outs and pitcher outs-recorded
        pitcher_stats[sim_i, pit_i, PS_OUTS] += outs_added
        outs[sim_i] += outs_added
        if outs[sim_i] >= 3:
            _vec_end_half_inning(sim_i, inning, is_top, outs, on_1b, on_2b, on_3b, balls, strikes)
        balls[sim_i] = strikes[sim_i] = 0
        batting_idx[sim_i] = (batting_idx[sim_i] + 1) % 9

    else:
        # Hit: credit H (and HR/2B/3B) to batter, H to pitcher
        is_hr = (code == "hr")
        is_2b = (event_str in ("double",))
        is_3b = (event_str in ("triple",))
        if code in ("single","double","triple","hr"):
            batter_stats[sim_i, team_i, slot_i, BS_H]  += 1
            pitcher_stats[sim_i, pit_i, PS_H]           += 1
            if is_hr:
                batter_stats[sim_i, team_i, slot_i, BS_HR] += 1
                pitcher_stats[sim_i, pit_i, PS_HR]          += 1
            if is_2b:
                batter_stats[sim_i, team_i, slot_i, BS_2B] += 1
            if is_3b:
                batter_stats[sim_i, team_i, slot_i, BS_3B] += 1

        # Advance runners and count runs/RBIs
        bases = {"single":1,"double":2,"triple":3,"hr":4,"walk":1,"hbp":1}.get(code, 0)
        runs  = base_runs
        new_1b = new_2b = new_3b = False
        new_runner_slot = np.array([-1, -1, -1], dtype=np.int32)

        if bases == 1:
            if on_3b[sim_i]:
                runs += 1
                # runner on 3b scores — runner_slot[2] scored
            if on_2b[sim_i]:
                new_3b = True
                new_runner_slot[2] = runner_slot[sim_i, 1]
            if on_1b[sim_i]:
                new_2b = True
                new_runner_slot[1] = runner_slot[sim_i, 0]
            new_1b = True
            new_runner_slot[0] = slot_i
        elif bases == 2:
            if on_3b[sim_i]: runs += 1
            if on_2b[sim_i]: runs += 1
            if on_1b[sim_i]:
                new_3b = True
                new_runner_slot[2] = runner_slot[sim_i, 0]
            new_2b = True
            new_runner_slot[1] = slot_i
        elif bases == 3:
            if on_3b[sim_i]: runs += 1
            if on_2b[sim_i]: runs += 1
            if on_1b[sim_i]: runs += 1
            new_3b = True
            new_runner_slot[2] = slot_i
        elif bases == 4:
            # HR: everyone scores including batter
            if on_3b[sim_i]: runs += 1
            if on_2b[sim_i]: runs += 1
            if on_1b[sim_i]: runs += 1
            runs += 1  # batter

        on_1b[sim_i], on_2b[sim_i], on_3b[sim_i] = new_1b, new_2b, new_3b
        runner_slot[sim_i, :] = new_runner_slot

        # RBIs = runs scored on this PA (batter gets credit)
        batter_stats[sim_i, team_i, slot_i, BS_RBI] += runs

        if is_top[sim_i]:
            away_score[sim_i] += runs
        else:
            home_score[sim_i] += runs
            if inning[sim_i] >= 9 and home_score[sim_i] > away_score[sim_i]:
                walkoff_mask[sim_i] = True

        pitcher_stats[sim_i, pit_i, PS_R] += runs
        balls[sim_i] = strikes[sim_i] = 0
        pitcher_stats[sim_i, pit_i, PS_OUTS] += 1 if outs_added > 0 else 0
        batting_idx[sim_i] = (batting_idx[sim_i] + 1) % 9


def _incremental_encode_step_batch(
    model, prev_ctx, pitch_feats, gs_feats, b_ctx, batter_ids, pt_tokens, oc_tokens,
    kv_caches: Optional[List["KVCache"]] = None,
    pitcher_latent: Optional[torch.Tensor] = None,
    batter_latent:  Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Incremental context update for a batch of parallel simulations.

    When kv_caches is provided (list of K KVCache objects, one per sim),
    each cache is advanced through the full Transformer stack — giving proper
    causal attention over all prior pitches rather than a residual approximation.

    When kv_caches is None (legacy path), falls back to the residual addition.
    """
    enc     = model.encoder
    b_emb   = enc.batter_emb(batter_ids)
    pt_emb  = enc.ptype_emb(pt_tokens)
    step_in = torch.cat([pitch_feats, gs_feats, b_ctx, b_emb, pt_emb], dim=-1)
    x_proj  = enc.step_proj(step_in) + prev_ctx   # [n_act, d_model]

    if kv_caches is None or pitcher_latent is None or batter_latent is None:
        # Legacy fallback: residual addition (fast but approximate)
        return enc.out_norm(x_proj)

    # Full incremental attention: process each simulation through its own KV cache.
    # This is O(n_act × T_cache) vs O(T²) for a full forward pass.
    outputs = []
    for i, cache in enumerate(kv_caches):
        x_i   = x_proj[i:i+1].unsqueeze(1)    # [1, 1, d_model]
        pl_i  = pitcher_latent[i:i+1]          # [1, pitcher_latent_dim]
        bl_i  = batter_latent[i:i+1]           # [1, batter_latent_dim]
        out_i = enc.forward_incremental(x_i, pl_i, bl_i, cache)
        outputs.append(out_i.squeeze(1))        # [1, d_model]
    return torch.cat(outputs, dim=0)            # [n_act, d_model]


def _incremental_encode_step(
    model, prev_ctx, pitch_feats, gs_feats, b_ctx, batter_id, pt_token, oc_token,
) -> torch.Tensor:
    enc     = model.encoder
    b_emb   = enc.batter_emb(batter_id)
    pt_emb  = enc.ptype_emb(pt_token)
    # oc_token kept in signature for API compatibility but not concatenated
    step_in = torch.cat([pitch_feats, gs_feats, b_ctx, b_emb, pt_emb], dim=-1)
    new_ctx = enc.step_proj(step_in) + prev_ctx
    return enc.out_norm(new_ctx)


def _reconstruct_game_state(game_df, context_end_idx: int) -> GameState:
    gs = GameState()
    if context_end_idx == 0:
        return gs
    for _, row in game_df.iloc[:context_end_idx].iterrows():
        event   = row.get("events", None)
        outcome = str(row.get("description", "ball"))
        if event is not None and str(event) != "nan":
            gs.apply_event(str(event))
        else:
            gs.apply_pitch_outcome(outcome)
    return gs


def _clone_game_state(gs: GameState) -> GameState:
    new = GameState()
    new.__dict__.update(gs.__dict__.copy())
    return new


def _make_context_batch(sample, context_end_idx, device):
    keys_to_slice = ["pitch_seq", "pitch_types", "outcomes", "at_bat_events",
                     "batter_ctx", "batter_ids", "mask"]
    batch = {k: sample[k][:context_end_idx].unsqueeze(0).to(device) for k in keys_to_slice}
    for k in ["pitcher_ctx", "pitcher_id", "batting_order", "game_ctx"]:
        batch[k] = sample[k].unsqueeze(0).to(device)
    return batch


def _make_empty_context_batch(sample, device):
    batch = {}
    for k in ["pitch_seq", "batter_ctx"]:
        dim      = sample[k].shape[-1]
        batch[k] = torch.zeros(1, 1, dim, device=device)
    for k in ["pitch_types", "outcomes", "at_bat_events", "batter_ids"]:
        batch[k] = torch.zeros(1, 1, dtype=torch.long, device=device)
    batch["mask"] = torch.ones(1, 1, dtype=torch.bool, device=device)
    for k in ["pitcher_ctx", "pitcher_id", "batting_order", "game_ctx"]:
        batch[k] = sample[k].unsqueeze(0).to(device)
    return batch


# =============================================================================
# 7.  EVALUATION SUMMARY
# =============================================================================

def _print_summary(results: List[SimResult]):
    """
    Print simulation evaluation metrics.

    Includes:
      • Accuracy at multiple probability thresholds (not just 0.5)
      • Brier score and log-loss
      • ECE (Expected Calibration Error) — 10-bin reliability diagram values
      • Mean predicted home win probability (calibration sanity check)
    """
    n = len(results)
    if n == 0:
        return

    labeled = [r for r in results if r.actual_home_win is not None]
    probs   = np.array([r.home_win_prob for r in labeled])
    actuals = np.array([float(r.actual_home_win) for r in labeled])

    # ── Standard metrics ──────────────────────────────────────────────────
    brier = float(np.mean((probs - actuals) ** 2))
    ll    = float(np.mean([-math.log(max(p if a else 1-p, 1e-7))
                            for p, a in zip(probs, actuals)]))

    # ── Accuracy at multiple thresholds ───────────────────────────────────
    # Hard threshold of 0.5 is misleading for probabilistic models; also
    # report at 0.55 and 0.60 which reflect higher-confidence predictions.
    print("\n" + "=" * 60)
    print(f"  SIMULATION SUMMARY  ({n} games,  {len(labeled)} labeled)")
    print("=" * 60)
    print(f"  Brier Score     : {brier:.4f}")
    print(f"  Log Loss        : {ll:.4f}")
    for thresh in [0.50, 0.55, 0.60]:
        confident = [(p, a) for p, a in zip(probs, actuals)
                     if p >= thresh or p <= 1 - thresh]
        if confident:
            ps, as_ = zip(*confident)
            acc = np.mean([(p >= thresh) == bool(a) for p, a in zip(ps, as_)])
            print(f"  Accuracy @{thresh:.2f}    : {acc:.4f}  ({len(confident)}/{len(labeled)} games)")

    # ── ECE: Expected Calibration Error ───────────────────────────────────
    # Divide predictions into 10 equal-width bins. For each bin compute
    # |mean_predicted - mean_actual|.  ECE = weighted average of these gaps.
    n_bins = 10
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece_sum, total_weight = 0.0, 0
    print(f"\n  Reliability diagram (P_pred → P_actual):")
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i+1]
        mask   = (probs >= lo) & (probs < hi) if i < n_bins - 1 else (probs >= lo)
        if mask.sum() == 0:
            continue
        p_mean = float(probs[mask].mean())
        a_mean = float(actuals[mask].mean())
        count  = int(mask.sum())
        gap    = abs(p_mean - a_mean)
        ece_sum      += gap * count
        total_weight += count
        bar = "█" * int(gap * 40)
        print(f"    [{lo:.1f}–{hi:.1f}]  n={count:4d}  pred={p_mean:.3f}  "
              f"actual={a_mean:.3f}  gap={gap:.3f}  {bar}")

    ece = ece_sum / max(total_weight, 1)
    print(f"\n  ECE             : {ece:.4f}  (lower = better calibrated)")
    print(f"  Mean P(home win): {probs.mean():.4f}")

    tie_rate = np.mean([r.tie_prob for r in results])
    if tie_rate > 0:
        print(f"  Tie rate        : {tie_rate:.4f}  (innings cap hits — should be <0.001)")
    print("=" * 60 + "\n")


# =============================================================================
# 8.  UTILITIES
# =============================================================================

class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder that converts numpy scalars and arrays to Python natives."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


def _resolve_device(device_str: str) -> torch.device:
    if device_str == "auto":
        if torch.cuda.is_available():         return torch.device("cuda")
        if torch.backends.mps.is_available(): return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_str)


def _save_json(obj: dict, path: Path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


# =============================================================================
# 9.  CLI
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="TransFusion Baseball Model")
    sub    = parser.add_subparsers(dest="command")

    p_train = sub.add_parser("train")
    p_train.add_argument("--cache_dir",      default="./baseball_cache")
    p_train.add_argument("--checkpoint_dir", default="./checkpoints")
    p_train.add_argument("--start_dt",       default="2015-03-22")
    p_train.add_argument("--end_dt",         default="2026-05-01")
    p_train.add_argument("--val_start_dt",   default="2025-03-20")
    p_train.add_argument("--test_start_dt",  default="2026-03-25")
    p_train.add_argument("--epochs",         type=int,   default=30)
    p_train.add_argument("--batch_size",     type=int,   default=32)
    p_train.add_argument("--lr",             type=float, default=3e-4)
    p_train.add_argument("--d_model",        type=int,   default=256)
    p_train.add_argument("--n_layers",       type=int,   default=8)
    p_train.add_argument("--n_heads",        type=int,   default=16)
    p_train.add_argument("--n_diff_steps",        type=int,   default=30)
    p_train.add_argument("--pitcher_latent_dim",  type=int,   default=64)
    p_train.add_argument("--batter_latent_dim",   type=int,   default=64)
    p_train.add_argument("--lambda_vae",          type=float, default=0.10)
    p_train.add_argument("--phase2_start",        type=int,   default=4,
                         help="Epoch to unfreeze encoder+pitch/diffusion heads")
    p_train.add_argument("--phase3_start",        type=int,   default=13,
                         help="Epoch to unfreeze outcome head (full joint training)")
    p_train.add_argument("--num_workers",    type=int,   default=0)
    p_train.add_argument("--device",         default="auto")
    p_train.add_argument("--seed",           type=int,   default=42)

    p_sim = sub.add_parser("simulate")
    p_sim.add_argument("--checkpoint",      required=True)
    p_sim.add_argument("--cache_dir",       default="./baseball_cache")
    p_sim.add_argument("--start_dt",        default="2015-03-22")
    p_sim.add_argument("--end_dt",          default="2026-05-01")
    p_sim.add_argument("--val_start_dt",    default="2025-03-20")
    p_sim.add_argument("--test_start_dt",   default="2026-03-25")
    p_sim.add_argument("--context_innings", type=float, default=3.0)
    p_sim.add_argument("--n_simulations",   type=int,   default=500)
    p_sim.add_argument("--split",           default="test", choices=["train", "val", "test"])
    p_sim.add_argument("--out_dir",         default="./sim_results")
    p_sim.add_argument("--device",          default="auto")
    p_sim.add_argument("--mh",              action="store_true")
    p_sim.add_argument("--lam",             type=float, default=1.0)
    p_sim.add_argument("--n_steps",         type=int,   default=500)
    p_sim.add_argument("--burn_in",         type=int,   default=100)
    p_sim.add_argument("--temperature",     type=float, default=0.85)

    p_mlp = sub.add_parser("train_in_play_mlp")
    p_mlp.add_argument("--cache_dir",    default="./baseball_cache")
    p_mlp.add_argument("--train_end_dt", default="2026-02-01")
    p_mlp.add_argument("--epochs",       type=int, default=20)
    p_mlp.add_argument("--hidden_dim",   type=int, default=64)
    p_mlp.add_argument("--device",       default="auto")

    return parser.parse_args()


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    args = parse_args()

    if args.command == "train":
        cfg_train = TrainConfig(
            cache_dir=args.cache_dir, checkpoint_dir=args.checkpoint_dir,
            start_dt=args.start_dt, end_dt=args.end_dt,
            val_start_dt=args.val_start_dt, test_start_dt=args.test_start_dt,
            epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
            num_workers=args.num_workers, device=args.device, seed=args.seed,
        )
        cfg_model = ModelConfig(
            d_model=args.d_model, n_layers=args.n_layers, n_heads=args.n_heads,
            n_diffusion_steps=args.n_diff_steps,
            pitcher_latent_dim=args.pitcher_latent_dim,
            batter_latent_dim=args.batter_latent_dim,
            lambda_vae=args.lambda_vae,
        )
        train(cfg_train, cfg_model)

    elif args.command == "simulate":
        cfg_sim = SimConfig(
            checkpoint=args.checkpoint, cache_dir=args.cache_dir,
            start_dt=args.start_dt, end_dt=args.end_dt,
            val_start_dt=args.val_start_dt, test_start_dt=args.test_start_dt,
            context_innings=args.context_innings, n_simulations=args.n_simulations,
            split=args.split, out_dir=args.out_dir, device=args.device,
            temperature=args.temperature,
        )
        if args.mh:
            simulate_games_mh(cfg_sim, ModelConfig(),
                               lam=args.lam, n_steps=args.n_steps, burn_in=args.burn_in)
        else:
            simulate_games(cfg_sim, ModelConfig())

    elif args.command == "train_in_play_mlp":
        train_in_play_mlp(
            cache_dir=args.cache_dir,
            train_end_dt=args.train_end_dt,
            epochs=args.epochs,
            hidden_dim=args.hidden_dim,
            device_str=args.device,
        )

    else:
        print("Usage: python new_transfusion.py [train|simulate|train_in_play_mlp] --help")