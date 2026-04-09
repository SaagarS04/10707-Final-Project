"""
TransFusion: Baseball Pitch Sequence Model + MCMC Game Simulator
================================================================
Architecture:
    TransFusion = Transformer encoder (context) + Diffusion decoder (continuous pitch features)
                + dual classification heads (pitch type, pitch outcome)

Training:
    Given a game sequence, the model learns to denoise the next pitch's continuous
    features (diffusion loss) while simultaneously classifying the next pitch type
    and the current pitch outcome (cross-entropy losses).

Simulation (MCMC):
    Given N context innings of a real game, autoregressively simulate the remainder
    pitch-by-pitch. At each step:
        1. Sample pitch_type from the model's categorical distribution
        2. Denoise a continuous pitch feature vector conditioned on context + pitch_type
        3. Sample outcome from the outcome head
        4. Advance game state (count, outs, runners, score) deterministically
        5. Repeat until 27 outs (or extras)
    Run K parallel simulations → aggregate win probabilities.

Usage:
    # Train
    python transfusion.py train \
        --cache_dir ./baseball_cache \
        --checkpoint_dir ./checkpoints \
        --epochs 40 \
        --batch_size 16

    # Simulate (specify context innings)
    python transfusion.py simulate \
        --checkpoint ./checkpoints/best.pt \
        --cache_dir ./baseball_cache \
        --context_innings 3.0 \
        --n_simulations 500 \
        --split test \
        --out_dir ./sim_results
"""

import argparse
import json
import math
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# ── Import dataset builder (must be in same directory or on PYTHONPATH) ──────
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


# =============================================================================
# 0.  CONFIG
# =============================================================================

@dataclass
class ModelConfig:
    # Input dims (overridden from dataset at runtime)
    pitch_feat_dim:   int = 30    # len(PITCH_CONTINUOUS_COLS + GAME_STATE_COLS)
    pitcher_feat_dim: int = 17    # len(PITCHER_STAT_COLS)
    batter_feat_dim:  int = 14    # len(BATTER_STAT_COLS)
    game_feat_dim:    int = 3     # len(GAME_CTX_COLS)

    # Vocab sizes (overridden from encoders at runtime)
    num_pitch_types: int = 20
    num_outcomes:    int = 30
    num_events:      int = 25
    num_batters:     int = 2000
    num_pitchers:    int = 1000

    # Transformer
    d_model:       int = 256
    n_heads:       int = 8
    n_layers:      int = 6
    d_ff:          int = 1024
    dropout:       float = 0.1
    max_seq_len:   int = 400

    # Diffusion
    n_diffusion_steps: int = 50
    beta_start:        float = 1e-4
    beta_end:          float = 0.02

    # Embeddings
    embed_dim: int = 64   # for batter/pitcher ID embeddings

    # Loss weights
    lambda_diffusion:  float = 1.0
    lambda_pitch_type: float = 0.5
    lambda_outcome:    float = 0.5


@dataclass
class TrainConfig:
    cache_dir:      str  = "./baseball_cache"
    checkpoint_dir: str  = "./checkpoints"
    start_dt:       str  = "2022-04-07"
    end_dt:         str  = "2024-11-01"
    val_start_dt:   str  = "2024-03-20"
    test_start_dt:  str  = "2024-10-01"
    epochs:         int  = 40
    batch_size:     int  = 16
    lr:             float = 3e-4
    weight_decay:   float = 1e-4
    grad_clip:      float = 1.0
    warmup_steps:   int  = 500
    log_every:      int  = 50
    val_every:      int  = 1       # validate every N epochs
    num_workers:    int  = 4
    seed:           int  = 42
    device:         str  = "auto"


@dataclass
class SimConfig:
    checkpoint:       str   = "./checkpoints/best.pt"
    cache_dir:        str   = "./baseball_cache"
    context_innings:  float = 3.0    # 0, 0.5, 1.0, ..., 8.5
    n_simulations:    int   = 500
    split:            str   = "test" # train / val / test
    out_dir:          str   = "./sim_results"
    max_game_pitches: int   = 400    # hard cap per simulation
    device:           str   = "auto"
    # Context innings encoding:
    #   0.0 = no pitches seen, simulate whole game
    #   1.0 = top of 1st complete (3 outs)
    #   1.5 = bottom of 1st complete (6 outs)
    #   N.0 = top of Nth inning complete
    #   N.5 = bottom of Nth inning complete


# =============================================================================
# 1.  DIFFUSION SCHEDULE
# =============================================================================

class CosineNoiseSchedule(nn.Module):
    """Cosine beta schedule (better than linear for small T)."""

    def __init__(self, T: int, s: float = 0.008):
        super().__init__()
        steps = T + 1
        x = torch.linspace(0, T, steps)
        alphas_cumprod = torch.cos(((x / T) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        betas = betas.clamp(0.0001, 0.9999)

        alphas     = 1.0 - betas
        alpha_bar  = torch.cumprod(alphas, dim=0)
        alpha_bar_prev = F.pad(alpha_bar[:-1], (1, 0), value=1.0)

        self.register_buffer("betas",         betas)
        self.register_buffer("alphas",        alphas)
        self.register_buffer("alpha_bar",     alpha_bar)
        self.register_buffer("alpha_bar_prev",alpha_bar_prev)
        self.register_buffer("sqrt_alpha_bar",         torch.sqrt(alpha_bar))
        self.register_buffer("sqrt_one_minus_alpha_bar",torch.sqrt(1.0 - alpha_bar))
        self.T = T

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None):
        """Forward diffusion: q(x_t | x_0)."""
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_ab  = self.sqrt_alpha_bar[t].view(-1, *([1] * (x0.dim() - 1)))
        sqrt_oab = self.sqrt_one_minus_alpha_bar[t].view(-1, *([1] * (x0.dim() - 1)))
        return sqrt_ab * x0 + sqrt_oab * noise, noise

    @torch.no_grad()
    def p_sample(self, model_output: torch.Tensor, x_t: torch.Tensor, t: int):
        """One reverse diffusion step: p(x_{t-1} | x_t)."""
        beta_t       = self.betas[t]
        alpha_t      = self.alphas[t]
        alpha_bar_t  = self.alpha_bar[t]
        alpha_bar_t1 = self.alpha_bar_prev[t]

        # Predicted x_0 from noise prediction
        x0_pred = (x_t - self.sqrt_one_minus_alpha_bar[t] * model_output) / self.sqrt_alpha_bar[t]
        x0_pred = x0_pred.clamp(-5.0, 5.0)

        # Posterior mean
        coef1 = beta_t * torch.sqrt(alpha_bar_t1) / (1.0 - alpha_bar_t)
        coef2 = (1.0 - alpha_bar_t1) * torch.sqrt(alpha_t) / (1.0 - alpha_bar_t)
        mean  = coef1 * x0_pred + coef2 * x_t

        if t == 0:
            return mean
        variance = beta_t * (1.0 - alpha_bar_t1) / (1.0 - alpha_bar_t)
        return mean + torch.sqrt(variance) * torch.randn_like(x_t)

    @torch.no_grad()
    def ddim_sample(
        self,
        model_fn,
        shape: Tuple,
        device: torch.device,
        n_steps: int = 20,
        eta: float = 0.0,
    ) -> torch.Tensor:
        """
        DDIM accelerated sampling — runs in n_steps << T denoising steps.
        model_fn: callable(x_t, t_tensor) → predicted noise
        """
        step_size = self.T // n_steps
        timesteps = list(range(0, self.T, step_size))[::-1]

        x = torch.randn(shape, device=device)
        for i, t_val in enumerate(timesteps):
            t_batch = torch.full((shape[0],), t_val, device=device, dtype=torch.long)
            eps     = model_fn(x, t_batch)

            ab   = self.alpha_bar[t_val]
            ab_p = self.alpha_bar[timesteps[i + 1]] if i + 1 < len(timesteps) else torch.tensor(1.0)
            ab_p = ab_p.to(device) if isinstance(ab_p, torch.Tensor) else torch.tensor(ab_p, device=device)

            x0_pred = (x - torch.sqrt(1 - ab) * eps) / torch.sqrt(ab)
            x0_pred = x0_pred.clamp(-5.0, 5.0)

            sigma = eta * torch.sqrt((1 - ab_p) / (1 - ab) * (1 - ab / ab_p))
            dir_x = torch.sqrt(1 - ab_p - sigma ** 2) * eps
            x     = torch.sqrt(ab_p) * x0_pred + dir_x
            if eta > 0:
                x = x + sigma * torch.randn_like(x)

        return x


# =============================================================================
# 2.  MODEL COMPONENTS
# =============================================================================

class ContextEncoder(nn.Module):
    """
    Encodes the observed pitch sequence into a context memory tensor.
    Input per step: pitch_seq features + batter_ctx + batter_id_emb + pitch_type_emb + outcome_emb
    Global conditioning: pitcher_ctx + pitcher_id_emb + game_ctx + batting_order_emb
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        # Per-step input projection
        per_step_in = (
            cfg.pitch_feat_dim
            + cfg.batter_feat_dim
            + cfg.embed_dim   # batter_id embedding
            + cfg.embed_dim   # pitch_type embedding
            + cfg.embed_dim   # outcome embedding
        )
        self.step_proj = nn.Linear(per_step_in, cfg.d_model)

        # Global conditioning (added to every step via cross-attention)
        global_in = (
            cfg.pitcher_feat_dim
            + cfg.embed_dim   # pitcher_id embedding
            + cfg.game_feat_dim
            + cfg.embed_dim   # batting_order mean pool embedding
        )
        self.global_proj = nn.Linear(global_in, cfg.d_model)

        # Embeddings
        self.batter_emb   = nn.Embedding(cfg.num_batters,     cfg.embed_dim, padding_idx=0)
        self.pitcher_emb  = nn.Embedding(cfg.num_pitchers,    cfg.embed_dim, padding_idx=0)
        self.ptype_emb    = nn.Embedding(cfg.num_pitch_types, cfg.embed_dim, padding_idx=0)
        self.outcome_emb  = nn.Embedding(cfg.num_outcomes,    cfg.embed_dim, padding_idx=0)
        self.order_emb    = nn.Embedding(cfg.num_batters,     cfg.embed_dim, padding_idx=0)

        # Positional encoding
        self.pos_enc = SinusoidalPositionalEncoding(cfg.d_model, cfg.max_seq_len)

        # Causal Transformer encoder (decoder-style with causal mask)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.d_ff,
            dropout=cfg.dropout,
            batch_first=True,
            norm_first=True,   # Pre-LN for stability
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=cfg.n_layers)

        self.out_norm = nn.LayerNorm(cfg.d_model)

    def forward(
        self,
        pitch_seq:     torch.Tensor,   # [B, T, F_pitch]
        pitch_types:   torch.Tensor,   # [B, T]
        outcomes:      torch.Tensor,   # [B, T]
        batter_ctx:    torch.Tensor,   # [B, T, F_batter]
        batter_ids:    torch.Tensor,   # [B, T]
        pitcher_ctx:   torch.Tensor,   # [B, F_pitcher]
        pitcher_id:    torch.Tensor,   # [B]
        batting_order: torch.Tensor,   # [B, 9]
        game_ctx:      torch.Tensor,   # [B, F_game]
        mask:          torch.Tensor,   # [B, T] True=valid
    ) -> torch.Tensor:                 # [B, T, d_model]

        B, T, _ = pitch_seq.shape

        # Per-step features
        b_emb  = self.batter_emb(batter_ids)      # [B, T, E]
        pt_emb = self.ptype_emb(pitch_types)       # [B, T, E]
        oc_emb = self.outcome_emb(outcomes)         # [B, T, E]

        step_in = torch.cat([pitch_seq, batter_ctx, b_emb, pt_emb, oc_emb], dim=-1)
        x = self.step_proj(step_in)                # [B, T, d_model]

        # Global conditioning — broadcast as learned bias added to every step
        p_emb     = self.pitcher_emb(pitcher_id)   # [B, E]
        ord_emb   = self.order_emb(batting_order).mean(dim=1)  # [B, E]
        global_in = torch.cat([pitcher_ctx, p_emb, game_ctx, ord_emb], dim=-1)
        g_vec     = self.global_proj(global_in).unsqueeze(1)    # [B, 1, d_model]
        x = x + g_vec

        # Positional encoding
        x = self.pos_enc(x)

        # Causal mask: each position can only attend to itself and prior positions
        causal_mask = nn.Transformer.generate_square_subsequent_mask(T, device=x.device)

        # Key padding mask: True = ignore (pad token)
        key_pad = ~mask  # [B, T]

        x = self.transformer(x, mask=causal_mask, src_key_padding_mask=key_pad)
        return self.out_norm(x)   # [B, T, d_model]


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class DiffusionDenoiser(nn.Module):
    """
    Predicts the noise in x_t given:
        - x_t:          noisy pitch features [B, F_pitch_cont]
        - t_emb:        sinusoidal time step embedding
        - context:      encoded context from ContextEncoder at the conditioning position [B, d_model]
        - pitch_type:   sampled/known pitch type embedding [B, E]
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        t_dim = cfg.d_model

        # Sinusoidal time embedding
        self.time_emb = nn.Sequential(
            SinusoidalTimeEmbedding(t_dim),
            nn.Linear(t_dim, t_dim),
            nn.SiLU(),
            nn.Linear(t_dim, t_dim),
        )

        # Pitch type conditioning
        self.ptype_proj = nn.Linear(cfg.embed_dim, t_dim)
        self.ptype_emb  = nn.Embedding(cfg.num_pitch_types, cfg.embed_dim, padding_idx=0)

        # Noisy input projection
        self.x_proj = nn.Linear(len(PITCH_CONTINUOUS_COLS), t_dim)

        # Main denoiser: context (d_model) + time (d_model) + x_proj → noise prediction
        self.net = nn.Sequential(
            nn.Linear(t_dim * 3, t_dim * 2),
            nn.SiLU(),
            nn.LayerNorm(t_dim * 2),
            nn.Linear(t_dim * 2, t_dim),
            nn.SiLU(),
            nn.LayerNorm(t_dim),
            nn.Linear(t_dim, len(PITCH_CONTINUOUS_COLS)),
        )

    def forward(
        self,
        x_t:       torch.Tensor,  # [B, F_cont]
        t:         torch.Tensor,  # [B] integer timesteps
        context:   torch.Tensor,  # [B, d_model]
        pitch_type: torch.Tensor, # [B] int
    ) -> torch.Tensor:            # [B, F_cont] predicted noise
        t_emb  = self.time_emb(t)                          # [B, d_model]
        pt_emb = self.ptype_proj(self.ptype_emb(pitch_type))  # [B, d_model]
        x_emb  = self.x_proj(x_t)                         # [B, d_model]

        h = torch.cat([x_emb + t_emb, context, pt_emb], dim=-1)
        return self.net(h)


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device) / (half - 1)
        ).float()
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        return torch.cat([args.sin(), args.cos()], dim=-1)


class ClassificationHeads(nn.Module):
    """
    Dual-head classifier operating on context vectors.
        - Pitch type head: predicts pitch_types[t+1] from context[t]
        - Outcome head:    predicts outcomes[t] from context[t] (current pitch result)
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.pitch_type_head = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model // 2),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model // 2, cfg.num_pitch_types),
        )
        self.outcome_head = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model // 2),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model // 2, cfg.num_outcomes),
        )

    def forward(self, context: torch.Tensor):
        return self.pitch_type_head(context), self.outcome_head(context)


# =============================================================================
# 3.  FULL TRANSFUSION MODEL
# =============================================================================

class TransFusion(nn.Module):
    """
    TransFusion for baseball pitch sequence modeling.

    Forward pass (training):
        1. Encode the full observed sequence through ContextEncoder (causal).
        2. For each position t, context[t] conditions on pitches 0..t.
        3. Diffusion denoiser: given noisy x_{t+1}, context[t], and pitch_type[t+1],
           predict the noise. Diffusion loss on pitch continuous features.
        4. Classification heads on context[t]:
            - Predict pitch_types[t+1]  (pitch type head)
            - Predict outcomes[t]       (outcome head)

    Inference / simulation:
        Given context pitches 0..C-1, predict pitch C onwards autoregressively.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        self.encoder   = ContextEncoder(cfg)
        self.denoiser  = DiffusionDenoiser(cfg)
        self.heads     = ClassificationHeads(cfg)
        self.schedule  = CosineNoiseSchedule(cfg.n_diffusion_steps)

        # How many continuous features the denoiser operates on
        self.n_cont = len(PITCH_CONTINUOUS_COLS)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Training forward pass. Returns dict of losses.
        """
        pitch_seq     = batch["pitch_seq"]       # [B, T, F]
        pitch_types   = batch["pitch_types"]     # [B, T]
        outcomes      = batch["outcomes"]        # [B, T]
        batter_ctx    = batch["batter_ctx"]      # [B, T, FB]
        batter_ids    = batch["batter_ids"]      # [B, T]
        pitcher_ctx   = batch["pitcher_ctx"]     # [B, FP]
        pitcher_id    = batch["pitcher_id"]      # [B]
        batting_order = batch["batting_order"]   # [B, 9]
        game_ctx      = batch["game_ctx"]        # [B, FG]
        mask          = batch["mask"]            # [B, T]

        B, T, _ = pitch_seq.shape

        # ── 1. Encode context (causal) ─────────────────────────────────────
        context = self.encoder(
            pitch_seq, pitch_types, outcomes,
            batter_ctx, batter_ids,
            pitcher_ctx, pitcher_id,
            batting_order, game_ctx, mask,
        )  # [B, T, d_model]

        # ── 2. Classification losses ───────────────────────────────────────
        pt_logits, oc_logits = self.heads(context)  # [B, T, V_*]

        # Pitch type: predict t+1 from context at t  (shift by 1)
        # Valid range: t in [0, T-2], target = pitch_types[t+1]
        pt_loss = F.cross_entropy(
            pt_logits[:, :-1].reshape(-1, self.cfg.num_pitch_types),
            pitch_types[:, 1:].reshape(-1),
            ignore_index=0,
        )

        # Outcome: predict outcome at t from context at t
        oc_loss = F.cross_entropy(
            oc_logits.reshape(-1, self.cfg.num_outcomes),
            outcomes.reshape(-1),
            ignore_index=0,
        )

        # ── 3. Diffusion loss on continuous pitch features ─────────────────
        # Target: continuous features of pitch t+1 (only the pitch-physics cols)
        x0 = pitch_seq[:, 1:, :self.n_cont].reshape(B * (T - 1), self.n_cont)  # [(B*(T-1)), F_cont]

        # Sample random diffusion timestep per sample
        t_diff = torch.randint(0, self.cfg.n_diffusion_steps, (B * (T - 1),), device=x0.device)

        # Forward diffusion
        x_t, noise = self.schedule.q_sample(x0, t_diff)

        # Context at position t (predicts position t+1)
        ctx_for_diff = context[:, :-1].reshape(B * (T - 1), -1)   # [(B*(T-1)), d_model]
        pt_for_diff  = pitch_types[:, 1:].reshape(B * (T - 1))     # [(B*(T-1))]

        # Predict noise
        noise_pred = self.denoiser(x_t, t_diff, ctx_for_diff, pt_for_diff)

        # Only compute loss on valid (non-padded) positions
        valid = mask[:, 1:].reshape(-1)  # [(B*(T-1))]
        diff_loss = F.mse_loss(noise_pred[valid], noise[valid])

        # ── 4. Weighted total loss ─────────────────────────────────────────
        loss = (
            self.cfg.lambda_diffusion  * diff_loss
            + self.cfg.lambda_pitch_type * pt_loss
            + self.cfg.lambda_outcome    * oc_loss
        )

        return {
            "loss":      loss,
            "diff_loss": diff_loss.detach(),
            "pt_loss":   pt_loss.detach(),
            "oc_loss":   oc_loss.detach(),
        }

    @torch.no_grad()
    def encode_context(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Return context vectors for all positions in batch.
        Used during simulation to get the context at the split point.
        """
        return self.encoder(
            batch["pitch_seq"], batch["pitch_types"], batch["outcomes"],
            batch["batter_ctx"], batch["batter_ids"],
            batch["pitcher_ctx"], batch["pitcher_id"],
            batch["batting_order"], batch["game_ctx"], batch["mask"],
        )

    @torch.no_grad()
    def predict_next_pitch_type(self, context_vec: torch.Tensor) -> torch.Tensor:
        """
        Given context vector at position t, return pitch type probability distribution.
        context_vec: [B, d_model]  → returns [B, num_pitch_types] probabilities
        """
        pt_logits, _ = self.heads(context_vec)
        return F.softmax(pt_logits, dim=-1)

    @torch.no_grad()
    def predict_outcome(self, context_vec: torch.Tensor) -> torch.Tensor:
        """
        Given context vector at position t, return outcome probability distribution.
        context_vec: [B, d_model]  → returns [B, num_outcomes] probabilities
        """
        _, oc_logits = self.heads(context_vec)
        return F.softmax(oc_logits, dim=-1)

    @torch.no_grad()
    def sample_pitch_features(
        self,
        context_vec: torch.Tensor,    # [B, d_model]
        pitch_type:  torch.Tensor,    # [B] int
        ddim_steps:  int = 20,
    ) -> torch.Tensor:
        """
        Sample continuous pitch features for the next pitch via DDIM.
        Returns [B, n_cont] normalized pitch features.
        """
        B = context_vec.shape[0]
        device = context_vec.device

        def _model_fn(x_t, t_batch):
            return self.denoiser(x_t, t_batch, context_vec, pitch_type)

        return self.schedule.ddim_sample(
            _model_fn, (B, self.n_cont), device, n_steps=ddim_steps
        )


# =============================================================================
# 4.  GAME STATE ENGINE
# =============================================================================

# Outcome categories (must match encoder vocabulary)
# These string keys map to groups used by the game state engine
BALL_OUTCOMES = {
    "ball", "blocked_ball", "pitchout", "intent_ball",
}
STRIKE_OUTCOMES = {
    "called_strike", "swinging_strike", "swinging_strike_blocked",
    "foul_tip", "missed_bunt",
}
FOUL_OUTCOMES = {"foul", "foul_bunt"}
IN_PLAY_OUTCOMES = {
    "hit_into_play", "hit_into_play_score", "hit_into_play_no_out",
}

# PA terminal events → (outs_added, runs_scored, runner_advance_code)
# runner_advance_code: 'out'=batter out, 'single','double','triple','hr','walk','hbp'
EVENT_TABLE = {
    # outs
    "field_out":              (1, 0, "out"),
    "strikeout":              (1, 0, "out"),
    "strikeout_double_play":  (2, 0, "out"),
    "double_play":            (2, 0, "out"),
    "grounded_into_double_play": (2, 0, "out"),
    "force_out":              (1, 0, "out"),
    "field_error":            (0, 0, "single"),   # error → treat as single (conservative)
    "fielders_choice":        (1, 0, "single"),
    "fielders_choice_out":    (1, 0, "out"),
    "sac_fly":                (1, 1, "out"),
    "sac_fly_double_play":    (2, 1, "out"),
    "sac_bunt":               (1, 0, "out"),
    "other_out":              (1, 0, "out"),
    # hits
    "single":                 (0, 0, "single"),
    "double":                 (0, 0, "double"),
    "triple":                 (0, 0, "triple"),
    "home_run":               (0, 0, "hr"),
    # walks / HBP
    "walk":                   (0, 0, "walk"),
    "intent_walk":            (0, 0, "walk"),
    "hit_by_pitch":           (0, 0, "hbp"),
    # catch interference
    "catcher_interf":         (0, 0, "walk"),
}


class GameState:
    """
    Lightweight baseball game state machine used inside simulations.
    Tracks: inning, top/bot, outs, runners (1b/2b/3b), home_score, away_score.
    """

    def __init__(self):
        self.inning       = 1
        self.is_top       = True    # True = top (away batting)
        self.outs         = 0
        self.on_1b        = False
        self.on_2b        = False
        self.on_3b        = False
        self.home_score   = 0
        self.away_score   = 0
        self.batting_idx  = 0       # current position in batting order (0-8)
        self.balls        = 0
        self.strikes      = 0

    def is_game_over(self, max_innings: int = 9) -> bool:
        # Bottom of 9th (or later) ends when home team takes the lead or inning ends with tie broken
        if self.inning > max_innings:
            return True
        if self.inning == max_innings and not self.is_top and self.outs == 3:
            return True
        return False

    def advance_runners(self, bases: int) -> int:
        """Move all runners forward `bases` bases. Returns runs scored."""
        runs = 0
        new_1b, new_2b, new_3b = False, False, False

        if bases == 1:  # single
            if self.on_3b: runs += 1
            if self.on_2b: new_3b = True
            if self.on_1b: new_2b = True
            new_1b = True
        elif bases == 2:  # double
            if self.on_3b: runs += 1
            if self.on_2b: runs += 1
            if self.on_1b: new_3b = True
            new_2b = True
        elif bases == 3:  # triple
            if self.on_3b: runs += 1
            if self.on_2b: runs += 1
            if self.on_1b: runs += 1
            new_3b = True
        elif bases == 4:  # home run
            if self.on_3b: runs += 1
            if self.on_2b: runs += 1
            if self.on_1b: runs += 1
            runs += 1  # batter scores
            new_1b = new_2b = new_3b = False

        self.on_1b, self.on_2b, self.on_3b = new_1b, new_2b, new_3b
        return runs

    def apply_walk(self) -> int:
        """Walk / HBP: advance batter to first, force runners if bases loaded."""
        runs = 0
        if self.on_1b and self.on_2b and self.on_3b:
            runs = 1
            self.on_3b = True
        elif self.on_1b and self.on_2b:
            self.on_3b = True
        elif self.on_1b:
            self.on_2b = True
        self.on_1b = True
        return runs

    def apply_event(self, event_str: str) -> int:
        """
        Apply a PA terminal event. Returns runs scored this PA.
        Updates outs, runners, scores.
        """
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
        """
        Apply a single pitch outcome. Returns PA terminal event string if PA ends,
        else None.
        """
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
        # in-play outcomes are handled by the event (not the pitch description)
        return None

    def _add_score(self, runs: int):
        if self.is_top:
            self.away_score += runs
        else:
            self.home_score += runs

    def _end_half_inning(self):
        self.outs = 0
        self.on_1b = self.on_2b = self.on_3b = False
        self.balls = self.strikes = 0
        if self.is_top:
            self.is_top = False
        else:
            self.inning += 1
            self.is_top = True

    def to_feature_vec(self, pitch_scaler) -> np.ndarray:
        """Produce the GAME_STATE_COLS portion of the pitch_seq input."""
        # Normalized using the same scaler used during training
        raw = {
            "balls":       float(self.balls),
            "strikes":     float(self.strikes),
            "outs_when_up":float(self.outs),
            "inning":      float(self.inning),
            "home_score":  float(self.home_score),
            "away_score":  float(self.away_score),
            "on_1b":       float(self.on_1b),
            "on_2b":       float(self.on_2b),
            "on_3b":       float(self.on_3b),
            "run_diff":    float(self.home_score - self.away_score),
        }
        return pitch_scaler.transform_row(raw, GAME_STATE_COLS)


def inning_number_to_context_outs(context_innings: float) -> int:
    """
    Convert the user-specified context_innings float to a number of completed outs
    that defines the boundary between observed and simulated.
        0.0 → 0 outs (simulate whole game)
        1.0 → 3 outs (top of 1st done)
        1.5 → 6 outs (bottom of 1st done)
        9.0 → 27 outs (full game)
    """
    full_innings = int(context_innings)
    half = (context_innings - full_innings) >= 0.4  # 0.5 half inning
    return full_innings * 6 + (3 if half else 0)


def find_context_split(game_df, context_innings: float) -> int:
    """
    Given a game DataFrame (sorted by at_bat_number, pitch_number),
    find the pitch index that corresponds to the end of `context_innings`.
    Returns the index of the first pitch to be simulated (0 = whole game).
    """
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
                return i + 1  # next pitch is first simulated

    return len(game_df)  # fallback: all context


# =============================================================================
# 5.  TRAINING
# =============================================================================

def get_lr_scheduler(optimizer, warmup_steps: int, total_steps: int):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.05, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train(cfg_train: TrainConfig, cfg_model: ModelConfig):
    # Setup
    torch.manual_seed(cfg_train.seed)
    np.random.seed(cfg_train.seed)
    random.seed(cfg_train.seed)

    device = _resolve_device(cfg_train.device)
    print(f"[train] Device: {device}")

    ckpt_dir = Path(cfg_train.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ── Build datasets ────────────────────────────────────────────────────
    builder = BaseballDatasetBuilder(
        start_dt             = cfg_train.start_dt,
        end_dt               = cfg_train.end_dt,
        val_start_dt         = cfg_train.val_start_dt,
        test_start_dt        = cfg_train.test_start_dt,
        cache_dir            = cfg_train.cache_dir,
        max_seq_len          = cfg_model.max_seq_len,
        min_pitches_per_game = 100,
    )
    train_ds, val_ds, test_ds, encoders = builder.build()

    # Patch model config with true dims
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

    # ── Model ─────────────────────────────────────────────────────────────
    model = TransFusion(cfg_model).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[train] TransFusion parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg_train.lr, weight_decay=cfg_train.weight_decay
    )
    total_steps  = cfg_train.epochs * len(train_loader)
    scheduler    = get_lr_scheduler(optimizer, cfg_train.warmup_steps, total_steps)
    scaler_amp   = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    # Save config
    _save_json(vars(cfg_model), ckpt_dir / "model_config.json")

    best_val_loss = float("inf")
    global_step   = 0

    for epoch in range(1, cfg_train.epochs + 1):
        model.train()
        epoch_losses = {"loss": 0.0, "diff_loss": 0.0, "pt_loss": 0.0, "oc_loss": 0.0}
        t0 = time.time()

        for step, batch in enumerate(train_loader):
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
                print(
                    f"  ep={epoch:3d} step={global_step:6d} "
                    f"loss={losses['loss'].item():.4f} "
                    f"diff={losses['diff_loss'].item():.4f} "
                    f"pt={losses['pt_loss'].item():.4f} "
                    f"oc={losses['oc_loss'].item():.4f} "
                    f"lr={lr:.2e}"
                )

        n = len(train_loader)
        elapsed = time.time() - t0
        print(
            f"[epoch {epoch:3d}] train_loss={epoch_losses['loss']/n:.4f}  "
            f"({elapsed:.0f}s)"
        )

        # ── Validation ────────────────────────────────────────────────────
        if epoch % cfg_train.val_every == 0:
            val_loss = _evaluate(model, val_loader, device)
            print(f"[epoch {epoch:3d}] val_loss={val_loss:.4f}")

            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "cfg_model": vars(cfg_model),
                },
                ckpt_dir / "latest.pt",
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(
                    {"epoch": epoch, "model_state": model.state_dict(),
                     "val_loss": val_loss, "cfg_model": vars(cfg_model)},
                    ckpt_dir / "best.pt",
                )
                print(f"  ✓ New best val_loss={val_loss:.4f}")

    print(f"[train] Done. Best val_loss={best_val_loss:.4f}")


@torch.no_grad()
def _evaluate(model: TransFusion, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total, count = 0.0, 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        losses = model(batch)
        total += losses["loss"].item()
        count += 1
    model.train()
    return total / max(count, 1)


# =============================================================================
# 6.  MCMC SIMULATION
# =============================================================================

@dataclass
class SimResult:
    game_pk:          int
    context_innings:  float
    n_simulations:    int
    home_win_prob:    float
    away_win_prob:    float
    tie_prob:         float
    mean_home_runs:   float
    mean_away_runs:   float
    std_home_runs:    float
    std_away_runs:    float
    actual_home_score: Optional[int]
    actual_away_score: Optional[int]
    actual_home_win:  Optional[bool]


def simulate_games(cfg_sim: SimConfig, cfg_model: ModelConfig):
    """Main simulation entry point."""
    device = _resolve_device(cfg_sim.device)
    out_dir = Path(cfg_sim.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load model ────────────────────────────────────────────────────────
    ckpt = torch.load(cfg_sim.checkpoint, map_location=device)
    saved_cfg = ckpt.get("cfg_model", {})
    for k, v in saved_cfg.items():
        if hasattr(cfg_model, k):
            setattr(cfg_model, k, v)

    model = TransFusion(cfg_model).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"[sim] Loaded checkpoint from {cfg_sim.checkpoint}  (epoch {ckpt.get('epoch','?')})")

    # ── Load dataset ──────────────────────────────────────────────────────
    builder = BaseballDatasetBuilder(
        start_dt      = "2022-04-07",
        end_dt        = "2024-11-01",
        val_start_dt  = "2024-03-20",
        test_start_dt = "2024-10-01",
        cache_dir     = cfg_sim.cache_dir,
        max_seq_len   = cfg_model.max_seq_len,
    )
    train_ds, val_ds, test_ds, encoders = builder.build()

    ds_map = {"train": train_ds, "val": val_ds, "test": test_ds}
    dataset: PitchSequenceDataset = ds_map[cfg_sim.split]
    print(f"[sim] Simulating {len(dataset)} games from '{cfg_sim.split}' split")
    print(f"[sim] Context: {cfg_sim.context_innings} innings  |  "
          f"N simulations per game: {cfg_sim.n_simulations}")

    results = []

    for game_idx in range(len(dataset)):
        sample   = dataset[game_idx]
        game_pk  = sample["game_pk"].item()
        game_df  = dataset.game_groups[game_pk]

        # Find the split point in pitch sequence
        context_end_idx = find_context_split(game_df, cfg_sim.context_innings)

        # Actual final score (ground truth)
        actual_home = int(game_df["home_score"].iloc[-1])
        actual_away = int(game_df["away_score"].iloc[-1])

        # ── Build context batch (first context_end_idx pitches) ───────────
        if context_end_idx == 0:
            # No context: use a single dummy step (zeros) to seed encoding
            ctx_batch = _make_empty_context_batch(sample, device)
            ctx_out_idx = 0
        else:
            ctx_batch   = _make_context_batch(sample, context_end_idx, device)
            ctx_out_idx = context_end_idx - 1   # last context position

        # Encode context
        with torch.no_grad():
            context_memory = model.encode_context(ctx_batch)  # [1, T_ctx, d_model]

        # ── Run K Monte Carlo simulations in parallel ─────────────────────
        home_scores, away_scores = [], []

        # Reconstruct game state at context boundary
        context_state = _reconstruct_game_state(game_df, context_end_idx)

        # Repeat context memory K times for parallel simulation
        K = cfg_sim.n_simulations
        ctx_vec = context_memory[:, ctx_out_idx, :].expand(K, -1)  # [K, d_model]

        # We'll run simulation one game at a time in parallel across K sims
        h_scores, a_scores = _run_parallel_simulations(
            model         = model,
            encoders      = encoders,
            pitch_scaler  = dataset.pitch_scaler,
            batter_scaler = dataset.batter_scaler,
            pitcher_scaler= dataset.pitcher_scaler,
            ctx_vec       = ctx_vec,
            game_state_template = context_state,
            sample        = sample,
            game_df       = game_df,
            context_end_idx = context_end_idx,
            K             = K,
            device        = device,
            max_pitches   = cfg_sim.max_game_pitches,
            cfg_model     = cfg_model,
        )
        home_scores.extend(h_scores)
        away_scores.extend(a_scores)

        # ── Aggregate results ─────────────────────────────────────────────
        home_wins  = sum(h > a for h, a in zip(home_scores, away_scores))
        away_wins  = sum(a > h for h, a in zip(home_scores, away_scores))
        ties       = K - home_wins - away_wins

        result = SimResult(
            game_pk           = game_pk,
            context_innings   = cfg_sim.context_innings,
            n_simulations     = K,
            home_win_prob     = home_wins / K,
            away_win_prob     = away_wins / K,
            tie_prob          = ties / K,
            mean_home_runs    = float(np.mean(home_scores)),
            mean_away_runs    = float(np.mean(away_scores)),
            std_home_runs     = float(np.std(home_scores)),
            std_away_runs     = float(np.std(away_scores)),
            actual_home_score = actual_home,
            actual_away_score = actual_away,
            actual_home_win   = actual_home > actual_away,
        )
        results.append(result)

        if (game_idx + 1) % 10 == 0 or game_idx == 0:
            correct = actual_home > actual_away and result.home_win_prob > 0.5
            correct = correct or (actual_away > actual_home and result.away_win_prob > 0.5)
            print(
                f"  [{game_idx+1:4d}/{len(dataset)}] game={game_pk}  "
                f"P(home)={result.home_win_prob:.3f}  "
                f"actual={'H' if actual_home > actual_away else 'A'}  "
                f"{'✓' if correct else '✗'}"
            )

    # ── Save results ──────────────────────────────────────────────────────
    results_path = out_dir / f"sim_results_{cfg_sim.context_innings}inn.json"
    out_list = [vars(r) for r in results]
    with open(results_path, "w") as f:
        json.dump(out_list, f, indent=2)
    print(f"[sim] Saved {len(results)} results to {results_path}")

    # ── Summary metrics ───────────────────────────────────────────────────
    _print_summary(results)
    return results


def _run_parallel_simulations(
    model, encoders, pitch_scaler, batter_scaler, pitcher_scaler,
    ctx_vec, game_state_template, sample, game_df, context_end_idx,
    K, device, max_pitches, cfg_model,
) -> Tuple[List[float], List[float]]:
    """
    Run K game simulations in parallel (vectorized across the K axis).
    Uses batched model inference for efficiency.
    Returns (home_scores, away_scores) lists of length K.
    """
    # Clone K independent game states
    game_states = [_clone_game_state(game_state_template) for _ in range(K)]

    # Current context vector for each simulation: [K, d_model]
    current_ctx = ctx_vec.clone()

    # Batter context for the next batter in each sim (use training-season batter means)
    # We'll look up the next batter from the batting order
    batting_order_raw = sample["batting_order"].tolist()   # [9] tokenized IDs

    # Pitch type and outcome index→string lookups (reverse encoders)
    pt_idx2str  = {v: k for k, v in encoders.pitch_type.items()}
    oc_idx2str  = {v: k for k, v in encoders.outcome.items()}
    ev_idx2str  = {v: k for k, v in encoders.event.items()}

    active = list(range(K))   # indices of simulations still running
    home_scores = [0.0] * K
    away_scores = [0.0] * K

    pitch_count = 0
    season = int(game_df.iloc[0].get("game_year", 2024))

    # Per-sim PA tracking
    pa_pitch_counts = [0] * K   # pitches in current PA

    while active and pitch_count < max_pitches:
        if not active:
            break

        idx = torch.tensor(active, device=device)
        ctx_active = current_ctx[idx]   # [n_active, d_model]
        n_active   = len(active)

        # ── Step 1: Sample next pitch type ────────────────────────────────
        pt_probs = model.predict_next_pitch_type(ctx_active)   # [n_active, V_pt]
        pt_samples = torch.multinomial(pt_probs, 1).squeeze(-1)  # [n_active]

        # ── Step 2: Sample continuous pitch features via DDIM ─────────────
        pitch_feats = model.sample_pitch_features(
            ctx_active, pt_samples, ddim_steps=10
        )  # [n_active, F_cont]

        # ── Step 3: Sample outcome ────────────────────────────────────────
        oc_probs   = model.predict_outcome(ctx_active)          # [n_active, V_oc]
        oc_samples = torch.multinomial(oc_probs, 1).squeeze(-1)  # [n_active]

        # ── Step 4: Advance each simulation's game state ──────────────────
        still_active = []
        for local_i, sim_i in enumerate(active):
            gs  = game_states[sim_i]
            pt  = pt_idx2str.get(pt_samples[local_i].item(), "FF")
            oc  = oc_idx2str.get(oc_samples[local_i].item(), "ball")

            # Check if this outcome terminates the PA
            terminal_event = gs.apply_pitch_outcome(oc)

            # If in-play, resolve with a PA-ending event sampled from event distribution
            if oc in IN_PLAY_OUTCOMES or terminal_event is None and oc in IN_PLAY_OUTCOMES:
                terminal_event = _sample_in_play_event(encoders, oc_probs[local_i])

            if terminal_event is not None:
                gs.apply_event(terminal_event)
                pa_pitch_counts[sim_i] = 0
            else:
                pa_pitch_counts[sim_i] += 1

            home_scores[sim_i] = gs.home_score
            away_scores[sim_i] = gs.away_score

            # ── Step 5: Update context vector for next step ───────────────
            # Build new pitch step features: concat sampled pitch feats + game state feats
            gs_feats = torch.tensor(
                gs.to_feature_vec(pitch_scaler), device=device, dtype=torch.float32
            )
            # Batter stats for current batter
            batter_ord_idx = gs.batting_idx
            batter_tok     = batting_order_raw[batter_ord_idx] if batter_ord_idx < len(batting_order_raw) else 0
            batter_mlbam   = _decode_batter_id(encoders, batter_tok)
            b_row          = dataset_batter_lut_lookup(model, encoders, batter_mlbam, season)
            b_ctx = torch.tensor(
                batter_scaler.transform_row(b_row, BATTER_STAT_COLS),
                device=device, dtype=torch.float32,
            )
            # Full new pitch feature: [F_cont | game_state | batter_ctx | batter_id | pt | oc]
            # We pass through the encoder step projection
            # For efficiency: do a single-step incremental encode via the encoder
            new_step_ctx = _incremental_encode_step(
                model        = model,
                prev_ctx     = ctx_active[local_i:local_i+1],
                pitch_feats  = pitch_feats[local_i:local_i+1],
                gs_feats     = gs_feats.unsqueeze(0),
                b_ctx        = b_ctx.unsqueeze(0),
                batter_id    = torch.tensor([batter_tok], device=device),
                pt_token     = pt_samples[local_i:local_i+1],
                oc_token     = oc_samples[local_i:local_i+1],
            )  # [1, d_model]
            current_ctx[sim_i] = new_step_ctx.squeeze(0)

            if not gs.is_game_over():
                still_active.append(sim_i)

        active = still_active
        pitch_count += 1

    return home_scores, away_scores


def _incremental_encode_step(
    model, prev_ctx, pitch_feats, gs_feats, b_ctx, batter_id, pt_token, oc_token
) -> torch.Tensor:
    """
    Approximate incremental context update for a single new pitch.
    Rather than re-encoding the full sequence (expensive), we compute the
    new step's contribution and update via learned projection.

    prev_ctx:    [1, d_model]   context from previous step
    pitch_feats: [1, F_cont]    sampled continuous pitch features
    gs_feats:    [1, F_gs]      game state features
    b_ctx:       [1, F_batter]  batter season stats
    batter_id:   [1]            tokenized batter id
    pt_token:    [1]            pitch type token
    oc_token:    [1]            outcome token
    """
    enc = model.encoder

    b_emb  = enc.batter_emb(batter_id)    # [1, E]
    pt_emb = enc.ptype_emb(pt_token)       # [1, E]
    oc_emb = enc.outcome_emb(oc_token)     # [1, E]

    # Concatenate pitch features + game state + embeddings
    step_in = torch.cat([pitch_feats, gs_feats, b_ctx, b_emb, pt_emb, oc_emb], dim=-1)

    # Project and add to previous context (residual approximation)
    new_ctx = enc.step_proj(step_in) + prev_ctx   # [1, d_model]
    return enc.out_norm(new_ctx)


def _sample_in_play_event(encoders: Encoders, oc_probs: torch.Tensor) -> str:
    """
    When a pitch is hit in play, sample a PA-ending event.
    Uses a simple prior over batted ball events.
    """
    in_play_events = {
        "single": 0.23, "double": 0.06, "triple": 0.008, "home_run": 0.04,
        "field_out": 0.45, "force_out": 0.08, "double_play": 0.04,
        "grounded_into_double_play": 0.03, "field_error": 0.01, "sac_fly": 0.005,
    }
    events = list(in_play_events.keys())
    probs  = np.array(list(in_play_events.values()))
    probs  = probs / probs.sum()
    return np.random.choice(events, p=probs)


def _reconstruct_game_state(game_df, context_end_idx: int) -> "GameState":
    """
    Walk through the first context_end_idx pitches of the game to reconstruct
    the exact game state at the context boundary.
    """
    gs = GameState()
    if context_end_idx == 0:
        return gs

    for i, row in game_df.iloc[:context_end_idx].iterrows():
        event   = row.get("events", None)
        outcome = str(row.get("description", "ball"))

        if event is not None and str(event) != "nan":
            gs.apply_event(str(event))
        else:
            gs.apply_pitch_outcome(outcome)

    return gs


def _clone_game_state(gs: "GameState") -> "GameState":
    new = GameState()
    new.__dict__.update(gs.__dict__.copy())
    return new


def _make_context_batch(
    sample: Dict[str, torch.Tensor],
    context_end_idx: int,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """Slice sample tensors to context_end_idx and add batch dim."""
    keys_to_slice = ["pitch_seq", "pitch_types", "outcomes", "at_bat_events",
                     "batter_ctx", "batter_ids", "mask"]
    batch = {}
    for k in keys_to_slice:
        batch[k] = sample[k][:context_end_idx].unsqueeze(0).to(device)

    for k in ["pitcher_ctx", "pitcher_id", "batting_order", "game_ctx"]:
        batch[k] = sample[k].unsqueeze(0).to(device)

    return batch


def _make_empty_context_batch(
    sample: Dict[str, torch.Tensor], device: torch.device
) -> Dict[str, torch.Tensor]:
    """Create a single-step zero context when context_innings=0."""
    batch = {}
    for k in ["pitch_seq", "batter_ctx"]:
        dim = sample[k].shape[-1]
        batch[k] = torch.zeros(1, 1, dim, device=device)
    for k in ["pitch_types", "outcomes", "at_bat_events", "batter_ids"]:
        batch[k] = torch.zeros(1, 1, dtype=torch.long, device=device)
    batch["mask"] = torch.ones(1, 1, dtype=torch.bool, device=device)
    for k in ["pitcher_ctx", "pitcher_id", "batting_order", "game_ctx"]:
        batch[k] = sample[k].unsqueeze(0).to(device)
    return batch


def _decode_batter_id(encoders: Encoders, token: int) -> int:
    """Reverse map token → MLBAM ID."""
    rev = {v: k for k, v in encoders.batter_id.items()}
    return rev.get(token, 0)


def dataset_batter_lut_lookup(model, encoders, mlbam_id: int, season: int) -> dict:
    """Return an empty dict if batter not in lookup; scaler will fill with means."""
    return {}   # The StatScaler.transform_row handles missing keys gracefully


def _print_summary(results: List[SimResult]):
    n = len(results)
    if n == 0:
        return

    correct_home = sum(
        1 for r in results
        if r.actual_home_win is not None
        and ((r.actual_home_win and r.home_win_prob > 0.5)
             or (not r.actual_home_win and r.away_win_prob > 0.5))
    )
    brier = np.mean([
        (r.home_win_prob - float(r.actual_home_win)) ** 2
        for r in results if r.actual_home_win is not None
    ])
    log_loss_vals = []
    for r in results:
        if r.actual_home_win is None:
            continue
        p = r.home_win_prob if r.actual_home_win else r.away_win_prob
        log_loss_vals.append(-math.log(max(p, 1e-7)))

    print("\n" + "=" * 55)
    print(f"  SIMULATION SUMMARY  ({n} games)")
    print("=" * 55)
    print(f"  Accuracy        : {correct_home / n:.4f}  ({correct_home}/{n})")
    print(f"  Brier Score     : {brier:.4f}")
    print(f"  Log Loss        : {np.mean(log_loss_vals):.4f}")
    print(f"  Mean P(home win): {np.mean([r.home_win_prob for r in results]):.4f}")
    print("=" * 55 + "\n")


# =============================================================================
# 7.  UTILITIES
# =============================================================================

def _resolve_device(device_str: str) -> torch.device:
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_str)


def _save_json(obj: dict, path: Path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


# =============================================================================
# 8.  CLI
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="TransFusion Baseball Model")
    sub = parser.add_subparsers(dest="command")

    # ── train ────────────────────────────────────────────────────────────
    p_train = sub.add_parser("train")
    p_train.add_argument("--cache_dir",      default="./baseball_cache")
    p_train.add_argument("--checkpoint_dir", default="./checkpoints")
    p_train.add_argument("--start_dt",       default="2022-04-07")
    p_train.add_argument("--end_dt",         default="2024-11-01")
    p_train.add_argument("--val_start_dt",   default="2024-03-20")
    p_train.add_argument("--test_start_dt",  default="2024-10-01")
    p_train.add_argument("--epochs",         type=int,   default=40)
    p_train.add_argument("--batch_size",     type=int,   default=16)
    p_train.add_argument("--lr",             type=float, default=3e-4)
    p_train.add_argument("--d_model",        type=int,   default=256)
    p_train.add_argument("--n_layers",       type=int,   default=6)
    p_train.add_argument("--n_heads",        type=int,   default=8)
    p_train.add_argument("--n_diff_steps",   type=int,   default=50)
    p_train.add_argument("--num_workers",    type=int,   default=4)
    p_train.add_argument("--device",         default="auto")
    p_train.add_argument("--seed",           type=int,   default=42)

    # ── simulate ─────────────────────────────────────────────────────────
    p_sim = sub.add_parser("simulate")
    p_sim.add_argument("--checkpoint",      required=True)
    p_sim.add_argument("--cache_dir",       default="./baseball_cache")
    p_sim.add_argument("--context_innings", type=float, default=3.0,
                       help="Innings of real context (0, 0.5, 1.0, ..., 8.5)")
    p_sim.add_argument("--n_simulations",   type=int,   default=500)
    p_sim.add_argument("--split",           default="test",
                       choices=["train", "val", "test"])
    p_sim.add_argument("--out_dir",         default="./sim_results")
    p_sim.add_argument("--device",          default="auto")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.command == "train":
        cfg_train = TrainConfig(
            cache_dir      = args.cache_dir,
            checkpoint_dir = args.checkpoint_dir,
            start_dt       = args.start_dt,
            end_dt         = args.end_dt,
            val_start_dt   = args.val_start_dt,
            test_start_dt  = args.test_start_dt,
            epochs         = args.epochs,
            batch_size     = args.batch_size,
            lr             = args.lr,
            num_workers    = args.num_workers,
            device         = args.device,
            seed           = args.seed,
        )
        cfg_model = ModelConfig(
            d_model           = args.d_model,
            n_layers          = args.n_layers,
            n_heads           = args.n_heads,
            n_diffusion_steps = args.n_diff_steps,
        )
        train(cfg_train, cfg_model)

    elif args.command == "simulate":
        cfg_sim = SimConfig(
            checkpoint      = args.checkpoint,
            cache_dir       = args.cache_dir,
            context_innings = args.context_innings,
            n_simulations   = args.n_simulations,
            split           = args.split,
            out_dir         = args.out_dir,
            device          = args.device,
        )
        simulate_games(cfg_sim, ModelConfig())

    else:
        print("Usage: python transfusion.py [train|simulate] --help")