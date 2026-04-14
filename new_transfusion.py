"""
TransFusion: Baseball Pitch Sequence Model + MCMC Game Simulator
================================================================
Architecture:
    TransFusion = Transformer encoder (context) + Diffusion decoder (continuous pitch features)
                + dual classification heads (pitch type, pitch outcome)
                + Pitcher VAE + Batter VAE (latent player representations)

    The VAEs map raw pitcher/batter season statistics to a learned latent space,
    providing smoother generalization across players — especially for players with
    limited data — and decoupling player identity from raw stat noise.

Training:
    Given a game sequence, the model learns to denoise the next pitch's continuous
    features (diffusion loss) while simultaneously classifying the next pitch type
    and the next pitch outcome (cross-entropy losses). Pitcher and batter VAEs are
    trained jointly with KL divergence and reconstruction losses.

    NOTE ON SCORE IN FEATURES:
    home_score / away_score / run_diff are VALID inputs — at position t they only
    reflect the score up to that pitch (chronological), not the final score.
    This is legitimate game context, not leakage.

    LEAKAGE FIX APPLIED:
    The outcome head now predicts outcomes[t+1] from context[t], consistent with
    the pitch type head. Previously it predicted outcomes[t] from context[t] which
    contained the outcome embedding of outcomes[t] — trivial self-prediction.

Simulation (MCMC):
    Given N context innings of a real game, autoregressively simulate the remainder
    pitch-by-pitch. At each step:
        1. Sample pitch_type from the model's categorical distribution
        2. Denoise a continuous pitch feature vector conditioned on context + pitch_type
        3. Sample outcome from the outcome head
        4. Advance game state (count, outs, runners, score) deterministically
        5. Repeat until 27 outs (or extras)
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
    d_ff:        int   = 2048
    dropout:     float = 0.2
    max_seq_len: int   = 400

    # Diffusion
    n_diffusion_steps: int   = 50
    beta_start:        float = 1e-4
    beta_end:          float = 0.02

    # Embeddings
    embed_dim: int = 128

    # VAE latent dimensions
    pitcher_latent_dim: int = 64   # latent space for pitcher stats
    batter_latent_dim:  int = 64   # latent space for batter stats

    # VAE loss weight — how much KL + recon contribute to total loss
    lambda_vae: float = 0.05

    # Loss weights (sum to 1.0)
    lambda_diffusion:  float = 0.20
    lambda_pitch_type: float = 0.40
    lambda_outcome:    float = 0.40


@dataclass
class TrainConfig:
    cache_dir:      str   = "./baseball_cache"
    checkpoint_dir: str   = "./checkpoints"
    start_dt:       str   = "2021-04-07"
    end_dt:         str   = "2026-05-01"
    val_start_dt:   str   = "2026-02-01"
    test_start_dt:  str   = "2026-03-25"
    epochs:         int   = 60
    batch_size:     int   = 16
    lr:             float = 2e-4
    weight_decay:   float = 1e-4
    grad_clip:      float = 1.0
    warmup_steps:   int   = 1000
    log_every:      int   = 50
    val_every:      int   = 1
    num_workers:    int   = 0
    seed:           int   = 42
    device:         str   = "auto"


@dataclass
class SimConfig:
    checkpoint:       str   = "./checkpoints/best.pt"
    cache_dir:        str   = "./baseball_cache"
    start_dt:         str   = "2021-04-07"
    end_dt:           str   = "2026-05-01"
    val_start_dt:     str   = "2026-02-01"
    test_start_dt:    str   = "2026-03-25"
    context_innings:  float = 3.0
    n_simulations:    int   = 500
    split:            str   = "test"
    out_dir:          str   = "./sim_results"
    max_game_pitches: int   = 400
    device:           str   = "auto"
    temperature:      float = 1.0


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

        alphas        = 1.0 - betas
        alpha_bar     = torch.cumprod(alphas, dim=0)
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
# 1b.  PLAYER VAEs
# =============================================================================

class PlayerVAE(nn.Module):
    """
    Variational Autoencoder for player season statistics.

    Encodes a raw stat vector (pitcher or batter) into a learned latent space
    z ~ N(μ, σ²) via the reparameterization trick. The decoder reconstructs
    the original stats from z, providing a reconstruction loss signal.

    During training: encode stats → sample z → decode → compute KL + recon.
    During inference: encode stats → use μ directly (no sampling noise).

    The latent vector z replaces the raw stat vector as input to the
    ContextEncoder's global conditioning, giving the transformer a smoother,
    more generalizable representation of pitcher/batter identity.

    For unseen players or players with few pitches, the encoder will map
    their stats close to the prior N(0, I), providing graceful degradation
    rather than out-of-distribution raw stat values.
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
        logvar = self.fc_logvar(h).clamp(-4.0, 4.0)  # clamp for stability
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor,
                       logvar: torch.Tensor) -> torch.Tensor:
        """Sample z = μ + ε·σ using the reparameterization trick."""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu  # deterministic at inference

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns (z, μ, log σ²).
        z is the sampled latent (or μ at inference).
        """
        mu, logvar = self.encode(x)
        z          = self.reparameterize(mu, logvar)
        return z, mu, logvar

    @staticmethod
    def kl_loss(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """KL divergence KL(N(μ,σ²) || N(0,I)), averaged over batch."""
        return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    def vae_loss(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full VAE loss: reconstruction MSE + KL divergence.
        Returns (z, recon_loss, kl_loss).
        """
        z, mu, logvar = self.forward(x)
        x_recon       = self.decode(z)
        recon_loss    = F.mse_loss(x_recon, x)
        kl            = self.kl_loss(mu, logvar)
        return z, recon_loss, kl


# =============================================================================
# 2.  MODEL COMPONENTS
# =============================================================================

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


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


class ContextEncoder(nn.Module):
    """
    Causal Transformer encoder over pitch sequences.

    Per-step input:  pitch_seq + batter_ctx + batter_id_emb + ptype_emb + outcome_emb
    Global context:  pitcher_latent (from VAE) + pitcher_id_emb + game_ctx
                     + batting_order_emb + batter_latent_mean (mean of lineup VAE latents)

    The VAE latent vectors replace raw pitcher_ctx and batter_ctx in the global
    conditioning, providing smoother player representations.

    Score columns (home_score, away_score, run_diff) are included in pitch_seq
    and are VALID — at position t they only reflect the score up to that pitch,
    never the final score. The causal mask enforces this.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        # Per-step input: pitch feats + per-pitch batter stats + embeddings
        # Note: batter_ctx is still used per-step for at-bat specific context
        per_step_in = (
            cfg.pitch_feat_dim
            + cfg.batter_feat_dim
            + cfg.embed_dim   # batter_id
            + cfg.embed_dim   # pitch_type
            + cfg.embed_dim   # outcome
        )
        self.step_proj = nn.Linear(per_step_in, cfg.d_model)

        # Global conditioning now uses VAE latents instead of raw stats
        global_in = (
            cfg.pitcher_latent_dim  # pitcher VAE latent (replaces raw pitcher_ctx)
            + cfg.embed_dim         # pitcher_id
            + cfg.game_feat_dim
            + cfg.embed_dim         # batting_order mean pool
            + cfg.batter_latent_dim # mean batter VAE latent over lineup
        )
        self.global_proj = nn.Linear(global_in, cfg.d_model)

        self.batter_emb  = nn.Embedding(cfg.num_batters,     cfg.embed_dim, padding_idx=0)
        self.pitcher_emb = nn.Embedding(cfg.num_pitchers,    cfg.embed_dim, padding_idx=0)
        self.ptype_emb   = nn.Embedding(cfg.num_pitch_types, cfg.embed_dim, padding_idx=0)
        self.outcome_emb = nn.Embedding(cfg.num_outcomes,    cfg.embed_dim, padding_idx=0)
        self.order_emb   = nn.Embedding(cfg.num_batters,     cfg.embed_dim, padding_idx=0)

        self.pos_enc = SinusoidalPositionalEncoding(cfg.d_model, cfg.max_seq_len)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model, nhead=cfg.n_heads,
            dim_feedforward=cfg.d_ff, dropout=cfg.dropout,
            batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=cfg.n_layers)
        self.out_norm    = nn.LayerNorm(cfg.d_model)

    def forward(
        self,
        pitch_seq:      torch.Tensor,  # [B, T, F_pitch]
        pitch_types:    torch.Tensor,  # [B, T]
        outcomes:       torch.Tensor,  # [B, T]
        batter_ctx:     torch.Tensor,  # [B, T, F_batter]
        batter_ids:     torch.Tensor,  # [B, T]
        pitcher_latent: torch.Tensor,  # [B, pitcher_latent_dim]  ← from VAE
        pitcher_id:     torch.Tensor,  # [B]
        batting_order:  torch.Tensor,  # [B, 9]
        game_ctx:       torch.Tensor,  # [B, F_game]
        batter_latent:  torch.Tensor,  # [B, batter_latent_dim]   ← mean over lineup VAE
        mask:           torch.Tensor,  # [B, T] True=valid
    ) -> torch.Tensor:                 # [B, T, d_model]

        B, T, _ = pitch_seq.shape

        b_emb  = self.batter_emb(batter_ids)
        pt_emb = self.ptype_emb(pitch_types)
        oc_emb = self.outcome_emb(outcomes)

        step_in = torch.cat([pitch_seq, batter_ctx, b_emb, pt_emb, oc_emb], dim=-1)
        x = self.step_proj(step_in)

        p_emb     = self.pitcher_emb(pitcher_id)
        ord_emb   = self.order_emb(batting_order).mean(dim=1)

        # Global conditioning uses VAE latents instead of raw stats
        global_in = torch.cat(
            [pitcher_latent, p_emb, game_ctx, ord_emb, batter_latent], dim=-1
        )
        g_vec = self.global_proj(global_in).unsqueeze(1)
        x = x + g_vec

        x = self.pos_enc(x)

        causal_mask = nn.Transformer.generate_square_subsequent_mask(T, device=x.device)
        key_pad     = ~mask

        x = self.transformer(x, mask=causal_mask, src_key_padding_mask=key_pad)
        return self.out_norm(x)


class DiffusionDenoiser(nn.Module):
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
        self.x_proj     = nn.Linear(len(PITCH_CONTINUOUS_COLS), t_dim)

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


class ClassificationHeads(nn.Module):
    """
    Dual classification heads:
      - pitch_type_head: predicts pitch_types[t+1] from context[t]
      - outcome_head:    predicts outcomes[t+1]    from context[t]

    Both are shifted-by-1 predictions — context[t] never contains the target.
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
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg      = cfg
        self.encoder  = ContextEncoder(cfg)
        self.denoiser = DiffusionDenoiser(cfg)
        self.heads    = ClassificationHeads(cfg)
        self.schedule = CosineNoiseSchedule(cfg.n_diffusion_steps)
        self.n_cont   = len(PITCH_CONTINUOUS_COLS)

        # Player VAEs — jointly trained with the main model
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

    def _encode_players(
        self,
        pitcher_ctx:   torch.Tensor,  # [B, F_pitcher]
        batter_ctx:    torch.Tensor,  # [B, T, F_batter] — per-pitch batter stats
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode pitcher and batter stats through their respective VAEs.

        For the batter, we encode each per-pitch batter stat vector, then
        mean-pool the latents over the sequence to get a single game-level
        batter representation for global conditioning.

        Returns:
            pitcher_z:       [B, pitcher_latent_dim]
            pitcher_recon_l: scalar
            pitcher_kl:      scalar
            batter_z_mean:   [B, batter_latent_dim]  — mean over sequence
            batter_kl:       scalar
        """
        # Pitcher VAE
        pitcher_z, pitcher_recon_l, pitcher_kl = self.pitcher_vae.vae_loss(pitcher_ctx)

        # Batter VAE — reshape [B, T, F] → [B*T, F], encode, reshape back
        B, T, F_bat = batter_ctx.shape
        batter_flat = batter_ctx.reshape(B * T, F_bat)
        batter_z_flat, batter_mu_flat, batter_logvar_flat = self.batter_vae(batter_flat)

        # KL over all batter positions
        batter_kl = PlayerVAE.kl_loss(batter_mu_flat, batter_logvar_flat)

        # Mean pool batter latents over sequence for global conditioning
        batter_z_seq  = batter_z_flat.reshape(B, T, self.cfg.batter_latent_dim)
        batter_z_mean = batter_z_seq.mean(dim=1)  # [B, batter_latent_dim]

        return pitcher_z, pitcher_recon_l, pitcher_kl, batter_z_mean, batter_kl

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
        pitcher_z, pitcher_recon_l, pitcher_kl, batter_z_mean, batter_kl = \
            self._encode_players(pitcher_ctx, batter_ctx)

        vae_loss = pitcher_recon_l + pitcher_kl + batter_kl

        # ── 2. Encode full sequence causally ──────────────────────────────
        context = self.encoder(
            pitch_seq, pitch_types, outcomes,
            batter_ctx, batter_ids,
            pitcher_z,      # ← VAE latent replaces raw pitcher_ctx
            pitcher_id,
            batting_order, game_ctx,
            batter_z_mean,  # ← VAE latent replaces raw batter mean pool
            mask,
        )  # [B, T, d_model]

        # ── 3. Classification losses (both shifted by 1) ───────────────────
        pt_logits, oc_logits = self.heads(context)

        pt_loss = F.cross_entropy(
            pt_logits[:, :-1].reshape(-1, self.cfg.num_pitch_types),
            pitch_types[:, 1:].reshape(-1),
            ignore_index=0,
            label_smoothing=0.05,
        )
        oc_loss = F.cross_entropy(
            oc_logits[:, :-1].reshape(-1, self.cfg.num_outcomes),
            outcomes[:, 1:].reshape(-1),
            ignore_index=0,
            label_smoothing=0.05,
        )

        # ── 4. Diffusion loss ─────────────────────────────────────────────
        x0 = pitch_seq[:, 1:, :self.n_cont].reshape(B * (T - 1), self.n_cont)
        t_diff = torch.randint(0, self.cfg.n_diffusion_steps, (B * (T - 1),), device=x0.device)
        x_t, noise = self.schedule.q_sample(x0, t_diff)

        ctx_for_diff = context[:, :-1].reshape(B * (T - 1), -1)
        pt_for_diff  = pitch_types[:, 1:].reshape(B * (T - 1))

        noise_pred = self.denoiser(x_t, t_diff, ctx_for_diff, pt_for_diff)

        valid     = mask[:, 1:].reshape(-1)
        diff_loss = F.mse_loss(noise_pred[valid], noise[valid])

        # ── 5. Weighted total loss ─────────────────────────────────────────
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
            "pitcher_recon": pitcher_recon_l.detach(),
            "pitcher_kl":    pitcher_kl.detach(),
            "batter_kl":     batter_kl.detach(),
        }

    @torch.no_grad()
    def _get_player_latents(
        self,
        pitcher_ctx:   torch.Tensor,  # [B, F_pitcher]
        batter_ctx:    torch.Tensor,  # [B, T, F_batter] or [B, F_batter]
        is_sequence:   bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get deterministic player latents (μ, no sampling) for inference.
        Returns (pitcher_z, batter_z_mean).
        """
        pitcher_mu, _ = self.pitcher_vae.encode(pitcher_ctx)

        if is_sequence:
            B, T, F = batter_ctx.shape
            batter_flat = batter_ctx.reshape(B * T, F)
            batter_mu_flat, _ = self.batter_vae.encode(batter_flat)
            batter_z_mean = batter_mu_flat.reshape(B, T, self.cfg.batter_latent_dim).mean(dim=1)
        else:
            batter_z_mean, _ = self.batter_vae.encode(batter_ctx)

        return pitcher_mu, batter_z_mean

    @torch.no_grad()
    def encode_context(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        pitcher_z, batter_z_mean = self._get_player_latents(
            batch["pitcher_ctx"], batch["batter_ctx"], is_sequence=True
        )
        return self.encoder(
            batch["pitch_seq"], batch["pitch_types"], batch["outcomes"],
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
        pt_logits, _ = self.heads(context_vec)
        return F.softmax(pt_logits / temperature, dim=-1)

    @torch.no_grad()
    def predict_next_outcome(self, context_vec: torch.Tensor,
                             temperature: float = 0.85) -> torch.Tensor:
        _, oc_logits = self.heads(context_vec)
        return F.softmax(oc_logits / temperature, dim=-1)

    @torch.no_grad()
    def sample_pitch_features(self, context_vec: torch.Tensor,
                               pitch_type: torch.Tensor,
                               ddim_steps: int = 10) -> torch.Tensor:
        B      = context_vec.shape[0]
        device = context_vec.device

        def _model_fn(x_t, t_batch):
            return self.denoiser(x_t, t_batch, context_vec, pitch_type)

        return self.schedule.ddim_sample(
            _model_fn, (B, self.n_cont), device, n_steps=ddim_steps
        )

    @torch.no_grad()
    def get_batter_latent(self, batter_stats: torch.Tensor) -> torch.Tensor:
        """Get batter latent for a single batter during simulation. [1, F] → [1, latent_dim]"""
        mu, _ = self.batter_vae.encode(batter_stats)
        return mu

    @torch.no_grad()
    def get_pitcher_latent(self, pitcher_stats: torch.Tensor) -> torch.Tensor:
        """Get pitcher latent for a single pitcher during simulation. [1, F] → [1, latent_dim]"""
        mu, _ = self.pitcher_vae.encode(pitcher_stats)
        return mu


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

    def is_game_over(self, min_innings: int = 9) -> bool:
        if self.inning < min_innings:
            return False
        if self.is_top and self.inning > min_innings and self.home_score != self.away_score:
            return True
        if self.is_top and self.inning > min_innings:
            return self.home_score != self.away_score
        return False

    def is_walkoff(self) -> bool:
        return (
            not self.is_top
            and self.inning >= 9
            and self.home_score > self.away_score
        )

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
        self.outs   = 0
        self.on_1b  = self.on_2b = self.on_3b = False
        self.balls  = self.strikes = 0
        if self.is_top:
            self.is_top = False
        else:
            self.inning += 1
            self.is_top = True
            if self.inning > 9:
                self.on_2b = True

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
        f"_vae{cfg_model.pitcher_latent_dim}"
    )


def _save_loss_plots(
    history:   Dict[str, List[float]],
    cfg_model: ModelConfig,
    cfg_train: TrainConfig,
    out_dir:   Path,
):
    if not _HAVE_MPL:
        print("[train] matplotlib not available — skipping loss plots")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    tag    = _model_tag(cfg_model)
    n_ep   = len(history["train_loss"])
    epochs = list(range(1, n_ep + 1))

    # Plot 1: total train + val loss
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(epochs, history["train_loss"], label="Train Loss",  color="steelblue", linewidth=1.8)
    ax.plot(epochs, history["val_loss"],   label="Val Loss",    color="tomato",    linewidth=1.8)
    best_ep = int(np.argmin(history["val_loss"])) + 1
    ax.axvline(best_ep, color="tomato", linestyle="--", alpha=0.55,
               label=f"Best val epoch={best_ep}")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.set_title(
        f"TransFusion — Total Loss\n"
        f"d={cfg_model.d_model}  L={cfg_model.n_layers}  H={cfg_model.n_heads}  "
        f"VAE latent={cfg_model.pitcher_latent_dim}/{cfg_model.batter_latent_dim}"
    )
    ax.legend(); ax.grid(True, alpha=0.3); fig.tight_layout()
    fname1 = out_dir / f"transfusion_loss_total_{tag}_ep{n_ep}.png"
    fig.savefig(fname1, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"[train] Loss plot (total)      → {fname1}")

    # Plot 2: component losses (4 panels now including VAE)
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    components = [
        ("diff_loss", "Diffusion MSE",   "steelblue"),
        ("pt_loss",   "Pitch-Type CE",   "seagreen"),
        ("oc_loss",   "Outcome CE",      "darkorange"),
        ("vae_loss",  "VAE Loss",        "mediumpurple"),
    ]
    for ax, (key, label, color) in zip(axes, components):
        train_vals = history.get(f"train_{key}", [])
        val_vals   = history.get(f"val_{key}",   [])
        ax.plot(epochs[:len(train_vals)], train_vals,
                label="Train", color=color, linewidth=1.8)
        if val_vals:
            ax.plot(epochs[:len(val_vals)], val_vals,
                    label="Val", color=color, linewidth=1.8, linestyle="--", alpha=0.8)
        ax.axvline(best_ep, color="tomato", linestyle=":", alpha=0.5)
        ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
        ax.set_title(label); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    fig.suptitle(
        f"TransFusion — Component Losses  |  "
        f"d={cfg_model.d_model}  L={cfg_model.n_layers}  H={cfg_model.n_heads}",
        fontsize=10,
    )
    fig.tight_layout()
    fname2 = out_dir / f"transfusion_loss_components_{tag}_ep{n_ep}.png"
    fig.savefig(fname2, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"[train] Loss plot (components) → {fname2}")


def train(cfg_train: TrainConfig, cfg_model: ModelConfig):
    torch.manual_seed(cfg_train.seed)
    np.random.seed(cfg_train.seed)
    random.seed(cfg_train.seed)

    device   = _resolve_device(cfg_train.device)
    ckpt_dir = Path(cfg_train.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    print(f"[train] Device: {device}")

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

    model = TransFusion(cfg_model).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[train] TransFusion parameters: {n_params:,}")

    optimizer   = torch.optim.AdamW(
        model.parameters(), lr=cfg_train.lr,
        weight_decay=cfg_train.weight_decay,
        betas=(0.9, 0.98), eps=1e-9,
    )
    total_steps = cfg_train.epochs * len(train_loader)
    scheduler   = get_lr_scheduler(optimizer, cfg_train.warmup_steps, total_steps)
    scaler_amp  = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    _save_json(vars(cfg_model), ckpt_dir / "model_config.json")

    best_val_loss = float("inf")
    global_step   = 0

    history: Dict[str, List[float]] = {
        "train_loss": [], "val_loss": [],
        "train_diff_loss": [], "train_pt_loss": [], "train_oc_loss": [], "train_vae_loss": [],
        "val_diff_loss":   [], "val_pt_loss":   [], "val_oc_loss":   [], "val_vae_loss":   [],
    }

    for epoch in range(1, cfg_train.epochs + 1):
        model.train()
        epoch_losses = {
            "loss": 0.0, "diff_loss": 0.0, "pt_loss": 0.0,
            "oc_loss": 0.0, "vae_loss": 0.0,
        }
        t0 = time.time()

        for step, batch in tqdm(
            enumerate(train_loader), total=len(train_loader),
            desc=f"Epoch {epoch}/{cfg_train.epochs}"
        ):
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
        print(f"[epoch {epoch:3d}] train_loss={epoch_losses['loss']/n:.4f}  "
              f"({time.time()-t0:.0f}s)")

        history["train_loss"].append(epoch_losses["loss"] / n)
        history["train_diff_loss"].append(epoch_losses["diff_loss"] / n)
        history["train_pt_loss"].append(epoch_losses["pt_loss"] / n)
        history["train_oc_loss"].append(epoch_losses["oc_loss"] / n)
        history["train_vae_loss"].append(epoch_losses["vae_loss"] / n)

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
    _save_loss_plots(
        history=history, cfg_model=cfg_model, cfg_train=cfg_train,
        out_dir=Path(cfg_train.checkpoint_dir) / "plots",
    )


@torch.no_grad()
def _evaluate(
    model: TransFusion, loader: DataLoader, device: torch.device
) -> Dict[str, float]:
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


def simulate_games_mh(cfg_sim: SimConfig, cfg_model: ModelConfig,
                       lam: float = 1.0, n_steps: int = 500,
                       burn_in: int = 100):
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
    print(f"[mh-sim] Loaded checkpoint (epoch {ckpt.get('epoch','?')})")
    print(f"[mh-sim] λ={lam}  steps={n_steps}  burn_in={burn_in}")

    builder = BaseballDatasetBuilder(
        start_dt=cfg_sim.start_dt, end_dt=cfg_sim.end_dt,
        val_start_dt=cfg_sim.val_start_dt, test_start_dt=cfg_sim.test_start_dt,
        cache_dir=cfg_sim.cache_dir, max_seq_len=cfg_model.max_seq_len,
        min_pitches_per_game=100,
    )
    train_ds, val_ds, test_ds, encoders = builder.build()
    _load_re24_table(cfg_sim.cache_dir)
    _load_in_play_probs(cfg_sim.cache_dir)

    dataset: PitchSequenceDataset = {
        "train": train_ds, "val": val_ds, "test": test_ds
    }[cfg_sim.split]
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

        current_traj = _simulate_one_game(
            model, encoders, dataset.pitch_scaler, dataset.batter_scaler,
            ctx_vec_1.expand(1, -1), context_state,
            sample["batting_order"].tolist(), device, cfg_sim.max_game_pitches,
            temperature=cfg_sim.temperature,
        )

        win_votes = 0
        n_accepted = 0
        post_burn_count = 0

        for step in range(n_steps + burn_in):
            boundaries = current_traj["half_inning_boundaries"]
            m_tau      = len(boundaries)
            if m_tau == 0:
                proposed_traj = current_traj
                accepted      = True
            else:
                split_idx  = random.randint(0, m_tau - 1)
                split_info = boundaries[split_idx]
                split_ctx_vec = _get_context_at_boundary(
                    model, sample, ctx_vec_1, split_info, device,
                    dataset.pitch_scaler, dataset.batter_scaler, encoders,
                    context_end_idx, cfg_sim.max_game_pitches,
                )
                split_gs = split_info["game_state"]
                proposed_traj = _simulate_one_game(
                    model, encoders, dataset.pitch_scaler, dataset.batter_scaler,
                    split_ctx_vec.expand(1, -1), split_gs,
                    sample["batting_order"].tolist(), device, cfg_sim.max_game_pitches,
                    prefix_phi_sum=split_info["prefix_phi_sum"],
                    temperature=cfg_sim.temperature,
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
                    current_traj = proposed_traj
                    n_accepted  += 1

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
            correct = (
                (result.actual_home_win and result.home_win_prob > 0.5)
                or (not result.actual_home_win and result.away_win_prob > 0.5)
            )
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
    temperature: float = 0.85,
) -> Dict:
    oc_idx2str = {v: k for k, v in encoders.outcome.items()}

    gs          = _clone_game_state(start_gs)
    current_ctx = ctx_vec_1.squeeze(0).clone()

    half_inning_boundaries = []
    total_phi              = prefix_phi_sum
    pitch_count            = 0
    hi_phi_sum             = 0.0

    while not gs.is_game_over() and pitch_count < max_pitches:
        ctx_active = current_ctx.unsqueeze(0)

        pt_probs  = model.predict_next_pitch_type(ctx_active, temperature=temperature)
        pt_sample = torch.multinomial(pt_probs, 1).squeeze(-1)

        pitch_feat = model.sample_pitch_features(ctx_active, pt_sample, ddim_steps=10)

        oc_probs  = model.predict_next_outcome(ctx_active, temperature=temperature)
        oc_sample = torch.multinomial(oc_probs, 1).squeeze(-1)

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

        gs_feats   = torch.tensor(gs.to_feature_vec(pitch_scaler), device=device, dtype=torch.float32)
        batter_tok = batting_order_raw[gs.batting_idx] if gs.batting_idx < len(batting_order_raw) else 0
        b_ctx      = torch.tensor(
            batter_scaler.transform_row({}, BATTER_STAT_COLS),
            device=device, dtype=torch.float32,
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
    print(f"[sim] Loaded checkpoint from {cfg_sim.checkpoint}  "
          f"(epoch {ckpt.get('epoch','?')})")

    builder = BaseballDatasetBuilder(
        start_dt=cfg_sim.start_dt, end_dt=cfg_sim.end_dt,
        val_start_dt=cfg_sim.val_start_dt, test_start_dt=cfg_sim.test_start_dt,
        cache_dir=cfg_sim.cache_dir, max_seq_len=cfg_model.max_seq_len,
        min_pitches_per_game=100,
    )
    train_ds, val_ds, test_ds, encoders = builder.build()
    _load_in_play_probs(cfg_sim.cache_dir)

    dataset: PitchSequenceDataset = {
        "train": train_ds, "val": val_ds, "test": test_ds
    }[cfg_sim.split]
    print(f"[sim] Simulating {len(dataset)} games from '{cfg_sim.split}' split")
    print(f"[sim] Context: {cfg_sim.context_innings} innings  |  "
          f"N simulations: {cfg_sim.n_simulations}  |  "
          f"Temperature: {cfg_sim.temperature}")

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

        context_state = _reconstruct_game_state(game_df, context_end_idx)
        K       = cfg_sim.n_simulations
        ctx_vec = context_memory[:, ctx_out_idx, :].expand(K, -1)

        h_scores, a_scores = _run_parallel_simulations(
            model=model, encoders=encoders,
            pitch_scaler=dataset.pitch_scaler,
            batter_scaler=dataset.batter_scaler,
            pitcher_scaler=dataset.pitcher_scaler,
            ctx_vec=ctx_vec, game_state_template=context_state,
            sample=sample, game_df=game_df,
            context_end_idx=context_end_idx,
            K=K, device=device, max_pitches=cfg_sim.max_game_pitches,
            cfg_model=cfg_model, temperature=cfg_sim.temperature,
        )

        home_wins = sum(h > a for h, a in zip(h_scores, a_scores))
        away_wins = sum(a > h for h, a in zip(h_scores, a_scores))

        result = SimResult(
            game_pk=game_pk, context_innings=cfg_sim.context_innings,
            n_simulations=K,
            home_win_prob=home_wins / K, away_win_prob=away_wins / K,
            tie_prob=0.0,
            mean_home_runs=float(np.mean(h_scores)),
            mean_away_runs=float(np.mean(a_scores)),
            std_home_runs=float(np.std(h_scores)),
            std_away_runs=float(np.std(a_scores)),
            actual_home_score=actual_home, actual_away_score=actual_away,
            actual_home_win=actual_home > actual_away,
        )
        results.append(result)

        if (game_idx + 1) % 10 == 0 or game_idx == 0:
            correct = (
                (result.actual_home_win and result.home_win_prob > 0.5)
                or (not result.actual_home_win and result.away_win_prob > 0.5)
            )
            print(f"  [{game_idx+1:4d}/{len(dataset)}] game={game_pk}  "
                  f"P(home)={result.home_win_prob:.3f}  "
                  f"actual={'H' if actual_home > actual_away else 'A'}  "
                  f"{'✓' if correct else '✗'}")

    out_path = out_dir / f"sim_results_{cfg_sim.context_innings}inn.json"
    with open(out_path, "w") as f:
        json.dump([vars(r) for r in results], f, indent=2)
    print(f"[sim] Saved {len(results)} results to {out_path}")
    _print_summary(results)
    return results


# ---------------------------------------------------------------------------
# Vectorized simulation helpers
# ---------------------------------------------------------------------------

_IN_PLAY_EVENTS_ARR = np.array([
    "single", "double", "triple", "home_run",
    "field_out", "force_out", "double_play", "grounded_into_double_play",
    "field_error", "sac_fly",
])
_IN_PLAY_PROBS_ARR = np.array([
    0.2106, 0.0654, 0.0056, 0.0460,
    0.5926, 0.0298, 0.0031, 0.0274,
    0.0096, 0.0098,
])
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
            probs = np.array(
                [probs_dict.get(e, 0.0) for e in _IN_PLAY_EVENTS_ARR], dtype=float
            )
            s = probs.sum()
            if s > 0:
                probs /= s
            _IN_PLAY_PROBS_ARR = probs
            _IN_PLAY_CUMPROBS  = np.cumsum(_IN_PLAY_PROBS_ARR)
            print(f"[in-play] Loaded in-play probs from {path}")
            return
    warnings.warn("[in-play] No in_play_probs.json found; using empirical fallback.")


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
    temperature: float = 0.85,
) -> Tuple[List[float], List[float]]:
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
        if oc_str in BALL_OUTCOMES or oc_str in {
            "hit_by_pitch", "automatic_ball", "bunt_foul_tip",
            "foul_pitchout", "pitchout", "intent_ball", "blocked_ball"
        }:
            return OC_BALL
        if oc_str in STRIKE_OUTCOMES:  return OC_STRIKE
        if oc_str in FOUL_OUTCOMES:    return OC_FOUL
        if oc_str in IN_PLAY_OUTCOMES: return OC_IN_PLAY
        return OC_OTHER

    oc_class = np.array(
        [_classify_outcome(oc_idx2str.get(i, "ball"))
         for i in range(max(oc_idx2str.keys()) + 1)],
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

    zero_b_ctx = torch.tensor(
        batter_scaler.transform_row({}, BATTER_STAT_COLS),
        dtype=torch.float32, device=device,
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

        ctx_active  = current_ctx[active]
        pt_probs    = model.predict_next_pitch_type(ctx_active, temperature=temperature)
        pt_samples  = torch.multinomial(pt_probs, 1).squeeze(-1)
        pitch_feats = model.sample_pitch_features(ctx_active, pt_samples, ddim_steps=10)
        oc_probs    = model.predict_next_outcome(ctx_active, temperature=temperature)
        oc_samples  = torch.multinomial(oc_probs, 1).squeeze(-1)
        oc_np       = oc_samples.cpu().numpy()

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

        n_in_play      = int(is_in_play.sum())
        in_play_events = _vec_sample_in_play(n_in_play) if n_in_play > 0 else np.array([])
        in_play_ptr    = 0

        for local_i in range(n_act):
            sim_i = active[local_i]
            if walk_from_ball[local_i]:
                _vec_apply_walk(sim_i, inning, is_top, home_score, away_score,
                                on_1b, on_2b, on_3b, walkoff_mask)
                balls[sim_i] = strikes[sim_i] = 0
                batting_idx[sim_i] = (batting_idx[sim_i] + 1) % 9
            elif ko_from_strike[local_i]:
                outs[sim_i] += 1
                balls[sim_i] = strikes[sim_i] = 0
                batting_idx[sim_i] = (batting_idx[sim_i] + 1) % 9
                if outs[sim_i] >= 3:
                    _vec_end_half_inning(sim_i, inning, is_top, outs,
                                         on_1b, on_2b, on_3b, balls, strikes)
            elif is_in_play[local_i]:
                ev = in_play_events[in_play_ptr]; in_play_ptr += 1
                _vec_apply_event(sim_i, ev, inning, is_top, outs,
                                  on_1b, on_2b, on_3b, home_score, away_score,
                                  balls, strikes, batting_idx, walkoff_mask)

        gs_feats    = _build_gs_feats_batch(active)
        b_idxs      = batting_idx[active] % n_batters
        batter_toks = torch.tensor(
            [batting_order_arr[b] for b in b_idxs], dtype=torch.long, device=device
        )
        b_ctx_batch = zero_b_ctx.unsqueeze(0).expand(n_act, -1)

        new_ctxs = _incremental_encode_step_batch(
            model=model, prev_ctx=ctx_active,
            pitch_feats=pitch_feats, gs_feats=gs_feats,
            b_ctx=b_ctx_batch, batter_ids=batter_toks,
            pt_tokens=pt_samples, oc_tokens=oc_samples,
        )
        current_ctx[active] = new_ctxs

        game_over   = (inning > 9) & is_top & (home_score != away_score)
        active_mask = ~game_over & ~walkoff_mask
        pitch_count += 1

    return home_score.tolist(), away_score.tolist()


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


def _vec_end_half_inning(sim_i, inning, is_top, outs, on_1b, on_2b, on_3b,
                          balls, strikes):
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
            _vec_end_half_inning(sim_i, inning, is_top, outs, on_1b, on_2b,
                                  on_3b, balls, strikes)
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


def _incremental_encode_step_batch(
    model, prev_ctx, pitch_feats, gs_feats, b_ctx, batter_ids, pt_tokens, oc_tokens,
) -> torch.Tensor:
    enc     = model.encoder
    b_emb   = enc.batter_emb(batter_ids)
    pt_emb  = enc.ptype_emb(pt_tokens)
    oc_emb  = enc.outcome_emb(oc_tokens)
    step_in = torch.cat([pitch_feats, gs_feats, b_ctx, b_emb, pt_emb, oc_emb], dim=-1)
    new_ctx = enc.step_proj(step_in) + prev_ctx
    return enc.out_norm(new_ctx)


def _incremental_encode_step(
    model, prev_ctx, pitch_feats, gs_feats, b_ctx, batter_id, pt_token, oc_token,
) -> torch.Tensor:
    enc     = model.encoder
    b_emb   = enc.batter_emb(batter_id)
    pt_emb  = enc.ptype_emb(pt_token)
    oc_emb  = enc.outcome_emb(oc_token)
    step_in = torch.cat([pitch_feats, gs_feats, b_ctx, b_emb, pt_emb, oc_emb], dim=-1)
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


def _make_context_batch(sample: Dict[str, torch.Tensor],
                        context_end_idx: int,
                        device: torch.device) -> Dict[str, torch.Tensor]:
    keys_to_slice = ["pitch_seq", "pitch_types", "outcomes", "at_bat_events",
                     "batter_ctx", "batter_ids", "mask"]
    batch = {k: sample[k][:context_end_idx].unsqueeze(0).to(device)
             for k in keys_to_slice}
    for k in ["pitcher_ctx", "pitcher_id", "batting_order", "game_ctx"]:
        batch[k] = sample[k].unsqueeze(0).to(device)
    return batch


def _make_empty_context_batch(sample: Dict[str, torch.Tensor],
                               device: torch.device) -> Dict[str, torch.Tensor]:
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
    n = len(results)
    if n == 0:
        return
    correct = sum(
        1 for r in results
        if r.actual_home_win is not None
        and ((r.actual_home_win and r.home_win_prob > 0.5)
             or (not r.actual_home_win and r.away_win_prob > 0.5))
    )
    brier = np.mean([
        (r.home_win_prob - float(r.actual_home_win)) ** 2
        for r in results if r.actual_home_win is not None
    ])
    ll = []
    for r in results:
        if r.actual_home_win is None:
            continue
        p = r.home_win_prob if r.actual_home_win else r.away_win_prob
        ll.append(-math.log(max(p, 1e-7)))

    print("\n" + "=" * 55)
    print(f"  SIMULATION SUMMARY  ({n} games)")
    print("=" * 55)
    print(f"  Accuracy        : {correct / n:.4f}  ({correct}/{n})")
    print(f"  Brier Score     : {brier:.4f}")
    print(f"  Log Loss        : {np.mean(ll):.4f}")
    print(f"  Mean P(home win): {np.mean([r.home_win_prob for r in results]):.4f}")
    print("=" * 55 + "\n")


# =============================================================================
# 8.  UTILITIES
# =============================================================================

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
    p_train.add_argument("--cache_dir",         default="./baseball_cache")
    p_train.add_argument("--checkpoint_dir",    default="./checkpoints")
    p_train.add_argument("--start_dt",          default="2015-03-22")
    p_train.add_argument("--end_dt",            default="2026-05-01")
    p_train.add_argument("--val_start_dt",      default="2025-03-20")
    p_train.add_argument("--test_start_dt",     default="2026-03-25")
    p_train.add_argument("--epochs",            type=int,   default=60)
    p_train.add_argument("--batch_size",        type=int,   default=16)
    p_train.add_argument("--lr",                type=float, default=2e-4)
    p_train.add_argument("--d_model",           type=int,   default=384)
    p_train.add_argument("--n_layers",          type=int,   default=10)
    p_train.add_argument("--n_heads",           type=int,   default=8)
    p_train.add_argument("--n_diff_steps",      type=int,   default=50)
    p_train.add_argument("--pitcher_latent_dim",type=int,   default=32)
    p_train.add_argument("--batter_latent_dim", type=int,   default=32)
    p_train.add_argument("--lambda_vae",        type=float, default=0.05)
    p_train.add_argument("--num_workers",       type=int,   default=0)
    p_train.add_argument("--device",            default="auto")
    p_train.add_argument("--seed",              type=int,   default=42)

    p_sim = sub.add_parser("simulate")
    p_sim.add_argument("--checkpoint",       required=True)
    p_sim.add_argument("--cache_dir",        default="./baseball_cache")
    p_sim.add_argument("--start_dt",         default="2021-04-07")
    p_sim.add_argument("--end_dt",           default="2026-05-01")
    p_sim.add_argument("--val_start_dt",     default="2025-03-20")
    p_sim.add_argument("--test_start_dt",    default="2026-03-25")
    p_sim.add_argument("--context_innings",  type=float, default=3.0)
    p_sim.add_argument("--n_simulations",    type=int,   default=500)
    p_sim.add_argument("--split",            default="test",
                       choices=["train", "val", "test"])
    p_sim.add_argument("--out_dir",          default="./sim_results")
    p_sim.add_argument("--device",           default="auto")
    p_sim.add_argument("--mh",               action="store_true")
    p_sim.add_argument("--lam",              type=float, default=1.0)
    p_sim.add_argument("--n_steps",          type=int,   default=500)
    p_sim.add_argument("--burn_in",          type=int,   default=100)
    p_sim.add_argument("--temperature",      type=float, default=0.85)

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
                               lam=args.lam, n_steps=args.n_steps,
                               burn_in=args.burn_in)
        else:
            simulate_games(cfg_sim, ModelConfig())

    else:
        print("Usage: python new_transfusion.py [train|simulate] --help")