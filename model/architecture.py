"""
PitchSequenceTransfusion — model architecture.

Transfusion-style model for pitch-by-pitch baseball game simulation.
One mathematical concept per component; no training logic here.

Architecture:
    Single transformer backbone with causal attention across pitch steps.
    Each timestep encodes the previous pitch (discrete + continuous) and the
    current game state, then produces predictions for the next pitch.

    Discrete outputs  → cross-entropy loss (LM head)
    Continuous outputs → DDPM diffusion loss (conditioned on transformer latent)

Input per timestep (concatenated into d_model):
    prev_pitch_type   (embedded, d_model // 4)
    prev_zone         (embedded, d_model // 4)
    prev_pitch_result (embedded, d_model // 4)
    [context, game_state, prev_continuous] → projected to remaining_dim

Output per timestep:
    pitch_type_logits    (n_pitch_types)
    zone_logits          (n_zones)
    pitch_result_logits  (n_pitch_results)
    at_bat_event_logits  (n_at_bat_events)
    continuous_latent    (d_model) — conditioning vector for DDPM
"""

import math

import torch
import torch.nn as nn

from model.vocab import (
    PITCH_TYPES, ZONES, PITCH_RESULTS, AT_BAT_EVENTS, CONTINUOUS_PITCH_COLS,
)
from model.diffusion import GaussianDiffusion1D, Unet1D


class SinusoidalPositionalEncoding(nn.Module):
    """Standard fixed sinusoidal positional encoding (Vaswani et al., 2017)."""

    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, seq_len: int) -> torch.Tensor:
        return self.pe[:seq_len]  # (seq_len, d_model)


class PitchSequenceTransfusion(nn.Module):
    """Transfusion model for pitch sequence prediction.

    Each forward pass takes a batch of pitch sequences (teacher-forced) and
    returns logits for all discrete heads plus a continuous latent for DDPM.
    No training logic lives here — see model/train.py.
    """

    def __init__(
        self,
        context_dim: int,
        n_pitch_types: int = len(PITCH_TYPES),
        n_zones: int = len(ZONES),
        n_pitch_results: int = len(PITCH_RESULTS),
        n_at_bat_events: int = len(AT_BAT_EVENTS),
        n_continuous: int = len(CONTINUOUS_PITCH_COLS),
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 6,
        dropout: float = 0.1,
        max_seq_len: int = 256,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_pitch_types = n_pitch_types
        self.n_zones = n_zones
        self.n_pitch_results = n_pitch_results
        self.n_at_bat_events = n_at_bat_events
        self.n_continuous = n_continuous

        # ── Discrete embeddings for previous pitch tokens ──
        # +1 for the <start> token used at at-bat boundaries.
        self.pitch_type_embed   = nn.Embedding(n_pitch_types + 1,   d_model // 4)
        self.zone_embed         = nn.Embedding(n_zones + 1,         d_model // 4)
        self.pitch_result_embed = nn.Embedding(n_pitch_results + 1, d_model // 4)

        # ── Continuous projection ──
        # game_state: balls(1) + strikes(1) + outs(1) + inning(1) + score_diff(1)
        #             + on_1b(1) + on_2b(1) + on_3b(1) + is_top(1) = 9
        game_state_dim = 9
        continuous_input_dim = context_dim + game_state_dim + n_continuous
        remaining_dim = d_model - 3 * (d_model // 4)
        self.continuous_proj = nn.Linear(continuous_input_dim, remaining_dim)

        # ── Positional encoding ──
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len=max_seq_len)

        # ── Causal transformer ──
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # ── Discrete output heads (LM loss) ──
        def _head(out_dim: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.LayerNorm(d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, out_dim),
            )

        self.pitch_type_head     = _head(n_pitch_types)
        self.zone_head           = _head(n_zones)
        self.pitch_result_head   = _head(n_pitch_results)
        self.at_bat_event_head   = _head(n_at_bat_events)

        # ── Continuous head: DDPM conditioned on transformer latent ──
        self.ddpm_unet = Unet1D(
            dim=d_model,
            channels=n_continuous,
            dim_mults=(1, 2, 4),
            self_condition=False,
            cond_dim=d_model,
        )
        self.ddpm = GaussianDiffusion1D(
            model=self.ddpm_unet,
            seq_length=max_seq_len,
            timesteps=1000,
            sampling_timesteps=20,      # DDIM fast sampling for inference
            loss_type="l2",
            objective="pred_noise",
            beta_schedule="cosine",
        )

        self._init_weights()

    def _init_weights(self) -> None:
        ddpm_modules = set(self.ddpm.modules())
        for m in self.modules():
            if m in ddpm_modules:
                continue
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def _causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Upper-triangular boolean mask for causal self-attention."""
        return torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()

    def forward(
        self,
        context: torch.Tensor,           # (B, S, context_dim)
        game_state: torch.Tensor,        # (B, S, 9)
        prev_pitch_type: torch.Tensor,   # (B, S) long
        prev_zone: torch.Tensor,         # (B, S) long
        prev_pitch_result: torch.Tensor, # (B, S) long
        prev_continuous: torch.Tensor,   # (B, S, n_continuous)
    ) -> dict[str, torch.Tensor]:
        B, S = prev_pitch_type.shape

        pt_emb  = self.pitch_type_embed(prev_pitch_type)       # (B, S, d//4)
        z_emb   = self.zone_embed(prev_zone)                   # (B, S, d//4)
        pr_emb  = self.pitch_result_embed(prev_pitch_result)   # (B, S, d//4)

        cont_input = torch.cat([context, game_state, prev_continuous], dim=-1)
        cont_emb = self.continuous_proj(cont_input)            # (B, S, remaining)

        x = torch.cat([pt_emb, z_emb, pr_emb, cont_emb], dim=-1)  # (B, S, d_model)
        x = x + self.pos_enc(S).unsqueeze(0)
        x = self.transformer(x, mask=self._causal_mask(S, x.device))

        return {
            "pitch_type_logits":   self.pitch_type_head(x),
            "zone_logits":         self.zone_head(x),
            "pitch_result_logits": self.pitch_result_head(x),
            "at_bat_event_logits": self.at_bat_event_head(x),
            "continuous_latent":   x,   # conditioning vector for DDPM
        }
