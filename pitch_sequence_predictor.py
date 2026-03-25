"""
Pitch Sequence Predictor — Transfusion-style model for full baseball game simulation.

Architecture (Transfusion):
  - Single transformer backbone with causal attention across pitch steps
  - Cross-entropy loss on discrete outputs (pitch_type, zone, pitch_result, at_bat_event)
  - Gaussian NLL loss on continuous outputs (velocity, spin, movement, location)
  - Autoregressive generation: context → pitch → result → next pitch → ...

Hierarchy:
  Pitch level:   predict pitch_type, zone, continuous attrs, pitch_result
  At-bat level:  when pitch_result is terminal (hit_into_play, strikeout, walk, HBP),
                 predict at_bat_event (single, double, field_out, etc.)
  Half-inning:   track outs, switch sides at 3 outs
  Game level:    9 innings, MLB extra-inning rules
"""

import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from collections import Counter

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

PITCH_TYPES = sorted([
    'CH', 'CS', 'CU', 'EP', 'FA', 'FC', 'FF', 'FS',
    'KC', 'KN', 'SI', 'SL', 'ST', 'SV',
])

PITCH_RESULTS = sorted([
    'ball', 'blocked_ball', 'called_strike', 'foul', 'foul_bunt',
    'foul_tip', 'hit_by_pitch', 'hit_into_play', 'missed_bunt',
    'swinging_strike', 'swinging_strike_blocked',
    'automatic_ball', 'automatic_strike', 'pitchout',
    'bunt_foul_tip', 'foul_pitchout', 'intent_ball',
])

AT_BAT_EVENTS = sorted([
    'catcher_interf', 'double', 'double_play', 'field_error',
    'field_out', 'fielders_choice', 'fielders_choice_out',
    'force_out', 'grounded_into_double_play', 'hit_by_pitch',
    'home_run', 'intent_walk', 'sac_bunt', 'sac_bunt_double_play',
    'sac_fly', 'sac_fly_double_play', 'single', 'strikeout',
    'strikeout_double_play', 'triple', 'triple_play', 'truncated_pa',
    'walk',
])

ZONES = list(range(1, 15))  # 1-14

# Continuous pitch features (model predicts these via Gaussian NLL / diffusion)
CONTINUOUS_PITCH_COLS = [
    'release_speed', 'plate_x', 'plate_z', 'pfx_x', 'pfx_z', 'release_spin_rate'
]

# Pitch results that end an at-bat
TERMINAL_RESULTS = {
    'hit_into_play',   # → need at_bat_event prediction
    'hit_by_pitch',    # → HBP event
}

# Results that are strikes
STRIKE_RESULTS = {
    'called_strike', 'swinging_strike', 'swinging_strike_blocked',
    'foul_tip', 'missed_bunt', 'automatic_strike',
}

# Results that are fouls (strike only if < 2 strikes)
FOUL_RESULTS = {
    'foul', 'foul_bunt', 'bunt_foul_tip', 'foul_pitchout',
}

# Results that are balls
BALL_RESULTS = {
    'ball', 'blocked_ball', 'automatic_ball', 'pitchout', 'intent_ball',
}


def build_vocab_maps():
    """Build string↔index mappings for all categorical variables."""
    pt_to_idx = {pt: i for i, pt in enumerate(PITCH_TYPES)}
    pr_to_idx = {pr: i for i, pr in enumerate(PITCH_RESULTS)}
    ev_to_idx = {ev: i for i, ev in enumerate(AT_BAT_EVENTS)}
    zone_to_idx = {z: i for i, z in enumerate(ZONES)}
    return pt_to_idx, pr_to_idx, ev_to_idx, zone_to_idx


# ─────────────────────────────────────────────────────────────────────────────
# Positional encoding
# ─────────────────────────────────────────────────────────────────────────────

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, seq_len):
        return self.pe[:seq_len]


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────

class PitchSequenceTransfusion(nn.Module):
    """
    Transfusion-style model for pitch sequence prediction.

    Each step in the sequence represents one pitch. The model:
      1. Embeds discrete inputs (pitch_type, zone, pitch_result from *previous* pitch)
      2. Projects continuous inputs (pitch attrs from previous pitch + game state)
      3. Runs through a causal transformer
      4. Predicts next pitch's discrete outputs (LM loss) and continuous outputs (NLL loss)

    Input per timestep (concatenated into one vector):
      - Game context features (static, repeated)
      - Count state: balls, strikes (updated by game engine)
      - Outs, inning, score, baserunners (updated by game engine)
      - Previous pitch embedding: pitch_type, zone, pitch_result (discrete → embedded)
      - Previous pitch continuous: release_speed, plate_x, plate_z, pfx_x, pfx_z, spin_rate

    Output per timestep:
      - pitch_type logits         (n_pitch_types)    — CE loss
      - zone logits               (n_zones)          — CE loss
      - pitch_result logits       (n_pitch_results)  — CE loss
      - at_bat_event logits       (n_at_bat_events)  — CE loss (only on terminal pitches)
      - continuous mean           (n_continuous)      — Gaussian NLL loss
      - continuous logvar         (n_continuous)      — Gaussian NLL loss
    """

    def __init__(
        self,
        context_dim,           # Dimension of static game context features
        n_pitch_types=len(PITCH_TYPES),
        n_zones=len(ZONES),
        n_pitch_results=len(PITCH_RESULTS),
        n_at_bat_events=len(AT_BAT_EVENTS),
        n_continuous=len(CONTINUOUS_PITCH_COLS),
        d_model=256,
        n_heads=8,
        n_layers=6,
        dropout=0.1,
        max_seq_len=256,       # Max pitches per at-bat sequence
    ):
        super().__init__()
        self.d_model = d_model
        self.n_pitch_types = n_pitch_types
        self.n_zones = n_zones
        self.n_pitch_results = n_pitch_results
        self.n_at_bat_events = n_at_bat_events
        self.n_continuous = n_continuous

        # ── Discrete embeddings for PREVIOUS pitch info ──
        self.pitch_type_embed = nn.Embedding(n_pitch_types + 1, d_model // 4)  # +1 for <start>
        self.zone_embed = nn.Embedding(n_zones + 1, d_model // 4)              # +1 for <start>
        self.pitch_result_embed = nn.Embedding(n_pitch_results + 1, d_model // 4)  # +1 for <start>

        # ── Continuous projection ──
        # game_state: balls(1) + strikes(1) + outs(1) + inning(1) + score_diff(1) +
        #             on_1b(1) + on_2b(1) + on_3b(1) + inning_topbot(1) = 9
        # prev_continuous: 6 pitch attributes
        # context_dim: static game/pitcher/batter features
        game_state_dim = 9
        continuous_input_dim = context_dim + game_state_dim + n_continuous
        remaining_dim = d_model - 3 * (d_model // 4)
        self.continuous_proj = nn.Linear(continuous_input_dim, remaining_dim)

        # ── Positional encoding ──
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len=max_seq_len)

        # ── Causal Transformer ──
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # ── Output heads (Transfusion: discrete heads get LM loss, continuous gets diffusion/NLL) ──
        # Discrete heads (cross-entropy)
        self.pitch_type_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_pitch_types),
        )
        self.zone_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_zones),
        )
        self.pitch_result_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_pitch_results),
        )
        self.at_bat_event_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_at_bat_events),
        )

        # Continuous heads (Gaussian NLL — Transfusion's diffusion component simplified)
        self.continuous_mean_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_continuous),
        )
        self.continuous_logvar_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_continuous),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def _make_causal_mask(self, seq_len, device):
        return torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()

    def forward(self, context, game_state, prev_pitch_type, prev_zone, prev_pitch_result, prev_continuous):
        """
        Args:
            context:           (batch, seq_len, context_dim) — static game features (repeated)
            game_state:        (batch, seq_len, 9)           — balls, strikes, outs, inning, etc.
            prev_pitch_type:   (batch, seq_len) long         — previous pitch type index
            prev_zone:         (batch, seq_len) long         — previous zone index
            prev_pitch_result: (batch, seq_len) long         — previous pitch result index
            prev_continuous:   (batch, seq_len, n_continuous) — previous continuous pitch attrs

        Returns:
            dict with logits and distributional params for all output heads
        """
        B, S = prev_pitch_type.shape

        # Embed discrete previous-pitch tokens
        pt_emb = self.pitch_type_embed(prev_pitch_type)       # (B, S, d_model//4)
        z_emb = self.zone_embed(prev_zone)                     # (B, S, d_model//4)
        pr_emb = self.pitch_result_embed(prev_pitch_result)    # (B, S, d_model//4)

        # Project continuous inputs
        cont_input = torch.cat([context, game_state, prev_continuous], dim=-1)
        cont_emb = self.continuous_proj(cont_input)            # (B, S, remaining_dim)

        # Combine all embeddings
        x = torch.cat([pt_emb, z_emb, pr_emb, cont_emb], dim=-1)  # (B, S, d_model)

        # Add positional encoding
        x = x + self.pos_enc(S).unsqueeze(0)

        # Causal attention mask
        mask = self._make_causal_mask(S, x.device)
        x = self.transformer(x, mask=mask)

        # Output heads
        return {
            'pitch_type_logits':   self.pitch_type_head(x),
            'zone_logits':         self.zone_head(x),
            'pitch_result_logits': self.pitch_result_head(x),
            'at_bat_event_logits': self.at_bat_event_head(x),
            'continuous_mean':     self.continuous_mean_head(x),
            'continuous_logvar':   torch.clamp(self.continuous_logvar_head(x), min=-10, max=2),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Dataset: build at-bat pitch sequences from CSVs
# ─────────────────────────────────────────────────────────────────────────────

class AtBatSequenceDataset(Dataset):
    """
    Each sample is one at-bat: a sequence of pitches with their context and targets.
    """

    def __init__(self, pitch_df, pitch_context_df, pitch_result_df, at_bat_target_df,
                 game_context_df, pt_to_idx, pr_to_idx, ev_to_idx, zone_to_idx,
                 context_columns, context_mean, context_std, pitch_mean, pitch_std,
                 max_pitches=20):
        """
        All dataframes should be aligned row-by-row (same ordering).
        """
        self.max_pitches = max_pitches
        self.pt_to_idx = pt_to_idx
        self.pr_to_idx = pr_to_idx
        self.ev_to_idx = ev_to_idx
        self.zone_to_idx = zone_to_idx
        self.context_columns = context_columns
        self.context_mean = context_mean
        self.context_std = context_std
        self.pitch_mean = pitch_mean
        self.pitch_std = pitch_std
        self.n_continuous = len(CONTINUOUS_PITCH_COLS)

        # Merge all data into one frame for grouping
        merged = pitch_df[['game_id', 'at_bat_id', 'pitch_id', 'pitch_type', 'zone'] + CONTINUOUS_PITCH_COLS].copy()
        merged['pitch_result'] = pitch_result_df['description'].values
        merged['at_bat_event'] = at_bat_target_df['events'].values

        # Attach game-state columns from pitch_context
        for col in ['balls', 'strikes', 'inning', 'outs_when_up', 'home_score',
                     'away_score', 'bat_score_diff', 'on_1b', 'on_2b', 'on_3b',
                     'inning_topbot_Top']:
            if col in pitch_context_df.columns:
                merged[col] = pitch_context_df[col].values

        # Attach game_id for context lookup
        self.game_context_df = game_context_df

        # Sort properly: by game then at_bat then pitch within at_bat
        merged = merged.sort_values(['game_id', 'at_bat_id', 'pitch_id']).reset_index(drop=True)

        # Group into at-bat sequences
        self.at_bats = []
        for (gid, abid), group in merged.groupby(['game_id', 'at_bat_id'], sort=False):
            group = group.sort_values('pitch_id')
            self.at_bats.append((gid, group))

    def __len__(self):
        return len(self.at_bats)

    def _normalize_context(self, ctx_row):
        ctx = ctx_row[self.context_columns].values.astype(np.float32)
        ctx = (ctx - self.context_mean) / self.context_std
        return ctx

    def __getitem__(self, idx):
        game_id, group = self.at_bats[idx]
        S = min(len(group), self.max_pitches)
        group = group.iloc[:S]

        n_ctx = len(self.context_columns)
        n_cont = self.n_continuous

        # ── Build context (static per at-bat) ──
        ctx_row = self.game_context_df[self.game_context_df['game_id'] == game_id]
        if len(ctx_row) > 0:
            ctx = self._normalize_context(ctx_row.iloc[0])
        else:
            ctx = np.zeros(n_ctx, dtype=np.float32)

        context = np.tile(ctx, (S, 1))  # (S, n_ctx)

        # ── Game state per pitch ──
        game_state = np.zeros((S, 9), dtype=np.float32)
        for i, (_, row) in enumerate(group.iterrows()):
            game_state[i, 0] = row.get('balls', 0)
            game_state[i, 1] = row.get('strikes', 0)
            game_state[i, 2] = row.get('outs_when_up', 0)
            game_state[i, 3] = row.get('inning', 1) / 9.0  # normalize
            game_state[i, 4] = row.get('bat_score_diff', 0) / 10.0  # normalize
            game_state[i, 5] = 1.0 if row.get('on_1b', 0) else 0.0
            game_state[i, 6] = 1.0 if row.get('on_2b', 0) else 0.0
            game_state[i, 7] = 1.0 if row.get('on_3b', 0) else 0.0
            game_state[i, 8] = row.get('inning_topbot_Top', 0)

        # ── Shifted inputs (teacher forcing: input at step i is output from step i-1) ──
        # For step 0, use <start> tokens
        start_pt = len(PITCH_TYPES)     # <start> index for pitch_type
        start_zone = len(ZONES)         # <start> index for zone
        start_pr = len(PITCH_RESULTS)   # <start> index for pitch_result

        prev_pitch_type = np.full(S, start_pt, dtype=np.int64)
        prev_zone = np.full(S, start_zone, dtype=np.int64)
        prev_pitch_result = np.full(S, start_pr, dtype=np.int64)
        prev_continuous = np.zeros((S, n_cont), dtype=np.float32)

        # Targets
        tgt_pitch_type = np.zeros(S, dtype=np.int64)
        tgt_zone = np.zeros(S, dtype=np.int64)
        tgt_pitch_result = np.zeros(S, dtype=np.int64)
        tgt_at_bat_event = np.full(S, -1, dtype=np.int64)  # -1 = ignore
        tgt_continuous = np.zeros((S, n_cont), dtype=np.float32)

        for i, (_, row) in enumerate(group.iterrows()):
            # Current pitch targets
            pt_str = row['pitch_type']
            tgt_pitch_type[i] = self.pt_to_idx.get(pt_str, 0)

            zone_val = row['zone']
            if pd.notna(zone_val):
                tgt_zone[i] = self.zone_to_idx.get(int(zone_val), 0)

            pr_str = row['pitch_result']
            tgt_pitch_result[i] = self.pr_to_idx.get(pr_str, 0)

            ev_str = row['at_bat_event']
            if pd.notna(ev_str) and ev_str in self.ev_to_idx:
                tgt_at_bat_event[i] = self.ev_to_idx[ev_str]

            # Continuous targets (normalized)
            cont_vals = row[CONTINUOUS_PITCH_COLS].values.astype(np.float32)
            cont_vals = np.nan_to_num(cont_vals, nan=0.0)
            tgt_continuous[i] = (cont_vals - self.pitch_mean) / self.pitch_std

            # Shifted: step i+1 gets step i's outputs as input
            if i + 1 < S:
                prev_pitch_type[i + 1] = tgt_pitch_type[i]
                prev_zone[i + 1] = tgt_zone[i]
                prev_pitch_result[i + 1] = tgt_pitch_result[i]
                prev_continuous[i + 1] = tgt_continuous[i]

        # Build mask: which positions are valid
        mask = np.ones(S, dtype=np.float32)

        return {
            'context': torch.tensor(context, dtype=torch.float32),
            'game_state': torch.tensor(game_state, dtype=torch.float32),
            'prev_pitch_type': torch.tensor(prev_pitch_type, dtype=torch.long),
            'prev_zone': torch.tensor(prev_zone, dtype=torch.long),
            'prev_pitch_result': torch.tensor(prev_pitch_result, dtype=torch.long),
            'prev_continuous': torch.tensor(prev_continuous, dtype=torch.float32),
            'tgt_pitch_type': torch.tensor(tgt_pitch_type, dtype=torch.long),
            'tgt_zone': torch.tensor(tgt_zone, dtype=torch.long),
            'tgt_pitch_result': torch.tensor(tgt_pitch_result, dtype=torch.long),
            'tgt_at_bat_event': torch.tensor(tgt_at_bat_event, dtype=torch.long),
            'tgt_continuous': torch.tensor(tgt_continuous, dtype=torch.float32),
            'mask': torch.tensor(mask, dtype=torch.float32),
            'seq_len': S,
        }


def collate_at_bats(batch):
    """Pad variable-length at-bat sequences to the longest in the batch."""
    max_len = max(item['seq_len'] for item in batch)
    B = len(batch)

    ctx_dim = batch[0]['context'].shape[-1]
    n_cont = batch[0]['prev_continuous'].shape[-1]

    out = {
        'context':           torch.zeros(B, max_len, ctx_dim),
        'game_state':        torch.zeros(B, max_len, 9),
        'prev_pitch_type':   torch.full((B, max_len), len(PITCH_TYPES), dtype=torch.long),  # pad as <start>
        'prev_zone':         torch.full((B, max_len), len(ZONES), dtype=torch.long),
        'prev_pitch_result': torch.full((B, max_len), len(PITCH_RESULTS), dtype=torch.long),
        'prev_continuous':   torch.zeros(B, max_len, n_cont),
        'tgt_pitch_type':    torch.zeros(B, max_len, dtype=torch.long),
        'tgt_zone':          torch.zeros(B, max_len, dtype=torch.long),
        'tgt_pitch_result':  torch.zeros(B, max_len, dtype=torch.long),
        'tgt_at_bat_event':  torch.full((B, max_len), -1, dtype=torch.long),
        'tgt_continuous':    torch.zeros(B, max_len, n_cont),
        'mask':              torch.zeros(B, max_len),
    }

    for i, item in enumerate(batch):
        s = item['seq_len']
        out['context'][i, :s]           = item['context']
        out['game_state'][i, :s]        = item['game_state']
        out['prev_pitch_type'][i, :s]   = item['prev_pitch_type']
        out['prev_zone'][i, :s]         = item['prev_zone']
        out['prev_pitch_result'][i, :s] = item['prev_pitch_result']
        out['prev_continuous'][i, :s]   = item['prev_continuous']
        out['tgt_pitch_type'][i, :s]    = item['tgt_pitch_type']
        out['tgt_zone'][i, :s]          = item['tgt_zone']
        out['tgt_pitch_result'][i, :s]  = item['tgt_pitch_result']
        out['tgt_at_bat_event'][i, :s]  = item['tgt_at_bat_event']
        out['tgt_continuous'][i, :s]    = item['tgt_continuous']
        out['mask'][i, :s]              = item['mask']

    return out


# ─────────────────────────────────────────────────────────────────────────────
# Loss: Transfusion = LM (discrete) + λ · Diffusion/NLL (continuous)
# ─────────────────────────────────────────────────────────────────────────────

def transfusion_loss(outputs, batch, lambda_continuous=5.0):
    """
    Combined loss from Equation 4 of the Transfusion paper:
        L = L_LM + λ · L_DDPM

    We use cross-entropy for all discrete heads (L_LM) and Gaussian NLL for
    continuous heads (L_DDPM simplified to parametric NLL).
    """
    mask = batch['mask']  # (B, S)
    B, S = mask.shape

    # ── Discrete losses (cross-entropy, masked) ──
    ce = nn.CrossEntropyLoss(reduction='none')

    # Pitch type
    pt_loss = ce(outputs['pitch_type_logits'].view(-1, len(PITCH_TYPES)),
                 batch['tgt_pitch_type'].view(-1)).view(B, S)
    pt_loss = (pt_loss * mask).sum() / mask.sum()

    # Zone
    z_loss = ce(outputs['zone_logits'].view(-1, len(ZONES)),
                batch['tgt_zone'].view(-1)).view(B, S)
    z_loss = (z_loss * mask).sum() / mask.sum()

    # Pitch result
    pr_loss = ce(outputs['pitch_result_logits'].view(-1, len(PITCH_RESULTS)),
                 batch['tgt_pitch_result'].view(-1)).view(B, S)
    pr_loss = (pr_loss * mask).sum() / mask.sum()

    # At-bat event (only on terminal pitches where tgt_at_bat_event != -1)
    ev_mask = (batch['tgt_at_bat_event'] != -1).float() * mask
    if ev_mask.sum() > 0:
        ev_tgt = batch['tgt_at_bat_event'].clamp(min=0)  # clamp -1 to 0 for safe indexing
        ev_loss = ce(outputs['at_bat_event_logits'].view(-1, len(AT_BAT_EVENTS)),
                     ev_tgt.view(-1)).view(B, S)
        ev_loss = (ev_loss * ev_mask).sum() / ev_mask.sum()
    else:
        ev_loss = torch.tensor(0.0, device=mask.device)

    lm_loss = pt_loss + z_loss + pr_loss + ev_loss

    # ── Continuous loss (Gaussian NLL) ──
    mean = outputs['continuous_mean']      # (B, S, n_cont)
    logvar = outputs['continuous_logvar']   # (B, S, n_cont)
    target = batch['tgt_continuous']        # (B, S, n_cont)
    var = torch.exp(logvar)
    nll = 0.5 * (logvar + (target - mean) ** 2 / var)  # (B, S, n_cont)
    nll = nll.mean(dim=-1)  # average over features → (B, S)
    nll_loss = (nll * mask).sum() / mask.sum()

    total = lm_loss + lambda_continuous * nll_loss

    return {
        'total': total,
        'lm_loss': lm_loss,
        'pt_loss': pt_loss,
        'zone_loss': z_loss,
        'pr_loss': pr_loss,
        'ev_loss': ev_loss,
        'nll_loss': nll_loss,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def train_model(model, train_dataset, val_dataset, epochs=30, batch_size=128, lr=3e-4,
                lambda_continuous=5.0):
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available()
                          else "cpu")
    model = model.to(device)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_at_bats, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            collate_fn=collate_at_bats, num_workers=0)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01, betas=(0.9, 0.95))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_val_loss = float('inf')
    best_state = None

    for epoch in range(epochs):
        model.train()
        train_losses = {k: 0.0 for k in ['total', 'lm_loss', 'nll_loss', 'pt_loss', 'pr_loss', 'ev_loss']}
        n_batches = 0

        for batch in train_loader:
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}

            optimizer.zero_grad()
            outputs = model(
                batch['context'], batch['game_state'],
                batch['prev_pitch_type'], batch['prev_zone'],
                batch['prev_pitch_result'], batch['prev_continuous'],
            )
            losses = transfusion_loss(outputs, batch, lambda_continuous=lambda_continuous)
            losses['total'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            for k in train_losses:
                train_losses[k] += losses[k].item()
            n_batches += 1

        scheduler.step()

        # Validation
        model.eval()
        val_losses = {k: 0.0 for k in ['total', 'lm_loss', 'nll_loss']}
        val_correct_pt = 0
        val_correct_pr = 0
        val_total = 0
        n_val = 0

        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
                outputs = model(
                    batch['context'], batch['game_state'],
                    batch['prev_pitch_type'], batch['prev_zone'],
                    batch['prev_pitch_result'], batch['prev_continuous'],
                )
                losses = transfusion_loss(outputs, batch, lambda_continuous=lambda_continuous)
                for k in val_losses:
                    val_losses[k] += losses[k].item()
                n_val += 1

                mask = batch['mask'].bool()
                val_correct_pt += (outputs['pitch_type_logits'].argmax(-1)[mask] == batch['tgt_pitch_type'][mask]).sum().item()
                val_correct_pr += (outputs['pitch_result_logits'].argmax(-1)[mask] == batch['tgt_pitch_result'][mask]).sum().item()
                val_total += mask.sum().item()

        marker = ''
        vl = val_losses['total'] / max(n_val, 1)
        if vl < best_val_loss:
            best_val_loss = vl
            best_state = copy.deepcopy(model.state_dict())
            marker = ' *'

        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train: {train_losses['total']/n_batches:.4f} "
              f"(LM: {train_losses['lm_loss']/n_batches:.4f}, NLL: {train_losses['nll_loss']/n_batches:.4f}) | "
              f"Val: {vl:.4f} | "
              f"PT Acc: {val_correct_pt/max(val_total,1):.2%}, PR Acc: {val_correct_pr/max(val_total,1):.2%} | "
              f"LR: {scheduler.get_last_lr()[0]:.6f}{marker}")

    print(f'\nRestoring best model (val loss: {best_val_loss:.4f})')
    model.load_state_dict(best_state)
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Game Simulation Engine
# ─────────────────────────────────────────────────────────────────────────────

class GameSimulator:
    """
    Full MLB game simulation using the Transfusion pitch sequence model.

    Simulates pitch-by-pitch with proper MLB rules:
      - Count tracking (balls, strikes, fouls)
      - 3-out half innings, top/bottom alternation
      - 9 innings, extra innings if tied (runner on 2B in extras since 2020)
      - Walk on 4 balls, strikeout on 3 strikes
      - At-bat event prediction when pitch_result is terminal
    """

    def __init__(self, model, context_features, pt_to_idx, pr_to_idx, ev_to_idx, zone_to_idx,
                 pitch_mean, pitch_std, device='cpu'):
        self.model = model
        self.context_features = context_features  # (context_dim,) normalized static features
        self.pt_to_idx = pt_to_idx
        self.pr_to_idx = pr_to_idx
        self.ev_to_idx = ev_to_idx
        self.zone_to_idx = zone_to_idx
        self.idx_to_pt = {v: k for k, v in pt_to_idx.items()}
        self.idx_to_pr = {v: k for k, v in pr_to_idx.items()}
        self.idx_to_ev = {v: k for k, v in ev_to_idx.items()}
        self.idx_to_zone = {v: k for k, v in zone_to_idx.items()}
        self.pitch_mean = pitch_mean
        self.pitch_std = pitch_std
        self.device = device
        self.model.eval()

    def simulate_game(self, temperature=1.0, verbose=True):
        """Simulate a full game, returning detailed log."""
        home_score = 0
        away_score = 0
        game_log = []
        inning = 1
        max_innings = 9

        while True:
            # Top of inning (away team bats)
            if verbose:
                print(f"\n{'='*50}")
                print(f"  Top of Inning {inning} | Away: {away_score} - Home: {home_score}")
                print(f"{'='*50}")

            runs, half_log = self._simulate_half_inning(
                inning=inning, is_top=True,
                home_score=home_score, away_score=away_score,
                extra_runner=(inning > max_innings),
                temperature=temperature, verbose=verbose,
            )
            away_score += runs
            game_log.append({'inning': inning, 'half': 'top', 'runs': runs, 'details': half_log})

            # Bottom of inning (home team bats)
            # In bottom of 9th+, if home is ahead, game is over
            if inning >= max_innings and home_score > away_score:
                break

            if verbose:
                print(f"\n{'='*50}")
                print(f"  Bottom of Inning {inning} | Away: {away_score} - Home: {home_score}")
                print(f"{'='*50}")

            runs, half_log = self._simulate_half_inning(
                inning=inning, is_top=False,
                home_score=home_score, away_score=away_score,
                extra_runner=(inning > max_innings),
                temperature=temperature, verbose=verbose,
            )
            home_score += runs
            game_log.append({'inning': inning, 'half': 'bottom', 'runs': runs, 'details': half_log})

            # Walk-off check: home team takes the lead in bottom of 9th+
            if inning >= max_innings and home_score > away_score:
                break

            # End of regulation or continue
            if inning >= max_innings:
                if home_score == away_score:
                    inning += 1
                    continue
                else:
                    break
            else:
                inning += 1

        if verbose:
            print(f"\n{'='*50}")
            print(f"  FINAL SCORE: Away {away_score} - Home {home_score}")
            print(f"{'='*50}")

        return {
            'home_score': home_score,
            'away_score': away_score,
            'innings': inning,
            'log': game_log,
        }

    def _simulate_half_inning(self, inning, is_top, home_score, away_score,
                              extra_runner=False, temperature=1.0, verbose=True):
        """Simulate one half-inning until 3 outs."""
        outs = 0
        runs = 0
        bases = [False, False, False]  # 1B, 2B, 3B
        half_log = []
        at_bat_num = 0

        # MLB extra innings: runner starts on 2B
        if extra_runner:
            bases[1] = True
            if verbose:
                print("  [Extra innings: ghost runner placed on 2B]")

        while outs < 3:
            at_bat_num += 1
            ab_result = self._simulate_at_bat(
                inning=inning, is_top=is_top, outs=outs,
                home_score=home_score, away_score=away_score,
                bases=bases, temperature=temperature, verbose=verbose,
            )
            half_log.append(ab_result)

            # Process the at-bat event
            event = ab_result['event']
            new_outs, new_runs, bases = self._apply_event(event, outs, bases)
            outs = new_outs
            runs += new_runs

            if verbose:
                base_str = ''
                if bases[0]: base_str += '1B '
                if bases[1]: base_str += '2B '
                if bases[2]: base_str += '3B '
                print(f"  → Event: {event} | Outs: {outs} | Runs this inning: {runs} | Bases: {base_str or 'empty'}")

            # Walk-off in bottom of inning
            if not is_top and inning >= 9:
                if (home_score + runs) > away_score:
                    if verbose:
                        print("  *** WALK-OFF! ***")
                    break

        return runs, half_log

    def _simulate_at_bat(self, inning, is_top, outs, home_score, away_score,
                         bases, temperature=1.0, verbose=True):
        """Simulate a single at-bat pitch by pitch."""
        balls = 0
        strikes = 0
        pitch_log = []

        # Build initial game state
        score_diff = (away_score - home_score) if is_top else (home_score - away_score)

        # History for autoregressive feeding
        prev_pt_idx = len(PITCH_TYPES)      # <start>
        prev_zone_idx = len(ZONES)           # <start>
        prev_pr_idx = len(PITCH_RESULTS)     # <start>
        prev_cont = np.zeros(len(CONTINUOUS_PITCH_COLS), dtype=np.float32)

        # Accumulate sequence history for transformer context
        history_pt = []
        history_zone = []
        history_pr = []
        history_cont = []
        history_gs = []

        with torch.no_grad():
            max_pitches = 30  # safety limit
            for pitch_num in range(max_pitches):
                # Build game state vector
                gs = np.array([
                    balls, strikes, outs, inning / 9.0, score_diff / 10.0,
                    float(bases[0]), float(bases[1]), float(bases[2]),
                    float(is_top),
                ], dtype=np.float32)

                # Add to history
                history_pt.append(prev_pt_idx)
                history_zone.append(prev_zone_idx)
                history_pr.append(prev_pr_idx)
                history_cont.append(prev_cont.copy())
                history_gs.append(gs)

                S = len(history_pt)

                # Build tensors (batch=1, seq_len=S)
                ctx = torch.tensor(self.context_features, dtype=torch.float32).unsqueeze(0).unsqueeze(0).expand(1, S, -1).to(self.device)
                gs_t = torch.tensor(np.array(history_gs), dtype=torch.float32).unsqueeze(0).to(self.device)
                pt_t = torch.tensor(history_pt, dtype=torch.long).unsqueeze(0).to(self.device)
                z_t = torch.tensor(history_zone, dtype=torch.long).unsqueeze(0).to(self.device)
                pr_t = torch.tensor(history_pr, dtype=torch.long).unsqueeze(0).to(self.device)
                cont_t = torch.tensor(np.array(history_cont), dtype=torch.float32).unsqueeze(0).to(self.device)

                outputs = self.model(ctx, gs_t, pt_t, z_t, pr_t, cont_t)

                # Take the last timestep's predictions
                pt_logits = outputs['pitch_type_logits'][0, -1] / temperature
                z_logits = outputs['zone_logits'][0, -1] / temperature
                pr_logits = outputs['pitch_result_logits'][0, -1] / temperature
                cont_mean = outputs['continuous_mean'][0, -1]
                cont_logvar = outputs['continuous_logvar'][0, -1]

                # Sample discrete outputs
                sampled_pt = torch.multinomial(F.softmax(pt_logits, dim=-1), 1).item()
                sampled_zone = torch.multinomial(F.softmax(z_logits, dim=-1), 1).item()
                sampled_pr = torch.multinomial(F.softmax(pr_logits, dim=-1), 1).item()

                # Sample continuous outputs
                std = torch.exp(0.5 * cont_logvar)
                sampled_cont = (cont_mean + std * torch.randn_like(std)).cpu().numpy()

                # Denormalize continuous values
                raw_cont = sampled_cont * self.pitch_std + self.pitch_mean

                pitch_type_str = self.idx_to_pt.get(sampled_pt, 'FF')
                zone_str = self.idx_to_zone.get(sampled_zone, 5)
                result_str = self.idx_to_pr.get(sampled_pr, 'ball')

                pitch_info = {
                    'pitch_num': pitch_num + 1,
                    'pitch_type': pitch_type_str,
                    'zone': zone_str,
                    'result': result_str,
                    'continuous': {col: float(raw_cont[i]) for i, col in enumerate(CONTINUOUS_PITCH_COLS)},
                    'count_before': f"{balls}-{strikes}",
                }

                # Update count based on result
                if result_str in STRIKE_RESULTS:
                    strikes += 1
                elif result_str in FOUL_RESULTS:
                    if strikes < 2:
                        strikes += 1
                elif result_str in BALL_RESULTS:
                    balls += 1

                pitch_info['count_after'] = f"{balls}-{strikes}"

                if verbose:
                    speed = raw_cont[0] if len(raw_cont) > 0 else 0
                    print(f"    P{pitch_num+1}: {pitch_type_str} {speed:.1f}mph → {result_str} ({balls}-{strikes})")

                pitch_log.append(pitch_info)

                # ── Check for at-bat termination ──
                event = None

                # Strikeout
                if strikes >= 3:
                    event = 'strikeout'
                    break

                # Walk
                if balls >= 4:
                    event = 'walk'
                    break

                # Hit by pitch
                if result_str == 'hit_by_pitch':
                    event = 'hit_by_pitch'
                    break

                # Ball in play → predict the at-bat event
                if result_str == 'hit_into_play':
                    ev_logits = outputs['at_bat_event_logits'][0, -1] / temperature
                    sampled_ev = torch.multinomial(F.softmax(ev_logits, dim=-1), 1).item()
                    event = self.idx_to_ev.get(sampled_ev, 'field_out')
                    break

                # Set up inputs for next pitch
                prev_pt_idx = sampled_pt
                prev_zone_idx = sampled_zone
                prev_pr_idx = sampled_pr
                prev_cont = sampled_cont

            # Safety: no event after max pitches
            if event is None:
                event = 'field_out'

        return {'pitches': pitch_log, 'event': event, 'final_count': f"{balls}-{strikes}"}

    def _apply_event(self, event, outs, bases):
        """Apply an at-bat event to update outs, runs, and baserunners."""
        runs = 0
        bases = list(bases)  # copy

        if event in ('strikeout', 'strikeout_double_play'):
            outs += 1
            if event == 'strikeout_double_play':
                outs += 1

        elif event == 'field_out':
            outs += 1
            # Advance runners one base on field out (simplified)
            if bases[2]:
                runs += 1
                bases[2] = False
            if bases[1]:
                bases[2] = True
                bases[1] = False

        elif event == 'force_out':
            outs += 1

        elif event == 'grounded_into_double_play':
            outs += 2
            # Remove lead runner
            if bases[0]:
                bases[0] = False

        elif event == 'double_play':
            outs += 2

        elif event == 'triple_play':
            outs += 3

        elif event == 'single':
            # All runners advance one base
            if bases[2]:
                runs += 1
                bases[2] = False
            if bases[1]:
                bases[2] = True
                bases[1] = False
            if bases[0]:
                bases[1] = True
                bases[0] = False
            bases[0] = True  # batter to first

        elif event == 'double':
            if bases[2]:
                runs += 1
                bases[2] = False
            if bases[1]:
                runs += 1
                bases[1] = False
            if bases[0]:
                bases[2] = True
                bases[0] = False
            bases[1] = True  # batter to second

        elif event == 'triple':
            runs += sum(bases)
            bases = [False, False, False]
            bases[2] = True  # batter to third

        elif event == 'home_run':
            runs += sum(bases) + 1  # all runners + batter score
            bases = [False, False, False]

        elif event in ('walk', 'hit_by_pitch', 'intent_walk', 'catcher_interf'):
            # Force advance
            if bases[0] and bases[1] and bases[2]:
                runs += 1
                bases[2] = True  # stays full
            elif bases[0] and bases[1]:
                bases[2] = True
            elif bases[0]:
                bases[1] = True
            bases[0] = True

        elif event == 'sac_fly':
            outs += 1
            if bases[2]:
                runs += 1
                bases[2] = False

        elif event == 'sac_fly_double_play':
            outs += 2
            if bases[2]:
                runs += 1
                bases[2] = False

        elif event == 'sac_bunt':
            outs += 1
            # Advance all runners
            if bases[2]:
                runs += 1
                bases[2] = False
            if bases[1]:
                bases[2] = True
                bases[1] = False
            if bases[0]:
                bases[1] = True
                bases[0] = False

        elif event == 'sac_bunt_double_play':
            outs += 2

        elif event == 'field_error':
            # Batter reaches, runners advance
            if bases[2]:
                runs += 1
                bases[2] = False
            if bases[1]:
                bases[2] = True
                bases[1] = False
            if bases[0]:
                bases[1] = True
                bases[0] = False
            bases[0] = True

        elif event in ('fielders_choice', 'fielders_choice_out'):
            outs += 1
            bases[0] = True

        elif event == 'truncated_pa':
            pass  # no change (rare edge case)

        outs = min(outs, 3)  # cap at 3
        return outs, runs, bases


# ─────────────────────────────────────────────────────────────────────────────
# Data loading helper
# ─────────────────────────────────────────────────────────────────────────────

def load_data(split='train'):
    """Load all CSVs for a given split and return aligned dataframes."""
    if split == 'train':
        suffix = '2020-05-12_2025-08-01'
    else:
        suffix = '2025-08-01_2025-11-03'

    prefix = f'{split}_'

    pitch_df = pd.read_csv(f'{prefix}pitch_{suffix}.csv')
    pitch_context_df = pd.read_csv(f'{prefix}pitch_context_{suffix}.csv')
    pitch_result_df = pd.read_csv(f'{prefix}pitch_result_{suffix}.csv')
    at_bat_target_df = pd.read_csv(f'{prefix}at_bat_target_{suffix}.csv')
    game_context_df = pd.read_csv(f'{prefix}game_context_{suffix}.csv')

    return pitch_df, pitch_context_df, pitch_result_df, at_bat_target_df, game_context_df


def prepare_game_context(game_context_df, is_train=True):
    """Process game context into numeric features, returning (df, columns, mean, std)."""
    gc = game_context_df.copy()
    gc = gc.drop(columns=['game_date'], errors='ignore')

    # Keep game_id for lookup but don't include in features
    game_ids = gc['game_id'] if 'game_id' in gc.columns else None
    gc_features = gc.drop(columns=['game_id'], errors='ignore')

    gc_features = pd.get_dummies(gc_features, drop_first=True).astype(float)
    columns = gc_features.columns

    values = gc_features.values
    if is_train:
        mean = values.mean(axis=0)
        std = values.std(axis=0)
        std[std < 1e-8] = 1.0
    else:
        mean = std = None  # will be provided externally

    gc_out = gc_features.copy()
    if game_ids is not None:
        gc_out['game_id'] = game_ids.values

    return gc_out, columns, mean, std


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading data...")
    train_pitch, train_pc, train_pr, train_ab, train_gc = load_data('train')
    test_pitch, test_pc, test_pr, test_ab, test_gc = load_data('test')

    pt_to_idx, pr_to_idx, ev_to_idx, zone_to_idx = build_vocab_maps()

    # Process game context
    print("Processing game context...")
    train_gc_proc, ctx_columns, ctx_mean, ctx_std = prepare_game_context(train_gc, is_train=True)

    # Align test game context to train columns
    test_gc_proc = test_gc.copy()
    test_gc_proc = test_gc_proc.drop(columns=['game_date'], errors='ignore')
    game_ids_test = test_gc_proc['game_id'] if 'game_id' in test_gc_proc.columns else None
    test_gc_features = test_gc_proc.drop(columns=['game_id'], errors='ignore')
    test_gc_features = pd.get_dummies(test_gc_features, drop_first=True).astype(float)
    test_gc_features = test_gc_features.reindex(columns=ctx_columns, fill_value=0)
    test_gc_proc = test_gc_features.copy()
    if game_ids_test is not None:
        test_gc_proc['game_id'] = game_ids_test.values

    # Compute continuous pitch normalization from training data
    print("Computing pitch normalization stats...")
    cont_values = train_pitch[CONTINUOUS_PITCH_COLS].values.astype(np.float32)
    cont_values = np.nan_to_num(cont_values, nan=0.0)
    pitch_mean_arr = cont_values.mean(axis=0)
    pitch_std_arr = cont_values.std(axis=0)
    pitch_std_arr[pitch_std_arr < 1e-8] = 1.0

    # Build datasets
    print("Building training dataset...")
    train_dataset = AtBatSequenceDataset(
        train_pitch, train_pc, train_pr, train_ab, train_gc_proc,
        pt_to_idx, pr_to_idx, ev_to_idx, zone_to_idx,
        ctx_columns, ctx_mean, ctx_std, pitch_mean_arr, pitch_std_arr,
    )
    print(f"  {len(train_dataset)} at-bats in training set")

    print("Building test dataset...")
    test_dataset = AtBatSequenceDataset(
        test_pitch, test_pc, test_pr, test_ab, test_gc_proc,
        pt_to_idx, pr_to_idx, ev_to_idx, zone_to_idx,
        ctx_columns, ctx_mean, ctx_std, pitch_mean_arr, pitch_std_arr,
    )
    print(f"  {len(test_dataset)} at-bats in test set")

    # Print base rates
    print("\n--- Base rates ---")
    pt_counts = Counter(train_pitch['pitch_type'].dropna())
    print(f"Most common pitch type: {pt_counts.most_common(1)[0]}")
    pr_counts = Counter(train_pr['description'].dropna())
    print(f"Most common pitch result: {pr_counts.most_common(1)[0]}")

    # Build model
    context_dim = len(ctx_columns)
    print(f"\nContext dim: {context_dim}")
    model = PitchSequenceTransfusion(
        context_dim=context_dim,
        d_model=256,
        n_heads=8,
        n_layers=6,
        dropout=0.1,
        max_seq_len=256,
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # Train
    print("\n--- Training ---")
    model = train_model(
        model, train_dataset, test_dataset,
        epochs=30, batch_size=128, lr=3e-4, lambda_continuous=5.0,
    )

    # Save
    torch.save({
        'model_state_dict': model.state_dict(),
        'ctx_columns': list(ctx_columns),
        'ctx_mean': ctx_mean,
        'ctx_std': ctx_std,
        'pitch_mean': pitch_mean_arr,
        'pitch_std': pitch_std_arr,
    }, 'pitch_sequence_model.pth')
    print("Model saved to pitch_sequence_model.pth")

    # Demo: simulate one game
    print("\n--- Simulating a game ---")
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available()
                          else "cpu")
    model = model.to(device)

    # Use first test game's context as demo
    sample_ctx = train_gc_proc[train_gc_proc.columns.intersection(ctx_columns)].iloc[0].values.astype(np.float32)
    sample_ctx = (sample_ctx - ctx_mean) / ctx_std

    simulator = GameSimulator(
        model=model,
        context_features=sample_ctx,
        pt_to_idx=pt_to_idx,
        pr_to_idx=pr_to_idx,
        ev_to_idx=ev_to_idx,
        zone_to_idx=zone_to_idx,
        pitch_mean=pitch_mean_arr,
        pitch_std=pitch_std_arr,
        device=device,
    )

    result = simulator.simulate_game(temperature=0.9, verbose=True)
    print(f"\nFinal: Away {result['away_score']} - Home {result['home_score']} ({result['innings']} innings)")
