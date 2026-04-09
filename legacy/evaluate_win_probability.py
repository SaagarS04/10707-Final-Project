"""
Monte Carlo Win Probability Evaluation
=======================================
Simulates N game paths per test game using the trained Transfusion pitch sequence
model.  Computes empirical P(home_win) = (# home wins) / N and compares against
baselines.

Baselines:
  1. Historical average:  always predict training-set home win rate
  2. Log5 method:         p = (h - h·a) / (h + a - 2·h·a)
  3. MLP classifier       on game-context + prefix-state features

Metrics:  Brier score · Log-loss · Accuracy · Calibration

Usage:
    # Train model first (if no checkpoint):
    python pitch_sequence_predictor.py

    # Run evaluation (uses cached data if available):
    python evaluate_win_probability.py --n_sims 10000 --n_games 100

    # Force reprocessing data (ignore cache):
    python evaluate_win_probability.py --no_cache --n_sims 5000 --n_games 50

    # Custom cache directory:
    python evaluate_win_probability.py --cache_dir my_cache --n_sims 1000

    # Train + evaluate with multithreading:
    python evaluate_win_probability.py --train --epochs 15 --n_threads 4 --n_sims 1000
"""

import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from collections import defaultdict
import concurrent.futures
import threading
from concurrent.futures import ThreadPoolExecutor
import pickle
from pathlib import Path

from mcmc_simulator import MHGameSampler
from pitch_data import compute_re24_from_pitch_context
from pitch_sequence_predictor import (
    PitchSequenceTransfusion,
    GameSequenceDataset,
    GameSimulator,
    collate_games,
    train_model,
    load_data,
    prepare_game_context,
    build_vocab_maps,
    PITCH_TYPES, PITCH_RESULTS, AT_BAT_EVENTS, ZONES,
    CONTINUOUS_PITCH_COLS,
    STRIKE_RESULTS, FOUL_RESULTS, BALL_RESULTS,
)


# ─────────────────────────────────────────────────────────────────────────────
# Precomputed result-category lookup tables (avoid dict lookups in hot loop)
# ─────────────────────────────────────────────────────────────────────────────

_N_PR = len(PITCH_RESULTS)
_PR_LIST = sorted(PITCH_RESULTS)   # consistent with build_vocab_maps
_IS_STRIKE = np.array([_PR_LIST[i] in STRIKE_RESULTS for i in range(_N_PR)], dtype=bool)
_IS_FOUL   = np.array([_PR_LIST[i] in FOUL_RESULTS   for i in range(_N_PR)], dtype=bool)
_IS_BALL   = np.array([_PR_LIST[i] in BALL_RESULTS    for i in range(_N_PR)], dtype=bool)
_IS_HBP    = np.array([_PR_LIST[i] == 'hit_by_pitch'  for i in range(_N_PR)], dtype=bool)
_IS_HIP    = np.array([_PR_LIST[i] == 'hit_into_play' for i in range(_N_PR)], dtype=bool)


# ─────────────────────────────────────────────────────────────────────────────
# Event application (standalone, same logic as GameSimulator._apply_event)
# ─────────────────────────────────────────────────────────────────────────────

def _apply_event(event, outs, b0, b1, b2):
    """Apply an at-bat event.  Returns (outs, runs, b0, b1, b2)."""
    runs = 0

    if event in ('strikeout', 'strikeout_double_play'):
        outs += 1
        if event == 'strikeout_double_play':
            outs += 1

    elif event == 'field_out':
        outs += 1
        if b2:
            runs += 1; b2 = False
        if b1:
            b2 = True; b1 = False

    elif event == 'force_out':
        outs += 1

    elif event == 'grounded_into_double_play':
        outs += 2
        if b0:
            b0 = False

    elif event == 'double_play':
        outs += 2

    elif event == 'triple_play':
        outs += 3

    elif event == 'single':
        if b2: runs += 1; b2 = False
        if b1: b2 = True;  b1 = False
        if b0: b1 = True;  b0 = False
        b0 = True

    elif event == 'double':
        if b2: runs += 1; b2 = False
        if b1: runs += 1; b1 = False
        if b0: b2 = True;  b0 = False
        b1 = True

    elif event == 'triple':
        runs += int(b0) + int(b1) + int(b2)
        b0 = b1 = False; b2 = True

    elif event == 'home_run':
        runs += int(b0) + int(b1) + int(b2) + 1
        b0 = b1 = b2 = False

    elif event in ('walk', 'hit_by_pitch', 'intent_walk', 'catcher_interf'):
        if b0 and b1 and b2:
            runs += 1
        elif b0 and b1:
            b2 = True
        elif b0:
            b1 = True
        b0 = True

    elif event == 'sac_fly':
        outs += 1
        if b2: runs += 1; b2 = False

    elif event == 'sac_fly_double_play':
        outs += 2
        if b2: runs += 1; b2 = False

    elif event == 'sac_bunt':
        outs += 1
        if b2: runs += 1; b2 = False
        if b1: b2 = True;  b1 = False
        if b0: b1 = True;  b0 = False

    elif event == 'sac_bunt_double_play':
        outs += 2

    elif event == 'field_error':
        if b2: runs += 1; b2 = False
        if b1: b2 = True;  b1 = False
        if b0: b1 = True;  b0 = False
        b0 = True

    elif event in ('fielders_choice', 'fielders_choice_out'):
        outs += 1
        b0 = True

    outs = min(outs, 3)
    return outs, runs, b0, b1, b2


# ─────────────────────────────────────────────────────────────────────────────
# Extract real game state at the start of a given inning (or half-inning)
# ─────────────────────────────────────────────────────────────────────────────

def extract_game_state_at_inning(game_id, start_inning, test_pitch, test_pc, test_pr, test_ab,
                                  pt_to_idx, pr_to_idx, ev_to_idx, zone_to_idx,
                                  pitch_mean, pitch_std):
    """
    Find the game state just before `start_inning` begins.

    Args:
        start_inning: float — the inning to start simulating FROM.
            Whole numbers (e.g. 8) → top of that inning (use innings 1-7 fully).
            Half values  (e.g. 7.5) → bottom of that inning (use through top of 7).
            1.0 → simulate from scratch.

    Returns:
        dict with keys: inning, is_top, outs, home_score, away_score,
                        bases (list of 3 bools), balls, strikes,
                        n_pitches_replayed (int), total_pitches (int)
        or None if the game can't be found
    """
    # Determine the integer inning and whether we start at bottom half
    is_half = (start_inning % 1) >= 0.4   # 7.5 → True, 8.0 → False
    base_inning = int(start_inning)        # 7.5 → 7, 8.0 → 8

    if start_inning <= 1.0:
        mask = test_pitch['game_id'] == game_id
        total = mask.sum()
        return {
            'inning': 1, 'is_top': True, 'outs': 0,
            'home_score': 0, 'away_score': 0,
            'bases': [False, False, False],
            'balls': 0, 'strikes': 0,
            'n_pitches_replayed': 0, 'total_pitches': total,
        }

    # Get all pitches for this game
    mask = test_pitch['game_id'] == game_id
    if mask.sum() == 0:
        return None

    game_pitches = test_pitch[mask].copy()
    game_pc = test_pc[mask].copy()

    # Sort by chronological pitch order
    game_pitches = game_pitches.sort_values(['at_bat_id', 'pitch_id'])
    sort_idx = game_pitches.index
    game_pc = game_pc.loc[sort_idx].reset_index(drop=True)
    game_pitches = game_pitches.reset_index(drop=True)

    total_pitches = len(game_pitches)

    innings = game_pc['inning'].values
    topbot  = game_pc['inning_topbot_Top'].values   # 1 = top, 0 = bottom

    if is_half:
        # e.g. 7.5 → use all of top of inning 7 (and everything before).
        # Start simulating from bottom of inning 7.
        # Prefix = pitches where (inning < base_inning) OR (inning == base_inning AND is_top)
        prefix_mask = (innings < base_inning) | ((innings == base_inning) & (topbot == 1))
        result_inning = base_inning
        result_is_top = False
    else:
        # e.g. 8 → use all of inning 7 (top + bottom). Start from top of 8.
        # Prefix = pitches where inning < base_inning
        prefix_mask = innings < base_inning
        result_inning = base_inning
        result_is_top = True

    n_prefix = int(prefix_mask.sum())

    if n_prefix == 0:
        return {
            'inning': 1, 'is_top': True, 'outs': 0,
            'home_score': 0, 'away_score': 0,
            'bases': [False, False, False],
            'balls': 0, 'strikes': 0,
            'n_pitches_replayed': 0, 'total_pitches': total_pitches,
        }

    if n_prefix >= total_pitches:
        n_prefix = total_pitches

    # Get the score from the FIRST pitch AFTER the prefix (cleanest boundary).
    # If that pitch doesn't exist, use the last prefix pitch.
    non_prefix_mask = ~prefix_mask
    if non_prefix_mask.any():
        first_after = np.where(non_prefix_mask)[0][0]
        row = game_pc.iloc[first_after]
    else:
        row = game_pc.iloc[n_prefix - 1]

    home_score = int(row.get('home_score', 0))
    away_score = int(row.get('away_score', 0))

    return {
        'inning': result_inning, 'is_top': result_is_top, 'outs': 0,
        'home_score': home_score, 'away_score': away_score,
        'bases': [False, False, False],
        'balls': 0, 'strikes': 0,
        'n_pitches_replayed': n_prefix, 'total_pitches': total_pitches,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Batched Game Simulation
# ─────────────────────────────────────────────────────────────────────────────

def simulate_games_batched(
    model,
    context_features,   # (ctx_dim,) numpy — already normalised
    N,                  # number of simulations
    device,
    idx_to_ev,          # {int: str}
    temperature=1.0,
    forward_chunk=2048,
    max_game_pitches=1200,
    init_state=None,    # dict with initial game state from prefix replay (or None)
):
    """
    Simulate *N* complete baseball games for a single game context.

    If init_state is provided, all N simulations start from that game state
    (inning, score, outs, bases, count) rather than from the beginning.

    All N simulations share the same static context but diverge via stochastic
    sampling.  The expensive forward passes are batched; per-simulation state
    updates use fast numpy vectorised ops + a short Python loop only for the
    ~25 % of sims whose at-bat ends on each step.

    Returns dict:  home_wins (int), home_scores (N,), away_scores (N,)
    """
    model.eval()
    n_cont = len(CONTINUOUS_PITCH_COLS)
    start_pt   = len(PITCH_TYPES)
    start_zone = len(ZONES)
    start_pr   = len(PITCH_RESULTS)
    MAX_AB     = 30            # max pitches per at-bat (safety cap)

    # ── state arrays (N,) ──
    if init_state is not None:
        inning     = np.full(N, init_state['inning'], dtype=np.int32)
        is_top     = np.full(N, init_state['is_top'], dtype=bool)
        outs       = np.full(N, init_state['outs'], dtype=np.int32)
        bases      = np.tile(np.array(init_state['bases'], dtype=bool), (N, 1))
        home_score = np.full(N, init_state['home_score'], dtype=np.int32)
        away_score = np.full(N, init_state['away_score'], dtype=np.int32)
        balls      = np.full(N, init_state['balls'], dtype=np.int32)
        strikes    = np.full(N, init_state['strikes'], dtype=np.int32)
    else:
        inning     = np.ones(N, dtype=np.int32)
        is_top     = np.ones(N, dtype=bool)          # True → away batting
        outs       = np.zeros(N, dtype=np.int32)
        bases      = np.zeros((N, 3), dtype=bool)    # columns: 1B 2B 3B
        home_score = np.zeros(N, dtype=np.int32)
        away_score = np.zeros(N, dtype=np.int32)
        balls      = np.zeros(N, dtype=np.int32)
        strikes    = np.zeros(N, dtype=np.int32)
    game_over  = np.zeros(N, dtype=bool)

    # ── per-sim at-bat history buffers ──
    ab_len   = np.zeros(N, dtype=np.int32)
    hist_pt  = np.full((N, MAX_AB), start_pt,   dtype=np.int64)
    hist_z   = np.full((N, MAX_AB), start_zone, dtype=np.int64)
    hist_pr  = np.full((N, MAX_AB), start_pr,   dtype=np.int64)
    hist_c   = np.zeros((N, MAX_AB, n_cont),    dtype=np.float32)
    hist_gs  = np.zeros((N, MAX_AB, 9),         dtype=np.float32)

    ctx_t = torch.tensor(context_features, dtype=torch.float32, device=device)

    with torch.no_grad():
        for step in range(max_game_pitches):
            active_mask = ~game_over
            if not active_mask.any():
                break
            aidx = np.where(active_mask)[0]
            na   = len(aidx)

            # ── 1. write game-state row for current pitch position ──
            pos = ab_len[aidx]
            # clamp to MAX_AB-1 for safety
            pos = np.minimum(pos, MAX_AB - 1)
            sd = np.where(is_top[aidx],
                          away_score[aidx] - home_score[aidx],
                          home_score[aidx] - away_score[aidx])
            gs_rows = np.stack([
                balls[aidx].astype(np.float32),
                strikes[aidx].astype(np.float32),
                outs[aidx].astype(np.float32),
                inning[aidx].astype(np.float32) / 9.0,
                sd.astype(np.float32) / 10.0,
                bases[aidx, 0].astype(np.float32),
                bases[aidx, 1].astype(np.float32),
                bases[aidx, 2].astype(np.float32),
                is_top[aidx].astype(np.float32),
            ], axis=1)                                       # (na,9)
            hist_gs[aidx, pos, :] = gs_rows

            # ── 2. construct batch tensors & run forward pass in chunks ──
            cur_lens = pos + 1                               # (na,)
            max_len  = int(cur_lens.max())

            # allocate output storage
            s_pt_all   = np.empty(na, dtype=np.int64)
            s_z_all    = np.empty(na, dtype=np.int64)
            s_pr_all   = np.empty(na, dtype=np.int64)
            s_c_all    = np.empty((na, n_cont), dtype=np.float32)
            ev_log_all = torch.empty(na, len(AT_BAT_EVENTS), device=device)

            for cs in range(0, na, forward_chunk):
                ce = min(cs + forward_chunk, na)
                cidx  = aidx[cs:ce]
                csz   = ce - cs
                clens = cur_lens[cs:ce]
                cml   = int(clens.max())

                ctx_b  = ctx_t.unsqueeze(0).unsqueeze(0).expand(csz, cml, -1)
                gs_b   = torch.tensor(hist_gs[cidx, :cml],  dtype=torch.float32, device=device)
                pt_b   = torch.tensor(hist_pt[cidx, :cml],  dtype=torch.long,    device=device)
                z_b    = torch.tensor(hist_z[cidx,  :cml],  dtype=torch.long,    device=device)
                pr_b   = torch.tensor(hist_pr[cidx, :cml],  dtype=torch.long,    device=device)
                c_b    = torch.tensor(hist_c[cidx,  :cml],  dtype=torch.float32, device=device)

                out = model(ctx_b, gs_b, pt_b, z_b, pr_b, c_b)

                lp = torch.tensor(clens - 1, dtype=torch.long, device=device)
                bi = torch.arange(csz, device=device)

                pt_lo = out['pitch_type_logits'][bi, lp]   / temperature
                z_lo  = out['zone_logits'][bi, lp]         / temperature
                pr_lo = out['pitch_result_logits'][bi, lp] / temperature
                ev_lo = out['at_bat_event_logits'][bi, lp] / temperature

                # Sample continuous via conditioned DDPM
                # Get transformer latent for conditioning: (csz, S, d_model) → (csz, d_model, S)
                latent = out['continuous_latent'].permute(0, 2, 1)
                seq_length = model.ddpm.seq_length
                # Pad or truncate to DDPM's expected seq_length
                if latent.shape[2] < seq_length:
                    latent = F.pad(latent, (0, seq_length - latent.shape[2]))
                elif latent.shape[2] > seq_length:
                    latent = latent[:, :, :seq_length]
                # DDIM sample (fast, ~20 steps configured in model)
                sampled_seq = model.ddpm.sample(batch_size=csz, cond=latent)
                # Extract continuous values at each sim's current position
                # sampled_seq: (csz, n_continuous, seq_length)
                lp_clamped = lp.clamp(max=seq_length - 1)
                sc = sampled_seq[bi, :, lp_clamped].cpu().numpy()

                sp = torch.multinomial(F.softmax(pt_lo, -1), 1).squeeze(-1).cpu().numpy()
                sz = torch.multinomial(F.softmax(z_lo,  -1), 1).squeeze(-1).cpu().numpy()
                sr = torch.multinomial(F.softmax(pr_lo, -1), 1).squeeze(-1).cpu().numpy()

                s_pt_all[cs:ce]   = sp
                s_z_all[cs:ce]    = sz
                s_pr_all[cs:ce]   = sr
                s_c_all[cs:ce]    = sc
                ev_log_all[cs:ce] = ev_lo
            
            # ── 3. vectorised count update ──
            safe_pr = np.clip(s_pr_all, 0, _N_PR - 1)
            is_strike = _IS_STRIKE[safe_pr]
            is_foul   = _IS_FOUL[safe_pr]
            is_ball   = _IS_BALL[safe_pr]
            is_hbp    = _IS_HBP[safe_pr]
            is_hip    = _IS_HIP[safe_pr]

            strikes[aidx[is_strike]] += 1
            foul_ok = is_foul & (strikes[aidx] < 2)
            strikes[aidx[foul_ok]] += 1
            balls[aidx[is_ball]] += 1

            # ── 4. write history for next pitch ──
            nxt = np.minimum(pos + 1, MAX_AB - 1)
            still = ~game_over[aidx]
            si = aidx[still]; sj = np.where(still)[0]
            hist_pt[si, nxt[still]] = s_pt_all[sj]
            hist_z[si,  nxt[still]] = s_z_all[sj]
            hist_pr[si, nxt[still]] = s_pr_all[sj]
            hist_c[si,  nxt[still]] = s_c_all[sj]
            ab_len[si] += 1
            
            # ── 5. at-bat termination check ──
            k_mask  = strikes[aidx] >= 3
            bb_mask = balls[aidx]   >= 4
            ended   = k_mask | bb_mask | is_hbp | is_hip | (ab_len[aidx] >= MAX_AB)
            if not ended.any():
                continue

            e_local = np.where(ended)[0]          # index into aidx
            e_sim   = aidx[ended]                  # global sim index

            # ── 6. determine event per ended at-bat ──
            for k_loc, k_sim in zip(e_local, e_sim):
                if is_hip[k_loc]:
                    ev_logit = ev_log_all[k_loc].unsqueeze(0)
                    ev_idx = torch.multinomial(F.softmax(ev_logit, -1), 1).item()
                    event = idx_to_ev.get(ev_idx, 'field_out')
                elif is_hbp[k_loc]:
                    event = 'hit_by_pitch'
                elif k_mask[k_loc]:
                    event = 'strikeout'
                elif bb_mask[k_loc]:
                    event = 'walk'
                else:
                    event = 'field_out'   # safety (MAX_AB hit)

                # apply event
                new_outs, runs, nb0, nb1, nb2 = _apply_event(
                    event, int(outs[k_sim]),
                    bool(bases[k_sim, 0]), bool(bases[k_sim, 1]), bool(bases[k_sim, 2]),
                )
                outs[k_sim] = new_outs
                bases[k_sim] = [nb0, nb1, nb2]
                if is_top[k_sim]:
                    away_score[k_sim] += runs
                else:
                    home_score[k_sim] += runs

                # ── walk-off ──
                if (not is_top[k_sim]) and inning[k_sim] >= 9:
                    if home_score[k_sim] > away_score[k_sim]:
                        game_over[k_sim] = True
                        _reset_ab(k_sim, ab_len, balls, strikes,
                                  hist_pt, hist_z, hist_pr, hist_c, hist_gs,
                                  start_pt, start_zone, start_pr)
                        continue

                # ── half-inning end ──
                if outs[k_sim] >= 3:
                    if is_top[k_sim]:
                        # Home already ahead after top of 9+ → game over
                        if inning[k_sim] >= 9 and home_score[k_sim] > away_score[k_sim]:
                            game_over[k_sim] = True
                            _reset_ab(k_sim, ab_len, balls, strikes,
                                      hist_pt, hist_z, hist_pr, hist_c, hist_gs,
                                      start_pt, start_zone, start_pr)
                            continue
                        # switch to bottom
                        is_top[k_sim] = False
                        outs[k_sim] = 0
                        bases[k_sim] = [False, False, False]
                        if inning[k_sim] > 9:
                            bases[k_sim, 1] = True      # ghost runner
                    else:
                        # end-of-inning: game over?
                        if inning[k_sim] >= 9 and home_score[k_sim] != away_score[k_sim]:
                            game_over[k_sim] = True
                            _reset_ab(k_sim, ab_len, balls, strikes,
                                      hist_pt, hist_z, hist_pr, hist_c, hist_gs,
                                      start_pt, start_zone, start_pr)
                            continue
                        inning[k_sim] += 1
                        is_top[k_sim] = True
                        outs[k_sim] = 0
                        bases[k_sim] = [False, False, False]
                        if inning[k_sim] > 9:
                            bases[k_sim, 1] = True      # ghost runner

                # reset at-bat
                _reset_ab(k_sim, ab_len, balls, strikes,
                          hist_pt, hist_z, hist_pr, hist_c, hist_gs,
                          start_pt, start_zone, start_pr)

    return {
        'home_wins':   int((home_score > away_score).sum()),
        'ties':        int((home_score == away_score).sum()),
        'home_scores': home_score,
        'away_scores': away_score,
    }


def _reset_ab(i, ab_len, balls, strikes,
              hist_pt, hist_z, hist_pr, hist_c, hist_gs,
              start_pt, start_zone, start_pr):
    ab_len[i]  = 0
    balls[i]   = 0
    strikes[i] = 0
    hist_pt[i] = start_pt
    hist_z[i]  = start_zone
    hist_pr[i] = start_pr
    hist_c[i]  = 0
    hist_gs[i] = 0


# ─────────────────────────────────────────────────────────────────────────────
# Baselines
# ─────────────────────────────────────────────────────────────────────────────

def log5(h, a):
    """Log5 win probability for home team.  h,a ∈ (0,1)."""
    denom = h + a - 2.0 * h * a
    if abs(denom) < 1e-9:
        return 0.5
    return (h - h * a) / denom


def mlp_baseline(train_gc, train_gt, test_gc, ctx_columns, ctx_mean, ctx_std,
                 test_prefix_states=None, eval_idx=None):
    """Train an MLP classifier on game-context features → home_win.

    When test_prefix_states is provided (list-of-dicts parallel to eval_idx),
    the prefix game-state is appended as extra features for the test rows that
    are actually evaluated.  Training data always has prefix features set to
    the start-of-game defaults (zeros) so the model learns how to use them.
    """
    from sklearn.neural_network import MLPClassifier

    PREFIX_COLS = ['inning', 'is_top', 'outs', 'home_score', 'away_score',
                   'on_1b', 'on_2b', 'on_3b', 'balls', 'strikes']
    n_prefix = len(PREFIX_COLS)

    # Prepare train features
    X_train = train_gc[ctx_columns].values.astype(np.float64)
    X_train = (X_train - ctx_mean) / ctx_std
    y_train = train_gt['home_win_exp'].values

    # Append start-of-game prefix features for training (all zeros / defaults)
    prefix_train = np.zeros((len(X_train), n_prefix), dtype=np.float64)
    prefix_train[:, 0] = 1.0   # inning = 1
    prefix_train[:, 1] = 1.0   # is_top = True
    X_train = np.hstack([X_train, prefix_train])

    # Prepare test features
    X_test = test_gc.reindex(columns=ctx_columns, fill_value=0).values.astype(np.float64)
    X_test = (X_test - ctx_mean) / ctx_std

    # Build prefix features for test set
    prefix_test = np.zeros((len(X_test), n_prefix), dtype=np.float64)
    prefix_test[:, 0] = 1.0   # default inning
    prefix_test[:, 1] = 1.0   # default is_top
    if test_prefix_states is not None and eval_idx is not None:
        for j, gi in enumerate(eval_idx):
            st = test_prefix_states[j]
            if st is not None:
                prefix_test[gi] = [
                    st['inning'], float(st['is_top']), st['outs'],
                    st['home_score'], st['away_score'],
                    float(st['bases'][0]), float(st['bases'][1]), float(st['bases'][2]),
                    st['balls'], st['strikes'],
                ]
    X_test = np.hstack([X_test, prefix_test])

    # Handle NaN
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_test  = np.nan_to_num(X_test,  nan=0.0, posinf=0.0, neginf=0.0)

    model = MLPClassifier(hidden_layer_sizes=(1000, 1000, 1000), max_iter=500,
                          early_stopping=True, validation_fraction=0.1,
                          random_state=42)
    model.fit(X_train, y_train)

    probs = model.predict_proba(X_test)[:, 1]
    return probs, model


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def brier_score(probs, actuals):
    return float(np.mean((probs - actuals) ** 2))


def log_loss(probs, actuals, eps=1e-7):
    p = np.clip(probs, eps, 1.0 - eps)
    return float(-np.mean(actuals * np.log(p) + (1 - actuals) * np.log(1 - p)))


def accuracy(probs, actuals):
    preds = (probs >= 0.5).astype(float)
    return float(np.mean(preds == actuals))


def calibration_table(probs, actuals, n_bins=10):
    """Return a DataFrame with calibration info per bucket."""
    bins = np.linspace(0, 1, n_bins + 1)
    rows = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (probs >= lo) & (probs < hi)
        if mask.sum() == 0:
            continue
        rows.append({
            'bin': f'{lo:.1f}-{hi:.1f}',
            'n': int(mask.sum()),
            'mean_pred': float(probs[mask].mean()),
            'mean_actual': float(actuals[mask].mean()),
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='MC Win Probability Evaluation')
    p.add_argument('--model_path', default='pitch_sequence_model.pth')
    p.add_argument('--train', action='store_true', help='Train model before evaluating')
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--lr', type=float, default=5e-4)
    p.add_argument('--n_sims', type=int, default=10, help='Simulations per game')
    p.add_argument('--n_games', type=int, default=0,
                   help='Number of test games to evaluate (0 = all)')
    p.add_argument('--temperature', type=float, default=1)
    p.add_argument('--forward_chunk', type=int, default=256,
                   help='Max batch size for model forward pass')
    p.add_argument('--n_threads', type=int, default=12,
                   help='Number of threads for parallel simulation')    
    p.add_argument('--cache_dir', type=str, default='cache',
                   help='Directory to store/load preprocessed data cache')
    p.add_argument('--no_cache', action='store_true',
                   help='Disable caching (always reprocess data)')    
    p.add_argument('--ddpm_steps', type=int, default=20,
                   help='DDIM sampling steps for continuous features (fewer=faster)')
    p.add_argument('--d_model', type=int, default=8, help='Transformer hidden dim')
    p.add_argument('--n_layers', type=int, default=2, help='Transformer layers')
    p.add_argument('--n_heads', type=int, default=2, help='Attention heads')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--train_pct', type=float, default=1)
    p.add_argument('--test_pct', type=float, default=1)
    p.add_argument('--prefix_inning', type=float, default=1.0,
                   help='Inning to start simulating from (1 = from scratch, '
                        '8 = use innings 1-7 as context, 7.5 = use through top of 7)')
    # MCMC options
    p.add_argument('--mcmc', action='store_true',
                   help='Use Metropolis-Hastings MCMC instead of batched Monte Carlo')
    p.add_argument('--mcmc_steps', type=int, default=500,
                   help='Number of post-burn-in MH steps per game')
    p.add_argument('--mcmc_burnin', type=int, default=100,
                   help='Number of burn-in steps to discard before collecting samples')
    p.add_argument('--lambda_cal', type=float, default=0.5,
                   help='RE24 calibration weight λ (0=pure model, 1=full calibration)')
    return p.parse_args()


def simulate_single_game(args_tuple):
    """
    Worker function for simulating a single game.
    Returns: (game_index, mc_prob, mc_avg_home, mc_avg_away)
    """
    (j, gi, model, test_gc_features, ctx_mean, ctx_std, n_sims, 
     device, idx_to_ev, temperature, forward_chunk, test_game_ids, 
     progress_counter, progress_lock, n_eval, t0, actuals, log5_probs,
     init_state) = args_tuple
    
    try:
        # Prepare context
        ctx_row = test_gc_features.iloc[gi].values.astype(np.float32)
        ctx_normed = (ctx_row - ctx_mean) / ctx_std
        ctx_normed = np.nan_to_num(ctx_normed, nan=0.0)
        result = simulate_games_batched(
            model, ctx_normed, n_sims, device,
            idx_to_ev=idx_to_ev,
            temperature=temperature,
            forward_chunk=forward_chunk,
            init_state=init_state,
        )
        
        mc_prob = result['home_wins'] / n_sims
        mc_avg_h = result['home_scores'].mean()
        mc_avg_a = result['away_scores'].mean()
       
        # Update progress with thread safety
        with progress_lock:
            progress_counter[0] += 1
            completed = progress_counter[0]
            elapsed = time.time() - t0
            per_game = elapsed / completed if completed > 0 else 0
            eta = per_game * (n_eval - completed)
            game_id = test_game_ids[gi] if test_game_ids is not None else gi
            actual = actuals[j]
            log5_prob = log5_probs[j]
            print(f'  [{completed:4d}/{n_eval}] {game_id}  '
                  f'MC: {mc_prob:.3f}  Log5: {log5_prob:.3f}  '
                  f'Actual: {actual:.0f}  '
                  f'Avg score: {mc_avg_h:.1f}-{mc_avg_a:.1f}  '
                  f'({per_game:.1f}s/game, ETA {eta/60:.0f}min)')
        
        return j, mc_prob, mc_avg_h, mc_avg_a
        
    except Exception as e:
        print(f'Error in game {gi}: {e}')
        return j, 0.5, 0.0, 0.0  # Default values on error


def save_preprocessed_data(cache_dir, data_dict):
    """Save preprocessed data to cache directory."""
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, 'preprocessed_data.pkl')
    print(f'Saving preprocessed data to {cache_file}...')
    with open(cache_file, 'wb') as f:
        pickle.dump(data_dict, f)
    print(f'Cache saved successfully.')


def load_preprocessed_data(cache_dir):
    """Load preprocessed data from cache directory."""
    cache_file = os.path.join(cache_dir, 'preprocessed_data.pkl')
    if not os.path.exists(cache_file):
        return None
    
    print(f'Loading preprocessed data from {cache_file}...')
    try:
        with open(cache_file, 'rb') as f:
            data_dict = pickle.load(f)
        print(f'Cache loaded successfully.')
        return data_dict
    except Exception as e:
        print(f'Failed to load cache: {e}')
        return None


def check_cache_validity(cache_dir, expected_files):
    """Check if cache is still valid by comparing file modification times."""
    cache_file = os.path.join(cache_dir, 'preprocessed_data.pkl')
    if not os.path.exists(cache_file):
        return False
    
    cache_time = os.path.getmtime(cache_file)
    
    # Check if any expected data files are newer than cache
    for file_path in expected_files:
        if os.path.exists(file_path):
            if os.path.getmtime(file_path) > cache_time:
                print(f'Cache outdated: {file_path} is newer than cache')
                return False
    
    return True


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available()
                          else 'mps' if torch.backends.mps.is_available()
                          else 'cpu')
    print(f'Device: {device}')

    # ── Check for cached preprocessed data ──
    cached_data = None
    expected_files = [
        'train_pitch_2020-05-12_2025-08-01.csv',
        'train_pitch_context_2020-05-12_2025-08-01.csv',
        'train_pitch_result_2020-05-12_2025-08-01.csv',
        'train_at_bat_target_2020-05-12_2025-08-01.csv',
        'train_game_context_2020-05-12_2025-08-01.csv',
        'test_pitch_2025-08-01_2025-11-03.csv',
        'test_pitch_context_2025-08-01_2025-11-03.csv',
        'test_pitch_result_2025-08-01_2025-11-03.csv',
        'test_at_bat_target_2025-08-01_2025-11-03.csv',
        'test_game_context_2025-08-01_2025-11-03.csv',
        'train_Game_target_2020-05-12_2025-08-01.csv',
        'test_Game_target_2025-08-01_2025-11-03.csv',
    ]
    
    if not args.no_cache:
        if check_cache_validity(args.cache_dir, expected_files):
            cached_data = load_preprocessed_data(args.cache_dir)
    
    if cached_data is not None:
        print('Using cached preprocessed data.')
        # Extract cached data
        train_pitch = cached_data['train_pitch']
        train_pc = cached_data['train_pc']
        train_pr = cached_data['train_pr']
        train_ab = cached_data['train_ab']
        train_gc = cached_data['train_gc']
        test_pitch = cached_data['test_pitch']
        test_pc = cached_data['test_pc']
        test_pr = cached_data['test_pr']
        test_ab = cached_data['test_ab']
        test_gc = cached_data['test_gc']
        train_gt = cached_data['train_gt']
        test_gt = cached_data['test_gt']
        pt_to_idx = cached_data['pt_to_idx']
        pr_to_idx = cached_data['pr_to_idx']
        ev_to_idx = cached_data['ev_to_idx']
        zone_to_idx = cached_data['zone_to_idx']
        idx_to_ev = cached_data['idx_to_ev']
        ctx_columns = cached_data['ctx_columns']
        ctx_mean = cached_data['ctx_mean']
        ctx_std = cached_data['ctx_std']
        test_game_ids = cached_data['test_game_ids']
        test_gc_features = cached_data['test_gc_features']
        train_gc_features = cached_data['train_gc_features']
        train_gc_proc = cached_data['train_gc_proc']
        test_gc_proc = cached_data['test_gc_proc']
        pitch_mean = cached_data['pitch_mean']
        pitch_std = cached_data['pitch_std']
        context_dim = cached_data['context_dim']
        re24_table = cached_data.get('re24_table', None)  # None triggers hardcoded fallback
    else:
        print('Processing data from scratch...')
        
        # ── Load data ──
        print('Loading data...')
        train_pitch, train_pc, train_pr, train_ab, train_gc = load_data('train')
        test_pitch,  test_pc,  test_pr,  test_ab,  test_gc  = load_data('test')

        train_gt = pd.read_csv('train_Game_target_2020-05-12_2025-08-01.csv')
        test_gt  = pd.read_csv('test_Game_target_2025-08-01_2025-11-03.csv')

        pt_to_idx, pr_to_idx, ev_to_idx, zone_to_idx = build_vocab_maps()
        idx_to_ev = {v: k for k, v in ev_to_idx.items()}

        # ── Process game context ──
        print('Processing game context...')
        train_gc_proc, ctx_columns, ctx_mean, ctx_std = prepare_game_context(train_gc, is_train=True)

        # Align test game context to train columns
        test_gc_proc = test_gc.drop(columns=['game_date'], errors='ignore').copy()
        test_game_ids = test_gc_proc['game_id'].values if 'game_id' in test_gc_proc.columns else None
        test_gc_features = test_gc_proc.drop(columns=['game_id'], errors='ignore')
        test_gc_features = pd.get_dummies(test_gc_features, drop_first=True).astype(float)
        test_gc_features = test_gc_features.reindex(columns=ctx_columns, fill_value=0)

        # Also need train features for MLP baseline (before normalization)
        train_gc_features = train_gc_proc.drop(columns=['game_id'], errors='ignore')

        # Continuous pitch normalisation
        print('Computing pitch normalisation stats...')
        cont_vals = train_pitch[CONTINUOUS_PITCH_COLS].values.astype(np.float32)
        cont_vals = np.nan_to_num(cont_vals, nan=0.0)
        pitch_mean = cont_vals.mean(axis=0)
        pitch_std  = cont_vals.std(axis=0)
        pitch_std[pitch_std < 1e-8] = 1.0

        context_dim = len(ctx_columns)

        # ── Compute RE24 table from training pitch context ──
        print('Computing RE24 run-expectancy table from training data...')
        re24_table = compute_re24_from_pitch_context(train_pc)

        # ── Save preprocessed data to cache ──
        if not args.no_cache:
            cache_data = {
                'train_pitch': train_pitch,
                'train_pc': train_pc,
                'train_pr': train_pr,
                'train_ab': train_ab,
                'train_gc': train_gc,
                'test_pitch': test_pitch,
                'test_pc': test_pc,
                'test_pr': test_pr,
                'test_ab': test_ab,
                'test_gc': test_gc,
                'train_gt': train_gt,
                'test_gt': test_gt,
                'pt_to_idx': pt_to_idx,
                'pr_to_idx': pr_to_idx,
                'ev_to_idx': ev_to_idx,
                'zone_to_idx': zone_to_idx,
                'idx_to_ev': idx_to_ev,
                'ctx_columns': ctx_columns,
                'ctx_mean': ctx_mean,
                'ctx_std': ctx_std,
                'test_game_ids': test_game_ids,
                'test_gc_features': test_gc_features,
                'train_gc_features': train_gc_features,
                'train_gc_proc': train_gc_proc,
                'test_gc_proc': test_gc_proc,
                'pitch_mean': pitch_mean,
                'pitch_std': pitch_std,
                'context_dim': context_dim,
                're24_table': re24_table,
            }
            save_preprocessed_data(args.cache_dir, cache_data)

    # ── Train or load model ──
    if args.train or not os.path.exists(args.model_path):
        print(f'\n{"="*60}')
        print('  Training pitch sequence model')
        print(f'{"="*60}')

        print('Building datasets...')
        train_gc_with_id = train_gc_features.copy()
        train_gc_with_id['game_id'] = train_gc_proc['game_id'].values if 'game_id' in train_gc_proc.columns else train_gc['game_id'].values

        test_gc_with_id = test_gc_features.copy()
        if test_game_ids is not None:
            test_gc_with_id['game_id'] = test_game_ids

        # Subset by game (not pitch) so we don't split mid-game
        if args.train_pct < 1.0:
            all_game_ids = train_pitch['game_id'].unique()
            n_games = max(1, int(len(all_game_ids) * args.train_pct))
            keep_games = set(all_game_ids[-n_games:])
            mask = train_pitch['game_id'].isin(keep_games)
            train_pitch = train_pitch[mask].reset_index(drop=True)
            train_pc = train_pc[mask].reset_index(drop=True)
            train_pr = train_pr[mask].reset_index(drop=True)
            train_ab = train_ab[mask].reset_index(drop=True)

        if args.test_pct < 1.0:
            all_test_ids = test_pitch['game_id'].unique()
            m_games = max(1, int(len(all_test_ids) * args.test_pct))
            keep_test = set(all_test_ids[-m_games:])
            mask = test_pitch['game_id'].isin(keep_test)
            test_pitch = test_pitch[mask].reset_index(drop=True)
            test_pc = test_pc[mask].reset_index(drop=True)
            test_pr = test_pr[mask].reset_index(drop=True)
            test_ab = test_ab[mask].reset_index(drop=True)

        # Print date ranges and pitch counts
        if 'game_date' in train_pitch.columns:
            print(f'  Train date range: {train_pitch["game_date"].min()} to {train_pitch["game_date"].max()}')
        if 'game_date' in test_pitch.columns:
            print(f'  Test  date range: {test_pitch["game_date"].min()} to {test_pitch["game_date"].max()}')
        print(f'  Train pitches: {len(train_pitch):,}')
        print(f'  Test  pitches: {len(test_pitch):,}')

        train_dataset = GameSequenceDataset(
            train_pitch, train_pc, train_pr, train_ab, train_gc_with_id,
            pt_to_idx, pr_to_idx, ev_to_idx, zone_to_idx,
            ctx_columns, ctx_mean, ctx_std, pitch_mean, pitch_std,
            max_pitches=352,
        )
        test_dataset = GameSequenceDataset(
            test_pitch, test_pc, test_pr, test_ab, test_gc_with_id,
            pt_to_idx, pr_to_idx, ev_to_idx, zone_to_idx,
            ctx_columns, ctx_mean, ctx_std, pitch_mean, pitch_std,
            max_pitches=352,
        )
        n_games_train = len(train_dataset) // len(GameSequenceDataset.INNING_CHOICES)
        n_games_test = len(test_dataset) // len(GameSequenceDataset.INNING_CHOICES)
        print(f'  Train: {len(train_dataset)} samples ({n_games_train} games x {len(GameSequenceDataset.INNING_CHOICES)} inning cuts)')
        print(f'  Test:  {len(test_dataset)} samples ({n_games_test} games x {len(GameSequenceDataset.INNING_CHOICES)} inning cuts)')

        model = PitchSequenceTransfusion(
            context_dim=context_dim, d_model=args.d_model, n_heads=args.n_heads,
            n_layers=args.n_layers, dropout=0, max_seq_len=352,
        )
        print(f'  Parameters: {sum(p.numel() for p in model.parameters()):,}')

        model = train_model(
            model, train_dataset, test_dataset,
            epochs=args.epochs, batch_size=32,
            lr=args.lr, lambda_continuous=1.0,
            collate_fn=collate_games,
        )

        torch.save({
            'model_state_dict': model.state_dict(),
            'ctx_columns': list(ctx_columns),
            'ctx_mean': ctx_mean,
            'ctx_std': ctx_std,
            'pitch_mean': pitch_mean,
            'pitch_std': pitch_std,
            'context_dim': context_dim,
            'd_model': args.d_model,
            'n_layers': args.n_layers,
            'n_heads': args.n_heads,
        }, args.model_path)
        print(f'Model saved to {args.model_path}')
    else:
        print(f'Loading model from {args.model_path}...')
        ckpt = torch.load(args.model_path, map_location='cpu', weights_only=False)
        # Use checkpoint's normalisation stats if available
        if 'ctx_mean' in ckpt:
            ctx_mean = ckpt['ctx_mean']
            ctx_std  = ckpt['ctx_std']
        if 'pitch_mean' in ckpt:
            pitch_mean = ckpt['pitch_mean']
            pitch_std  = ckpt['pitch_std']
        if 'ctx_columns' in ckpt:
            ctx_columns = ckpt['ctx_columns']
            test_gc_features = test_gc_features.reindex(columns=ctx_columns, fill_value=0)
            context_dim = len(ctx_columns)

        d_model = ckpt.get('d_model', args.d_model)
        n_layers = ckpt.get('n_layers', args.n_layers)
        n_heads = ckpt.get('n_heads', args.n_heads)
        model = PitchSequenceTransfusion(
            context_dim=context_dim, d_model=d_model, n_heads=n_heads,
            n_layers=n_layers, dropout=0.1, max_seq_len=352,
        )
        model.load_state_dict(ckpt['model_state_dict'])

    # Apply DDIM sampling steps from CLI arg
    model.ddpm.sampling_timesteps = args.ddpm_steps
    model.ddpm.is_ddim_sampling = args.ddpm_steps < model.ddpm.num_timesteps

    model = model.to(device).eval()

    # ── Select test games ──
    n_test = len(test_gt)
    if args.n_games > 0:
        n_eval = min(args.n_games, n_test)
        eval_idx = np.random.choice(n_test, n_eval, replace=False)
        eval_idx.sort()
    else:
        n_eval = n_test
        eval_idx = np.arange(n_test)

    print(f'\n{"="*60}')
    print(f'  Evaluating on {n_eval} test games  ×  {args.n_sims} sims each')
    print(f'{"="*60}')

    # ── Compute baselines ──
    print('\nComputing baselines...')
    actuals = test_gt['home_win_exp'].values[eval_idx]
    hist_rate = train_gt['home_win_exp'].mean()

    # Log5
    log5_probs = np.zeros(n_eval)
    for j, gi in enumerate(eval_idx):
        h = test_gc.iloc[gi].get('home_team_win_pct', 0.5)
        a = test_gc.iloc[gi].get('away_team_win_pct', 0.5)
        if pd.isna(h) or h == 0: h = 0.5
        if pd.isna(a) or a == 0: a = 0.5
        log5_probs[j] = log5(h, a)

    # Compute true final scores for eval games (max score seen across all pitches)
    true_home_scores = np.zeros(n_eval)
    true_away_scores = np.zeros(n_eval)
    for j, gi in enumerate(eval_idx):
        game_id = test_game_ids[gi] if test_game_ids is not None else gi
        gm = test_pitch['game_id'] == game_id
        if gm.any():
            true_home_scores[j] = test_pc.loc[gm, 'home_score'].max()
            true_away_scores[j] = test_pc.loc[gm, 'away_score'].max()

    # Pre-compute prefix states for eval games (used by both MLP baseline and MC sim)
    prefix_states = [None] * n_eval
    if args.prefix_inning > 1:
        print(f'  Extracting prefix states (sim from inning {args.prefix_inning})...')
        for j, gi in enumerate(eval_idx):
            game_id = test_game_ids[gi] if test_game_ids is not None else gi
            prefix_states[j] = extract_game_state_at_inning(
                game_id, args.prefix_inning,
                test_pitch, test_pc, test_pr, test_ab,
                pt_to_idx, pr_to_idx, ev_to_idx, zone_to_idx,
                pitch_mean, pitch_std,
            )

    # MLP classifier
    try:
        mlp_probs_all, mlp_model = mlp_baseline(
            train_gc_features, train_gt, test_gc_features,
            ctx_columns, ctx_mean, ctx_std,
            test_prefix_states=prefix_states if args.prefix_inning > 1 else None,
            eval_idx=eval_idx,
        )
        mlp_probs = mlp_probs_all[eval_idx]
        have_mlp = True
    except Exception as e:
        print(f'  MLP classifier failed: {e}')
        mlp_probs = np.full(n_eval, hist_rate)
        have_mlp = False

    # ── Simulation (MC or MCMC) ──
    use_mcmc = getattr(args, 'mcmc', False)
    sim_label = 'MCMC (MH)' if use_mcmc else 'Monte Carlo'
    print(f'\nRunning {sim_label} simulations using {args.n_threads} threads...')
    mc_probs  = np.zeros(n_eval)
    mc_avg_hs = np.zeros(n_eval)
    mc_avg_as = np.zeros(n_eval)
    mcmc_acceptance_rates = np.zeros(n_eval)  # only populated when --mcmc is set

    # Thread-safe progress tracking
    progress_counter = [0]  # Use list for mutability
    progress_lock = threading.Lock()
    t0 = time.time()

    if use_mcmc:
        # ── MCMC path: one MH chain per game, single-threaded ──
        # (Chains are inherently sequential; parallelism across games is future work.)
        for j, gi in enumerate(eval_idx):
            ctx_row = test_gc_features.iloc[gi].values.astype(np.float32)
            ctx_normed = (ctx_row - ctx_mean) / ctx_std
            ctx_normed = np.nan_to_num(ctx_normed, nan=0.0)

            simulator = GameSimulator(
                model=model,
                context_features=ctx_normed,
                pt_to_idx=pt_to_idx,
                pr_to_idx=pr_to_idx,
                ev_to_idx=ev_to_idx,
                zone_to_idx=zone_to_idx,
                pitch_mean=pitch_mean,
                pitch_std=pitch_std,
                device=device,
            )
            sampler = MHGameSampler(
                simulator=simulator,
                lambda_cal=args.lambda_cal,
                temperature=args.temperature,
                re24_table=re24_table,
            )
            chain_result = sampler.run_chain(
                n_steps=args.mcmc_steps,
                burn_in=args.mcmc_burnin,
            )

            mc_probs[j] = chain_result['win_probability']
            mcmc_acceptance_rates[j] = chain_result['acceptance_rate']
            # MCMC doesn't track per-game average scores; leave at 0.

            elapsed = time.time() - t0
            per_game = elapsed / (j + 1)
            eta = per_game * (n_eval - j - 1)
            game_id = test_game_ids[gi] if test_game_ids is not None else gi
            actual = actuals[j]
            print(f'  [{j+1:4d}/{n_eval}] {game_id}  '
                  f'MCMC: {mc_probs[j]:.3f}  Log5: {log5_probs[j]:.3f}  '
                  f'Actual: {actual:.0f}  '
                  f'AcceptRate: {mcmc_acceptance_rates[j]:.2f}  '
                  f'({per_game:.1f}s/game, ETA {eta/60:.0f}min)')

    elif args.n_threads == 1:
        # Single-threaded MC (original behavior)
        for j, gi in enumerate(eval_idx):
            # Prepare context
            ctx_row = test_gc_features.iloc[gi].values.astype(np.float32)
            ctx_normed = (ctx_row - ctx_mean) / ctx_std
            ctx_normed = np.nan_to_num(ctx_normed, nan=0.0)

            # Use pre-computed prefix state
            init_state = prefix_states[j]

            result = simulate_games_batched(
                model, ctx_normed, args.n_sims, device,
                idx_to_ev=idx_to_ev,
                temperature=args.temperature,
                forward_chunk=args.forward_chunk,
                init_state=init_state,
            )
            mc_probs[j]  = result['home_wins'] / args.n_sims
            mc_avg_hs[j] = result['home_scores'].mean()
            mc_avg_as[j] = result['away_scores'].mean()

            elapsed = time.time() - t0
            per_game = elapsed / (j + 1)
            eta = per_game * (n_eval - j - 1)
            game_id = test_game_ids[gi] if test_game_ids is not None else gi
            actual = actuals[j]
            print(f'  [{j+1:4d}/{n_eval}] {game_id}  '
                  f'MC: {mc_probs[j]:.3f}  Log5: {log5_probs[j]:.3f}  '
                  f'Actual: {actual:.0f}  '
                  f'Avg score: {mc_avg_hs[j]:.1f}-{mc_avg_as[j]:.1f}  '
                  f'({per_game:.1f}s/game, ETA {eta/60:.0f}min)')
    else:
        # Multi-threaded execution
        # Prepare arguments for worker threads
        thread_args = []
        for j, gi in enumerate(eval_idx):
            # Use pre-computed prefix state
            init_state = prefix_states[j]
            args_tuple = (j, gi, model, test_gc_features, ctx_mean, ctx_std, args.n_sims,
                         device, idx_to_ev, args.temperature, args.forward_chunk, test_game_ids,
                         progress_counter, progress_lock, n_eval, t0, actuals, log5_probs,
                         init_state)
            thread_args.append(args_tuple)

        # Execute simulations in parallel
        with ThreadPoolExecutor(max_workers=args.n_threads) as executor:
            # Submit all tasks
            futures = [executor.submit(simulate_single_game, args_tuple) 
                      for args_tuple in thread_args]
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(futures):
                j, mc_prob, mc_avg_h, mc_avg_a = future.result()
                mc_probs[j] = mc_prob
                mc_avg_hs[j] = mc_avg_h
                mc_avg_as[j] = mc_avg_a

    total_time = time.time() - t0
    print(f'\nSimulation complete: {total_time:.0f}s total, {total_time/n_eval:.1f}s per game')

    # ── Report ──
    print(f'\n{"="*60}')
    print(f'  Results  ({n_eval} games, {args.n_sims} sims each)')
    print(f'{"="*60}')

    hist_probs = np.full(n_eval, hist_rate)

    methods = {
        'Historical avg': hist_probs,
        'Log5':           log5_probs,
    }
    if have_mlp:
        methods['MLP classifier'] = mlp_probs
    sim_method_label = 'MCMC (MH)' if use_mcmc else 'MC simulation'
    methods[sim_method_label] = mc_probs

    print(f'\n{"Method":<20} {"Brier↓":>8} {"LogLoss↓":>10} {"Accuracy":>10}')
    print('-' * 52)
    for name, probs in methods.items():
        bs  = brier_score(probs, actuals)
        ll  = log_loss(probs, actuals)
        acc = accuracy(probs, actuals)
        print(f'{name:<20} {bs:>8.4f} {ll:>10.4f} {acc:>10.2%}')

    # Calibration for simulation method
    print(f'\nCalibration ({sim_method_label}):')
    cal = calibration_table(mc_probs, actuals, n_bins=5)
    print(cal.to_string(index=False))

    if use_mcmc:
        print(f'\nMCMC Chain Diagnostics:')
        print(f'  Mean acceptance rate: {mcmc_acceptance_rates.mean():.3f}')
        print(f'  Min / Max acceptance: {mcmc_acceptance_rates.min():.3f} / {mcmc_acceptance_rates.max():.3f}')
        print(f'  (λ={args.lambda_cal}, steps={args.mcmc_steps}, burn-in={args.mcmc_burnin})')
    else:
        # Score distribution (only meaningful for batched MC)
        print(f'\nMC Score Distributions:')
        print(f'  Avg home score: {mc_avg_hs.mean():.2f} ± {mc_avg_hs.std():.2f}')
        print(f'  Avg away score: {mc_avg_as.mean():.2f} ± {mc_avg_as.std():.2f}')

    # Save results
    prefix_home = np.array([ps['home_score'] if ps else 0 for ps in prefix_states])
    prefix_away = np.array([ps['away_score'] if ps else 0 for ps in prefix_states])

    results_df = pd.DataFrame({
        'game_idx': eval_idx,
        'game_id': [test_game_ids[i] if test_game_ids is not None else i for i in eval_idx],
        'actual': actuals,
        'mc_prob': mc_probs,
        'log5_prob': log5_probs,
        'mlp_prob': mlp_probs if have_mlp else hist_rate,
        'hist_prob': hist_rate,
        'mc_avg_home': mc_avg_hs,
        'mc_avg_away': mc_avg_as,
        'true_home_score': true_home_scores.astype(int),
        'true_away_score': true_away_scores.astype(int),
        'prefix_home_score': prefix_home.astype(int),
        'prefix_away_score': prefix_away.astype(int),
    })
    prefix_tag = f'_frominn{args.prefix_inning}' if args.prefix_inning > 1 else ''
    out_path = f'win_prob_results_{n_eval}games_{args.n_sims}sims{prefix_tag}.csv'
    results_df.to_csv(out_path, index=False)
    print(f'\nResults saved to {out_path}')


if __name__ == '__main__':
    main()
