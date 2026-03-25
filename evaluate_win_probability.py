"""
Monte Carlo Win Probability Evaluation
=======================================
Simulates N game paths per test game using the trained Transfusion pitch sequence
model.  Computes empirical P(home_win) = (# home wins) / N and compares against
baselines.

Baselines:
  1. Historical average:  always predict training-set home win rate
  2. Log5 method:         p = (h - h·a) / (h + a - 2·h·a)
  3. Logistic regression  on game-context features

Metrics:  Brier score · Log-loss · Accuracy · Calibration

Usage:
    # Train model first (if no checkpoint):
    python pitch_sequence_predictor.py

    # Run evaluation:
    python evaluate_win_probability.py --n_sims 10000 --n_games 100

    # Or train + evaluate in one go:
    python evaluate_win_probability.py --train --epochs 15 --n_sims 1000 --n_games 50
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

from pitch_sequence_predictor import (
    PitchSequenceTransfusion,
    AtBatSequenceDataset,
    collate_at_bats,
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
):
    """
    Simulate *N* complete baseball games for a single game context.

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
                c_mu  = out['continuous_mean'][bi, lp]
                c_lv  = out['continuous_logvar'][bi, lp]

                sp = torch.multinomial(F.softmax(pt_lo, -1), 1).squeeze(-1).cpu().numpy()
                sz = torch.multinomial(F.softmax(z_lo,  -1), 1).squeeze(-1).cpu().numpy()
                sr = torch.multinomial(F.softmax(pr_lo, -1), 1).squeeze(-1).cpu().numpy()
                sc = (c_mu + torch.exp(0.5 * c_lv) * torch.randn_like(c_mu)).cpu().numpy()

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


def logistic_baseline(train_gc, train_gt, test_gc, ctx_columns, ctx_mean, ctx_std):
    """Train a logistic regression on game-context features → home_win."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    # Prepare train features
    X_train = train_gc[ctx_columns].values.astype(np.float64)
    X_train = (X_train - ctx_mean) / ctx_std
    y_train = train_gt['home_win_exp'].values

    # Prepare test features
    X_test = test_gc.reindex(columns=ctx_columns, fill_value=0).values.astype(np.float64)
    X_test = (X_test - ctx_mean) / ctx_std

    # Handle NaN
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_test  = np.nan_to_num(X_test,  nan=0.0, posinf=0.0, neginf=0.0)

    model = LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs')
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
    p.add_argument('--epochs', type=int, default=15)
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--n_sims', type=int, default=10000, help='Simulations per game')
    p.add_argument('--n_games', type=int, default=0,
                   help='Number of test games to evaluate (0 = all)')
    p.add_argument('--temperature', type=float, default=0.9)
    p.add_argument('--forward_chunk', type=int, default=2048,
                   help='Max batch size for model forward pass')
    p.add_argument('--d_model', type=int, default=256, help='Transformer hidden dim')
    p.add_argument('--n_layers', type=int, default=6, help='Transformer layers')
    p.add_argument('--n_heads', type=int, default=8, help='Attention heads')
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available()
                          else 'mps' if torch.backends.mps.is_available()
                          else 'cpu')
    print(f'Device: {device}')

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

    # Also need train features for logistic regression (before normalization)
    train_gc_features = train_gc_proc.drop(columns=['game_id'], errors='ignore')

    # Continuous pitch normalisation
    print('Computing pitch normalisation stats...')
    cont_vals = train_pitch[CONTINUOUS_PITCH_COLS].values.astype(np.float32)
    cont_vals = np.nan_to_num(cont_vals, nan=0.0)
    pitch_mean = cont_vals.mean(axis=0)
    pitch_std  = cont_vals.std(axis=0)
    pitch_std[pitch_std < 1e-8] = 1.0

    context_dim = len(ctx_columns)

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

        train_dataset = AtBatSequenceDataset(
            train_pitch, train_pc, train_pr, train_ab, train_gc_with_id,
            pt_to_idx, pr_to_idx, ev_to_idx, zone_to_idx,
            ctx_columns, ctx_mean, ctx_std, pitch_mean, pitch_std,
        )
        test_dataset = AtBatSequenceDataset(
            test_pitch, test_pc, test_pr, test_ab, test_gc_with_id,
            pt_to_idx, pr_to_idx, ev_to_idx, zone_to_idx,
            ctx_columns, ctx_mean, ctx_std, pitch_mean, pitch_std,
        )
        print(f'  Train: {len(train_dataset)} at-bats, Test: {len(test_dataset)} at-bats')

        model = PitchSequenceTransfusion(
            context_dim=context_dim, d_model=args.d_model, n_heads=args.n_heads,
            n_layers=args.n_layers, dropout=0.1, max_seq_len=256,
        )
        print(f'  Parameters: {sum(p.numel() for p in model.parameters()):,}')

        model = train_model(
            model, train_dataset, test_dataset,
            epochs=args.epochs, batch_size=args.batch_size,
            lr=args.lr, lambda_continuous=5.0,
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
            n_layers=n_layers, dropout=0.1, max_seq_len=256,
        )
        model.load_state_dict(ckpt['model_state_dict'])

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

    # Logistic regression
    try:
        lr_probs_all, lr_model = logistic_baseline(
            train_gc_features, train_gt, test_gc_features,
            ctx_columns, ctx_mean, ctx_std,
        )
        lr_probs = lr_probs_all[eval_idx]
        have_lr = True
    except Exception as e:
        print(f'  Logistic regression failed: {e}')
        lr_probs = np.full(n_eval, hist_rate)
        have_lr = False

    # ── MC simulation ──
    print('\nRunning Monte Carlo simulations...')
    mc_probs  = np.zeros(n_eval)
    mc_avg_hs = np.zeros(n_eval)
    mc_avg_as = np.zeros(n_eval)

    t0 = time.time()
    for j, gi in enumerate(eval_idx):
        # Prepare context
        ctx_row = test_gc_features.iloc[gi].values.astype(np.float32)
        ctx_normed = (ctx_row - ctx_mean) / ctx_std
        ctx_normed = np.nan_to_num(ctx_normed, nan=0.0)

        result = simulate_games_batched(
            model, ctx_normed, args.n_sims, device,
            idx_to_ev=idx_to_ev,
            temperature=args.temperature,
            forward_chunk=args.forward_chunk,
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
    if have_lr:
        methods['Logistic reg'] = lr_probs
    methods['MC simulation'] = mc_probs

    print(f'\n{"Method":<20} {"Brier↓":>8} {"LogLoss↓":>10} {"Accuracy":>10}')
    print('-' * 52)
    for name, probs in methods.items():
        bs  = brier_score(probs, actuals)
        ll  = log_loss(probs, actuals)
        acc = accuracy(probs, actuals)
        print(f'{name:<20} {bs:>8.4f} {ll:>10.4f} {acc:>10.2%}')

    # Calibration for MC simulation
    print(f'\nCalibration (MC simulation):')
    cal = calibration_table(mc_probs, actuals, n_bins=5)
    print(cal.to_string(index=False))

    # Score distribution
    print(f'\nMC Score Distributions:')
    print(f'  Avg home score: {mc_avg_hs.mean():.2f} ± {mc_avg_hs.std():.2f}')
    print(f'  Avg away score: {mc_avg_as.mean():.2f} ± {mc_avg_as.std():.2f}')

    # Save results
    results_df = pd.DataFrame({
        'game_idx': eval_idx,
        'game_id': [test_game_ids[i] if test_game_ids is not None else i for i in eval_idx],
        'actual': actuals,
        'mc_prob': mc_probs,
        'log5_prob': log5_probs,
        'lr_prob': lr_probs if have_lr else hist_rate,
        'hist_prob': hist_rate,
        'mc_avg_home': mc_avg_hs,
        'mc_avg_away': mc_avg_as,
    })
    out_path = f'win_prob_results_{n_eval}games_{args.n_sims}sims.csv'
    results_df.to_csv(out_path, index=False)
    print(f'\nResults saved to {out_path}')


if __name__ == '__main__':
    main()
