"""
Quick end-to-end test of the MCMC sampler with visualizations.

Pulls a short date range of Statcast data, trains a small model for a few
epochs, runs the MH chain on a handful of games, and produces four plots:

  1. Win Probability Over Game  — P(home win) at each inning boundary,
                                   estimated from the chain's score trajectories.
  2. MCMC Chain Convergence     — running mean of P(home win) over chain steps,
                                   with burn-in region shaded.
  3. Score Progression          — mean ± 1σ score differential by inning across
                                   all chain trajectories for one game.
  4. RE24 Heatmap               — empirical expected-runs table as a 3×8 grid.

Usage:
    python quick_test.py                        # defaults
    python quick_test.py --n_games 2 --mcmc_steps 50 --mcmc_burnin 20 --epochs 1
    python quick_test.py --lambda_cal 0.0       # sanity check: acceptance rate ≈ 1.0
    PYTORCH_ENABLE_MPS_FALLBACK=1 python quick_test.py   # MPS with CPU fallback
"""

import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')   # non-interactive backend; swap to 'TkAgg' if you want a window
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from pitch_data import get_transformed_data, compute_re24_from_pitch_context
from pitch_sequence_predictor import (
    PitchSequenceTransfusion,
    AtBatSequenceDataset,
    GameSimulator,
    collate_at_bats,
    train_model,
    build_vocab_maps,
    CONTINUOUS_PITCH_COLS,
)
from mcmc_simulator import MHGameSampler, GameTrajectory


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description='Quick MCMC end-to-end test')
    p.add_argument('--train_start', default='2024-04-01')
    p.add_argument('--train_end',   default='2024-04-30')
    p.add_argument('--test_start',  default='2024-05-01')
    p.add_argument('--test_end',    default='2024-05-07')
    p.add_argument('--epochs',      type=int,   default=3)
    p.add_argument('--d_model',     type=int,   default=64)
    p.add_argument('--n_layers',    type=int,   default=2)
    p.add_argument('--n_heads',     type=int,   default=4)
    p.add_argument('--n_games',     type=int,   default=5)
    p.add_argument('--mcmc_steps',  type=int,   default=200)
    p.add_argument('--mcmc_burnin', type=int,   default=50)
    p.add_argument('--lambda_cal',  type=float, default=0.5)
    p.add_argument('--temperature', type=float, default=2.0)
    p.add_argument('--seed',        type=int,   default=42)
    p.add_argument('--plot',        action='store_true', default=True,
                   help='Save diagnostic plots to mcmc_plots.png')
    p.add_argument('--no_plot',     action='store_true',
                   help='Disable plotting')
    return p.parse_args()


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

def score_diff_by_inning(trajectory: GameTrajectory) -> dict:
    """Return home_score - away_score after each completed inning (1-indexed)."""
    home, away = 0, 0
    scores = {}
    for hi in trajectory.half_innings:
        if hi.is_top:
            away += hi.runs
        else:
            home += hi.runs
            scores[hi.inning] = home - away
    return scores


def win_prob_by_inning(trajectories: list) -> tuple:
    """
    Compute mean P(home win) and score differential statistics at each inning
    boundary across the chain's post-burn-in trajectories.

    Returns:
        innings:    sorted list of inning numbers present in the data.
        wp_mean:    mean P(home win) at each inning boundary.
        diff_mean:  mean score differential at each inning boundary.
        diff_std:   std of score differential at each inning boundary.
    """
    # For each trajectory, record (score_diff, home_wins) at each inning end
    from collections import defaultdict
    diffs_by_inning = defaultdict(list)
    wins_by_inning  = defaultdict(list)

    for traj in trajectories:
        sd = score_diff_by_inning(traj)
        for inn, diff in sd.items():
            diffs_by_inning[inn].append(diff)
            wins_by_inning[inn].append(float(diff > 0))

    innings   = sorted(diffs_by_inning.keys())
    wp_mean   = [np.mean(wins_by_inning[i])  for i in innings]
    diff_mean = [np.mean(diffs_by_inning[i]) for i in innings]
    diff_std  = [np.std(diffs_by_inning[i])  for i in innings]

    return innings, wp_mean, diff_mean, diff_std


def plot_all(all_results: list, re24_table: dict, args, save_path='mcmc_plots.png'):
    """
    Produce a 2×2 figure:
      [0,0] Win probability over game (inning-by-inning) for each evaluated game
      [0,1] MCMC chain convergence (running mean of P(home win) over steps)
      [1,0] Score progression for first game (mean ± 1σ score diff by inning)
      [1,1] RE24 heatmap
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f'MCMC Diagnostics  |  λ={args.lambda_cal}  steps={args.mcmc_steps}  '
        f'burn-in={args.mcmc_burnin}',
        fontsize=13, fontweight='bold',
    )

    colors = plt.cm.tab10.colors

    # ── [0,0] Win probability over game ──────────────────────────────────────
    ax = axes[0, 0]
    for idx, (game_id, result, actual) in enumerate(all_results):
        innings, wp_mean, _, _ = win_prob_by_inning(result['trajectories'])
        color = colors[idx % len(colors)]
        ax.plot(innings, wp_mean, marker='o', color=color, linewidth=1.8,
                label=f'{game_id[:12]}… (actual={int(actual)})')

    ax.axhline(0.5, color='grey', linestyle='--', linewidth=0.8, alpha=0.6)
    ax.set_xlabel('Inning')
    ax.set_ylabel('P(home win)')
    ax.set_title('Win Probability by Inning')
    ax.set_ylim(-0.05, 1.05)
    ax.set_xticks(range(1, 10))
    ax.legend(fontsize=7, loc='upper right')

    # ── [0,1] Chain convergence ───────────────────────────────────────────────
    ax = axes[0, 1]
    for idx, (game_id, result, actual) in enumerate(all_results):
        win_samples = result['win_samples']
        running_mean = np.cumsum(win_samples) / np.arange(1, len(win_samples) + 1)
        color = colors[idx % len(colors)]
        ax.plot(running_mean, color=color, linewidth=1.5,
                label=f'{game_id[:12]}…')

    ax.axhline(0.5, color='grey', linestyle='--', linewidth=0.8, alpha=0.6)
    ax.set_xlabel('Chain step (post burn-in)')
    ax.set_ylabel('Running P(home win)')
    ax.set_title('MCMC Chain Convergence')
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=7, loc='upper right')

    # ── [1,0] Score progression for first game ───────────────────────────────
    ax = axes[1, 0]
    game_id, result, actual = all_results[0]
    innings, _, diff_mean, diff_std = win_prob_by_inning(result['trajectories'])
    diff_mean = np.array(diff_mean)
    diff_std  = np.array(diff_std)

    ax.plot(innings, diff_mean, marker='o', color=colors[0], linewidth=2, label='Mean diff')
    ax.fill_between(innings,
                    diff_mean - diff_std,
                    diff_mean + diff_std,
                    alpha=0.25, color=colors[0], label='±1σ')
    ax.axhline(0, color='grey', linestyle='--', linewidth=0.8, alpha=0.6)
    ax.set_xlabel('Inning')
    ax.set_ylabel('Score differential (home − away)')
    ax.set_title(f'Score Progression: {game_id[:20]}… (actual={int(actual)})')
    ax.set_xticks(range(1, 10))
    ax.legend(fontsize=8)

    # ── [1,1] RE24 heatmap ────────────────────────────────────────────────────
    ax = axes[1, 1]
    base_states = [
        (False, False, False), (True,  False, False),
        (False, True,  False), (True,  True,  False),
        (False, False, True),  (True,  False, True),
        (False, True,  True),  (True,  True,  True),
    ]
    base_labels = [
        '---', '1--', '-2-', '12-',
        '--3', '1-3', '-23', '123',
    ]
    grid = np.array([
        [re24_table[(b1, b2, b3, outs)] for (b1, b2, b3) in base_states]
        for outs in range(3)
    ])

    im = ax.imshow(grid, cmap='YlOrRd', aspect='auto')
    ax.set_xticks(range(8))
    ax.set_xticklabels(base_labels, fontsize=8)
    ax.set_yticks(range(3))
    ax.set_yticklabels(['0 outs', '1 out', '2 outs'])
    ax.set_title('Empirical RE24 Table (expected runs)')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Annotate each cell with the value
    for r in range(3):
        for c in range(8):
            ax.text(c, r, f'{grid[r, c]:.2f}', ha='center', va='center',
                    fontsize=7, color='black')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f'\nPlots saved to {save_path}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    do_plot = args.plot and not args.no_plot

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = (
        'cuda' if torch.cuda.is_available()
        else 'mps' if torch.backends.mps.is_available()
        else 'cpu'
    )
    print(f'Device: {device}')

    # ── Pull data ──────────────────────────────────────────────────────────────
    print(f'\nPulling training data ({args.train_start} → {args.train_end})...')
    train_data = get_transformed_data(args.train_start, args.train_end, weather=False)

    print(f'Pulling test data ({args.test_start} → {args.test_end})...')
    test_data = get_transformed_data(args.test_start, args.test_end, weather=False)

    train_pitch = train_data['pitch']
    train_pc    = train_data['pitch_context']
    train_pr    = train_data['pitch_result']
    train_ab    = train_data['at_bat_target']
    train_gc    = train_data['game_context']
    re24_table  = train_data['re24_table']

    test_pitch  = test_data['pitch']
    test_pc     = test_data['pitch_context']
    test_pr     = test_data['pitch_result']
    test_ab     = test_data['at_bat_target']
    test_gc     = test_data['game_context']
    test_gt     = test_data['Game_target']

    print(f'  Train: {len(train_gc)} games, {len(train_pc)} pitches')
    print(f'  Test:  {len(test_gc)} games, {len(test_pc)} pitches')

    # ── Vocab and normalization ────────────────────────────────────────────────
    pt_to_idx, pr_to_idx, ev_to_idx, zone_to_idx = build_vocab_maps()

    gc_num      = train_gc.select_dtypes(include=[float, int]).drop(columns=['game_date'], errors='ignore')
    ctx_columns = gc_num.columns.tolist()
    ctx_mean    = gc_num.mean().values.astype(np.float32)
    ctx_std     = gc_num.std().values.astype(np.float32)
    ctx_std[ctx_std < 1e-8] = 1.0
    context_dim = len(ctx_columns)

    cont_vals  = train_pitch[CONTINUOUS_PITCH_COLS].values.astype(np.float32)
    cont_vals  = np.nan_to_num(cont_vals, nan=0.0)
    pitch_mean = cont_vals.mean(axis=0)
    pitch_std  = cont_vals.std(axis=0)
    pitch_std[pitch_std < 1e-8] = 1.0

    # ── Build datasets ─────────────────────────────────────────────────────────
    def make_dataset(pitch_df, pc_df, pr_df, ab_df, gc_df):
        return AtBatSequenceDataset(
            pitch_df=pitch_df,
            pitch_context_df=pc_df,
            pitch_result_df=pr_df,
            at_bat_target_df=ab_df,
            game_context_df=gc_df,
            pt_to_idx=pt_to_idx,
            pr_to_idx=pr_to_idx,
            ev_to_idx=ev_to_idx,
            zone_to_idx=zone_to_idx,
            context_columns=ctx_columns,
            context_mean=ctx_mean,
            context_std=ctx_std,
            pitch_mean=pitch_mean,
            pitch_std=pitch_std,
        )

    train_dataset = make_dataset(train_pitch, train_pc, train_pr, train_ab, train_gc)
    test_dataset  = make_dataset(test_pitch,  test_pc,  test_pr,  test_ab,  test_gc)
    print(f'  Train at-bats: {len(train_dataset)}  |  Test at-bats: {len(test_dataset)}')

    # ── Train model ────────────────────────────────────────────────────────────
    print(f'\nTraining model ({args.epochs} epochs, d_model={args.d_model}, '
          f'layers={args.n_layers}, heads={args.n_heads})...')
    model = PitchSequenceTransfusion(
        context_dim=context_dim,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
    ).to(device)

    model = train_model(
        model=model,
        train_dataset=train_dataset,
        val_dataset=test_dataset,
        epochs=args.epochs,
        batch_size=64,
        lr=1e-3,
    )
    model.eval()

    # ── Run MCMC on test games ─────────────────────────────────────────────────
    n_eval      = min(args.n_games, len(test_gc))
    test_game_ids = list(test_gc.index[:n_eval])
    actuals       = test_gt.values[:n_eval].astype(float)

    print(f'\nRunning MCMC on {n_eval} games '
          f'(λ={args.lambda_cal}, steps={args.mcmc_steps}, burn-in={args.mcmc_burnin})...')

    all_results = []   # list of (game_id, result_dict, actual)
    mc_probs    = np.zeros(n_eval)

    for j, game_id in enumerate(test_game_ids):
        gc_row     = test_gc.loc[game_id, ctx_columns].values.astype(np.float32)
        ctx_normed = np.nan_to_num((gc_row - ctx_mean) / ctx_std, nan=0.0)

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
        result = sampler.run_chain(
            n_steps=args.mcmc_steps,
            burn_in=args.mcmc_burnin,
            return_trajectories=do_plot,
        )

        mc_probs[j] = result['win_probability']
        all_results.append((game_id, result, actuals[j]))

        print(f'  [{j+1}/{n_eval}] {game_id}  '
              f'P(home win)={mc_probs[j]:.3f}  '
              f'Actual={actuals[j]:.0f}  '
              f'AcceptRate={result["acceptance_rate"]:.2f}')

    # ── Summary metrics ────────────────────────────────────────────────────────
    brier    = np.mean((mc_probs - actuals) ** 2)
    eps      = 1e-7
    logloss  = -np.mean(actuals * np.log(mc_probs + eps) + (1 - actuals) * np.log(1 - mc_probs + eps))
    accuracy = np.mean((mc_probs > 0.5) == actuals)
    mean_acc = np.mean([r['acceptance_rate'] for _, r, _ in all_results])

    print(f'\n── Results ──────────────────────────────────────────')
    print(f'  Brier score      : {brier:.4f}')
    print(f'  Log loss         : {logloss:.4f}')
    print(f'  Accuracy         : {accuracy:.2%}')
    print(f'  Mean accept rate : {mean_acc:.3f}')
    if args.lambda_cal == 0.0:
        print('  (λ=0: acceptance rate should be ~1.0 — equivalent to plain MC)')

    # ── RE24 table ─────────────────────────────────────────────────────────────
    print(f'\n── Empirical RE24 table ({args.train_start} → {args.train_end}) ──')
    print(f'  {"State":<30} {"Exp Runs":>10}')
    for outs in range(3):
        for b1 in (False, True):
            for b2 in (False, True):
                for b3 in (False, True):
                    key   = (b1, b2, b3, outs)
                    state = f'1B={int(b1)} 2B={int(b2)} 3B={int(b3)} Outs={outs}'
                    print(f'  {state:<30} {re24_table[key]:>10.3f}')

    # ── Plots ──────────────────────────────────────────────────────────────────
    if do_plot:
        plot_all(all_results, re24_table, args, save_path='mcmc_plots.png')


if __name__ == '__main__':
    main()
