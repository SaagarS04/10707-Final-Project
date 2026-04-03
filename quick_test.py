"""
Quick end-to-end test of the MCMC sampler.

Pulls ~1 month of Statcast data, trains a small model for a few epochs,
then runs the MH chain on a handful of games and reports acceptance rate,
win probability estimates, and calibration against actual outcomes.

Usage:
    python quick_test.py                        # defaults: April 2024 train, May 2024 test
    python quick_test.py --train_start 2024-04-01 --train_end 2024-04-30
    python quick_test.py --epochs 5 --n_games 5 --mcmc_steps 200 --lambda_cal 0.0
"""

import argparse
import numpy as np
import torch

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
from mcmc_simulator import MHGameSampler


def parse_args():
    p = argparse.ArgumentParser(description='Quick MCMC end-to-end test')
    p.add_argument('--train_start', default='2024-04-01')
    p.add_argument('--train_end',   default='2024-04-30')
    p.add_argument('--test_start',  default='2024-05-01')
    p.add_argument('--test_end',    default='2024-05-07')
    p.add_argument('--epochs',      type=int,   default=3,    help='Training epochs (keep low for speed)')
    p.add_argument('--d_model',     type=int,   default=64,   help='Transformer hidden dim')
    p.add_argument('--n_layers',    type=int,   default=2,    help='Transformer layers')
    p.add_argument('--n_heads',     type=int,   default=4,    help='Attention heads')
    p.add_argument('--n_games',     type=int,   default=5,    help='Number of test games to evaluate')
    p.add_argument('--mcmc_steps',  type=int,   default=200,  help='Post-burn-in MH steps per game')
    p.add_argument('--mcmc_burnin', type=int,   default=50,   help='MH burn-in steps to discard')
    p.add_argument('--lambda_cal',  type=float, default=0.5,  help='RE24 calibration weight (0=MC, 1=full)')
    p.add_argument('--temperature', type=float, default=2.0,  help='Sampling temperature')
    p.add_argument('--seed',        type=int,   default=42)
    return p.parse_args()


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f'Device: {device}')

    # ── Pull data ──────────────────────────────────────────────────────────────
    print(f'\nPulling training data ({args.train_start} → {args.train_end})...')
    train_data = get_transformed_data(args.train_start, args.train_end, weather=False)

    print(f'Pulling test data ({args.test_start} → {args.test_end})...')
    test_data = get_transformed_data(args.test_start, args.test_end, weather=False)

    train_pitch      = train_data['pitch']
    train_pc         = train_data['pitch_context']
    train_pr         = train_data['pitch_result']
    train_ab         = train_data['at_bat_target']
    train_gc         = train_data['game_context']
    train_gt         = train_data['Game_target']
    re24_table       = train_data['re24_table']

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

    # Game context normalization
    gc_num = train_gc.select_dtypes(include=[float, int]).drop(columns=['game_date'], errors='ignore')
    ctx_columns = gc_num.columns.tolist()
    ctx_mean = gc_num.mean().values.astype(np.float32)
    ctx_std  = gc_num.std().values.astype(np.float32)
    ctx_std[ctx_std < 1e-8] = 1.0
    context_dim = len(ctx_columns)

    # Continuous pitch normalization
    cont_vals = train_pitch[CONTINUOUS_PITCH_COLS].values.astype(np.float32)
    cont_vals = np.nan_to_num(cont_vals, nan=0.0)
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
    print(f'\nTraining model ({args.epochs} epochs, d_model={args.d_model}, layers={args.n_layers})...')
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
    test_game_ids = list(test_gc.index[:args.n_games])
    actuals = test_gt.values[:args.n_games].astype(float)

    print(f'\nRunning MCMC on {len(test_game_ids)} games '
          f'(λ={args.lambda_cal}, steps={args.mcmc_steps}, burn-in={args.mcmc_burnin})...')

    mc_probs          = np.zeros(len(test_game_ids))
    acceptance_rates  = np.zeros(len(test_game_ids))

    for j, game_id in enumerate(test_game_ids):
        # Normalize this game's context
        gc_row = test_gc.loc[game_id, ctx_columns].values.astype(np.float32)
        ctx_normed = (gc_row - ctx_mean) / ctx_std
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
        result = sampler.run_chain(n_steps=args.mcmc_steps, burn_in=args.mcmc_burnin)

        mc_probs[j]         = result['win_probability']
        acceptance_rates[j] = result['acceptance_rate']

        print(f'  [{j+1}/{len(test_game_ids)}] {game_id}  '
              f'P(home win)={mc_probs[j]:.3f}  '
              f'Actual={actuals[j]:.0f}  '
              f'AcceptRate={acceptance_rates[j]:.2f}')

    # ── Summary ────────────────────────────────────────────────────────────────
    brier = np.mean((mc_probs - actuals) ** 2)
    eps = 1e-7
    logloss = -np.mean(actuals * np.log(mc_probs + eps) + (1 - actuals) * np.log(1 - mc_probs + eps))
    accuracy = np.mean((mc_probs > 0.5) == actuals)

    print(f'\n── Results ──────────────────────────────────')
    print(f'  Brier score : {brier:.4f}')
    print(f'  Log loss    : {logloss:.4f}')
    print(f'  Accuracy    : {accuracy:.2%}')
    print(f'  Mean accept : {acceptance_rates.mean():.3f}')
    if args.lambda_cal == 0.0:
        print('  (λ=0: acceptance rate should be ~1.0, equivalent to plain MC)')

    # Print RE24 table for inspection
    print(f'\n── Empirical RE24 table (from {args.train_start}→{args.train_end}) ──')
    print(f'  {"State":<30} {"Exp Runs":>10}')
    for outs in range(3):
        for b1 in (False, True):
            for b2 in (False, True):
                for b3 in (False, True):
                    key = (b1, b2, b3, outs)
                    state = f'1B={int(b1)} 2B={int(b2)} 3B={int(b3)} Outs={outs}'
                    print(f'  {state:<30} {re24_table[key]:>10.3f}')


if __name__ == '__main__':
    main()
