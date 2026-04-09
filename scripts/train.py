"""
Train PitchSequenceTransfusion and save the checkpoint.

Thin entry point — all logic lives in model/. This script:
  1. Loads model-ready tables from data/processed/
  2. Builds AtBatSequenceDataset for train and val splits
  3. Calls model.train.train()
  4. Saves the checkpoint

Usage:
    python scripts/train.py \\
        --data   data/processed \\
        --out    checkpoints/model.pt \\
        --epochs 30 \\
        --batch  128 \\
        --lr     3e-4 \\
        --lambda-cont 5.0 \\
        --d-model 256 \\
        --n-heads 8 \\
        --n-layers 6 \\
        --max-seq 256
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.tables import load_all
from model.architecture import PitchSequenceTransfusion
from model.dataset import AtBatSequenceDataset, collate_at_bats
from model.train import train
from model.vocab import build_vocab_maps, CONTINUOUS_PITCH_COLS


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train PitchSequenceTransfusion.")
    p.add_argument("--data",        default="data/processed")
    p.add_argument("--out",         default="checkpoints/model.pt")
    p.add_argument("--epochs",      type=int,   default=30)
    p.add_argument("--batch",       type=int,   default=128)
    p.add_argument("--lr",          type=float, default=3e-4)
    p.add_argument("--lambda-cont", type=float, default=5.0, dest="lambda_cont")
    p.add_argument("--d-model",     type=int,   default=256, dest="d_model")
    p.add_argument("--n-heads",     type=int,   default=8,   dest="n_heads")
    p.add_argument("--n-layers",    type=int,   default=6,   dest="n_layers")
    p.add_argument("--max-seq",     type=int,   default=256, dest="max_seq")
    return p.parse_args()


def _build_dataset(
    pitch_seqs: pd.DataFrame,
    pregame: pd.DataFrame,
    split: str,
    pt_to_idx, pr_to_idx, ev_to_idx, zone_to_idx,
    ctx_columns, ctx_mean, ctx_std,
    pitch_mean, pitch_std,
    max_seq: int,
) -> AtBatSequenceDataset:
    """Slice pitch_seqs and pregame to a single split and build the dataset."""
    game_pks = set(pregame[pregame["split"] == split].index)
    ps = pitch_seqs[pitch_seqs["game_pk"].isin(game_pks)].copy()

    # Rename columns to match original CSV format expected by AtBatSequenceDataset.
    pitch_df = ps[["game_pk", "at_bat_number", "pitch_number",
                   "pitch_type", "zone"] + CONTINUOUS_PITCH_COLS].copy()
    pitch_df = pitch_df.rename(columns={
        "game_pk": "game_id",
        "at_bat_number": "at_bat_id",
        "pitch_number": "pitch_id",
    })

    pitch_context_df = ps[["game_pk", "at_bat_number", "pitch_number",
                            "inning", "inning_topbot", "outs_when_up",
                            "home_score", "away_score", "bat_score_diff",
                            "balls", "strikes", "on_1b", "on_2b", "on_3b"]].copy()
    # One-hot encode inning_topbot.
    pitch_context_df["inning_topbot_Top"] = (pitch_context_df["inning_topbot"] == "Top").astype(float)
    pitch_context_df = pitch_context_df.drop(columns=["inning_topbot"])
    pitch_context_df = pitch_context_df.rename(columns={
        "game_pk": "game_id",
        "at_bat_number": "at_bat_id",
        "pitch_number": "pitch_id",
    })

    pitch_result_df = ps[["description"]].copy()
    at_bat_target_df = ps[["events"]].copy()

    # Game context: pregame features for this split.
    gc = pregame[pregame["split"] == split].reset_index()
    gc = gc.rename(columns={"game_pk": "game_id"})
    gc_features = gc.drop(columns=["split", "home_win", "home_win_exp",
                                    "home_score", "away_score"], errors="ignore")
    gc_features = pd.get_dummies(gc_features, drop_first=True).astype(float)
    gc_features = gc_features.reindex(columns=ctx_columns, fill_value=0.0)
    gc_features["game_id"] = gc["game_id"].values

    return AtBatSequenceDataset(
        pitch_df=pitch_df,
        pitch_context_df=pitch_context_df,
        pitch_result_df=pitch_result_df,
        at_bat_target_df=at_bat_target_df,
        game_context_df=gc_features,
        pt_to_idx=pt_to_idx, pr_to_idx=pr_to_idx,
        ev_to_idx=ev_to_idx, zone_to_idx=zone_to_idx,
        context_columns=ctx_columns,
        context_mean=ctx_mean, context_std=ctx_std,
        pitch_mean=pitch_mean, pitch_std=pitch_std,
        max_pitches=max_seq,
    )


def main() -> None:
    args = parse_args()

    print("=== Loading data ===")
    tables = load_all(args.data)
    pregame    = tables["pregame_context"]
    pitch_seqs = tables["pitch_sequences"]

    pt_to_idx, pr_to_idx, ev_to_idx, zone_to_idx = build_vocab_maps()

    # Build context column set and normalization stats from the training split only.
    train_pregame = pregame[pregame["split"] == "train"].copy()
    train_pregame = train_pregame.drop(columns=["split", "home_win", "home_win_exp",
                                                 "home_score", "away_score",
                                                 "game_date"], errors="ignore")
    train_features = pd.get_dummies(train_pregame, drop_first=True).astype(float)
    ctx_columns = train_features.columns
    ctx_mean = train_features.values.mean(axis=0).astype(np.float32)
    ctx_std  = train_features.values.std(axis=0).astype(np.float32)
    ctx_std[ctx_std < 1e-8] = 1.0

    # Continuous pitch normalization from training data only.
    train_pitches = pitch_seqs[pitch_seqs["game_pk"].isin(
        set(pregame[pregame["split"] == "train"].index)
    )]
    cont_vals = train_pitches[CONTINUOUS_PITCH_COLS].values.astype(np.float32)
    cont_vals = np.nan_to_num(cont_vals, nan=0.0)
    pitch_mean = cont_vals.mean(axis=0)
    pitch_std  = cont_vals.std(axis=0)
    pitch_std[pitch_std < 1e-8] = 1.0

    print("=== Building datasets ===")
    train_ds = _build_dataset(
        pitch_seqs, pregame, "train",
        pt_to_idx, pr_to_idx, ev_to_idx, zone_to_idx,
        ctx_columns, ctx_mean, ctx_std, pitch_mean, pitch_std, args.max_seq,
    )
    val_ds = _build_dataset(
        pitch_seqs, pregame, "val",
        pt_to_idx, pr_to_idx, ev_to_idx, zone_to_idx,
        ctx_columns, ctx_mean, ctx_std, pitch_mean, pitch_std, args.max_seq,
    )
    print(f"  Train: {len(train_ds)} at-bats  |  Val: {len(val_ds)} at-bats")

    print("=== Building model ===")
    context_dim = len(ctx_columns)
    model = PitchSequenceTransfusion(
        context_dim=context_dim,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        max_seq_len=args.max_seq,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}  |  context_dim: {context_dim}")

    print("=== Training ===")
    model, history = train(
        model, train_ds, val_ds,
        epochs=args.epochs,
        batch_size=args.batch,
        lr=args.lr,
        lambda_continuous=args.lambda_cont,
        collate_fn=collate_at_bats,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "ctx_columns":  list(ctx_columns),
        "ctx_mean":     ctx_mean,
        "ctx_std":      ctx_std,
        "pitch_mean":   pitch_mean,
        "pitch_std":    pitch_std,
        "d_model":      args.d_model,
        "n_heads":      args.n_heads,
        "n_layers":     args.n_layers,
        "max_seq_len":  args.max_seq,
        "context_dim":  context_dim,
    }, out_path)
    print(f"\nCheckpoint saved → {out_path}")


if __name__ == "__main__":
    main()
