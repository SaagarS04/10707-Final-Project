"""
Training loop for PitchSequenceTransfusion.

Responsibility: take a model, two datasets, and hyperparameters → return a
trained model. No data loading, no evaluation, no checkpointing strategy beyond
returning the best-val-loss state. All of those concerns belong to scripts/train.py.
"""

from __future__ import annotations

import copy
import os

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from model.dataset import (
    AtBatSequenceDataset,
    collate_at_bats,
    compute_class_weights,
    transfusion_loss,
)


def train(
    model,
    train_dataset: Dataset,
    val_dataset: Dataset,
    epochs: int = 30,
    batch_size: int = 128,
    lr: float = 3e-4,
    lambda_continuous: float = 5.0,
    collate_fn=None,
) -> tuple:
    """Train PitchSequenceTransfusion and return (model, history).

    Defaults to AtBatSequenceDataset collate. Pass collate_games explicitly
    when training on GameSequenceDataset.

    Args:
        model:             PitchSequenceTransfusion instance.
        train_dataset:     Training dataset (canonical: AtBatSequenceDataset).
        val_dataset:       Validation dataset (same type as train).
        epochs:            Number of training epochs.
        batch_size:        Minibatch size.
        lr:                Initial learning rate for AdamW.
        lambda_continuous: Weight on DDPM loss term (λ in Transfusion loss).
        collate_fn:        Collate function. Defaults to collate_at_bats.

    Returns:
        (model, history) where model has the best-val-loss weights loaded and
        history is a dict of per-epoch metric lists.
    """
    if collate_fn is None:
        collate_fn = collate_at_bats

    device = (
        torch.device("cuda") if torch.cuda.is_available()
        else torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cpu")
    )
    model = model.to(device)

    n_workers = min(12, os.cpu_count() or 0)
    pin = device.type == "cuda"
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=n_workers,
        pin_memory=pin, persistent_workers=(n_workers > 0),
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=n_workers,
        pin_memory=pin, persistent_workers=(n_workers > 0),
    )

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01, betas=(0.9, 0.95))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5)

    use_amp = device.type == "cuda"
    scaler  = torch.amp.GradScaler("cuda", enabled=use_amp)

    print("Computing class weights from training data...")
    class_weights = compute_class_weights(train_dataset, device=str(device))

    loss_keys = ["total", "lm_loss", "pt_loss", "zone_loss", "pr_loss", "ev_loss", "ddpm_loss"]
    history: dict = {k: {"train": [], "val": []} for k in loss_keys}
    history["pt_acc"]  = {"train": [], "val": []}
    history["pr_acc"]  = {"train": [], "val": []}
    history["lr"]      = []

    best_val_loss = float("inf")
    best_state    = None

    for epoch in range(epochs):
        # ── Train ──
        model.train()
        tr = {k: 0.0 for k in loss_keys}
        tr_correct_pt = tr_correct_pr = tr_total = n_tr = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [train]"):
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            optimizer.zero_grad()

            with torch.amp.autocast("cuda", enabled=use_amp):
                outputs = model(
                    batch["context"], batch["game_state"],
                    batch["prev_pitch_type"], batch["prev_zone"],
                    batch["prev_pitch_result"], batch["prev_continuous"],
                )
            losses = transfusion_loss(model, outputs, batch, lambda_continuous, class_weights)

            scaler.scale(losses["total"]).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            for k in loss_keys:
                tr[k] += losses[k].item()
            n_tr += 1

            with torch.no_grad():
                m = batch["mask"].bool()
                tr_correct_pt += (outputs["pitch_type_logits"].argmax(-1)[m] == batch["tgt_pitch_type"][m]).sum().item()
                tr_correct_pr += (outputs["pitch_result_logits"].argmax(-1)[m] == batch["tgt_pitch_result"][m]).sum().item()
                tr_total      += m.sum().item()

        # ── Validate ──
        model.eval()
        vl = {k: 0.0 for k in loss_keys}
        vl_correct_pt = vl_correct_pr = vl_total = n_vl = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [val]"):
                batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
                with torch.amp.autocast("cuda", enabled=use_amp):
                    outputs = model(
                        batch["context"], batch["game_state"],
                        batch["prev_pitch_type"], batch["prev_zone"],
                        batch["prev_pitch_result"], batch["prev_continuous"],
                    )
                losses = transfusion_loss(model, outputs, batch, lambda_continuous, class_weights)
                for k in loss_keys:
                    vl[k] += losses[k].item()
                n_vl += 1

                m = batch["mask"].bool()
                vl_correct_pt += (outputs["pitch_type_logits"].argmax(-1)[m] == batch["tgt_pitch_type"][m]).sum().item()
                vl_correct_pr += (outputs["pitch_result_logits"].argmax(-1)[m] == batch["tgt_pitch_result"][m]).sum().item()
                vl_total      += m.sum().item()

        scheduler.step(vl["total"])

        val_total_loss = vl["total"] / max(n_vl, 1)
        marker = ""
        if val_total_loss < best_val_loss:
            best_val_loss = val_total_loss
            best_state    = copy.deepcopy(model.state_dict())
            marker = " *"

        nb = max(n_tr, 1)
        nv = max(n_vl, 1)
        print(
            f"Epoch {epoch+1}/{epochs}{marker}\n"
            f"  Train  total={tr['total']/nb:.4f}  "
            f"pt={tr['pt_loss']/nb:.4f}  zone={tr['zone_loss']/nb:.4f}  "
            f"pr={tr['pr_loss']/nb:.4f}  ev={tr['ev_loss']/nb:.4f}  "
            f"ddpm={tr['ddpm_loss']/nb:.4f}  "
            f"PT={tr_correct_pt/max(tr_total,1):.2%}  PR={tr_correct_pr/max(tr_total,1):.2%}\n"
            f"  Val    total={val_total_loss:.4f}  "
            f"pt={vl['pt_loss']/nv:.4f}  zone={vl['zone_loss']/nv:.4f}  "
            f"pr={vl['pr_loss']/nv:.4f}  ev={vl['ev_loss']/nv:.4f}  "
            f"ddpm={vl['ddpm_loss']/nv:.4f}  "
            f"PT={vl_correct_pt/max(vl_total,1):.2%}  PR={vl_correct_pr/max(vl_total,1):.2%}  "
            f"LR={scheduler.get_last_lr()[0]:.2e}"
        )

        for k in loss_keys:
            history[k]["train"].append(tr[k] / nb)
            history[k]["val"].append(vl[k] / nv)
        history["pt_acc"]["train"].append(tr_correct_pt / max(tr_total, 1))
        history["pt_acc"]["val"].append(vl_correct_pt / max(vl_total, 1))
        history["pr_acc"]["train"].append(tr_correct_pr / max(tr_total, 1))
        history["pr_acc"]["val"].append(vl_correct_pr / max(vl_total, 1))
        history["lr"].append(scheduler.get_last_lr()[0])

    print(f"\nRestoring best model (val loss: {best_val_loss:.4f})")
    model.load_state_dict(best_state)
    return model, history
