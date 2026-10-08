"""Training entry point for PhyGlobT.

Faithful port of the paper training loop (``Phy_unbalance`` script used to
produce the released flagship checkpoint): PhysiTrackModel + PhysiTrackLoss
(focal association loss + physics consistency), threshold-search evaluation on
the validation set, and ``best_model.pt`` saved with the same top-level keys
as the released checkpoint (``model_state_dict``, ``best_threshold``, ``f1``,
``level_stats``, ...).

Usage::

    python -m phyglobt.train \\
        --data_dir ./data/scenes \\
        --save_dir ./checkpoints \\
        --max_trajectories 256 --batch_size 1 --epochs 200

Train/val split (80/20 by default) is done internally per
``CrossSensorTrajectoryDataset``.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .dataset import create_data_loaders
from .model import PhysiTrackLoss, PhysiTrackModel


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_metrics_at_threshold(P, Y, mask_A, mask_B, thresh):
    """Per-pair metrics over the masked (A x B) association submatrix."""
    P_bin = (P > thresh).float()
    valid = mask_A.unsqueeze(-1) * mask_B.unsqueeze(1)
    P_bin = P_bin * valid
    Y = Y * valid
    tp = ((P_bin == 1) & (Y == 1)).sum().item()
    fp = ((P_bin == 1) & (Y == 0)).sum().item()
    fn = ((P_bin == 0) & (Y == 1)).sum().item()
    tn = ((P_bin == 0) & (Y == 0)).sum().item()
    acc = (tp + tn) / (tp + fp + fn + tn + 1e-8)
    prec = tp / (tp + fp + 1e-8)
    rec = tp / (tp + fn + 1e-8)
    f1 = 2 * prec * rec / (prec + rec + 1e-8)
    return acc, prec, rec, f1


def find_best_threshold(P, Y, mask_A, mask_B, thresh_list=None):
    """Sweep the decision threshold on the val set and return (best, best_f1)."""
    if thresh_list is None:
        thresh_list = np.linspace(0.01, 0.99, 50)
    best_f1 = -1.0
    best_thresh = 0.5
    for thresh in thresh_list:
        _, _, _, f1 = compute_metrics_at_threshold(P, Y, mask_A, mask_B, thresh)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
    return best_thresh, best_f1


def compute_confusion_matrix(P, Y, mask_A, mask_B, thresh):
    P_bin = (P > thresh).float()
    valid = mask_A.unsqueeze(-1) * mask_B.unsqueeze(1)
    P_bin = P_bin * valid
    Y = Y * valid
    tp = ((P_bin == 1) & (Y == 1)).sum().item()
    fp = ((P_bin == 1) & (Y == 0)).sum().item()
    fn = ((P_bin == 0) & (Y == 1)).sum().item()
    tn = ((P_bin == 0) & (Y == 0)).sum().item()
    return tp, fp, fn, tn


def compute_metrics_from_confusion(tp, fp, fn, tn):
    prec = tp / (tp + fp + 1e-8)
    rec = tp / (tp + fn + 1e-8)
    f1 = 2 * prec * rec / (prec + rec + 1e-8)
    acc = (tp + tn) / (tp + fp + fn + tn + 1e-8)
    specificity = tn / (tn + fp + 1e-8)
    return {
        'precision': prec, 'recall': rec, 'f1': f1,
        'accuracy': acc, 'specificity': specificity,
        'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
    }


@torch.no_grad()
def evaluate(model, val_loader, device):
    """Validate: aggregate probs/labels per density level, sweep threshold,
    return (best_thresh, best_f1, level_stats, overall_metrics, avg_val_loss)."""
    model.eval()
    val_loss = 0.0
    all_probs, all_Y, all_maskA, all_maskB, all_levels = [], [], [], [], []
    levels_present = {-1: False, 0: False, 1: False, 2: False}

    for batch in val_loader:
        traj_A, traj_B, relations, weights, ts_A, ts_B, mask_A, mask_B, levels = batch
        traj_A, traj_B = traj_A.to(device), traj_B.to(device)
        relations = relations.to(device)
        ts_A, ts_B = ts_A.to(device), ts_B.to(device)
        mask_A, mask_B = mask_A.to(device), mask_B.to(device)
        if mask_A.sum() == 0 or mask_B.sum() == 0:
            continue
        P, logits, aux = model(traj_A, traj_B, ts_A, ts_B, mask_A, mask_B)
        _, _, _, _, phys_A, phys_B = aux
        loss, _ = PhysiTrackLoss()(logits, relations, mask_A, mask_B, phys_A, phys_B)
        if not torch.isnan(loss):
            val_loss += loss.item()
        probs = torch.sigmoid(logits)
        bs = probs.shape[0]
        for b in range(bs):
            level_id = levels[b].item()
            all_probs.append(probs[b:b + 1])
            all_Y.append(relations[b:b + 1])
            all_maskA.append(mask_A[b:b + 1])
            all_maskB.append(mask_B[b:b + 1])
            all_levels.append(level_id)
            levels_present[level_id] = True

    if not all_probs:
        raise RuntimeError("No valid validation batches.")

    probs_cat = torch.cat(all_probs, dim=0)
    Y_cat = torch.cat(all_Y, dim=0)
    maskA_cat = torch.cat(all_maskA, dim=0)
    maskB_cat = torch.cat(all_maskB, dim=0)
    avg_val_loss = val_loss / len(all_probs)

    best_thresh, best_f1 = find_best_threshold(probs_cat, Y_cat, maskA_cat, maskB_cat)

    level_stats = {l: {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0} for l in (-1, 0, 1, 2)}
    for probs_b, Y_b, mA_b, mB_b, level_id in zip(all_probs, all_Y, all_maskA, all_maskB, all_levels):
        tp, fp, fn, tn = compute_confusion_matrix(probs_b, Y_b, mA_b, mB_b, best_thresh)
        level_stats[level_id]['tp'] += tp
        level_stats[level_id]['fp'] += fp
        level_stats[level_id]['fn'] += fn
        level_stats[level_id]['tn'] += tn

    total_tp = sum(level_stats[l]['tp'] for l in (0, 1, 2))
    total_fp = sum(level_stats[l]['fp'] for l in (0, 1, 2))
    total_fn = sum(level_stats[l]['fn'] for l in (0, 1, 2))
    total_tn = sum(level_stats[l]['tn'] for l in (0, 1, 2))
    overall_metrics = compute_metrics_from_confusion(total_tp, total_fp, total_fn, total_tn)
    return best_thresh, best_f1, level_stats, overall_metrics, avg_val_loss


def train_one_run(save_dir, train_loader, val_loader, model, criterion,
                  optimizer, scheduler, num_epochs, patience, device, resume_path=None):
    os.makedirs(save_dir, exist_ok=True)
    best_val_f1 = -1.0
    start_epoch = 0

    if resume_path is not None and os.path.isfile(resume_path):
        print(f"Resuming training from: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        best_val_f1 = checkpoint.get('f1', -1.0)
        start_epoch = checkpoint.get('epoch', 0) + 1

    for epoch in range(start_epoch, num_epochs):
        model.train()
        train_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            traj_A, traj_B, relations, weights, ts_A, ts_B, mask_A, mask_B, _ = batch
            traj_A, traj_B = traj_A.to(device), traj_B.to(device)
            relations = relations.to(device)
            ts_A, ts_B = ts_A.to(device), ts_B.to(device)
            mask_A, mask_B = mask_A.to(device), mask_B.to(device)
            if mask_A.sum() == 0 or mask_B.sum() == 0:
                continue
            optimizer.zero_grad()
            P, logits, aux = model(traj_A, traj_B, ts_A, ts_B, mask_A, mask_B)
            _, _, _, _, phys_A, phys_B = aux
            loss, loss_parts = criterion(logits, relations, mask_A, mask_B, phys_A, phys_B)
            if torch.isnan(loss):
                print("Warning: NaN loss, skipping batch")
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
            n_batches += 1

        if n_batches == 0:
            print("No valid train batches, skipping epoch")
            continue
        avg_train_loss = train_loss / n_batches

        best_thresh, best_f1, level_stats, overall_metrics, avg_val_loss = evaluate(
            model, val_loader, device
        )
        scheduler.step(avg_val_loss)

        level_names = {0: '50_100', 1: '100_200', 2: '200_plus', -1: 'unknown'}
        print(f"\nEpoch {epoch + 1:03d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        print(f"Overall (thresh={best_thresh:.3f}): Acc={overall_metrics['accuracy']:.4f} "
              f"Prec={overall_metrics['precision']:.4f} Rec={overall_metrics['recall']:.4f} "
              f"F1={overall_metrics['f1']:.4f}")
        for level_id in (0, 1, 2, -1):
            st = level_stats[level_id]
            if st['tp'] + st['fp'] + st['fn'] + st['tn'] > 0:
                m = compute_metrics_from_confusion(st['tp'], st['fp'], st['fn'], st['tn'])
                print(f"  {level_names[level_id]}: Prec={m['precision']:.4f} Rec={m['recall']:.4f} F1={m['f1']:.4f} "
                      f"(TP={st['tp']}, FP={st['fp']}, FN={st['fn']}, TN={st['tn']})")

        if overall_metrics['f1'] > best_val_f1:
            best_val_f1 = overall_metrics['f1']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
                'best_threshold': best_thresh,
                'accuracy': overall_metrics['accuracy'],
                'precision': overall_metrics['precision'],
                'recall': overall_metrics['recall'],
                'f1': overall_metrics['f1'],
                'specificity': overall_metrics['specificity'],
                'level_stats': level_stats,
            }, os.path.join(save_dir, 'best_model.pt'))
            print(f"  -> Best model saved (F1={best_val_f1:.4f})")

    print(f"\nTraining completed. Best F1: {best_val_f1:.4f}")


def parse_args():
    parser = argparse.ArgumentParser(description="PhyGlobT training")
    parser.add_argument("--data_dir", type=str, default="./data/scenes")
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    parser.add_argument("--max_trajectories", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--lambda_assoc", type=float, default=1.0)
    parser.add_argument("--lambda_phys", type=float, default=0.2)
    parser.add_argument("--neg_pos_ratio", type=int, default=5)
    parser.add_argument("--focal_alpha", type=float, default=0.25)
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    parser.add_argument("--sinkhorn_tau", type=float, default=0.5)
    parser.add_argument("--use_topology", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--resume", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model_config = {
        'input_dim': 5, 'hidden_dim': 128, 'phys_dim': 64, 'lstm_layers': 2, 'dropout': 0.2,
        'use_topology': args.use_topology, 'topo_num_layers': 2, 'topo_nhead': 8,
        'topo_dim_feedforward': 256, 'sinkhorn_iters': 20, 'sinkhorn_tau': args.sinkhorn_tau,
        'sinkhorn_epsilon': 1e-3, 'dustbin_cost': 0.0, 'learn_dustbin_cost': False,
        'auto_dustbin_mass': True, 'dustbin_mass_ratio': 0.2, 'sinkhorn_stop_thresh': 1e-5,
        'use_transformer_encoder': False,
    }

    data_config = {
        'max_trajectories': args.max_trajectories,
        'time_steps': 32,
        'use_interpolation': False,
        'sparsify': True,
        'max_time_gap': 1800,
        'max_distance': 10.0,
        'mask_augment': False,
        'random_seed': args.seed,
        'train_ratio': 0.8,
        'test_ratio': 0.0,
        'batch_size': args.batch_size,
        'num_workers': args.num_workers,
        'pin_memory': True,
    }
    train_loader, val_loader, _, train_dataset, val_dataset, _ = create_data_loaders(
        args.data_dir, data_config
    )
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    if len(train_dataset) == 0:
        print("No training data found; exit.")
        return

    model = PhysiTrackModel(model_config).to(device)
    criterion = PhysiTrackLoss(
        lambda_assoc=args.lambda_assoc, lambda_phys=args.lambda_phys,
        neg_pos_ratio=args.neg_pos_ratio,
        focal_alpha=args.focal_alpha, focal_gamma=args.focal_gamma,
    )
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

    train_one_run(
        save_dir=args.save_dir,
        train_loader=train_loader,
        val_loader=val_loader,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=args.epochs,
        patience=args.patience,
        device=device,
        resume_path=args.resume,
    )


if __name__ == "__main__":
    main()