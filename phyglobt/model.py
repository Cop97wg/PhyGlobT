"""
PhyGlobT core model — Physics-Informed Global Topology Encoding for
Multi-Radar Track-to-Track Association.

Extracted verbatim from the paper implementation (``Phy_unbalance`` variant,
the one matching the released flagship checkpoint; unbalanced Sinkhorn with a
dustbin row/column so unmatched tracks are allowed).

Components
----------
* Phy-KRE      : ``PhysicalResidualEncoder`` — CV-model kinematics residual features
* BiLSTM       : ``TrajectoryFeatureEncoder`` — deep temporal motion encoding
* Phy-GTE      : ``GlobalTopologyEncoder`` — density-independent Transformer self-attention
* Phy-Sinkhorn : ``UnbalancedSinkhornSolver`` — differentiable balanced/unbalanced assignment
* ``PhysiTrackModel`` ties them together; ``PhysiTrackLoss`` is the training objective.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class PhysicalResidualEncoder(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.proj = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, traj, timestamps):
        B, N, T, C = traj.shape
        if T < 2:
            return torch.zeros(B, N, self.hidden_dim, device=traj.device)

        pos = traj[..., :2]
        sog = traj[..., 2:3]
        cog = traj[..., 3:4]
        cog_rad = cog * torch.pi / 180.0
        vx = sog * torch.cos(cog_rad)
        vy = sog * torch.sin(cog_rad)
        vel = torch.cat([vx, vy], dim=-1)

        dt = timestamps[..., 1:] - timestamps[..., :-1]
        dt_exp = dt.unsqueeze(-1)

        pos_cur = pos[..., :-1, :]
        pos_next = pos[..., 1:, :]
        vel_cur = vel[..., :-1, :]
        pos_pred = pos_cur + vel_cur * dt_exp
        pos_res = pos_next - pos_pred

        vel_next = vel[..., 1:, :]
        vel_res = vel_next - vel_cur

        residual = torch.cat([pos_res, vel_res], dim=-1)
        residual_pooled = residual.mean(dim=2)
        residual_pooled = torch.nan_to_num(residual_pooled, nan=0.0, posinf=1.0, neginf=-1.0)
        residual_pooled = torch.clamp(residual_pooled, -5.0, 5.0)
        feat = self.proj(residual_pooled)
        return feat


class TrajectoryFeatureEncoder(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=128, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                            batch_first=True, dropout=dropout, bidirectional=False)
        self.proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, traj, mask=None):
        B, N, T, D = traj.shape
        traj_flat = traj.view(B * N, T, D)
        traj_flat = torch.nan_to_num(traj_flat, nan=0.0, posinf=10.0, neginf=-10.0)
        traj_flat = torch.clamp(traj_flat, -10.0, 10.0)

        _, (h_n, _) = self.lstm(traj_flat)
        feat = h_n[-1].view(B, N, -1)
        feat = self.proj(feat)
        if mask is not None:
            feat = feat * mask.unsqueeze(-1)
        feat = torch.nan_to_num(feat, nan=0.0, posinf=1.0, neginf=-1.0)
        return feat


class PositionalEncoding(nn.Module):
    """可学习的位置编码（适配 Ours 模型架构）"""
    def __init__(self, d_model, max_len=100):
        super().__init__()
        self.pe = nn.Parameter(torch.randn(1, max_len, d_model) * 0.1)

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


class TrajectoryTransformerEncoder(nn.Module):
    """Transformer 轨迹编码器，替代 LSTM 编码器
    输入: (B, N, T, input_dim), time_mask: (B, N, T)
    输出: (B, N, d_model)
    """
    def __init__(self, input_dim=5, d_model=128, nhead=4, num_layers=2, dropout=0.2, max_len=100):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(d_model, d_model)
        self.d_model = d_model

    def forward(self, traj, time_mask):
        B, N, T, D = traj.shape
        traj_flat = traj.reshape(B * N, T, D)
        mask_flat = time_mask.reshape(B * N, T)

        x = self.input_proj(traj_flat)
        x = self.pos_encoder(x)

        key_padding_mask = (mask_flat == 0)
        valid_mask = (mask_flat.sum(dim=1) > 0)

        if valid_mask.any():
            x_valid = x[valid_mask]
            key_padding_mask_valid = key_padding_mask[valid_mask]
            out_valid = self.transformer(x_valid, src_key_padding_mask=key_padding_mask_valid)
        else:
            out_valid = torch.empty(0, T, self.d_model, device=traj.device)

        out_flat = torch.zeros(B * N, T, self.d_model, device=traj.device)
        if valid_mask.any():
            out_flat[valid_mask] = out_valid

        lengths = mask_flat.sum(dim=1, keepdim=True)
        sum_emb = (out_flat * mask_flat.unsqueeze(-1)).sum(dim=1)
        valid_len = lengths.clamp(min=1)
        emb_flat = sum_emb / valid_len

        zero_len_mask = (lengths.squeeze(-1) == 0)
        if zero_len_mask.any():
            emb_flat[zero_len_mask] = 0.0

        emb_flat = self.output_proj(emb_flat)
        emb = emb_flat.reshape(B, N, -1)
        return emb


class GlobalTopologyEncoder(nn.Module):
    def __init__(self, d_model=128, nhead=8, num_layers=2, dim_feedforward=256, dropout=0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x, mask):
        padding_mask = (mask == 0)
        x = x * mask.unsqueeze(-1)
        out = self.transformer(x, src_key_padding_mask=padding_mask)
        out = self.out_proj(out)
        out = out * mask.unsqueeze(-1)
        out = torch.nan_to_num(out, nan=0.0, posinf=1.0, neginf=-1.0)
        out = torch.clamp(out, -10.0, 10.0)
        return out


class CostPredictor(nn.Module):
    def __init__(self, feat_dim, hidden_dim=64):
        super().__init__()
        self.feat_proj = nn.Linear(feat_dim, hidden_dim)
        self.similarity = nn.Bilinear(hidden_dim, hidden_dim, 1)
        nn.init.constant_(self.similarity.bias, -2.0)

    def forward(self, x_A, x_B):
        hA = self.feat_proj(x_A)
        hB = self.feat_proj(x_B)
        hA = torch.clamp(hA, -5.0, 5.0)
        hB = torch.clamp(hB, -5.0, 5.0)
        hA_exp = hA.unsqueeze(2).expand(-1, -1, hB.size(1), -1)
        hB_exp = hB.unsqueeze(1).expand(-1, hA.size(1), -1, -1)
        logits = self.similarity(hA_exp, hB_exp).squeeze(-1)
        logits = torch.clamp(logits, -20.0, 20.0)
        return logits


class UnbalancedSinkhornSolver(nn.Module):
    """
    不均衡 Sinkhorn 求解器，通过扩展代价矩阵（添加垃圾桶）允许部分航迹不匹配。
    """
    def __init__(self, num_iters=20, tau=1.0, epsilon=1e-3,
                 dustbin_cost=0.0, learn_dustbin_cost=False,
                 dustbin_mass_ratio=0.2, auto_dustbin_mass=True,
                 stop_thresh=1e-5):
        super().__init__()
        self.num_iters = num_iters
        self.tau = tau
        self.epsilon = epsilon
        self.stop_thresh = stop_thresh
        self.auto_dustbin_mass = auto_dustbin_mass
        self.dustbin_mass_ratio = dustbin_mass_ratio
        
        if learn_dustbin_cost:
            self.dustbin_cost = nn.Parameter(torch.tensor(dustbin_cost, dtype=torch.float32))
        else:
            self.register_buffer('dustbin_cost', torch.tensor(dustbin_cost, dtype=torch.float32))

    def forward(self, logits, mask0=None, mask1=None):
        B, N0, N1 = logits.shape
        device = logits.device
        
        if mask0 is None:
            mask0 = torch.ones(B, N0, device=device)
        if mask1 is None:
            mask1 = torch.ones(B, N1, device=device)
        
        logits_masked = logits.clone()
        logits_masked = logits_masked.masked_fill((mask0 == 0).unsqueeze(-1), -1e9)
        logits_masked = logits_masked.masked_fill((mask1 == 0).unsqueeze(1), -1e9)
        
        valid0 = mask0.sum(dim=1)
        valid1 = mask1.sum(dim=1)
        
        a = mask0 / (valid0.unsqueeze(1) + 1e-8)
        b = mask1 / (valid1.unsqueeze(1) + 1e-8)
        a = a * mask0
        b = b * mask1
        
        if self.auto_dustbin_mass:
            dustbin_mass_a = 1.0 - a.sum(dim=1)
            dustbin_mass_b = 1.0 - b.sum(dim=1)
            dustbin_mass_a = torch.clamp(dustbin_mass_a, min=0.0)
            dustbin_mass_b = torch.clamp(dustbin_mass_b, min=0.0)
        else:
            dustbin_mass_a = torch.full((B,), self.dustbin_mass_ratio, device=device)
            dustbin_mass_b = torch.full((B,), self.dustbin_mass_ratio, device=device)
            a = a * (1.0 - dustbin_mass_a.unsqueeze(1))
            b = b * (1.0 - dustbin_mass_b.unsqueeze(1))
        
        a_ext = torch.cat([a, dustbin_mass_a.unsqueeze(1)], dim=1)
        b_ext = torch.cat([b, dustbin_mass_b.unsqueeze(1)], dim=1)
        
        dustbin_val = self.dustbin_cost.to(device)
        logits_ext = torch.full((B, N0+1, N1+1), dustbin_val, device=device)
        logits_ext[:, :N0, :N1] = logits_masked
        logits_ext = torch.clamp(logits_ext, -20.0, 20.0)
        
        K = torch.exp(logits_ext / self.tau)
        K = K + self.epsilon
        
        u = torch.ones(B, N0+1, device=device)
        v = torch.ones(B, N1+1, device=device)
        
        for i in range(self.num_iters):
            Kv = torch.matmul(K, v.unsqueeze(-1)).squeeze(-1)
            u_new = a_ext / (Kv + 1e-12)
            u_new = torch.nan_to_num(u_new, nan=1.0, posinf=1.0, neginf=1.0)
            
            KTu = torch.matmul(K.transpose(1, 2), u_new.unsqueeze(-1)).squeeze(-1)
            v_new = b_ext / (KTu + 1e-12)
            v_new = torch.nan_to_num(v_new, nan=1.0, posinf=1.0, neginf=1.0)
            
            u, v = u_new, v_new
        
        P_ext = u.unsqueeze(-1) * K * v.unsqueeze(1)
        P_ext = torch.clamp(P_ext, 0.0, 1.0)
        P = P_ext[:, :N0, :N1]
        P = P * mask0.unsqueeze(-1) * mask1.unsqueeze(1)
        P = torch.nan_to_num(P, nan=0.0)
        return P


class PhysiTrackModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        input_dim = config.get('input_dim', 5)
        hidden_dim = config.get('hidden_dim', 128)
        phys_dim = config.get('phys_dim', 64)
        lstm_layers = config.get('lstm_layers', 2)
        dropout = config.get('dropout', 0.2)
        use_topology = config.get('use_topology', True)
        topo_num_layers = config.get('topo_num_layers', 2)
        topo_nhead = config.get('topo_nhead', 8)
        topo_dim_feedforward = config.get('topo_dim_feedforward', 256)

        self.use_topology = use_topology
        self.phys_encoder = PhysicalResidualEncoder(phys_dim)

        # 编码器选择: LSTM (默认) 或 Transformer
        self.use_transformer_encoder = config.get('use_transformer_encoder', False)
        transformer_nhead = config.get('transformer_nhead', 4)
        transformer_num_layers = config.get('transformer_num_layers', 2)
        if self.use_transformer_encoder:
            self.traj_encoder = TrajectoryTransformerEncoder(
                input_dim=input_dim,
                d_model=hidden_dim,
                nhead=transformer_nhead,
                num_layers=transformer_num_layers,
                dropout=dropout,
                max_len=100
            )
            print(f"  [PhysiTrackModel] Using TransformerEncoder (nhead={transformer_nhead}, layers={transformer_num_layers})")
        else:
            self.traj_encoder = TrajectoryFeatureEncoder(input_dim, hidden_dim, lstm_layers, dropout)

        self.fusion_dim = hidden_dim + phys_dim

        if use_topology:
            self.topology_encoder = GlobalTopologyEncoder(
                d_model=self.fusion_dim,
                nhead=topo_nhead,
                num_layers=topo_num_layers,
                dim_feedforward=topo_dim_feedforward,
                dropout=dropout
            )
            self.cost_predictor = CostPredictor(self.fusion_dim, hidden_dim)
        else:
            self.cost_predictor = self._make_legacy_cost_predictor(hidden_dim, phys_dim, hidden_dim)

        self.sinkhorn = UnbalancedSinkhornSolver(
            num_iters=config.get('sinkhorn_iters', 20),
            tau=config.get('sinkhorn_tau', 1.0),
            epsilon=config.get('sinkhorn_epsilon', 1e-3),
            dustbin_cost=config.get('dustbin_cost', 0.0),
            learn_dustbin_cost=config.get('learn_dustbin_cost', False),
            auto_dustbin_mass=config.get('auto_dustbin_mass', True),
            dustbin_mass_ratio=config.get('dustbin_mass_ratio', 0.2),
            stop_thresh=config.get('sinkhorn_stop_thresh', 1e-5)
        )

    def _make_legacy_cost_predictor(self, feat_dim, phys_dim, hidden_dim):
        class LegacyCostPredictor(nn.Module):
            def __init__(self, feat_dim, phys_dim, hidden_dim):
                super().__init__()
                self.feat_proj = nn.Linear(feat_dim, hidden_dim)
                self.phys_proj = nn.Linear(phys_dim, hidden_dim)
                self.similarity = nn.Bilinear(hidden_dim, hidden_dim, 1)
                nn.init.constant_(self.similarity.bias, -2.0)

            def forward(self, feat_A, phys_A, feat_B, phys_B):
                hA = self.feat_proj(feat_A) + self.phys_proj(phys_A)
                hB = self.feat_proj(feat_B) + self.phys_proj(phys_B)
                hA = torch.clamp(hA, -5.0, 5.0)
                hB = torch.clamp(hB, -5.0, 5.0)
                hA_exp = hA.unsqueeze(2).expand(-1, -1, hB.size(1), -1)
                hB_exp = hB.unsqueeze(1).expand(-1, hA.size(1), -1, -1)
                logits = self.similarity(hA_exp, hB_exp).squeeze(-1)
                return torch.clamp(logits, -20.0, 20.0)
        return LegacyCostPredictor(feat_dim, phys_dim, hidden_dim)

    def forward(self, traj_A, traj_B, timestamps_A, timestamps_B, mask_A, mask_B):
        traj_A = torch.nan_to_num(traj_A, nan=0.0, posinf=10.0, neginf=-10.0)
        traj_B = torch.nan_to_num(traj_B, nan=0.0, posinf=10.0, neginf=-10.0)
        timestamps_A = torch.nan_to_num(timestamps_A, nan=0.0)
        timestamps_B = torch.nan_to_num(timestamps_B, nan=0.0)

        phys_A = self.phys_encoder(traj_A, timestamps_A)
        phys_B = self.phys_encoder(traj_B, timestamps_B)

        if self.use_transformer_encoder:
            # 创建时间步掩码 (B,N,T): timestamp>0 为有效，同时乘以轨迹掩码
            time_mask_A = (timestamps_A > 0).float() * mask_A.unsqueeze(-1)
            time_mask_B = (timestamps_B > 0).float() * mask_B.unsqueeze(-1)
            feat_A = self.traj_encoder(traj_A, time_mask_A)
            feat_B = self.traj_encoder(traj_B, time_mask_B)
        else:
            feat_A = self.traj_encoder(traj_A, mask_A)
            feat_B = self.traj_encoder(traj_B, mask_B)

        fused_A = torch.cat([feat_A, phys_A], dim=-1)
        fused_B = torch.cat([feat_B, phys_B], dim=-1)

        if self.use_topology:
            fused_A = self.topology_encoder(fused_A, mask_A)
            fused_B = self.topology_encoder(fused_B, mask_B)
            logits = self.cost_predictor(fused_A, fused_B)
        else:
            # LegacyCostPredictor expects separate feat_A, phys_A, feat_B, phys_B
            logits = self.cost_predictor(feat_A, phys_A, feat_B, phys_B)

        mask_A_bool = mask_A.bool()
        mask_B_bool = mask_B.bool()
        logits = logits.masked_fill(~mask_A_bool.unsqueeze(-1), -1e9)
        logits = logits.masked_fill(~mask_B_bool.unsqueeze(1), -1e9)

        P = self.sinkhorn(logits, mask_A.float(), mask_B.float())
        aux = (fused_A, fused_B, feat_A, feat_B, phys_A, phys_B)
        return P, logits, aux


def focal_loss(logits, targets, alpha=0.25, gamma=2.0):
    probs = torch.sigmoid(logits)
    p_t = probs * targets + (1 - probs) * (1 - targets)
    focal_weight = (1 - p_t) ** gamma
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    loss = -alpha_t * focal_weight * torch.log(p_t.clamp(min=1e-7))
    return loss.mean()


def association_loss_balanced(logits, Y, mask_A, mask_B, neg_pos_ratio=2,
                              focal_alpha=0.25, focal_gamma=2.0):
    valid_pairs = mask_A.unsqueeze(-1) * mask_B.unsqueeze(1)
    valid_mask = valid_pairs.bool()
    logits_valid = logits[valid_mask]
    Y_valid = Y[valid_mask]

    pos_mask = (Y_valid == 1)
    neg_mask = (Y_valid == 0)
    pos_indices = torch.where(pos_mask)[0]
    neg_indices = torch.where(neg_mask)[0]
    num_pos = len(pos_indices)

    if num_pos == 0:
        return torch.tensor(0.0, device=logits.device, requires_grad=True)

    num_neg_target = num_pos * neg_pos_ratio
    if len(neg_indices) > num_neg_target:
        perm = torch.randperm(len(neg_indices), device=neg_indices.device)
        selected_neg = neg_indices[perm[:num_neg_target]]
    else:
        selected_neg = neg_indices

    selected_indices = torch.cat([pos_indices, selected_neg])
    logits_sampled = logits_valid[selected_indices]
    Y_sampled = Y_valid[selected_indices]

    return focal_loss(logits_sampled, Y_sampled, alpha=focal_alpha, gamma=focal_gamma)


def physical_consistency_loss(phys_A, phys_B, Y, mask_A, mask_B):
    phys_A_norm = F.normalize(phys_A, p=2, dim=-1, eps=1e-8)
    phys_B_norm = F.normalize(phys_B, p=2, dim=-1, eps=1e-8)
    sim = torch.bmm(phys_A_norm, phys_B_norm.transpose(1, 2))
    valid = (mask_A.unsqueeze(-1) * mask_B.unsqueeze(1)).bool()
    Y_bool = Y.bool() & valid

    pos_loss = 0.0
    neg_loss = 0.0
    pos_count = 0
    neg_count = 0
    for b in range(Y.shape[0]):
        pos_indices = torch.nonzero(Y_bool[b], as_tuple=False)
        if pos_indices.size(0) > 0:
            pos_sim = sim[b][Y_bool[b]]
            pos_loss += (1 - pos_sim).sum()
            pos_count += pos_indices.size(0)
        neg_mask = valid[b] & (~Y.bool()[b])
        if neg_mask.any():
            neg_indices = torch.nonzero(neg_mask, as_tuple=False)
            n_neg = min(1000, neg_indices.size(0))
            idx = torch.randint(0, neg_indices.size(0), (n_neg,), device=neg_indices.device)
            neg_sim = sim[b][neg_indices[idx, 0], neg_indices[idx, 1]]
            neg_loss += (neg_sim + 1).clamp(min=0).sum()
            neg_count += n_neg
    if pos_count > 0:
        pos_loss = pos_loss / pos_count
    if neg_count > 0:
        neg_loss = neg_loss / neg_count
    return pos_loss + 0.1 * neg_loss


class PhysiTrackLoss(nn.Module):
    def __init__(self, lambda_assoc=1.0, lambda_phys=0.2, neg_pos_ratio=2,
                 focal_alpha=0.25, focal_gamma=2.0):
        super().__init__()
        self.lambda_assoc = lambda_assoc
        self.lambda_phys = lambda_phys
        self.neg_pos_ratio = neg_pos_ratio
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

    def forward(self, logits, Y, mask_A, mask_B, phys_A, phys_B):
        loss_assoc = association_loss_balanced(
            logits, Y, mask_A, mask_B,
            neg_pos_ratio=self.neg_pos_ratio,
            focal_alpha=self.focal_alpha,
            focal_gamma=self.focal_gamma
        )
        total = self.lambda_assoc * loss_assoc
        loss_phys = 0.0
        if phys_A is not None and phys_B is not None:
            loss_phys = physical_consistency_loss(phys_A, phys_B, Y, mask_A, mask_B)
            total += self.lambda_phys * loss_phys
        return total, {'assoc': loss_assoc.item(), 'phys': loss_phys.item()}
