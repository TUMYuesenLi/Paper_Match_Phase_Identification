import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import warnings
warnings.filterwarnings('ignore')


class TypeAwareAttentionPool(nn.Module):
    def __init__(self, in_dim, hidden_dim=128,
                 use_role: bool = True, num_roles: int = 4,
                 use_team_embed: bool = True):
        super().__init__()
        self.use_role = use_role
        self.use_team_embed = use_team_embed

        if use_role:
            self.role_emb = nn.Embedding(num_roles, in_dim)
        if use_team_embed:
            self.team_emb = nn.Embedding(2, in_dim)

        in_feat = in_dim \
                  + (in_dim if use_role else 0) \
                  + (in_dim if use_team_embed else 0)

        self.scorer = nn.Sequential(
            nn.Linear(in_feat, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    @torch.no_grad()
    def _safe_embed(self, emb: nn.Embedding, idx: torch.Tensor, out_dim: int):

        N = idx.numel()
        device = idx.device
        out = torch.zeros(N, out_dim, device=device)
        valid = idx >= 0
        if valid.any():
            out[valid] = emb(idx[valid])
        return out

    def forward(self, x: torch.Tensor,
                node_type: torch.Tensor,
                role_id: torch.Tensor = None,
                return_attention: bool = False):

        N, D = x.shape
        feats = [x]

        if self.use_role:
            if role_id is None:
                raise ValueError("role_id is required when use_role=True")
            role_vec = self._safe_embed(self.role_emb, role_id, D)
            feats.append(role_vec)
        if self.use_team_embed:
            team_vec = self._safe_embed(self.team_emb, node_type, D)
            feats.append(team_vec)

        aug = torch.cat(feats, dim=-1)          # [N, *]
        raw = self.scorer(aug).squeeze(-1)
        valid_nt = (node_type >= 0)
        alpha = torch.zeros_like(raw)

        for t in (0, 1):
            mask = valid_nt & (node_type == t)
            if mask.any():
                alpha[mask] = torch.softmax(raw[mask], dim=0)

        graph_emb = (alpha.unsqueeze(-1) * x).sum(dim=0)  # [D]

        if return_attention:
            return graph_emb, alpha
        return graph_emb


class FrameLevelGNN(MessagePassing):
    def __init__(self, node_dim, edge_dim, global_dim, out_dim, use_global=True):
        super(FrameLevelGNN, self).__init__(aggr='mean')
        self.use_global = use_global

        self.msg_mlp = nn.Sequential(
            nn.Linear(2 * node_dim + edge_dim, out_dim),
            # nn.ReLU(),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(out_dim, out_dim)
        )
        self.residual = nn.Linear(node_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)

        if self.use_global:
            self.global_mlp = nn.Sequential(
                nn.Linear(global_dim, out_dim),
                # nn.ReLU(),
                nn.LeakyReLU(negative_slope=0.01),
                nn.Linear(out_dim, out_dim)
            )

    def forward(self, x, edge_index, edge_attr, global_feat):
        # residual = self.residual(x)
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        # out = self.norm(out + residual)

        if self.use_global:
            global_feat = self.global_mlp(global_feat)

        return out, global_feat

    def message(self, x_i, x_j, edge_attr):
        if edge_attr.dim() == 2:
            edge_attr = edge_attr  # shape: [E, 1] (already correct)
        elif edge_attr.dim() == 1:
            edge_attr = edge_attr.unsqueeze(-1)  # shape: [E, 1]
        else:
            raise ValueError(f"edge_attr has unexpected shape: {edge_attr.shape}")

        msg_input = torch.cat([x_i, x_j, edge_attr], dim=-1)
        return self.msg_mlp(msg_input)


def _chk(name, t):
    if t is None: return
    if not torch.isfinite(t).all():
        bad = (~torch.isfinite(t)).nonzero(as_tuple=False)[:5]
        raise RuntimeError(f"[NaN/Inf] {name} find infinity value; e.g: {bad.tolist()}")

# import torch, os
# torch.autograd.set_detect_anomaly(True)  # 定位反向的 NaN 源头（慢但有效）

class SoccerGNN(nn.Module):
    def __init__(self, node_dim, edge_dim, global_dim, action_dim=32, d_model=256,
                 gnn_layers=2, trans_layers=2, nhead=8, dim_feedforward=512,
                 dropout=0.1, num_classes=3):
        super().__init__()

        self.gnn_layers = nn.ModuleList()
        for i in range(gnn_layers):
            in_dim = node_dim if i == 0 else d_model
            self.gnn_layers.append(FrameLevelGNN(in_dim, edge_dim, global_dim, d_model, use_global=(i == 0)))

        self.gate_layer = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Sigmoid()
        )

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
        self.num_classes = num_classes
        hidden_dim = d_model // 2
        self.pool = TypeAwareAttentionPool(in_dim=d_model, hidden_dim=hidden_dim,
                                                            use_role=True,
                                                            num_roles=5,
                                                            use_team_embed=True
                                                        )

        if num_classes == 2:
            self.classifier = nn.Sequential(
                nn.Linear(d_model, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(d_model, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, num_classes)
            )

        self.fusion_mlp = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.label_embed = nn.Embedding(num_classes, d_model)
        self.input_proj = nn.Linear(d_model + action_dim, d_model)

    def forward(self, batch_episodes, masks, device):
        frame_predictions = []
        proportions = []
        for i, episode in enumerate(batch_episodes):
            x_list = []
            label_list = []
            action_x_list = []
            node_type_list = []
            for frame in episode:
                x = frame['player'].x.to(device)
                edge_index = frame['player', 'teammate', 'player'].edge_index.to(device)
                edge_attr = frame['player', 'teammate', 'player'].edge_attr.to(device)
                if edge_attr.dim() == 1:
                    edge_attr = edge_attr.unsqueeze(-1)
                action_feat = frame['match_action'].x.to(device)
                global_feat = frame['traditional'].x.to(device)
                global_feat = torch.cat([global_feat, action_feat], dim=-1)
                action_x_list.append(action_feat)
                for gnn in self.gnn_layers:
                    x, global_feat = gnn(x, edge_index, edge_attr, global_feat)
                xg = torch.cat([x, global_feat.expand(x.size(0), -1)], dim=-1)
                x = self.fusion_mlp(xg)
                role_id = frame['player'].position.to(device).long()
                node_type = frame['player'].poss.to(device).long()
                x_pool = self.pool(x, node_type, role_id)  # [D]
                x_list.append(x_pool)
                node_type_list.append(node_type)

            frame_features = torch.stack(x_list, dim=0)  # [T, D]
            frame_features = self.norm(self.dropout(frame_features))
            frame_logit = self.classifier(frame_features)
            frame_predictions.append(frame_logit)
        return pad_sequence(frame_predictions, batch_first=True)

    def predict_episode(self, episode, device
                        ):
        x_list = []
        action_x_list = []
        attention_weights_list = []
        node_type_list = []
        for frame in episode:
            x = frame['player'].x.to(device)
            edge_index = frame['player', 'teammate', 'player'].edge_index.to(device)
            edge_attr = frame['player', 'teammate', 'player'].edge_attr.to(device)
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.unsqueeze(-1)
            global_feat = frame['traditional'].x.to(device)
            action_feat = frame['match_action'].x.to(device)
            global_feat = torch.cat([global_feat, action_feat], dim=-1)
            action_x_list.append(action_feat)
            for gnn in self.gnn_layers:
                x, global_feat = gnn(x, edge_index, edge_attr, global_feat)
            xg = torch.cat([x, global_feat.expand(x.size(0), -1)], dim=-1)
            x = self.fusion_mlp(xg)
            node_type = frame['player'].poss.to(device).long()
            role_id = frame['player'].position.to(device).long()
            pooled, attn_weights = self.pool(x, node_type, role_id, return_attention=True)
            attention_weights_list.append(attn_weights.cpu().numpy())
            x_list.append(pooled)
            node_type_list.append(node_type)
        frame_features = torch.stack(x_list, dim=0)  # [T, D]
        frame_features = self.norm(self.dropout(frame_features))
        frame_logit = self.classifier(frame_features)
        preds = (torch.sigmoid(frame_logit) > 0.5).long().squeeze(-1) if self.num_classes == 2 else torch.argmax(frame_logit, dim=-1)
        return frame_logit, preds, attention_weights_list


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean',
                 ignore_index=None, debug_checks=True,
                 time_pool='logsumexp', lse_normalize=True, eps=1e-8):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.debug_checks = debug_checks
        self.time_pool = time_pool
        self.lse_normalize = lse_normalize
        self.eps = eps

        if alpha is not None:
            if isinstance(alpha, (list, tuple)):
                alpha = torch.tensor(alpha, dtype=torch.float32)
            else:
                alpha = alpha.float()
            self.register_buffer("alpha", alpha)
        else:
            self.alpha = None

    def _reduce(self, loss, eff_mask):
        if self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'mean':
            denom = eff_mask.to(loss.dtype).sum().clamp_min(1.0)
            return loss.sum() / denom
        else:
            return loss

    def forward(self, logits, targets, mask=None, mode='auto'):
        if logits.dim() not in (2, 3):
            raise ValueError(f"Expect logits dim 2 or 3, got {logits.shape}")

        B = logits.shape[0]
        C = logits.size(-1)
        if mode == 'auto':
            if logits.dim() == 3 and targets.dim() == 2:
                mode = 'frame'
            elif (logits.dim() == 3 and targets.dim() == 1) or (logits.dim() == 2 and targets.dim() == 1):
                mode = 'sequence'
            else:
                raise ValueError(
                    f"Cannot auto-resolve mode for logits{tuple(logits.shape)} targets{tuple(targets.shape)}")

        if mode == 'frame':
            assert logits.dim() == 3 and targets.dim() == 2, "frame mode expects logits [B,T,C], targets [B,T]"
            B, T, C = logits.shape
            x = logits.reshape(-1, C).contiguous()  # [N, C]
            t = targets.reshape(-1).long().contiguous()  # [N]

            if mask is None:
                eff = torch.ones_like(t, dtype=torch.bool, device=t.device)
            else:
                eff = mask.reshape(-1).bool().to(t.device)

            if self.ignore_index is not None:
                eff = eff & (t != self.ignore_index)

            if self.debug_checks and eff.any():
                t_eff = t[eff]
                assert t_eff.dtype in (torch.int64, torch.long)
                assert int(t_eff.min()) >= 0, f"min target={int(t_eff.min())}"
                assert int(t_eff.max()) < C, f"max target={int(t_eff.max())} >= C({C})"
                if self.alpha is not None:
                    assert self.alpha.numel() == C, f"alpha size={self.alpha.numel()} vs C={C}"
                assert torch.isfinite(x).all(), "logits contain NaN/Inf"

            safe_idx = t.clone()
            safe_idx[~eff] = 0

            log_probs = F.log_softmax(x, dim=-1)  # [N, C]
            probs = log_probs.exp()
            pt = probs.gather(1, safe_idx.unsqueeze(1)).squeeze(1)  # [N]
            log_pt = log_probs.gather(1, safe_idx.unsqueeze(1)).squeeze(1)

            if self.alpha is not None:
                alpha_t = torch.ones_like(pt, device=pt.device)
                alpha_vals = self.alpha.to(pt.device)
                alpha_t[eff] = alpha_vals[t[eff]]
            else:
                alpha_t = 1.0

            loss = -alpha_t * (1.0 - pt).pow(self.gamma) * log_pt  # [N]
            loss = loss * eff.to(loss.dtype)
            return self._reduce(loss, eff)

        elif mode == 'sequence':
            if logits.dim() == 2:
                assert targets.dim() == 1 and targets.shape[0] == B
                t = targets.long()
                if mask is None:
                    eff_seq = torch.ones_like(t, dtype=torch.bool, device=t.device)
                else:
                    eff_seq = mask.view(-1).bool().to(t.device)

                if self.ignore_index is not None:
                    eff_seq = eff_seq & (t != self.ignore_index)

                x_seq = logits
                if self.debug_checks and eff_seq.any():
                    t_eff = t[eff_seq]
                    assert int(t_eff.min()) >= 0 and int(t_eff.max()) < C
                    if self.alpha is not None:
                        assert self.alpha.numel() == C

                log_probs = F.log_softmax(x_seq, dim=-1)  # [B, C]
                probs = log_probs.exp()
                pt = probs[torch.arange(B, device=t.device), t]  # [B]
                log_pt = log_probs[torch.arange(B, device=t.device), t]
                if self.alpha is not None:
                    alpha_t = self.alpha.to(logits.device)[t]
                else:
                    alpha_t = 1.0
                loss = -alpha_t * (1.0 - pt).pow(self.gamma) * log_pt
                loss = loss * eff_seq.to(loss.dtype)
                return self._reduce(loss, eff_seq)

            else:
                assert targets.dim() == 1 and (mask is None or mask.dim() == 2)
                B, T, C = logits.shape
                t = targets.long()

                if mask is None:
                    m = torch.ones((B, T), dtype=torch.bool, device=logits.device)
                else:
                    m = mask.bool().to(logits.device)
                    assert m.shape == (B, T)

                eff_seq = m.any(dim=1)
                if self.ignore_index is not None:
                    eff_seq = eff_seq & (t != self.ignore_index)

                if not eff_seq.any():
                    return logits.new_tensor(0.0)

                x = logits[eff_seq]  # [B_eff, T, C]
                m_eff = m[eff_seq]  # [B_eff, T]
                t_eff = t[eff_seq]  # [B_eff]

                if self.debug_checks:
                    assert torch.isfinite(x).all(), "logits contain NaN/Inf"
                    assert int(t_eff.min()) >= 0 and int(t_eff.max()) < C, f"targets out of range for C={C}"

                if self.time_pool == 'mean':
                    denom = m_eff.sum(dim=1, keepdim=True).clamp_min(1).to(x.dtype)  # [B_eff,1]
                    x_seq = (x * m_eff.unsqueeze(-1).to(x.dtype)).sum(dim=1) / denom  # [B_eff, C]

                    log_probs = F.log_softmax(x_seq, dim=-1)
                    probs = log_probs.exp()
                    pt = probs[torch.arange(x_seq.size(0), device=x.device), t_eff]
                    log_pt = log_probs[torch.arange(x_seq.size(0), device=x.device), t_eff]

                elif self.time_pool == 'max':
                    x_masked = x.masked_fill(~m_eff.unsqueeze(-1), float('-inf'))
                    x_seq = x_masked.max(dim=1).values
                    x_seq = torch.where(torch.isfinite(x_seq), x_seq, x.new_zeros(x_seq.shape))
                    log_probs = F.log_softmax(x_seq, dim=-1)
                    probs = log_probs.exp()
                    pt = probs[torch.arange(x_seq.size(0), device=x.device), t_eff]
                    log_pt = log_probs[torch.arange(x_seq.size(0), device=x.device), t_eff]

                elif self.time_pool == 'logsumexp':
                    x_masked = x.masked_fill(~m_eff.unsqueeze(-1), float('-inf'))
                    lse = torch.logsumexp(x_masked, dim=1)  # [B_eff, C]
                    if self.lse_normalize:
                        valid_len = m_eff.sum(dim=1, keepdim=True).clamp_min(1).to(x.dtype)
                        x_seq = lse - valid_len.log()
                    else:
                        x_seq = lse
                    log_probs = F.log_softmax(x_seq, dim=-1)
                    probs = log_probs.exp()
                    pt = probs[torch.arange(x_seq.size(0), device=x.device), t_eff]
                    log_pt = log_probs[torch.arange(x_seq.size(0), device=x.device), t_eff]

                elif self.time_pool == 'noisy_or':
                    p = F.softmax(x, dim=-1)  # [B_eff,T,C]
                    p = torch.clamp(p, self.eps, 1 - self.eps)
                    idx = t_eff.view(-1, 1, 1).expand(-1, p.size(1), 1)  # [B_eff,T,1]
                    p_true = p.gather(dim=-1, index=idx).squeeze(-1)  # [B_eff,T]
                    one_minus = 1.0 - p_true
                    one_minus = torch.where(m_eff, one_minus, torch.ones_like(one_minus))
                    prod = one_minus.prod(dim=1)  # [B_eff]
                    pt = 1.0 - prod  # p_true_seq
                    pt = torch.clamp(pt, self.eps, 1 - self.eps)
                    log_pt = torch.log(pt)
                else:
                    raise ValueError(f"Unknown time_pool: {self.time_pool}")

                if self.alpha is not None:
                    alpha_t = self.alpha.to(logits.device)[t_eff]
                else:
                    alpha_t = 1.0

                loss = -alpha_t * (1.0 - pt).pow(self.gamma) * log_pt  # [B_eff]
                return self._reduce(loss, eff_seq[eff_seq].to(loss.dtype))
        else:
            raise ValueError(f"Unknown mode: {mode}")


class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean',
                 ignore_index=None, time_pool='noisy_or', mode='auto', eps=1e-8):

        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.time_pool = time_pool
        self.mode = mode
        if isinstance(alpha, (list, tuple)):
            self.alpha_neg, self.alpha_pos = float(alpha[0]), float(alpha[1])
            self.alpha_scalar = None
        else:
            self.alpha_scalar = float(alpha) if alpha is not None else None
            self.alpha_neg = self.alpha_pos = None
        self.eps = eps

    def _alpha_factor(self, t_float):
        if self.alpha_scalar is not None:
            return t_float * self.alpha_scalar + (1.0 - t_float) * (1.0 - self.alpha_scalar)
        elif self.alpha_pos is not None:
            return t_float * self.alpha_pos + (1.0 - t_float) * self.alpha_neg
        else:
            return 1.0

    def _reduce(self, loss, eff):
        if self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'mean':
            denom = eff.to(loss.dtype).sum().clamp_min(1.0)
            return loss.sum() / denom
        else:
            return loss

    def forward(self, logits, targets, mask=None, mode=None):
        if logits.dim() == 3 and logits.size(-1) == 1:
            logits = logits.squeeze(-1)  # -> [B,T]
        if logits.dim() == 2 and logits.size(1) == 1:
            logits = logits.squeeze(1)   # -> [B]

        if mode is None or mode == 'auto':
            # 自动判断：targets [B] → sequence；targets [B,T] → frame
            mode = 'sequence' if targets.dim() == 1 else 'frame'

        if mode == 'frame':
            # 帧级 focal（BCE with logits），mask/ignore_index 在时间维逐元素应用
            assert logits.dim() == 2, f"frame mode expect logits [B,T]，got {tuple(logits.shape)}"
            assert targets.dim() == 2, f"frame mode expect targets [B,T]，got {tuple(targets.shape)}"
            B, T = logits.shape
            t = targets.reshape(-1).long()
            x = logits.reshape(-1)

            if mask is None:
                eff = torch.ones_like(t, dtype=torch.bool)
            else:
                eff = mask.reshape(-1).bool()

            if self.ignore_index is not None:
                eff = eff & (t != self.ignore_index)

            if eff.any():
                t_eff = t[eff]
                assert int(t_eff.min()) >= 0 and int(t_eff.max()) <= 1, \
                    f"frame mode needs label in {{0,1}}，but find [{int(t_eff.min())}, {int(t_eff.max())}]"

            t_float = t.float()
            bce = F.binary_cross_entropy_with_logits(x, t_float, reduction='none')  # [B*T]
            p = torch.sigmoid(x)
            pt = torch.where(t_float > 0.5, p, 1 - p)
            focal = (1.0 - pt).pow(self.gamma)
            alpha = self._alpha_factor(t_float)
            loss = alpha * focal * bce
            loss = loss * eff.to(loss.dtype)
            return self._reduce(loss, eff)

        elif mode == 'sequence':
            if logits.dim() == 1:
                B = logits.shape[0]
                t = targets.long()
                if mask is None:
                    eff = torch.ones_like(t, dtype=torch.bool)
                else:
                    eff = mask.bool()

                if self.ignore_index is not None:
                    eff = eff & (t != self.ignore_index)

                if eff.any():
                    t_eff = t[eff]
                    assert int(t_eff.min()) >= 0 and int(t_eff.max()) <= 1

                t_float = t.float()
                bce = F.binary_cross_entropy_with_logits(logits, t_float, reduction='none')
                p = torch.sigmoid(logits)
                pt = torch.where(t_float > 0.5, p, 1 - p)
                focal = (1.0 - pt).pow(self.gamma)
                alpha = self._alpha_factor(t_float)
                loss = alpha * focal * bce
                loss = loss * eff.to(loss.dtype)
                return self._reduce(loss, eff)

            assert logits.dim() == 2 and targets.dim() == 1, \
                f"sequence mode expect logits [B,T], targets [B]，got logits{tuple(logits.shape)}, targets{tuple(targets.shape)}"
            B, T = logits.shape
            t = targets.long()
            t_float = t.float()

            if mask is None:
                m = torch.ones((B, T), dtype=torch.bool, device=logits.device)
            else:
                m = mask.bool().to(logits.device)
                assert m.shape == (B, T)

            eff = m.any(dim=1)
            if self.ignore_index is not None:
                eff = eff & (t != self.ignore_index)

            if not eff.any():
                return logits.new_tensor(0.0)

            x = logits[eff]     # [B_eff, T]
            m_eff = m[eff]      # [B_eff, T]
            t_eff = t_float[eff]# [B_eff]

            if self.time_pool == 'mean':
                denom = m_eff.sum(dim=1).clamp_min(1).to(x.dtype)
                x_seq = (x * m_eff.to(x.dtype)).sum(dim=1) / denom          # [B_eff] (logit 均值)

                # BCE with logits
                bce = F.binary_cross_entropy_with_logits(x_seq, t_eff, reduction='none')
                p_seq = torch.sigmoid(x_seq)
                pt = torch.where(t_eff > 0.5, p_seq, 1 - p_seq)

            elif self.time_pool == 'max':
                x_masked = x.masked_fill(~m_eff, float('-inf'))
                x_seq = x_masked.max(dim=1).values
                x_seq = torch.where(torch.isfinite(x_seq), x_seq, x.new_zeros(x_seq.shape))  # 若全无效，置0
                bce = F.binary_cross_entropy_with_logits(x_seq, t_eff, reduction='none')
                p_seq = torch.sigmoid(x_seq)
                pt = torch.where(t_eff > 0.5, p_seq, 1 - p_seq)

            elif self.time_pool == 'logsumexp':
                x_masked = x.masked_fill(~m_eff, float('-inf'))
                lse = torch.logsumexp(x_masked, dim=1)                       # [B_eff]
                valid_len = m_eff.sum(dim=1).clamp_min(1).to(x.dtype)
                x_seq = lse - valid_len.log()                                 # 相当于对概率做平均的 logits 近似
                bce = F.binary_cross_entropy_with_logits(x_seq, t_eff, reduction='none')
                p_seq = torch.sigmoid(x_seq)
                pt = torch.where(t_eff > 0.5, p_seq, 1 - p_seq)

            elif self.time_pool == 'noisy_or':
                p = torch.sigmoid(x)                                          # [B_eff, T]
                p = torch.clamp(p, self.eps, 1 - self.eps)
                one_minus_p = 1.0 - p
                one_minus_p = torch.where(m_eff, one_minus_p, torch.ones_like(one_minus_p))
                prod = one_minus_p.prod(dim=1)                                # ∏(1-p_t)
                p_seq = 1.0 - prod                                            # [B_eff]
                p_seq = torch.clamp(p_seq, self.eps, 1 - self.eps)

                bce = -(t_eff * torch.log(p_seq) + (1.0 - t_eff) * torch.log(1.0 - p_seq))
                pt = torch.where(t_eff > 0.5, p_seq, 1 - p_seq)
            else:
                raise ValueError(f"Unknown time_pool: {self.time_pool}")

            focal = (1.0 - pt).pow(self.gamma)
            alpha = self._alpha_factor(t_eff)
            loss = alpha * focal * bce                                        # [B_eff]

            return self._reduce(loss, eff.new_ones(loss.shape, dtype=torch.float32, device=loss.device))

        else:
            raise ValueError(f"Unknown mode: {mode}")


class RatioLoss(nn.Module):
    def __init__(self, reduction='mean', mode='mse', class_weights=None):
        super().__init__()
        self.reduction = reduction
        self.mode = mode
        self.class_weights = class_weights

    def forward(self, pred, target):
        if self.mode == 'mse':
            loss = F.mse_loss(pred, target, reduction=self.reduction)
        elif self.mode == 'mae':
            loss = F.l1_loss(pred, target, reduction=self.reduction)
        elif self.mode == 'kl':
            loss = F.kl_div(pred.log(), target, reduction=self.reduction)
        elif self.mode == 'cosine':
            loss = 1 - F.cosine_similarity(pred, target, dim=-1)
            loss = loss.mean() if self.reduction == 'mean' else loss.sum()
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")
        if self.class_weights is not None:
            weight = self.class_weights.view(1, -1).to(pred.device)
            loss = loss * weight
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
