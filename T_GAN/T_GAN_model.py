import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import math


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


class TemporalAttentionPool(nn.Module):
    def __init__(self, dim, hidden=128):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1)
        )

    def forward(self, x):  # x: [T, D] 或 [T, N, D]
        if x.dim() == 2:  # [T, D]
            w = torch.softmax(self.attn(x).squeeze(-1), dim=0)  # [T]
            return (w.unsqueeze(-1) * x).sum(dim=0), w  # [D], [T]
        elif x.dim() == 3:  # [T, N, D]
            T, N, D = x.shape
            w = torch.softmax(self.attn(x).squeeze(-1), dim=0)  # [T, N]
            return (w.unsqueeze(-1) * x).sum(dim=0), w  # [N, D], [T, N]
        else:
            raise ValueError("Unexpected shape")


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        if x.dim() == 2:  # [T, D]
            return self.dropout(x + self.pe[:x.size(0), :].squeeze(1))
        elif x.dim() == 3:  # [T, B, D]
            return self.dropout(x + self.pe[:x.size(0), :])
        else:
            raise ValueError(f"Unexpected input shape: {x.shape}")


# ===================== Event-gated + Event-biased Transformer blocks =====================
class EventGate(nn.Module):

    def __init__(self, d_model: int, d_event: int):
        super().__init__()
        self.to_gate = nn.Sequential(
            nn.Linear(d_event, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor, e: torch.Tensor, event_mask: torch.Tensor) -> torch.Tensor:
        """
        x: [B,T,D]
        e: [B,T,E]
        event_mask: [B,T] (0/1)
        """
        g = self.to_gate(e)  # [B,T,D]
        g = g * event_mask.unsqueeze(-1).to(g.dtype)  # 无事件帧 gate=0
        return x + g * x


class EventBiasMultiheadSelfAttention(nn.Module):
    def __init__(self, d_model: int, nhead: int, d_event: int, per_head: bool = True, dropout: float = 0.1):
        super().__init__()
        assert d_model % nhead == 0
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.per_head = per_head
        self.dropout = dropout

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

        out_dim = nhead if per_head else 1
        self.event_score = nn.Sequential(
            nn.Linear(d_event, d_model),
            nn.ReLU(),
            nn.Linear(d_model, out_dim)  # -> [B,T,H] or [B,T,1]
        )

    def forward(self,
                x: torch.Tensor,
                e: torch.Tensor,
                event_mask: torch.Tensor,
                key_padding_mask: torch.Tensor = None,
                return_attn: bool = False):
        B, T, D = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, T, self.nhead, self.head_dim).transpose(1, 2)  # [B,H,T,hd]
        k = k.view(B, T, self.nhead, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.nhead, self.head_dim).transpose(1, 2)

        # ---- event bias:  key=j ----
        s = self.event_score(e)  # [B,T,H] or [B,T,1]
        s = s * event_mask.unsqueeze(-1).to(s.dtype)
        if self.per_head:
            # [B,T,H] -> [B,H,1,T]
            bias = s.permute(0, 2, 1).unsqueeze(2)
        else:
            # [B,T,1] -> [B,1,1,T]
            bias = s.transpose(1, 2).unsqueeze(1)

        # logits: [B,H,T,T]
        logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        logits = logits + bias  # broadcast to [B,H,T,T]

        if key_padding_mask is not None:
            logits = logits.masked_fill(key_padding_mask[:, None, None, :], float("-inf"))

        attn = torch.softmax(logits, dim=-1)  # [B,H,T,T]
        attn = F.dropout(attn, p=self.dropout, training=self.training)

        y = torch.matmul(attn, v)  # [B,H,T,hd]
        y = y.transpose(1, 2).contiguous().view(B, T, D)
        y = self.out(y)

        if return_attn:
            return y, attn
        return y


class EventBiasedTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, d_event: int,
                 dim_feedforward: int = 512, dropout: float = 0.1, per_head_bias: bool = True):
        super().__init__()
        self.self_attn = EventBiasMultiheadSelfAttention(
            d_model=d_model, nhead=nhead, d_event=d_event, per_head=per_head_bias, dropout=dropout
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )

    def forward(self, x, e, event_mask, key_padding_mask=None, return_attn: bool = False):
        h = self.norm1(x)
        if return_attn:
            h, attn = self.self_attn(h, e, event_mask, key_padding_mask=key_padding_mask, return_attn=True)
        else:
            h = self.self_attn(h, e, event_mask, key_padding_mask=key_padding_mask, return_attn=False)
        x = x + self.drop(h)

        h2 = self.norm2(x)
        h2 = self.ffn(h2)
        x = x + self.drop(h2)

        if return_attn:
            return x, attn
        return x


class EventBiasedTransformerEncoder(nn.Module):
    def __init__(self, num_layers: int, d_model: int, nhead: int, d_event: int,
                 dim_feedforward: int = 512, dropout: float = 0.1, per_head_bias: bool = True):
        super().__init__()
        self.layers = nn.ModuleList([
            EventBiasedTransformerEncoderLayer(
                d_model=d_model, nhead=nhead, d_event=d_event,
                dim_feedforward=dim_feedforward, dropout=dropout, per_head_bias=per_head_bias
            )
            for _ in range(num_layers)
        ])

    def forward(self, x, e, event_mask, key_padding_mask=None, return_attn: bool = False):
        attn_all = []
        for layer in self.layers:
            if return_attn:
                x, attn = layer(x, e, event_mask, key_padding_mask=key_padding_mask, return_attn=True)
                attn_all.append(attn)
            else:
                x = layer(x, e, event_mask, key_padding_mask=key_padding_mask, return_attn=False)
        if return_attn:
            return x, attn_all
        return x


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
        else:
            return out

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
        raise RuntimeError(f"[NaN/Inf] {name} 发现非有限值; 例: {bad.tolist()}")


class SoccerGNNTransformer(nn.Module):
    def __init__(self, node_dim, edge_dim, global_dim, action_dim=18, action_emb_dim=8, d_model=256,
                 gnn_layers=2, trans_layers=2, nhead=8, dim_feedforward=512,
                 dropout=0.1, num_classes=3):
        super().__init__()

        self.gnn_layers = nn.ModuleList()
        for i in range(gnn_layers):
            in_dim = node_dim if i == 0 else d_model
            self.gnn_layers.append(FrameLevelGNN(in_dim, edge_dim, global_dim, d_model, use_global=False))

        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)

        self.transformer = EventBiasedTransformerEncoder(
            num_layers=trans_layers,
            d_model=d_model,
            nhead=nhead,
            d_event=action_emb_dim,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            per_head_bias=True
        )

        self.event_gate = EventGate(d_model=d_model, d_event=action_emb_dim)

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
        self.num_classes = num_classes
        hidden_dim = d_model // 2
        self.pool = TypeAwareAttentionPool(in_dim=d_model, hidden_dim=hidden_dim,
                                           use_role=True,
                                           num_roles=5,
                                           use_team_embed=True
                                           )
        self.tpool = TemporalAttentionPool(d_model, hidden_dim)

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
        self.episode_context_mlp = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
        )
        self.transformer_input_proj = nn.Linear(d_model + action_emb_dim, d_model)
        self.transformer_input_proj_with_counter = nn.Linear(d_model + action_emb_dim + 8, d_model)
        self.transformer_input_proj_with_traditional_counter = nn.Linear(d_model + action_emb_dim + 8 + 17, d_model)
        self.transformer_input_proj_with_traditional = nn.Linear(d_model + action_emb_dim + 17, d_model)

        self.action_emb = nn.Embedding(action_dim, action_emb_dim)
        self.action_mlp = nn.Sequential(
            nn.Linear(action_emb_dim + 10, action_emb_dim),
            nn.ReLU(),
            nn.Linear(action_emb_dim, action_emb_dim),
        )

    def forward(self, batch_episodes, masks, y_key, device,
                sequence_level=False, counter_features=False, traditional_features=False):
        frame_predictions = []
        max_len = masks.shape[1]

        for i, episode in enumerate(batch_episodes):
            x_list = []
            action_x_list = []
            event_mask_list = []
            counter_x_list = []
            node_type_list = []
            traditional_x_list = []

            for frame in episode:
                x = frame['player'].x.to(device)
                edge_index = frame['player', 'teammate', 'player'].edge_index.to(device)
                edge_attr = frame['player', 'teammate', 'player'].edge_attr.to(device)
                if edge_attr.dim() == 1:
                    edge_attr = edge_attr.unsqueeze(-1)

                global_feat = frame['global'].x[:, 0:-3].to(device)
                if traditional_features:
                    traditional_feat = frame['traditional'].x.to(device)
                    traditional_x_list.append(traditional_feat)
                ma = frame['match_action'].x.to(device)  # float tensor
                action_type_id = ma[:, 0].long().clamp(min=0, max=self.action_emb.num_embeddings - 1)  # [1]
                action_other = ma[:, 1:].float()  # [1,9]
                action_type_emb = self.action_emb(action_type_id)  # [1, action_emb_dim]
                action_feat = torch.cat([action_type_emb, action_other], dim=-1).view(-1)  # [action_emb_dim+9]
                action_feat = self.action_mlp(action_feat)  # [action_emb_dim]
                action_x_list.append(action_feat)

                if ('event_mask' in frame) and hasattr(frame['event_mask'], 'x'):
                    em = frame['event_mask'].x.view(-1).to(device)
                else:
                    em = (ma[:, -1].long().view(-1) != 0).long()
                event_mask_list.append(em)

                if counter_features:
                    counter_feat = frame['global'].x[:, -3:].to(device)
                    counter_feat = self.episode_context_mlp(counter_feat).view(-1)
                    counter_x_list.append(counter_feat)

                for gnn in self.gnn_layers:
                    x = gnn(x, edge_index, edge_attr, global_feat)

                role_id = frame['player'].position.to(device).long()
                node_type = frame['player'].poss.to(device).long()
                x_pool = self.pool(x, node_type, role_id)  # [D]
                x_list.append(x_pool)
                node_type_list.append(node_type)

            frame_features_full = torch.stack(x_list, dim=0)  # [T_full, D]
            frame_action_features = torch.stack(action_x_list, dim=0)

            if traditional_features and counter_features:
                frame_traditional_features = torch.stack(traditional_x_list, dim=0)
                frame_traditional_features = frame_traditional_features.squeeze(1)
                frame_counter_features = torch.stack(counter_x_list, dim=0)
                frame_features_full = torch.cat(
                    [frame_features_full, frame_action_features, frame_counter_features, frame_traditional_features],
                    dim=-1
                )
                frame_features_full = self.transformer_input_proj_with_traditional_counter(frame_features_full)

            elif traditional_features:
                frame_traditional_features = torch.stack(traditional_x_list, dim=0)
                frame_traditional_features = frame_traditional_features.squeeze(1)
                frame_features_full = torch.cat(
                    [frame_features_full, frame_action_features, frame_traditional_features],
                    dim=-1
                )
                frame_features_full = self.transformer_input_proj_with_traditional(frame_features_full)

            elif counter_features:
                frame_counter_features = torch.stack(counter_x_list, dim=0)
                frame_features_full = torch.cat(
                    [frame_features_full, frame_action_features, frame_counter_features],
                    dim=-1
                )
                frame_features_full = self.transformer_input_proj_with_counter(frame_features_full)
            else:
                frame_features_full = torch.cat(
                    [frame_features_full, frame_action_features],
                    dim=-1
                )
                frame_features_full = self.transformer_input_proj(frame_features_full)

            T_full = frame_features_full.size(0)

            frame_features = frame_features_full  # [T_full, D]
            mask_i = masks[i, :T_full].to(frame_features.device)  # [T_full]

            # ---- Transformer batch_first）----
            valid_len = int(mask_i.sum().item())
            frame_features = self.pos_encoder(frame_features)  # [T_full, D]

            x_bt = frame_features.unsqueeze(0)  # [1, T, D]
            e_bt = frame_action_features.unsqueeze(0)  # [1, T, E]

            event_mask = torch.cat(event_mask_list, dim=0)[:T_full].to(frame_features.device)  # [T]
            m_bt = event_mask.unsqueeze(0).float()  # [1, T]

            key_padding_mask = (~mask_i.bool()).unsqueeze(0)  # [1, T]

            x_bt = self.event_gate(x_bt, e_bt, m_bt)

            if valid_len == 0:
                transformer_out_bt = torch.zeros_like(x_bt)  # [1, T, D]
            else:
                x_valid = x_bt[:, :valid_len, :]
                e_valid = e_bt[:, :valid_len, :]
                m_valid = m_bt[:, :valid_len]
                kpm_valid = key_padding_mask[:, :valid_len]
                out_valid = self.transformer(
                    x_valid, e_valid, m_valid,
                    key_padding_mask=kpm_valid,
                    return_attn=False
                )  # [1, L, D]

                transformer_out_bt = torch.zeros_like(x_bt)
                transformer_out_bt[:, :valid_len, :] = out_valid

            transformer_out = transformer_out_bt.squeeze(0)  # [T, D]
            transformer_out = self.norm(self.dropout(transformer_out))

            if sequence_level:
                transformer_out, frame_weights = self.tpool(transformer_out)
                frame_logit = self.classifier(transformer_out)  # [C] 或 [1]
                frame_predictions.append(frame_logit.unsqueeze(0))  # [1, C]
            else:
                frame_logit = self.classifier(transformer_out)  # [T_sparse, C] or [T_sparse, 1]
                frame_predictions.append(frame_logit)

        return pad_sequence(frame_predictions, batch_first=True)

    def predict_episode(self, episode, device,
                        sequence_level=False, node_level=False,
                        return_GNN=False,
                        return_transformer=False,
                        return_attn=False,
                        counter_features=False,
                        traditional_features=False
                        ):
        x_list = []
        action_x_list = []
        event_mask_list = []
        counter_x_list = []
        traditional_x_list = []
        attention_weights_list = []
        frame_weights_list = []
        node_type_list = []

        for frame in episode:
            x = frame['player'].x.to(device)
            edge_index = frame['player', 'teammate', 'player'].edge_index.to(device)
            edge_attr = frame['player', 'teammate', 'player'].edge_attr.to(device)
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.unsqueeze(-1)
            global_feat = frame['global'].x[:, 0:-3].to(device)
            if traditional_features:
                traditional_feat = frame['traditional'].x.to(device)
                traditional_x_list.append(traditional_feat)
            ma = frame['match_action'].x.to(device)  # float tensor
            action_type_id = ma[:, 0].long().clamp(min=0, max=self.action_emb.num_embeddings - 1)  # [1]
            action_other = ma[:, 1:].float()  # [1,9]
            action_type_emb = self.action_emb(action_type_id)  # [1, action_emb_dim]
            action_feat = torch.cat([action_type_emb, action_other], dim=-1).view(-1)  # [action_emb_dim+9]
            action_feat = self.action_mlp(action_feat)  # [action_emb_dim]
            action_x_list.append(action_feat)

            if ('event_mask' in frame) and hasattr(frame['event_mask'], 'x'):
                em = frame['event_mask'].x.view(-1).to(device)
            else:
                em = (ma[:, -1].long().view(-1) != 0).long()
            event_mask_list.append(em)

            if counter_features:
                counter_feat = frame['global'].x[:, -3:].to(device)
                counter_feat = self.episode_context_mlp(counter_feat).view(-1)
                counter_x_list.append(counter_feat)

            for gnn in self.gnn_layers:
                # x, global_feat = gnn(x, edge_index, edge_attr, global_feat)
                x = gnn(x, edge_index, edge_attr, global_feat)
            if node_level:
                x_list.append(x)
            else:
                node_type = frame['player'].poss.to(device).long()
                role_id = frame['player'].position.to(device).long()
                pooled, attn_weights = self.pool(x, node_type, role_id, return_attention=True)
                x_list.append(pooled)
                # xg = torch.cat([pooled, global_feat.view(-1)], dim=-1)
                # x = self.fusion_mlp(xg)
                attention_weights_list.append(attn_weights.cpu().numpy())
                # x_list.append(x)
                node_type_list.append(node_type)

        if node_level:
            frame_features = torch.stack(x_list, dim=0)  # [T, N, D]
            frame_features = self.pos_encoder(frame_features)
            frame_features = frame_features.permute(1, 0, 2)  # [N, T, D] (batch_first)

            T_full = frame_features.size(1)
            frame_action_features = torch.stack(action_x_list, dim=0)[:T_full]  # [T, E]
            event_mask = torch.cat(event_mask_list, dim=0)[:T_full].to(frame_features.device)  # [T]

            e_bt = frame_action_features.unsqueeze(0).expand(frame_features.size(0), -1, -1)  # [N, T, E]
            m_bt = event_mask.unsqueeze(0).expand(frame_features.size(0), -1).float()  # [N, T]
            key_padding_mask = torch.zeros((frame_features.size(0), T_full), dtype=torch.bool,
                                           device=frame_features.device)

            frame_features = self.event_gate(frame_features, e_bt, m_bt)
            transformer_out = self.transformer(
                frame_features, e_bt, m_bt,
                key_padding_mask=key_padding_mask,
                return_attn=False
            )  # [N, T, D]
            transformer_out = self.norm(self.dropout(transformer_out))
            transformer_out = transformer_out.permute(1, 0, 2)  # [T, N, D]

            if sequence_level:
                transformer_out, frame_weights = self.tpool(transformer_out)

            frame_logit = self.classifier(transformer_out)
            preds = (torch.sigmoid(frame_logit) > 0.5).long().squeeze(-1) if self.num_classes == 2 else torch.argmax(
                frame_logit, dim=-1)
            node_probs = torch.sigmoid(frame_logit) if self.num_classes == 2 else F.softmax(frame_logit, dim=-1)
            return frame_logit, preds, node_probs, transformer_out

        frame_features_full = torch.stack(x_list, dim=0)  # [T_full, D]
        frame_action_features = torch.stack(action_x_list, dim=0)

        if traditional_features and counter_features:
            frame_traditional_features = torch.stack(traditional_x_list, dim=0)
            frame_traditional_features = frame_traditional_features.squeeze(1)
            frame_counter_features = torch.stack(counter_x_list, dim=0)
            frame_features_full = torch.cat(
                [frame_features_full, frame_action_features, frame_counter_features, frame_traditional_features],
                dim=-1
            )
            frame_features_full = self.transformer_input_proj_with_traditional_counter(frame_features_full)
        elif traditional_features:
            frame_traditional_features = torch.stack(traditional_x_list, dim=0)
            frame_traditional_features = frame_traditional_features.squeeze(1)
            frame_features_full = torch.cat(
                [frame_features_full, frame_action_features, frame_traditional_features],
                dim=-1
            )
            frame_features_full = self.transformer_input_proj_with_traditional(frame_features_full)
        elif counter_features:
            frame_counter_features = torch.stack(counter_x_list, dim=0)
            frame_features_full = torch.cat(
                [frame_features_full, frame_action_features, frame_counter_features],
                dim=-1
            )
            frame_features_full = self.transformer_input_proj_with_counter(frame_features_full)
        else:
            frame_features_full = torch.cat(
                [frame_features_full, frame_action_features],
                dim=-1
            )
            frame_features_full = self.transformer_input_proj(frame_features_full)

        if return_GNN:
            return frame_features_full

        T_full = frame_features_full.size(0)

        indices = torch.arange(0, T_full, device=frame_features_full.device)
        frame_features = frame_features_full  # [T_full, D]

        frame_features = self.pos_encoder(frame_features)  # [T_full, D]

        # batch_first
        x_bt = frame_features.unsqueeze(0)  # [1,T,D]
        e_bt = frame_action_features.unsqueeze(0)  # [1,T,E]
        event_mask = torch.cat(event_mask_list, dim=0)[:T_full].to(frame_features.device)  # [T]
        m_bt = event_mask.unsqueeze(0).float()  # [1,T]
        key_padding_mask = torch.zeros((1, T_full), dtype=torch.bool, device=frame_features.device)  # 无padding

        # gate
        x_bt = self.event_gate(x_bt, e_bt, m_bt)

        if return_attn:
            transformer_out_bt, attn_all = self.transformer(
                x_bt, e_bt, m_bt,
                key_padding_mask=key_padding_mask,
                return_attn=True
            )
            attn = attn_all[0]  # [1,H,T,T] (只有一层时)
        else:
            transformer_out_bt = self.transformer(
                x_bt, e_bt, m_bt,
                key_padding_mask=key_padding_mask,
                return_attn=False
            )
            attn = None

        transformer_out = transformer_out_bt.squeeze(0)  # [T,D]
        transformer_out = self.norm(self.dropout(transformer_out))
        if return_transformer:
            return transformer_out

        if sequence_level:
            seq_repr, frame_weights = self.tpool(transformer_out)
            frame_weights_list.append(frame_weights)
            frame_logit = self.classifier(seq_repr)
            preds = (torch.sigmoid(frame_logit) > 0.5).long().squeeze(-1) if self.num_classes == 2 \
                else torch.argmax(frame_logit, dim=-1)
            if return_attn:
                return frame_logit, preds, seq_repr, attention_weights_list, frame_weights_list, attn, event_mask
            return frame_logit, preds, seq_repr, attention_weights_list, frame_weights_list


        else:
            _, frame_weights = self.tpool(transformer_out)
            frame_weights_list.append(frame_weights)

            frame_logit = self.classifier(transformer_out)  # [T_sparse, C]
            preds = (torch.sigmoid(frame_logit) > 0.5).long().squeeze(-1) if self.num_classes == 2 \
                else torch.argmax(frame_logit, dim=-1)
            if return_attn:
                return frame_logit, preds, transformer_out, attention_weights_list, frame_weights_list, attn, event_mask
            return frame_logit, preds, transformer_out, attention_weights_list, frame_weights_list
