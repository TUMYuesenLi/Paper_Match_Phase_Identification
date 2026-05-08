import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import math


class TemporalAttentionPool(nn.Module):
    def __init__(self, dim, hidden=128):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1)
        )
    def forward(self, x):
        if x.dim() == 2:        # [T, D]
            w = torch.softmax(self.attn(x).squeeze(-1), dim=0)   # [T]
            return (w.unsqueeze(-1) * x).sum(dim=0), w            # [D], [T]
        elif x.dim() == 3:      # [T, N, D]
            T, N, D = x.shape
            w = torch.softmax(self.attn(x).squeeze(-1), dim=0)   # [T, N]
            return (w.unsqueeze(-1) * x).sum(dim=0), w            # [N, D], [T, N]
        else:
            raise ValueError("Unexpected shape")


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

        # padding key 不允许被 attend
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


class SoccerTransformer(nn.Module):
    def __init__(self, global_dim, d_model=256, action_dim=18, action_emb_dim=8
                 , trans_layers=2, nhead=8, dim_feedforward=512,
                 dropout=0.1, num_classes=3):
        super().__init__()

        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )

        self.gate_layer = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Sigmoid()
        )

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
        self.num_classes = num_classes
        hidden_dim = d_model // 2

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
        #
        self.transformer_input_proj = nn.Linear(global_dim + action_emb_dim, d_model)

        self.action_emb = nn.Embedding(action_dim, action_emb_dim)
        self.action_mlp = nn.Sequential(
            nn.Linear(action_emb_dim + 10, action_emb_dim),
            nn.ReLU(),
            nn.Linear(action_emb_dim, action_emb_dim),
        )

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

    def forward(self, batch_episodes, masks, y_key, device):
        frame_predictions = []
        for i, episode in enumerate(batch_episodes):
            x_list = []
            action_x_list = []
            event_mask_list = []
            for frame in episode:
                x = frame['traditional'].x.to(device)
                x_list.append(x)
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

            frame_features = torch.stack(x_list, dim=0).squeeze(1)  # [T, D]
            frame_action_features = torch.stack(action_x_list, dim=0)
            frame_features = torch.cat([frame_features, frame_action_features], dim=-1)

            frame_features = self.transformer_input_proj(frame_features)
            frame_features = self.pos_encoder(frame_features)

            # transformer [T, D] → [T, D]
            T = frame_features.size(0)
            x_in = frame_features.unsqueeze(1)  # [T, 1, D]  (S, N, E) batch_first=False

            mask_i = masks[i, :T].to(x_in.device)
            valid_len = int(mask_i.sum().item())
            x_bt = frame_features.unsqueeze(0)  # [1, T, D]
            e_bt = frame_action_features.unsqueeze(0)  # [1, T, E]

            event_mask = torch.cat(event_mask_list, dim=0)[:T].to(frame_features.device)  # [T]
            m_bt = event_mask.unsqueeze(0).float()  # [1, T]

            key_padding_mask = (~mask_i.bool()).unsqueeze(0)  # [1, T]

            x_bt = self.event_gate(x_bt, e_bt, m_bt)

            if valid_len == 0:
                transformer_out_bt = torch.zeros_like(frame_features)  # [T, D]
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

            frame_logit = self.classifier(transformer_out)

            frame_predictions.append(frame_logit)
        return pad_sequence(frame_predictions, batch_first=True)

    def predict_episode(self, episode, device):
        x_list = []
        action_x_list = []
        attention_weights_list = []
        frame_weights_list = []
        node_type_list = []
        event_mask_list=[]
        for frame in episode:
            x = frame['traditional'].x.to(device)
            x_list.append(x)
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
        frame_features = torch.stack(x_list, dim=0).squeeze(1)  # [T, D]
        frame_action_features = torch.stack(action_x_list, dim=0)
        frame_features = torch.cat([frame_features, frame_action_features], dim=-1)
        frame_features = self.transformer_input_proj(frame_features)
        frame_features = self.pos_encoder(frame_features)
        T = frame_features.size(0)
        x_bt = frame_features.unsqueeze(0)                    # [1,T,D]
        e_bt = frame_action_features.unsqueeze(0)             # [1,T,E]
        event_mask = torch.cat(event_mask_list, dim=0)[:T].to(frame_features.device)  # [T]
        m_bt = event_mask.unsqueeze(0).float()                # [1,T]
        key_padding_mask = torch.zeros((1, T), dtype=torch.bool, device=frame_features.device)

        # gate
        x_bt = self.event_gate(x_bt, e_bt, m_bt)
        transformer_out_bt = self.transformer(
            x_bt, e_bt, m_bt,
            key_padding_mask=key_padding_mask,
            return_attn=False
        )
        attn = None
        transformer_out = transformer_out_bt.squeeze(0)        # [T,D]

        # transformer  [T, D] → [T, D]
        transformer_out = self.norm(self.dropout(transformer_out))
        frame_logit = self.classifier(transformer_out)
        preds = (torch.sigmoid(frame_logit) > 0.5).long().squeeze(-1) if self.num_classes == 2 else torch.argmax(frame_logit, dim=-1)
        return frame_logit, preds, transformer_out, attention_weights_list, frame_weights_list

