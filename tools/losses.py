import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
warnings.filterwarnings('ignore')


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean',
                 ignore_index=None, debug_checks=True,
                 time_pool='logsumexp', lse_normalize=True, eps=1e-8):
        """
        alpha: 类别权重 [C] 或 None
        gamma: 聚焦参数
        reduction: 'none' | 'mean' | 'sum'
        ignore_index: 被忽略的标签值（如 -100）
        debug_checks: 开启严格断言，便于定位数据问题
        time_pool: 序列级聚合：'mean' | 'max' | 'logsumexp' | 'noisy_or'
        lse_normalize: logsumexp 后是否减去 log(有效长度)，近似“平均”的 logits
        eps: 数值稳定用
        """
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
        """
        - 帧级: logits [B,T,C], targets [B,T], mask [B,T], mode='frame' 或 'auto'
        - 序列级(已聚合): logits [B,C], targets [B], mode='sequence'
        - 序列级(在loss内聚合): logits [B,T,C], targets [B], mask [B,T], mode='sequence'
        """
        if logits.dim() not in (2, 3):
            raise ValueError(f"Expect logits dim 2 or 3, got {logits.shape}")

        B = logits.shape[0]
        C = logits.size(-1)

        # 自动判断模式
        if mode == 'auto':
            if logits.dim() == 3 and targets.dim() == 2:
                mode = 'frame'
            elif (logits.dim() == 3 and targets.dim() == 1) or (logits.dim() == 2 and targets.dim() == 1):
                mode = 'sequence'
            else:
                raise ValueError(
                    f"Cannot auto-resolve mode for logits{tuple(logits.shape)} targets{tuple(targets.shape)}")

        if mode == 'frame':
            # ----- 帧级 focal：按元素应用 mask/ignore，再展平 -----
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

            # Debug 检查（只对有效元素）
            if self.debug_checks and eff.any():
                t_eff = t[eff]
                assert t_eff.dtype in (torch.int64, torch.long)
                assert int(t_eff.min()) >= 0, f"min target={int(t_eff.min())}"
                assert int(t_eff.max()) < C, f"max target={int(t_eff.max())} >= C({C})"
                if self.alpha is not None:
                    assert self.alpha.numel() == C, f"alpha size={self.alpha.numel()} vs C={C}"
                assert torch.isfinite(x).all(), "logits contain NaN/Inf"

            # 安全 gather：对无效位置用 0 占位，随后用 eff 清零损失
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
            # ----- 序列级 focal -----
            if logits.dim() == 2:
                # 已是序列级 [B, C]
                assert targets.dim() == 1 and targets.shape[0] == B
                t = targets.long()
                if mask is None:
                    eff_seq = torch.ones_like(t, dtype=torch.bool, device=t.device)
                else:
                    # 允许 mask 是 [B] 或 [B,1]（非必须）
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
                # 在 loss 内进行时间聚合：logits [B,T,C], targets [B], mask [B,T]
                assert targets.dim() == 1 and (mask is None or mask.dim() == 2)
                B, T, C = logits.shape
                t = targets.long()

                if mask is None:
                    m = torch.ones((B, T), dtype=torch.bool, device=logits.device)
                else:
                    m = mask.bool().to(logits.device)
                    assert m.shape == (B, T)

                eff_seq = m.any(dim=1)  # 至少一个有效帧
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

                # ---- 时间聚合 ----
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
                    # 用概率聚合“至少一个帧为该类”：p_true_seq = 1 - ∏(1 - p_true_frame)
                    p = F.softmax(x, dim=-1)  # [B_eff,T,C]
                    p = torch.clamp(p, self.eps, 1 - self.eps)
                    # 取真类的逐帧概率
                    idx = t_eff.view(-1, 1, 1).expand(-1, p.size(1), 1)  # [B_eff,T,1]
                    p_true = p.gather(dim=-1, index=idx).squeeze(-1)  # [B_eff,T]
                    # 只在有效帧上连乘
                    one_minus = 1.0 - p_true
                    one_minus = torch.where(m_eff, one_minus, torch.ones_like(one_minus))
                    prod = one_minus.prod(dim=1)  # [B_eff]
                    pt = 1.0 - prod  # p_true_seq
                    pt = torch.clamp(pt, self.eps, 1 - self.eps)
                    log_pt = torch.log(pt)
                else:
                    raise ValueError(f"Unknown time_pool: {self.time_pool}")

                # α 权重
                if self.alpha is not None:
                    alpha_t = self.alpha.to(logits.device)[t_eff]
                else:
                    alpha_t = 1.0

                loss = -alpha_t * (1.0 - pt).pow(self.gamma) * log_pt  # [B_eff]
                # 对序列级 reduction（eff_seq 只在有效序列为 True）
                return self._reduce(loss, eff_seq[eff_seq].to(loss.dtype))
        else:
            raise ValueError(f"Unknown mode: {mode}")


class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean',
                 ignore_index=None, time_pool='noisy_or', mode='auto', eps=1e-8):
        """
        alpha: 标量(正类权重) 或 [alpha_neg, alpha_pos]
        gamma: 聚焦参数
        reduction: 'none' | 'mean' | 'sum'
        ignore_index: 可选，被忽略的标签值（如 -100）
        time_pool: 序列级聚合方式：'mean' | 'max' | 'logsumexp' | 'noisy_or'
        mode: 'auto' | 'frame' | 'sequence'
        eps: 数值稳定用
        """
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.time_pool = time_pool
        self.mode = mode
        # 统一 alpha
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
        """
        支持的输入：
        - 帧级：logits [B,T] 或 [B,T,1]; targets [B,T]; mask [B,T]
        - 序列级（在loss里聚合）：logits [B,T]/[B,T,1]; targets [B]; mask [B,T]
        - 已是序列级：logits [B]/[B,1]; targets [B]; mask 可省略或 [B](1/0)
        """
        if logits.dim() == 3 and logits.size(-1) == 1:
            logits = logits.squeeze(-1)  # -> [B,T]
        if logits.dim() == 2 and logits.size(1) == 1:
            logits = logits.squeeze(1)   # -> [B]

        if mode is None or mode == 'auto':
            # 自动判断：targets [B] → sequence；targets [B,T] → frame
            mode = 'sequence' if targets.dim() == 1 else 'frame'

        if mode == 'frame':
            # 帧级 focal（BCE with logits），mask/ignore_index 在时间维逐元素应用
            assert logits.dim() == 2, f"frame mode 期望 logits [B,T]，got {tuple(logits.shape)}"
            assert targets.dim() == 2, f"frame mode 期望 targets [B,T]，got {tuple(targets.shape)}"
            B, T = logits.shape
            # print(targets)
            t = targets.reshape(-1).long()
            x = logits.reshape(-1)
            # print(logits)

            if mask is None:
                eff = torch.ones_like(t, dtype=torch.bool)
            else:
                eff = mask.reshape(-1).bool()

            if self.ignore_index is not None:
                eff = eff & (t != self.ignore_index)

            if eff.any():
                t_eff = t[eff]
                assert int(t_eff.min()) >= 0 and int(t_eff.max()) <= 1, \
                    f"frame mode 需要标签在{{0,1}}，但找到 [{int(t_eff.min())}, {int(t_eff.max())}]"

            t_float = t.float()
            bce = F.binary_cross_entropy_with_logits(x, t_float, reduction='none')  # [B*T]
            p = torch.sigmoid(x)
            # print(p)
            # pt = torch.where(t_float > 0.5, p, 1 - p)
            p = p.clamp(min=self.eps, max=1.0 - self.eps)
            # pt = p * targets + (1.0 - p) * (1.0 - targets)  # [N]
            pt = torch.where(t_float > 0.5, p, 1 - p)
            # print(pt)
            focal = (1.0 - pt).pow(self.gamma)
            alpha = self._alpha_factor(t_float)
            loss = alpha * focal * bce
            loss = loss * eff.to(loss.dtype)
            # print(loss)
            return self._reduce(loss, eff)

        elif mode == 'sequence':
            # 在时间维先聚合，再做序列级二分类 focal
            # 两种输入：
            # a) logits [B,T], targets [B], mask [B,T]
            # b) logits [B],   targets [B]（此时等同已序列级）
            if logits.dim() == 1:
                # 已是序列级
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

            # 序列级（在loss里聚合）：logits [B,T]
            assert logits.dim() == 2 and targets.dim() == 1, \
                f"sequence mode 期望 logits [B,T], targets [B]，got logits{tuple(logits.shape)}, targets{tuple(targets.shape)}"
            B, T = logits.shape
            t = targets.long()
            t_float = t.float()

            if mask is None:
                m = torch.ones((B, T), dtype=torch.bool, device=logits.device)
            else:
                m = mask.bool().to(logits.device)
                assert m.shape == (B, T)

            # 忽略样本
            eff = m.any(dim=1)  # 有至少1个有效帧
            if self.ignore_index is not None:
                eff = eff & (t != self.ignore_index)

            # 仅对有效序列计算
            if not eff.any():
                return logits.new_tensor(0.0)

            x = logits[eff]     # [B_eff, T]
            m_eff = m[eff]      # [B_eff, T]
            t_eff = t_float[eff]# [B_eff]

            # ---- 时间聚合为序列级 ----
            if self.time_pool == 'mean':
                denom = m_eff.sum(dim=1).clamp_min(1).to(x.dtype)
                x_seq = (x * m_eff.to(x.dtype)).sum(dim=1) / denom          # [B_eff] (logit 均值)

                # BCE with logits
                bce = F.binary_cross_entropy_with_logits(x_seq, t_eff, reduction='none')
                p_seq = torch.sigmoid(x_seq)
                pt = torch.where(t_eff > 0.5, p_seq, 1 - p_seq)

            elif self.time_pool == 'max':
                # 只在有效帧上取最大 logit
                x_masked = x.masked_fill(~m_eff, float('-inf'))
                x_seq = x_masked.max(dim=1).values
                x_seq = torch.where(torch.isfinite(x_seq), x_seq, x.new_zeros(x_seq.shape))  # 若全无效，置0
                bce = F.binary_cross_entropy_with_logits(x_seq, t_eff, reduction='none')
                p_seq = torch.sigmoid(x_seq)
                pt = torch.where(t_eff > 0.5, p_seq, 1 - p_seq)

            elif self.time_pool == 'logsumexp':
                # 稳定的 softmax-like 聚合：logsumexp - log(有效长度)
                x_masked = x.masked_fill(~m_eff, float('-inf'))
                lse = torch.logsumexp(x_masked, dim=1)                       # [B_eff]
                valid_len = m_eff.sum(dim=1).clamp_min(1).to(x.dtype)
                x_seq = lse - valid_len.log()                                 # 相当于对概率做平均的 logits 近似
                bce = F.binary_cross_entropy_with_logits(x_seq, t_eff, reduction='none')
                p_seq = torch.sigmoid(x_seq)
                pt = torch.where(t_eff > 0.5, p_seq, 1 - p_seq)

            elif self.time_pool == 'noisy_or':
                # p_seq = 1 - ∏(1 - p_t)（仅在有效帧上）
                p = torch.sigmoid(x)                                          # [B_eff, T]
                p = torch.clamp(p, self.eps, 1 - self.eps)
                one_minus_p = 1.0 - p
                one_minus_p = torch.where(m_eff, one_minus_p, torch.ones_like(one_minus_p))
                prod = one_minus_p.prod(dim=1)                                # ∏(1-p_t)
                p_seq = 1.0 - prod                                            # [B_eff]
                p_seq = torch.clamp(p_seq, self.eps, 1 - self.eps)

                # 用概率版本的 BCE（非 logits）
                bce = -(t_eff * torch.log(p_seq) + (1.0 - t_eff) * torch.log(1.0 - p_seq))
                pt = torch.where(t_eff > 0.5, p_seq, 1 - p_seq)
            else:
                raise ValueError(f"Unknown time_pool: {self.time_pool}")

            focal = (1.0 - pt).pow(self.gamma)
            alpha = self._alpha_factor(t_eff)
            loss = alpha * focal * bce                                        # [B_eff]

            # reduction（这里的 eff 已经是序列级）
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
            # [1, num_classes]，扩展到 [B, num_classes]
            weight = self.class_weights.view(1, -1).to(pred.device)
            loss = loss * weight
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
