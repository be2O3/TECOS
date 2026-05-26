import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicNonOverlapWindowNet(nn.Module):
    """
    End-to-end window head for SLADE.

    Input:
        x: [B, 2, L]
           channel 0 = source recovery score
           channel 1 = source drift score

    Behavior:
        1) point-level temporal encoding with Conv1d backbone
        2) non-overlapping base blocks of size Wb
        3) optional history-only block context with a fixed history length
        4) point logits + block logits

    Notes:
        - block context is causal/history-only
        - incomplete tail shorter than one base block is discarded in the block branch
        - point branch still outputs length-L logits
        - history_len means: for block j, use recent history_len past blocks
          [j-history_len, ..., j-1]
        - inside that fixed history span, the model learns attention weights over the
          past blocks
    """

    def __init__(
        self,
        base_window_size=4,
        history_len=3,
        hidden=64,
        gate_hidden=64,
        kernel_size=5,
        num_layers=2,
        dropout=0.1,
        temp=1.0,
        use_block_context=True,
        in_ch=2,
        pooling_type="softmax",
        context_direction="bidirectional",
    ):
        super().__init__()
        self.Wb = int(base_window_size)
        self.history_len = int(history_len)
        self.hidden = int(hidden)
        self.temp = float(temp)
        self.use_block_context = bool(use_block_context)
        self.in_ch = int(in_ch)
        self.pooling_type = str(pooling_type).lower()
        self.context_direction = str(context_direction).lower()
        self.kernel_size = int(kernel_size)

        if self.Wb < 1:
            raise ValueError(f"base_window_size must be >= 1, got {self.Wb}")
        if self.history_len < 1:
            raise ValueError(f"history_len must be >= 1, got {self.history_len}")
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError(f"kernel_size should be positive odd, got {kernel_size}")
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")
        if self.in_ch != 2:
            raise ValueError(f"This version expects in_ch=2 ([rec, drift]), got {self.in_ch}")
        if self.pooling_type not in {"softmax", "mean", "max", "meanmax"}:
            raise ValueError(
                "pooling_type must be one of {'softmax','mean','max','meanmax'}, "
                f"got {self.pooling_type}"
            )
        if self.context_direction not in {"bidirectional", "past", "future"}:
            raise ValueError(
                "context_direction must be one of {'bidirectional','past','future'}, "
                f"got {self.context_direction}"
            )

        self.conv_layers = nn.ModuleList([
            nn.Conv1d(self.in_ch, hidden, kernel_size=kernel_size, padding=0)
        ])
        for _ in range(num_layers - 1):
            self.conv_layers.append(nn.Conv1d(hidden, hidden, kernel_size=kernel_size, padding=0))
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.point_head = nn.Conv1d(hidden, 1, kernel_size=1)

        # Used only when pooling_type == "softmax". The gate is an
        # intra-window attention scorer conditioned on point evidence,
        # window context, and relative position inside the window.
        self.point_pos_dim = min(16, hidden)
        self.point_pos_emb = nn.Embedding(self.Wb, self.point_pos_dim)
        self.point_gate_net = nn.Sequential(
            nn.Linear(hidden * 3 + self.in_ch + self.point_pos_dim, gate_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(gate_hidden, 1),
        )

        # history attention over a fixed-length causal block window
        # input = [current block, candidate history block, current - candidate]
        self.history_gate_net = nn.Sequential(
            nn.Linear(hidden * 3, gate_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(gate_hidden, 1),
        )

        block_in_dim = hidden * 3 if self.use_block_context else hidden
        self.block_head = nn.Sequential(
            nn.Linear(block_in_dim, gate_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(gate_hidden, 1),
        )

    def _masked_conv1d(self, conv, h):
        pad = self.kernel_size // 2
        h = F.pad(h, (pad, pad))

        if self.context_direction == "bidirectional":
            weight = conv.weight
        else:
            mask = torch.zeros(
                1,
                1,
                self.kernel_size,
                device=conv.weight.device,
                dtype=conv.weight.dtype,
            )
            if self.context_direction == "past":
                mask[:, :, :pad + 1] = 1
            elif self.context_direction == "future":
                mask[:, :, pad:] = 1
            else:
                raise ValueError(f"Unknown context_direction={self.context_direction}")
            weight = conv.weight * mask

        return F.conv1d(
            h,
            weight,
            conv.bias,
            stride=conv.stride,
            padding=0,
            dilation=conv.dilation,
            groups=conv.groups,
        )

    def _encode_temporal(self, h):
        for conv in self.conv_layers:
            h = self._masked_conv1d(conv, h)
            h = self.activation(h)
            h = self.dropout(h)
        return h

    def _build_raw_features(self, x, lengths=None):
        """
        x: [B, 2, L]
        returns: [B, 2, L]
        """
        if x.dim() != 3 or x.size(1) != self.in_ch:
            raise ValueError(f"Expected x shape [B,{self.in_ch},L], got {x.shape}")

        if lengths is None:
            return x

        B, C, L = x.shape
        device = x.device
        idx = torch.arange(L, device=device)[None, :]
        mask = (idx < lengths[:, None]).float().unsqueeze(1)  # [B,1,L]
        return x * mask

    def _make_base_blocks(self, h, raw_feats, point_valid_mask=None):
        """
        h: [B, H, L]
        raw_feats: [B, F, L]

        returns:
            h_blk:    [B, Nblk, Wb, H]
            raw_blk:  [B, Nblk, Wb, F]
            mask_blk: [B, Nblk, Wb] bool, or None
        """
        Wb = self.Wb
        B, H, L = h.shape
        _, Fdim, _ = raw_feats.shape

        usable = (L // Wb) * Wb
        h_use = h[:, :, :usable]
        r_use = raw_feats[:, :, :usable]
        nblk = usable // Wb

        h_blk = h_use.view(B, H, nblk, Wb).permute(0, 2, 3, 1).contiguous()
        raw_blk = r_use.view(B, Fdim, nblk, Wb).permute(0, 2, 3, 1).contiguous()

        if point_valid_mask is None:
            mask_blk = None
        else:
            m_use = point_valid_mask[:, :usable].to(device=h.device, dtype=torch.bool)
            mask_blk = m_use.view(B, nblk, Wb).contiguous()

        return h_blk, raw_blk, mask_blk

    def _encode_base_blocks(self, h_blk, raw_blk, mask_blk=None):
        """
        h_blk:   [B, Nblk, Wb, H]
        raw_blk: [B, Nblk, Wb, F]

        mask_blk:
            [B,Nblk,Wb] bool. False positions are padding tokens and are
            excluded from every pooling mode.

        pooling_type:
            - softmax: intra-window attentive pooling with point features,
                       window context, and relative position
            - mean:    masked mean pooling
            - max:     masked max pooling
            - meanmax: 0.5 * masked mean + 0.5 * masked max
        """
        if mask_blk is None:
            mask_f = None
            valid_count = None
            valid_any = None
        else:
            mask_f = mask_blk.to(dtype=h_blk.dtype).unsqueeze(-1)  # [B,Nblk,Wb,1]
            valid_count_raw = mask_f.sum(dim=2)                    # [B,Nblk,1]
            valid_any = valid_count_raw > 0
            valid_count = valid_count_raw.clamp_min(1.0)           # [B,Nblk,1]

        if self.pooling_type == "mean":
            point_gates = None
            if mask_f is None:
                blk_repr = h_blk.mean(dim=2)
            else:
                blk_repr = (h_blk * mask_f).sum(dim=2) / valid_count
            return blk_repr, point_gates

        if self.pooling_type == "max":
            point_gates = None
            if mask_f is None:
                blk_repr = h_blk.max(dim=2).values
            else:
                h_masked = h_blk.masked_fill(~mask_blk.unsqueeze(-1), -1e9)
                blk_repr = h_masked.max(dim=2).values
                blk_repr = torch.where(valid_any, blk_repr, torch.zeros_like(blk_repr))
            return blk_repr, point_gates

        if self.pooling_type == "meanmax":
            point_gates = None
            if mask_f is None:
                h_mean = h_blk.mean(dim=2)
                h_max = h_blk.max(dim=2).values
            else:
                h_mean = (h_blk * mask_f).sum(dim=2) / valid_count
                h_masked = h_blk.masked_fill(~mask_blk.unsqueeze(-1), -1e9)
                h_max = h_masked.max(dim=2).values
                h_max = torch.where(valid_any, h_max, torch.zeros_like(h_max))
            blk_repr = 0.5 * h_mean + 0.5 * h_max
            return blk_repr, point_gates

        B, Nblk, Wb, H = h_blk.shape

        if mask_f is None:
            win_context = h_blk.mean(dim=2, keepdim=True)       # [B,Nblk,1,H]
        else:
            win_context = (h_blk * mask_f).sum(dim=2, keepdim=True) / valid_count.unsqueeze(2)

        win_context = win_context.expand(-1, -1, Wb, -1)        # [B,Nblk,Wb,H]
        rel_to_context = h_blk - win_context

        pos_idx = torch.arange(Wb, device=h_blk.device)
        pos_emb = self.point_pos_emb(pos_idx).view(1, 1, Wb, self.point_pos_dim)
        pos_emb = pos_emb.expand(B, Nblk, -1, -1)               # [B,Nblk,Wb,P]

        gate_in = torch.cat(
            [h_blk, raw_blk, win_context, rel_to_context, pos_emb],
            dim=-1,
        )                                                       # [B,Nblk,Wb,3H+F+P]
        gate_logits = self.point_gate_net(gate_in).squeeze(-1)  # [B,Nblk,Wb]

        gate_logits = gate_logits / max(self.temp, 1e-6)
        if mask_blk is not None:
            gate_logits = gate_logits.masked_fill(~mask_blk, -1e9)

        point_gates = torch.softmax(gate_logits, dim=-1)
        if mask_blk is not None:
            point_gates = point_gates * mask_blk.to(dtype=point_gates.dtype)
            point_gates = point_gates / point_gates.sum(dim=-1, keepdim=True).clamp_min(1e-8)

        blk_repr = (point_gates.unsqueeze(-1) * h_blk).sum(dim=2)
        return blk_repr, point_gates

    def _make_history_windows(self, blk_repr):
        """
        blk_repr: [B, Nblk, H]

        returns:
            hist_win: [B, Nblk, history_len, H]

        For block j, hist_win[:, j] contains recent history_len past blocks
        [j-history_len, ..., j-1]. Missing left history is handled by left
        replicate padding, so early blocks still have a full-length history window.
        """
        B, Nblk, H = blk_repr.shape
        hist_len = self.history_len

        blk_t = blk_repr.transpose(1, 2).contiguous()           # [B,H,Nblk]
        blk_pad = F.pad(blk_t, (hist_len, 0), mode="replicate")
        hist_win = blk_pad.unfold(dimension=2, size=hist_len, step=1)[:, :, :Nblk, :]
        # [B,H,Nblk,hist_len], where window j corresponds to [j-history_len, ..., j-1]
        hist_win = hist_win.permute(0, 2, 3, 1).contiguous()           # [B,Nblk,hist_len,H]
        return hist_win

    def _history_context(self, blk_repr):
        """
        blk_repr: [B, Nblk, H]

        Fixed-length history-only / causal context.
        The model does NOT choose among multiple nested scales anymore.
        Instead, it always looks at recent history_len past blocks and
        learns attention weights inside that fixed history span.
        """
        hist_win = self._make_history_windows(blk_repr)                 # [B,Nblk,K,H]
        cur = blk_repr.unsqueeze(2).expand_as(hist_win)                 # [B,Nblk,K,H]

        hist_gate_in = torch.cat([cur, hist_win, cur - hist_win], dim=-1)
        hist_logits = self.history_gate_net(hist_gate_in).squeeze(-1)   # [B,Nblk,K]
        history_weights = torch.softmax(hist_logits / max(self.temp, 1e-6), dim=-1)

        dyn_context = (history_weights.unsqueeze(-1) * hist_win).sum(dim=2)
        max_context = hist_win.max(dim=2).values
        return dyn_context, max_context, history_weights, hist_win

    def forward(self, x, lengths=None, point_valid_mask=None, return_attn=False):
        """
        x: [B, 2, L]

        returns:
            point_logits: [B, L]
            block_logits: [B, Nblk]
        """
        if x.dim() != 3:
            raise ValueError(f"Expected x shape [B,{self.in_ch},L], got {x.shape}")
        B, C, L = x.shape
        if C != self.in_ch:
            raise ValueError(f"Expected x shape [B,{self.in_ch},L], got {x.shape}")
        if L < self.Wb:
            raise ValueError(f"Sequence length L={L} < base_window_size Wb={self.Wb}")

        if point_valid_mask is None and lengths is not None:
            idx = torch.arange(L, device=x.device)[None, :]
            point_valid_mask = idx < lengths[:, None]
        elif point_valid_mask is not None:
            if point_valid_mask.shape != (B, L):
                raise ValueError(
                    f"Expected point_valid_mask shape [{B},{L}], got {point_valid_mask.shape}"
                )
            point_valid_mask = point_valid_mask.to(device=x.device, dtype=torch.bool)

        raw_feats = self._build_raw_features(x, lengths=lengths)  # [B,2,L]
        h = self._encode_temporal(raw_feats)                      # [B,H,L]
        point_logits = self.point_head(h).squeeze(1)              # [B,L]

        h_blk, raw_blk, mask_blk = self._make_base_blocks(
            h,
            raw_feats,
            point_valid_mask=point_valid_mask,
        )
        blk_repr, point_gates = self._encode_base_blocks(h_blk, raw_blk, mask_blk=mask_blk)

        if self.use_block_context:
            dyn_context, max_context, history_weights, history_blocks = self._history_context(blk_repr)
            final_repr = torch.cat([blk_repr, dyn_context, max_context], dim=-1)
        else:
            dyn_context = None
            max_context = None
            history_weights = None
            history_blocks = None
            final_repr = blk_repr

        block_logits = self.block_head(final_repr).squeeze(-1)    # [B,Nblk]

        if return_attn:
            extras = {
                "point_gates": point_gates,
                "history_weights": history_weights,
                "history_blocks": history_blocks,
                "block_repr": blk_repr,
                "dyn_context": dyn_context,
                "max_context": max_context,
                "block_point_mask": mask_blk,
            }
            return point_logits, block_logits, extras

        return point_logits, block_logits


def compute_block_pos_weight(src_labels_by_node: dict, base_window_size=4, cap=50.0):
    """
    Compute pos_weight for block/window BCE loss.
    Window label is OR-pooled within each non-overlapping base block.
    """
    Wb = int(base_window_size)
    pos = 0
    neg = 0
    for ys in src_labels_by_node.values():
        if len(ys) < Wb:
            continue
        y = torch.tensor(ys, dtype=torch.float32).view(1, 1, -1)
        y_blk = F.max_pool1d(y, kernel_size=Wb, stride=Wb).view(-1)
        pos += int((y_blk > 0.5).sum().item())
        neg += int((y_blk <= 0.5).sum().item())
    if pos == 0:
        return None
    return float(min(neg / max(pos, 1), cap))


def compute_point_pos_weight(src_labels_by_node: dict, cap=50.0):
    """
    Compute pos_weight for optional point-level BCE loss.
    """
    pos = 0
    neg = 0
    for ys in src_labels_by_node.values():
        if len(ys) == 0:
            continue
        y = torch.tensor(ys, dtype=torch.float32).view(-1)
        pos += int((y > 0.5).sum().item())
        neg += int((y <= 0.5).sum().item())
    if pos == 0:
        return None
    return float(min(neg / max(pos, 1), cap))


def get_valid_block_mask(lengths, nblk, window_size, device):
    """
    Build mask for valid non-overlapping blocks.

    lengths: [B]
    returns: [B, nblk] boolean tensor
    """
    valid_nblk = torch.div(lengths, int(window_size), rounding_mode="floor")
    idx = torch.arange(nblk, device=device)[None, :]
    return idx < valid_nblk[:, None]


__all__ = [
    "DynamicNonOverlapWindowNet",
    "compute_block_pos_weight",
    "compute_point_pos_weight",
    "get_valid_block_mask",
]
