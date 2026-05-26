import math
import random

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.nn.utils.rnn import pad_sequence

def set_all_seeds(seed: int, deterministic: bool = True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def select_src_two_channel_tensor(args, src_rec_bound, src_drift_bound):
    """
    return [B,2]:
      col 0 = rec
      col 1 = drift
    """
    src_rec = src_rec_bound.reshape(-1)
    src_drift = src_drift_bound.reshape(-1)

    if args.only_rec_score or args.only_recovery_loss_score:
        src_drift = torch.zeros_like(src_drift)
    elif args.only_drift_score or args.only_drift_loss_score:
        src_rec = torch.zeros_like(src_rec)

    return torch.stack([src_rec, src_drift], dim=-1)  # [B,2]


def group_labels_by_source(sources_np, labels_np):
    out = {}
    for nid, lb in zip(sources_np, labels_np):
        nid = int(nid)
        out.setdefault(nid, []).append(float(lb))
    return out


def compute_block_pos_weight(epoch_src_labels: dict, base_window_size=4, cap=50.0):
    Wb = int(base_window_size)
    pos = 0
    neg = 0
    for ys in epoch_src_labels.values():
        if len(ys) < Wb:
            continue
        y = torch.tensor(ys, dtype=torch.float32).view(1, 1, -1)
        y_blk = F.max_pool1d(y, kernel_size=Wb, stride=Wb).view(-1)
        pos += int((y_blk > 0.5).sum().item())
        neg += int((y_blk <= 0.5).sum().item())
    if pos == 0:
        return None
    return float(min(neg / max(pos, 1), cap))


def compute_point_pos_weight(epoch_src_labels: dict, cap=50.0):
    pos = 0
    neg = 0
    for ys in epoch_src_labels.values():
        y = np.asarray(ys, dtype=np.float32).reshape(-1)
        if y.size == 0:
            continue
        pos += int((y > 0.5).sum())
        neg += int((y <= 0.5).sum())
    if pos == 0:
        return None
    return float(min(neg / max(pos, 1), cap))


def make_block_mask(lengths, nblk, Wb, device):
    block_lengths = torch.div(lengths, Wb, rounding_mode='floor')
    idx = torch.arange(nblk, device=device).unsqueeze(0)
    return idx < block_lengths.unsqueeze(1)


def safe_nanargmax(values):
    arr = np.asarray(values, dtype=np.float32)
    if arr.size == 0 or np.all(np.isnan(arr)):
        return None
    return int(np.nanargmax(arr))


def summarize_eval_win_coverage(seq_rec: dict, seq_drift: dict, seq_labels: dict, window_size: int):
    """
    Coverage summary aligned with partial-tail conv evaluation.
    Keep the last partial window if it contains at least one real point.
    """
    eval_node_count = 0
    eval_window_count = 0
    eval_pos_window_count = 0
    eval_neg_window_count = 0

    Wb = int(window_size)

    for nid, rec_seq in seq_rec.items():
        drift_seq = seq_drift.get(nid, None)
        y_seq = seq_labels.get(nid, None)
        if drift_seq is None or y_seq is None:
            continue

        L = min(len(rec_seq), len(drift_seq), len(y_seq))
        if L <= 0:
            continue

        target_len = max(Wb, int(math.ceil(L / Wb) * Wb))

        y_pad = np.zeros(target_len, dtype=np.float32)
        point_mask = np.zeros(target_len, dtype=np.float32)

        y_pad[:L] = np.asarray(y_seq[:L], dtype=np.float32)
        point_mask[:L] = 1.0

        y_blk = y_pad.reshape(-1, Wb).max(axis=1)
        blk_valid = point_mask.reshape(-1, Wb).max(axis=1) > 0

        y_blk_valid = y_blk[blk_valid]
        if y_blk_valid.size == 0:
            continue

        eval_node_count += 1
        eval_window_count += int(y_blk_valid.size)
        eval_pos_window_count += int((y_blk_valid > 0.5).sum())
        eval_neg_window_count += int((y_blk_valid <= 0.5).sum())

    return eval_node_count, eval_window_count, eval_pos_window_count, eval_neg_window_count



def detach_residual_buffers(score_buffer, label_buffer):
    for nid in list(score_buffer.keys()):
        score_buffer[nid] = [t.detach() for t in score_buffer[nid]]
    for nid in list(label_buffer.keys()):
        label_buffer[nid] = [t.detach() for t in label_buffer[nid]]


def append_and_collect_ready_chunks(
    score_buffer,
    label_buffer,
    node_id: int,
    score_t: torch.Tensor,
    label_t: torch.Tensor,
    chunk_len: int,
    step_len: int,
):
    """
    score_t: [2]
    """
    score_buffer.setdefault(node_id, []).append(score_t)
    label_buffer.setdefault(node_id, []).append(label_t)

    ready_x = []
    ready_y = []

    while len(score_buffer[node_id]) >= chunk_len:
        x_chunk = torch.stack(score_buffer[node_id][:chunk_len], dim=0)  # [chunk_len,2]
        y_chunk = torch.stack(label_buffer[node_id][:chunk_len], dim=0)  # [chunk_len]
        ready_x.append(x_chunk)
        ready_y.append(y_chunk)

        del score_buffer[node_id][:step_len]
        del label_buffer[node_id][:step_len]

    return ready_x, ready_y


def flush_tail_buffers_with_padding_token(
    score_buffer,
    label_buffer,
    window_size: int,
):
    """
    Keep residual tails by padding with dummy tokens to the nearest window multiple.

    For each node:
      - if residual length r == 0: skip
      - target_len = max(Wb, ceil(r / Wb) * Wb)
      - x is padded with dummy token [0, 0]
      - y is padded with 0
      - point_valid_mask marks real points only

    Returns:
      ready_x: list[Tensor[target_len, 2]]
      ready_y: list[Tensor[target_len]]
      ready_lengths: list[int]                      # original valid length r
      ready_point_masks: list[Tensor[target_len]]  # 1 for real, 0 for pad
      padded_node_ids: list[int]
      raw_tail_lengths: dict[nid] = r
    """
    Wb = int(window_size)

    ready_x = []
    ready_y = []
    ready_lengths = []
    ready_point_masks = []
    padded_node_ids = []
    raw_tail_lengths = {}

    for nid in sorted(score_buffer.keys()):
        xs = score_buffer.get(nid, [])
        ys = label_buffer.get(nid, [])

        if len(xs) == 0 or len(ys) == 0:
            continue
        if len(xs) != len(ys):
            raise ValueError(
                f"Residual buffer length mismatch for node {nid}: "
                f"len(xs)={len(xs)} vs len(ys)={len(ys)}"
            )

        r = len(xs)
        target_len = max(Wb, int(math.ceil(r / Wb) * Wb))

        x_real = torch.stack(xs, dim=0)   # [r, 2]
        y_real = torch.stack(ys, dim=0)   # [r]

        if target_len > r:
            x_pad = torch.zeros(
                target_len - r,
                x_real.size(-1),
                dtype=x_real.dtype,
                device=x_real.device,
            )
            y_pad = torch.zeros(
                target_len - r,
                dtype=y_real.dtype,
                device=y_real.device,
            )
            x_chunk = torch.cat([x_real, x_pad], dim=0)   # [target_len, 2]
            y_chunk = torch.cat([y_real, y_pad], dim=0)   # [target_len]
        else:
            x_chunk = x_real
            y_chunk = y_real

        point_valid_mask = torch.zeros(target_len, dtype=torch.float32, device=y_chunk.device)
        point_valid_mask[:r] = 1.0

        ready_x.append(x_chunk)
        ready_y.append(y_chunk)
        ready_lengths.append(int(r))
        ready_point_masks.append(point_valid_mask)
        padded_node_ids.append(int(nid))
        raw_tail_lengths[int(nid)] = int(r)

    return ready_x, ready_y, ready_lengths, ready_point_masks, padded_node_ids, raw_tail_lengths


def compute_conv_loss_from_ready(
    conv_head,
    ready_x,
    ready_y,
    device,
    window_size,
    criterion_block,
    batch_size,
    ready_lengths=None,
    ready_point_masks=None,
):
    """
    ready_x: list of Tensor[L_i, 2]
    ready_y: list of Tensor[L_i]
    ready_lengths: optional list[int], original valid lengths before padding
    ready_point_masks: optional list of Tensor[L_i], 1 for real point, 0 for pad

    Supports variable-length tail chunks by padding to the batch max length.
    The downstream objective keeps only block BCE loss.
    Conv point loss and the old auxiliary representation margin loss are disabled.
    """
    if len(ready_x) == 0:
        return None

    total = None
    denom = 0
    Wb = int(window_size)

    for s in range(0, len(ready_x), batch_size):
        e = min(len(ready_x), s + batch_size)

        xb_list = [t.to(device) for t in ready_x[s:e]]
        yb_list = [t.to(device) for t in ready_y[s:e]]

        xb = pad_sequence(xb_list, batch_first=True, padding_value=0.0)   # [B,Lmax,2]
        yb = pad_sequence(yb_list, batch_first=True, padding_value=0.0)   # [B,Lmax]

        if ready_lengths is None:
            lengths = torch.tensor(
                [t.size(0) for t in xb_list],
                dtype=torch.long,
                device=device,
            )
        else:
            lengths = torch.tensor(ready_lengths[s:e], dtype=torch.long, device=device)

        if ready_point_masks is None:
            point_valid_mask = torch.ones_like(yb, dtype=torch.float32, device=device)
        else:
            point_valid_mask = pad_sequence(
                [m.to(device) for m in ready_point_masks[s:e]],
                batch_first=True,
                padding_value=0.0,
            )  # [B,Lmax]

        xb_in = xb.permute(0, 2, 1).contiguous()   # [B,2,Lmax]
        point_logits, block_logits = conv_head(
            xb_in,
            lengths=lengths,
            point_valid_mask=point_valid_mask,
        )
        del point_logits

        y_blk = F.max_pool1d(
            yb.unsqueeze(1),
            kernel_size=Wb,
            stride=Wb,
        ).squeeze(1)  # [B,Nblk]

        blk_valid = F.max_pool1d(
            point_valid_mask.unsqueeze(1),
            kernel_size=Wb,
            stride=Wb,
        ).squeeze(1) > 0  # [B,Nblk]

        block_loss_raw = criterion_block(block_logits, y_blk)  # [B,Nblk]
        if blk_valid.any():
            block_loss = block_loss_raw[blk_valid].mean()
        else:
            block_loss = block_loss_raw.mean() * 0.0

        loss = block_loss

        total = loss * xb.size(0) if total is None else total + loss * xb.size(0)
        denom += xb.size(0)

    return total / max(denom, 1)


@torch.no_grad()
def collect_eval_node_sequences(
    dcl_tgn,
    data_split,
    batch_size,
    n_neighbors,
    device,
    negative_nodes,
    distance_metric,
    mi_method,
    args,
    reset_memory=True,
):
    seq_rec = {}
    seq_drift = {}
    seq_labels = {}

    sources_t = torch.from_numpy(data_split.sources).long().to(device)
    destinations_t = torch.from_numpy(data_split.destinations).long().to(device)
    timestamps_t = torch.from_numpy(data_split.timestamps).float().to(device)
    labels_t = torch.from_numpy(data_split.labels).float().to(device)

    num_instance = len(data_split.sources)
    num_batch = math.ceil(num_instance / batch_size)

    ngh_finder = dcl_tgn.neighbor_finder
    src_neighbors, _, src_neighbors_time = ngh_finder.get_temporal_neighbor_tqdm(
        data_split.sources, data_split.timestamps, n_neighbors
    )
    dst_neighbors, _, dst_neighbors_time = ngh_finder.get_temporal_neighbor_tqdm(
        data_split.destinations, data_split.timestamps, n_neighbors
    )

    src_neighbors_t = torch.from_numpy(src_neighbors).long().to(device)
    dst_neighbors_t = torch.from_numpy(dst_neighbors).long().to(device)
    src_neighbors_time_t = torch.from_numpy(src_neighbors_time).long().to(device)
    dst_neighbors_time_t = torch.from_numpy(dst_neighbors_time).long().to(device)

    dcl_tgn.eval()
    if reset_memory:
        dcl_tgn.memory.__init_memory__()

    for k in range(num_batch):
        s_idx = k * batch_size
        e_idx = min(num_instance, s_idx + batch_size)

        sources_batch = sources_t[s_idx:e_idx]
        destinations_batch = destinations_t[s_idx:e_idx]
        timestamps_batch = timestamps_t[s_idx:e_idx]
        labels_batch = labels_t[s_idx:e_idx]
        src_neighbors_batch = src_neighbors_t[s_idx:e_idx]
        dst_neighbors_batch = dst_neighbors_t[s_idx:e_idx]
        src_neighbors_time_batch = src_neighbors_time_t[s_idx:e_idx]
        dst_neighbors_time_batch = dst_neighbors_time_t[s_idx:e_idx]

        src_rec_score, src_drift_score, _, _ = dcl_tgn.compute_anomaly_score(
            sources_batch,
            destinations_batch,
            timestamps_batch,
            src_neighbors_batch,
            dst_neighbors_batch,
            src_neighbors_time_batch,
            dst_neighbors_time_batch,
            n_neighbors,
            negative_nodes,
            distance_metric,
            mi_method,
        )

        src_2ch = select_src_two_channel_tensor(args, src_rec_score, src_drift_score)

        for idx, src_node in enumerate(sources_batch):
            nid = int(src_node.item())
            seq_rec.setdefault(nid, []).append(float(src_2ch[idx, 0].item()))
            seq_drift.setdefault(nid, []).append(float(src_2ch[idx, 1].item()))
            seq_labels.setdefault(nid, []).append(float(labels_batch[idx].item()))

    return seq_rec, seq_drift, seq_labels


@torch.no_grad()
def warmup_memory_on_train_data(
    dcl_tgn,
    train_data_sources,
    train_data_destinations,
    train_data_timestamps,
    train_data_src_neighbors,
    train_data_dst_neighbors,
    train_data_src_neighbors_time,
    train_data_dst_neighbors_time,
    batch_size,
    n_neighbors,
    negative_nodes,
    distance_metric,
    mi_method,
    collect_scores=False,
):
    """
    Replay train interactions in chronological order to bring TGN memory to the
    train/test boundary before evaluating test interactions.

    This function intentionally uses compute_anomaly_score because that path
    advances memory exactly like test-time inference, but all returned scores
    are discarded unless collect_scores=True.
    """
    dcl_tgn.eval()

    num_instance = train_data_sources.size(0)
    num_batch = math.ceil(num_instance / batch_size)
    raw_rec_scores = []
    raw_drift_scores = []

    for k in range(num_batch):
        s_idx = k * batch_size
        e_idx = min(num_instance, s_idx + batch_size)

        src_rec_score, src_drift_score, _, _ = dcl_tgn.compute_anomaly_score(
            train_data_sources[s_idx:e_idx],
            train_data_destinations[s_idx:e_idx],
            train_data_timestamps[s_idx:e_idx],
            train_data_src_neighbors[s_idx:e_idx],
            train_data_dst_neighbors[s_idx:e_idx],
            train_data_src_neighbors_time[s_idx:e_idx],
            train_data_dst_neighbors_time[s_idx:e_idx],
            n_neighbors,
            negative_nodes,
            distance_metric,
            mi_method,
        )

        if collect_scores:
            raw_rec_scores.extend(src_rec_score.reshape(-1).detach().cpu().numpy().astype(np.float32))
            raw_drift_scores.extend(src_drift_score.reshape(-1).detach().cpu().numpy().astype(np.float32))

    if not collect_scores:
        return None, None

    return (
        np.asarray(raw_rec_scores, dtype=np.float32),
        np.asarray(raw_drift_scores, dtype=np.float32),
    )


@torch.no_grad()
def eval_conv_on_sequences(
    conv_head,
    seq_rec: dict,
    seq_drift: dict,
    seq_labels: dict,
    device,
    window_size,
    batch_size,
):
    """
    Keep partial tail windows during conv evaluation.

    Strategy:
      - for each source sequence with L > 0:
          target_len = max(Wb, ceil(L / Wb) * Wb)
          pad x with dummy token [0, 0]
          pad y with 0
          build point_valid_mask to mark real points
      - run conv_head on padded sequence, but pass original lengths
      - block validity is determined by pooled point_valid_mask
      - final ROC-AUC is computed over all valid blocks, including the last partial tail block
    """
    Wb = int(window_size)

    xs = []
    ys = []
    point_masks = []
    lengths = []

    for nid, rec_seq in seq_rec.items():
        drift_seq = seq_drift.get(nid, None)
        y_seq = seq_labels.get(nid, None)
        if drift_seq is None or y_seq is None:
            continue

        L = min(len(rec_seq), len(drift_seq), len(y_seq))
        if L <= 0:
            continue

        target_len = max(Wb, int(math.ceil(L / Wb) * Wb))

        x_seq = torch.zeros(target_len, 2, dtype=torch.float32)
        y_pad = torch.zeros(target_len, dtype=torch.float32)
        point_valid_mask = torch.zeros(target_len, dtype=torch.float32)

        rec_np = np.asarray(rec_seq[:L], dtype=np.float32)
        drift_np = np.asarray(drift_seq[:L], dtype=np.float32)
        y_np = np.asarray(y_seq[:L], dtype=np.float32)

        x_seq[:L, 0] = torch.from_numpy(rec_np)
        x_seq[:L, 1] = torch.from_numpy(drift_np)
        y_pad[:L] = torch.from_numpy(y_np)
        point_valid_mask[:L] = 1.0

        xs.append(x_seq)
        ys.append(y_pad)
        point_masks.append(point_valid_mask)
        lengths.append(int(L))

    if len(xs) == 0:
        return float("nan"), float("nan")

    probs_all = []
    y_all = []
    conv_head.eval()

    for s in range(0, len(xs), batch_size):
        e = min(len(xs), s + batch_size)

        x_pad = pad_sequence(xs[s:e], batch_first=True, padding_value=0.0).to(device)   # [B,Lmax,2]
        y_pad = pad_sequence(ys[s:e], batch_first=True, padding_value=0.0).to(device)   # [B,Lmax]
        point_mask_pad = pad_sequence(point_masks[s:e], batch_first=True, padding_value=0.0).to(device)  # [B,Lmax]
        lengths_t = torch.tensor(lengths[s:e], dtype=torch.long, device=device)

        x_in = x_pad.permute(0, 2, 1).contiguous()   # [B,2,Lmax]
        _, block_logits = conv_head(
            x_in,
            lengths=lengths_t,
            point_valid_mask=point_mask_pad,
        )

        probs = torch.sigmoid(block_logits)   # [B,Nblk]

        y_blk = F.max_pool1d(
            y_pad.unsqueeze(1),
            kernel_size=Wb,
            stride=Wb,
        ).squeeze(1)   # [B,Nblk]

        blk_valid = F.max_pool1d(
            point_mask_pad.unsqueeze(1),
            kernel_size=Wb,
            stride=Wb,
        ).squeeze(1) > 0   # [B,Nblk]

        if blk_valid.any():
            probs_all.append(probs[blk_valid].detach().cpu().numpy())
            y_all.append(y_blk[blk_valid].detach().cpu().numpy())

    probs_flat = np.concatenate(probs_all) if probs_all else np.array([], dtype=np.float32)
    y_flat = np.concatenate(y_all) if y_all else np.array([], dtype=np.float32)

    if probs_flat.size == 0 or len(np.unique(y_flat)) < 2:
        return float("nan"), float("nan")

    auc = float(roc_auc_score(y_flat, probs_flat))
    ap = float(average_precision_score(y_flat, probs_flat))
    return auc, ap



