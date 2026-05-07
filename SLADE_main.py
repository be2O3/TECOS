import math
import logging
import time
import sys
import random
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

from model.SLADE_TGN import SLADE_TGN
from model.conv1d_wei import DynamicNonOverlapWindowNet
from utils.utils import get_neighbor_finder, fit_quantile_scale
from utils.data_processing import get_data_node_classification
try:
    from utils.data_processing import apply_label_delay_by_source
except ImportError:
    def apply_label_delay_by_source(data, delay=0):
        delay = int(delay)
        if delay > 0:
            raise ImportError(
                "apply_label_delay_by_source is not available in utils.data_processing. "
                "Please sync the updated utils/data_processing.py before using --label_delay."
            )
        return data


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


def _select_src_two_channel_tensor(args, src_rec_bound, src_drift_bound):
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


def _group_labels_by_source(sources_np, labels_np):
    out = {}
    for nid, lb in zip(sources_np, labels_np):
        nid = int(nid)
        out.setdefault(nid, []).append(float(lb))
    return out


def _compute_block_pos_weight(epoch_src_labels: dict, base_window_size=4, cap=50.0):
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


def _compute_point_pos_weight(epoch_src_labels: dict, cap=50.0):
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


def _make_block_mask(lengths, nblk, Wb, device):
    block_lengths = torch.div(lengths, Wb, rounding_mode='floor')
    idx = torch.arange(nblk, device=device).unsqueeze(0)
    return idx < block_lengths.unsqueeze(1)


def _safe_nanargmax(values):
    arr = np.asarray(values, dtype=np.float32)
    if arr.size == 0 or np.all(np.isnan(arr)):
        return None
    return int(np.nanargmax(arr))


def _summarize_eval_win_coverage(seq_rec: dict, seq_drift: dict, seq_labels: dict, window_size: int):
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



def _detach_residual_buffers(score_buffer, label_buffer):
    for nid in list(score_buffer.keys()):
        score_buffer[nid] = [t.detach() for t in score_buffer[nid]]
    for nid in list(label_buffer.keys()):
        label_buffer[nid] = [t.detach() for t in label_buffer[nid]]


def _append_and_collect_ready_chunks(
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


def _flush_tail_buffers_with_padding_token(
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


def _compute_conv_loss_from_ready(
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
def _collect_eval_node_sequences(
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

        src_2ch = _select_src_two_channel_tensor(args, src_rec_score, src_drift_score)

        for idx, src_node in enumerate(sources_batch):
            nid = int(src_node.item())
            seq_rec.setdefault(nid, []).append(float(src_2ch[idx, 0].item()))
            seq_drift.setdefault(nid, []).append(float(src_2ch[idx, 1].item()))
            seq_labels.setdefault(nid, []).append(float(labels_batch[idx].item()))

    return seq_rec, seq_drift, seq_labels


@torch.no_grad()
def _warmup_memory_on_train_data(
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
def _eval_conv_on_sequences(
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



if __name__ == '__main__':
    parser = argparse.ArgumentParser('dynamic contrastive anomaly detection (End-to-End SLADE + 2ch conv1d)')

    parser.add_argument('-d', '--data', type=str, default='wikipedia')
    parser.add_argument('--bs', type=int, default=100)
    parser.add_argument('--n_degree', type=int, default=20)
    parser.add_argument('--n_head', type=int, default=2)
    parser.add_argument('--n_epoch', type=int, default=10)
    parser.add_argument('--lr', type=float, default=3e-6)
    parser.add_argument('--n_runs', type=int, default=3)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--drop_out', type=float, default=0.1)
    parser.add_argument('--mi_method', type=str, default='mine', choices=['mine', 'infonce'])

    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--message_dim', type=int, default=128)
    parser.add_argument('--memory_dim', type=int, default=256)
    parser.add_argument('--agg_type', type=str, default='TGAT')
    parser.add_argument('--negative_memory_type', type=str, default='train')
    parser.add_argument('--message_updater', type=str, default='mlp')
    parser.add_argument('--memory_updater', type=str, default='gru', choices=['gru', 'aggrgru', 'rnn', 'transformer'])

    parser.add_argument('--training_ratio', type=float, default=0.85)
    parser.add_argument('--label_delay', type=int, default=0)
    parser.add_argument('--delay_apply_to', type=str, default='both', choices=['train', 'test', 'both'])
    parser.add_argument('--lr_decay', type=float, default=0.8)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--srf', type=float, default=0.08)
    parser.add_argument('--drf', type=float, default=0.08)

    parser.add_argument('--only_drift_loss_score', action='store_true')
    parser.add_argument('--only_recovery_loss_score', action='store_true')
    parser.add_argument('--only_drift_score', action='store_true')
    parser.add_argument('--only_rec_score', action='store_true')
    parser.add_argument('--distance_metric', type=str, default='cosine', choices=['cosine', 'euclidean'])
    # Disabled for now: the old evolving-neighbor path mutates timestamps without
    # synchronized edge indices. Keep the attribute false to avoid accidental use.
    parser.set_defaults(test_inference_time=False)
    parser.add_argument('--quantile', type=float, default=0.95)

    parser.add_argument('--stage2_window_size', type=int, default=1)
    parser.add_argument('--stage2_hidden', type=int, default=64)
    parser.add_argument('--stage2_dropout', type=float, default=0.1)
    parser.add_argument('--stage2_lr', type=float, default=5e-4)
    parser.add_argument('--stage2_batch_size', type=int, default=128)
    parser.add_argument('--stage2_lambda_point', type=float, default=0.0,
                        help='Deprecated/ignored. Conv point loss is disabled in this version.')
    parser.add_argument('--stage2_pos_weight_cap', type=float, default=300)
    parser.add_argument('--stage2_kernel_size', type=int, default=5)
    parser.add_argument('--stage2_num_layers', type=int, default=2)
    parser.add_argument('--stage2_history_len', type=int, default=1)
    parser.add_argument('--stage2_chunk_num_blocks', type=int, default=5)
    parser.add_argument('--stage2_no_block_context', action='store_true')
    parser.add_argument('--stage2_pooling_type', type=str, default='softmax',
                        choices=['softmax', 'mean', 'max', 'meanmax'])
    parser.add_argument('--stage2_context_direction', type=str, default='bidirectional',
                        choices=['bidirectional', 'past', 'future'],
                        help='Conv1D temporal context: bidirectional, past-only, or future-only.')

    parser.add_argument('--e2e_chunk_batches', type=int, default=8)
    parser.add_argument('--e2e_alpha', type=float, default=1.0)
    parser.add_argument('--e2e_beta', type=float, default=1.0)
    parser.add_argument('--e2e_grad_clip', type=float, default=1.0)

    try:
        args = parser.parse_args()
    except Exception:
        parser.print_help()
        sys.exit(0)

    set_all_seeds(args.seed, deterministic=True)

    BATCH_SIZE = args.bs
    NUM_NEIGHBORS = args.n_degree
    NUM_EPOCH = args.n_epoch
    NUM_HEADS = args.n_head
    DROP_OUT = args.drop_out
    GPU = args.gpu
    DATA = args.data
    NUM_LAYER = 1
    LEARNING_RATE = args.lr
    MESSAGE_DIM = args.message_dim
    MEMORY_DIM = args.memory_dim
    memory_agg_type = args.agg_type
    negative_memory_type = args.negative_memory_type
    message_updater = args.message_updater
    distance_metric = args.distance_metric
    mi_method = args.mi_method
    q = args.quantile

    Wb = int(args.stage2_window_size)
    stage2_history_len = int(args.stage2_history_len)
    chunk_num_blocks = max(int(args.stage2_chunk_num_blocks), stage2_history_len)
    chunk_len = Wb * chunk_num_blocks
    chunk_step = 1

    Path('log/').mkdir(parents=True, exist_ok=True)

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    fh = logging.FileHandler(
        'log/{}__e2e_slade_conv2ch__srf{}_drf{}_epoch{}_lr{}_bs{}_memdim{}_msgdim{}_{}.log'.format(
            DATA, args.srf, args.drf, NUM_EPOCH, LEARNING_RATE, BATCH_SIZE,
            MEMORY_DIM, MESSAGE_DIM, str(time.time())
        )
    )
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)

    logger.info(args)
    logger.info('Data Processing')

    full_data, train_data, test_data = get_data_node_classification(DATA, training_ratio=args.training_ratio)
    if args.label_delay > 0:
        if args.delay_apply_to in {'train', 'both'}:
            train_data = apply_label_delay_by_source(train_data, delay=args.label_delay)
        if args.delay_apply_to in {'test', 'both'}:
            test_data = apply_label_delay_by_source(test_data, delay=args.label_delay)
        logger.info(
            'Applied label delay | delay=%d | split=%s | train_pos=%d | test_pos=%d',
            args.label_delay,
            args.delay_apply_to,
            int((np.asarray(train_data.labels) > 0).sum()),
            int((np.asarray(test_data.labels) > 0).sum()),
        )

    max_idx = max(full_data.unique_nodes)
    train_ngh_finder = get_neighbor_finder(train_data, uniform=False, max_node_idx=max_idx)
    full_ngh_finder = get_neighbor_finder(full_data, uniform=False, max_node_idx=max_idx)

    src_neighbors, _, src_neighbors_time = train_ngh_finder.get_temporal_neighbor_tqdm(
        train_data.sources, train_data.timestamps, NUM_NEIGHBORS
    )
    dst_neighbors, _, dst_neighbors_time = train_ngh_finder.get_temporal_neighbor_tqdm(
        train_data.destinations, train_data.timestamps, NUM_NEIGHBORS
    )

    device_string = f'cuda:{GPU}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_string)

    train_labels_by_node = _group_labels_by_source(train_data.sources, train_data.labels)
    block_pw = _compute_block_pos_weight(train_labels_by_node, base_window_size=Wb, cap=args.stage2_pos_weight_cap)

    conv_radius = (int(args.stage2_kernel_size) // 2) * int(args.stage2_num_layers)
    logger.info(
        'End2End conv config | Wb={} | history_len={} | chunk_num_blocks={} | chunk_len={} | chunk_step={} | pooling_type={} | kernel_size={} | num_layers={} | effective_radius={} | context_direction={} | block_context={}'.format(
            Wb, stage2_history_len, chunk_num_blocks, chunk_len, chunk_step, args.stage2_pooling_type,
            args.stage2_kernel_size, args.stage2_num_layers, conv_radius,
            args.stage2_context_direction, not args.stage2_no_block_context,
        )
    )
    if args.stage2_context_direction != 'bidirectional' and not args.stage2_no_block_context:
        logger.warning(
            'Directional Conv1D ablation still uses causal block history context; '
            'use --stage2_no_block_context for a pure Conv1D context-direction ablation.'
        )
    logger.info(
        'End2End pos_weight | block_pw={} | conv_point_loss=disabled'.format(
            'None' if block_pw is None else f'{block_pw:.6f}'
        )
    )

    best_conv_run = []
    best_conv_ap_run = []

    for i in range(args.n_runs):
        run_seed = int(args.seed + i)
        set_all_seeds(run_seed, deterministic=True)
        logger.info('Dynamic anomaly detection start - run: {} | run_seed: {}'.format(i, run_seed))

        dcl_tgn = SLADE_TGN(
            neighbor_finder=train_ngh_finder,
            n_nodes=full_data.n_unique_nodes,
            n_edges=full_data.n_interactions,
            device=device,
            n_layers=NUM_LAYER,
            n_heads=NUM_HEADS,
            dropout=DROP_OUT,
            message_dimension=MESSAGE_DIM,
            memory_dimension=MEMORY_DIM,
            n_neighbors=NUM_NEIGHBORS,
            memory_agg_type=memory_agg_type,
            negative_memory_type=negative_memory_type,
            message_updater=message_updater,
            memory_updater=args.memory_updater,
            src_reg_factor=args.srf,
            dst_reg_factor=args.drf,
            only_drift_loss=args.only_drift_loss_score,
            only_recovery_loss=args.only_recovery_loss_score,
            mi_method=mi_method,
        ).to(device)

        conv_head = DynamicNonOverlapWindowNet(
            base_window_size=Wb,
            history_len=stage2_history_len,
            hidden=args.stage2_hidden,
            gate_hidden=args.stage2_hidden,
            kernel_size=args.stage2_kernel_size,
            num_layers=args.stage2_num_layers,
            dropout=args.stage2_dropout,
            temp=1.0,
            use_block_context=(not args.stage2_no_block_context),
            in_ch=2,
            pooling_type=args.stage2_pooling_type,
            context_direction=args.stage2_context_direction,
        ).to(device)

        if block_pw is not None:
            criterion_block = nn.BCEWithLogitsLoss(reduction='none', pos_weight=torch.tensor([float(block_pw)], device=device))
        else:
            criterion_block = nn.BCEWithLogitsLoss(reduction='none')

        # Conv point loss is intentionally removed in this version.

        train_data_sources = torch.from_numpy(train_data.sources).long().to(device)
        train_data_destinations = torch.from_numpy(train_data.destinations).long().to(device)
        train_data_timestamps = torch.from_numpy(train_data.timestamps).float().to(device)
        train_data_labels = torch.from_numpy(train_data.labels).float().to(device)
        train_data_src_neighbors = torch.from_numpy(src_neighbors).long().to(device)
        train_data_dst_neighbors = torch.from_numpy(dst_neighbors).long().to(device)
        train_data_src_neighbors_time = torch.from_numpy(src_neighbors_time).long().to(device)
        train_data_dst_neighbors_time = torch.from_numpy(dst_neighbors_time).long().to(device)

        num_instance = len(train_data.sources)
        num_batch = math.ceil(num_instance / BATCH_SIZE)

        optimizer = torch.optim.Adam(
            [
                {'params': dcl_tgn.parameters(), 'lr': args.lr},
                {'params': conv_head.parameters(), 'lr': args.stage2_lr},
            ],
            weight_decay=args.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_decay)

        negative_train_nodes = torch.from_numpy(np.array(list(set(train_data.destinations) | set(train_data.sources)))).long().to(device)

        time_list = []
        train_time_list = []
        conv_win_aucs = []
        conv_win_aps = []

        for epoch in range(NUM_EPOCH):
            dcl_tgn.train()
            conv_head.train()

            dcl_tgn.memory.__init_memory__()
            dcl_tgn.set_neighbor_finder(train_ngh_finder)

            score_buffer = {}
            label_buffer = {}
            epoch_m_loss = []
            epoch_conv_loss = []
            chunk_ctr_losses = []
            chunk_conv_losses = []

            train_conv_node_ids = set()
            train_conv_chunk_count = 0
            train_conv_window_count = 0
            train_conv_pos_window_count = 0
            train_conv_neg_window_count = 0

            tail_pad_node_count = 0
            tail_pad_chunk_count = 0
            tail_pad_window_count = 0
            tail_pad_pos_window_count = 0
            tail_pad_neg_window_count = 0

            train_start = time.time()
            optimizer.zero_grad(set_to_none=True)

            for k in tqdm(range(num_batch), desc=f'run{i}-epoch{epoch}'):
                s_idx = k * BATCH_SIZE
                e_idx = min(num_instance, s_idx + BATCH_SIZE)

                sources_batch = train_data_sources[s_idx:e_idx]
                destinations_batch = train_data_destinations[s_idx:e_idx]
                timestamps_batch = train_data_timestamps[s_idx:e_idx]
                labels_batch = train_data_labels[s_idx:e_idx]
                src_neighbors_batch = train_data_src_neighbors[s_idx:e_idx]
                dst_neighbors_batch = train_data_dst_neighbors[s_idx:e_idx]
                src_neighbors_time_batch = train_data_src_neighbors_time[s_idx:e_idx]
                dst_neighbors_time_batch = train_data_dst_neighbors_time[s_idx:e_idx]

                contrastive_loss, src_rec_bound, dst_rec_bound, src_drift_bound, dst_drift_bound = dcl_tgn.compute_node_diff_score(
                    sources_batch,
                    destinations_batch,
                    timestamps_batch,
                    src_neighbors_batch,
                    dst_neighbors_batch,
                    src_neighbors_time_batch,
                    dst_neighbors_time_batch,
                    NUM_NEIGHBORS,
                    negative_train_nodes,
                    distance_metric,
                    mi_method,
                )

                chunk_ctr_losses.append(contrastive_loss)
                epoch_m_loss.append(float(contrastive_loss.item()))

                src_score_2ch = _select_src_two_channel_tensor(args, src_rec_bound, src_drift_bound)  # [B,2]

                ready_x = []
                ready_y = []
                for idx, src_node in enumerate(sources_batch):
                    nid = int(src_node.item())
                    new_x, new_y = _append_and_collect_ready_chunks(
                        score_buffer=score_buffer,
                        label_buffer=label_buffer,
                        node_id=nid,
                        score_t=src_score_2ch[idx],
                        label_t=labels_batch[idx],
                        chunk_len=chunk_len,
                        step_len=chunk_step,
                    )
                    if len(new_x) > 0:
                        train_conv_node_ids.add(nid)
                        train_conv_chunk_count += int(len(new_x))
                        for y_chunk in new_y:
                            y_blk = F.max_pool1d(
                                y_chunk.view(1, 1, -1),
                                kernel_size=Wb,
                                stride=Wb,
                            ).view(-1)
                            pos_blk = int((y_blk > 0.5).sum().item())
                            neg_blk = int((y_blk <= 0.5).sum().item())
                            train_conv_pos_window_count += pos_blk
                            train_conv_neg_window_count += neg_blk
                            train_conv_window_count += (pos_blk + neg_blk)
                    ready_x.extend(new_x)
                    ready_y.extend(new_y)

                conv_loss = _compute_conv_loss_from_ready(
                    conv_head=conv_head,
                    ready_x=ready_x,
                    ready_y=ready_y,
                    device=device,
                    window_size=Wb,
                    criterion_block=criterion_block,
                    batch_size=args.stage2_batch_size,
                    ready_lengths=None,
                    ready_point_masks=None,
                )

                if conv_loss is not None:
                    chunk_conv_losses.append(conv_loss)
                    epoch_conv_loss.append(float(conv_loss.detach().item()))

                is_chunk_end = ((k + 1) % max(int(args.e2e_chunk_batches), 1) == 0) or (k == num_batch - 1)
                if is_chunk_end:
                    if len(chunk_ctr_losses) == 0 and len(chunk_conv_losses) == 0:
                        _detach_residual_buffers(score_buffer, label_buffer)
                        dcl_tgn.memory.detach_memory()
                        optimizer.zero_grad(set_to_none=True)
                        continue

                    total_loss = None
                    if len(chunk_ctr_losses) > 0:
                        ctr_loss_mean = torch.stack(chunk_ctr_losses).mean()
                        total_loss = args.e2e_alpha * ctr_loss_mean
                    else:
                        ctr_loss_mean = None

                    if len(chunk_conv_losses) > 0:
                        conv_loss_mean = torch.stack(chunk_conv_losses).mean()
                        total_loss = args.e2e_beta * conv_loss_mean if total_loss is None else total_loss + args.e2e_beta * conv_loss_mean
                    else:
                        conv_loss_mean = None

                    total_loss.backward()

                    if args.e2e_grad_clip is not None and args.e2e_grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(list(dcl_tgn.parameters()) + list(conv_head.parameters()), max_norm=float(args.e2e_grad_clip))

                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

                    _detach_residual_buffers(score_buffer, label_buffer)
                    dcl_tgn.memory.detach_memory()
                    chunk_ctr_losses = []
                    chunk_conv_losses = []

            # ------------------------------------------------------------
            # Epoch-end residual flush with padding token + masked supervision
            # ------------------------------------------------------------
            tail_ready_x, tail_ready_y, tail_ready_lengths, tail_point_masks, tail_padded_node_ids, tail_raw_lengths = _flush_tail_buffers_with_padding_token(
                score_buffer=score_buffer,
                label_buffer=label_buffer,
                window_size=Wb,
            )

            if len(tail_ready_x) > 0:
                tail_pad_node_count = int(len(tail_padded_node_ids))
                tail_pad_chunk_count = int(len(tail_ready_y))

                for y_chunk, pmask in zip(tail_ready_y, tail_point_masks):
                    y_blk = F.max_pool1d(
                        y_chunk.view(1, 1, -1),
                        kernel_size=Wb,
                        stride=Wb,
                    ).view(-1)

                    blk_valid = F.max_pool1d(
                        pmask.view(1, 1, -1),
                        kernel_size=Wb,
                        stride=Wb,
                    ).view(-1) > 0

                    if blk_valid.any():
                        y_blk_valid = y_blk[blk_valid]
                        pos_blk = int((y_blk_valid > 0.5).sum().item())
                        neg_blk = int((y_blk_valid <= 0.5).sum().item())
                        tail_pad_pos_window_count += pos_blk
                        tail_pad_neg_window_count += neg_blk
                        tail_pad_window_count += (pos_blk + neg_blk)

                tail_conv_loss = _compute_conv_loss_from_ready(
                    conv_head=conv_head,
                    ready_x=tail_ready_x,
                    ready_y=tail_ready_y,
                    device=device,
                    window_size=Wb,
                    criterion_block=criterion_block,
                    batch_size=args.stage2_batch_size,
                    ready_lengths=tail_ready_lengths,
                    ready_point_masks=tail_point_masks,
                )

                if tail_conv_loss is not None:
                    optimizer.zero_grad(set_to_none=True)
                    tail_conv_loss.backward()

                    if args.e2e_grad_clip is not None and args.e2e_grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(
                            list(conv_head.parameters()),
                            max_norm=float(args.e2e_grad_clip),
                        )

                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

                    epoch_conv_loss.append(float(tail_conv_loss.detach().item()))

                train_conv_node_ids.update(tail_padded_node_ids)
                train_conv_chunk_count += tail_pad_chunk_count
                train_conv_window_count += tail_pad_window_count
                train_conv_pos_window_count += tail_pad_pos_window_count
                train_conv_neg_window_count += tail_pad_neg_window_count

            score_buffer.clear()
            label_buffer.clear()

            train_end = time.time()
            train_time_list.append(train_end - train_start)
            scheduler.step()

            eval_ngh_finder = train_ngh_finder if args.test_inference_time else full_ngh_finder
            dcl_tgn.set_neighbor_finder(eval_ngh_finder)

            dcl_tgn.eval()
            dcl_tgn.memory.__init_memory__()
            raw_rec_scores_np, raw_drift_scores_np = _warmup_memory_on_train_data(
                dcl_tgn=dcl_tgn,
                train_data_sources=train_data_sources,
                train_data_destinations=train_data_destinations,
                train_data_timestamps=train_data_timestamps,
                train_data_src_neighbors=train_data_src_neighbors,
                train_data_dst_neighbors=train_data_dst_neighbors,
                train_data_src_neighbors_time=train_data_src_neighbors_time,
                train_data_dst_neighbors_time=train_data_dst_neighbors_time,
                batch_size=BATCH_SIZE,
                n_neighbors=NUM_NEIGHBORS,
                negative_nodes=negative_train_nodes,
                distance_metric=distance_metric,
                mi_method=mi_method,
                collect_scores=True,
            )
            M_rec = fit_quantile_scale(raw_rec_scores_np, q=q)
            M_drift = fit_quantile_scale(raw_drift_scores_np, q=q)

            start = time.time()
            dcl_tgn.memory.__init_memory__()
            _warmup_memory_on_train_data(
                dcl_tgn=dcl_tgn,
                train_data_sources=train_data_sources,
                train_data_destinations=train_data_destinations,
                train_data_timestamps=train_data_timestamps,
                train_data_src_neighbors=train_data_src_neighbors,
                train_data_dst_neighbors=train_data_dst_neighbors,
                train_data_src_neighbors_time=train_data_src_neighbors_time,
                train_data_dst_neighbors_time=train_data_dst_neighbors_time,
                batch_size=BATCH_SIZE,
                n_neighbors=NUM_NEIGHBORS,
                negative_nodes=negative_train_nodes,
                distance_metric=distance_metric,
                mi_method=mi_method,
                collect_scores=False,
            )
            eval_seq_rec, eval_seq_drift, eval_seq_labels = _collect_eval_node_sequences(
                dcl_tgn=dcl_tgn,
                data_split=test_data,
                batch_size=BATCH_SIZE,
                n_neighbors=NUM_NEIGHBORS,
                device=device,
                negative_nodes=negative_train_nodes,
                distance_metric=distance_metric,
                mi_method=mi_method,
                args=args,
                reset_memory=False,
            )
            eval_win_node_count, eval_win_window_count, eval_win_pos_window_count, eval_win_neg_window_count = _summarize_eval_win_coverage(
                seq_rec=eval_seq_rec,
                seq_drift=eval_seq_drift,
                seq_labels=eval_seq_labels,
                window_size=Wb,
            )
            win_auc, win_ap = _eval_conv_on_sequences(
                conv_head=conv_head,
                seq_rec=eval_seq_rec,
                seq_drift=eval_seq_drift,
                seq_labels=eval_seq_labels,
                device=device,
                window_size=Wb,
                batch_size=args.stage2_batch_size,
            )

            end = time.time()
            time_list.append(end - start)

            mean_mloss = float(np.mean(epoch_m_loss)) if len(epoch_m_loss) > 0 else 0.0
            mean_convloss = float(np.mean(epoch_conv_loss)) if len(epoch_conv_loss) > 0 else 0.0
            logger.info(
                'E2E | run {} epoch {} | ctr_loss: {:.4f} | conv_loss: {:.4f} | conv_win_auc: {} | conv_win_ap: {} | train_conv_nodes: {} | train_conv_windows: {} | train_pos_windows: {} | train_neg_windows: {} | tail_pad_nodes: {} | tail_pad_chunks: {} | tail_pad_windows: {} | tail_pad_pos_windows: {} | tail_pad_neg_windows: {} | eval_win_nodes: {} | eval_win_windows: {} | eval_pos_windows: {} | eval_neg_windows: {}'.format(
                    i, epoch, mean_mloss, mean_convloss,
                    'nan' if np.isnan(win_auc) else f'{float(win_auc):.4f}',
                    'nan' if np.isnan(win_ap) else f'{float(win_ap):.4f}',
                    len(train_conv_node_ids),
                    int(train_conv_window_count),
                    int(train_conv_pos_window_count),
                    int(train_conv_neg_window_count),
                    int(tail_pad_node_count),
                    int(tail_pad_chunk_count),
                    int(tail_pad_window_count),
                    int(tail_pad_pos_window_count),
                    int(tail_pad_neg_window_count),
                    int(eval_win_node_count),
                    int(eval_win_window_count),
                    int(eval_win_pos_window_count),
                    int(eval_win_neg_window_count),
                )
            )

            conv_win_aucs.append(float(win_auc))
            conv_win_aps.append(float(win_ap))

        train_time_np = np.array(train_time_list, dtype=np.float32)
        infer_time_np = np.array(time_list, dtype=np.float32)
        logger.info('Run {} | training time mean: {:.4f} std: {:.4f}'.format(i, np.mean(train_time_np), np.std(train_time_np)))
        logger.info('Run {} | inference time mean: {:.4f} std: {:.4f}'.format(i, np.mean(infer_time_np), np.std(infer_time_np)))

        best_conv_idx = _safe_nanargmax(conv_win_aucs)
        if best_conv_idx is None:
            best_conv_auc = float('nan')
            best_conv_ap = float('nan')
        else:
            best_conv_auc = float(conv_win_aucs[best_conv_idx])
            best_conv_ap = float(conv_win_aps[best_conv_idx])

        logger.info(
            'Run {} | best outer epoch by conv_win_auc = {} | best conv_win_auc = {} | best conv_win_ap = {}'.format(
                i,
                'None' if best_conv_idx is None else best_conv_idx,
                'nan' if np.isnan(best_conv_auc) else f'{best_conv_auc:.4f}',
                'nan' if np.isnan(best_conv_ap) else f'{best_conv_ap:.4f}',
            )
        )

        best_conv_run.append(best_conv_auc)
        best_conv_ap_run.append(best_conv_ap)

    best_conv_run = np.array(best_conv_run, dtype=np.float32)
    best_conv_ap_run = np.array(best_conv_ap_run, dtype=np.float32)

    logger.info('Test end')
    logger.info('train_ratio: {:.2f}'.format(args.training_ratio))
    logger.info('E2E best conv win auc mean: {:.4f} std: {:.4f}'.format(np.nanmean(best_conv_run), np.nanstd(best_conv_run)))
    logger.info('E2E best conv win ap mean: {:.4f} std: {:.4f}'.format(np.nanmean(best_conv_ap_run), np.nanstd(best_conv_ap_run)))
