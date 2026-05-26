import argparse
import logging
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from model.SLADE_TGN import SLADE_TGN
from model.contextual_supervision import DynamicNonOverlapWindowNet
from utils.data_processing import get_data_node_classification
from utils.e2e_training import (
    append_and_collect_ready_chunks,
    collect_eval_node_sequences,
    compute_block_pos_weight,
    compute_conv_loss_from_ready,
    detach_residual_buffers,
    eval_conv_on_sequences,
    flush_tail_buffers_with_padding_token,
    group_labels_by_source,
    safe_nanargmax,
    select_src_two_channel_tensor,
    summarize_eval_win_coverage,
    warmup_memory_on_train_data,
    set_all_seeds,
)
from utils.utils import fit_quantile_scale, get_neighbor_finder
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

    train_labels_by_node = group_labels_by_source(train_data.sources, train_data.labels)
    block_pw = compute_block_pos_weight(train_labels_by_node, base_window_size=Wb, cap=args.stage2_pos_weight_cap)

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

                src_score_2ch = select_src_two_channel_tensor(args, src_rec_bound, src_drift_bound)  # [B,2]

                ready_x = []
                ready_y = []
                for idx, src_node in enumerate(sources_batch):
                    nid = int(src_node.item())
                    new_x, new_y = append_and_collect_ready_chunks(
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

                conv_loss = compute_conv_loss_from_ready(
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
                        detach_residual_buffers(score_buffer, label_buffer)
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

                    detach_residual_buffers(score_buffer, label_buffer)
                    dcl_tgn.memory.detach_memory()
                    chunk_ctr_losses = []
                    chunk_conv_losses = []

            # ------------------------------------------------------------
            # Epoch-end residual flush with padding token + masked supervision
            # ------------------------------------------------------------
            tail_ready_x, tail_ready_y, tail_ready_lengths, tail_point_masks, tail_padded_node_ids, tail_raw_lengths = flush_tail_buffers_with_padding_token(
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

                tail_conv_loss = compute_conv_loss_from_ready(
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
            raw_rec_scores_np, raw_drift_scores_np = warmup_memory_on_train_data(
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
            warmup_memory_on_train_data(
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
            eval_seq_rec, eval_seq_drift, eval_seq_labels = collect_eval_node_sequences(
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
            eval_win_node_count, eval_win_window_count, eval_win_pos_window_count, eval_win_neg_window_count = summarize_eval_win_coverage(
                seq_rec=eval_seq_rec,
                seq_drift=eval_seq_drift,
                seq_labels=eval_seq_labels,
                window_size=Wb,
            )
            win_auc, win_ap = eval_conv_on_sequences(
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

        best_conv_idx = safe_nanargmax(conv_win_aucs)
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
