import math

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from utils.utils import normalize_clip


def make_windows_from_src_dict_nonoverlap(
    src_scores_dict: dict,
    src_labels_dict: dict,
    window_size: int = 2,
    score_reduce: str = "max",
):
    """
    Non-overlap source-node windows aligned with conv evaluation.

    The last partial tail window is kept if it contains at least one real point.
    Window score and label are computed only over real points.
    """
    assert window_size >= 1
    assert score_reduce in ("mean", "max")

    y_true_all = []
    y_score_all = []
    meta = []

    for sid in src_scores_dict.keys():
        if sid not in src_labels_dict:
            continue

        scores = np.asarray(src_scores_dict[sid], dtype=np.float32)
        labels = np.asarray(src_labels_dict[sid], dtype=np.int64)

        if len(scores) != len(labels):
            raise ValueError(
                f"sid={sid} scores/labels length mismatch: {len(scores)} vs {len(labels)}"
            )

        L = int(len(scores))
        if L <= 0:
            continue

        target_len = max(window_size, int(math.ceil(L / window_size) * window_size))

        for st in range(0, target_len, window_size):
            ed = min(st + window_size, L)
            if ed <= st:
                continue

            w_scores = scores[st:ed]
            w_labels = labels[st:ed]

            if score_reduce == "mean":
                w_score = float(w_scores.mean())
            else:
                w_score = float(w_scores.max())

            w_label = int((w_labels == 1).any())

            y_score_all.append(w_score)
            y_true_all.append(w_label)
            meta.append((sid, st, ed))

    return (
        np.asarray(y_true_all, dtype=np.int64),
        np.asarray(y_score_all, dtype=np.float32),
        meta,
    )


def window_level_metrics(
    test_src_pred_scores,
    test_src_labels,
    window_size=2,
    score_reduce="max",
    return_details=False,
):
    y_true, y_score, meta = make_windows_from_src_dict_nonoverlap(
        test_src_pred_scores,
        test_src_labels,
        window_size=window_size,
        score_reduce=score_reduce,
    )

    if y_true.size == 0 or np.unique(y_true).size < 2:
        auc = float("nan")
    else:
        auc = float(roc_auc_score(y_true, y_score))

    if return_details:
        return auc, y_true, y_score, meta
    return auc


def eval_anomaly_node_detection(
    model,
    data,
    batch_size,
    n_neighbors,
    device,
    only_rec_score=False,
    only_drift_score=False,
    test_inference_time=False,
    negative_nodes=None,
    distance_metric="cosine",
    M_rec=None,
    M_drift=None,
    mi_method="mine",
    window_size=2,
    window_score_reduce="max",
):
    """
    Return:
      auc_roc: point-level ROC-AUC
      win_auc: non-overlap source-window ROC-AUC with partial tail windows kept
      pred_score: point-level anomaly scores
      pred_mask: whether score at each position has been filled
    """
    if test_inference_time:
        raise NotImplementedError(
            "test_inference_time is disabled because the old evolving-neighbor "
            "path mutates timestamps without synchronized edge indices."
        )

    pred_score = np.zeros(len(data.sources), dtype=np.float32)
    pred_mask = np.zeros(len(data.sources), dtype=bool)

    num_instance = len(data.sources)
    num_batch = math.ceil(num_instance / batch_size)

    test_data_labels = torch.from_numpy(data.labels).float().to(device)

    with torch.no_grad():
        model.eval()

        test_src_pred_scores = {}
        test_src_labels = {}

        for k in range(num_batch):
            s_idx = k * batch_size
            e_idx = min(num_instance, s_idx + batch_size)

            sources_batch = torch.from_numpy(data.sources[s_idx:e_idx]).long().to(device)
            destinations_batch = torch.from_numpy(data.destinations[s_idx:e_idx]).long().to(device)
            timestamps_batch = torch.from_numpy(data.timestamps[s_idx:e_idx]).float().to(device)
            labels_batch = test_data_labels[s_idx:e_idx].float().to(device)

            src_neighbors_batch_np, _, src_neighbors_time_batch_np = model.neighbor_finder.get_temporal_neighbor(
                data.sources[s_idx:e_idx], data.timestamps[s_idx:e_idx], n_neighbors
            )
            dst_neighbors_batch_np, _, dst_neighbors_time_batch_np = model.neighbor_finder.get_temporal_neighbor(
                data.destinations[s_idx:e_idx], data.timestamps[s_idx:e_idx], n_neighbors
            )

            src_neighbors_batch = torch.from_numpy(src_neighbors_batch_np).long().to(device)
            dst_neighbors_batch = torch.from_numpy(dst_neighbors_batch_np).long().to(device)
            src_neighbors_time_batch = torch.from_numpy(src_neighbors_time_batch_np).long().to(device)
            dst_neighbors_time_batch = torch.from_numpy(dst_neighbors_time_batch_np).long().to(device)

            source_recovery_score, source_drift_score, _, _ = model.compute_anomaly_score(
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

            src_rec_np = source_recovery_score.reshape(-1).detach().cpu().numpy()
            src_drift_np = source_drift_score.reshape(-1).detach().cpu().numpy()

            if np.isnan(src_rec_np).sum() > 0:
                raise ValueError("NaN detected in source recovery scores")

            if only_drift_score:
                batch_scores = 1 - normalize_clip(src_drift_np, M_drift)
            elif only_rec_score:
                batch_scores = 1 - normalize_clip(src_rec_np, M_rec)
            else:
                batch_scores = 1 - normalize_clip(src_drift_np + src_rec_np, M_drift + M_rec)

            pred_score[s_idx:e_idx] = batch_scores.astype(np.float32)
            pred_mask[s_idx:e_idx] = True

            for idx, src_node in enumerate(sources_batch):
                sid = int(src_node.item())
                test_src_pred_scores.setdefault(sid, []).append(float(batch_scores[idx]))
                test_src_labels.setdefault(sid, []).append(float(labels_batch[idx].item()))

        win_auc = window_level_metrics(
            test_src_pred_scores,
            test_src_labels,
            window_size=window_size,
            score_reduce=window_score_reduce,
            return_details=False,
        )

        if len(np.unique(data.labels)) < 2:
            auc_roc = float("nan")
        else:
            auc_roc = float(roc_auc_score(data.labels, pred_score))

        return auc_roc, win_auc, pred_score, pred_mask
