import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import math

class MergeLayer(torch.nn.Module):
    def __init__(self, dim1, dim2, dim3, dim4):
        super().__init__()
        self.fc1 = torch.nn.Linear(dim1 + dim2, dim3)
        self.fc2 = torch.nn.Linear(dim3, dim4)
        self.act = torch.nn.ReLU()

        torch.nn.init.xavier_normal_(self.fc1.weight)
        torch.nn.init.xavier_normal_(self.fc2.weight)

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        h = self.act(self.fc1(x))
        return self.fc2(h)
    


def get_neighbor_finder(data, uniform=False, max_node_idx=None):
    max_node_idx = max(data.sources.max(), data.destinations.max()) if max_node_idx is None else max_node_idx
    adj_list = [[] for _ in range(max_node_idx + 1)]
    for source, destination, edge_idx, timestamp in zip(data.sources, data.destinations,
                                                      data.edge_idxs,
                                                      data.timestamps):
        adj_list[source].append((destination, edge_idx, timestamp))
        adj_list[destination].append((source, edge_idx, timestamp))
    return NeighborFinder(adj_list, uniform=uniform)

# class TransformerCell(nn.Module):
#     def __init__(self, message_dimension, memory_dimension, num_heads=2, num_layers=1, dropout=0.1):
#         super(TransformerCell, self).__init__()
#         print("[DEBUG] Using TransformerMemoryUpdater")
#         self.input_proj = nn.Linear(message_dimension, memory_dimension)

#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=memory_dimension,
#             nhead=num_heads,
#             dim_feedforward=4*memory_dimension,
#             dropout=dropout,
#             batch_first=True
#         )
#         self.transformer_encoder = nn.TransformerEncoder(
#             encoder_layer,
#             num_layers=num_layers   
#         )
    
#     def forward(self, messages, memory):
#         # messages: (B, message_dimension)
#         # memory: (B, memory_dimension)
#         # return: (B, memory_dimension)
#         mem_token = memory.unsqueeze(1)  # (B, 1, D)
#         msg_token = self.input_proj(messages).unsqueeze(1)  # (B, 1, D)

#         seq = torch.cat([mem_token, msg_token], dim=1)  # (B, 2, D)
#         L = seq.size(1)
#         mask = torch.full((L, L), float('-inf'), device=seq.device)
#         mask = torch.triu(mask, diagonal=1)  # (2, 2)

#         out = self.transformer_encoder(seq, mask=mask)  # (B, 2, D)
#         # out = self.transformer_encoder(seq)  # (B, 2, D)
#         updated_memory = out[:, -1, :]  # (B, D)
#         return updated_memory

class NeighborFinder:
    def __init__(self, adj_list, uniform=False, seed=None):
        self.node_to_neighbors = []
        self.node_to_edge_idxs = []
        self.node_to_edge_timestamps = []

        for neighbors in adj_list:
            sorted_neighhbors = sorted(neighbors, key=lambda x: x[2])
            self.node_to_neighbors.append(np.array([x[0] for x in sorted_neighhbors]))
            self.node_to_edge_idxs.append(np.array([x[1] for x in sorted_neighhbors]))
            self.node_to_edge_timestamps.append(np.array([x[2] for x in sorted_neighhbors]))
        self.uniform = uniform

        if seed is not None:
            self.seed = seed
            self.random_state = np.random.RandomState(self.seed)

    def find_before(self, src_idx, cut_time):
        i = np.searchsorted(self.node_to_edge_timestamps[src_idx], cut_time)
        return self.node_to_neighbors[src_idx][:i], self.node_to_edge_idxs[src_idx][:i], self.node_to_edge_timestamps[src_idx][:i]
    
    def get_temporal_neighbor_tqdm(self, source_nodes, timestamps, n_neighbors=20):
        assert (len(source_nodes) == len(timestamps))
        tmp_n_neighbors = n_neighbors if n_neighbors > 0 else 1
        # NB! All interactions described in these matrices are sorted in each row by time
        neighbors = np.zeros((len(source_nodes), tmp_n_neighbors)).astype(
        np.int32)  # each entry in position (i,j) represent the id of the item targeted by user src_idx_l[i] with an interaction happening before cut_time_l[i]
        edge_times = np.zeros((len(source_nodes), tmp_n_neighbors)).astype(
        np.float32)  # each entry in position (i,j) represent the timestamp of an interaction between user src_idx_l[i] and item neighbors[i,j] happening before cut_time_l[i]
        edge_idxs = np.zeros((len(source_nodes), tmp_n_neighbors)).astype(
        np.int32)  # each entry in position (i,j) represent the interaction index of an interaction between user src_idx_l[i] and item neighbors[i,j] happening before cut_time_l[i]

        for i, (source_node, timestamp) in tqdm(enumerate(zip(source_nodes, timestamps))):
            source_neighbors, source_edge_idxs, source_edge_times = self.find_before(source_node,
                                                   timestamp) # extracts all neighbors, interactions indexes and timestamps of all interactions of user source_node happening before cut_time
            
            if len(source_neighbors) > 0 and n_neighbors > 0:
                if self.uniform:  # if we are applying uniform sampling, shuffles the data above before sampling
                    sampled_idx = np.random.randint(0, len(source_neighbors), n_neighbors)

                    neighbors[i, :] = source_neighbors[sampled_idx]
                    edge_times[i, :] = source_edge_times[sampled_idx]
                    edge_idxs[i, :] = source_edge_idxs[sampled_idx]

                    # re-sort based on time
                    pos = edge_times[i, :].argsort()
                    neighbors[i, :] = neighbors[i, :][pos]
                    edge_times[i, :] = edge_times[i, :][pos]
                    edge_idxs[i, :] = edge_idxs[i, :][pos]
                else:
                    source_edge_times = source_edge_times[-n_neighbors:]
                    source_neighbors = source_neighbors[-n_neighbors:]
                    source_edge_idxs = source_edge_idxs[-n_neighbors:]

                    assert (len(source_neighbors) <= n_neighbors)
                    assert (len(source_edge_times) <= n_neighbors)
                    assert (len(source_edge_idxs) <= n_neighbors)

                    neighbors[i, n_neighbors - len(source_neighbors):] = source_neighbors
                    edge_times[i, n_neighbors - len(source_edge_times):] = source_edge_times
                    edge_idxs[i, n_neighbors - len(source_edge_idxs):] = source_edge_idxs

        return neighbors, edge_idxs, edge_times
    
    def get_temporal_neighbor(self, source_nodes, timestamps, n_neighbors=20):
        assert (len(source_nodes) == len(timestamps))
        tmp_n_neighbors = n_neighbors if n_neighbors > 0 else 1
        # NB! All interactions described in these matrices are sorted in each row by time
        neighbors = np.zeros((len(source_nodes), tmp_n_neighbors)).astype(
        np.int32)  # each entry in position (i,j) represent the id of the item targeted by user src_idx_l[i] with an interaction happening before cut_time_l[i]
        edge_times = np.zeros((len(source_nodes), tmp_n_neighbors)).astype(
        np.float32)  # each entry in position (i,j) represent the timestamp of an interaction between user src_idx_l[i] and item neighbors[i,j] happening before cut_time_l[i]
        edge_idxs = np.zeros((len(source_nodes), tmp_n_neighbors)).astype(
        np.int32)  # each entry in position (i,j) represent the interaction index of an interaction between user src_idx_l[i] and item neighbors[i,j] happening before cut_time_l[i]

        for i, (source_node, timestamp) in enumerate(zip(source_nodes, timestamps)):
            source_neighbors, source_edge_idxs, source_edge_times = self.find_before(source_node,
                                                   timestamp)
            
            if len(source_neighbors) > 0 and n_neighbors > 0:
                    if self.uniform:  # if we are applying uniform sampling, shuffles the data above before sampling
                        sampled_idx = np.random.randint(0, len(source_neighbors), n_neighbors)

                        neighbors[i, :] = source_neighbors[sampled_idx]
                        edge_times[i, :] = source_edge_times[sampled_idx]
                        edge_idxs[i, :] = source_edge_idxs[sampled_idx]

                        # re-sort based on time
                        pos = edge_times[i, :].argsort()
                        neighbors[i, :] = neighbors[i, :][pos]
                        edge_times[i, :] = edge_times[i, :][pos]
                        edge_idxs[i, :] = edge_idxs[i, :][pos]
                    else:
                        source_edge_times = source_edge_times[-n_neighbors:]
                        source_neighbors = source_neighbors[-n_neighbors:]
                        source_edge_idxs = source_edge_idxs[-n_neighbors:]

                        assert (len(source_neighbors) <= n_neighbors)
                        assert (len(source_edge_times) <= n_neighbors)
                        assert (len(source_edge_idxs) <= n_neighbors)

                        neighbors[i, n_neighbors - len(source_neighbors):] = source_neighbors
                        edge_times[i, n_neighbors - len(source_edge_times):] = source_edge_times
                        edge_idxs[i, n_neighbors - len(source_edge_idxs):] = source_edge_idxs

        return neighbors, edge_idxs, edge_times


def propagate_early_labels(node_label_dict: dict, k: int = 3):
    if k is None or k <= 1:
        return
    for nid, labels in node_label_dict.items():
        L = len(labels)
        for i in range(L):
            if labels[i] == 1:
                start = max(0, i - (k - 1))
                for j in range(start, i):
                    labels[j] = 1


def augment_mi_noise_only(x_seq, 
                          noise_std=0.03,
                        #   scale_range=(0.85, 1.15),
                        #   time_mask_prob=0.2,
                        #   time_mask_length=1,  # 固定遮挡1个时间步（安全！）
                          device="cpu"):
    """
    纯噪声增强（仅作用于x_seq，绝对不改变y_seq/窗口标签）
    专为窗口大小=2设计：遮挡1个点不会改变窗口标签（OR操作保留另一点信号）
    """
    B, L = x_seq.shape
    x_aug = x_seq.clone()
    
    # 1) 高斯噪声（全局）
    if noise_std > 0:
        noise = torch.randn_like(x_aug, device=device) * noise_std
        x_aug = x_aug + noise
    
    # # 2) 随机缩放（样本级）
    # if scale_range is not None:
    #     scale = torch.empty(B, 1, device=device).uniform_(*scale_range)
    #     x_aug = x_aug * scale
    
    # # 3) 时间遮挡（固定遮挡1个点，安全！）
    # if time_mask_prob > 0 and time_mask_length > 0:
    #     for b in range(B):
    #         if torch.rand(1, device=device).item() < time_mask_prob:
    #             # 确保遮挡位置有效（序列长度>1）
    #             if L > time_mask_length:
    #                 start = torch.randint(0, L - time_mask_length + 1, (1,), device=device).item()
    #                 x_aug[b, start:start+time_mask_length] = 0.0
    
    # 4) 物理约束：MI分数 >=0
    x_aug = torch.clamp(x_aug, min=0.0)
    return x_aug


def fit_quantile_scale(mi_train, q=0.95, eps=1e-8):
    M = np.quantile(mi_train, q)
    return max(M, eps)

def normalize_clip(mi_vals, M):
    return np.clip(mi_vals / M, 0.0, 1.0)

