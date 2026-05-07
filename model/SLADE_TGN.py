import logging
import numpy as np
import torch
import os
import csv

from utils.utils import *
from modules.memory import Memory
from modules.message_function import get_message_function
from modules.memory_updater import get_memory_updater
from modules.embedding_module import get_embedding_module
from model.time_encoding import TimeEncode, Fixed_time_encode
from utils.MI import mi_lower_bound
from torch_scatter import scatter
import torch.nn as nn
import torch.nn.functional as F
import pdb

class SLADE_TGN(torch.nn.Module):
    def __init__(self, neighbor_finder, device, n_nodes, n_edges, n_layers=1, 
                 n_heads=2, dropout=0.1, message_dimension=128, memory_dimension=256,
                 n_neighbors=10, memory_agg_type='TGAT',negative_memory_type='train', message_updater='mlp', memory_updater='gru',
                 src_reg_factor=1, dst_reg_factor=1, only_drift_loss=False, only_recovery_loss=False, mi_method="mine_lower_bound"):
        super(SLADE_TGN, self).__init__()

        self.n_layers = n_layers
        self.neighbor_finder = neighbor_finder
        self.device = device
        self.logger = logging.getLogger(__name__)

        # self.mi_debug = False
        # self.mi_csv_path = "log/mi_debug_stats_euclidean.csv"
        # os.makedirs(os.path.dirname(self.mi_csv_path), exist_ok=True)
        # self.mi_step = 0
        self.memory_updater_type = memory_updater

        # if not os.path.exists(self.mi_csv_path):
        #     with open(self.mi_csv_path, "w", newline="") as f:
        #         writer = csv.writer(f)
        #         writer.writerow([
        #             "step", "role", "score_type", "distance_metric", "mean", "std", "min", "max", "N"
        #         ])

        self.n_node_features = memory_dimension
        self.n_nodes = n_nodes + 1 # first node memory is empty (because of zero neighbor masks)
        self.n_edge_features = memory_dimension
        self.n_edges = n_edges
        
        self.embedding_dimension = self.n_node_features
        self.n_neighbors = n_neighbors
        self.src_reg_factor = src_reg_factor
        self.dst_reg_factor = dst_reg_factor
        self.only_drift_loss = only_drift_loss
        self.only_recovery_loss = only_recovery_loss
        
        self.time_encoder = Fixed_time_encode(dimension=self.n_node_features)
        
        self.memory = None
        self.bias_src_memory = nn.Parameter(torch.zeros(memory_dimension), requires_grad=False)
        self.bias_dst_memory = nn.Parameter(torch.zeros(memory_dimension), requires_grad=False)

        self.memory_agg_type = memory_agg_type
        self.negative_memory_type = negative_memory_type

        self.memory_dimension = memory_dimension
        self.raw_message_dimension = self.memory_dimension + self.time_encoder.dimension

        if message_updater=='mlp':
            self.message_dimension = message_dimension
        else:
            self.message_dimension = self.raw_message_dimension

        self.memory = Memory(n_nodes=self.n_nodes,
                            memory_dimension=self.memory_dimension,
                            input_dimension=self.message_dimension,
                            message_dimension=self.message_dimension,
                            raw_message_dimension=self.raw_message_dimension,
                            device=device,
                            memory_updater=self.memory_updater_type)
        
        
        self.message_function = get_message_function(module_type=message_updater,
                                                    raw_message_dimension=self.raw_message_dimension,
                                                    message_dimension=self.message_dimension)
        mu_in_dim = self.raw_message_dimension if self.memory_updater_type == "transformer" else self.message_dimension
        self.memory_updater = get_memory_updater(module_type=self.memory_updater_type,
                                                memory=self.memory,
                                                message_dimension=mu_in_dim,
                                                memory_dimension=self.memory_dimension,
                                                device=device)
        
        self.embedding_module_recovery = get_embedding_module(module_type="graph_attention_recovery",
                                                    memory=self.memory,
                                                    neighbor_finder=self.neighbor_finder,
                                                    time_encoder=self.time_encoder,
                                                    n_layers=self.n_layers,
                                                    n_node_features=self.n_node_features,
                                                    n_edge_features=self.n_edge_features,
                                                    n_time_features=self.n_node_features,
                                                    embedding_dimension=self.embedding_dimension,
                                                    device=self.device,
                                                    n_heads=n_heads, dropout=dropout,
                                                    use_memory=True,
                                                    n_neighbors=self.n_neighbors)
    # def enable_mi_debug(self, flag: bool = True):
    #     self.mi_debug = flag
    #     mode = "ON" if flag else "OFF"
    #     self.logger.info(f"MI Debugging is {mode}")


    def compute_temporal_embeddings(self, source_nodes, destination_nodes, edge_times,
                                    src_neighbors, dst_neighbors,src_neighbors_time, dst_neighbors_time, n_neighbors, test=False):
        
        n_samples = len(source_nodes)
        nodes = torch.concat([source_nodes, destination_nodes])
        timestamps = torch.concat([edge_times, edge_times])
        neighbors = torch.concat([src_neighbors, dst_neighbors])
        neighbors_time = torch.concat([src_neighbors_time, dst_neighbors_time])
        memory_updater = self.memory_updater_type
        memory = None

        if memory_updater == "transformer":
            all_nodes = torch.concat([nodes,neighbors.reshape(-1)]).to(self.device)
            memory, _ = self.get_updated_memory_tensor(all_nodes, self.memory.messages_tensor, self.memory.messages_time)
        else:
            if test:
                all_nodes = torch.concat([nodes,neighbors.reshape(-1)])
                memory, _ = self.get_updated_memory_tensor(all_nodes, self.memory.messages_tensor, self.memory.messages_time)
            else:
                memory, _ = self.get_updated_memory_tensor(torch.arange(1, self.n_nodes), self.memory.messages_tensor, self.memory.messages_time)
                
        pos_node_embedding, dst_node_embedding = None, None
        if self.memory_agg_type == 'TGAT':
            node_embedding_TGAT = self.embedding_module_recovery.compute_recovery_memory_embedding(memory=memory,
                                                                    source_nodes=nodes,
                                                                    timestamps=timestamps, neighbors=neighbors, neighbors_time=neighbors_time,
                                                                    n_layers=self.n_layers,
                                                                    n_neighbors=n_neighbors)
            pos_node_embedding = node_embedding_TGAT[:n_samples]
            dst_node_embedding = node_embedding_TGAT[n_samples:]
        else:
            raise AssertionError('memory agg type is wrong!')
        
        node_memory = memory[source_nodes]
        dst_node_memory = memory[destination_nodes]

        self.update_memory_tensor(nodes, self.memory.messages_tensor, self.memory.messages_time)
        self.memory.clear_messages(nodes)

        unique_sources, source_message, source_message_ts, source_dt = self.get_raw_messages(source_nodes, destination_nodes, edge_times)        
        unique_destinations, dst_message, dst_message_ts, dst_dt = self.get_raw_messages(destination_nodes, source_nodes, edge_times)

        self.memory.store_raw_messages(unique_sources, source_message, source_message_ts)
        self.memory.store_raw_messages(unique_destinations, dst_message, dst_message_ts)
        return node_memory, pos_node_embedding, dst_node_memory, dst_node_embedding
    


    def compute_node_diff_score(self, source_nodes, destination_nodes, edge_times, src_neighbors, dst_neighbors, src_neighbors_time, dst_neighbors_time, n_neighbors, seen_nodes, distance_metric, mi_method):
        n_samples = len(source_nodes)
        prev_memory = self.memory.get_memory(source_nodes) # [B, D]
        prev_dst_memory = self.memory.get_memory(destination_nodes) # [B, D]

        node_memory, pos_node_embedding, dst_node_memory, dst_node_embedding = self.compute_temporal_embeddings(source_nodes, destination_nodes, edge_times, src_neighbors, dst_neighbors, src_neighbors_time, dst_neighbors_time, n_neighbors)
        if self.negative_memory_type == 'random':
            random_node = np.random.randint(0, self.n_nodes, size=n_samples)
            negative_memory = self.memory.get_memory(random_node)
        elif self.negative_memory_type == 'train':
            negative_memory = self.memory.get_memory(seen_nodes)#[8311, D]
        else:
            raise AssertionError("negative memory type is wrong")
        

        src_rec_bound = mi_lower_bound(pos_node_embedding, node_memory, negative_memory, distance_metric, mi_method) #[B]
        # src_rec_max = fit_quantile_scale(src_rec_bound.detach().cpu().numpy())
        dst_rec_bound = mi_lower_bound(dst_node_embedding, dst_node_memory, negative_memory, distance_metric, mi_method) #[B]
        # dst_rec_max = fit_quantile_scale(dst_rec_bound.detach().cpu().numpy())
        src_drift_bound = mi_lower_bound(node_memory, prev_memory, negative_memory, distance_metric, mi_method) #[B]
        # src_drift_max = fit_quantile_scale(src_drift_bound.detach().cpu().numpy())
        dst_drift_bound = mi_lower_bound(dst_node_memory, prev_dst_memory, negative_memory, distance_metric, mi_method) #[B]
        # dst_drift_max = fit_quantile_scale(dst_drift_bound.detach().cpu().numpy())

        if self.only_drift_loss:      # only memory drift loss
            contrastive_loss = - src_drift_bound - dst_drift_bound

        elif self.only_recovery_loss: # only memory reconstruciton loss
            contrastive_loss = - dst_rec_bound - src_rec_bound
        else:
            contrastive_loss = - self.src_reg_factor * src_rec_bound - self.dst_reg_factor * dst_rec_bound - src_drift_bound - dst_drift_bound


        contrastive_loss = contrastive_loss.mean()

        # return contrastive_loss, src_rec_max, dst_rec_max, src_drift_max, dst_drift_max
        return contrastive_loss, src_rec_bound, dst_rec_bound, src_drift_bound, dst_drift_bound


    def compute_anomaly_score(self, source_nodes, destination_nodes, edge_times, src_neighbors, dst_neighbors, src_neighbors_time, dst_neighbors_time, n_neighbors, negative_nodes, distance_metric, mi_method):
        prev_memory = self.memory.get_memory(source_nodes)
        prev_dst_memory = self.memory.get_memory(destination_nodes)
        node_memory, pos_node_embedding, dst_node_memory, dst_node_embedding = self.compute_temporal_embeddings(source_nodes, destination_nodes, edge_times, src_neighbors, dst_neighbors,
                                                                                                                src_neighbors_time, dst_neighbors_time, n_neighbors, test=True)

        negative_memory = self.memory.get_memory(negative_nodes)
        source_recovery_score = mi_lower_bound(pos_node_embedding, node_memory, negative_memory, distance_metric, mi_method)
        dst_recovery_score = mi_lower_bound(dst_node_embedding, dst_node_memory, negative_memory, distance_metric, mi_method)
        source_drift_score = mi_lower_bound(node_memory, prev_memory, negative_memory, distance_metric, mi_method)
        dst_drift_score = mi_lower_bound(dst_node_memory, prev_dst_memory, negative_memory, distance_metric, mi_method)

        # if getattr(self, "mi_debug", False):
        #     with torch.no_grad():
        #         def _stats(x: torch.Tensor):
        #             return (
        #                 x.mean().item(),
        #                 x.std().item(),
        #                 x.min().item(),
        #                 x.max().item(),
        #                 x.numel(),
        #             )
                # rec_mean, rec_std, rec_min, rec_max, rec_n = _stats(source_recovery_score)
                # drift_mean, drift_std, drift_min, drift_max, drift_n = _stats(source_drift_score)

                # self.mi_step += 1
                # with open(self.mi_csv_path, "a", newline="") as f:
                #     writer = csv.writer(f)
                #     writer.writerow([
                #         self.mi_step, "src", "recovery", distance_metric, rec_mean, rec_std, rec_min, rec_max, rec_n
                #     ])
                #     writer.writerow([
                #         self.mi_step, "src", "drift", distance_metric, drift_mean, drift_std, drift_min, drift_max, drift_n
                #     ])

        return source_recovery_score, source_drift_score, dst_recovery_score, dst_drift_score


    def get_raw_messages(self, source_nodes, destination_nodes, edge_times):
        destination_memory = self.memory.get_memory(destination_nodes)

        source_time_delta = edge_times - self.memory.last_update[source_nodes]
        source_time_delta_encoding = self.time_encoder(source_time_delta.unsqueeze(dim=1)).view(len(
        source_nodes), -1)
        source_message = torch.cat([destination_memory, source_time_delta_encoding], dim=1)
        source_nodes_torch = source_nodes
        (nid, idx) = torch.unique(source_nodes_torch, return_inverse=True)
        message = scatter(source_message, idx, reduce='mean', dim=0)
        message_ts = scatter(edge_times, idx, reduce='max')
        unique_sources = nid

        # new: 把 dt也聚合
        dt_aggr = scatter(source_time_delta, idx, reduce='mean')

        return unique_sources, message, message_ts, dt_aggr


    def update_memory_tensor(self, nodes, messages_tensor, messages_ts_tensor):
        # Aggregate messages for the same nodes

        unique_node_ids = torch.unique(nodes).to(self.device)
        mask = (messages_ts_tensor[unique_node_ids] != 0)

        masked_unique_nodes = unique_node_ids[mask]
        unique_node_messages = messages_tensor[unique_node_ids][mask]
        unique_node_ts = messages_ts_tensor[unique_node_ids][mask]
        
        if len(masked_unique_nodes) > 0:
            if self.memory_updater_type == "transformer":
                unique_messages = unique_node_messages
            else:
                unique_messages = self.message_function.compute_message(unique_node_messages)
        else:
            unique_messages = None

            # Update the memory with the aggregated messages
        self.memory_updater.update_memory(masked_unique_nodes, unique_messages,
                                            timestamps=unique_node_ts)

    def get_updated_memory_tensor(self, nodes, messages_tensor, messages_ts_tensor):
        # Aggregate messages for the same nodes

        unique_node_ids = torch.unique(nodes).to(self.device)
        mask = (messages_ts_tensor[unique_node_ids] != 0)

        masked_unique_nodes = unique_node_ids[mask]
        unique_node_messages = messages_tensor[unique_node_ids][mask]
        unique_node_ts = messages_ts_tensor[unique_node_ids][mask]

        if len(masked_unique_nodes) > 0:
            if self.memory_updater_type == "transformer":
                unique_messages = unique_node_messages
            else:
                unique_messages = self.message_function.compute_message(unique_node_messages)
        else:
            unique_messages = None
        updated_memory, updated_last_update = self.memory_updater.get_updated_memory(masked_unique_nodes,
                                                                                    unique_messages,
                                                                                    timestamps=unique_node_ts)
        return updated_memory, updated_last_update

    
    def set_neighbor_finder(self, neighbor_finder):
        self.neighbor_finder = neighbor_finder
        self.embedding_module_recovery.neighbor_finder = neighbor_finder
    
    def caculate_mi_classification(self, mi_bound, labels):
        
        return




