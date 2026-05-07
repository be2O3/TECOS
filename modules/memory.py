import torch
from torch import nn

from collections import deque
from torch_geometric.nn import Linear 
from copy import deepcopy
import torch.nn.init as init
import numpy as np


class Memory(nn.Module):
    def __init__(self, n_nodes, memory_dimension, input_dimension, message_dimension=None, raw_message_dimension=None,
               device="cpu", memory_updater="gru", combination_method='sum',use_random_memory=False, msg_history_len=10, mem_history_len=10, memory_history_retrieve="last"):
        super(Memory, self).__init__()
        self.n_nodes = n_nodes
        self.memory_dimension = memory_dimension
        self.input_dimension = input_dimension
        self.message_dimension = message_dimension
        self.raw_message_dimension = raw_message_dimension
        self.device = device
        self.use_random_memory = use_random_memory
        self.memory_updater = memory_updater

        self.combination_method = combination_method

        if memory_updater == "transformer":# new
            self.msg_history_len = msg_history_len
            self.msg_history_dim = raw_message_dimension
            # 环形cache
            self.msg_history = nn.Parameter(
                    torch.zeros(self.n_nodes, self.msg_history_len, self.msg_history_dim, device=self.device), requires_grad=False,
                )
            self.dt_history = nn.Parameter(
                    torch.zeros(self.n_nodes, self.msg_history_len, device=self.device), requires_grad=False,
                )
            self.history_ptr = nn.Parameter(
                    torch.zeros(self.n_nodes, dtype=torch.long, device=self.device), requires_grad=False,
                )
            
        if memory_updater == "aggrgru":
        # memory_history
            self.memory_history_retrieve = memory_history_retrieve
            self.mem_history_len = mem_history_len

            self.memory_history = deque(maxlen=mem_history_len)
            for _ in range(mem_history_len):
                self.memory_history.append(
                    torch.zeros(self.n_nodes, self.memory_dimension, requires_grad=False).to(self.device)
                )

        self.__init_memory__()

    def __init_memory__(self):
        if self.use_random_memory:
            self.memory = nn.Parameter(torch.zeros((self.n_nodes, self.memory_dimension)).to(self.device), requires_grad=False) 
            init.xavier_normal_(self.memory)
        else:
            self.memory = nn.Parameter(torch.zeros((self.n_nodes, self.memory_dimension)).to(self.device), requires_grad=False) 
        self.last_update = nn.Parameter(torch.zeros(self.n_nodes).to(self.device), requires_grad=False)

        self.messages_tensor = nn.Parameter(torch.zeros(self.n_nodes, self.raw_message_dimension).to(self.device),requires_grad=False)
        self.messages_time = nn.Parameter(torch.zeros(self.n_nodes).to(self.device),requires_grad=False)

        if self.memory_updater == "aggrgru":
        # 重新构造 deque（最干净），不要复用旧 tensor
            self.memory_history = deque(maxlen=self.mem_history_len)
            for _ in range(self.mem_history_len):
                self.memory_history.append(
                    torch.zeros(self.n_nodes, self.memory_dimension, device=self.device).detach()
                )


    def push_msg_history(self, node_ids, msg_vec, dt_vec):
        ptr = self.history_ptr[node_ids]  # [B]
        self.msg_history[node_ids, ptr, :] = msg_vec.detach()
        self.dt_history[node_ids, ptr] = dt_vec.detach()
        self.history_ptr[node_ids] = (ptr + 1) % self.msg_history_len

    def get_msg_history(self, node_ids):
        ptr = self.history_ptr[node_ids]
        B = node_ids.numel()
        L = self.msg_history_len

        # indices: [B, L] 每行是（ptr, ptr+1,...,ptr+L-1）% L
        arr = torch.arange(L, device=self.device).unsqueeze(0).expand(B, L)
        idx = (ptr.unsqueeze(1) + arr) % L  # [B, L]

        msg_seq = self.msg_history[node_ids.unsqueeze(1), idx]  # [B, L, D]
        dt_seq = self.dt_history[node_ids.unsqueeze(1), idx]  # [B, L]
        return msg_seq, dt_seq
    

    def get_memory_all(self):
        if self.memory_updater == "aggrgru":
            if self.memory_history_retrieve == "last":
                return self.memory_history[-1]
            elif self.memory_history_retrieve == "mean":
                return torch.stack(list(self.memory_history), dim=0).mean(dim=0)
            else:
                raise ValueError(f"Unknown memory_history_retrieve={self.memory_history_retrieve}")
        else:
            return self.memory


    def store_raw_messages(self, nodes, messages, message_ts):
        self.messages_tensor[nodes] = messages
        self.messages_time[nodes] = message_ts

    def get_memory(self, node_idxs):
        if self.memory_updater == "aggrgru":
            if self.memory_history_retrieve == "last":
                return self.memory_history[-1][node_idxs]
            elif self.memory_history_retrieve == "mean":
                return torch.stack([m[node_idxs] for m in self.memory_history], dim=0).mean(dim=0)
        else:
            return self.memory[node_idxs, :]
    
    def set_memory(self, node_idxs, values):
        if self.memory_updater == "aggrgru":
            for i in range(len(self.memory_history) - 1):
                self.memory_history[i][node_idxs] = self.memory_history[i + 1][node_idxs]
            self.memory_history[-1][node_idxs] = values.detach()
        else:
            self.memory[node_idxs, :] = values
    
    
            

    def get_last_update(self, node_idxs):
        return self.last_update[node_idxs]

    def detach_memory(self):
        self.memory.detach_()
        self.messages_tensor.detach_()
        self.messages_time.detach_()
    

    def clear_messages(self, nodes):

        unique_nodes = torch.unique(nodes)
        self.messages_tensor[unique_nodes].zero_()
        self.messages_time[unique_nodes].zero_()

