from torch import nn
import torch

class MemoryUpdater(nn.Module):
    def update_memory(self, unique_node_ids, unique_messages, timestamps):
        pass


class SequenceMemoryUpdater(MemoryUpdater):
    def __init__(self, memory, message_dimension, memory_dimension, device):
        super(SequenceMemoryUpdater, self).__init__()
        self.memory = memory
        self.message_dimension = message_dimension
        self.device = device
    
    def update_memory(self, unique_node_ids, unique_messages, timestamps):
        if len(unique_node_ids) <= 0:
            return
        assert (self.memory.get_last_update(unique_node_ids) <= timestamps).all().item(), "Trying to " \
                                                                                     "update memory to time in the past"

        memory = self.memory.get_memory(unique_node_ids)
        self.memory.last_update[unique_node_ids] = timestamps
        updated_memory = self.memory_updater(unique_messages, memory)

        self.memory.set_memory(unique_node_ids, updated_memory)
    
    # def get_updated_memory(self, unique_node_ids, unique_messages, timestamps):
    #     if len(unique_node_ids) <= 0:
    #         return self.memory.memory.data.clone(), self.memory.last_update.data.clone()
    #     assert (self.memory.get_last_update(unique_node_ids) <= timestamps).all().item(), "Trying to " \
    #                                                                                  "update memory to time in the past"
        
    #     updated_memory = self.memory.memory.data.clone()
    #     updated_memory[unique_node_ids] = self.memory_updater(unique_messages, updated_memory[unique_node_ids])

    #     updated_last_update = self.memory.last_update.data.clone()
    #     updated_last_update[unique_node_ids] = timestamps

    #     return updated_memory, updated_last_update
    def get_updated_memory(self, unique_node_ids, unique_messages, timestamps):
        if len(unique_node_ids) <= 0:
            # 返回“当前快照”
            cur = self.memory.get_memory_all()
            return cur.data.clone(), self.memory.last_update.data.clone()

        assert (self.memory.get_last_update(unique_node_ids) <= timestamps).all().item(), \
            "Trying to update memory to time in the past"

        base_memory = self.memory.get_memory_all()          # [N, D]（aggrgru: deque最后帧；普通: Parameter）
        updated_memory = base_memory.data.clone()           # 只做一次 clone
        updated_memory[unique_node_ids] = self.memory_updater(
            unique_messages, base_memory[unique_node_ids]
        )
        updated_last_update = self.memory.last_update.data.clone()
        updated_last_update[unique_node_ids] = timestamps

        return updated_memory, updated_last_update


class TransformerMemoryUpdater(SequenceMemoryUpdater):
    def __init__(self, memory, message_dimension, memory_dimension, device, nhead=4, num_layers=2, dropout=0.1, history_len=10):
        super().__init__(memory, message_dimension, memory_dimension, device)
        self.memory_dimension = memory_dimension
        self.history_len = history_len

        self.msg_proj = nn.Linear(message_dimension, memory_dimension) \
            if message_dimension != memory_dimension else nn.Identity()
        
        self.dt_mlp = nn.Sequential(
            nn.Linear(1, memory_dimension),
            nn.GELU(),
            nn.Linear(memory_dimension, memory_dimension)
        )
        
        enc_layer = nn.TransformerEncoderLayer(
            d_model=memory_dimension,
            nhead=nhead,
            dim_feedforward=4*memory_dimension,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation='gelu'
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        self.pos_emb = nn.Parameter(torch.zeros(1, self.history_len, memory_dimension))
        nn.init.normal_(self.pos_emb, std=0.02)
    
    def _encode_seq(self, msg_seq, dt_seq):
        # msg_seq: [B, L, msg_dim], dt_seq: [B, L]
        x = self.msg_proj(msg_seq)  # [B, L, D]
        dt_tok = self.dt_mlp(dt_seq.unsqueeze(-1))  # [B, L, D]
        x = x + dt_tok + self.pos_emb[:, : x.size(1), :]
        return x

    def _compute_updated(self, node_ids, unique_messages, timestamps, update_history=False):
         # 1) 先把当前聚合 message 推入 history（训练时 update_history=True）
        # dt 用 timestamps - last_update（这里 last_update 还没更新）
        cur_mem = self.memory.get_memory(node_ids)  # [B, D]
        last = self.memory.get_last_update(node_ids)
        dt = (timestamps - last).clamp(min=0)

        if update_history:
            self.memory.push_msg_history(node_ids, unique_messages, dt)

        # 2) 取出 L 条 message history（已经包含当前这条）
        msg_seq, dt_seq = self.memory.get_msg_history(node_ids)  # [B, L, msg_dim], [B, L]
        x = self._encode_seq(msg_seq, dt_seq)                    # [B, L, D]

        out = self.encoder(x)            # [B, L, D]
        updated = out[:, -1, :]          # 用最新 token 作为新 memory
        return updated
    
    def update_memory(self, unique_node_ids, unique_messages, timestamps):
        if len(unique_node_ids) <= 0:
            return
        assert (self.memory.get_last_update(unique_node_ids) <= timestamps).all().item()

        updated = self._compute_updated(unique_node_ids, unique_messages, timestamps, update_history=True)
        self.memory.last_update[unique_node_ids] = timestamps
        self.memory.set_memory(unique_node_ids, updated)
    
    def get_updated_memory(self, unique_node_ids, unique_messages, timestamps):
        if len(unique_node_ids) <= 0:
            return self.memory.memory.data.clone(), self.memory.last_update.data.clone()
        assert (self.memory.get_last_update(unique_node_ids) <= timestamps).all().item()

        updated_all = self.memory.memory.data.clone()
        updated_last = self.memory.last_update.data.clone()
        updated_last[unique_node_ids] = timestamps

        updated_part = self._compute_updated(unique_node_ids, unique_messages, timestamps, update_history=False)
        updated_all[unique_node_ids] = updated_part
        return updated_all, updated_last

class GRUMemoryUpdater(SequenceMemoryUpdater):
    def __init__(self, memory, message_dimension, memory_dimension, device):
        super(GRUMemoryUpdater, self).__init__(memory, message_dimension, memory_dimension, device)
        self.memory_updater = nn.GRUCell(input_size=message_dimension,
                                     hidden_size=memory_dimension)        

class RNNMemoryUpdater(SequenceMemoryUpdater):
    def __init__(self, memory, message_dimension, memory_dimension, device):
        super(RNNMemoryUpdater, self).__init__(memory, message_dimension, memory_dimension, device)
        self.memory_updater = nn.RNNCell(input_size=message_dimension,
                                     hidden_size=memory_dimension)
        

def get_memory_updater(module_type, memory, message_dimension, memory_dimension, device):
    if module_type == "gru" or module_type == "aggrgru":
        return GRUMemoryUpdater(memory, message_dimension, memory_dimension, device)
    elif module_type == "rnn":
        return RNNMemoryUpdater(memory, message_dimension, memory_dimension, device)
    elif module_type == "transformer":
        return TransformerMemoryUpdater(memory, message_dimension, memory_dimension, device,
                                        nhead=4, num_layers=2, dropout=0.1)
    else:
        AssertionError('memory updater type is wrong!')

