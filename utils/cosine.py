import torch
import torch.nn.functional as F

def cosine_neg_mean_score(node_embedding, negative_memory):

    # 余弦相似度矩阵 [B, K]
    sim = cosine_similarity(node_embedding, negative_memory) # [B, K]
    z = torch.logsumexp(sim, dim=1) - torch.log(
        torch.tensor(sim.size(1), device=sim.device, dtype=sim.dtype)
    )  # [B]
    return z

def neg_mean_cosine(q, neg_mem):
    # 对k个负样本做均值，返回逐样本 [B]
    cos_sim = cosine_similarity(q, neg_mem) # [B, K]
    z = cos_sim.mean(dim=1)
    return z # [B]

def cosine_similarity(z1, z2):
    z1 = F.normalize(z1, p=2, dim=1)
    z2 = F.normalize(z2, p=2, dim=1)
    res = torch.mm(z1, z2.t())
    return res