import torch
import torch.nn.functional as F

def score_dot_pos(x, y):
    # 正样本打分
    x = F.normalize(x, p=2, dim=1)
    y = F.normalize(y, p=2, dim=1)
    return x @ y.t()

def score_dot_neg(q, neg_mem):
    # 负样本打分
    # q: [B, D]
    # neg_mem: [K, D]
    # return [B, K]
    q = F.normalize(q, p=2, dim=1)
    neg_mem = F.normalize(neg_mem, p=2, dim=-1)
    scores = q @ neg_mem.t()
    return scores

def neg_logmeanexp_dot(q, neg_mem):
    scores = score_dot_neg(q, neg_mem)  # [B, K]
    K = torch.tensor(scores.shape[1], device=scores.device, dtype=scores.dtype)
    return torch.logsumexp(scores, dim=1) - torch.log(K)

def neg_mean_dot(q, neg_mem):
    scores = score_dot_neg(q, neg_mem)  # [B, K]
    return scores.mean(dim=1)