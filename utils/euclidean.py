import torch
import torch.nn.functional as F

def score_neg_sq_l2_pos(x, y):
    # x: [B, D]
    # y: [B, D]
    # x = F.normalize(x, p=2, dim=1)
    # y = F.normalize(y, p=2, dim=1)
    return -((x-y).pow(2).sum(dim=1))


def score_neg_sq_l2_neg(q, neg_mem):
    # 负样本打分
    # q: [B, D]
    # neg_mem: [K, D]
    # return [B, K]
    # q = F.normalize(q, p=2, dim=1)
    # neg_mem = F.normalize(neg_mem, p=2, dim=1)
    dist = torch.cdist(q.float(), neg_mem.float(), p=2)
    scores = -(dist ** 2)
    return scores # [B, K]


def neg_logmeanexp_neg_sq_l2(q, neg_mem):
    # 对k个负样本做 log-mean-exp，返回逐样本 [B]
    scores = score_neg_sq_l2_neg(q, neg_mem)  # [B, K]
    z = torch.logsumexp(scores, dim=1) - torch.log(
        torch.tensor(scores.size(1), device=scores.device, dtype=scores.dtype)
    )  # [B]
    return z

def neg_mean_neg_sq_l2(q, neg_mem):
    scores = score_neg_sq_l2_neg(q, neg_mem)  # [B, K]
    return scores.mean(dim=1)