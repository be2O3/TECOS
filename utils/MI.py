import os
import csv
import torch
import torch.nn.functional as F
from utils.cosine import cosine_similarity, cosine_neg_mean_score, neg_mean_cosine
from utils.dot import score_dot_pos, neg_logmeanexp_dot, score_dot_neg
from utils.euclidean import score_neg_sq_l2_pos, neg_logmeanexp_neg_sq_l2, score_neg_sq_l2_neg

def js_fgan_lower_bound(f_pos, f_neg):
    """Lower bound on Jensen-Shannon divergence from Nowozin et al. (2016)."""
    pos_term = -F.softplus(-f_pos)
    neg_term = F.softplus(f_neg).mean(dim=1)
    return pos_term - neg_term

def mine_lower_bound(pos_node_embedding, node_memory, negative_memory, distance_metric):
    if distance_metric == 'cosine':
        f_pos = torch.diag(cosine_similarity(pos_node_embedding, node_memory))
        f_neg = cosine_neg_mean_score(pos_node_embedding, negative_memory)
        js = js_fgan_lower_bound(f_pos, cosine_similarity(pos_node_embedding, negative_memory))
        # neg_mat = cosine_similarity(pos_node_embedding, negative_memory)
    
    elif distance_metric == 'euclidean':
        f_pos = score_neg_sq_l2_pos(pos_node_embedding, node_memory)
        f_neg = neg_logmeanexp_neg_sq_l2(pos_node_embedding, negative_memory)
        js = js_fgan_lower_bound(f_pos, score_neg_sq_l2_neg(pos_node_embedding, negative_memory))
        # neg_mat = score_neg_sq_l2_neg(pos_node_embedding, negative_memory)
    elif distance_metric == 'dot':
            f_pos = torch.diag(score_dot_pos(pos_node_embedding, node_memory))
            f_neg = neg_logmeanexp_dot(pos_node_embedding, negative_memory)
            js = js_fgan_lower_bound(f_pos, score_dot_neg(pos_node_embedding, negative_memory))
    
    dv = f_pos - f_neg
    with torch.no_grad():
        dv_js = dv - js
    mine = js + dv_js
    return mine

def infonce_lower_bound(pos_node_embedding, node_memory, negative_memory, distance_metric):
    if distance_metric == 'cosine':
        f_pos = torch.exp(torch.diag(cosine_similarity(pos_node_embedding, node_memory)))
        f_neg = torch.exp(cosine_similarity(pos_node_embedding, negative_memory)).sum(dim=1)
        return torch.log(f_pos / f_neg)
    elif distance_metric == 'euclidean':
        f_pos = torch.exp(score_neg_sq_l2_pos(pos_node_embedding, node_memory))
        f_neg = torch.exp(score_neg_sq_l2_neg(pos_node_embedding, negative_memory)).sum(dim=1)
        return torch.log(f_pos / f_neg)
    elif distance_metric == 'dot':
        f_pos = torch.exp(torch.diag(score_dot_pos(pos_node_embedding, node_memory)))
        f_neg = torch.exp(score_dot_neg(pos_node_embedding, negative_memory)).sum(dim=1)
        return torch.log(f_pos / f_neg)


def mi_lower_bound(pos_node_embedding, node_memory, negative_memory, distance_metric, mi_method):
    if mi_method == 'mine':
        return mine_lower_bound(pos_node_embedding, node_memory, negative_memory, distance_metric)
    elif mi_method == 'infonce':
        return infonce_lower_bound(pos_node_embedding, node_memory, negative_memory, distance_metric)

